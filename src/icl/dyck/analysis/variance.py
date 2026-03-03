import gc

import torch
from typing import Optional, Sequence

import icl.utils.notebook_utils as nu
from icl.dyck.legacy.dyck_task_vec import get_dyck_sampler
from icl.dyck.dyck_utils import sample_binary_mask
from icl.utils.logger import setup_logger

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = setup_logger(__name__)


def compute_p1_variance_dyck_padded(
    exp_name: str,
    layers: Optional[Sequence[int]] = None,
    B: int = 64,
    n_masks: int = 30,
    step: Optional[int] = None,
    n_minor: int = 32,
    n_ood: int = 0,
    verbose: bool = False,
) -> dict:
    """
    Compute Var(H | dyck_prefix, task) at each Dyck position index for one or
    more layers of the Dyck task.

    For each task k and Dyck position index j, the Dyck prefix steps[0:j+1] is
    deterministic. The variance captures residual variability from random Markov
    noise tokens and mask placement across multiple masks.

    Parameters
    ----------
    exp_name : str
        Experiment name (folder under results/dyck/).
    layers : list of int, optional
        Layer indices to extract hidden representations from. If None, uses all
        layers. A single int is also accepted.
    B : int, default=64
        Batch size per (task, mask) forward pass.
    n_masks : int, default=30
        Number of independently-sampled masks for position diversity.
    step : int, optional
        Checkpoint step. If None, uses the final checkpoint.
    n_minor : int, default=32
        Number of minor tasks to subsample. Use -1 for none, or a large number
        (e.g. 1000000) for all available.
    n_ood : int, default=0
        Number of OOD tasks.
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    dict with keys:
        - 'var_pos': dict mapping layer_idx -> list of floats (variance per Dyck position)
        - 'var_pos_per_task': dict mapping layer_idx -> (n_tasks, n_dyck) tensor
        - 'var_pos_norm': dict mapping layer_idx -> list of floats (normalized variance)
        - 'n_tasks': int
        - 'n_dyck_positions': int
        - 'layers': list of int — layer indices used
        - 'n_masks': int
        - 'B': int
        - 'samples_per_task': int
    """
    _, _sampler_orig, config = nu.load_everything("dyck", exp_name)

    if step is None:
        step = config.training.num_epochs

    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval()
    device = config.device
    model.to(device)

    n_layers = len(model.layers)

    if layers is None:
        layers = list(range(n_layers))
    elif isinstance(layers, int):
        layers = [layers]
    else:
        layers = list(layers)

    n_minor_val = n_minor if n_minor is not None else 1000000
    if n_minor_val == -1:
        n_minor_val = 0
    sampler, _ = get_dyck_sampler(exp_name, n_minor_val, n_ood)

    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    seq_len_padded = sampler.seq_len
    d_model = config.model.emb_dim

    masks_list = [sample_binary_mask(config).to(device) for _ in range(n_masks)]

    masks_info = []
    for mask in masks_list:
        dyck_real = torch.nonzero(mask == 1, as_tuple=True)[0]
        pp_raw = 2 * dyck_real + 1
        valid = pp_raw < seq_len_padded
        pp = pp_raw[valid].to(device=device, dtype=torch.long)
        if len(pp) > 0:
            masks_info.append({'mask': mask, 'padded_positions': pp, 'n_pos': len(pp)})

    if not masks_info:
        raise RuntimeError("No masks produced valid Dyck positions.")

    min_n_pos = min(info['n_pos'] for info in masks_info)

    if verbose:
        logger.info(f"Dyck P1 variance: {n_tasks} tasks, {len(masks_info)} masks, "
                     f"{min_n_pos} Dyck positions, layers {layers}, B={B}")

    caches = {}

    def make_hook(li):
        def hook_fn(module, inp, out):
            if torch.is_tensor(out):
                caches[li] = out.detach()
            elif isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                caches[li] = out[0].detach()
        return hook_fn

    handles = []
    for li in layers:
        handles.append(model.layers[li].attn_block.register_forward_hook(make_hook(li)))

    sum_h = {li: torch.zeros((n_tasks, min_n_pos, d_model), dtype=torch.float64) for li in layers}
    sum_h2 = {li: torch.zeros((n_tasks, min_n_pos), dtype=torch.float64) for li in layers}
    count = torch.zeros((n_tasks,), dtype=torch.int64)

    try:
        for task_idx in range(n_tasks):
            for m_info in masks_info:
                mask = m_info['mask']
                pp = m_info['padded_positions'][:min_n_pos]

                demo_data, _ = sampler.generate(
                    mode="testing", task=task_idx, num_samples=B,
                    dyck_mask=mask.clone(),
                )
                demo_data = demo_data.to(device)

                caches.clear()
                _ = model(demo_data)

                for li in layers:
                    h = caches[li].index_select(dim=1, index=pp).cpu().to(torch.float64)
                    sum_h[li][task_idx] += h.sum(dim=0)
                    sum_h2[li][task_idx] += (h ** 2).sum(dim=-1).sum(dim=0)

                count[task_idx] += B

                del demo_data
                caches.clear()

            if verbose and (task_idx == 0 or (task_idx + 1) % 50 == 0 or task_idx == n_tasks - 1):
                logger.info(f"  Task {task_idx + 1}/{n_tasks} done")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        for h in handles:
            h.remove()

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    eps = 1e-8
    n = count.unsqueeze(1).to(torch.float64)  # (n_tasks, 1)

    var_pos_dict = {}
    var_per_task_dict = {}
    var_pos_norm_dict = {}
    mean_norm_sq_dict = {}

    for li in layers:
        mean_h = sum_h[li] / n.unsqueeze(2)
        mean_h2 = sum_h2[li] / n
        mean_norm_sq = (mean_h ** 2).sum(dim=-1)

        var_per_task = (mean_h2 - mean_norm_sq).clamp(min=0.0)
        var_pos = var_per_task.mean(dim=0)

        norm_per_task = var_per_task / (mean_norm_sq + eps)
        var_pos_norm = norm_per_task.mean(dim=0)

        var_pos_dict[li] = var_pos.tolist()
        var_per_task_dict[li] = var_per_task
        var_pos_norm_dict[li] = var_pos_norm.tolist()
        mean_norm_sq_dict[li] = mean_norm_sq

    return {
        'layers': layers,
        'positions': list(range(min_n_pos)),
        'var_pos': var_pos_dict,
        'var_pos_per_task': var_per_task_dict,
        'var_pos_norm': var_pos_norm_dict,
        'mean_norm_sq_per_task': mean_norm_sq_dict,
        'n_tasks': n_tasks,
        'n_dyck_positions': min_n_pos,
        'n_masks': len(masks_info),
        'B': B,
        'samples_per_task': int(count[0].item()),
    }


@torch.no_grad()
def compute_p1_variance_dyck(
    exp_name: str,
    layers: Optional[Sequence[int]] = None,
    B: int = 64,
    n_masks: int = 30,
    step: Optional[int] = None,
    n_minor: int = 32,
    n_ood: int = 0,
    verbose: bool = False,
) -> dict:
    """
    Non-padded Dyck counterpart of ``compute_p1_variance_dyck_padded``.

    Computes Var(H | dyck_prefix, task) at Dyck-token positions for models trained
    with non-padded Dyck sequences.

    Position convention:
    - Dyck token locations are identified by ``dyck_mask == 1`` on the real
      (non-padded) timeline.
    - Hidden states are extracted at those same real-token positions, truncated
      to ``< seq_len - 1`` so each position corresponds to a valid next-token
      prediction context.
    """
    _, _sampler_orig, config = nu.load_everything("dyck", exp_name)

    if step is None:
        step = config.training.num_epochs

    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval()
    device = config.device
    model.to(device)

    n_layers = len(model.layers)

    if layers is None:
        layers = list(range(n_layers))
    elif isinstance(layers, int):
        layers = [layers]
    else:
        layers = list(layers)

    n_minor_val = n_minor if n_minor is not None else 1000000
    if n_minor_val == -1:
        n_minor_val = 0
    sampler, _ = get_dyck_sampler(exp_name, n_minor_val, n_ood)

    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    seq_len_real = sampler.seq_len
    d_model = config.model.emb_dim

    masks_list = [sample_binary_mask(config).to(device) for _ in range(n_masks)]

    masks_info = []
    for mask in masks_list:
        dyck_real = torch.nonzero(mask == 1, as_tuple=True)[0]
        rp = dyck_real[dyck_real < (seq_len_real - 1)].to(device=device, dtype=torch.long)
        if len(rp) > 0:
            masks_info.append({'mask': mask, 'real_positions': rp, 'n_pos': len(rp)})

    if not masks_info:
        raise RuntimeError("No masks produced valid non-padded Dyck positions.")

    min_n_pos = min(info['n_pos'] for info in masks_info)

    if verbose:
        logger.info(
            f"Dyck nonpadded P1 variance: {n_tasks} tasks, {len(masks_info)} masks, "
            f"{min_n_pos} Dyck positions, layers {layers}, B={B}"
        )

    caches = {}

    def make_hook(li):
        def hook_fn(module, inp, out):
            if torch.is_tensor(out):
                caches[li] = out.detach()
            elif isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                caches[li] = out[0].detach()
        return hook_fn

    handles = []
    for li in layers:
        handles.append(model.layers[li].attn_block.register_forward_hook(make_hook(li)))

    sum_h = {li: torch.zeros((n_tasks, min_n_pos, d_model), dtype=torch.float64) for li in layers}
    sum_h2 = {li: torch.zeros((n_tasks, min_n_pos), dtype=torch.float64) for li in layers}
    count = torch.zeros((n_tasks,), dtype=torch.int64)

    try:
        for task_idx in range(n_tasks):
            for m_info in masks_info:
                mask = m_info['mask']
                rp = m_info['real_positions'][:min_n_pos]

                demo_data, _ = sampler.generate(
                    mode="testing", task=task_idx, num_samples=B,
                    dyck_mask=mask.clone(),
                )
                demo_data = demo_data.to(device)

                caches.clear()
                _ = model(demo_data)

                for li in layers:
                    h = caches[li].index_select(dim=1, index=rp).cpu().to(torch.float64)
                    sum_h[li][task_idx] += h.sum(dim=0)
                    sum_h2[li][task_idx] += (h ** 2).sum(dim=-1).sum(dim=0)

                count[task_idx] += B

                del demo_data
                caches.clear()

            if verbose and (task_idx == 0 or (task_idx + 1) % 50 == 0 or task_idx == n_tasks - 1):
                logger.info(f"  Task {task_idx + 1}/{n_tasks} done")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        for h in handles:
            h.remove()

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    eps = 1e-8
    n = count.unsqueeze(1).to(torch.float64)

    var_pos_dict = {}
    var_per_task_dict = {}
    var_pos_norm_dict = {}
    mean_norm_sq_dict = {}

    for li in layers:
        mean_h = sum_h[li] / n.unsqueeze(2)
        mean_h2 = sum_h2[li] / n
        mean_norm_sq = (mean_h ** 2).sum(dim=-1)

        var_per_task = (mean_h2 - mean_norm_sq).clamp(min=0.0)
        var_pos = var_per_task.mean(dim=0)

        norm_per_task = var_per_task / (mean_norm_sq + eps)
        var_pos_norm = norm_per_task.mean(dim=0)

        var_pos_dict[li] = var_pos.tolist()
        var_per_task_dict[li] = var_per_task
        var_pos_norm_dict[li] = var_pos_norm.tolist()
        mean_norm_sq_dict[li] = mean_norm_sq

    return {
        'layers': layers,
        'positions': list(range(min_n_pos)),
        'var_pos': var_pos_dict,
        'var_pos_per_task': var_per_task_dict,
        'var_pos_norm': var_pos_norm_dict,
        'mean_norm_sq_per_task': mean_norm_sq_dict,
        'n_tasks': n_tasks,
        'n_dyck_positions': min_n_pos,
        'n_masks': len(masks_info),
        'B': B,
        'samples_per_task': int(count[0].item()),
    }


def plot_p1_variance_dyck(
    exp_name: str,
    layers: Optional[Sequence[int]] = None,
    B: int = 64,
    n_masks: int = 30,
    step: Optional[int] = None,
    n_minor: int = 32,
    n_ood: int = 0,
    verbose: bool = False,
    figsize: tuple = (8, 6),
    log_x: bool = True,
    show: bool = True,
    title: Optional[str] = None,
) -> dict:
    """
    Compute and plot normalized P1 variance for non-padded Dyck sequences.

    This is the non-padded Dyck plotting counterpart of the coin/latent
    non-padded helpers: it calls ``compute_p1_variance_dyck`` and
    plots ``var_pos_norm`` vs Dyck position for each selected layer.

    Returns
    -------
    dict
        ``{'results', 'fig', 'ax'}``, where ``results`` is the full output of
        ``compute_p1_variance_dyck``.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is not installed. Install it with: pip install matplotlib")

    results = compute_p1_variance_dyck(
        exp_name=exp_name,
        layers=layers,
        B=B,
        n_masks=n_masks,
        step=step,
        n_minor=n_minor,
        n_ood=n_ood,
        verbose=verbose,
    )

    fig, ax = plt.subplots(figsize=figsize)

    for layer_idx in results['layers']:
        positions = results['positions']
        var_pos_norm = results['var_pos_norm'][layer_idx]
        ax.plot(
            positions, var_pos_norm, 'o-',
            label=f'Layer {layer_idx}',
            linewidth=2, markersize=6,
        )

    ax.set_xlabel('Dyck Position' + (' (log scale)' if log_x else ''), fontsize=16)
    ax.set_ylabel('Normalized P1 Variance', fontsize=16)
    if log_x:
        ax.set_xscale('log')
    if len(positions) > 0:
        ax.set_xticks(positions)
        ax.set_xticklabels([str(int(p)) for p in positions])
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=16)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()

    return {
        'results': results,
        'fig': fig,
        'ax': ax,
    }
