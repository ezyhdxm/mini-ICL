import gc

import torch
from typing import Optional, Sequence

import icl.utils.notebook_utils as nu
get_dyck_sampler = None
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
    mean_h_dict = {}

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
        mean_h_dict[li] = mean_h  # (n_tasks, n_pos, D)

    return {
        'layers': layers,
        'positions': list(range(min_n_pos)),
        'var_pos': var_pos_dict,
        'var_pos_per_task': var_per_task_dict,
        'var_pos_norm': var_pos_norm_dict,
        'mean_norm_sq_per_task': mean_norm_sq_dict,
        'mean_h_per_task': mean_h_dict,
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
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title("", fontsize=18)

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


def _compute_dyck_r2(results, prefix_k, sampler):
    """Compute R² from pre-collected variance results.

    Parameters
    ----------
    results : dict
        Output of ``compute_p1_variance_dyck``.
    prefix_k : int or None
        Number of most-recent planted Dyck characters to condition on.
        ``None`` means full prefix (equivalent to conditioning on task id).
    sampler : DyckSampler
        Needed to retrieve Dyck strings for prefix grouping.

    Returns
    -------
    dict : ``{layer_idx: list of R² per Dyck position}``
    """
    eps = 1e-10
    n_tasks = results['n_tasks']
    n_pos = len(results['positions'])
    n_per_cell = results['samples_per_task']

    if prefix_k is None:
        # Full prefix: each task is its own cell (same as before)
        r2_dict = {}
        for li in results['layers']:
            var_per_task = results['var_pos_per_task'][li]
            mean_h = results['mean_h_per_task'][li]
            grand_mean = mean_h.mean(dim=0)
            ss_between = (n_per_cell * ((mean_h - grand_mean.unsqueeze(0)) ** 2).sum(dim=-1)).sum(dim=0)
            ss_within = (n_per_cell * var_per_task).sum(dim=0)
            ss_total = ss_between + ss_within
            r2_dict[li] = (1.0 - ss_within / (ss_total + eps)).tolist()
        return r2_dict

    # Truncated prefix: group tasks by last-k Dyck characters at each position
    all_dyck_strings = []
    for task_idx in range(n_tasks):
        path = sampler.get_task_dyck_path(task_idx).cpu().tolist()
        all_dyck_strings.append(path)

    r2_dict = {}
    for li in results['layers']:
        var_per_task = results['var_pos_per_task'][li]   # (n_tasks, n_pos)
        mean_h = results['mean_h_per_task'][li]          # (n_tasks, n_pos, D)
        D = mean_h.shape[-1]

        r2_per_pos = []
        for j in range(n_pos):
            # Build cell grouping by last-k prefix at Dyck position j
            prefix_to_tasks = {}
            for task_idx in range(n_tasks):
                dyck_str = all_dyck_strings[task_idx]
                start = max(0, j + 1 - prefix_k)
                key = tuple(dyck_str[start:j + 1])
                if key not in prefix_to_tasks:
                    prefix_to_tasks[key] = []
                prefix_to_tasks[key].append(task_idx)

            # Compute grand mean at this position
            grand_mean_j = mean_h[:, j, :].mean(dim=0)  # (D,)

            ss_between_j = 0.0
            ss_within_j = 0.0
            for _prefix_key, task_ids in prefix_to_tasks.items():
                n_group = len(task_ids)
                # Group cell mean = weighted average of task cell means
                group_mean = mean_h[task_ids, j, :].mean(dim=0)  # (D,)

                # Between: this group vs grand mean
                ss_between_j += n_group * n_per_cell * float(
                    ((group_mean - grand_mean_j) ** 2).sum()
                )

                # Within: variance within each task (unchanged) +
                #          variance of task means around group mean
                for t_idx in task_ids:
                    ss_within_j += n_per_cell * float(var_per_task[t_idx, j])
                    ss_within_j += n_per_cell * float(
                        ((mean_h[t_idx, j, :] - group_mean) ** 2).sum()
                    )

            ss_total_j = ss_between_j + ss_within_j
            r2_j = 1.0 - ss_within_j / (ss_total_j + eps)
            r2_per_pos.append(r2_j)

        r2_dict[li] = r2_per_pos

    return r2_dict


def plot_task_vector_r2_dyck(
    exp_name: str,
    layers: Optional[Sequence[int]] = None,
    prefix_k: Optional[int] = None,
    batch_size: int = 64,
    n_masks: int = 30,
    step: Optional[int] = None,
    n_minor: int = 0,
    n_ood: int = 0,
    verbose: bool = False,
    figsize: tuple = (5, 3.2),
    log_x: bool = False,
    show: bool = True,
    show_ylabel: bool = True,
    print_summary: bool = True,
) -> dict:
    """Task-token vector R² for the Dyck task.

    Measures what fraction of hidden-state variance at each Dyck position
    is explained by conditioning on the last ``prefix_k`` planted Dyck
    characters (and task identity when prefixes coincide).

    Parameters
    ----------
    exp_name : str
    layers : list of int, optional
    prefix_k : int or None
        Number of most-recent planted Dyck characters to condition on.
        ``None`` (default) uses the full prefix, which is equivalent to
        conditioning on task identity since the Dyck string is deterministic
        per task.  Use ``prefix_k=1`` for current-token conditioning
        (analogous to coin/latent), ``prefix_k=3`` for last-3, etc.
    batch_size : int
    n_masks : int
    step : int, optional
    n_minor : int
    n_ood : int
    verbose : bool
    figsize, log_x, show, show_ylabel, print_summary : plot options

    Returns
    -------
    dict
        ``{'results', 'r2', 'prefix_k', 'fig'}``.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is not installed.")

    results = compute_p1_variance_dyck(
        exp_name=exp_name,
        layers=layers,
        B=batch_size,
        n_masks=n_masks,
        step=step,
        n_minor=n_minor,
        n_ood=n_ood,
        verbose=verbose,
    )

    # Load sampler to get Dyck strings for prefix grouping
    n_minor_val = n_minor if n_minor is not None else 1000000
    if n_minor_val == -1:
        n_minor_val = 0
    sampler, _ = get_dyck_sampler(exp_name, n_minor_val, n_ood)

    positions = results['positions']
    r2_dict = _compute_dyck_r2(results, prefix_k, sampler)

    prefix_label = f"k={prefix_k}" if prefix_k is not None else "full prefix"

    if print_summary:
        header = f"{'Layer':>6} {'Pos':>5} {'R²':>8}"
        print("=" * len(header))
        print(f"  Dyck task-token vector R² ({prefix_label})")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for li in results['layers']:
            for j, pos in enumerate(positions):
                print(f"{li:>6} {pos:>5} {r2_dict[li][j]:>8.4f}")
        print("=" * len(header))

    from icl.utils.separability import _layer_style

    fig, ax = plt.subplots(figsize=figsize)
    for li in results['layers']:
        vals = [1.0 - v for v in r2_dict[li]]
        ax.plot(
            positions, vals,
            label=str(li),
            **_layer_style(li, len(positions)),
        )

    ax.set_xlabel("Dyck position", fontsize=13)
    if show_ylabel:
        ax.set_ylabel("Residual variance ratio", fontsize=13)
    if log_x and len(positions) > 1:
        ax.set_xscale("symlog", linthresh=1)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=12)
    ax.legend(title="Layer", fontsize=12, title_fontsize=12,
              framealpha=0.9, loc="center left",
              bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0.3, handlelength=2.2)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        'results': results,
        'r2': r2_dict,
        'prefix_k': prefix_k,
        'fig': fig,
    }
