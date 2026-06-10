"""Task-vector R², P1 variance, and ANOVA separability for the Latent Markov task
on **non-padded** sequences.
"""

import gc
import torch
from typing import Dict, Optional, Sequence

import icl.utils.notebook_utils as nu
from icl.coin.analysis._helpers import compute_hiddens_token_conditioned_coin
from icl.latent_markov.analysis.ood import get_latent_sampler
from icl.utils.logger import setup_logger
from icl.linear.p1_variance import (
    compute_p1_variance_multi_layer,
    extract_plotting_data_multi_layer as extract_p1_plotting_data_multi_layer,
)

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Hiddens cache (holds one entry to avoid recomputation across analyses)
# ---------------------------------------------------------------------------
_hiddens_cache: dict = {}


def _get_hiddens_cached(
    exp_name: str,
    layers,
    batch_size: int,
    positions_of_interest,
    n_minor: int,
    step,
    verbose: bool,
    task_batch_size: int = 8,
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
    state: str = "hidden",
    max_tasks: Optional[int] = None,
) -> tuple:
    """Return (all_hiddens, token_info), reusing cache when parameters match."""
    key = (
        exp_name,
        tuple(layers) if layers is not None else None,
        batch_size,
        tuple(positions_of_interest) if positions_of_interest is not None else None,
        n_minor,
        step,
        post_layernorm,
        extraction_point,
        state,
        max_tasks,
    )
    if key in _hiddens_cache:
        if verbose:
            logger.info("[latent cache] reusing cached token-conditioned hiddens")
        return _hiddens_cache[key]

    result = get_token_conditioned_hiddens(
        exp_name=exp_name,
        layers=layers,
        batch_size=batch_size,
        positions_of_interest=positions_of_interest,
        n_minor=n_minor,
        step=step,
        verbose=verbose,
        task_batch_size=task_batch_size,
        post_layernorm=post_layernorm,
        extraction_point=extraction_point,
        state=state,
        max_tasks=max_tasks,
    )
    _hiddens_cache.clear()
    _hiddens_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# 1.  get_token_conditioned_hiddens
# ---------------------------------------------------------------------------

def get_token_conditioned_hiddens(
    exp_name: str,
    layers: Optional[Sequence] = None,
    batch_size: int = 16,
    positions_of_interest: Optional[list] = None,
    max_unique_tokens: Optional[int] = None,
    n_minor: int = 0,
    step: Optional[int] = None,
    verbose: bool = False,
    task_batch_size: int = 8,
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
    state: str = "hidden",
    max_tasks: Optional[int] = None,
) -> tuple:
    """
    Get token-conditioned hidden representations for Latent Markov task (non-padded).

    Non-padded counterpart of token-conditioned hiddens for latent. Fixes tokens
    at specific positions and extracts hiddens **at the same position** (the real
    token itself — no following PAD token).
    Always uses ``n_ood=0``. ``n_minor`` is capped at ``sampler.n_minor_tasks``.

    For order=1 latent Markov, tokens are state values in {0, ..., num_states-1}.

    Parameters
    ----------
    exp_name : str
    layers : list, optional
        ``None`` → all layers.
    batch_size : int
    positions_of_interest : list, optional
        Position indices ``[0, seq_len-2]``. ``None`` → all.
    max_unique_tokens : int, optional
    n_minor : int
        Capped at ``sampler.n_minor_tasks``.
    step : int, optional
    verbose : bool
    task_batch_size : int
        Number of tasks to batch into a single forward pass.

    Returns
    -------
    all_hiddens : torch.Tensor
        ``(L, n_positions, n_unique_tokens, n_tasks, batch_size, n_embd)``
    token_info : dict
    """
    _, _sampler_orig, config = nu.load_everything("latent", exp_name)

    if step is None:
        step = config.training.num_epochs

    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True
    )
    model.eval().to(config.device)

    sampler, k_minor, _ = get_latent_sampler(exp_name, n_minor, n_ood=0)

    if layers is None:
        layers = list(range(len(model.layers)))

    if verbose:
        logger.info(
            f"Computing non-padded token-conditioned hiddens for latent exp: {exp_name}"
        )
        logger.info(
            f"Layers: {layers}, Batch size: {batch_size}, "
            f"n_tasks: {sampler.n_major_tasks + sampler.n_minor_tasks}"
        )

    all_hiddens, token_info = compute_hiddens_token_conditioned_coin(
        config=config,
        model=model,
        sampler=sampler,
        layers=layers,
        batch_size=batch_size,
        positions_of_interest=positions_of_interest,
        max_unique_tokens=max_unique_tokens,
        task_batch_size=task_batch_size,
        post_layernorm=post_layernorm,
        extraction_point=extraction_point,
        state=state,
        max_tasks=max_tasks,
    )

    if verbose:
        logger.info(f"Computed hiddens shape: {all_hiddens.shape}")
        logger.info(f"Token info - positions: {token_info['positions']}")
        logger.info(f"Token info - n_unique_tokens: {token_info['n_unique_tokens']}")

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return all_hiddens, token_info


# ---------------------------------------------------------------------------
# 2.  task_vector_r2
# ---------------------------------------------------------------------------

def plot_task_vector_r2_latent(
    exp_name: str,
    layers: Optional[Sequence] = None,
    batch_size: int = 16,
    positions_of_interest: Optional[list] = None,
    n_minor: int = 0,
    step: Optional[int] = None,
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
    verbose: bool = False,
    figsize: tuple = (5, 3.2),
    log_x: bool = True,
    show: bool = True,
    show_ylabel: bool = True,
    print_summary: bool = True,
    task_batch_size: int = 8,
    max_tasks: Optional[int] = None,
) -> dict:
    """Task-token vector R² for the Latent Markov task.

    Measures what fraction of hidden-state variance at each position
    is explained by knowing (task, current token).  R² → 1 indicates
    long-context stability.

    Results are cached: calling this followed by
    ``plot_anova_separability_latent`` with the same data parameters
    will not recompute hidden states.

    Parameters
    ----------
    exp_name : str
    layers : list, optional
    batch_size : int
    positions_of_interest : list, optional
    n_minor : int
    step : int, optional
    post_layernorm : bool
    extraction_point : str
        ``"post_attn"`` (default) or ``"post_mlp"`` (residual stream).
    verbose : bool
    figsize, log_x, show, show_ylabel, print_summary : plot options
    task_batch_size : int
        Number of tasks to batch into a single forward pass.

    Returns
    -------
    dict
        ``{'all_hiddens', 'token_info', 'r2_results', 'fig'}``.
    """
    from icl.utils.separability import (
        task_vector_r2_multi,
        plot_task_vector_r2,
        print_task_vector_r2_summary,
    )

    all_hiddens, token_info = _get_hiddens_cached(
        exp_name, layers, batch_size, positions_of_interest, n_minor, step, verbose,
        task_batch_size=task_batch_size,
        post_layernorm=post_layernorm,
        extraction_point=extraction_point,
        max_tasks=max_tasks,
    )

    if layers is None:
        _, _, config = nu.load_everything("latent", exp_name)
        layers = list(range(config.model.num_layers))

    r2_results = task_vector_r2_multi(
        all_hiddens=all_hiddens,
        token_info=token_info,
        layers=layers,
        positions=positions_of_interest,
    )

    if print_summary:
        print_task_vector_r2_summary(r2_results, positions=positions_of_interest)

    fig = plot_task_vector_r2(
        r2_results, figsize=figsize, log_x=log_x, show=show,
        show_ylabel=show_ylabel,
    )

    return {
        "all_hiddens": all_hiddens,
        "token_info": token_info,
        "r2_results": r2_results,
        "fig": fig,
    }


# ---------------------------------------------------------------------------
# 2b.  Blocked task-vector R²
# ---------------------------------------------------------------------------

def plot_task_vector_r2_blocked_latent(
    exp_name: str,
    layers: Optional[Sequence] = None,
    batch_size: int = 16,
    positions_of_interest: Optional[list] = None,
    n_minor: int = 0,
    step: Optional[int] = None,
    block_size: int = 10,
    show_per_position: bool = True,
    verbose: bool = False,
    figsize: tuple = (5, 3.2),
    log_x: bool = True,
    show: bool = True,
    show_ylabel: bool = True,
    task_batch_size: int = 8,
) -> dict:
    r"""Blocked task-vector R² for the Latent Markov task.

    Like ``plot_task_vector_r2_latent``, but pools positions into
    non-overlapping blocks of size ``block_size`` and estimates a
    **single** set of cell means :math:`\theta_{k,a}` per block (shared
    across positions within the block).  Each position retains its own
    mean :math:`\mu_t`.

    The model tested within each block is:

    .. math::
        h_{t,k,a,b} = \mu_t + \theta_{k,a} + \varepsilon_{t,k,a,b}

    where :math:`\theta_{k,a}` is constant across positions in the block.
    A drop relative to per-position R² indicates position–task interaction
    (e.g. from rotary embeddings).

    Parameters
    ----------
    exp_name : str
    layers : list, optional
    batch_size : int
    positions_of_interest : list, optional
    n_minor : int
    step : int, optional
    block_size : int
        Number of adjacent positions per block.
    show_per_position : bool
        If True, overlay the per-position R² in lighter style.
    verbose : bool
    figsize, log_x, show, show_ylabel : plot options
    task_batch_size : int

    Returns
    -------
    dict
        ``{'all_hiddens', 'token_info', 'blocked_r2', 'per_pos_r2', 'fig'}``.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from icl.utils.separability import task_vector_r2_multi

    all_hiddens, token_info = _get_hiddens_cached(
        exp_name, layers, batch_size, positions_of_interest, n_minor, step,
        verbose, task_batch_size=task_batch_size,
    )
    # all_hiddens: (L, n_positions, V_max, K, B, D)

    if layers is None:
        _, _, config = nu.load_everything("latent", exp_name)
        layers = list(range(config.model.num_layers))
    layers_list = list(layers)

    all_positions = token_info["positions"]
    n_unique = token_info["n_unique_tokens"]

    if positions_of_interest is None:
        positions_of_interest = list(all_positions)
    pos_to_idx = {p: i for i, p in enumerate(all_positions)}

    # ---- divide positions into blocks ----
    sorted_pos = sorted(p for p in positions_of_interest if p in pos_to_idx)
    blocks = []
    for i in range(0, len(sorted_pos), block_size):
        blocks.append(sorted_pos[i : i + block_size])

    # ---- per-position R² (for comparison) ----
    per_pos_r2 = None
    if show_per_position:
        per_pos_r2 = task_vector_r2_multi(
            all_hiddens=all_hiddens,
            token_info=token_info,
            layers=layers_list,
            positions=positions_of_interest,
        )

    # ---- blocked R²: shared θ_{k,a} within each block ----
    blocked_r2: Dict[int, list] = {}  # layer -> list of (block_center, r2)

    for l_idx, l_num in enumerate(layers_list):
        if l_idx >= all_hiddens.shape[0]:
            continue
        block_results = []

        for block_pos in blocks:
            # minimum V across block for balanced cells
            V_block = min(n_unique.get(p, all_hiddens.shape[2]) for p in block_pos)
            K = all_hiddens.shape[3]
            B = all_hiddens.shape[4]
            D = all_hiddens.shape[5]
            n_t = len(block_pos)

            # Gather cell means and compute per-position mean μ_t
            # cell_means_per_t: (n_t, V, K, D)
            cell_means_per_t = torch.zeros(n_t, V_block, K, D)
            mu_per_t = torch.zeros(n_t, D)
            for ti, p in enumerate(block_pos):
                pi = pos_to_idx[p]
                h_p = all_hiddens[l_idx, pi, :V_block, :, :, :].float()  # (V, K, B, D)
                cell_means_per_t[ti] = h_p.mean(dim=2)  # (V, K, D)
                mu_per_t[ti] = h_p.mean(dim=(0, 1, 2))  # (D,)

            # Shared θ_{k,a} from position-demeaned cell means
            demeaned_cell_means = cell_means_per_t - mu_per_t[:, None, None, :]
            theta_ka = demeaned_cell_means.mean(dim=0)  # (V, K, D)

            # Compute block R²
            ss_total_block = 0.0
            ss_resid_block = 0.0
            for ti, p in enumerate(block_pos):
                pi = pos_to_idx[p]
                h_p = all_hiddens[l_idx, pi, :V_block, :, :, :].float()  # (V, K, B, D)
                h_demeaned = h_p - mu_per_t[ti]
                ss_total_block += (h_demeaned ** 2).sum().item()
                resid = h_demeaned - theta_ka.unsqueeze(2)  # (V, K, B, D)
                ss_resid_block += (resid ** 2).sum().item()

            r2_block = 1.0 - ss_resid_block / (ss_total_block + 1e-10)
            block_center = np.mean(block_pos)
            block_results.append((block_center, r2_block, block_pos))

        blocked_r2[l_num] = block_results

    # ---- plot ----
    _COLORS = [
        "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
        "#56B4E9", "#F0E442", "#000000",
    ]
    _LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

    fig, ax = plt.subplots(figsize=figsize)

    for i, l_num in enumerate(sorted(blocked_r2.keys())):
        color = _COLORS[i % len(_COLORS)]
        ls = _LINESTYLES[i % len(_LINESTYLES)]

        # per-position R² (light)
        if show_per_position and per_pos_r2 is not None and l_num in per_pos_r2:
            pos_r = per_pos_r2[l_num]
            pp = sorted(pos_r.keys())
            ax.plot(
                pp, [pos_r[p].r2 for p in pp],
                color=color, linestyle=ls, linewidth=1.0, alpha=0.35,
            )

        # blocked R² (bold)
        centers = [br[0] for br in blocked_r2[l_num]]
        r2_vals = [br[1] for br in blocked_r2[l_num]]
        ax.plot(
            centers, r2_vals,
            color=color, linestyle=ls, linewidth=2.5, marker="o",
            markersize=4, label=f"Layer {l_num}",
        )

    ax.set_xlabel("Position", fontsize=13)
    if show_ylabel:
        ax.set_ylabel(r"$R^2$  (blocked $\theta_{k,a}$)", fontsize=13)
    if log_x:
        ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "all_hiddens": all_hiddens,
        "token_info": token_info,
        "blocked_r2": blocked_r2,
        "per_pos_r2": per_pos_r2,
        "fig": fig,
    }


def _fit_probe_r2_per_position_latent(
    hiddens_by_layer: Dict[int, torch.Tensor],
    posteriors: torch.Tensor,
    logits: torch.Tensor,
    real_tokens: torch.Tensor,
    n_major: int,
    layers: Sequence[int],
    positions: Sequence[int],
    validation_split: Optional[float] = None,
    include_position_bias: bool = True,
    include_logit: bool = True,
) -> Dict[int, Dict[int, float]]:
    """Joint OLS across positions; return per-position R²."""
    n_seq = posteriors.shape[0]
    seq_perm = torch.randperm(n_seq)
    use_split = (
        validation_split is not None
        and 0.0 < float(validation_split) < 1.0
        and n_seq > 1
    )
    if use_split:
        n_seq_train = max(1, int(n_seq * (1.0 - float(validation_split))))
        n_seq_train = min(n_seq_train, n_seq - 1)
        seq_tr = seq_perm[:n_seq_train]
        seq_va = seq_perm[n_seq_train:]
    else:
        # In-sample fit/eval (no train/val split).
        seq_tr = seq_perm
        seq_va = seq_perm

    post_maj = posteriors[:, :, :n_major].float()
    p = len(positions)
    n_vocab = int(real_tokens.max().item()) + 1
    use_pos_bias = include_position_bias and p > 1
    results: Dict[int, Dict[int, float]] = {}

    for layer in layers:
        h_layer = hiddens_by_layer[layer].float()  # (N, P, D)
        ytr = h_layer[seq_tr].reshape(-1, h_layer.shape[-1])
        yva = h_layer[seq_va].reshape(-1, h_layer.shape[-1])
        x_main_tr = post_maj[seq_tr].reshape(-1, post_maj.shape[-1])
        x_main_va = post_maj[seq_va].reshape(-1, post_maj.shape[-1])

        rt_tr = real_tokens[seq_tr].reshape(-1).long()
        rt_va = real_tokens[seq_va].reshape(-1).long()
        x_tok_tr = torch.zeros(rt_tr.shape[0], n_vocab, dtype=torch.float32)
        x_tok_tr.scatter_(1, rt_tr.unsqueeze(1), 1.0)
        x_tok_va = torch.zeros(rt_va.shape[0], n_vocab, dtype=torch.float32)
        x_tok_va.scatter_(1, rt_va.unsqueeze(1), 1.0)

        xtr_parts = [x_main_tr, x_tok_tr]
        xva_parts = [x_main_va, x_tok_va]
        if include_logit:
            xtr_parts.append(logits[seq_tr].reshape(-1, logits.shape[-1]).float())
            xva_parts.append(logits[seq_va].reshape(-1, logits.shape[-1]).float())
        if use_pos_bias:
            pos_tr = torch.arange(p).unsqueeze(0).expand(seq_tr.shape[0], p).reshape(-1)
            pos_va = torch.arange(p).unsqueeze(0).expand(seq_va.shape[0], p).reshape(-1)
            x_pos_tr = torch.zeros(pos_tr.shape[0], p, dtype=torch.float32)
            x_pos_tr.scatter_(1, pos_tr.unsqueeze(1), 1.0)
            x_pos_va = torch.zeros(pos_va.shape[0], p, dtype=torch.float32)
            x_pos_va.scatter_(1, pos_va.unsqueeze(1), 1.0)
            xtr_parts.append(x_pos_tr)
            xva_parts.append(x_pos_va)

        xtr = torch.cat(xtr_parts, dim=1)
        xva = torch.cat(xva_parts, dim=1)
        ones = torch.ones(xtr.shape[0], 1, dtype=xtr.dtype)
        w_aug = torch.linalg.pinv(torch.cat([xtr, ones], dim=1)) @ ytr
        w = w_aug[:-1, :]
        b = w_aug[-1, :]
        pred_va = (xva @ w + b).reshape(seq_va.shape[0], p, -1)
        yva_tensor = h_layer[seq_va]

        layer_results: Dict[int, float] = {}
        for p_idx, pos in enumerate(positions):
            yva_p = yva_tensor[:, p_idx, :]
            pred_p = pred_va[:, p_idx, :]
            ss_res = ((yva_p - pred_p) ** 2).sum().item()
            ss_tot = ((yva_p - yva_p.mean(dim=0)) ** 2).sum().item()
            layer_results[pos] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        results[layer] = layer_results
    return results


def plot_probe_fit_r2_latent(
    exp_name: str,
    layers: Optional[Sequence[int]] = None,
    batch_size: int = 64,
    n_samples: int = 2000,
    positions_of_interest: Optional[Sequence[int]] = None,
    n_minor: int = 0,
    step: Optional[int] = None,
    validation_split: Optional[float] = None,
    include_position_bias: bool = True,
    uniform_sampling: bool = True,
    sample_mode: str = "train",
    include_logit: bool = True,
    verbose: bool = False,
    figsize: tuple = (5, 3.2),
    log_x: bool = True,
    show: bool = True,
    show_ylabel: bool = True,
) -> dict:
    """Probe-fit R² by position using latent hidden predictor features.

    By default (`validation_split=None`) this reports in-sample fit R²
    (no train/val split). Set `validation_split` to a value in (0, 1)
    to report held-out validation R² instead.
    Uses a shared fit across selected positions; with
    `include_position_bias=True` adds one-hot position nuisance features.
    """
    import matplotlib.pyplot as plt
    from icl.latent_markov.analysis.probes import _collect_multi_layer_data

    if layers is None:
        _, _, config = nu.load_everything("latent", exp_name)
        layers = list(range(config.model.num_layers))
    else:
        layers = list(layers)

    if positions_of_interest is None:
        _, sampler, _ = nu.load_everything("latent", exp_name)
        positions = list(range(min(10, sampler.seq_len)))
    else:
        positions = list(positions_of_interest)

    data = _collect_multi_layer_data(
        exp_name=exp_name,
        layers=layers,
        B=batch_size,
        n_samples=n_samples,
        step=step,
        n_minor=n_minor,
        positions=positions,
        uniform_sampling=uniform_sampling,
        sample_mode=sample_mode,
        verbose=verbose,
    )

    r2_results = _fit_probe_r2_per_position_latent(
        hiddens_by_layer=data["hiddens_by_layer"],
        posteriors=data["posteriors"],
        logits=data["logits"],
        real_tokens=data["real_tokens"],
        n_major=data["n_major"],
        layers=layers,
        positions=positions,
        validation_split=validation_split,
        include_position_bias=include_position_bias,
        include_logit=include_logit,
    )

    _COLORS = [
        "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
        "#56B4E9", "#F0E442", "#000000",
    ]
    _LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

    fig, ax = plt.subplots(figsize=figsize)
    for i, layer in enumerate(sorted(r2_results.keys())):
        pos_list = sorted(r2_results[layer].keys())
        vals = [r2_results[layer][p] for p in pos_list]
        ax.plot(
            pos_list, vals, label=f"Layer {layer}",
            color=_COLORS[i % len(_COLORS)],
            linestyle=_LINESTYLES[i % len(_LINESTYLES)],
            linewidth=2.2,
        )

    ax.set_xlabel("Position", fontsize=13)
    if show_ylabel:
        ax.set_ylabel("Probe-fit $R^2$", fontsize=13)
    if log_x and len(positions) > 1 and min(positions) >= 0:
        ax.set_xscale("symlog", linthresh=1)
    ax.set_ylim(None, 1.02)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10, framealpha=0.9, loc="best",
              borderaxespad=0.3, handlelength=1.8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "r2_results": r2_results,
        "positions": positions,
        "layers": layers,
        "include_position_bias": include_position_bias,
        "include_logit": include_logit,
        "fig": fig,
    }


# ---------------------------------------------------------------------------
# 3.  plot_p1_variance (legacy metric, kept for backward compat)
# ---------------------------------------------------------------------------

def plot_p1_variance(
    exp_name: str,
    layers: Optional[Sequence] = None,
    batch_size: int = 16,
    positions_of_interest: Optional[list] = None,
    max_unique_tokens: Optional[int] = None,
    n_minor: int = 0,
    step: Optional[int] = None,
    verbose: bool = False,
    eps: float = 1e-8,
    figsize: tuple = (8, 6),
    log_x: bool = True,
    show: bool = True,
    title: Optional[str] = None,
    task_batch_size: int = 8,
    max_tasks: Optional[int] = None,
) -> dict:
    """
    Compute and plot P1 variance (normalized conditional residual variance)
    for the Latent Markov task on non-padded sequences.

    All-in-one convenience function that:

    1. Loads model/sampler.
    2. Collects token-conditioned hiddens via
       ``get_token_conditioned_hiddens``.
    3. Computes P1 variance via ``compute_p1_variance_multi_layer``.
    4. Plots normalised P1 variance vs position for each layer.

    Always uses ``n_ood=0``. ``n_minor`` is capped at ``sampler.n_minor_tasks``.

    Parameters
    ----------
    exp_name : str
    layers : list, optional
        ``None`` → all layers.
    batch_size : int
    positions_of_interest : list, optional
        ``None`` → all positions ``[0, seq_len-2]``.
    max_unique_tokens : int, optional
    n_minor : int
        Capped at ``sampler.n_minor_tasks``.
    step : int, optional
    verbose : bool
    eps : float
    figsize : tuple
    log_x : bool
        Use log scale on x-axis.
    show : bool
    title : str, optional
    task_batch_size : int
        Number of tasks to batch into a single forward pass.

    Returns
    -------
    dict
        ``{'all_hiddens', 'token_info', 'results_dict', 'plotting_data',
        'fig', 'ax'}``.
    """
    import matplotlib.pyplot as plt

    all_hiddens, token_info = _get_hiddens_cached(
        exp_name, layers, batch_size, positions_of_interest, n_minor, step, verbose,
        task_batch_size=task_batch_size,
        max_tasks=max_tasks,
    )

    L = all_hiddens.shape[0]
    if layers is None:
        _, _, config = nu.load_everything("latent", exp_name)
        layers = list(range(config.model.num_layers))

    results_dict = compute_p1_variance_multi_layer(
        all_hiddens=all_hiddens,
        token_info=token_info,
        layers=layers,
        eps=eps,
    )

    plotting_data = extract_p1_plotting_data_multi_layer(results_dict)

    if verbose:
        logger.info(f"Computed P1 variance for {len(results_dict)} layers")

    fig, ax = plt.subplots(figsize=figsize)

    for layer_idx in plotting_data['layers']:
        positions = plotting_data['positions']
        var_pos_norm = plotting_data['var_pos_norm'][layer_idx]
        ax.plot(
            positions, var_pos_norm, 'o-',
            label=f'Layer {layer_idx}',
            linewidth=2, markersize=6,
        )

    ax.set_xlabel('Position' + (' (log scale)' if log_x else ''), fontsize=16)
    ax.set_ylabel('Normalized P1 Variance', fontsize=16)
    if log_x:
        ax.set_xscale('log')
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()

    return {
        'all_hiddens': all_hiddens,
        'token_info': token_info,
        'results_dict': results_dict,
        'plotting_data': plotting_data,
        'fig': fig,
        'ax': ax,
    }


# ---------------------------------------------------------------------------
# 4.  ANOVA separability
# ---------------------------------------------------------------------------

def plot_anova_separability_latent(
    exp_name: str,
    layers: Optional[Sequence] = None,
    batch_size: int = 16,
    positions_of_interest: Optional[list] = None,
    n_minor: int = 0,
    step: Optional[int] = None,
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
    verbose: bool = False,
    figsize: tuple = (5, 3.2),
    log_x: bool = True,
    show: bool = True,
    show_ylabel: bool = True,
    print_summary: bool = True,
    task_batch_size: int = 8,
) -> dict:
    """Two-way ANOVA test of additive separability for the Latent Markov task.

    Tests whether θ_{k,a} ≈ θ_k + ν_a by decomposing the token-conditioned
    cell means into task main effect, token main effect, and interaction.

    For Markov chains the interaction is expected to be significant because
    the transition probability p_k(s_t, ·) couples task k with token s_t.

    Results are cached: calling this after ``plot_task_vector_r2_latent``
    with the same data parameters will not recompute hidden states.

    Parameters
    ----------
    exp_name : str
    layers : list, optional
        ``None`` → all layers.
    batch_size : int
    positions_of_interest : list, optional
        ``None`` → all positions ``[0, seq_len-2]``.
    n_minor : int
        Capped at ``sampler.n_minor_tasks``.
    step : int, optional
    post_layernorm : bool
    extraction_point : str
        ``"post_attn"`` (default) or ``"post_mlp"`` (residual stream).
    verbose : bool
    figsize : tuple
    log_x : bool
    show : bool
    print_summary : bool
    task_batch_size : int
        Number of tasks to batch into a single forward pass.

    Returns
    -------
    dict
        ``{'all_hiddens', 'token_info', 'anova_results',
        'fig_interaction', 'fig_separability'}``.
    """
    from icl.utils.separability import (
        anova_separability_multi,
        plot_anova_separability,
        print_anova_summary,
    )

    all_hiddens, token_info = _get_hiddens_cached(
        exp_name, layers, batch_size, positions_of_interest, n_minor, step, verbose,
        task_batch_size=task_batch_size,
        post_layernorm=post_layernorm,
        extraction_point=extraction_point,
    )

    if layers is None:
        _, _, config = nu.load_everything("latent", exp_name)
        layers = list(range(config.model.num_layers))

    anova_results = anova_separability_multi(
        all_hiddens=all_hiddens,
        token_info=token_info,
        layers=layers,
        positions=positions_of_interest,
    )

    if print_summary:
        print_anova_summary(anova_results, positions=positions_of_interest)

    fig_int, fig_sep = plot_anova_separability(
        anova_results,
        figsize=figsize,
        log_x=log_x,
        show=show,
        show_ylabel=show_ylabel,
    )

    return {
        "all_hiddens": all_hiddens,
        "token_info": token_info,
        "anova_results": anova_results,
        "fig_interaction": fig_int,
        "fig_separability": fig_sep,
    }
