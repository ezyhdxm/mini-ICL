"""Task-vector R², P1 variance, and ANOVA separability for the Coin task."""

from typing import Dict, Optional, Sequence

import icl.utils.notebook_utils as nu
import torch
from icl.utils.logger import setup_logger
from icl.linear.p1_variance import compute_p1_variance_multi_layer
from icl.linear.p1_variance import extract_plotting_data_multi_layer as extract_p1_plotting_data_multi_layer
from icl.coin.analysis._helpers import get_token_conditioned_hiddens_coin

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Hiddens cache (holds one entry to avoid recomputation across analyses)
# ---------------------------------------------------------------------------
_hiddens_cache: dict = {}


def _get_hiddens_cached(
    exp_name: str,
    layers: Optional[list],
    batch_size: int,
    positions_of_interest: Optional[list],
    n_minor: int,
    step: Optional[int],
    verbose: bool,
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
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
    )
    if key in _hiddens_cache:
        if verbose:
            logger.info("[coin cache] reusing cached token-conditioned hiddens")
        return _hiddens_cache[key]

    result = get_token_conditioned_hiddens_coin(
        exp_name=exp_name,
        layers=layers,
        batch_size=batch_size,
        positions_of_interest=positions_of_interest,
        n_minor=n_minor,
        step=step,
        verbose=verbose,
        post_layernorm=post_layernorm,
        extraction_point=extraction_point,
    )
    _hiddens_cache.clear()
    _hiddens_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Task-vector R²
# ---------------------------------------------------------------------------

def plot_task_vector_r2_coin(
    exp_name: str,
    layers: Optional[list] = None,
    batch_size: int = 16,
    positions_of_interest: Optional[list] = None,
    n_minor: int = 0,
    step: Optional[int] = None,
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
    verbose: bool = False,
    figsize: tuple = (6, 4),
    log_x: bool = True,
    show: bool = True,
    show_ylabel: bool = True,
    pool_positions: bool = False,
    print_summary: bool = True,
) -> dict:
    """Task-token vector R² for the Coin task.

    Measures what fraction of hidden-state variance at each position
    is explained by knowing (task, current token).  R² → 1 indicates
    long-context stability.

    If ``pool_positions=True``, sums SS_total and SS_within across all
    selected positions first, then computes a pooled R² per layer:
        R²_pool = 1 - (Σ_pos SS_within) / (Σ_pos SS_total).
    The pooled value is displayed as a flat line across positions.

    Results are cached: calling this followed by
    ``plot_anova_separability_coin`` with the same data parameters
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
    figsize, log_x, show, print_summary : plot options

    Returns
    -------
    dict
        ``{'all_hiddens', 'token_info', 'r2_results', 'fig'}``.
    """
    from icl.utils.separability import (
        TaskVectorR2Result,
        task_vector_r2_multi,
        plot_task_vector_r2,
        print_task_vector_r2_summary,
    )

    all_hiddens, token_info = _get_hiddens_cached(
        exp_name, layers, batch_size, positions_of_interest, n_minor, step, verbose,
        post_layernorm=post_layernorm,
        extraction_point=extraction_point,
    )

    if layers is None:
        _, _, config = nu.load_everything("coin", exp_name)
        layers = list(range(config.model.num_layers))

    r2_results = task_vector_r2_multi(
        all_hiddens=all_hiddens,
        token_info=token_info,
        layers=layers,
        positions=positions_of_interest,
    )

    if pool_positions:
        pooled_results = {}
        pos_used = (
            list(positions_of_interest)
            if positions_of_interest is not None
            else list(token_info["positions"])
        )
        for l_num, pos_dict in r2_results.items():
            valid_pos = [p for p in pos_used if p in pos_dict]
            if not valid_pos:
                pooled_results[l_num] = {}
                continue
            ss_total = sum(pos_dict[p].ss_total for p in valid_pos)
            ss_within = sum(pos_dict[p].ss_within for p in valid_pos)
            r2_pool = 1.0 - ss_within / (ss_total + 1e-10)
            ref = pos_dict[valid_pos[0]]
            pooled_results[l_num] = {}
            for p in valid_pos:
                pooled_results[l_num][p] = TaskVectorR2Result(
                    r2=r2_pool,
                    ss_total=ss_total,
                    ss_between=ss_total - ss_within,
                    ss_within=ss_within,
                    n_tasks=ref.n_tasks,
                    n_tokens=ref.n_tokens,
                    n_batch=ref.n_batch,
                    layer_num=l_num,
                    position=p,
                )
        r2_results = pooled_results

    if print_summary:
        if pool_positions:
            print("\nPooled task-token vector R² across positions:")
            for l_num in sorted(r2_results.keys()):
                vals = list(r2_results[l_num].values())
                if not vals:
                    continue
                print(f"  Layer {l_num}: R²_pool = {vals[0].r2:.4f}")
        else:
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


def _fit_probe_r2_per_position_coin(
    hiddens_by_layer: Dict[int, torch.Tensor],
    posteriors_all: torch.Tensor,
    real_tokens_all: torch.Tensor,
    layers: Sequence[int],
    positions: Sequence[int],
    validation_split: float = 0.2,
    include_position_bias: bool = True,
) -> Dict[int, Dict[int, float]]:
    """Joint OLS across positions; return per-position val R²."""
    n_seq = posteriors_all.shape[0]
    n_seq_train = max(1, int(n_seq * (1.0 - validation_split)))
    n_seq_train = min(n_seq_train, n_seq - 1) if n_seq > 1 else n_seq_train
    seq_perm = torch.randperm(n_seq)
    seq_tr = seq_perm[:n_seq_train]
    seq_va = seq_perm[n_seq_train:]

    p = len(positions)
    n_vocab = int(real_tokens_all.max().item()) + 1
    use_pos_bias = include_position_bias and p > 1
    results: Dict[int, Dict[int, float]] = {}

    for layer in layers:
        h_layer = hiddens_by_layer[layer].float()  # (N, P, D)
        ytr = h_layer[seq_tr].reshape(-1, h_layer.shape[-1])
        yva = h_layer[seq_va].reshape(-1, h_layer.shape[-1])

        x_main_tr = posteriors_all[seq_tr].reshape(-1, posteriors_all.shape[-1]).float()
        x_main_va = posteriors_all[seq_va].reshape(-1, posteriors_all.shape[-1]).float()

        rt_tr = real_tokens_all[seq_tr].reshape(-1).long()
        rt_va = real_tokens_all[seq_va].reshape(-1).long()
        x_tok_tr = torch.zeros(rt_tr.shape[0], n_vocab, dtype=torch.float32)
        x_tok_tr.scatter_(1, rt_tr.unsqueeze(1), 1.0)
        x_tok_va = torch.zeros(rt_va.shape[0], n_vocab, dtype=torch.float32)
        x_tok_va.scatter_(1, rt_va.unsqueeze(1), 1.0)

        xtr_parts = [x_main_tr, x_tok_tr]
        xva_parts = [x_main_va, x_tok_va]
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


def plot_probe_fit_r2_coin(
    exp_name: str,
    layers: Optional[Sequence[int]] = None,
    batch_size: int = 64,
    n_samples: int = 2000,
    positions_of_interest: Optional[Sequence[int]] = None,
    n_minor: int = 0,
    step: Optional[int] = None,
    validation_split: float = 0.2,
    include_position_bias: bool = True,
    uniform_sampling: bool = True,
    sample_mode: str = "train",
    extraction_point: str = "post_attn",
    verbose: bool = False,
    figsize: tuple = (6, 4),
    log_x: bool = True,
    show: bool = True,
    show_ylabel: bool = True,
) -> dict:
    """Probe-fit R² by position using joint OLS for Coin.

    Fits one shared model across selected positions and reports per-position
    R². Optionally adds one-hot position nuisance features so the baseline
    hidden mean can vary by position.
    """
    import matplotlib.pyplot as plt
    from icl.coin.analysis.probes import _collect_coin_probe_data

    if layers is None:
        _, _, config = nu.load_everything("coin", exp_name)
        layers = list(range(config.model.num_layers))
    else:
        layers = list(layers)

    if positions_of_interest is None:
        _, sampler, _ = nu.load_everything("coin", exp_name)
        positions = list(range(min(10, sampler.seq_len)))
    else:
        positions = list(positions_of_interest)

    data = _collect_coin_probe_data(
        exp_name=exp_name,
        layers=layers,
        B=batch_size,
        n_samples=n_samples,
        step=step,
        n_minor=n_minor,
        positions=positions,
        uniform_sampling=uniform_sampling,
        sample_mode=sample_mode,
        extraction_point=extraction_point,
        verbose=verbose,
    )

    r2_results = _fit_probe_r2_per_position_coin(
        hiddens_by_layer=data["hiddens_by_layer"],
        posteriors_all=data["posteriors_all"],
        real_tokens_all=data["real_tokens_all"],
        layers=layers,
        positions=positions,
        validation_split=validation_split,
        include_position_bias=include_position_bias,
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

    ax.set_xlabel("Position", fontsize=14)
    if show_ylabel:
        ax.set_ylabel("Probe-fit $R^2$", fontsize=14)
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
        "fig": fig,
    }


# ---------------------------------------------------------------------------
# P1 variance (legacy metric, kept for backward compat)
# ---------------------------------------------------------------------------

def plot_p1_variance_coin(
    exp_name: str,
    layers: Optional[list] = None,
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
) -> dict:
    """
    Compute and plot P1 variance (normalized conditional residual variance)
    for the Coin task.

    All-in-one convenience function that:

    1. Loads model/sampler.
    2. Collects token-conditioned hiddens via
       ``get_token_conditioned_hiddens_coin``.
    3. Computes P1 variance via ``compute_p1_variance_multi_layer``.
    4. Plots normalised P1 variance vs position for each layer.

    Always uses ``n_ood=0``.  ``n_minor`` is capped at
    ``sampler.n_minor_tasks``.

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

    Returns
    -------
    dict
        ``{'all_hiddens', 'token_info', 'results_dict', 'plotting_data',
        'fig', 'ax'}``.
    """
    import matplotlib.pyplot as plt

    all_hiddens, token_info = _get_hiddens_cached(
        exp_name, layers, batch_size, positions_of_interest, n_minor, step, verbose,
    )

    L = all_hiddens.shape[0]
    if layers is None:
        _, _, config = nu.load_everything("coin", exp_name)
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
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title("", fontsize=18)

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
# ANOVA separability
# ---------------------------------------------------------------------------

def plot_anova_separability_coin(
    exp_name: str,
    layers: Optional[list] = None,
    batch_size: int = 16,
    positions_of_interest: Optional[list] = None,
    n_minor: int = 0,
    step: Optional[int] = None,
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
    verbose: bool = False,
    figsize: tuple = (6, 4),
    log_x: bool = True,
    show: bool = True,
    show_ylabel: bool = True,
    print_summary: bool = True,
) -> dict:
    """Two-way ANOVA test of additive separability for the Coin task.

    Tests whether θ_{k,a} ≈ θ_k + ν_a by decomposing the token-conditioned
    cell means into task main effect, token main effect, and interaction.

    Results are cached: calling this after ``plot_task_vector_r2_coin``
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
        post_layernorm=post_layernorm,
        extraction_point=extraction_point,
    )

    if layers is None:
        _, _, config = nu.load_everything("coin", exp_name)
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


