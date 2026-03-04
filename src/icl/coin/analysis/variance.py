"""Task-vector R², P1 variance, and ANOVA separability for the Coin task.

Task variance functions have been moved to legacy:
    ``icl.coin.legacy.task_variance_coin``
"""

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
) -> tuple:
    """Return (all_hiddens, token_info), reusing cache when parameters match."""
    key = (
        exp_name,
        tuple(layers) if layers is not None else None,
        batch_size,
        tuple(positions_of_interest) if positions_of_interest is not None else None,
        n_minor,
        step,
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
    verbose: bool = False,
    figsize: tuple = (6, 4),
    log_x: bool = True,
    show: bool = True,
    show_ylabel: bool = True,
    print_summary: bool = True,
) -> dict:
    """Task-token vector R² for the Coin task.

    Measures what fraction of hidden-state variance at each position
    is explained by knowing (task, current token).  R² → 1 indicates
    long-context stability.

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
    verbose : bool
    figsize, log_x, show, print_summary : plot options

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


def _fit_probe_r2_per_position_coin(
    hiddens_by_layer: Dict[int, torch.Tensor],
    posteriors_all: torch.Tensor,
    real_tokens_all: torch.Tensor,
    layers: Sequence[int],
    positions: Sequence[int],
    validation_split: float = 0.2,
) -> Dict[int, Dict[int, float]]:
    """Fit OLS h ~ [posterior, one_hot(token)] and return val R² per position."""
    n_seq = posteriors_all.shape[0]
    n_seq_train = max(1, int(n_seq * (1.0 - validation_split)))
    n_seq_train = min(n_seq_train, n_seq - 1) if n_seq > 1 else n_seq_train
    seq_perm = torch.randperm(n_seq)
    seq_tr = seq_perm[:n_seq_train]
    seq_va = seq_perm[n_seq_train:]

    results: Dict[int, Dict[int, float]] = {}
    for layer in layers:
        h_layer = hiddens_by_layer[layer].float()  # (N, P, D)
        layer_results: Dict[int, float] = {}
        for p_idx, pos in enumerate(positions):
            ytr = h_layer[seq_tr, p_idx, :]
            yva = h_layer[seq_va, p_idx, :]
            x_main_tr = posteriors_all[seq_tr, p_idx, :].float()
            x_main_va = posteriors_all[seq_va, p_idx, :].float()

            rt_tr = real_tokens_all[seq_tr, p_idx].long()
            rt_va = real_tokens_all[seq_va, p_idx].long()
            n_vocab = int(real_tokens_all[:, p_idx].max().item()) + 1
            x_tok_tr = torch.zeros(rt_tr.shape[0], n_vocab, dtype=torch.float32)
            x_tok_tr.scatter_(1, rt_tr.unsqueeze(1), 1.0)
            x_tok_va = torch.zeros(rt_va.shape[0], n_vocab, dtype=torch.float32)
            x_tok_va.scatter_(1, rt_va.unsqueeze(1), 1.0)

            xtr = torch.cat([x_main_tr, x_tok_tr], dim=1)
            xva = torch.cat([x_main_va, x_tok_va], dim=1)
            ones = torch.ones(xtr.shape[0], 1, dtype=xtr.dtype)
            w_aug = torch.linalg.pinv(torch.cat([xtr, ones], dim=1)) @ ytr
            w = w_aug[:-1, :]
            b = w_aug[-1, :]
            pred_va = xva @ w + b

            ss_res = ((yva - pred_va) ** 2).sum().item()
            ss_tot = ((yva - yva.mean(dim=0)) ** 2).sum().item()
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
    uniform_sampling: bool = True,
    sample_mode: str = "train",
    verbose: bool = False,
    figsize: tuple = (6, 4),
    log_x: bool = True,
    show: bool = True,
    show_ylabel: bool = True,
) -> dict:
    """Probe-fit R² by position using OLS (posterior + token) for Coin."""
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
        verbose=verbose,
    )

    r2_results = _fit_probe_r2_per_position_coin(
        hiddens_by_layer=data["hiddens_by_layer"],
        posteriors_all=data["posteriors_all"],
        real_tokens_all=data["real_tokens_all"],
        layers=layers,
        positions=positions,
        validation_split=validation_split,
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


# ---------------------------------------------------------------------------
# Legacy re-exports (backward compatibility)
# ---------------------------------------------------------------------------
from icl.coin.legacy.task_variance_coin import (  # noqa: F401, E402
    get_task_variance_coin,
    plot_task_variance_coin,
)
