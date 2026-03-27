"""Two-way ANOVA: additive separability for discrete tokens."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch


@dataclass
class ANOVAResult:
    """ANOVA decomposition for a single (layer, position)."""

    ss_total: float
    ss_task: float
    ss_token: float
    ss_interaction: float

    eta2_task: float
    eta2_token: float
    eta2_interaction: float

    separability_r2: float

    mse_within: float

    n_tasks: int
    n_tokens: int
    n_batch: int

    layer_num: Optional[int] = None
    position: Optional[int] = None


def anova_separability(
    cell_hiddens: torch.Tensor,
    eps: float = 1e-10,
) -> ANOVAResult:
    """Core two-way ANOVA on a single (layer, position) slice.

    Parameters
    ----------
    cell_hiddens : torch.Tensor
        Shape ``(n_tokens, n_tasks, batch_size, D)``.
        Raw hidden states *before* averaging over batch.
    eps : float
        Guard against division by zero.

    Returns
    -------
    ANOVAResult
    """
    n_tokens, n_tasks, B, D = cell_hiddens.shape
    h = cell_hiddens.float()

    # ---- cell means: θ̂_{a,k} ----
    cell_means = h.mean(dim=2)  # (|V|, K, D)

    # ---- marginals ----
    task_mean = cell_means.mean(dim=0)   # θ̂_{.k}  (K, D)
    token_mean = cell_means.mean(dim=1)  # θ̂_{a.}  (|V|, D)
    grand_mean = cell_means.mean(dim=(0, 1))  # θ̂_{..}  (D,)

    # ---- effects ----
    tau = task_mean - grand_mean          # (K, D)
    nu = token_mean - grand_mean          # (|V|, D)
    interaction = (
        cell_means
        - task_mean.unsqueeze(0)
        - token_mean.unsqueeze(1)
        + grand_mean
    )  # (|V|, K, D)

    # ---- sum of squares ----
    ss_task = n_tokens * (tau ** 2).sum().item()
    ss_token = n_tasks * (nu ** 2).sum().item()
    ss_interaction = (interaction ** 2).sum().item()
    ss_total = ss_task + ss_token + ss_interaction

    # ---- proportions ----
    eta2_task = ss_task / (ss_total + eps)
    eta2_token = ss_token / (ss_total + eps)
    eta2_interaction = ss_interaction / (ss_total + eps)
    sep_r2 = 1.0 - eta2_interaction

    # ---- within-cell variance (noise floor) ----
    residuals = h - cell_means.unsqueeze(2)  # (|V|, K, B, D)
    mse_within = (residuals ** 2).sum(dim=-1).mean().item()  # avg over all cells and batch

    return ANOVAResult(
        ss_total=ss_total,
        ss_task=ss_task,
        ss_token=ss_token,
        ss_interaction=ss_interaction,
        eta2_task=eta2_task,
        eta2_token=eta2_token,
        eta2_interaction=eta2_interaction,
        separability_r2=sep_r2,
        mse_within=mse_within,
        n_tasks=n_tasks,
        n_tokens=n_tokens,
        n_batch=B,
    )


def anova_separability_multi(
    all_hiddens: torch.Tensor,
    token_info: dict,
    layers: Optional[Sequence[int]] = None,
    positions: Optional[Sequence[int]] = None,
    eps: float = 1e-10,
) -> Dict[int, Dict[int, ANOVAResult]]:
    """Run ANOVA across multiple layers and positions.

    Parameters
    ----------
    all_hiddens : torch.Tensor
        Shape ``(L, n_positions, n_tokens, n_tasks, batch_size, D)``.
        Output of ``compute_hiddens_token_conditioned_coin`` (or latent
        equivalent).
    token_info : dict
        Must contain ``'positions'`` (list of position indices) and
        ``'n_unique_tokens'`` (dict mapping position to count).
    layers : sequence of int, optional
        Layer *numbers* (not indices). ``None`` → ``range(L)``.
    positions : sequence of int, optional
        Subset of positions to analyse.  ``None`` → all in *token_info*.
    eps : float

    Returns
    -------
    dict
        ``{layer_num: {position: ANOVAResult, ...}, ...}``
    """
    L = all_hiddens.shape[0]
    all_positions = token_info["positions"]

    if layers is None:
        layers = list(range(L))
    if positions is None:
        positions = list(all_positions)

    results: Dict[int, Dict[int, ANOVAResult]] = {}

    for l_idx, l_num in enumerate(layers):
        if l_idx >= L:
            continue
        layer_results: Dict[int, ANOVAResult] = {}

        for pos in positions:
            if pos not in all_positions:
                continue
            pos_idx = all_positions.index(pos)
            n_unique = token_info["n_unique_tokens"].get(pos, all_hiddens.shape[2])

            cell_hiddens = all_hiddens[l_idx, pos_idx, :n_unique]  # (V', K, B, D)

            res = anova_separability(cell_hiddens, eps=eps)
            res.layer_num = l_num
            res.position = pos
            layer_results[pos] = res

        results[l_num] = layer_results

    return results


def plot_anova_separability(
    results: Dict[int, Dict[int, ANOVAResult]],
    figsize: tuple = (6, 4),
    log_x: bool = True,
    show: bool = True,
    show_ylabel: bool = True,
):
    """Plot η²_interaction and separability R² as two separate figures.

    Parameters
    ----------
    results : dict
        Output of :func:`anova_separability_multi`.

    Returns
    -------
    (fig_interaction, fig_sep) : tuple of Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np  # noqa: F401

    layers = sorted(results.keys())
    if not layers:
        return None, None

    _COLORS = [
        "#0072B2",  # blue
        "#E69F00",  # amber
        "#009E73",  # teal
        "#D55E00",  # vermillion
        "#CC79A7",  # pink
        "#56B4E9",  # sky blue
        "#F0E442",  # yellow
        "#000000",  # black
    ]
    _LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

    def _style(i):
        return dict(
            color=_COLORS[i % len(_COLORS)],
            linestyle=_LINESTYLES[i % len(_LINESTYLES)],
            linewidth=2.2,
        )

    # ---- Figure 1: η²_interaction ----
    fig1, ax1 = plt.subplots(figsize=figsize)
    pos_list = []
    for i, l_num in enumerate(layers):
        pos_results = results[l_num]
        pos_list = sorted(pos_results.keys())
        if not pos_list:
            continue
        eta2_int = [pos_results[p].eta2_interaction for p in pos_list]
        ax1.plot(pos_list, eta2_int, label=f"Layer {l_num}", **_style(i))

    ax1.set_xlabel("Position", fontsize=14)
    if show_ylabel:
        ax1.set_ylabel("$\\eta^2_{\\mathrm{interaction}}$", fontsize=14)
    if log_x and len(pos_list) > 1 and min(pos_list) >= 0:
        ax1.set_xscale("symlog", linthresh=1)
    ax1.set_ylim(-0.02, None)
    ax1.tick_params(labelsize=12)
    ax1.legend(fontsize=10, framealpha=0.9, loc="best",
               borderaxespad=0.3, handlelength=1.8)
    ax1.grid(True, alpha=0.25, linewidth=0.5)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig1)

    # ---- Figure 2: separability R² ----
    fig2, ax2 = plt.subplots(figsize=figsize)
    for i, l_num in enumerate(layers):
        pos_results = results[l_num]
        pos_list = sorted(pos_results.keys())
        if not pos_list:
            continue
        sep_r2 = [pos_results[p].separability_r2 for p in pos_list]
        ax2.plot(pos_list, sep_r2, label=f"Layer {l_num}", **_style(i))

    ax2.set_xlabel("Position", fontsize=14)
    if show_ylabel:
        ax2.set_ylabel("Separability $R^2$", fontsize=14)
    if log_x and len(pos_list) > 1 and min(pos_list) >= 0:
        ax2.set_xscale("symlog", linthresh=1)
    ax2.set_ylim(None, 1.00)
    ax2.tick_params(labelsize=12)
    ax2.legend(fontsize=10, framealpha=0.9, loc="best",
               borderaxespad=0.3, handlelength=1.8)
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig2)

    return fig1, fig2


def print_anova_summary(
    results: Dict[int, Dict[int, ANOVAResult]],
    positions: Optional[Sequence[int]] = None,
):
    """Print a formatted summary table."""
    layers = sorted(results.keys())
    if not layers:
        return

    sample_layer = results[layers[0]]
    all_pos = sorted(sample_layer.keys())
    if positions is not None:
        all_pos = [p for p in all_pos if p in positions]

    header = (
        f"{'Layer':>6} {'Pos':>5} {'η²_task':>9} {'η²_token':>9} "
        f"{'η²_inter':>9} {'sep R²':>8} {'MSE_w':>10}"
    )
    print("=" * len(header))
    print("  Two-way ANOVA: additive separability of θ_{k,a}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for l_num in layers:
        pos_results = results[l_num]
        for pos in all_pos:
            if pos not in pos_results:
                continue
            r = pos_results[pos]
            print(
                f"{l_num:>6} {pos:>5} {r.eta2_task:>9.4f} {r.eta2_token:>9.4f} "
                f"{r.eta2_interaction:>9.4f} {r.separability_r2:>8.4f} "
                f"{r.mse_within:>10.2f}"
            )

    print("=" * len(header))
