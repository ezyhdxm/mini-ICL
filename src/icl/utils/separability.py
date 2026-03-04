"""
Task-vector evaluation metrics: R², additive separability (ANOVA / ANCOVA).

- **Task-vector R²**: fraction of hidden-state variance explained by
  (task, token), testing long-context stability.
- **ANOVA separability**: tests θ_{k,a} = θ_k + ν_a for discrete tokens.
- **ANCOVA separability**: tests slope homogeneity for continuous covariates.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch


# =========================================================================
# Task-vector R²: fraction of h variance explained by (task, token)
# =========================================================================


@dataclass
class TaskVectorR2Result:
    """R² for a single (layer, position)."""

    r2: float
    ss_total: float
    ss_between: float
    ss_within: float

    n_tasks: int
    n_tokens: int
    n_batch: int

    layer_num: Optional[int] = None
    position: Optional[int] = None


def task_vector_r2(
    cell_hiddens: torch.Tensor,
    eps: float = 1e-10,
) -> TaskVectorR2Result:
    """Compute R² = 1 − SS_within / SS_total for a single (layer, position).

    Measures what fraction of hidden-state variance is explained by
    knowing (task, token).  Uses the law-of-total-variance decomposition:
        SS_total = SS_between + SS_within
    where SS_between captures variation of cell means θ̂_{k,a} and
    SS_within is the residual after conditioning on (k, a).

    Parameters
    ----------
    cell_hiddens : torch.Tensor
        Shape ``(n_tokens, n_tasks, batch_size, D)``.
    eps : float

    Returns
    -------
    TaskVectorR2Result
    """
    h = cell_hiddens.float()
    V, K, B, D = h.shape

    grand_mean = h.mean(dim=(0, 1, 2))  # (D,)
    cell_means = h.mean(dim=2)  # (V, K, D)

    ss_total = ((h - grand_mean) ** 2).sum().item()
    ss_within = ((h - cell_means.unsqueeze(2)) ** 2).sum().item()
    ss_between = ss_total - ss_within

    r2 = 1.0 - ss_within / (ss_total + eps)

    return TaskVectorR2Result(
        r2=r2,
        ss_total=ss_total,
        ss_between=ss_between,
        ss_within=ss_within,
        n_tasks=K,
        n_tokens=V,
        n_batch=B,
    )


def task_vector_r2_multi(
    all_hiddens: torch.Tensor,
    token_info: dict,
    layers: Optional[Sequence[int]] = None,
    positions: Optional[Sequence[int]] = None,
    eps: float = 1e-10,
) -> Dict[int, Dict[int, TaskVectorR2Result]]:
    """Compute task-vector R² across layers and positions.

    Parameters
    ----------
    all_hiddens : torch.Tensor
        Shape ``(L, n_positions, n_tokens, n_tasks, batch_size, D)``.
    token_info : dict
        Must contain ``'positions'`` and ``'n_unique_tokens'``.
    layers, positions : optional
        Subsets to analyse.  ``None`` → all.

    Returns
    -------
    dict
        ``{layer_num: {position: TaskVectorR2Result, ...}, ...}``
    """
    L = all_hiddens.shape[0]
    all_positions = token_info["positions"]

    if layers is None:
        layers = list(range(L))
    if positions is None:
        positions = list(all_positions)

    results: Dict[int, Dict[int, TaskVectorR2Result]] = {}

    for l_idx, l_num in enumerate(layers):
        if l_idx >= L:
            continue
        layer_results: Dict[int, TaskVectorR2Result] = {}

        for pos in positions:
            if pos not in all_positions:
                continue
            pos_idx = all_positions.index(pos)
            n_unique = token_info["n_unique_tokens"].get(pos, all_hiddens.shape[2])

            cell_hiddens = all_hiddens[l_idx, pos_idx, :n_unique]

            res = task_vector_r2(cell_hiddens, eps=eps)
            res.layer_num = l_num
            res.position = pos
            layer_results[pos] = res

        results[l_num] = layer_results

    return results


def plot_task_vector_r2(
    results: Dict[int, Dict[int, TaskVectorR2Result]],
    figsize: tuple = (6, 4),
    log_x: bool = True,
    show: bool = True,
    show_ylabel: bool = True,
):
    """Plot task-token vector R² across positions.

    Returns
    -------
    fig : Figure
    """
    import matplotlib.pyplot as plt

    layers = sorted(results.keys())
    if not layers:
        return None

    _COLORS = [
        "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
        "#56B4E9", "#F0E442", "#000000",
    ]
    _LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

    def _style(i):
        return dict(
            color=_COLORS[i % len(_COLORS)],
            linestyle=_LINESTYLES[i % len(_LINESTYLES)],
            linewidth=2.2,
        )

    fig, ax = plt.subplots(figsize=figsize)
    pos_list = []
    for i, l_num in enumerate(layers):
        pos_results = results[l_num]
        pos_list = sorted(pos_results.keys())
        if not pos_list:
            continue
        r2_vals = [pos_results[p].r2 for p in pos_list]
        ax.plot(pos_list, r2_vals, label=f"Layer {l_num}", **_style(i))

    ax.set_xlabel("Position", fontsize=14)
    if show_ylabel:
        ax.set_ylabel("Task-token vector $R^2$", fontsize=14)
    if log_x and len(pos_list) > 1 and min(pos_list) >= 0:
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

    return fig


def print_task_vector_r2_summary(
    results: Dict[int, Dict[int, TaskVectorR2Result]],
    positions: Optional[Sequence[int]] = None,
):
    """Print a formatted task-vector R² table."""
    layers = sorted(results.keys())
    if not layers:
        return

    sample_layer = results[layers[0]]
    all_pos = sorted(sample_layer.keys())
    if positions is not None:
        all_pos = [p for p in all_pos if p in positions]

    header = f"{'Layer':>6} {'Pos':>5} {'R²':>8} {'SS_betw':>12} {'SS_with':>12}"
    print("=" * len(header))
    print("  Task-vector R²: Var(E[h|z,s_t]) / Var(h)")
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
                f"{l_num:>6} {pos:>5} {r.r2:>8.4f} "
                f"{r.ss_between:>12.1f} {r.ss_within:>12.1f}"
            )

    print("=" * len(header))


# =========================================================================
# Two-way ANOVA: additive separability for discrete tokens
# =========================================================================


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
    import numpy as np

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
    ax2.set_ylim(None, 1.005)
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


# =========================================================================
# ANCOVA: additive separability for continuous covariates (linear regression)
# =========================================================================


@dataclass
class ANCOVAResult:
    """ANCOVA slope-homogeneity test for a single (layer, position)."""

    r2_additive: float
    r2_full: float
    separability_gap: float

    ss_total: float
    ss_res_additive: float
    ss_res_full: float

    n_tasks: int
    n_covariate_dims: int
    n_samples: int

    layer_num: Optional[int] = None
    position: Optional[int] = None


def ancova_separability(
    hiddens: torch.Tensor,
    covariates: torch.Tensor,
    task_labels: torch.Tensor,
    eps: float = 1e-10,
) -> ANCOVAResult:
    """ANCOVA slope-homogeneity test for a single (layer, position).

    Compares an additive model (common slopes across tasks) against a
    full interaction model (task-specific slopes) to quantify whether
    the effect of the continuous covariate x_t on h is task-independent.

    The additive model is::

        h = one_hot(k) @ W_task + x_t @ W_x + b

    The full model uses per-task slopes with no redundant columns::

        h = one_hot(k) @ W_task + [one_hot(k) ⊗ x_t] @ W_int + b

    (The shared x_t term is dropped from the full model because it
    is in the column span of the interaction terms.)

    All computations use float64 for numerical stability.

    Parameters
    ----------
    hiddens : torch.Tensor
        Shape ``(N, D)`` — hidden states.
    covariates : torch.Tensor
        Shape ``(N, d)`` — continuous covariates (e.g. x_t).
    task_labels : torch.Tensor
        Shape ``(N,)`` — integer task labels in ``{0, ..., K-1}``.
    eps : float

    Returns
    -------
    ANCOVAResult
    """
    h = hiddens.double()
    x = covariates.double()
    labels = task_labels.long()

    N, D = h.shape
    d = x.shape[1]
    K = int(labels.max().item()) + 1

    one_hot_k = torch.zeros(N, K, dtype=torch.float64, device=h.device)
    one_hot_k.scatter_(1, labels.unsqueeze(1), 1.0)

    def _ols_r2(X, Y):
        ones = torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)
        X_aug = torch.cat([X, ones], dim=1)
        W = torch.linalg.lstsq(X_aug, Y).solution
        pred = X_aug @ W
        ss_res = ((Y - pred) ** 2).sum().item()
        ss_tot = ((Y - Y.mean(dim=0)) ** 2).sum().item()
        r2 = 1.0 - ss_res / (ss_tot + eps)
        return r2, ss_res, ss_tot

    # Additive: h = one_hot(k) @ W_task + x_t @ W_x + b
    X_add = torch.cat([one_hot_k, x], dim=1)  # (N, K + d)
    r2_add, ss_res_add, ss_tot = _ols_r2(X_add, h)

    # Full: h = one_hot(k) @ W_task + [one_hot(k) ⊗ x_t] @ W_int + b
    # The interaction columns subsume the shared x columns (Σ_k oh_k⊗x = x),
    # so we omit the standalone x to avoid rank deficiency.
    interaction = one_hot_k.unsqueeze(2) * x.unsqueeze(1)  # (N, K, d)
    interaction = interaction.reshape(N, K * d)
    X_full = torch.cat([one_hot_k, interaction], dim=1)  # (N, K + K*d)
    r2_full, ss_res_full, _ = _ols_r2(X_full, h)

    gap = r2_full - r2_add

    return ANCOVAResult(
        r2_additive=r2_add,
        r2_full=r2_full,
        separability_gap=gap,
        ss_total=ss_tot,
        ss_res_additive=ss_res_add,
        ss_res_full=ss_res_full,
        n_tasks=K,
        n_covariate_dims=d,
        n_samples=N,
    )


def ancova_separability_from_hiddens(
    all_hiddens: torch.Tensor,
    demo_data: torch.Tensor,
    layers: Optional[Sequence[int]] = None,
    positions: Optional[Sequence[int]] = None,
    eps: float = 1e-10,
) -> Dict[int, Dict[int, ANCOVAResult]]:
    """Run ANCOVA across layers and positions for linear regression.

    Parameters
    ----------
    all_hiddens : torch.Tensor
        Shape ``(L, n_tasks, n_points, batch_size, D)``.
    demo_data : torch.Tensor
        Shape ``(batch_size, n_points, n_dims)``.
        Input vectors x_t (shared across tasks).
    layers : sequence of int, optional
        Layer numbers.  ``None`` → ``range(L)``.
    positions : sequence of int, optional
        Point indices to analyse.  ``None`` → all.
    eps : float

    Returns
    -------
    dict
        ``{layer_num: {position: ANCOVAResult, ...}, ...}``
    """
    L, n_tasks, n_points, B, D = all_hiddens.shape
    n_dims = demo_data.shape[-1]

    if layers is None:
        layers = list(range(L))
    if positions is None:
        positions = list(range(n_points))

    task_labels = (
        torch.arange(n_tasks)
        .unsqueeze(1)
        .expand(n_tasks, B)
        .reshape(-1)
    )

    results: Dict[int, Dict[int, ANCOVAResult]] = {}

    for l_idx, l_num in enumerate(layers):
        if l_idx >= L:
            continue
        layer_results: Dict[int, ANCOVAResult] = {}

        for pos in positions:
            if pos >= n_points:
                continue

            h = all_hiddens[l_idx, :, pos, :, :]  # (n_tasks, B, D)
            h_flat = h.reshape(n_tasks * B, D)

            x = demo_data[:, pos, :]  # (B, n_dims)
            x_flat = x.unsqueeze(0).expand(n_tasks, B, n_dims).reshape(n_tasks * B, n_dims)

            res = ancova_separability(h_flat, x_flat, task_labels, eps=eps)
            res.layer_num = l_num
            res.position = pos
            layer_results[pos] = res

        results[l_num] = layer_results

    return results


def plot_ancova_separability(
    results: Dict[int, Dict[int, ANCOVAResult]],
    figsize: tuple = (6, 4),
    log_x: bool = True,
    show: bool = True,
    show_ylabel: bool = True,
):
    """Plot ANCOVA separability gap and R² as separate figures.

    Returns
    -------
    (fig_gap, fig_r2) : tuple of Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    layers = sorted(results.keys())
    if not layers:
        return None, None

    _COLORS = [
        "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
        "#56B4E9", "#F0E442", "#000000",
    ]
    _LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

    def _style(i):
        return dict(
            color=_COLORS[i % len(_COLORS)],
            linestyle=_LINESTYLES[i % len(_LINESTYLES)],
            linewidth=2.2,
        )

    # ---- Figure 1: separability gap ----
    fig1, ax1 = plt.subplots(figsize=figsize)
    pos_list = []
    for i, l_num in enumerate(layers):
        pos_results = results[l_num]
        pos_list = sorted(pos_results.keys())
        if not pos_list:
            continue
        gaps = [pos_results[p].separability_gap for p in pos_list]
        ax1.plot(pos_list, gaps, label=f"Layer {l_num}", **_style(i))

    ax1.set_xlabel("Position", fontsize=14)
    if show_ylabel:
        ax1.set_ylabel("$R^2_{\\mathrm{full}} - R^2_{\\mathrm{additive}}$", fontsize=14)
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

    # ---- Figure 2: additive R² ----
    fig2, ax2 = plt.subplots(figsize=figsize)
    for i, l_num in enumerate(layers):
        pos_results = results[l_num]
        pos_list = sorted(pos_results.keys())
        if not pos_list:
            continue
        r2_add = [pos_results[p].r2_additive for p in pos_list]
        ax2.plot(pos_list, r2_add, label=f"Layer {l_num}", **_style(i))

    ax2.set_xlabel("Position", fontsize=14)
    if show_ylabel:
        ax2.set_ylabel("Additive $R^2$", fontsize=14)
    if log_x and len(pos_list) > 1 and min(pos_list) >= 0:
        ax2.set_xscale("symlog", linthresh=1)
    ax2.set_ylim(None, 1.005)
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


def print_ancova_summary(
    results: Dict[int, Dict[int, ANCOVAResult]],
    positions: Optional[Sequence[int]] = None,
):
    """Print a formatted ANCOVA summary table."""
    layers = sorted(results.keys())
    if not layers:
        return

    sample_layer = results[layers[0]]
    all_pos = sorted(sample_layer.keys())
    if positions is not None:
        all_pos = [p for p in all_pos if p in positions]

    header = (
        f"{'Layer':>6} {'Pos':>5} {'R²_add':>8} {'R²_full':>8} "
        f"{'gap':>8} {'N':>7} {'K':>3} {'d':>3}"
    )
    print("=" * len(header))
    print("  ANCOVA: slope homogeneity (additive separability)")
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
                f"{l_num:>6} {pos:>5} {r.r2_additive:>8.4f} {r.r2_full:>8.4f} "
                f"{r.separability_gap:>8.4f} {r.n_samples:>7} "
                f"{r.n_tasks:>3} {r.n_covariate_dims:>3}"
            )

    print("=" * len(header))
