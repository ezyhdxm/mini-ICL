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


def _layer_style(layer_num, n_positions=50):
    """Deterministic visual style keyed by *layer number* (not enumeration index).

    Ensures the same layer always receives the same colour/marker across
    different plots, even when different subsets of layers are shown.
    """
    _MARKERS = [
        "o", "s", "^", "D", "v", ">", "<", "p", "h", "*",
        "X", "P", "d", "8", "H",
    ]
    _LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
    import matplotlib.pyplot as plt
    cmap_colors = plt.cm.tab20.colors          # 20 distinct colours
    return dict(
        color=cmap_colors[layer_num % len(cmap_colors)],
        linestyle=_LINESTYLES[layer_num % len(_LINESTYLES)],
        marker=_MARKERS[layer_num % len(_MARKERS)],
        markersize=4,
        markevery=max(1, n_positions // 8),
        linewidth=2.0,
    )


def plot_task_vector_r2(
    results: Dict[int, Dict[int, TaskVectorR2Result]],
    figsize: tuple = (6, 4),
    log_x: bool = True,
    show: bool = True,
    show_ylabel: bool = True,
):
    """Plot residual variance ratio (1 − R²) across positions.

    Returns
    -------
    fig : Figure
    """
    import matplotlib.pyplot as plt

    layers = sorted(results.keys())
    if not layers:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    pos_list = []
    for l_num in layers:
        pos_results = results[l_num]
        pos_list = sorted(pos_results.keys())
        if not pos_list:
            continue
        vals = [1.0 - pos_results[p].r2 for p in pos_list]
        ax.plot(pos_list, vals, label=str(l_num),
                **_layer_style(l_num, len(pos_list)))

    ax.set_xlabel("Position", fontsize=14)
    if show_ylabel:
        ax.set_ylabel("Residual variance ratio", fontsize=14)
    if log_x and len(pos_list) > 1 and min(pos_list) >= 0:
        ax.set_xscale("symlog", linthresh=1)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=12)
    _ncol = 2 if len(layers) > 6 else 1
    ax.legend(title="Layer", fontsize=12, title_fontsize=12,
              framealpha=0.9, loc="best", ncol=_ncol,
              borderaxespad=0.3, handlelength=2.2)
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

    r2_additive_val: Optional[float] = None
    r2_full_val: Optional[float] = None
    separability_gap_val: Optional[float] = None


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


def mlp_ancova_separability(
    hiddens: torch.Tensor,
    covariates: torch.Tensor,
    task_labels: torch.Tensor,
    hidden_dim: int = 256,
    n_hidden_layers: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    n_epochs: int = 800,
    val_fraction: float = 0.2,
    patience: int = 80,
    batch_size: int = 512,
    verbose: bool = False,
    eps: float = 1e-10,
) -> ANCOVAResult:
    """MLP-based separability test for a single (layer, position).

    Replaces the linear covariate model in ``ancova_separability`` with
    a small MLP so that nonlinear token effects are properly captured.

    Additive model::

        h = onehot(k) @ W_task + MLP_shared(x_t) + b

    Full (interaction) model::

        h = MLP(onehot(k), x_t)

    Both models are trained with Adam + early stopping on a held-out
    validation set. The gap  R²_full − R²_additive  measures genuine
    task–token interaction that persists even after accounting for
    nonlinear token effects.

    Parameters
    ----------
    hiddens : (N, D) tensor
    covariates : (N, d) tensor
    task_labels : (N,) long tensor  — values in {0, ..., K-1}
    hidden_dim : int — MLP hidden width
    n_hidden_layers : int — number of hidden layers (≥1)
    lr, weight_decay : float — Adam parameters
    n_epochs : int — max training epochs
    val_fraction : float — held-out fraction for early stopping
    patience : int — early-stopping patience (epochs without improvement)
    batch_size : int — minibatch size (0 = full batch)
    verbose : bool
    eps : float

    Returns
    -------
    ANCOVAResult  (same dataclass as the linear version)
    """
    import torch.nn as nn
    import torch.optim as optim

    h = hiddens.float()
    x = covariates.float()
    labels = task_labels.long()

    N, D = h.shape
    d = x.shape[1]
    K = int(labels.max().item()) + 1
    device = h.device

    one_hot_k = torch.zeros(N, K, dtype=torch.float32, device=device)
    one_hot_k.scatter_(1, labels.unsqueeze(1), 1.0)

    # ---- train/val split (deterministic) ----
    n_val = max(1, int(N * val_fraction))
    n_train = N - n_val
    perm = torch.randperm(N, device=device)
    idx_train, idx_val = perm[:n_train], perm[n_train:]

    def _make_mlp(in_dim, out_dim):
        layers = []
        cur = in_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(cur, hidden_dim))
            layers.append(nn.GELU())
            cur = hidden_dim
        layers.append(nn.Linear(cur, out_dim))
        return nn.Sequential(*layers).to(device)

    def _count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _train_model(model, X_train, Y_train, X_val, Y_val, model_name="model"):
        n_params = _count_params(model)
        if verbose:
            print(f"  [{model_name}] params={n_params:,d}, "
                  f"training up to {n_epochs} epochs ...")

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        final_epoch = 0
        bs = batch_size if batch_size > 0 else n_train

        for epoch in range(n_epochs):
            model.train()
            shuf = torch.randperm(X_train.shape[0], device=device)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, X_train.shape[0], bs):
                idx = shuf[start:start + bs]
                pred = model(X_train[idx])
                loss = ((pred - Y_train[idx]) ** 2).sum() / idx.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = ((val_pred - Y_val) ** 2).mean().item()

            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose and (epoch % 200 == 0 or epoch == n_epochs - 1):
                print(f"    epoch {epoch:4d}: train_mse={epoch_loss / n_batches:.4f}  "
                      f"val_mse={val_loss:.4f}")

            final_epoch = epoch
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  [{model_name}] early stop at epoch {epoch} "
                          f"(best_val_mse={best_val_loss:.4f})")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        if verbose and epochs_no_improve < patience:
            print(f"  [{model_name}] finished {final_epoch + 1} epochs "
                  f"(best_val_mse={best_val_loss:.4f})")
        return model

    def _compute_r2(model, X, Y):
        model.eval()
        with torch.no_grad():
            pred = model(X)
        ss_res = ((Y - pred) ** 2).sum().item()
        ss_tot = ((Y - Y.mean(dim=0)) ** 2).sum().item()
        return 1.0 - ss_res / (ss_tot + eps), ss_res, ss_tot

    # ---- OLS warm-start ----
    # Solve h = onehot(k) @ W_task + x_t @ W_x + b via closed-form OLS.
    # Both models are initialized from this solution so the MLP only needs
    # to learn the nonlinear residual (guarantees R² >= linear R²).
    X_ols = torch.cat([one_hot_k, x, torch.ones(N, 1, device=device)], dim=1)
    W_ols = torch.linalg.lstsq(X_ols, h).solution  # (K+d+1, D)
    W_task_ols = W_ols[:K]      # (K, D)
    W_x_ols = W_ols[K:K + d]   # (d, D)
    b_ols = W_ols[K + d]       # (D,)

    def _zero_init_last_linear(seq):
        for module in reversed(list(seq.modules())):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
                break

    # ---- Additive model: h = W_task @ onehot(k) + W_x @ x_t + MLP(x_t) + b ----
    # Linear skip from OLS + MLP residual (zero-init output).
    class AdditiveModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.task_linear = nn.Linear(K, D, bias=False)
            self.token_linear = nn.Linear(d, D, bias=False)
            self.token_mlp = _make_mlp(d, D)
            self.bias = nn.Parameter(torch.zeros(D))

            with torch.no_grad():
                self.task_linear.weight.copy_(W_task_ols.T)
                self.token_linear.weight.copy_(W_x_ols.T)
                self.bias.copy_(b_ols)
            _zero_init_last_linear(self.token_mlp)

        def forward(self, inputs):
            oh, xt = inputs[:, :K], inputs[:, K:]
            return (self.task_linear(oh) + self.token_linear(xt)
                    + self.token_mlp(xt) + self.bias)

    # ---- Full model: linear_skip([onehot(k), x_t]) + MLP([onehot(k), x_t]) ----
    # Linear skip from OLS + MLP residual (zero-init output).
    class FullModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_skip = nn.Linear(K + d, D)
            self.mlp = _make_mlp(K + d, D)

            with torch.no_grad():
                W_skip = torch.cat([W_task_ols, W_x_ols], dim=0)  # (K+d, D)
                self.linear_skip.weight.copy_(W_skip.T)
                self.linear_skip.bias.copy_(b_ols)
            _zero_init_last_linear(self.mlp)

        def forward(self, inputs):
            return self.linear_skip(inputs) + self.mlp(inputs)

    X_combined = torch.cat([one_hot_k, x], dim=1)  # (N, K+d)
    X_tr, X_va = X_combined[idx_train], X_combined[idx_val]
    Y_tr, Y_va = h[idx_train], h[idx_val]

    # Report OLS baseline R² for reference
    with torch.no_grad():
        h_ols = X_ols @ W_ols
        ss_res_ols = ((h - h_ols) ** 2).sum().item()
        ss_tot_all = ((h - h.mean(dim=0)) ** 2).sum().item()
        r2_ols = 1.0 - ss_res_ols / (ss_tot_all + eps)

    if verbose:
        print(f"[mlp_ancova] N={N}, K={K}, d_cov={d}, D_hidden={D}, "
              f"train={n_train}, val={n_val}, "
              f"arch={n_hidden_layers}x{hidden_dim}")
        print(f"[mlp_ancova] OLS baseline R² = {r2_ols:.6f} (warm-start)")

    add_model = AdditiveModel()
    add_model = _train_model(
        add_model, X_tr, Y_tr, X_va, Y_va,
        model_name="additive: W_task @ onehot(k) + MLP(x_t) + b",
    )
    r2_add, ss_res_add, ss_tot = _compute_r2(add_model, X_combined, h)
    r2_add_val, _, _ = _compute_r2(add_model, X_va, Y_va)

    if verbose:
        print(f"  [additive] R²_full={r2_add:.6f}  R²_val={r2_add_val:.6f}")

    full_model = FullModel()
    full_model = _train_model(
        full_model, X_tr, Y_tr, X_va, Y_va,
        model_name="full: MLP(onehot(k), x_t)",
    )
    r2_full, ss_res_full, _ = _compute_r2(full_model, X_combined, h)
    r2_full_val, _, _ = _compute_r2(full_model, X_va, Y_va)

    if verbose:
        print(f"  [full]     R²_full={r2_full:.6f}  R²_val={r2_full_val:.6f}")
        print(f"  gap(full_data)={r2_full - r2_add:.6f}  "
              f"gap(val)={r2_full_val - r2_add_val:.6f}")

    gap = r2_full - r2_add
    gap_val = r2_full_val - r2_add_val

    del add_model, full_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        r2_additive_val=r2_add_val,
        r2_full_val=r2_full_val,
        separability_gap_val=gap_val,
    )


def mlp_ancova_separability_from_hiddens(
    all_hiddens: torch.Tensor,
    demo_data: torch.Tensor,
    layers: "Optional[Sequence[int]]" = None,
    positions: "Optional[Sequence[int]]" = None,
    **mlp_kwargs,
) -> Dict[int, Dict[int, ANCOVAResult]]:
    """Run MLP-based ANCOVA across layers and positions for linear regression.

    Same interface as ``ancova_separability_from_hiddens`` but uses
    ``mlp_ancova_separability`` internally.

    Parameters
    ----------
    all_hiddens : (L, n_tasks, n_points, batch_size, D)
    demo_data : (batch_size, n_points, n_dims)
    layers, positions : optional index lists
    **mlp_kwargs : forwarded to ``mlp_ancova_separability``

    Returns
    -------
    dict  {layer_num: {position: ANCOVAResult}}
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

    is_verbose = mlp_kwargs.get("verbose", False)
    total_fits = len(layers) * len(positions)
    fit_count = 0

    results: Dict[int, Dict[int, ANCOVAResult]] = {}

    for l_idx, l_num in enumerate(layers):
        if l_idx >= L:
            continue
        layer_results: Dict[int, ANCOVAResult] = {}

        for pos in positions:
            if pos >= n_points:
                continue

            fit_count += 1
            if is_verbose:
                print(f"\n{'='*60}")
                print(f"[mlp_ancova] Layer {l_num}, Position {pos}  "
                      f"({fit_count}/{total_fits})")
                print(f"{'='*60}")

            h = all_hiddens[l_idx, :, pos, :, :]  # (n_tasks, B, D)
            h_flat = h.reshape(n_tasks * B, D)

            x = demo_data[:, pos, :]  # (B, n_dims)
            x_flat = x.unsqueeze(0).expand(n_tasks, B, n_dims).reshape(
                n_tasks * B, n_dims,
            )

            res = mlp_ancova_separability(
                h_flat, x_flat, task_labels, **mlp_kwargs,
            )
            res.layer_num = l_num
            res.position = pos
            layer_results[pos] = res

        results[l_num] = layer_results

    return results


def mlp_ancova_separability_joint(
    all_hiddens_layer: torch.Tensor,
    demo_data: torch.Tensor,
    positions: "Optional[Sequence[int]]" = None,
    layer_num: int = 0,
    fit_position: bool = False,
    hidden_dim: int = 256,
    n_hidden_layers: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    n_epochs: int = 800,
    val_fraction: float = 0.2,
    patience: int = 80,
    batch_size: int = 512,
    verbose: bool = False,
    eps: float = 1e-10,
) -> Dict[int, ANCOVAResult]:
    """Joint MLP-based separability test across positions for one layer.

    Instead of fitting a separate model at every position, this function
    pools data across all positions and fits *one* additive model and
    *one* full model.  ``W_task`` is shared.

    When ``fit_position=True``, normalised position ``t/T`` is appended
    to the covariate vector so the MLP can learn position-dependent token
    effects.  When ``False`` (default), only ``x_t`` is used.

    Both models are warm-started from the pooled OLS solution.
    After joint fitting, R² is evaluated **per position** separately.

    Parameters
    ----------
    all_hiddens_layer : (n_tasks, n_points, B, D)  — one layer
    demo_data : (B, n_points, n_dims)
    positions : list of position indices to use
    layer_num : int — for labelling results
    fit_position : bool — include normalised position as an extra input
    hidden_dim, n_hidden_layers : MLP architecture
    lr, weight_decay : Adam parameters
    n_epochs, patience : training budget
    val_fraction : held-out fraction
    batch_size : minibatch size
    verbose : bool
    eps : float

    Returns
    -------
    dict  {position: ANCOVAResult}
    """
    import torch.nn as nn
    import torch.optim as optim

    n_tasks, n_points, B, D = all_hiddens_layer.shape
    d = demo_data.shape[-1]
    K = n_tasks
    device = all_hiddens_layer.device

    if positions is None:
        positions = list(range(n_points))
    n_pos = len(positions)
    pos_max = max(max(positions), 1)

    # ---- Flatten data across (position, task, batch) ----
    h_parts, x_parts, k_parts = [], [], []
    p_parts = [] if fit_position else None
    pos_slices: Dict[int, tuple] = {}
    offset = 0
    for pos in positions:
        for ki in range(K):
            h_parts.append(all_hiddens_layer[ki, pos, :, :])     # (B, D)
            x_parts.append(demo_data[:, pos, :])                  # (B, d)
            k_parts.append(torch.full((B,), ki, dtype=torch.long, device=device))
            if fit_position:
                p_parts.append(torch.full((B, 1), pos / pos_max,
                                          dtype=torch.float32, device=device))
        pos_slices[pos] = (offset, offset + K * B)
        offset += K * B

    h_all = torch.cat(h_parts, dim=0).float()       # (N, D)
    x_all = torch.cat(x_parts, dim=0).float()       # (N, d)
    k_all = torch.cat(k_parts, dim=0)               # (N,)
    N = h_all.shape[0]

    if fit_position:
        p_all = torch.cat(p_parts, dim=0)            # (N, 1)
        cov_all = torch.cat([x_all, p_all], dim=1)   # (N, d+1)
        d_cov = d + 1
    else:
        cov_all = x_all                               # (N, d)
        d_cov = d

    one_hot_k = torch.zeros(N, K, dtype=torch.float32, device=device)
    one_hot_k.scatter_(1, k_all.unsqueeze(1).to(device), 1.0)

    # ---- train/val split ----
    n_val = max(1, int(N * val_fraction))
    n_train = N - n_val
    perm = torch.randperm(N, device=device)
    idx_train, idx_val = perm[:n_train], perm[n_train:]

    # ---- Helpers (same as per-position version) ----
    def _make_mlp(in_dim, out_dim):
        layers_list = []
        cur = in_dim
        for _ in range(n_hidden_layers):
            layers_list.append(nn.Linear(cur, hidden_dim))
            layers_list.append(nn.GELU())
            cur = hidden_dim
        layers_list.append(nn.Linear(cur, out_dim))
        return nn.Sequential(*layers_list).to(device)

    def _zero_init_last_linear(seq):
        for module in reversed(list(seq.modules())):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
                break

    def _count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _train_model(model, X_train, Y_train, X_val, Y_val, model_name="model"):
        n_params = _count_params(model)
        if verbose:
            print(f"  [{model_name}] params={n_params:,d}, "
                  f"training up to {n_epochs} epochs ...")
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.01)
        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        final_epoch = 0
        bs = batch_size if batch_size > 0 else n_train
        for epoch in range(n_epochs):
            model.train()
            shuf = torch.randperm(X_train.shape[0], device=device)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, X_train.shape[0], bs):
                idx = shuf[start:start + bs]
                pred = model(X_train[idx])
                loss = ((pred - Y_train[idx]) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = ((val_pred - Y_val) ** 2).mean().item()
            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if verbose and (epoch % 200 == 0 or epoch == n_epochs - 1):
                print(f"    epoch {epoch:4d}: train_mse={epoch_loss / n_batches:.4f}  "
                      f"val_mse={val_loss:.4f}")
            final_epoch = epoch
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  [{model_name}] early stop at epoch {epoch} "
                          f"(best_val_mse={best_val_loss:.4f})")
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        if verbose and epochs_no_improve < patience:
            print(f"  [{model_name}] finished {final_epoch + 1} epochs "
                  f"(best_val_mse={best_val_loss:.4f})")
        return model

    def _compute_r2(model, X, Y):
        model.eval()
        with torch.no_grad():
            pred = model(X)
        ss_res = ((Y - pred) ** 2).sum().item()
        ss_tot = ((Y - Y.mean(dim=0)) ** 2).sum().item()
        return 1.0 - ss_res / (ss_tot + eps), ss_res, ss_tot

    # ---- OLS warm-start (pooled across positions) ----
    X_ols = torch.cat([one_hot_k, cov_all,
                       torch.ones(N, 1, device=device)], dim=1)
    W_ols = torch.linalg.lstsq(X_ols, h_all).solution
    W_task_ols = W_ols[:K]                  # (K, D)
    W_cov_ols = W_ols[K:K + d_cov]         # (d_cov, D)
    b_ols = W_ols[K + d_cov]               # (D,)

    with torch.no_grad():
        h_ols_pred = X_ols @ W_ols
        r2_ols = 1.0 - (((h_all - h_ols_pred) ** 2).sum().item()
                        / (((h_all - h_all.mean(0)) ** 2).sum().item() + eps))

    pos_tag = "+pos" if fit_position else ""
    if verbose:
        print(f"[mlp_ancova_joint] Layer {layer_num}: "
              f"N={N} ({n_pos} pos x {K} tasks x {B} batch), "
              f"K={K}, d_cov={d}{pos_tag}, D_hidden={D}, "
              f"train={n_train}, val={n_val}, "
              f"arch={n_hidden_layers}x{hidden_dim}")
        print(f"[mlp_ancova_joint] OLS baseline R² = {r2_ols:.6f} (warm-start)")

    # ---- Additive: h = W_task @ oh + Linear(cov) + MLP(cov) + b ----
    class AdditiveModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.task_linear = nn.Linear(K, D, bias=False)
            self.token_linear = nn.Linear(d_cov, D, bias=False)
            self.token_mlp = _make_mlp(d_cov, D)
            self.bias = nn.Parameter(torch.zeros(D, device=device))
            with torch.no_grad():
                self.task_linear.weight.copy_(W_task_ols.T)
                self.token_linear.weight.copy_(W_cov_ols.T)
                self.bias.copy_(b_ols)
            _zero_init_last_linear(self.token_mlp)

        def forward(self, inputs):
            oh = inputs[:, :K]
            cov = inputs[:, K:K + d_cov]
            return (self.task_linear(oh) + self.token_linear(cov)
                    + self.token_mlp(cov) + self.bias)

    # ---- Full: h = Linear(oh, cov) + MLP(oh, cov) ----
    class FullModel(nn.Module):
        def __init__(self):
            super().__init__()
            in_dim = K + d_cov
            self.linear_skip = nn.Linear(in_dim, D)
            self.mlp = _make_mlp(in_dim, D)
            with torch.no_grad():
                W_skip = torch.cat([W_task_ols, W_cov_ols], dim=0)
                self.linear_skip.weight.copy_(W_skip.T)
                self.linear_skip.bias.copy_(b_ols)
            _zero_init_last_linear(self.mlp)

        def forward(self, inputs):
            return self.linear_skip(inputs) + self.mlp(inputs)

    X_combined = torch.cat([one_hot_k, cov_all], dim=1)
    X_tr = X_combined[idx_train]
    X_va = X_combined[idx_val]
    Y_tr = h_all[idx_train]
    Y_va = h_all[idx_val]

    cov_desc = "x,pos" if fit_position else "x"
    add_model = AdditiveModel()
    add_model = _train_model(
        add_model, X_tr, Y_tr, X_va, Y_va,
        model_name=f"additive(joint): W_task@oh + MLP({cov_desc}) + b",
    )

    full_model = FullModel()
    full_model = _train_model(
        full_model, X_tr, Y_tr, X_va, Y_va,
        model_name=f"full(joint): MLP(oh, {cov_desc})",
    )

    # ---- Per-position R² evaluation ----
    results: Dict[int, ANCOVAResult] = {}
    for pos in positions:
        s, e = pos_slices[pos]
        X_pos = X_combined[s:e]
        Y_pos = h_all[s:e]

        r2_add, ss_res_add, ss_tot_pos = _compute_r2(add_model, X_pos, Y_pos)
        r2_full, ss_res_full, _ = _compute_r2(full_model, X_pos, Y_pos)

        # val-only R² for this position
        val_mask = (idx_val >= s) & (idx_val < e)
        if val_mask.any():
            val_idx_local = idx_val[val_mask] - s
            X_pos_val = X_pos[val_idx_local]
            Y_pos_val = Y_pos[val_idx_local]
            r2_add_val, _, _ = _compute_r2(add_model, X_pos_val, Y_pos_val)
            r2_full_val, _, _ = _compute_r2(full_model, X_pos_val, Y_pos_val)
        else:
            r2_add_val = r2_add
            r2_full_val = r2_full

        gap = r2_full - r2_add
        gap_val = r2_full_val - r2_add_val

        results[pos] = ANCOVAResult(
            r2_additive=r2_add,
            r2_full=r2_full,
            separability_gap=gap,
            ss_total=ss_tot_pos,
            ss_res_additive=ss_res_add,
            ss_res_full=ss_res_full,
            n_tasks=K,
            n_covariate_dims=d,
            n_samples=e - s,
            layer_num=layer_num,
            position=pos,
            r2_additive_val=r2_add_val,
            r2_full_val=r2_full_val,
            separability_gap_val=gap_val,
        )

    if verbose:
        for pos in positions:
            r = results[pos]
            print(f"  pos={pos:3d}: R²_add={r.r2_additive:.4f} "
                  f"R²_full={r.r2_full:.4f}  gap={r.separability_gap:.4f}  "
                  f"R²_add_v={r.r2_additive_val:.4f} "
                  f"R²_full_v={r.r2_full_val:.4f}  gap_v={r.separability_gap_val:.4f}")

    del add_model, full_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def mlp_ancova_separability_joint_from_hiddens(
    all_hiddens: torch.Tensor,
    demo_data: torch.Tensor,
    layers: "Optional[Sequence[int]]" = None,
    positions: "Optional[Sequence[int]]" = None,
    **mlp_kwargs,
) -> Dict[int, Dict[int, ANCOVAResult]]:
    """Run joint MLP ANCOVA across layers (one joint fit per layer).

    Parameters
    ----------
    all_hiddens : (L, n_tasks, n_points, batch_size, D)
    demo_data : (batch_size, n_points, n_dims)
    layers, positions : optional index lists
    **mlp_kwargs : forwarded to ``mlp_ancova_separability_joint``

    Returns
    -------
    dict  {layer_num: {position: ANCOVAResult}}
    """
    L = all_hiddens.shape[0]
    if layers is None:
        layers = list(range(L))

    results: Dict[int, Dict[int, ANCOVAResult]] = {}
    for l_idx, l_num in enumerate(layers):
        if l_idx >= L:
            continue
        results[l_num] = mlp_ancova_separability_joint(
            all_hiddens[l_idx],
            demo_data,
            positions=positions,
            layer_num=l_num,
            **mlp_kwargs,
        )
    return results


# =========================================================================
# Averaging-based task vector estimation and additive-model R²
# =========================================================================


def estimate_task_vectors_by_averaging(
    all_hiddens_layer: torch.Tensor,
    estimation_positions: Sequence[int],
) -> tuple:
    """Estimate task vectors by averaging hidden states per task.

    Pools hidden states across ``estimation_positions`` and batch
    samples for each task, then centers.

    Parameters
    ----------
    all_hiddens_layer : (n_tasks, n_points, B, D)
        Hidden states for one layer.
    estimation_positions : list of int
        Late position indices to average over.

    Returns
    -------
    (task_vecs, grand_mean)
        task_vecs : (K, D) centred task vectors (sum to zero)
        grand_mean : (D,) grand mean across tasks
    """
    K, n_points, B, D = all_hiddens_layer.shape

    task_means = torch.zeros(K, D, dtype=torch.float32)
    for k in range(K):
        parts = [all_hiddens_layer[k, t, :, :] for t in estimation_positions]
        task_means[k] = torch.cat(parts, dim=0).float().mean(dim=0)

    grand_mean = task_means.mean(dim=0)
    task_vecs = task_means - grand_mean.unsqueeze(0)

    return task_vecs, grand_mean


def per_position_task_vectors(
    all_hiddens_layer: torch.Tensor,
    per_position_mean: bool = True,
) -> tuple:
    """Compute task vectors at every position independently.

    Parameters
    ----------
    all_hiddens_layer : (K, T, B, D)
        Hidden states for one layer, grouped by task.
    per_position_mean : bool
        If True, centre with a position-specific grand mean mu_t.
        If False, use a single grand mean pooled across all positions.

    Returns
    -------
    (task_vecs_by_pos, grand_means)
        task_vecs_by_pos : (K, T, D) centred task vectors at each position
        grand_means : (T, D) grand mean at each position (or broadcast)
    """
    K, T, B, D = all_hiddens_layer.shape

    task_means = all_hiddens_layer.float().mean(dim=2)  # (K, T, D)

    if per_position_mean:
        grand_means = task_means.mean(dim=0)            # (T, D)
    else:
        grand_means = task_means.mean(dim=(0, 1)).unsqueeze(0).expand(T, D)  # (T, D)

    task_vecs_by_pos = task_means - grand_means.unsqueeze(0)  # (K, T, D)

    return task_vecs_by_pos, grand_means


# ---- Balanced (orthogonal-design) variants --------------------------------

def per_position_task_vectors_balanced(
    cell_hiddens: torch.Tensor,
    per_position_mean: bool = True,
) -> tuple:
    """Compute task vectors at every position from a balanced design.

    Uses cell means from token-conditioned (interventional) data and
    uniformly averages over tokens, eliminating token leakage.  This is
    the per-position analogue of the task effect in a two-way ANOVA.

    Parameters
    ----------
    cell_hiddens : (T, V, K, B, D)
        Token-conditioned hidden states for one layer.
        ``T`` positions, ``V`` unique tokens, ``K`` tasks, ``B`` samples.
    per_position_mean : bool
        If True, centre with a position-specific grand mean.

    Returns
    -------
    (task_vecs_by_pos, grand_means)
        task_vecs_by_pos : (K, T, D)
        grand_means      : (T, D)
    """
    T, V, K, B, D = cell_hiddens.shape
    cell_means = cell_hiddens.float().mean(dim=3)       # (T, V, K, D)
    task_means = cell_means.mean(dim=1)                 # (T, K, D)

    if per_position_mean:
        grand_means = task_means.mean(dim=1)            # (T, D)
    else:
        grand_means = task_means.mean(dim=(0, 1)).unsqueeze(0).expand(T, D)

    task_vecs_by_pos = (
        task_means - grand_means.unsqueeze(1)
    ).permute(1, 0, 2)                                  # (K, T, D)

    return task_vecs_by_pos, grand_means


def estimate_task_vectors_by_averaging_balanced(
    cell_hiddens: torch.Tensor,
    estimation_positions: Sequence[int],
) -> tuple:
    """Estimate reference task vectors from a balanced design.

    Pools cell means across ``estimation_positions`` and uniformly
    averages over tokens, then centres.

    Parameters
    ----------
    cell_hiddens : (T, V, K, B, D)
        Token-conditioned hidden states for one layer.
    estimation_positions : list of int
        Position indices (into the T axis) to pool over.

    Returns
    -------
    (task_vecs, grand_mean)
        task_vecs  : (K, D) centred, sum-to-zero
        grand_mean : (D,)
    """
    h = cell_hiddens.float()
    est = h[estimation_positions]                       # (T_est, V, K, B, D)
    est_cell_means = est.mean(dim=3)                    # (T_est, V, K, D)
    task_means = est_cell_means.mean(dim=(0, 1))        # (K, D)
    grand_mean = task_means.mean(dim=0)                 # (D,)
    task_vecs = task_means - grand_mean.unsqueeze(0)    # (K, D)
    return task_vecs, grand_mean


def per_position_token_vectors_balanced(
    cell_hiddens: torch.Tensor,
    per_position_mean: bool = True,
) -> torch.Tensor:
    """Compute token (main-effect) vectors at every position from a balanced design.

    Parameters
    ----------
    cell_hiddens : (T, V, K, B, D)
        Token-conditioned hidden states for one layer.
    per_position_mean : bool
        If True, centre with a position-specific grand mean.

    Returns
    -------
    token_vecs_by_pos : (V, T, D)
    """
    T, V, K, B, D = cell_hiddens.shape
    cell_means = cell_hiddens.float().mean(dim=3)       # (T, V, K, D)
    token_means = cell_means.mean(dim=2)                # (T, V, D)

    if per_position_mean:
        grand_means = cell_means.mean(dim=(1, 2))       # (T, D)
    else:
        grand_means = cell_means.mean(dim=(0, 1, 2)).unsqueeze(0).expand(T, D)

    token_vecs_by_pos = (
        token_means - grand_means.unsqueeze(1)
    ).permute(1, 0, 2)                                  # (V, T, D)

    return token_vecs_by_pos


def estimate_task_and_token_vectors_jointly(
    all_hiddens_layer: torch.Tensor,
    token_ids: torch.Tensor,
    estimation_positions: Sequence[int],
) -> tuple:
    """Jointly estimate task and token vectors via two-way additive ANOVA.

    Solves the OLS problem

        h̃_i = Σ_{k} z_{ik} θ_k + Σ_{s} w_{is} ν_s + ε_i

    on position-demeaned hidden states pooled across estimation
    positions, using drop-one-column coding for identifiability and
    reconstructing sum-to-zero vectors afterwards.

    This avoids the bias that arises when task and token effects are
    estimated by independent averaging under an unbalanced design
    (e.g. latent Markov chains where the token distribution depends
    on the task).

    Parameters
    ----------
    all_hiddens_layer : (K, n_points, B, D)
    token_ids : (K, B, seq_len)
    estimation_positions : list of int

    Returns
    -------
    task_vecs : (K, D)  centred, sum-to-zero
    token_vecs : (V, D) centred, sum-to-zero
    grand_mean : (D,)   grand mean of *raw* hidden states
    """
    K, n_points, B, D = all_hiddens_layer.shape

    h_parts, task_parts, tok_parts = [], [], []
    task_labels = torch.arange(K).unsqueeze(1).expand(K, B).reshape(K * B)

    for t in estimation_positions:
        h_t = all_hiddens_layer[:, t, :, :].reshape(K * B, D).float()
        mu_t = h_t.mean(dim=0)
        h_parts.append(h_t - mu_t)
        task_parts.append(task_labels)
        tok_parts.append(token_ids[:, :, t].reshape(K * B))

    h_pooled = torch.cat(h_parts, dim=0)           # (M, D)
    z_task = torch.cat(task_parts, dim=0).long()    # (M,)
    z_tok = torch.cat(tok_parts, dim=0).long()      # (M,)

    M = h_pooled.shape[0]
    V = int(z_tok.max().item()) + 1

    # Design matrix: task indicators (K-1) | token indicators (V-1)
    # No intercept needed — data is already position-demeaned (zero mean).
    n_cols = (K - 1) + (V - 1)
    X = torch.zeros(M, n_cols, dtype=h_pooled.dtype, device=h_pooled.device)
    for k in range(K - 1):
        X[:, k] = (z_task == k).float()
    for s in range(V - 1):
        X[:, (K - 1) + s] = (z_tok == s).float()

    W = torch.linalg.lstsq(X, h_pooled).solution   # (n_cols, D)

    # Reconstruct sum-to-zero task vectors
    task_coefs = torch.cat([W[:K - 1], torch.zeros(1, D, dtype=W.dtype, device=W.device)], dim=0)
    task_vecs = task_coefs - task_coefs.mean(dim=0, keepdim=True)

    # Reconstruct sum-to-zero token vectors
    tok_coefs = torch.cat([W[K - 1 : K - 1 + V - 1], torch.zeros(1, D, dtype=W.dtype, device=W.device)], dim=0)
    token_vecs = tok_coefs - tok_coefs.mean(dim=0, keepdim=True)

    # Grand mean of raw (non-demeaned) hidden states
    raw_parts = [
        all_hiddens_layer[:, t, :, :].reshape(K * B, D).float()
        for t in estimation_positions
    ]
    grand_mean = torch.cat(raw_parts, dim=0).mean(dim=0)

    return task_vecs, token_vecs, grand_mean


def estimate_token_vectors_by_averaging(
    all_hiddens_layer: torch.Tensor,
    token_ids: torch.Tensor,
    estimation_positions: Sequence[int],
    task_vecs: Optional[torch.Tensor] = None,
) -> tuple:
    """Estimate token (vocab) vectors by averaging hidden states per token.

    At each estimation position the per-position mean is subtracted to
    remove position-encoding effects.  When ``task_vecs`` is provided
    the task effect is also subtracted before averaging by token
    identity, preventing task--token confounding from contaminating
    the token vectors (critical when the token distribution depends
    on the task, e.g. latent Markov chains).

    Parameters
    ----------
    all_hiddens_layer : (K, n_points, B, D)
        Hidden states for one layer.
    token_ids : (K, B, seq_len)
        Integer token ids for each (task, sample, position).
    estimation_positions : list of int
        Position indices to average over (same as for task vectors).
    task_vecs : (K, D), optional
        Centred task vectors (sum-to-zero).  When provided, each
        sample's task effect is subtracted before averaging by token,
        so the resulting token vectors are not confounded with the
        task effect.

    Returns
    -------
    token_vecs : (V, D)
        Centred token vectors.
    grand_mean : (D,)
        Grand mean across all pooled (raw) samples.
    """
    K, n_points, B, D = all_hiddens_layer.shape

    # Task labels: sample i in [k*B, (k+1)*B) belongs to task k
    task_labels = (
        torch.arange(K).unsqueeze(1).expand(K, B).reshape(K * B)
    )

    h_parts = []
    s_parts = []
    for t in estimation_positions:
        h_t = all_hiddens_layer[:, t, :, :].reshape(K * B, D).float()
        mu_t = h_t.mean(dim=0)
        h_demeaned = h_t - mu_t

        if task_vecs is not None:
            h_demeaned = h_demeaned - task_vecs.to(h_demeaned.device)[task_labels].float()

        h_parts.append(h_demeaned)
        s_parts.append(token_ids[:, :, t].reshape(K * B))

    h_pooled = torch.cat(h_parts, dim=0)   # (M, D)
    s_pooled = torch.cat(s_parts, dim=0).long()

    raw_parts = [
        all_hiddens_layer[:, t, :, :].reshape(K * B, D).float()
        for t in estimation_positions
    ]
    grand_mean = torch.cat(raw_parts, dim=0).mean(dim=0)

    V = int(s_pooled.max().item()) + 1
    token_means = torch.zeros(V, D, dtype=torch.float32)
    for s in range(V):
        mask = (s_pooled == s)
        if mask.any():
            token_means[s] = h_pooled[mask].mean(dim=0)
        else:
            token_means[s] = torch.zeros(D, dtype=torch.float32)

    return token_means, grand_mean


@dataclass
class AveragingR2Result:
    """R² from the averaging-based task-vector test at one (layer, position)."""

    r2_task: float
    r2_additive: float

    ss_total: float
    ss_task: float
    ss_token: float

    n_tasks: int
    n_samples: int

    layer_num: Optional[int] = None
    position: Optional[int] = None


def _simplex_project_coeffs(
    task_vecs: torch.Tensor,
    h_centered: torch.Tensor,
) -> torch.Tensor:
    """Per-sample simplex-constrained coefficients and reconstruction.

    Solves  min_{β ≥ 0, Σβ=1} ‖h − Θᵀβ‖²  approximately by first
    computing the unconstrained affine solution (Σβ=1) then projecting
    onto the probability simplex.

    Parameters
    ----------
    task_vecs : (K, D) centred task vectors (Σ_k θ_k ≈ 0)
    h_centered : (N, D) mean-subtracted hidden states

    Returns
    -------
    h_hat : (N, D) simplex-constrained reconstruction  Σ β*_k θ_k
    """
    import numpy as np
    from icl.utils.linear_algebra_utils import _project_onto_simplex_np

    K, D = task_vecs.shape
    device = h_centered.device

    V = task_vecs.float().to(device)
    anchor = V[-1]                          # (D,)
    diff = V[:-1] - anchor                  # (K-1, D)
    pinv = torch.linalg.pinv(diff)          # (D, K-1)

    r = h_centered.float() - anchor.unsqueeze(0)   # (N, D)
    gamma = r @ pinv                                # (N, K-1)
    beta_last = 1.0 - gamma.sum(dim=1, keepdim=True)
    beta = torch.cat([gamma, beta_last], dim=1)     # (N, K)

    beta_np = beta.cpu().numpy()
    beta_np = _project_onto_simplex_np(beta_np)     # (N, K)
    beta_simplex = torch.from_numpy(beta_np).float().to(device)

    h_hat = beta_simplex @ V                        # (N, D)
    return h_hat


def task_subspace_r2_at_position(
    task_vecs: torch.Tensor,
    hiddens: torch.Tensor,
    covariates: Optional[torch.Tensor] = None,
    fit_token: str = "linear",
    grand_mean: Optional[torch.Tensor] = None,
    token_vecs: Optional[torch.Tensor] = None,
    token_ids: Optional[torch.Tensor] = None,
    simplex: bool = True,
    eps: float = 1e-10,
) -> AveragingR2Result:
    """Fraction of hidden-state variance explained by task subspace (+ token).

    Subtracts a mean vector and projects onto span(task_vecs), optionally
    fitting a linear token model on the residual.

    Parameters
    ----------
    task_vecs : (K, D) centred task vectors
    hiddens : (n_tasks, B, D) hidden states at one position
    covariates : (B, d) or (n_tasks, B, d)
        Covariates x_t.  Used when ``fit_token="linear"``.
        Shape ``(B, d)`` is broadcast identically to every task
        (e.g. shared input in linear regression).  Shape
        ``(n_tasks, B, d)`` provides per-task covariates (e.g. one-hot
        tokens that differ across tasks).
    fit_token : "none" | "linear" | "anova"
        ``"linear"``: OLS on ``covariates`` (continuous or one-hot).
        ``"anova"``: subtract the known token vector (looked up via
        ``token_ids``), then project the remainder onto the task
        subspace via drop-one.
        ``"none"``: no token model.
    grand_mean : (D,), optional
        If provided, subtract this fixed mean instead of the per-position
        sample mean.  Useful for testing what happens without per-position
        demeaning.
    token_vecs : (V, D), optional
        Pre-estimated centred token vectors.  Required when
        ``fit_token="anova"``.
    token_ids : (n_tasks, B), optional
        Integer token identity at this position for each (task, sample).
        Required when ``fit_token="anova"``.
    simplex : bool
        If True (default), constrain task coefficients β to the
        probability simplex (β ≥ 0, Σβ = 1).  If False, use
        unconstrained orthogonal projection.
    eps : float

    Returns
    -------
    AveragingR2Result
    """
    K, D = task_vecs.shape
    n_tasks, B, _ = hiddens.shape
    device = hiddens.device

    h_flat = hiddens.reshape(n_tasks * B, D).float()
    N = h_flat.shape[0]

    if grand_mean is not None:
        mu_t = grand_mean.to(device).float()
    else:
        mu_t = h_flat.mean(dim=0)
    h_centered = h_flat - mu_t.unsqueeze(0)

    ss_total = (h_centered ** 2).sum().item()

    V = task_vecs.to(device).float()  # (K, D)

    if simplex:
        # ---- Task-only R² (simplex-constrained) ----
        h_task_hat = _simplex_project_coeffs(V, h_centered)
        ss_task_residual = ((h_centered - h_task_hat) ** 2).sum().item()
        ss_task = ss_total - ss_task_residual
        r2_task = ss_task / (ss_total + eps)

        # ---- Task + token R² ----
        if fit_token == "none" or (covariates is None and token_vecs is None):
            r2_additive = r2_task
            ss_token = 0.0
        elif fit_token == "linear":
            residual = h_centered - h_task_hat  # (N, D)

            if covariates.ndim == 3:
                d = covariates.shape[2]
                x_flat = covariates.to(device).float().reshape(N, d)
            else:
                d = covariates.shape[1]
                x_flat = covariates.to(device).float()
                x_flat = x_flat.unsqueeze(0).expand(n_tasks, B, d).reshape(N, d)

            XtX = x_flat.T @ x_flat
            XtX_reg = XtX + eps * torch.eye(d, device=XtX.device)
            W_x = torch.linalg.solve(XtX_reg, x_flat.T @ residual)
            h_token = x_flat @ W_x

            ss_token = (h_token ** 2).sum().item()
            ss_residual = ((residual - h_token) ** 2).sum().item()
            r2_additive = 1.0 - ss_residual / (ss_total + eps)
        elif fit_token == "anova":
            if token_vecs is None or token_ids is None:
                raise ValueError(
                    "fit_token='anova' requires token_vecs and token_ids"
                )
            Vt = token_vecs.to(device).float()
            ids_flat = token_ids.reshape(N).long()
            h_tok_effect = Vt[ids_flat]                    # (N, D)
            h_no_tok = h_centered - h_tok_effect
            h_task_hat_nt = _simplex_project_coeffs(V, h_no_tok)
            h_additive_hat = h_task_hat_nt + h_tok_effect
            ss_residual = ((h_centered - h_additive_hat) ** 2).sum().item()
            ss_additive = ss_total - ss_residual
            r2_additive = ss_additive / (ss_total + eps)
            ss_token = ss_additive - ss_task
        else:
            raise ValueError(
                f"fit_token must be 'none', 'linear', or 'anova', "
                f"got {fit_token!r}"
            )
    else:
        # ---- Task-only R² (unconstrained projection) ----
        V_basis = V[:-1]  # (K-1, D), full rank
        P = V_basis.T @ torch.linalg.solve(V_basis @ V_basis.T, V_basis)
        h_task = (h_centered @ P)  # (N, D)

        ss_task = (h_task ** 2).sum().item()
        r2_task = ss_task / (ss_total + eps)

        # ---- Task + token R² ----
        if fit_token == "none" or covariates is None:
            r2_additive = r2_task
            ss_token = 0.0
        elif fit_token == "linear":
            residual = h_centered - h_task  # (N, D)

            if covariates.ndim == 3:
                d = covariates.shape[2]
                x_flat = covariates.to(device).float().reshape(N, d)
            else:
                d = covariates.shape[1]
                x_flat = covariates.to(device).float()
                x_flat = x_flat.unsqueeze(0).expand(n_tasks, B, d).reshape(N, d)

            XtX = x_flat.T @ x_flat
            XtX_reg = XtX + eps * torch.eye(d, device=XtX.device)
            W_x = torch.linalg.solve(XtX_reg, x_flat.T @ residual)
            h_token = x_flat @ W_x

            ss_token = (h_token ** 2).sum().item()
            ss_residual = ((residual - h_token) ** 2).sum().item()
            r2_additive = 1.0 - ss_residual / (ss_total + eps)
        elif fit_token == "anova":
            if token_vecs is None or token_ids is None:
                raise ValueError(
                    "fit_token='anova' requires token_vecs and token_ids"
                )
            Vt = token_vecs.to(device).float()
            Vt_basis = Vt[:-1]
            combined = torch.cat([V_basis, Vt_basis], dim=0)
            Q, _ = torch.linalg.qr(combined.T)
            P_comb = Q @ Q.T
            h_additive = h_centered @ P_comb

            ss_additive = (h_additive ** 2).sum().item()
            r2_additive = ss_additive / (ss_total + eps)
            ss_token = ss_additive - ss_task
        else:
            raise ValueError(
                f"fit_token must be 'none', 'linear', or 'anova', "
                f"got {fit_token!r}"
            )

    return AveragingR2Result(
        r2_task=r2_task,
        r2_additive=r2_additive,
        ss_total=ss_total,
        ss_task=ss_task,
        ss_token=ss_token,
        n_tasks=n_tasks,
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
    """Plot ANCOVA interaction proportion and separability R².

    To match the interpretation of :func:`plot_anova_separability`, we define

        η²_interaction = (R²_full - R²_additive) / R²_full
        separability R² = R²_additive / R²_full = 1 - η²_interaction

    so that both ANOVA and ANCOVA plots share the same semantics:
    small η²_interaction (or large separability R²) means additivity holds.

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

    def _eta2_interaction(res: ANCOVAResult) -> float:
        if res.r2_full <= 0:
            return 0.0
        return res.separability_gap / res.r2_full

    # ---- Figure 1: η²_interaction ----
    fig1, ax1 = plt.subplots(figsize=figsize)
    pos_list = []
    for i, l_num in enumerate(layers):
        pos_results = results[l_num]
        pos_list = sorted(pos_results.keys())
        if not pos_list:
            continue
        eta2_int = [_eta2_interaction(pos_results[p]) for p in pos_list]
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
        sep_r2 = [1.0 - _eta2_interaction(pos_results[p]) for p in pos_list]
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

    sample_res = sample_layer[all_pos[0]] if all_pos else None
    has_val = sample_res is not None and sample_res.r2_additive_val is not None

    if has_val:
        header = (
            f"{'Layer':>6} {'Pos':>5} "
            f"{'R²_add':>8} {'R²_full':>8} {'gap':>8} "
            f"{'R²_add_v':>9} {'R²_ful_v':>9} {'gap_v':>8} "
            f"{'N':>7} {'K':>3} {'d':>3}"
        )
    else:
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
            if has_val:
                print(
                    f"{l_num:>6} {pos:>5} "
                    f"{r.r2_additive:>8.4f} {r.r2_full:>8.4f} "
                    f"{r.separability_gap:>8.4f} "
                    f"{r.r2_additive_val:>9.4f} {r.r2_full_val:>9.4f} "
                    f"{r.separability_gap_val:>8.4f} "
                    f"{r.n_samples:>7} "
                    f"{r.n_tasks:>3} {r.n_covariate_dims:>3}"
                )
            else:
                print(
                    f"{l_num:>6} {pos:>5} {r.r2_additive:>8.4f} {r.r2_full:>8.4f} "
                    f"{r.separability_gap:>8.4f} {r.n_samples:>7} "
                    f"{r.n_tasks:>3} {r.n_covariate_dims:>3}"
                )

    print("=" * len(header))
