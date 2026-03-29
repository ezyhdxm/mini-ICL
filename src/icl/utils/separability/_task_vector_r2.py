"""Task-vector R²: fraction of hidden-state variance explained by (task, token)."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch


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
    figsize: tuple = (5, 3.2),
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

    ax.set_xlabel("Position", fontsize=13)
    if show_ylabel:
        ax.set_ylabel("Residual variance ratio", fontsize=13)
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


def plot_task_vector_r2_on_ax(
    ax,
    results: Dict[int, Dict[int, TaskVectorR2Result]],
    *,
    log_x: bool = True,
    show_ylabel: bool = True,
):
    """Draw residual variance ratio (1 − R²) on an existing *ax*."""
    layers = sorted(results.keys())
    pos_list: list = []
    for l_num in layers:
        pos_results = results[l_num]
        pos_list = sorted(pos_results.keys())
        if not pos_list:
            continue
        vals = [1.0 - pos_results[p].r2 for p in pos_list]
        ax.plot(pos_list, vals, label=str(l_num),
                **_layer_style(l_num, len(pos_list)))

    ax.set_xlabel("Position")
    if show_ylabel:
        ax.set_ylabel("Residual variance ratio")
    if log_x and len(pos_list) > 1 and min(pos_list) >= 0:
        ax.set_xscale("symlog", linthresh=1)
    ax.set_ylim(-0.02, 1.02)
    _ncol = 2 if len(layers) > 6 else 1
    ax.legend(title="Layer", framealpha=0.9, loc="best", ncol=_ncol,
              borderaxespad=0.3, handlelength=2.2)
    ax.grid(True, alpha=0.25, linewidth=0.5)


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
