"""Plotting functions for the Real-LLM ICL notebook."""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Enough distinct colours for up to 8 ID tasks (tab10-inspired, dark-first).
_ID_COLORS = [
    "#1a5276",  # navy
    "#1e8449",  # forest green
    "#7d3c98",  # purple
    "#d35400",  # burnt orange
    "#148f77",  # teal
    "#5dade2",  # sky blue
    "#58d68d",  # light green
    "#c39bd3",  # lavender
]
_OOD_COLORS = ["#e74c3c", "#e67e22", "#8e44ad"]


def plot_performance_bars(
    id_perf: dict,
    ood_perf: dict,
) -> plt.Figure:
    """Bar charts of ICL accuracy and cross-entropy loss for ID and OOD tasks.

    Args:
        id_perf:  {task_name: {"accuracy": float, "mean_loss": float}}
        ood_perf: {task_name: {"accuracy": float, "mean_loss": float}}

    Returns:
        matplotlib Figure.
    """
    id_names   = list(id_perf.keys())
    ood_names  = list(ood_perf.keys())
    all_names  = id_names + ood_names
    accuracies = [id_perf[n]["accuracy"]  for n in id_names]  + \
                 [ood_perf[n]["accuracy"] for n in ood_names]
    losses     = [id_perf[n]["mean_loss"]  for n in id_names]  + \
                 [ood_perf[n]["mean_loss"] for n in ood_names]
    colors     = ["#2ca02c"] * len(id_names) + ["#d62728"] * len(ood_names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(all_names))

    ax = axes[0]
    bars = ax.bar(x, accuracies, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Next-Token Accuracy")
    ax.set_title("ICL Next-Token Accuracy (ID vs OOD)")
    ax.set_ylim(0, 1.05)
    for b, acc in zip(bars, accuracies):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    bars = ax.bar(x, losses, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("ICL Cross-Entropy Loss (ID vs OOD)")
    for b, loss_val in zip(bars, losses):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                f"{loss_val:.2f}", ha="center", va="bottom", fontsize=8)

    legend_elements = [
        Patch(facecolor="#2ca02c", edgecolor="k", label="ID"),
        Patch(facecolor="#d62728", edgecolor="k", label="OOD"),
    ]
    for a in axes:
        a.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.show()
    return fig


def plot_r2_vs_layer(
    id_r2: dict,
    ood_r2: dict,
    layers_range: list[int],
    model_name: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot task-subspace R² vs layer for ID and OOD groups.

    Left panel: individual curves per task.
    Right panel: mean ID vs mean OOD with shaded area.

    Args:
        id_r2:        {task_name: np.ndarray (L,)}
        ood_r2:       {task_name: np.ndarray (L,)}
        layers_range: list of layer indices (x-axis).
        model_name:   optional model name for the plot title.
        save_path:    if given, save the figure to this path at dpi=150.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)

    ax = axes[0]
    for i, (name, r2) in enumerate(id_r2.items()):
        ax.plot(layers_range, r2, color=_ID_COLORS[i % len(_ID_COLORS)], lw=2.0,
                label=f'ID: {name.replace("_", " ")}')
    for i, (name, r2) in enumerate(ood_r2.items()):
        ax.plot(layers_range, r2, color=_OOD_COLORS[i % len(_OOD_COLORS)], lw=2.0,
                ls="--", label=f'OOD: {name.replace("_", " ")}')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Task-subspace R²")
    ax.set_title("ID vs OOD — Task Subspace Alignment")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    mean_id_r2  = np.mean(np.stack(list(id_r2.values()),  axis=0), axis=0)
    mean_ood_r2 = np.mean(np.stack(list(ood_r2.values()), axis=0), axis=0)
    ax.plot(layers_range, mean_id_r2,  color="#1a5276", lw=2.5, label="ID mean")
    ax.plot(layers_range, mean_ood_r2, color="#e74c3c", lw=2.5, ls="--", label="OOD mean")
    ax.fill_between(layers_range, mean_id_r2,  alpha=0.12, color="#1a5276")
    ax.fill_between(layers_range, mean_ood_r2, alpha=0.12, color="#e74c3c")
    ax.set_xlabel("Layer")
    title = "Mean R²: ID vs OOD"
    if model_name:
        title += f"  ({model_name.split('/')[-1]})"
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_lambda_id(
    beta_pos_id: dict,
    task_names: list[str],
    layers: list[int],
    T_traj: int,
    N_SHOTS: int,
    K: int,
    save_dir: str = "..",
) -> None:
    """Plot mean affine coefficient λ_k vs shot position for ID tasks.

    One figure per layer in ``layers``, with one subplot per ID task arranged
    in two rows.  The curve for the task whose task vector matches the subplot
    is highlighted.

    Args:
        beta_pos_id: {layer_idx: {task_name: np.ndarray (T, N_eval, K)}}
        task_names:  list of ID task names (= keys of beta_pos_id[l]).
        layers:      layer indices to plot.
        T_traj:      total time steps (N_SHOTS + 1).
        N_SHOTS:     number of in-context shots (for x-axis labels).
        K:           number of ID tasks.
        save_dir:    directory to save PNG files.
    """
    import math

    positions_x = list(range(1, T_traj + 1))
    pos_labels  = [str(p) for p in range(1, N_SHOTS + 1)] + ["Q"]
    tick_pos    = positions_x[::2] + [positions_x[-1]]
    tick_labels = [pos_labels[i] for i in range(0, len(pos_labels), 2)] + [pos_labels[-1]]

    for l_idx in layers:
        n_tasks = len(task_names)
        n_cols  = math.ceil(n_tasks / 2)
        n_rows  = 2
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4.5 * n_cols, 7),
            sharey=True,
        )
        # Flatten to 1-D list; hide unused axes in the last row if n_tasks is odd
        axes_flat = axes.flatten()
        for k_ax in range(n_tasks, len(axes_flat)):
            axes_flat[k_ax].set_visible(False)

        for k_ax, task_name in enumerate(task_names):
            ax = axes_flat[k_ax]
            b_mean = beta_pos_id[l_idx][task_name].mean(axis=1)  # (T, K)
            for k_vec, vec_name in enumerate(task_names):
                is_match = (k_vec == k_ax)
                ax.plot(
                    positions_x, b_mean[:, k_vec],
                    color=_ID_COLORS[k_vec % len(_ID_COLORS)],
                    lw=2.5 if is_match else 1.2,
                    alpha=0.95 if is_match else 0.45,
                    ls="-" if is_match else "--",
                    label=vec_name.replace("_", " "),
                )
            ax.set_title(task_name.replace("_", " "), fontsize=13)
            ax.set_xlabel("Shot position", fontsize=12)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, fontsize=11)
            # Only the leftmost column of each row gets a y-label
            if k_ax % n_cols == 0:
                ax.set_ylabel(r"Mean affine coefficient $\beta_k$", fontsize=12)
            ax.set_ylim(-0.1, 1.05)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"real_llm_lambda_id_l{l_idx:02d}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)


def plot_lambda_ood(
    beta_pos_ood: dict,
    task_names: list[str],
    layers: list[int],
    T_traj: int,
    N_SHOTS: int,
    K: int,
    save_dir: str = "..",
) -> None:
    """Plot mean affine coefficient λ_k vs shot position for OOD tasks.

    One figure per layer in ``layers``, with one subplot per OOD task.
    All task-vector curves are shown with equal weight (no highlighting).

    Args:
        beta_pos_ood: {layer_idx: {ood_name: np.ndarray (T, N_eval, K)}}
        task_names:   list of ID task names (for vector labels in legend).
        layers:       layer indices to plot.
        T_traj:       total time steps (N_SHOTS + 1).
        N_SHOTS:      number of in-context shots (for x-axis labels).
        K:            number of ID tasks (for the 1/K reference line).
        save_dir:     directory to save PNG files.
    """
    positions_x = list(range(1, T_traj + 1))
    pos_labels  = [str(p) for p in range(1, N_SHOTS + 1)] + ["Q"]
    tick_pos    = positions_x[::2] + [positions_x[-1]]
    tick_labels = [pos_labels[i] for i in range(0, len(pos_labels), 2)] + [pos_labels[-1]]

    for l_idx in layers:
        ood_names = list(beta_pos_ood[l_idx].keys())
        n_ood = len(ood_names)
        fig, axes = plt.subplots(1, n_ood, figsize=(5 * n_ood, 4), sharey=True)
        if n_ood == 1:
            axes = [axes]

        for k_ax, ood_name in enumerate(ood_names):
            ax = axes[k_ax]
            b_mean = beta_pos_ood[l_idx][ood_name].mean(axis=1)  # (T, K)
            for k_vec, vec_name in enumerate(task_names):
                ax.plot(
                    positions_x, b_mean[:, k_vec],
                    color=_ID_COLORS[k_vec % len(_ID_COLORS)], lw=2.0,
                    label=vec_name.replace("_", " "),
                )
            ax.axhline(1 / K, color="gray", ls=":", lw=1.0, alpha=0.5, label="1/K")
            ax.set_title(f"OOD: {ood_name.replace('_', ' ')}", fontsize=11)
            ax.set_xlabel("Shot position")
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels)
            if k_ax == 0:
                ax.set_ylabel(r"Mean affine coefficient $\beta_k$")
            ax.set_ylim(-0.1, 1.05)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)

        fig.suptitle(
            f"OOD  $\\beta_k$ vs position — Layer {l_idx}  (should stay diffuse)",
            fontsize=12,
        )
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"real_llm_lambda_ood_l{l_idx:02d}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)
