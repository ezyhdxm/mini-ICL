"""Shared plotting helper for beta/alpha trajectory figures.

The three experiment modules (coin, linear, latent) all produce a
``results_by_task`` dict  { task_id → {'beta': (B,T,K), 'post': (B,T,K)} }
using almost identical plotting code.  This module extracts that logic into a
single composable function so the three panels can be drawn onto pre-existing
axes for a compact combined figure.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


_DEFAULT_COLORS = ("#0072B2", "#E69F00", "#009E73")
_MARKERS = ("o", "s", "^", "D", "v", "P", "*", "X")


def plot_beta_alpha_on_ax(
    ax,
    results_by_task: Dict[int, Dict[str, np.ndarray]],
    k_major: int,
    task_ids: List[int],
    pidx: np.ndarray,
    *,
    project_beta_simplex: bool = False,
    beta_errbar: str = "quantile",
    corner_colors: Tuple[str, ...] = _DEFAULT_COLORS,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    add_labels: bool = False,
) -> Tuple[float, float]:
    """Draw beta/alpha trajectories for a single task onto *ax*.

    Parameters
    ----------
    ax :
        Matplotlib Axes to draw on.
    results_by_task :
        Dict mapping each task id to ``{'beta': (B, T, K), 'post': (B, T, K)}``.
    k_major :
        Number of major tasks (= K dimension of the arrays).
    task_ids :
        Which task ids to plot as separate rows.  For the paper figure this is
        always ``[0]`` (Task 1 only), which means *ax* receives one row of data.
    pidx :
        1-D array of position indices to include on the x-axis.
    project_beta_simplex :
        If True, project beta_mean onto the probability simplex before plotting.
    beta_errbar :
        ``'quantile'`` (25th–75th pct) or ``'std'``.
    corner_colors :
        Colours for the K major tasks.
    show_ylabel :
        If True, label the y-axis with "Task {tid+1}".
    show_xlabel :
        If True, label the x-axis with "Position t".
    add_labels :
        If True, attach legend labels ($\\beta_k$, $\\alpha_k$) to the artists.
        Pass True only for the panel that will carry the figure legend.

    Returns
    -------
    y_lo, y_hi : float
        Data range (before padding) so the caller can set a shared y-limit.
    """
    import matplotlib.colors as mcolors

    corner_rgb = [mcolors.to_rgb(c) for c in corner_colors]
    T_plot = len(pidx)
    ts = pidx.astype(float)

    # Jitter x positions so overlapping error bars are distinguishable.
    jitter_width = 1.0
    jitter_offsets = np.linspace(-jitter_width / 2, jitter_width / 2, k_major)

    # Show all points in the first 5 positions, then thin by 2 with per-task offset.
    dense_cutoff = 5
    thin_step = 2
    show_masks = []
    for col in range(k_major):
        m = np.zeros(T_plot, dtype=bool)
        m[:dense_cutoff] = True
        for i in range(dense_cutoff, T_plot):
            if (i - dense_cutoff + col) % thin_step == 0:
                m[i] = True
        show_masks.append(m)

    y_lo, y_hi = np.inf, -np.inf

    for row, tid in enumerate(task_ids):
        beta_all = results_by_task[tid]["beta"][:, pidx, :]  # (B, T_plot, K)
        post_all = results_by_task[tid]["post"][:, pidx, :]  # (B, T_plot, K)

        beta_mean = beta_all.mean(axis=0)   # (T_plot, K)
        beta_std  = beta_all.std(axis=0)

        if project_beta_simplex:
            for t in range(beta_mean.shape[0]):
                v = beta_mean[t]
                u = np.sort(v)[::-1]
                cssv = np.cumsum(u) - 1.0
                rho = np.nonzero(u > cssv / np.arange(1, len(u) + 1))[0][-1]
                theta = cssv[rho] / (rho + 1.0)
                beta_mean[t] = np.maximum(v - theta, 0.0)

        if beta_errbar == "quantile":
            beta_q_lo = np.percentile(beta_all, 25, axis=0)
            beta_q_hi = np.percentile(beta_all, 75, axis=0)

        post_mean = post_all.mean(axis=0)
        post_q10  = np.percentile(post_all, 10, axis=0)
        post_q90  = np.percentile(post_all, 90, axis=0)

        for col in range(k_major):
            c  = corner_rgb[col]
            mk = _MARKERS[col % len(_MARKERS)]
            ts_j = ts + jitter_offsets[col]
            sm   = show_masks[col]

            if beta_errbar == "quantile":
                yerr_lo = np.clip(beta_mean[sm, col] - beta_q_lo[sm, col], 0, None)
                yerr_hi = np.clip(beta_q_hi[sm, col] - beta_mean[sm, col], 0, None)
                yerr = [yerr_lo, yerr_hi]
            else:
                yerr = beta_std[sm, col]

            if project_beta_simplex:
                bm = beta_mean[sm, col]
                if isinstance(yerr, list):
                    yerr[0] = np.minimum(yerr[0], bm)
                    yerr[1] = np.minimum(yerr[1], 1.0 - bm)
                else:
                    yerr = [np.minimum(yerr, bm), np.minimum(yerr, 1.0 - bm)]

            beta_label = rf"$\beta_{{{col+1}}}$" if add_labels else None
            alpha_label = rf"$\alpha_{{{col+1}}}$" if add_labels else None

            ax.errorbar(
                ts_j[sm], beta_mean[sm, col],
                yerr=yerr,
                fmt=mk, color=c, markersize=3.5, linewidth=1.2,
                capsize=2, capthick=0.8, elinewidth=0.8,
                label=beta_label, zorder=3,
            )
            ax.plot(
                ts, post_mean[:, col], color=c, lw=2.2, ls="--", alpha=0.9,
                label=alpha_label, zorder=2,
            )
            ax.fill_between(
                ts, post_q10[:, col], post_q90[:, col],
                color=c, alpha=0.15, linewidth=0, zorder=1,
            )

        if beta_errbar == "quantile":
            data_lo = beta_q_lo.ravel()
            data_hi = beta_q_hi.ravel()
        else:
            data_lo = (beta_mean - beta_std).ravel()
            data_hi = (beta_mean + beta_std).ravel()

        all_vals = np.concatenate([data_lo, data_hi, post_q10.ravel(), post_q90.ravel()])
        y_lo = min(y_lo, float(np.nanmin(all_vals)))
        y_hi = max(y_hi, float(np.nanmax(all_vals)))

        ax.grid(axis="y", alpha=0.3)
        if show_ylabel:
            k_major_label = k_major  # for OOD tasks
            task_label = f"Task {tid + 1}" if tid < k_major_label else f"OOD {tid + 1}"
            ax.set_ylabel(task_label)
        if show_xlabel:
            ax.set_xlabel("Position $t$")

    return y_lo, y_hi
