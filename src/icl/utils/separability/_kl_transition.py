"""Shared plotting helper for KL-transition heatmap figures.

All three experiment modules (coin, linear, latent) compute a
``log10_ratio_matrix`` and ``alpha_matrix`` over (k, training-step) grids and
render them as pcolormesh heatmaps.  This module extracts that rendering logic
into a single composable function so the three panels can share one set of
axes inside a compact combined figure.
"""

from __future__ import annotations

import numpy as np


_REL_VMAX = 2.0
_CMAP_NAME = "RdBu_r"


def _centers_to_edges(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    if vals.ndim != 1 or vals.size == 0:
        raise ValueError("Need a non-empty 1-D array of centers.")
    if vals.size == 1:
        w = 1.0 if vals[0] == 0 else max(1.0, abs(vals[0]) * 0.05)
        return np.array([vals[0] - w, vals[0] + w], dtype=float)
    mids = 0.5 * (vals[:-1] + vals[1:])
    left  = vals[0]  - (mids[0]  - vals[0])
    right = vals[-1] + (vals[-1] - mids[-1])
    return np.concatenate(([left], mids, [right]))


def plot_kl_transition_on_ax(
    ax,
    out: dict,
    *,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    show_colorbar: bool = False,
    fig=None,
):
    """Render the KL-ratio heatmap from a pre-computed result dict onto *ax*.

    Parameters
    ----------
    ax :
        Matplotlib Axes to draw on.
    out :
        Dict returned by any of the three ``plot_kl_model_vs_two_bayes_*_transition_across_k``
        functions.  Required keys:
        ``step_grid``, ``k_values_loaded``, ``log10_ratio_matrix``, ``alpha_matrix``.
    show_ylabel :
        If True, add the y-axis label.
    show_xlabel :
        If True, add the x-axis label.
    show_colorbar :
        If True, attach a colorbar to *ax* (pass this only for the rightmost panel).
        *fig* must be provided when ``show_colorbar=True``.
    fig :
        The parent Figure; required when ``show_colorbar=True``.

    Returns
    -------
    mesh : QuadMesh
        The pcolormesh artist, so the caller can attach a colorbar elsewhere.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    step_grid    = np.asarray(out["step_grid"],           dtype=float)
    ks           = np.asarray(out["k_values_loaded"],     dtype=float)
    log_ratio    = np.asarray(out["log10_ratio_matrix"],  dtype=float)
    alpha_matrix = np.asarray(out["alpha_matrix"],        dtype=float)

    x_edges = _centers_to_edges(step_grid)
    y_edges = _centers_to_edges(ks)

    rel_norm = mcolors.TwoSlopeNorm(vmin=-_REL_VMAX, vcenter=0.0, vmax=_REL_VMAX)
    cmap_rel = plt.get_cmap(_CMAP_NAME).copy()
    cmap_rel.set_bad(color="#f0f0f0")

    log_ratio_ma = np.ma.masked_invalid(log_ratio)

    mesh = ax.pcolormesh(
        x_edges, y_edges, log_ratio_ma,
        cmap=cmap_rel, norm=rel_norm, shading="auto",
    )

    # Apply per-cell alpha transparency (closeness modulation).
    rel_facecolors = cmap_rel(rel_norm(log_ratio_ma.filled(0.0)))
    rel_facecolors[..., -1] = alpha_matrix
    mesh.set_facecolor(rel_facecolors.reshape(-1, 4))

    ax.set_yticks(ks)
    ax.grid(False)
    ax.set_xlabel("Training Step")
    if not show_xlabel:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel(r"$\log_2(N_{\mathrm{minor}})$")

    if show_colorbar:
        assert fig is not None, "fig must be provided when show_colorbar=True"
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label(
            r"$\log(\mathrm{KL}_{\mathrm{Bayesian}} / \mathrm{KL}_{\mathrm{extrapolation}})$",
        )

    return mesh
