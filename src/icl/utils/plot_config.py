"""
Central plotting configuration for paper-ready figures.
- Default title is empty (figures go in papers without redundant titles).
- Font sizes increased for readability in print.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Sequence, Tuple

# Paper-friendly font sizes (increase from matplotlib defaults for readability)
_PAPER_RC = {
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
}


def apply_paper_style():
    """Apply paper-friendly matplotlib defaults. Call at start of plotting modules."""
    matplotlib.rcParams.update(_PAPER_RC)


# Apply when this module is imported (so all icl plots use it)
apply_paper_style()


# ---------------------------------------------------------------------------
# Smooth simplex (ternary) heatmap
# ---------------------------------------------------------------------------

_SQRT3_2 = np.sqrt(3) / 2


def _bary_to_cart(alphas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (N, 3) barycentric coords to 2-D Cartesian on the standard
    equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt3/2)."""
    x = alphas[:, 1] + alphas[:, 2] * 0.5
    y = alphas[:, 2] * _SQRT3_2
    return x, y


def _point_in_triangle(px, py, tol=1e-6):
    """Boolean mask: True for grid points inside the equilateral triangle."""
    s = py / _SQRT3_2
    t = (px - 0.5 * py / _SQRT3_2)
    return (s >= -tol) & (s <= 1 + tol) & (t >= -tol) & (t <= 1 - s + tol)


def _bin_and_smooth(
    bary_x: np.ndarray,
    bary_y: np.ndarray,
    vals: np.ndarray,
    grid_res: int = 128,
    sigma: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin scattered data onto a regular grid, average per cell, then Gaussian-smooth.

    Returns xi, yi, Zi (smoothed 2-D array), mask (triangle interior).
    """
    from scipy.ndimage import gaussian_filter
    from scipy.stats import binned_statistic_2d

    x_edges = np.linspace(-0.02, 1.02, grid_res + 1)
    y_edges = np.linspace(-0.02, _SQRT3_2 + 0.02, grid_res + 1)

    stat, _, _, _ = binned_statistic_2d(
        bary_x, bary_y, vals,
        statistic="mean", bins=[x_edges, y_edges],
    )
    count, _, _, _ = binned_statistic_2d(
        bary_x, bary_y, vals,
        statistic="count", bins=[x_edges, y_edges],
    )
    # stat has shape (grid_res, grid_res) with x along axis-0, y along axis-1
    # transpose so that rows = y, cols = x for imshow with origin="lower"
    Zi = stat.T
    Ci = count.T

    xi = 0.5 * (x_edges[:-1] + x_edges[1:])
    yi = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xi, Yi = np.meshgrid(xi, yi)
    mask = _point_in_triangle(Xi, Yi)

    # Fill empty bins inside triangle with nearest non-empty neighbor
    empty = np.isnan(Zi) | (Ci < 1)
    if empty.any() and (~empty).any():
        from scipy.interpolate import griddata
        filled_pts = np.column_stack([Xi[~empty], Yi[~empty]])
        filled_vals = Zi[~empty]
        fill = griddata(filled_pts, filled_vals,
                        (Xi[empty], Yi[empty]), method="nearest")
        Zi[empty] = fill

    # Gaussian smooth (only over finite values)
    Zi_safe = np.nan_to_num(Zi, nan=0.0)
    Zi_smooth = gaussian_filter(Zi_safe, sigma=sigma)
    weight = gaussian_filter((~np.isnan(Zi)).astype(float), sigma=sigma)
    weight = np.where(weight > 0, weight, 1.0)
    Zi_smooth = Zi_smooth / weight

    Zi_smooth[~mask] = np.nan

    return xi, yi, Zi_smooth, mask


def _render_simplex_panels(
    bary_x: np.ndarray,
    bary_y: np.ndarray,
    data_list: Sequence[np.ndarray],
    titles: Sequence[str],
    *,
    metric_label: str = "KL divergence",
    mean_fmt: str = "mean KL = {:.4f}",
    vmin: float,
    vmax: float,
    figsize: Tuple[float, float] = (10, 3.8),
    wspace: float = -0.55,
    cmap: str = "magma_r",
    grid_res: int = 128,
    smooth_sigma: float = 2.0,
) -> plt.Figure:
    """Low-level helper: draw smooth simplex heatmap panels via binning + Gaussian blur.

    Callers are responsible for show/close.
    """
    fig, axes = plt.subplots(
        1, len(data_list), figsize=figsize,
        gridspec_kw={"wspace": wspace},
    )
    if not hasattr(axes, "__len__"):
        axes = [axes]

    n_panels = len(titles)
    im = None
    for idx, (ax, vals, title) in enumerate(zip(axes, data_list, titles)):
        xi, yi, Zi, _ = _bin_and_smooth(
            bary_x, bary_y, vals,
            grid_res=grid_res, sigma=smooth_sigma,
        )

        im = ax.imshow(
            Zi, extent=[xi[0], xi[-1], yi[0], yi[-1]],
            origin="lower", aspect="equal",
            cmap=cmap, vmin=vmin, vmax=vmax,
            interpolation="bilinear", rasterized=True,
        )

        tri_x = [0, 1, 0.5, 0]
        tri_y = [0, 0, _SQRT3_2, 0]
        ax.plot(tri_x, tri_y, "k-", lw=1.2)

        if idx == 0:
            ax.text(-0.03, -0.02, r"$\alpha_1$",
                    ha="right", va="top", fontsize=15)
        if idx == n_panels - 1:
            ax.text(1.03, -0.02, r"$\alpha_2$",
                    ha="left", va="top", fontsize=15)
        ax.text(0.5, _SQRT3_2 + 0.03, r"$\alpha_3$",
                ha="center", va="bottom", fontsize=15)
        ax.text(0.5, -0.08, title,
                ha="center", va="top", fontsize=14)
        ax.text(0.5, -0.18, mean_fmt.format(vals.mean()),
                ha="center", va="top", fontsize=12, color="0.4")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.22, _SQRT3_2 + 0.08)

    cbar = fig.colorbar(
        im, ax=list(axes), orientation="horizontal",
        fraction=0.04, pad=0.08, aspect=30,
    )
    cbar.set_label(metric_label, fontsize=14)
    cbar.ax.tick_params(labelsize=11)
    fig.subplots_adjust(bottom=0.12, wspace=wspace)

    return fig


def plot_simplex_panels(
    alphas: np.ndarray,
    data_list: Sequence[np.ndarray],
    titles: Sequence[str],
    *,
    metric_label: str = "KL divergence",
    mean_fmt: str = "mean KL = {:.4f}",
    color_vmin: Optional[float] = None,
    color_vmax: Optional[float] = None,
    figsize: Tuple[float, float] = (10, 3.8),
    wspace: float = -0.15,
    cmap: str = "magma_r",
    grid_res: int = 128,
    smooth_sigma: float = 2.0,
    show: bool = True,
) -> plt.Figure:
    """Draw smooth simplex heatmap panels via binned averaging + Gaussian blur.

    Parameters
    ----------
    alphas : (N, 3) array of barycentric coordinates.
    data_list : list of (N,) arrays, one per panel.
    titles : list of panel titles.
    metric_label : colorbar label.
    mean_fmt : format string for the per-panel mean annotation.
    grid_res : number of bins along each axis.
    smooth_sigma : Gaussian kernel σ (in grid-cell units) for smoothing.
    """
    bary_x, bary_y = _bary_to_cart(alphas)

    vmin = color_vmin if color_vmin is not None else min(d.min() for d in data_list)
    vmax = color_vmax if color_vmax is not None else max(d.max() for d in data_list)

    return _render_simplex_panels(
        bary_x, bary_y, data_list, titles,
        metric_label=metric_label, mean_fmt=mean_fmt,
        vmin=vmin, vmax=vmax,
        figsize=figsize, wspace=wspace, cmap=cmap,
        grid_res=grid_res, smooth_sigma=smooth_sigma,
    )


def _draw_single_simplex_panel(
    ax,
    bary_x: np.ndarray,
    bary_y: np.ndarray,
    vals: np.ndarray,
    vmin: float,
    vmax: float,
    title: str,
    mean_fmt: str,
    *,
    cmap: str = "magma_r",
    grid_res: int = 128,
    smooth_sigma: float = 2.0,
    show_alpha1: bool = True,
    show_alpha2: bool = True,
    show_alpha3: bool = True,
    title_x: float = 0.5,
    title_ha: str = "center",
):
    """Draw a single simplex heatmap panel onto *ax* and return the imshow artist."""
    xi, yi, Zi, _ = _bin_and_smooth(
        bary_x, bary_y, vals, grid_res=grid_res, sigma=smooth_sigma,
    )
    im = ax.imshow(
        Zi, extent=[xi[0], xi[-1], yi[0], yi[-1]],
        origin="lower", aspect="equal",
        cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="bilinear", rasterized=True,
    )
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, _SQRT3_2, 0]
    ax.plot(tri_x, tri_y, "k-", lw=1.2)

    if show_alpha1:
        ax.text(-0.03, -0.02, r"$\alpha_1$", ha="right", va="top", fontsize=11)
    if show_alpha2:
        ax.text(1.03, -0.02, r"$\alpha_2$", ha="left",  va="top", fontsize=11)
    if show_alpha3:
        ax.text(0.5, _SQRT3_2 + 0.03, r"$\alpha_3$", ha="center", va="bottom", fontsize=11)

    ax.text(title_x, -0.08, title, ha=title_ha, va="top", fontsize=11)
    ax.text(title_x, -0.17, mean_fmt.format(vals.mean()), ha=title_ha, va="top", fontsize=9, color="0.4")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.22, _SQRT3_2 + 0.08)
    return im


def _render_simplex_combined_fig(
    panels,
    title_left: str,
    title_right: str,
    figsize: Tuple[float, float],
    cmap: str,
    grid_res: int,
    smooth_sigma: float,
    kl_vmin: float,
    kl_vmax: float,
    rmse_vmin: float,
    rmse_vmax: float,
) -> plt.Figure:
    """Shared renderer for injection-simplex combined figures.

    Parameters
    ----------
    panels : list of (result_dict, vmin, vmax, key_left, key_right, mean_fmt)
    title_left / title_right : column labels for the left and right triangle panels.
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    import matplotlib.colors as mcolors

    fig = plt.figure(figsize=figsize)

    # Panels occupy [0.01, 0.82]; the remaining 0.18 is split between the
    # alpha-corner-label spillover and the dual-sided colorbar.
    outer = GridSpec(
        1, 3, figure=fig,
        left=0.01, right=0.82,
        wspace=0.08,
    )

    last_ax_right = None
    for col_idx, (result, vmin, vmax, key_left, key_right, mean_fmt) in enumerate(panels):
        bary_x, bary_y = _bary_to_cart(result["alphas"])
        data_left  = result[key_left]
        data_right = result[key_right]

        # Negative wspace pulls the paired triangles close together.
        inner = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[col_idx],
            wspace=-0.18,
        )
        ax_left  = fig.add_subplot(inner[0])
        ax_right = fig.add_subplot(inner[1])
        last_ax_right = ax_right

        # α₁ label only on far-left panel; α₂ only on far-right panel.
        show_a1 = (col_idx == 0)
        show_a2 = (col_idx == 2)

        _draw_single_simplex_panel(
            ax_left, bary_x, bary_y, data_left, vmin, vmax,
            title_left, mean_fmt, cmap=cmap, grid_res=grid_res, smooth_sigma=smooth_sigma,
            show_alpha1=show_a1, show_alpha2=False, show_alpha3=True,
            title_x=0.5, title_ha="center",
        )
        _draw_single_simplex_panel(
            ax_right, bary_x, bary_y, data_right, vmin, vmax,
            title_right, mean_fmt, cmap=cmap, grid_res=grid_res, smooth_sigma=smooth_sigma,
            show_alpha1=False, show_alpha2=show_a2, show_alpha3=True,
            title_x=0.5, title_ha="center",
        )

    # ── Single dual-sided colorbar, matched to axes height ────────────────────
    # Force layout so get_position() returns the final figure-coordinate bounds.
    fig.canvas.draw()
    pos = last_ax_right.get_position()

    n_ticks = 8
    tick_pos  = np.linspace(0.0, 1.0, n_ticks)
    kl_vals   = np.linspace(kl_vmin,   kl_vmax,   n_ticks)
    rmse_vals = np.linspace(rmse_vmin, rmse_vmax, n_ticks)

    cb_ax = fig.add_axes([0.87, pos.y0, 0.020, pos.height])
    norm  = mcolors.Normalize(vmin=0.0, vmax=1.0)
    sm    = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar  = fig.colorbar(sm, cax=cb_ax)

    cbar.set_ticks(tick_pos)
    cbar.set_ticklabels([f"{v:.1f}" for v in kl_vals], fontsize=9)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    cbar.set_label("KL divergence", fontsize=10, labelpad=8)

    ax_rmse_twin = cbar.ax.twinx()
    ax_rmse_twin.set_ylim(0.0, 1.0)
    ax_rmse_twin.set_yticks(tick_pos)
    ax_rmse_twin.set_yticklabels([f"{int(v)}" for v in rmse_vals], fontsize=9)
    ax_rmse_twin.set_ylabel("RMSE", fontsize=10, labelpad=8)

    return fig


def render_injection_simplex_combined(
    coin_result: dict,
    linear_result: dict,
    latent_result: dict,
    *,
    kl_vmin: float = 0.0,
    kl_vmax: float = 0.7,
    rmse_vmin: float = 0.0,
    rmse_vmax: float = 7.0,
    title_base: str = "Unsteered",
    title_inj: str = "Steered",
    figsize: Tuple[float, float] = (12, 3.8),
    cmap: str = "magma_r",
    grid_res: int = 128,
    smooth_sigma: float = 2.0,
) -> plt.Figure:
    """6-panel figure: [Unsteered | Steered] for E1, E2, E3."""
    panels = [
        (coin_result,   kl_vmin,   kl_vmax,   "kl_baseline",   "kl_injected",   "mean KL = {:.4f}"),
        (linear_result, rmse_vmin, rmse_vmax,  "rmse_baseline", "rmse_injected", "mean RMSE = {:.4f}"),
        (latent_result, kl_vmin,   kl_vmax,   "kl_baseline",   "kl_injected",   "mean KL = {:.4f}"),
    ]
    return _render_simplex_combined_fig(
        panels, title_base, title_inj,
        figsize, cmap, grid_res, smooth_sigma,
        kl_vmin, kl_vmax, rmse_vmin, rmse_vmax,
    )


def render_mode_task_comparison_combined(
    coin_result: dict,
    linear_result: dict,
    latent_result: dict,
    *,
    kl_vmin: float = 0.0,
    kl_vmax: float = 0.7,
    rmse_vmin: float = 0.0,
    rmse_vmax: float = 7.0,
    title_inj: str = "Steered",
    title_mode: str = r"Mode task $k^\star$",
    figsize: Tuple[float, float] = (12, 3.8),
    cmap: str = "magma_r",
    grid_res: int = 128,
    smooth_sigma: float = 2.0,
) -> plt.Figure:
    """6-panel figure: [Steered | Mode task k*] for E1, E2, E3."""
    panels = [
        (coin_result,   kl_vmin,   kl_vmax,   "kl_injected",   "kl_mode",   "mean KL = {:.4f}"),
        (linear_result, rmse_vmin, rmse_vmax,  "rmse_injected", "rmse_mode", "mean RMSE = {:.4f}"),
        (latent_result, kl_vmin,   kl_vmax,   "kl_injected",   "kl_mode",   "mean KL = {:.4f}"),
    ]
    return _render_simplex_combined_fig(
        panels, title_inj, title_mode,
        figsize, cmap, grid_res, smooth_sigma,
        kl_vmin, kl_vmax, rmse_vmin, rmse_vmax,
    )


def plot_simplex_panel_pairs(
    alphas: np.ndarray,
    data_base: np.ndarray,
    data_inj: np.ndarray,
    data_mode: np.ndarray,
    *,
    title_base: str = "Unsteered",
    title_inj: str = "Steered",
    title_mode: str = r"Mode task $k^\star$",
    metric_label: str = "KL divergence",
    mean_fmt: str = "mean KL = {:.4f}",
    color_vmin: Optional[float] = None,
    color_vmax: Optional[float] = None,
    figsize: Tuple[float, float] = (10, 3.8),
    wspace: float = -0.55,
    cmap: str = "magma_r",
    grid_res: int = 128,
    smooth_sigma: float = 2.0,
) -> Tuple[plt.Figure, plt.Figure]:
    """Create two 2-panel simplex figures.

    Returns
    -------
    fig1 : Unsteered vs Steered
    fig2 : Steered vs Mode task k*
    """
    bary_x, bary_y = _bary_to_cart(alphas)

    all_data = [data_base, data_inj, data_mode]
    vmin = color_vmin if color_vmin is not None else min(d.min() for d in all_data)
    vmax = color_vmax if color_vmax is not None else max(d.max() for d in all_data)

    common = dict(
        metric_label=metric_label, mean_fmt=mean_fmt,
        vmin=vmin, vmax=vmax,
        figsize=figsize, wspace=wspace, cmap=cmap,
        grid_res=grid_res, smooth_sigma=smooth_sigma,
    )

    fig1 = _render_simplex_panels(
        bary_x, bary_y,
        [data_base, data_inj], [title_base, title_inj],
        **common,
    )
    fig2 = _render_simplex_panels(
        bary_x, bary_y,
        [data_inj, data_mode], [title_inj, title_mode],
        **common,
    )
    return fig1, fig2
