"""Trajectory rendering (lines, scatter points, end markers) and
legend / cosmetics / reference stars."""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

from icl.utils.traj_plot._helpers import (
    _set_gid,
    _strip_b_suffix,
    _rgba,
    _r2_value_to_area,
)


# ============================================================
# Trajectory drawing
# ============================================================

def _draw_trajectories(
    ax, X_proj, J, sizes_area, *,
    idx_major, idx_ood, idx_minor,
    t_show, t_show_ood, t_base, aA, T,
    major_rgbs, ood_style, minor_style,
    labels, orig_task_idx, major_base_to_color,
    gid_prefix,
    # line params
    use_gradient_line, line_width, line_alpha_end, line_alpha_start_factor, line_alpha_power,
    major_linestyle, minor_linestyle,
    major_line_width, ood_line_alpha_factor,
    # point params
    mid_size_factor,
    point_alpha_start, point_alpha_end, point_alpha_power,
    show_pow2_anchors,
    # end marker params
    include_last, end_marker_alpha, end_marker_edge_color, end_marker_edge_width,
):
    """Draw all trajectory lines, scatter points, and end markers."""

    def _draw_gradient_polyline(Pxy, rgb, ls, gid_suffix, z=1.5,
                                lw_override=None, alpha_scale=1.0):
        if Pxy.shape[0] < 2:
            return None

        segs = np.stack([Pxy[:-1], Pxy[1:]], axis=1)
        nseg = segs.shape[0]
        uu = np.linspace(0.0, 1.0, nseg)

        a0 = float(line_alpha_end) * float(alpha_scale)
        a1 = float(line_alpha_end) * float(line_alpha_start_factor) * float(alpha_scale)
        al = a0 + (a1 - a0) * (uu ** float(line_alpha_power))
        al = np.clip(al, 0.0, 1.0)

        cols = np.column_stack(
            [
                np.full(nseg, rgb[0]),
                np.full(nseg, rgb[1]),
                np.full(nseg, rgb[2]),
                al,
            ]
        )

        lw = float(lw_override) if lw_override is not None else float(line_width)
        lc = LineCollection(
            segs,
            colors=cols,
            linewidths=lw,
            linestyles=ls,
            zorder=float(z),
        )
        ax.add_collection(lc)
        return _set_gid(lc, gid_prefix, gid_suffix)

    def _alpha_at(t_abs):
        u = t_abs.astype(float) / max(1.0, float(T - 1))
        a = float(point_alpha_end) + (float(point_alpha_start) - float(point_alpha_end)) * (
            (1.0 - u) ** float(point_alpha_power)
        )
        return np.clip(a, 0.0, 1.0)

    def _draw_points(Pfull, k, rgb, gid_suffix, t_abs=None):
        if t_abs is None:
            t_abs = t_show
        a_local = _alpha_at(t_abs)
        Pxy = Pfull[t_abs] + J[k, t_abs]

        fc = np.column_stack(
            [
                np.full_like(a_local, rgb[0], dtype=float),
                np.full_like(a_local, rgb[1], dtype=float),
                np.full_like(a_local, rgb[2], dtype=float),
                a_local,
            ]
        )

        coll = ax.scatter(
            Pxy[:, 0],
            Pxy[:, 1],
            s=sizes_area[k, t_abs] * float(mid_size_factor),
            marker="o",
            facecolors=fc,
            edgecolors="none",
            linewidths=0.0,
            zorder=2.2,
        )
        return _set_gid(coll, gid_prefix, gid_suffix)

    def _draw_pow2_anchors(Pfull, k, rgb, gid_suffix):
        if (not show_pow2_anchors) or (t_base.size == 0) or (aA is None):
            return None

        Pxy = Pfull[t_base] + J[k, t_base]

        fc = np.column_stack(
            [
                np.full_like(aA, rgb[0], dtype=float),
                np.full_like(aA, rgb[1], dtype=float),
                np.full_like(aA, rgb[2], dtype=float),
                aA,
            ]
        )

        coll = ax.scatter(
            Pxy[:, 0],
            Pxy[:, 1],
            s=sizes_area[k, t_base] * float(mid_size_factor),
            marker="o",
            facecolors=fc,
            edgecolors="none",
            linewidths=0.0,
            zorder=2.55,
        )
        return _set_gid(coll, gid_prefix, gid_suffix)

    def _draw_end(Pfull, k, rgb, gid_group):
        if (not include_last) or T <= 0:
            return None

        tlast = T - 1
        jxy = J[k, tlast]

        px, py = float(Pfull[tlast, 0] + jxy[0]), float(Pfull[tlast, 1] + jxy[1])

        en = ax.scatter(
            [px],
            [py],
            s=[sizes_area[k, -1] * float(mid_size_factor)],
            marker="o",
            facecolors=[(rgb[0], rgb[1], rgb[2], float(end_marker_alpha))],
            edgecolors=[end_marker_edge_color],
            linewidths=float(end_marker_edge_width),
            zorder=3.35,
        )
        return _set_gid(en, gid_prefix, f"{gid_group}_end:{k}")

    def _draw_one(k, Pfull, rgb, ls, gid_group, t_show_local=None,
                  lw_override=None, alpha_scale=1.0):
        t_line = t_show if t_show_local is None else t_show_local
        Pxy_line = Pfull[t_line]
        if use_gradient_line:
            _draw_gradient_polyline(Pxy_line, rgb, ls, f"{gid_group}_gline:{k}", z=1.6,
                                    lw_override=lw_override, alpha_scale=alpha_scale)
        else:
            lw = float(lw_override) if lw_override is not None else float(line_width)
            a = float(line_alpha_end) * float(alpha_scale)
            ln = ax.plot(
                Pxy_line[:, 0],
                Pxy_line[:, 1],
                color=(rgb[0], rgb[1], rgb[2], np.clip(a, 0.0, 1.0)),
                lw=lw,
                linestyle=ls,
                zorder=1.6,
            )[0]
            _set_gid(ln, gid_prefix, f"{gid_group}_line:{k}")

        _draw_points(Pfull, k, rgb, f"{gid_group}_pts:{k}", t_abs=t_line)
        _draw_pow2_anchors(Pfull, k, rgb, f"{gid_group}_anchors:{k}")
        _draw_end(Pfull, k, rgb, gid_group)

    def _major_rgb_for_k(k):
        if orig_task_idx is not None:
            try:
                ti = int(orig_task_idx[int(k)])
            except Exception:
                ti = None
            if ti is not None and 0 <= ti < 3:
                return major_rgbs[ti]
        base = _strip_b_suffix(labels[int(k)])
        ci = int(major_base_to_color.get(base, 0))
        return major_rgbs[ci]

    _major_lw = float(major_line_width) if major_line_width is not None else None
    _ood_alpha = float(ood_line_alpha_factor)

    for k in idx_major.tolist():
        _draw_one(int(k), X_proj[int(k)], _major_rgb_for_k(int(k)), major_linestyle, "major",
                  lw_override=_major_lw)

    for k in idx_ood.tolist():
        rgb, ls = ood_style[int(k)]
        _draw_one(int(k), X_proj[int(k)], rgb, ls, "ood", t_show_local=t_show_ood,
                  alpha_scale=_ood_alpha)

    for k in idx_minor.tolist():
        rgb = minor_style[int(k)]
        _draw_one(int(k), X_proj[int(k)], rgb, minor_linestyle, "minor")


# ============================================================
# Legend, cosmetics, and reference stars
# ============================================================

def _draw_legend_cosmetics_refs(
    ax, F_proj, major_rgbs, *,
    gid_prefix,
    # legend
    major_colors, major_linestyle, major_legend_prefix,
    maj_names=None,
    idx_ood, ood_base_color, ood_legend_label,
    idx_minor, minor_base_color, minor_linestyle,
    show_legend,
    R2, size_min, size_max, mid_size_factor,
    # cosmetics
    base_fontsize, title, axis_margin, hide_ticks, despine,
    # reference stars
    show_final_task_vecs, final_marker, final_marker_size,
    final_edge_color, final_edge_width,
    show_simplex_triangle, simplex_triangle_color,
    simplex_triangle_alpha, simplex_triangle_lw, simplex_triangle_ls,
):
    """Draw legend entries, axis cosmetics, reference stars, and simplex triangle."""

    # --- Legend handles ---
    handles = []
    for i in range(3):
        if maj_names is not None and i < len(maj_names):
            label = str(maj_names[i])
        else:
            label = f"{major_legend_prefix}{i + 1}"
        handles.append(
            Line2D(
                [0],
                [0],
                color=_rgba(major_colors[i], 0.95),
                lw=2.4,
                linestyle=major_linestyle,
                label=label,
            )
        )

    if idx_ood.size > 0:
        handles.append(
            Line2D(
                [0],
                [0],
                color=_rgba(ood_base_color, 0.95),
                lw=2.4,
                linestyle="--",
                label=ood_legend_label,
            )
        )

    if idx_minor.size > 0:
        handles.append(
            Line2D(
                [0],
                [0],
                color=_rgba(minor_base_color, 0.95),
                lw=2.4,
                linestyle=minor_linestyle,
                label="Minor trajectories",
            )
        )

    _fs = int(base_fontsize)
    _fs_large = _fs + 2

    if show_legend:
        leg1 = ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=_fs)

        r2_min = float(np.min(R2))
        r2_max = float(np.max(R2))

        r2_levels = np.quantile(R2, [0.2, 0.5, 0.8, 0.95]).astype(float)

        r2_levels = np.unique(np.round(r2_levels, 2))
        if r2_levels.size < 3:
            r2_levels = np.unique(np.round(np.array([r2_min, 0.5 * (r2_min + r2_max), r2_max]), 2))

        size_handles = []
        size_labels = []
        for v in r2_levels:
            area = _r2_value_to_area(
                v,
                r2_min=r2_min,
                r2_max=r2_max,
                size_min=size_min,
                size_max=size_max,
            )
            area *= float(mid_size_factor)

            h = ax.scatter(
                [], [],
                s=area,
                marker="o",
                facecolors=[(0, 0, 0, 0.18)],
                edgecolors=[(0, 0, 0, 0.45)],
                linewidths=0.9,
            )
            size_handles.append(h)
            size_labels.append(f"R² = {float(v):.2f}")

        leg2 = ax.legend(  # noqa: F841
            handles=size_handles,
            labels=size_labels,
            title="Marker size",
            scatterpoints=1,
            frameon=False,
            loc="upper right",
            borderpad=0.3,
            labelspacing=0.6,
            handletextpad=0.8,
            fontsize=_fs,
            title_fontsize=_fs,
        )

        ax.add_artist(leg1)
    else:
        if getattr(ax, "legend_", None) is not None:
            ax.legend_.remove()

    # --- Axis cosmetics ---
    ax.set_title(title, fontsize=_fs_large)
    ax.margins(float(axis_margin))
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(False)

    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for sp in ax.spines.values():
            sp.set_visible(False)
    else:
        ax.set_xlabel("axis 1", fontsize=_fs_large)
        ax.set_ylabel("axis 2", fontsize=_fs_large)
        ax.tick_params(labelsize=_fs)
        if despine:
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.spines["left"].set_alpha(0.45)
            ax.spines["bottom"].set_alpha(0.45)

    # --- Reference stars ---
    if show_final_task_vecs:
        for i in range(3):
            rgb = major_rgbs[i]
            ref = ax.scatter(
                [F_proj[i, 0]],
                [F_proj[i, 1]],
                s=float(final_marker_size),
                marker=final_marker,
                facecolors=[(rgb[0], rgb[1], rgb[2], 1.0)],
                edgecolors=[final_edge_color],
                linewidths=float(final_edge_width),
                zorder=4.2,
            )
            _set_gid(ref, gid_prefix, f"ref_vec:{i}")

        if show_simplex_triangle and F_proj.shape[0] == 3:
            tri_xs = np.append(F_proj[:, 0], F_proj[0, 0])
            tri_ys = np.append(F_proj[:, 1], F_proj[0, 1])
            tri_line, = ax.plot(
                tri_xs, tri_ys,
                color=simplex_triangle_color,
                alpha=simplex_triangle_alpha,
                lw=simplex_triangle_lw,
                linestyle=simplex_triangle_ls,
                zorder=1.0,
            )
            _set_gid(tri_line, gid_prefix, "simplex_triangle")
