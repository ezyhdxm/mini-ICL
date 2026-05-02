"""Main orchestrator for the trajectory R² projection plot."""

import numpy as np
import matplotlib.pyplot as plt

from icl.utils.traj_plot._helpers import (
    _project_to_plane,
    _rigid_align_2d,
    _cleanup_previous_artists,
)
from icl.utils.traj_plot._prepare import _prepare_inputs
from icl.utils.traj_plot._time_indices import _compute_time_indices
from icl.utils.traj_plot._styles import _compute_group_styles
from icl.utils.traj_plot._draw import _draw_trajectories, _draw_legend_cosmetics_refs
from icl.utils.traj_plot._annotate import _annotate_trajectories


def project_with_r2_trajectories_group_colors_mpl(
    task_vecs_over_all_time,
    final_task_vecs,
    r2_scores,
    task_labels=None,

    # --- grouping ---
    n_ood=None,
    n_minor=0,
    b_ood=1,
    b_minor=1,
    b_major=1,
    use_mean=True,

    # --- axis reuse behavior ---
    ax=None,
    figsize=(7, 5.5),
    dpi=150,
    clear_ax=False,
    cleanup_previous=True,
    gid_prefix="proj_r2_traj_clear",

    # --- R² -> marker size ---
    size_min=4,
    size_max=12,

    # --- POINTS: show more ---
    max_points=5,
    mid_size_factor=0.55,

    # --- points alpha: early a bit darker, late lighter (end marker emphasized separately) ---
    point_alpha_start=0.34,
    point_alpha_end=0.25,
    point_alpha_power=1.6,

    # --- highlight pow2 anchors lightly ---
    show_pow2_anchors=True,
    anchor_alpha_floor=0.25,
    anchor_skip_ends=True,

    # --- NEW: time indices to always include AND annotate ---
    # Accepts int or iterable of ints. Negative indices allowed (python-style): -1 = last.
    must_include_times=None,

    # --- jitter to reduce overlap ---
    jitter_points=True,
    jitter_scale=0.0022,
    jitter_seed=0,

    # --- group base colors ---
    major_colors=("#1f77b4", "#9467bd", "#17becf"),
    ood_base_color="#ff7f0e",
    minor_base_color="#2ca02c",

    # --- line styles ---
    major_linestyle="-",
    minor_linestyle=":",
    ood_vary_linestyle=True,
    ood_linestyle_cycle=None,

    # --- LINES: darker at the beginning, lighter towards the end ---
    use_gradient_line=True,
    line_width=1.7,
    line_alpha_end=0.30,           # start alpha (darker)
    line_alpha_start_factor=0.60,  # end alpha = line_alpha_end * factor (lighter)
    line_alpha_power=1.45,

    # --- end emphasis ---
    include_last=True,
    end_marker_alpha=0.98,
    end_marker_edge_color=(0, 0, 0, 0.35),
    end_marker_edge_width=1.0,

    # --- OOD style variety ---
    ood_style_seed=0,
    ood_hue_jitter=0.10,
    ood_sat_jitter=0.20,
    ood_val_jitter=0.15,

    # --- reference vectors ---
    show_final_task_vecs=True,
    final_marker="*",
    final_marker_size=80,
    final_edge_color=(0, 0, 0, 0.95),
    final_edge_width=1.5,

    # --- simplex triangle ---
    show_simplex_triangle=True,
    simplex_triangle_color=(0.4, 0.4, 0.4),
    simplex_triangle_alpha=0.25,
    simplex_triangle_lw=1.2,
    simplex_triangle_ls="--",

    # --- cosmetics ---
    title="",
    hide_ticks=True,
    despine=True,

    # ============================================================
    # Annotations (boxes + leader lines)
    # ============================================================
    annotate=True,
    annotate_k=None,                  # None => annotate ALL OOD trajectories

    annotate_start=True,
    annotate_end=True,

    annotate_times=(0, -1),           # Start + End only by default

    annotation_fontsize=8,
    annotation_box_alpha=0.45,
    annotation_box_pad=0.20,
    annotation_box_lw=0.90,
    annotation_line_alpha=0.60,
    annotation_line_width=0.95,
    annotation_pad_px=7,

    # leader band (display px)
    annotation_max_leader_px=42,
    annotation_min_gap_px=6,
    annotation_prefer_length_scale=0.40,

    # keep boxes separated (display px)
    annotation_sep_px=6,

    # point avoidance
    annotation_point_clear_px=None,
    annotation_point_collision_cap=0,
    annotation_point_penalty=350.0,
    annotation_density_weight=14.0,

    # density field grid
    annotation_density_grid=(220, 220),
    annotation_density_blur_k=9,

    # solver controls
    annotation_candidate_angles_deg=(0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180),
    annotation_candidate_radii_n=8,
    annotation_beam_width=48,
    annotation_beam_expand=28,
    annotation_max_tries=4,
    annotation_max_leader_growth=1.55,
    annotation_polish_passes=2,
    annotation_debug=False,

    # --- override the automatic t_show selection ---
    t_show_override=None,

    # --- per-group time overrides ---
    ood_t_show_override=None,

    # --- per-group line style overrides ---
    major_line_width=None,
    ood_line_alpha_factor=1.0,

    # --- legend label customization ---
    major_legend_prefix="Maj: ",
    ood_legend_label="OOD",

    # --- font sizes ---
    base_fontsize=9,

    # --- axis padding ---
    axis_margin=0.03,

    # --- marker-size legend: explicit R² ticks ---
    legend_r2_values=None,

    # --- compatibility alias ---
    b_maj=None,  # alias for b_major
    step=None,
    show_legend=True,

    # --- rigid alignment in 2D after projection ---
    align_major_rigid=True,
    align_major_target=(0.0, 0.0),
    align_major_use_rotation=True,
    align_major_ref_dir=(1.0, 0.0),
    align_major_rotate_about="major1",

    # --- selective rendering of trajectories (does NOT affect projection) ---
    # None  -> draw all trajectories of that group
    # []    -> draw no trajectories of that group
    # [k,...] -> draw only trajectories whose index is in the list
    render_indices_major=None,
    render_indices_ood=None,
):
    """
    Key behavior:
      - must_include_times: those time indices are included in the polyline AND annotated.
      - annotate_start / annotate_end: can disable Start/End annotation without touching annotate_times.
        (must_include_times still forces annotation if it includes 0 or T-1.)
      - The "20th point" (index 19) is NOT annotated unless 19 is in must_include_times.
      - If any annotation boxes overlap, drop boxes iteratively until no overlaps remain.
    """

    # ---- normalize bool-ish flags ----
    clear_ax = bool(clear_ax)
    cleanup_previous = bool(cleanup_previous)
    use_mean = bool(use_mean)
    include_last = bool(include_last)
    show_pow2_anchors = bool(show_pow2_anchors)
    anchor_skip_ends = bool(anchor_skip_ends)
    jitter_points = bool(jitter_points)
    show_final_task_vecs = bool(show_final_task_vecs)
    hide_ticks = bool(hide_ticks)
    despine = bool(despine)
    annotate = bool(annotate)
    annotate_start = bool(annotate_start)
    annotate_end = bool(annotate_end)
    annotation_debug = bool(annotation_debug)

    # --- compatibility: accept b_maj as alias for b_major ---
    if b_maj is not None:
        try:
            b_maj_int = int(b_maj)
        except Exception as e:
            raise ValueError("b_maj must be int-like") from e
        try:
            b_major_int = int(b_major)
        except Exception:
            b_major_int = None
        if b_major_int is not None and b_major_int != 1 and b_major_int != b_maj_int:
            raise ValueError("Pass only one of b_major or b_maj (alias), or set them equal.")
        b_major = b_maj_int

    # ============================================================
    # 1. Prepare inputs (conversion, grouping, labels)
    # ============================================================

    data = _prepare_inputs(
        task_vecs_over_all_time, final_task_vecs, r2_scores, task_labels,
        n_ood=n_ood, n_minor=n_minor, b_ood=b_ood, b_minor=b_minor, b_major=b_major,
        use_mean=use_mean, ood_style_seed=ood_style_seed,
    )

    X       = data["X"]
    R2      = data["R2"]
    labels  = data["labels"]
    F       = data["F"]
    K, T, D = data["K"], data["T"], data["D"]
    idx_major = data["idx_major"]
    idx_ood   = data["idx_ood"]
    idx_minor = data["idx_minor"]
    orig_task_idx      = data["orig_task_idx"]
    major_base_to_color = data["major_base_to_color"]
    maj_names           = data.get("maj_names", None)

    # ============================================================
    # 2. Project to 2D plane
    # ============================================================

    X_proj, F_proj = _project_to_plane(X, np.asarray(F, dtype=float))

    if align_major_rigid:
        X_proj, F_proj = _rigid_align_2d(
            X_proj, F_proj,
            target=align_major_target,
            use_rotation=align_major_use_rotation,
            ref_dir=align_major_ref_dir,
            rotate_about=align_major_rotate_about,
        )

    # ============================================================
    # 3. Compute time indices, anchors, sizes
    # ============================================================

    ti = _compute_time_indices(
        T, R2,
        must_include_times=must_include_times,
        annotate_times=annotate_times,
        annotate=annotate, annotate_start=annotate_start, annotate_end=annotate_end,
        t_show_override=t_show_override, ood_t_show_override=ood_t_show_override,
        max_points=max_points, include_last=include_last,
        show_pow2_anchors=show_pow2_anchors, anchor_skip_ends=anchor_skip_ends,
        point_alpha_start=point_alpha_start, point_alpha_end=point_alpha_end,
        point_alpha_power=point_alpha_power, anchor_alpha_floor=anchor_alpha_floor,
        size_min=size_min, size_max=size_max,
    )

    t_show     = ti["t_show"]
    t_base     = ti["t_base"]
    aA         = ti["aA"]
    ann_times  = ti["ann_times"]
    must_times = ti["must_times"]
    t_show_ood = ti["t_show_ood"]
    sizes_area = ti["sizes_area"]

    # ============================================================
    # 4. Figure / Axes
    # ============================================================

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
        if clear_ax:
            ax.clear()
        elif cleanup_previous:
            _cleanup_previous_artists(ax, gid_prefix)

    ax.margins(0.10)

    # ============================================================
    # 5. Jitter
    # ============================================================

    x_all = X_proj[..., 0].ravel()
    y_all = X_proj[..., 1].ravel()
    x_rng = float(np.ptp(x_all) + 1e-9)
    y_rng = float(np.ptp(y_all) + 1e-9)

    if jitter_points:
        rng = np.random.default_rng(int(jitter_seed))
        J = rng.normal(size=(K, T, 2)).astype(float)
        J[:, :, 0] *= float(jitter_scale) * x_rng
        J[:, :, 1] *= float(jitter_scale) * y_rng
    else:
        J = np.zeros((K, T, 2), dtype=float)

    # ============================================================
    # 6. Compute group styles
    # ============================================================

    major_rgbs, ood_style, minor_style, ood_linestyle_cycle = _compute_group_styles(
        idx_ood, idx_minor,
        major_colors=major_colors, ood_base_color=ood_base_color, minor_base_color=minor_base_color,
        ood_linestyle_cycle=ood_linestyle_cycle, ood_style_seed=ood_style_seed,
        ood_hue_jitter=ood_hue_jitter, ood_sat_jitter=ood_sat_jitter, ood_val_jitter=ood_val_jitter,
        ood_vary_linestyle=ood_vary_linestyle,
    )

    # ============================================================
    # 7. Draw trajectories
    # ============================================================

    _draw_trajectories(
        ax, X_proj, J, sizes_area,
        idx_major=idx_major, idx_ood=idx_ood, idx_minor=idx_minor,
        t_show=t_show, t_show_ood=t_show_ood, t_base=t_base, aA=aA, T=T,
        major_rgbs=major_rgbs, ood_style=ood_style, minor_style=minor_style,
        labels=labels, orig_task_idx=orig_task_idx, major_base_to_color=major_base_to_color,
        gid_prefix=gid_prefix,
        use_gradient_line=use_gradient_line, line_width=line_width,
        line_alpha_end=line_alpha_end, line_alpha_start_factor=line_alpha_start_factor,
        line_alpha_power=line_alpha_power,
        major_linestyle=major_linestyle, minor_linestyle=minor_linestyle,
        major_line_width=major_line_width, ood_line_alpha_factor=ood_line_alpha_factor,
        mid_size_factor=mid_size_factor,
        point_alpha_start=point_alpha_start, point_alpha_end=point_alpha_end,
        point_alpha_power=point_alpha_power,
        show_pow2_anchors=show_pow2_anchors,
        include_last=include_last, end_marker_alpha=end_marker_alpha,
        end_marker_edge_color=end_marker_edge_color, end_marker_edge_width=end_marker_edge_width,
        render_indices_major=render_indices_major,
        render_indices_ood=render_indices_ood,
    )

    # ============================================================
    # 8. Legend, cosmetics, reference stars
    # ============================================================

    _draw_legend_cosmetics_refs(
        ax, F_proj, major_rgbs,
        gid_prefix=gid_prefix,
        major_colors=major_colors, major_linestyle=major_linestyle,
        major_legend_prefix=major_legend_prefix,
        maj_names=maj_names,
        idx_ood=idx_ood, ood_base_color=ood_base_color, ood_legend_label=ood_legend_label,
        idx_minor=idx_minor, minor_base_color=minor_base_color, minor_linestyle=minor_linestyle,
        show_legend=show_legend,
        R2=R2, size_min=size_min, size_max=size_max, mid_size_factor=mid_size_factor,
        base_fontsize=base_fontsize, title=title, axis_margin=axis_margin,
        hide_ticks=hide_ticks, despine=despine,
        show_final_task_vecs=show_final_task_vecs, final_marker=final_marker,
        final_marker_size=final_marker_size, final_edge_color=final_edge_color,
        final_edge_width=final_edge_width,
        show_simplex_triangle=show_simplex_triangle, simplex_triangle_color=simplex_triangle_color,
        simplex_triangle_alpha=simplex_triangle_alpha, simplex_triangle_lw=simplex_triangle_lw,
        simplex_triangle_ls=simplex_triangle_ls,
        legend_r2_values=legend_r2_values,
    )

    # ============================================================
    # 9. Annotations
    # ============================================================

    return _annotate_trajectories(
        fig, ax,
        X_proj=X_proj, J=J, R2=R2, sizes_area=sizes_area, F_proj=F_proj,
        t_show=t_show, ann_times=ann_times, must_times=must_times, T=T, K=K,
        idx_ood=idx_ood, ood_style=ood_style, major_rgbs=major_rgbs,
        orig_task_idx=orig_task_idx, labels=labels, major_base_to_color=major_base_to_color,
        gid_prefix=gid_prefix,
        mid_size_factor=mid_size_factor,
        show_final_task_vecs=show_final_task_vecs, final_marker_size=final_marker_size,
        annotate=annotate, annotate_k=annotate_k,
        annotation_fontsize=annotation_fontsize,
        annotation_box_alpha=annotation_box_alpha, annotation_box_pad=annotation_box_pad,
        annotation_box_lw=annotation_box_lw,
        annotation_line_alpha=annotation_line_alpha, annotation_line_width=annotation_line_width,
        annotation_pad_px=annotation_pad_px,
        annotation_max_leader_px=annotation_max_leader_px,
        annotation_min_gap_px=annotation_min_gap_px,
        annotation_prefer_length_scale=annotation_prefer_length_scale,
        annotation_sep_px=annotation_sep_px,
        annotation_point_clear_px=annotation_point_clear_px,
        annotation_point_collision_cap=annotation_point_collision_cap,
        annotation_point_penalty=annotation_point_penalty,
        annotation_density_weight=annotation_density_weight,
        annotation_density_grid=annotation_density_grid,
        annotation_density_blur_k=annotation_density_blur_k,
        annotation_candidate_angles_deg=annotation_candidate_angles_deg,
        annotation_candidate_radii_n=annotation_candidate_radii_n,
        annotation_beam_width=annotation_beam_width, annotation_beam_expand=annotation_beam_expand,
        annotation_max_tries=annotation_max_tries, annotation_max_leader_growth=annotation_max_leader_growth,
        annotation_polish_passes=annotation_polish_passes,
        annotation_debug=annotation_debug,
    )
