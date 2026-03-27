"""Compute which time indices to display, pow2 anchors, and R²→size mapping."""

import numpy as np

from icl.utils.traj_plot._helpers import (
    _normalize_time_list,
    _pow2_time_indices,
    _r2_to_sizes_area,
)


def _compute_time_indices(
    T, R2, *,
    must_include_times, annotate_times, annotate, annotate_start, annotate_end,
    t_show_override, ood_t_show_override,
    max_points, include_last,
    show_pow2_anchors, anchor_skip_ends,
    point_alpha_start, point_alpha_end, point_alpha_power,
    anchor_alpha_floor,
    size_min, size_max,
):
    """
    Returns a dict with keys:
        t_show, t_base, aA, ann_times, must_times, t_show_ood, sizes_area
    """
    must_times = set(_normalize_time_list(must_include_times, T))

    ann_times = set(_normalize_time_list(annotate_times, T)) | set(must_times)
    if 19 in ann_times and 19 not in must_times:
        ann_times.discard(19)

    if (not annotate_start) and (0 in ann_times) and (0 not in must_times):
        ann_times.discard(0)
    if (not annotate_end) and ((T - 1) in ann_times) and ((T - 1) not in must_times):
        ann_times.discard(T - 1)

    if t_show_override is not None:
        t_show = np.unique(np.array(t_show_override, dtype=int))
        t_show = t_show[(t_show >= 0) & (t_show < T)]
        t_show.sort()
        t_base_full = np.array([], dtype=int)
    else:
        t_base_full = _pow2_time_indices(T, include_last=include_last)

        m = int(max(2, min(T, int(max_points))))
        t_dense = np.unique(np.round(np.linspace(0, T - 1, m)).astype(int))

        t_show = np.unique(np.concatenate([t_base_full, t_dense, np.array([0, T - 1], dtype=int)]))
        if must_times:
            t_show = np.unique(np.concatenate([t_show, np.array(sorted(must_times), dtype=int)]))
        if annotate and ann_times:
            t_show = np.unique(np.concatenate([t_show, np.array(sorted(ann_times), dtype=int)]))

        t_show.sort()

    # OOD time indices
    if ood_t_show_override is not None:
        _ood_ts = []
        for _t in ood_t_show_override:
            _t = int(_t)
            if _t < 0:
                _t = T + _t
            if 0 <= _t < T:
                _ood_ts.append(_t)
        t_show_ood = np.unique(np.array(_ood_ts, dtype=int))
        t_show_ood.sort()
        if t_show_ood.size == 0:
            t_show_ood = None
    else:
        t_show_ood = None

    # Pow2 anchor alphas
    if show_pow2_anchors and t_base_full.size > 0:
        t_base = t_base_full.copy()
        if anchor_skip_ends and T > 1:
            t_base = t_base[(t_base != 0) & (t_base != (T - 1))]

        if t_base.size > 0:
            uA = t_base.astype(float) / max(1.0, float(T - 1))
            aA = float(point_alpha_end) + (float(point_alpha_start) - float(point_alpha_end)) * (
                (1.0 - uA) ** float(point_alpha_power)
            )
            aA = np.clip(np.maximum(aA, float(anchor_alpha_floor)), 0.0, 1.0)
        else:
            aA = None
    else:
        t_base = np.array([], dtype=int)
        aA = None

    sizes_area = _r2_to_sizes_area(R2, size_min=size_min, size_max=size_max)

    return dict(
        t_show=t_show, t_base=t_base, aA=aA,
        ann_times=ann_times, must_times=must_times,
        t_show_ood=t_show_ood, sizes_area=sizes_area,
    )
