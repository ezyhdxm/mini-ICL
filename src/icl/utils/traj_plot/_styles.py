"""OOD / minor colour jittering and linestyle assignment."""

import numpy as np
import matplotlib.colors as mcolors

from icl.utils.traj_plot._helpers import _jitter_color


def _compute_group_styles(
    idx_ood, idx_minor, *,
    major_colors, ood_base_color, minor_base_color,
    ood_linestyle_cycle, ood_style_seed,
    ood_hue_jitter, ood_sat_jitter, ood_val_jitter,
    ood_vary_linestyle,
):
    """
    Returns (major_rgbs, ood_style, minor_style, ood_linestyle_cycle).
    """
    if len(major_colors) != 3:
        raise ValueError("major_colors must have length 3.")
    major_rgbs = [mcolors.to_rgb(c) for c in major_colors]

    if ood_linestyle_cycle is None:
        ood_linestyle_cycle = [
            (0, (6, 2)),
            (0, (4, 2, 1, 2)),
            (0, (2.5, 1.2)),
            (0, (7, 2, 2, 2)),
            (0, (3, 1, 1, 1)),
        ]

    rng_ood = np.random.default_rng(int(ood_style_seed))
    ood_style = {}
    for j, k in enumerate(idx_ood.tolist()):
        rgb = _jitter_color(
            ood_base_color,
            rng_ood,
            hue_j=float(ood_hue_jitter),
            sat_j=float(ood_sat_jitter),
            val_j=float(ood_val_jitter),
        )
        ls = (
            ood_linestyle_cycle[j % len(ood_linestyle_cycle)]
            if (ood_vary_linestyle and len(ood_linestyle_cycle) > 0)
            else "--"
        )
        ood_style[int(k)] = (rgb, ls)

    rng_minor = np.random.default_rng(int(ood_style_seed) + 999)
    minor_style = {}
    for j, k in enumerate(idx_minor.tolist()):
        rgb = _jitter_color(minor_base_color, rng_minor, hue_j=0.02, sat_j=0.06, val_j=0.06)
        minor_style[int(k)] = rgb

    return major_rgbs, ood_style, minor_style, ood_linestyle_cycle
