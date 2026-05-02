#!/usr/bin/env python3
"""Render a step-by-step buildup of the E2 simplex-trajectory plot.

Inputs: a single .npz file produced by

    python scripts/fig_linear_traj_aligned.py [...] --dump-npz <PATH>

which contains the full kwargs dict that drives
``project_with_r2_trajectories_group_colors_mpl``.

Outputs: 7 PNGs whose framing (xlim/ylim) and projection are identical to the
"full" stage, so they can be swapped via Beamer ``\\only`` overlays without any
visual jump.

Usage (from project root):

    python scripts/fig_simplex_buildup.py \\
        --npz path/to/_buildup_e2_inputs.npz \\
        --outdir path/to/output \\
        --prefix e2_stage
"""

import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from icl.utils.traj_plot import project_with_r2_trajectories_group_colors_mpl


# Each stage decides:
#   - which majority trajectories to draw (None=all, []=none, list of internal k)
#   - which OOD trajectories to draw (same convention)
#   - whether to override t_show / ood_t_show
#   - whether to suppress the always-on end marker (include_last)
STAGES = [
    dict(name="01_full",
         render_indices_major=None, render_indices_ood=None,
         t_show=None, ood_t_show=None, include_last=True),
    dict(name="02_vertices",
         render_indices_major=[],   render_indices_ood=[],
         t_show=None, ood_t_show=None, include_last=True),
    dict(name="03_proj_start",
         render_indices_major="single", render_indices_ood=[],
         t_show=[0], ood_t_show=None, include_last=False,
         # Add a manual highlight scatter on top of the (tiny) natural marker
         # so the audience can see exactly where the chosen trajectory starts,
         # without distorting the marker-size legend gradient.
         highlight_start=True),
    dict(name="04_proj_full",
         render_indices_major="single", render_indices_ood=[],
         t_show=None, ood_t_show=None, include_last=True),
    dict(name="05_majority",
         render_indices_major=None, render_indices_ood=[],
         t_show=None, ood_t_show=None, include_last=True),
    dict(name="06_ood_initial",
         render_indices_major=[],   render_indices_ood=None,
         t_show=None, ood_t_show=[0], include_last=False),
    dict(name="07_ood_full",
         render_indices_major=[],   render_indices_ood=None,
         t_show=None, ood_t_show=None, include_last=True),
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--npz", required=True,
        help="Path to the .npz produced by fig_linear_traj_aligned.py --dump-npz.",
    )
    p.add_argument(
        "--outdir", default=str(PROJECT_ROOT / "paper_figs" / "buildup"),
        help="Directory where the per-stage PNGs are written.",
    )
    p.add_argument(
        "--prefix", default="e2_stage",
        help="Filename prefix; stages save as <prefix>_<NN>_<name>.png.",
    )
    p.add_argument(
        "--dpi", type=int, default=300,
        help="Output dpi (matches the original plot).",
    )
    return p.parse_args()


def _load_plot_kw(npz_path: str) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    if "plot_kw" not in data.files:
        raise KeyError(f"npz {npz_path} does not contain key 'plot_kw'; got {data.files}")
    plot_kw = data["plot_kw"].item()
    if not isinstance(plot_kw, dict):
        raise TypeError(f"plot_kw in {npz_path} is not a dict (got {type(plot_kw)})")
    return dict(plot_kw)


def _add_start_highlight(ax, kw: dict, *, single_major_k: int) -> None:
    """Scatter a large visible marker at the t=0 position of the chosen
    majority trajectory.

    We locate the existing per-trajectory scatter by its gid (set by the
    plotter) and read its 2D coordinates; this avoids replicating the
    projection logic and naturally tracks any centring/rotation alignment
    applied inside ``project_with_r2_trajectories_group_colors_mpl``.
    """
    gid_prefix = kw.get("gid_prefix", "proj_r2_traj_clear")
    target_gid = f"{gid_prefix}:major_pts:{int(single_major_k)}"

    for artist in list(ax.collections):
        get_gid = getattr(artist, "get_gid", None)
        if get_gid is None or get_gid() != target_gid:
            continue
        offsets = artist.get_offsets()
        if len(offsets) == 0:
            continue
        # First offset == first displayed time index; with t_show=[0] this is
        # exactly the trajectory's start position.
        x, y = float(offsets[0][0]), float(offsets[0][1])

        # Reuse the existing point's RGB (so the highlight matches the task's
        # blue), but force full opacity and a noticeably larger marker.
        try:
            face = artist.get_facecolors()[0]
            rgb = (float(face[0]), float(face[1]), float(face[2]))
        except Exception:
            rgb = (0.10, 0.30, 0.55)

        ax.scatter(
            [x], [y],
            s=110, marker="o",
            facecolor=rgb, edgecolor="white", linewidth=1.2,
            alpha=0.95, zorder=5.0,
        )
        return


def _resolve_single_major_index(plot_kw: dict) -> int:
    """First internal index of the first majority task.

    With ``use_mean=False`` and the (K,T,B,D) input expansion in
    ``_prepare_inputs``, internal majority indices are
    ``[0, 1, ..., 3 * b_major - 1]``, where the first ``b_major`` indices all
    belong to majority task 0. We pick index 0 — the first group of task 0.
    """
    return 0


# Visual overrides applied to EVERY stage so the OOD trajectories read clearly
# in the buildup. The dump from fig_linear_traj_aligned.py uses
# ood_line_alpha_factor=0.38 and a 5-cycle dashed pattern which is appropriate
# for the dense standalone figure but too faint / noisy in the buildup, where
# the OOD trajectories are the entire focus of overlays 8-9.
GLOBAL_KW_OVERRIDES = dict(
    ood_line_alpha_factor=0.9,        # was 0.38
    ood_vary_linestyle=False,         # disable 5-pattern dash cycle ...
    ood_linestyle_cycle=[(0, (4, 1.5))],  # ... use one consistent short-dash for all OOD
    # Keep the dump's native (7, 3.0) figsize so the E2 panel matches the
    # Qwen panel's aspect ratio exactly -- otherwise the two images render at
    # mismatched heights (or, with padding, the Qwen triangle ends up visibly
    # smaller than the E2 triangle).
)


def _save_stage(stage_idx: int, stage: dict, plot_kw: dict, *,
                xlim, ylim, outdir: str, prefix: str, dpi: int,
                single_major_k: int):
    name = stage["name"]
    log.info("Stage %02d (%s): rendering ...", stage_idx, name)

    kw = dict(plot_kw)
    kw.update(GLOBAL_KW_OVERRIDES)

    # --- selective rendering ---
    rim = stage["render_indices_major"]
    rio = stage["render_indices_ood"]
    if rim == "single":
        kw["render_indices_major"] = [int(single_major_k)]
    else:
        kw["render_indices_major"] = rim
    kw["render_indices_ood"] = rio

    # --- per-stage time overrides (preserve original if not specified) ---
    if stage["t_show"] is not None:
        kw["t_show_override"] = list(stage["t_show"])
    if stage["ood_t_show"] is not None:
        kw["ood_t_show_override"] = list(stage["ood_t_show"])

    # --- end-marker control (suppress for "start-only" stages) ---
    kw["include_last"] = bool(stage.get("include_last", True))

    # --- per-stage extra overrides (e.g. boost marker size for stage 3) ---
    extra = stage.get("extra_kw")
    if extra:
        kw.update(extra)

    # Render
    fig, ax, *_ = project_with_r2_trajectories_group_colors_mpl(**kw)
    # Match the aspect tweak applied in fig_linear_traj_aligned.py.
    ax.set_aspect(0.85, adjustable="datalim")

    # --- optional "start position" highlight (stage 3) ---
    # Drawn via a separate scatter so we don't distort the size legend.
    if stage.get("highlight_start", False):
        _add_start_highlight(ax, kw, single_major_k=single_major_k)

    # Lock framing so every stage shares the same projection footprint.
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    out_path = os.path.join(outdir, f"{prefix}_{name}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    log.info("Saved -> %s", out_path)

    # Capture limits from the FIRST stage only (the full plot).
    captured_xlim = ax.get_xlim()
    captured_ylim = ax.get_ylim()

    plt.close(fig)
    return captured_xlim, captured_ylim


def main():
    args = parse_args()

    plot_kw = _load_plot_kw(args.npz)
    log.info("Loaded plot_kw with %d keys from %s", len(plot_kw), args.npz)

    single_major_k = _resolve_single_major_index(plot_kw)
    log.info("Single-trajectory internal index: %d", single_major_k)

    os.makedirs(args.outdir, exist_ok=True)

    xlim = ylim = None
    for i, stage in enumerate(STAGES, start=1):
        # The FIRST stage establishes the canonical (xlim, ylim) — pass None so
        # matplotlib computes them from the full data; afterward, every stage
        # forces those exact limits.
        cap_xlim, cap_ylim = _save_stage(
            i, stage, plot_kw,
            xlim=xlim, ylim=ylim,
            outdir=args.outdir, prefix=args.prefix, dpi=args.dpi,
            single_major_k=single_major_k,
        )
        if xlim is None and ylim is None:
            xlim, ylim = cap_xlim, cap_ylim
            log.info("Locked framing: xlim=%s  ylim=%s",
                     tuple(round(float(v), 4) for v in xlim),
                     tuple(round(float(v), 4) for v in ylim))


if __name__ == "__main__":
    main()
