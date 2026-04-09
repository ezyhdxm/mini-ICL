#!/usr/bin/env python3
"""Latent trajectory projection plot aligned with the RealLLM visual style.

Replicates the data-loading and task-vector estimation logic of
``traj_posterior_projection_plot`` but aligns the visual presentation with the
RealLLM trajectory plots:

  - Multiple trajectories per task (pre-averaged groups of N_GROUP samples)
    instead of a single mean trajectory.
  - Matching marker sizes, line widths, alphas, and colour palette.
  - Same display times as RealLLM: ``display_t = [0, 1, 2, 3, T-1]`` (5 points
    along the sequence); OOD polylines use ``[0, 1, -1]`` (3 of those 5).
  - R² computed per time on raw batch samples, then averaged within groups
    (identical to the RealLLM notebook loop).

**OOD vs major ink (side-by-side figures):** RealLLM still draws ``n_groups``
bundles for *both* ID and OOD (12 OOD tasks × 3 = 36 polylines vs 3 × 3 = 9
for majors).  By default we use ``b_ood=1`` so each OOD task shows one bundle
(12 polylines), which matches how dense the centre looks next to LLM plots.
Pass ``--b-ood 0`` to use ``n_groups`` for OOD like the RealLLM notebook.

**Defaults (latent-specific):** Task vectors use the **last sequence position
only**, averaged over ``--task-vec-samples`` batch sequences (default **128**).
Trajectory bundles average ``--n-group`` sequences each (default **64**); with
default ``B=128`` you get **2** bundles per task.  For pooled windows use
``--task-vec-window late30|half|all`` instead of ``last``.

**Higher projection R²:** Try ``--layer -1`` or ``--boost-r2`` (last layer).

**More samples:** ``--high-sample`` sets ``B=240`` and ``n_group=80`` (legacy).

**Marker-size legend:** If ``legend_r2_values`` is omitted,
``_draw_legend_cosmetics_refs`` uses quantiles at 0.2 / 0.5 / 0.8 (rounded)
plus a **final tick at the global max** ``max(R²)`` (not the 95th percentile).
If fewer than three levels remain, fall back to ``r2_min``, midpoint, ``r2_max``.
Override with ``--legend-r2 ...`` if needed.

The ``traj_posterior_projection_plot`` function itself is NOT modified.

Usage (from project root):
    python scripts/fig_latent_traj_aligned.py
    python scripts/fig_latent_traj_aligned.py --layer 4 --B 128 --n-group 64 --task-vec-samples 128
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

import icl.utils.notebook_utils as nu
from icl.utils.unified_interface import get_exp_name, _get_hiddens_at_real_positions
from icl.utils.separability import estimate_task_vectors_by_averaging
from icl.utils.linear_algebra_utils import estimate_lambda_with_r2
from icl.utils.traj_plot import project_with_r2_trajectories_group_colors_mpl


def _real_llm_marker_legend_r2_levels(R2: np.ndarray) -> np.ndarray:
    """Match ``_draw_legend_cosmetics_refs`` when ``legend_r2_values`` is None."""
    R2 = np.asarray(R2, dtype=float)
    r2_min = float(np.min(R2))
    r2_max = float(np.max(R2))
    q20, q50, q80 = np.quantile(R2, [0.2, 0.5, 0.8]).astype(float)
    levels = []
    for q in (q20, q50, q80):
        rq = round(float(q), 2)
        if not levels or rq > levels[-1] + 1e-9:
            levels.append(rq)
    if not levels or r2_max > levels[-1] + 1e-9:
        levels.append(float(r2_max))
    else:
        levels[-1] = float(r2_max)
    r2_levels = np.asarray(levels, dtype=float)
    if r2_levels.size < 3:
        r2_levels = np.unique(
            np.round(np.array([r2_min, 0.5 * (r2_min + r2_max), r2_max]), 2)
        )
    return r2_levels


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=10,
                   help="Experiment index passed to get_exp_name('latent', k=...)")
    p.add_argument(
        "--layer", type=int, default=4,
        help="Layer index (use -1 for last layer, matching traj_posterior default)",
    )
    p.add_argument(
        "--task-vec-window", type=str, default="last",
        choices=("last", "late30", "half", "all"),
        help="Task-vector pooling: last position only (default), or late30/half/all via averaging helper",
    )
    p.add_argument(
        "--task-vec-samples", type=int, default=128,
        help="When task_vec_window=last: number of batch sequences to average per major task (cap=min with B)",
    )
    p.add_argument(
        "--boost-r2",
        action="store_true",
        help="Shorthand: --layer -1 (last transformer layer)",
    )
    p.add_argument("--n-ood", type=int, default=12,
                   help="Number of OOD tasks (12 matches RealLLM)")
    p.add_argument("--B", type=int, default=128,
                   help="Batch size per task (must be ≥ --task-vec-samples when window=last; divisible by --n-group)")
    p.add_argument("--n-group", type=int, default=64,
                   help="Sequences averaged per trajectory bundle (default 64 → 2 bundles when B=128)")
    p.add_argument(
        "--high-sample",
        action="store_true",
        help="Use B=240 and n_group=80 (still 3 bundles, but 80 sequences averaged per bundle; "
             "task vectors also use more batch at late positions)",
    )
    p.add_argument(
        "--b-ood", type=int, default=1,
        help="OOD bundles per task (default 1 = less centre clutter; 0 = use n_groups, RealLLM notebook)",
    )
    p.add_argument("--extraction-point", type=str, default="post_mlp",
                   choices=["post_attn", "post_mlp"],
                   help="Where to extract hidden states")
    p.add_argument("--outdir", type=str, default=str(PROJECT_ROOT / "paper_figs"),
                   help="Output directory for the figure")
    p.add_argument(
        "--stem", type=str, default=None,
        help="Output PNG basename without .png (overrides default/boost_r2/high_n naming)",
    )
    p.add_argument(
        "--legend-r2", type=float, nargs="*", default=None,
        metavar="R2",
        help="Optional explicit marker-legend R² ticks; default: RealLLM quantile rule in traj_plot._draw",
    )
    p.add_argument("--no-show", action="store_true",
                   help="Don't call plt.show()")
    return p.parse_args()


def main():
    args = parse_args()
    if args.high_sample:
        log.info("--high-sample: B=240, n_group=80 (3 trajectory bundles, 80-sample group means)")
        args.B = 240
        args.n_group = 80
    if args.boost_r2:
        args.layer = -1
        log.info("--boost-r2: using last transformer layer")

    exp_name = get_exp_name("latent", k=args.k)
    log.info("Experiment: %s  (k=%d)", exp_name, args.k)

    # ── Load experiment config ───────────────────────────────────────────
    _, _, config, *_ = nu.load_everything("latent", exp_name)
    n_layers = int(config.model.num_layers)
    layer_idx = int(args.layer)
    if layer_idx < 0:
        layer_idx = n_layers + layer_idx
    if not (0 <= layer_idx < n_layers):
        raise ValueError(f"layer index out of range: got {args.layer} → {layer_idx}, n_layers={n_layers}")

    # ── Extract hidden states ────────────────────────────────────────────
    log.info("Extracting hidden states (layer=%d, B=%d, n_ood=%d, %s) ...",
             layer_idx, args.B, args.n_ood, args.extraction_point)
    hiddens, _ = _get_hiddens_at_real_positions(
        task_name="latent", exp_name=exp_name,
        n_minor=0, n_ood=args.n_ood, B=args.B,
        extraction_point=args.extraction_point,
    )

    hiddens_layer = hiddens[layer_idx].to(torch.float32)  # (K, T, B, D)
    K, T, B_actual, D = hiddens_layer.shape
    k_major = 3
    log.info("K=%d  T=%d  B=%d  D=%d", K, T, B_actual, D)

    # ── Estimate task vectors from major tasks by averaging ──────────────
    hiddens_major = hiddens_layer[:k_major]
    if args.task_vec_window == "last":
        n_tv = min(int(args.task_vec_samples), B_actual)
        if n_tv < int(args.task_vec_samples):
            log.warning(
                "task_vec_samples=%d > B=%d; using n_tv=%d for last-position task means",
                args.task_vec_samples, B_actual, n_tv,
            )
        h_last = hiddens_major[:, T - 1, :n_tv, :].float()  # (3, n_tv, D)
        task_means = h_last.mean(dim=1)  # (3, D)
        grand_mean = task_means.mean(dim=0)
        task_vecs = task_means - grand_mean.unsqueeze(0)
        log.info(
            "task_vec_window=last: position T-1=%d, mean over %d sequences per major task",
            T - 1, n_tv,
        )
    elif args.task_vec_window == "late30":
        estimation_positions = list(range(max(0, T - 30), T))
        log.info("task_vec_window=late30 → %d positions (%d..%d)",
                 len(estimation_positions), estimation_positions[0], estimation_positions[-1])
        task_vecs, grand_mean = estimate_task_vectors_by_averaging(
            hiddens_major, estimation_positions,
        )
    elif args.task_vec_window == "half":
        estimation_positions = list(range(T // 2, T))
        log.info("task_vec_window=half → %d positions", len(estimation_positions))
        task_vecs, grand_mean = estimate_task_vectors_by_averaging(
            hiddens_major, estimation_positions,
        )
    else:
        estimation_positions = list(range(T))
        log.info("task_vec_window=all → %d positions", len(estimation_positions))
        task_vecs, grand_mean = estimate_task_vectors_by_averaging(
            hiddens_major, estimation_positions,
        )
    final_task_vecs = task_vecs.float()  # (k_major, D), centred

    # ── Centre (RealLLM-style) ───────────────────────────────────────────
    h_centered = hiddens_layer - grand_mean.float()  # (K, T, B, D)

    n_groups = B_actual // args.n_group
    n_used = n_groups * args.n_group
    if n_used != B_actual:
        log.warning(
            "B=%d is not divisible by N_GROUP=%d; using first n_used=%d samples",
            B_actual, args.n_group, n_used,
        )
    log.info("N_GROUP=%d  n_groups=%d  n_used=%d", args.n_group, n_groups, n_used)

    # Display times: exact RealLLM choice (ID: four early + final; OOD uses 3 via override)
    if T < 5:
        display_t = list(range(T))
    else:
        display_t = [0, 1, 2, 3, T - 1]
    T_disp = len(display_t)
    log.info("display_t (RealLLM-aligned): %s  →  T_disp=%d", display_t, T_disp)

    # ── R² per display time on raw batch, then group-average (RealLLM notebook) ─
    log.info("Computing R² (per-time on n_used samples, then group mean) ...")
    h_eval = h_centered[:, :, :n_used, :]  # (K, T, n_used, D)
    r2_list = []
    for t_idx in display_t:
        h_t = h_eval[:, t_idx, :, :]  # (K, n_used, D)
        _, r2_t, _, _ = estimate_lambda_with_r2(
            final_task_vecs, h_t, is_zero_mean=True,
        )
        r2_list.append(r2_t)
    r2_disp = np.stack(r2_list, axis=-1)  # (K, n_used, T_disp)
    r2_grp = (
        r2_disp.reshape(K, n_groups, args.n_group, T_disp)
        .mean(axis=2)
    )  # (K, n_groups, T_disp)

    # ── Group-averaged trajectories at display times only ─────────────────
    h_grp = (
        h_eval.reshape(K, T, n_groups, args.n_group, D)
        .mean(dim=3)
    )  # (K, T, n_groups, D)
    h_disp = h_grp[:, display_t, :, :]  # (K, T_disp, n_groups, D)

    b_ood = n_groups if args.b_ood == 0 else max(1, min(int(args.b_ood), n_groups))
    log.info("b_major=%d  b_ood=%d  (majors: %d trajs, OOD: %d trajs)",
             n_groups, b_ood, 3 * n_groups, args.n_ood * b_ood)

    levels_preview = _real_llm_marker_legend_r2_levels(r2_grp)
    log.info(
        "Marker legend R² (RealLLM / traj_plot quantiles): %s",
        [float(x) for x in levels_preview],
    )
    log.info(
        "r2_grp stats: min=%.4f max=%.4f mean=%.4f",
        float(np.min(r2_grp)), float(np.max(r2_grp)), float(np.mean(r2_grp)),
    )

    # ── Plot with RealLLM-aligned visual parameters ──────────────────────
    log.info("Plotting ...")
    plot_kw = dict(
        task_vecs_over_all_time=h_disp.numpy(),          # (K, T_disp, n_groups, D)
        final_task_vecs=final_task_vecs.numpy(),          # (k_major, D)
        r2_scores=r2_grp,                                # (K, n_groups, T_disp)
        n_ood=args.n_ood,
        n_minor=0,
        use_mean=False,
        b_major=n_groups,
        b_ood=b_ood,
        ood_t_show_override=[0, 1, -1],
        ood_line_alpha_factor=0.38,
        annotate=False,
        show_pow2_anchors=False,
        figsize=(7, 3.0),
        title='',
        ood_legend_label='OOD',
        major_colors=('#1a5276', '#2471a3', '#5dade2'),
        ood_base_color='#e74c3c',
        size_min=5,
        size_max=20,
        mid_size_factor=0.45,
        jitter_scale=0.013,
        line_alpha_end=0.75,
        point_alpha_start=0.90,
        point_alpha_end=0.50,
        major_line_width=2.5,
        line_width=1.8,
        end_marker_alpha=1.0,
        base_fontsize=13,
        ood_style_seed=0,
    )
    if args.legend_r2 is not None and len(args.legend_r2) > 0:
        plot_kw["legend_r2_values"] = list(args.legend_r2)

    fig, ax, *_ = project_with_r2_trajectories_group_colors_mpl(**plot_kw)
    ax.set_aspect(0.85, adjustable='datalim')

    os.makedirs(args.outdir, exist_ok=True)
    if args.stem:
        stem = args.stem
    elif args.boost_r2:
        stem = "latent_traj_aligned_boost_r2"
    elif args.high_sample:
        stem = "latent_traj_aligned_high_n"
    else:
        stem = "latent_traj_aligned"
    fname = os.path.join(args.outdir, f"{stem}.png")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    log.info("Saved → %s", fname)

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
