#!/usr/bin/env python3
"""Linear-regression simplex trajectory plot aligned with the E3 slide style.

This script adapts the presentation-oriented logic from
``fig_latent_traj_aligned.py`` to the E2 linear-regression setting.

Important differences from E3:
  - E2 trajectories live over regression example positions (`n_points`), not
    latent-token dynamics.
  - The linear notebook historically used late-window task-vector estimation,
    so this script exposes both `last` and `last10` style estimation windows.
  - We still mirror the slide aesthetics: multiple major-task bundles,
    reduced OOD clutter, explicit major/OOD display times, and marker sizes
    driven by projection R^2.

Usage (from project root):
    python scripts/fig_linear_traj_aligned.py
    python scripts/fig_linear_traj_aligned.py --k 10 --layer 10 --task-vec-window last10
    python scripts/fig_linear_traj_aligned.py --task-vec-window last --B 128 --n-group 64
"""

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from icl.linear.linear_path_utils import get_path_to_exp_dir, load_model_task_config
from icl.utils.linear_algebra_utils import estimate_lambda_with_r2
from icl.utils.separability import estimate_task_vectors_by_averaging
from icl.utils.traj_plot import project_with_r2_trajectories_group_colors_mpl
from icl.utils.unified_interface import _get_hiddens_at_real_positions, get_exp_name


def _real_llm_marker_legend_r2_levels(r2: np.ndarray) -> np.ndarray:
    """Use the same marker-legend quantile heuristic as the E3 script."""
    r2 = np.asarray(r2, dtype=float)
    r2_min = float(np.min(r2))
    r2_max = float(np.max(r2))
    q20, q50, q80 = np.quantile(r2, [0.2, 0.5, 0.8]).astype(float)
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
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--k", type=int, default=10, help="Experiment index for `get_exp_name('linear', k=...)`.")
    p.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Optional explicit experiment name. Overrides hash-based lookup.",
    )
    p.add_argument("--layer", type=int, default=10, help="Layer index for plotting. Use -1 for the last layer.")
    p.add_argument(
        "--task-vec-window",
        type=str,
        default="last10",
        choices=("last", "last10", "half", "all"),
        help="Task-vector estimation window. `last10` matches the generic linear default more closely.",
    )
    p.add_argument(
        "--task-vec-samples",
        type=int,
        default=128,
        help="When `task-vec-window=last`, average over at most this many sequences per major task.",
    )
    p.add_argument("--n-ood", type=int, default=12, help="Number of OOD tasks to display.")
    p.add_argument(
        "--ood-seed-offset",
        type=int,
        default=0,
        help="Integer offset added to the deterministic linear OOD sampling seed.",
    )
    p.add_argument(
        "--ood-far-oversample",
        type=int,
        default=1,
        help=(
            "If >1, sample (this * n_ood) candidate OOD weights and keep the n_ood with "
            "the smallest projected variance fraction onto the majority span. Useful for "
            "demos that want OOD trajectories to stay clearly outside the simplex (small R^2)."
        ),
    )
    p.add_argument("--B", type=int, default=128, help="Batch size per task for hidden-state extraction.")
    p.add_argument(
        "--n-group",
        type=int,
        default=64,
        help="Number of sequences averaged into one displayed trajectory bundle.",
    )
    p.add_argument(
        "--b-ood",
        type=int,
        default=1,
        help="OOD bundles per task (default 1 = less clutter; 0 = use all bundles).",
    )
    p.add_argument(
        "--major-display-times",
        type=int,
        nargs="*",
        default=None,
        metavar="T",
        help="Optional explicit displayed times for major trajectories. Default mirrors the E3 slide style.",
    )
    p.add_argument(
        "--ood-display-times",
        type=int,
        nargs="*",
        default=[0, 1, -1],
        metavar="T",
        help="Displayed times for OOD trajectories. Negative values follow Python indexing.",
    )
    p.add_argument(
        "--extraction-point",
        type=str,
        default="post_mlp",
        choices=("post_attn", "post_mlp"),
        help="Where to extract hidden states.",
    )
    p.add_argument("--n-layer", type=int, default=16, help="Linear model depth for experiment lookup.")
    p.add_argument("--total-steps", type=int, default=30_000, help="Training steps for experiment lookup.")
    p.add_argument("--warmup-steps", type=int, default=15_000, help="Warmup steps for experiment lookup.")
    p.add_argument("--batch-size", type=int, default=256, help="Training batch size for experiment lookup.")
    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Optional max grad norm for experiment lookup when needed.",
    )
    p.add_argument("--outdir", type=str, default=str(PROJECT_ROOT / "paper_figs"), help="Output directory.")
    p.add_argument("--stem", type=str, default=None, help="Output PNG basename without extension.")
    p.add_argument(
        "--dump-npz",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional path to dump the full kwargs dict for project_with_r2_trajectories_group_colors_mpl. "
             "Saved as an object pickle inside an .npz file under key 'plot_kw'. "
             "Allows downstream scripts (e.g. fig_simplex_buildup.py) to replay the exact projection.",
    )
    p.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional plot title. Default is empty to match slide-ready figures.",
    )
    p.add_argument(
        "--legend-r2",
        type=float,
        nargs="*",
        default=None,
        metavar="R2",
        help="Optional explicit marker-legend R^2 ticks.",
    )
    p.add_argument("--no-show", action="store_true", help="Do not call `plt.show()`.")
    return p.parse_args()


def _resolve_display_times(t_total: int, override: list[int] | None) -> list[int]:
    if override:
        out = []
        for t in override:
            idx = int(t)
            if idx < 0:
                idx = t_total + idx
            if 0 <= idx < t_total:
                out.append(idx)
        out = sorted(set(out))
        if out:
            return out
    if t_total < 5:
        return list(range(t_total))
    return [0, 1, 2, 3, t_total - 1]


def _resolve_ood_times(t_total: int, override: list[int] | None) -> list[int]:
    if not override:
        return [0, t_total - 1]
    out = []
    for t in override:
        idx = int(t)
        if idx < 0:
            idx = t_total + idx
        if 0 <= idx < t_total:
            out.append(idx)
    out = sorted(set(out))
    return out or [0, t_total - 1]


def _task_vecs_from_window(hiddens_major: torch.Tensor, window: str, task_vec_samples: int):
    """Return `(task_vecs, grand_mean, description)` for the selected estimation window."""
    _, t_total, b_total, _ = hiddens_major.shape
    if window == "last":
        n_tv = min(int(task_vec_samples), b_total)
        h_last = hiddens_major[:, t_total - 1, :n_tv, :].float()
        task_means = h_last.mean(dim=1)
        grand_mean = task_means.mean(dim=0)
        task_vecs = task_means - grand_mean.unsqueeze(0)
        desc = f"last position only (t={t_total - 1}, n_tv={n_tv})"
        return task_vecs.float(), grand_mean.float(), desc
    if window == "last10":
        positions = list(range(max(0, t_total - 10), t_total))
    elif window == "half":
        positions = list(range(t_total // 2, t_total))
    elif window == "all":
        positions = list(range(t_total))
    else:
        raise ValueError(f"Unknown task vector window: {window}")
    task_vecs, grand_mean = estimate_task_vectors_by_averaging(hiddens_major, positions)
    desc = f"{window} positions ({positions[0]}..{positions[-1]})"
    return task_vecs.float(), grand_mean.float(), desc


def _load_config_only(exp_name: str):
    exp_dir = get_path_to_exp_dir(exp_name)
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config for experiment `{exp_name}` at `{config_path}`.")
    with open(config_path, "r") as f:
        return json.load(f), exp_dir


def _maybe_find_matching_existing_exp(args) -> str | None:
    """Best-effort fallback when `get_exp_name(...)` no longer matches local hashes."""
    config_paths = sorted((PROJECT_ROOT / "results" / "linear").glob("*/config.json"))
    for config_path in config_paths:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        try:
            if int(cfg["model"]["n_layer"]) != int(args.n_layer):
                continue
            if int(cfg["training"]["total_steps"]) != int(args.total_steps):
                continue
            if int(cfg["training"]["warmup_steps"]) != int(args.warmup_steps):
                continue
            if int(cfg["task"]["batch_size"]) != int(args.batch_size):
                continue
            if args.max_grad_norm is not None:
                if float(cfg["training"].get("max_grad_norm")) != float(args.max_grad_norm):
                    continue
            if int(cfg["task"].get("n_points", 64)) != 64:
                continue
        except Exception:
            continue
        return config_path.parent.name
    return None


def main():
    args = parse_args()
    if args.exp_name is not None:
        exp_name = args.exp_name
        log.info("Experiment: %s  (explicit)", exp_name)
    else:
        exp_kwargs = dict(
            n_layer=args.n_layer,
            total_steps=args.total_steps,
            warmup_steps=args.warmup_steps,
            batch_size=args.batch_size,
        )
        if args.max_grad_norm is not None:
            exp_kwargs["max_grad_norm"] = args.max_grad_norm
        exp_name = get_exp_name("linear", args.k, **exp_kwargs)
        try:
            _cfg_dict, exp_dir = _load_config_only(exp_name)
            log.info("Experiment: %s  (k=%d)", exp_name, args.k)
        except FileNotFoundError:
            fallback_exp_name = _maybe_find_matching_existing_exp(args)
            if fallback_exp_name is None:
                raise
            exp_name = fallback_exp_name
            _cfg_dict, exp_dir = _load_config_only(exp_name)
            log.warning(
                "Hash lookup did not match a local config; falling back to existing run `%s`.",
                exp_name,
            )

    cfg_dict, exp_dir = _load_config_only(exp_name)
    n_layers = int(cfg_dict["model"]["n_layer"])
    layer_idx = int(args.layer)
    if layer_idx < 0:
        layer_idx = n_layers + layer_idx
    if not (0 <= layer_idx < n_layers):
        raise ValueError(f"Layer index out of range: got {args.layer} -> {layer_idx}, n_layers={n_layers}")

    log.info(
        "Extracting linear hidden states (layer=%d, B=%d, n_ood=%d, %s) ...",
        layer_idx,
        args.B,
        args.n_ood,
        args.extraction_point,
    )
    checkpoint_path = os.path.join(exp_dir, "checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Missing checkpoint for experiment `{exp_name}`. Expected `{checkpoint_path}`. "
            "This checkout appears to include config/log files but not the trained weights."
        )
    hiddens, _ = _get_hiddens_at_real_positions(
        task_name="linear",
        exp_name=exp_name,
        n_minor=0,
        n_ood=args.n_ood,
        B=args.B,
        extraction_point=args.extraction_point,
        linear_ood_seed_offset=args.ood_seed_offset,
        ood_far_oversample=args.ood_far_oversample,
    )

    hiddens_layer = hiddens[layer_idx].to(torch.float32)  # (K, T, B, D)
    k_total, t_total, b_total, d_model = hiddens_layer.shape
    k_major = 3
    log.info("K=%d  T=%d  B=%d  D=%d", k_total, t_total, b_total, d_model)

    hiddens_major = hiddens_layer[:k_major]
    task_vecs, grand_mean, tv_desc = _task_vecs_from_window(
        hiddens_major, args.task_vec_window, args.task_vec_samples
    )
    final_task_vecs = task_vecs.float()
    log.info("Task-vector estimation: %s", tv_desc)

    h_centered = hiddens_layer - grand_mean.float()

    n_groups = b_total // args.n_group
    n_used = n_groups * args.n_group
    if n_groups < 1:
        raise ValueError(f"B={b_total} must be >= n_group={args.n_group}")
    if n_used != b_total:
        log.warning(
            "B=%d is not divisible by n_group=%d; using first n_used=%d samples",
            b_total,
            args.n_group,
            n_used,
        )
    log.info("n_group=%d  n_groups=%d  n_used=%d", args.n_group, n_groups, n_used)

    display_t = _resolve_display_times(t_total, args.major_display_times)
    ood_display_t = _resolve_ood_times(t_total, args.ood_display_times)
    log.info("Major display_t: %s", display_t)
    log.info("OOD display_t: %s", ood_display_t)

    log.info("Computing R^2 at displayed times on raw samples, then averaging within groups ...")
    h_eval = h_centered[:, :, :n_used, :]
    r2_list = []
    for t_idx in display_t:
        h_t = h_eval[:, t_idx, :, :]
        _, r2_t, _, _ = estimate_lambda_with_r2(final_task_vecs, h_t, is_zero_mean=True)
        r2_list.append(r2_t)
    r2_disp = np.stack(r2_list, axis=-1)  # (K, n_used, T_disp)
    r2_grp = r2_disp.reshape(k_total, n_groups, args.n_group, len(display_t)).mean(axis=2)

    h_grp = h_eval.reshape(k_total, t_total, n_groups, args.n_group, d_model).mean(dim=3)
    h_disp = h_grp[:, display_t, :, :]

    b_ood = n_groups if args.b_ood == 0 else max(1, min(int(args.b_ood), n_groups))
    log.info(
        "b_major=%d  b_ood=%d  (majors: %d trajs, OOD: %d trajs)",
        n_groups,
        b_ood,
        3 * n_groups,
        args.n_ood * b_ood,
    )

    levels_preview = _real_llm_marker_legend_r2_levels(r2_grp)
    log.info("Marker legend R^2 levels: %s", [float(x) for x in levels_preview])
    log.info(
        "r2_grp stats: min=%.4f max=%.4f mean=%.4f",
        float(np.min(r2_grp)),
        float(np.max(r2_grp)),
        float(np.mean(r2_grp)),
    )

    log.info("Plotting ...")
    plot_kw = dict(
        task_vecs_over_all_time=h_disp.cpu().numpy(),
        final_task_vecs=final_task_vecs.cpu().numpy(),
        r2_scores=r2_grp,
        n_ood=args.n_ood,
        n_minor=0,
        use_mean=False,
        b_major=n_groups,
        b_ood=b_ood,
        t_show_override=list(range(len(display_t))),
        ood_t_show_override=[display_t.index(t) for t in ood_display_t if t in display_t],
        ood_line_alpha_factor=0.38,
        annotate=False,
        show_pow2_anchors=False,
        figsize=(7, 3.0),
        title=args.title,
        major_legend_prefix="Task ",
        ood_legend_label="OOD",
        major_colors=("#1a5276", "#2471a3", "#5dade2"),
        ood_base_color="#e74c3c",
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
    if args.legend_r2:
        plot_kw["legend_r2_values"] = list(args.legend_r2)

    if args.dump_npz is not None:
        dump_path = args.dump_npz
        os.makedirs(os.path.dirname(os.path.abspath(dump_path)) or ".", exist_ok=True)
        np.savez(dump_path, plot_kw=np.array(plot_kw, dtype=object))
        log.info("Dumped plot kwargs -> %s", dump_path)

    fig, ax, *_ = project_with_r2_trajectories_group_colors_mpl(**plot_kw)
    ax.set_aspect(0.85, adjustable="datalim")

    os.makedirs(args.outdir, exist_ok=True)
    if args.stem:
        stem = args.stem
    else:
        stem = f"linear_traj_aligned_k{args.k}_l{layer_idx}_{args.task_vec_window}"
    out_path = os.path.join(args.outdir, f"{stem}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    log.info("Saved -> %s", out_path)

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
