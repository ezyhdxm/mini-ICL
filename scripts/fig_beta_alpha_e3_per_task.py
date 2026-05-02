#!/usr/bin/env python3
"""Per-task beta/alpha trajectory figure for E3 (Latent Markov chains).

Generates a 1x3 figure where each panel shows exactly one task's
(alpha_k dashed line + beta_k markers with errorbars) pair. This replaces
the dense 3-panel (E1/E2/E3) figure on the P3 alignment defense slide,
which overlays K=3 tasks per panel.

Compute is expensive (loads the trained Latent Markov model and runs
B=1024 forward passes), so the (beta, post) arrays are cached to an .npz
in ``paper_figs/`` and re-used on subsequent runs unless ``--recompute``
is passed.

Usage (from project root):
    python scripts/fig_beta_alpha_e3_per_task.py
    python scripts/fig_beta_alpha_e3_per_task.py --layer 3 --recompute
    python scripts/fig_beta_alpha_e3_per_task.py --no-simplex
"""

import argparse
import logging
import shutil
import time
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

from icl.utils.plot_config import apply_paper_style
from icl.utils.separability import plot_beta_alpha_one_col_on_ax
from icl.utils.unified_interface import get_exp_name

apply_paper_style()


_TASK_ID = 0  # 0-indexed: data is generated from task 1
_PLOT_POSITIONS = list(range(60))
_ESTIM_POSITIONS = list(range(40, 60))
_PIDX = np.array(_PLOT_POSITIONS)


def _panel_title(col: int, true_col: int) -> str:
    """Short panel title that makes the data-generating task explicit.

    Each panel is the posterior probability that the latent task equals ``k``;
    when ``col == true_col`` we annotate it as the data-generating task so the
    audience immediately sees why that curve goes to 1 while the others decay.
    """
    base = rf"$k\,{{=}}\,{col + 1}$"
    if col == true_col:
        return base + r"  (true task)"
    return base


def _cache_path(layer: int) -> Path:
    return PROJECT_ROOT / "paper_figs" / f"_beta_alpha_e3_l{layer}.npz"


def _save_path(layer: int, simplex: bool) -> Path:
    sim = "simplex" if simplex else "nosimplex"
    return PROJECT_ROOT / "paper_figs" / f"beta_alpha_traj_e3_l{layer}_per_task_{sim}.png"


def _compute_latent(layer: int):
    """Run the E3 (Latent Markov) trajectory computation."""
    from icl.latent_markov.analysis import traj_averaging_projection_plot
    exp_name = get_exp_name("latent", -1, total_steps=30_000, warmup_steps=15_000)
    out = traj_averaging_projection_plot(
        exp_name,
        layer_index=layer,
        task_ids=[_TASK_ID],
        estimation_positions=_ESTIM_POSITIONS,
        plot_positions=_PLOT_POSITIONS,
        B=1024,
        per_position_mean=True,
        # Caller decides whether to project; we always cache the raw beta and
        # apply the simplex projection inside the plot helper instead.
        project_beta_simplex=False,
        show_legend=False,
        beta_errbar="quantile",
        extraction_point="post_mlp",
        show=False,
    )
    plt.close("all")
    results = out["results_by_task"]
    k_major = results[_TASK_ID]["beta"].shape[2]
    return results, k_major


def _load_or_compute(layer: int, recompute: bool):
    cache = _cache_path(layer)
    if (not recompute) and cache.exists():
        log.info(f"loading cached arrays from {cache}")
        d = np.load(cache, allow_pickle=False)
        beta = d["beta"]
        post = d["post"]
        k_major = int(d["k_major"])
        results = {_TASK_ID: {"beta": beta, "post": post}}
        return results, k_major

    log.info("[E3] computing beta/alpha trajectories ...")
    t0 = time.time()
    results, k_major = _compute_latent(layer)
    log.info(f"[E3] done in {time.time() - t0:.1f}s")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        beta=results[_TASK_ID]["beta"],
        post=results[_TASK_ID]["post"],
        k_major=np.int64(k_major),
    )
    log.info(f"cached -> {cache}")
    return results, k_major


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-task beta/alpha trajectory figure for E3 (Latent Markov)."
    )
    p.add_argument("--layer", type=int, default=3, metavar="L",
                   help="Layer index for E3 latent Markov (default: 3, matches E3 panel of the existing combined figure)")
    p.add_argument("--simplex", action=argparse.BooleanOptionalAction, default=True,
                   help="Project the beta mean onto the probability simplex before plotting (default: --simplex)")
    p.add_argument("--recompute", action="store_true",
                   help="Ignore the cached .npz and re-run the model forward passes")
    p.add_argument("--copy-to", type=Path,
                   default=None,
                   help="If set, also copy the rendered PNG into this directory")
    return p.parse_args()


def main():
    args = parse_args()
    save_path = _save_path(args.layer, args.simplex)
    log.info(f"layer={args.layer}  simplex={args.simplex}  output={save_path}")

    results, k_major = _load_or_compute(args.layer, args.recompute)

    log.info(f"composing 1x{k_major} per-task figure ...")
    # Slightly narrower than the old 3-experiment figure since each panel only
    # carries one (alpha, beta) pair instead of K=3 overlapped.
    fig, axes = plt.subplots(1, k_major, figsize=(11, 2.6), sharey=True)

    y_lo_global, y_hi_global = np.inf, -np.inf
    for col, ax in enumerate(axes):
        # Pass a fresh copy of the results dict per call -- the in-place
        # simplex projection inside the helper would otherwise mutate shared
        # arrays across panels.
        beta = results[_TASK_ID]["beta"].copy()
        post = results[_TASK_ID]["post"].copy()
        local = {_TASK_ID: {"beta": beta, "post": post}}

        y_lo, y_hi = plot_beta_alpha_one_col_on_ax(
            ax,
            local,
            k_major=k_major,
            tid=_TASK_ID,
            col=col,
            pidx=_PIDX,
            project_beta_simplex=args.simplex,
            beta_errbar="quantile",
            show_ylabel=False,
            show_xlabel=True,
            add_labels=True,
        )
        y_lo_global = min(y_lo_global, y_lo)
        y_hi_global = max(y_hi_global, y_hi)
        ax.set_title(_panel_title(col, true_col=_TASK_ID), fontsize=12)
        ax.legend(fontsize=10, loc="center right", framealpha=0.85)

    if y_hi_global > y_lo_global:
        pad = 0.05 * (y_hi_global - y_lo_global)
    else:
        pad = 0.05
    y_lo_final = max(y_lo_global - pad, -0.05)
    y_hi_final = min(y_hi_global + pad, 1.10)
    for ax in axes:
        ax.set_ylim(y_lo_final, y_hi_final)

    fig.tight_layout(w_pad=1.2)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved -> {save_path}")

    if args.copy_to is not None:
        dest = args.copy_to / save_path.name
        try:
            shutil.copy2(save_path, dest)
            log.info(f"copied -> {dest}")
        except Exception as e:
            log.warning(f"failed to copy to {dest}: {e}")


if __name__ == "__main__":
    main()
