#!/usr/bin/env python3
"""Generate combined beta/alpha trajectory figure across E1, E2, E3.

Produces a single 1×3 subplot figure (Bayesian posterior alignment) instead
of three separate PNGs, eliminating redundant y-axis labels and inter-panel
gaps from LaTeX subfigure composition.

Usage (from project root):
    python scripts/fig_beta_alpha_traj_combined.py
    python scripts/fig_beta_alpha_traj_combined.py --coin-layer 5 --linear-layer 9 --latent-layer 5
    python scripts/fig_beta_alpha_traj_combined.py --no-simplex
"""

import argparse
import logging
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
from icl.utils.separability import plot_beta_alpha_on_ax
from icl.utils.unified_interface import get_exp_name

apply_paper_style()

_TASK_IDS = [0]
_PLOT_POSITIONS = list(range(60))
_ESTIM_POSITIONS = list(range(40, 60))
_PIDX = np.array(_PLOT_POSITIONS)


def _save_path(coin_layer: int, linear_layer: int, latent_layer: int, simplex: bool) -> Path:
    sim = "simplex" if simplex else "nosimplex"
    name = f"beta_alpha_traj_c{coin_layer}_l{linear_layer}_t{latent_layer}_{sim}.png"
    return PROJECT_ROOT / "paper_figs" / name


def _compute_coin(layer: int, simplex: bool, vocab_size: int = 6):
    from icl.coin.analysis import traj_averaging_projection_plot_coin
    exp_name = get_exp_name("coin", -1, vocab_size=vocab_size)
    out = traj_averaging_projection_plot_coin(
        exp_name,
        layer_index=layer,
        task_ids=_TASK_IDS,
        estimation_positions=_ESTIM_POSITIONS,
        plot_positions=_PLOT_POSITIONS,
        B=1024,
        show_legend=False,
        extraction_point="post_mlp",
        post_layernorm=False,
        beta_errbar="quantile",
        per_position_mean=True,
        project_beta_simplex=simplex,
        show=False,
    )
    plt.close("all")
    results = out["results_by_task"]
    k_major = results[_TASK_IDS[0]]["beta"].shape[2]
    return results, k_major, simplex


def _compute_linear(layer: int, simplex: bool):
    from icl.linear.analysis import traj_averaging_projection_plot
    exp_name = get_exp_name(
        "linear", -1,
        n_layer=16, total_steps=30_000,
        warmup_steps=15_000, batch_size=256,
        max_grad_norm=1.0,
    )
    out = traj_averaging_projection_plot(
        exp_name,
        layer_index=layer,
        task_ids=_TASK_IDS,
        estimation_positions=_ESTIM_POSITIONS,
        plot_positions=_PLOT_POSITIONS,
        B=1024,
        per_position_mean=True,
        project_beta_simplex=simplex,
        show_legend=False,
        beta_errbar="quantile",
        extraction_point="post_mlp",
        show=False,
    )
    plt.close("all")
    results = out["results_by_task"]
    k_major = results[_TASK_IDS[0]]["beta"].shape[2]
    return results, k_major, simplex


def _compute_latent(layer: int, simplex: bool):
    from icl.latent_markov.analysis import traj_averaging_projection_plot
    exp_name = get_exp_name("latent", -1, total_steps=30_000, warmup_steps=15_000)
    out = traj_averaging_projection_plot(
        exp_name,
        layer_index=layer,
        task_ids=_TASK_IDS,
        estimation_positions=_ESTIM_POSITIONS,
        plot_positions=_PLOT_POSITIONS,
        B=1024,
        per_position_mean=True,
        project_beta_simplex=simplex,
        show_legend=False,
        beta_errbar="quantile",
        extraction_point="post_mlp",
        show=False,
    )
    plt.close("all")
    results = out["results_by_task"]
    k_major = results[_TASK_IDS[0]]["beta"].shape[2]
    return results, k_major, simplex


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate combined beta/alpha trajectory figure (E1/E2/E3)."
    )
    p.add_argument("--coin-layer",   type=int, default=3, metavar="L",
                   help="Layer index for E1 Coins (default: 3)")
    p.add_argument("--linear-layer", type=int, default=9, metavar="L",
                   help="Layer index for E2 Linear (default: 9)")
    p.add_argument("--latent-layer", type=int, default=3, metavar="L",
                   help="Layer index for E3 Latent (default: 3)")
    p.add_argument("--simplex", action=argparse.BooleanOptionalAction, default=True,
                   help="Use simplex projection for beta coefficients (default: --simplex)")
    p.add_argument("--coin-vocab-size", type=int, default=6, metavar="V",
                   help="Vocabulary size for E1 Coins experiments (default: 16)")
    return p.parse_args()


def main():
    args = parse_args()
    save_path = _save_path(args.coin_layer, args.linear_layer, args.latent_layer, args.simplex)

    log.info(
        f"layers: coin={args.coin_layer}  linear={args.linear_layer}  latent={args.latent_layer}"
        f"  simplex={args.simplex}"
    )
    log.info(f"output: {save_path}")

    panels = [
        ("E1 (Coins)",  lambda: _compute_coin(args.coin_layer,   args.simplex, args.coin_vocab_size)),
        ("E2 (Linear)", lambda: _compute_linear(args.linear_layer, args.simplex)),
        ("E3 (Latent)", lambda: _compute_latent(args.latent_layer, args.simplex)),
    ]

    t_total = time.time()
    data = []
    for label, compute_fn in panels:
        log.info(f"[{label}] computing beta/alpha trajectories …")
        t0 = time.time()
        data.append(compute_fn())
        log.info(f"[{label}] done in {time.time() - t0:.1f}s")

    log.info("Composing figure …")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)

    y_lo_global, y_hi_global = np.inf, -np.inf
    for idx, (ax, (results, k_major, proj_simplex)) in enumerate(zip(axes, data)):
        y_lo, y_hi = plot_beta_alpha_on_ax(
            ax, results, k_major, _TASK_IDS, _PIDX,
            project_beta_simplex=proj_simplex,
            beta_errbar="quantile",
            show_ylabel=(idx == 0),
            show_xlabel=True,
            add_labels=(idx == len(axes) - 1),
        )
        y_lo_global = min(y_lo_global, y_lo)
        y_hi_global = max(y_hi_global, y_hi)
        if idx > 0:
            ax.tick_params(labelleft=False)

    # Shared y-limits with a small pad, clamped to a sensible range.
    # β and α are probability values in [0, 1]; the auto-detected range can be
    # dragged below 0 by wide quantile error bars at early positions.
    if y_hi_global > y_lo_global:
        pad = 0.05 * (y_hi_global - y_lo_global)
    else:
        pad = 0.05
    y_lo_final = max(y_lo_global - pad, -0.05)
    y_hi_final = min(y_hi_global + pad,  1.10)
    for ax in axes:
        ax.set_ylim(y_lo_final, y_hi_final)

    # Legend on the right side of the last panel.
    axes[-1].legend(
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        framealpha=0.8,
    )

    fig.tight_layout(w_pad=1.0)
    # Make room for the legend that sticks out to the right.
    fig.subplots_adjust(right=0.88)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    log.info(f"Saved → {save_path}  (total {time.time() - t_total:.1f}s)")
    plt.close(fig)


if __name__ == "__main__":
    main()
