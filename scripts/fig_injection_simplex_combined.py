#!/usr/bin/env python3
"""Generate combined injection-simplex figure across E1, E2, E3.

Produces a single 6-panel figure (Unsteered / Steered for each experiment)
with a single dual-sided colorbar: KL divergence ticks on the left,
RMSE ticks on the right (sharing the same colormap at a 10× ratio).

Usage (from project root):
    python scripts/fig_injection_simplex_combined.py
"""

import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_PATH      = PROJECT_ROOT / "paper_figs" / "injection_simplex_combined.png"
SAVE_PATH_MODE = PROJECT_ROOT / "paper_figs" / "mode_task_comparison_combined.png"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from icl.utils.plot_config import (
    apply_paper_style,
    render_injection_simplex_combined,
    render_mode_task_comparison_combined,
)
from icl.utils.unified_interface import get_exp_name

apply_paper_style()


def _compute_coin(vocab_size: int = 16, layer: int = 3):
    from icl.coin.analysis.interventions import intervene_averaging_injection_coin
    exp_name = get_exp_name("coin", -1, vocab_size=vocab_size)
    return intervene_averaging_injection_coin(
        exp_name=exp_name,
        layer=layer,
        verbose=False,
        estimation_positions=range(25, 40),
        eval_positions=range(25, 40),
        estimation_B=1024,
        B=10_000,
        per_position_mean=True,
        use_per_position_vecs=True,
        extraction_point="post_mlp",
        post_layernorm=False,
        color_vmin=0,
        color_vmax=0.7,
        show=False,
    )


def _compute_linear():
    from icl.linear.analysis.interventions import intervene_averaging_injection
    exp_name = get_exp_name(
        "linear", -1,
        n_layer=16,
        total_steps=30_000,
        warmup_steps=15_000,
        batch_size=256,
        max_grad_norm=1.0,
    )
    return intervene_averaging_injection(
        exp_name=exp_name,
        layer=10,
        per_position_mean=True,
        estimation_positions=range(25, 40),
        eval_positions=range(40),
        verbose=False,
        estimation_B=1024,
        B=10_000,
        extraction_point="post_mlp",
        color_vmin=0,
        color_vmax=7,
        show=False,
    )


def _compute_latent():
    from icl.latent_markov.analysis.interventions import intervene_averaging_injection
    exp_name = get_exp_name(
        "latent", k=-1,
        total_steps=30_000,
        warmup_steps=15_000,
    )
    return intervene_averaging_injection(
        exp_name=exp_name,
        layer=3,
        verbose=False,
        estimation_positions=range(25, 40),
        eval_positions=range(40),
        estimation_B=1024,
        per_position_mean=True,
        B=10_000,
        color_vmin=0,
        color_vmax=0.7,
        extraction_point="post_mlp",
        show=False,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Generate combined injection-simplex figure (E1/E2/E3).")
    p.add_argument("--coin-vocab-size", type=int, default=16, metavar="V",
                   help="Vocabulary size for E1 Coins experiments (default: 16)")
    p.add_argument("--coin-layer", type=int, default=3,
                   help="Layer index for E1 Coins steering (default: 3)")
    return p.parse_args()


def main():
    args = parse_args()
    panels = [
        ("E1 (Coins)",  lambda _v=args.coin_vocab_size, _l=args.coin_layer: _compute_coin(_v, _l)),
        ("E2 (Linear)", _compute_linear),
        ("E3 (Latent)", _compute_latent),
    ]

    t_total = time.time()
    results = []
    for label, compute_fn in panels:
        log.info(f"[{label}] computing injection simplex …")
        t0 = time.time()
        out = compute_fn()
        plt.close("all")
        results.append(out)
        log.info(f"[{label}] done in {time.time() - t0:.1f}s")

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    log.info("Composing injection-simplex figure …")
    fig_inj = render_injection_simplex_combined(
        coin_result=results[0],
        linear_result=results[1],
        latent_result=results[2],
        kl_vmin=0.0,
        kl_vmax=0.7,
        rmse_vmin=0.0,
        rmse_vmax=7.0,
        figsize=(13, 3.8),
    )
    fig_inj.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
    log.info(f"Saved → {SAVE_PATH}")
    plt.close(fig_inj)

    log.info("Composing mode-task comparison figure …")
    fig_mode = render_mode_task_comparison_combined(
        coin_result=results[0],
        linear_result=results[1],
        latent_result=results[2],
        kl_vmin=0.0,
        kl_vmax=0.7,
        rmse_vmin=0.0,
        rmse_vmax=7.0,
        figsize=(13, 3.8),
    )
    fig_mode.savefig(SAVE_PATH_MODE, dpi=300, bbox_inches="tight")
    log.info(f"Saved → {SAVE_PATH_MODE}  (total {time.time() - t_total:.1f}s)")
    plt.close(fig_mode)


if __name__ == "__main__":
    main()
