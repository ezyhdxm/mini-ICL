#!/usr/bin/env python3
"""Generate combined task-vector R² figure across E1, E2, E3.

Produces a single 1×3 subplot figure instead of three separate PNGs,
eliminating redundant y-axis labels and inter-panel gaps from LaTeX
subfigure composition.

Usage (from project root):
    python scripts/fig_task_vector_r2_combined.py
"""

import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_PATH = PROJECT_ROOT / "paper_figs" / "task_vector_r2_combined.png"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from icl.utils.plot_config import apply_paper_style
from icl.utils.separability import (
    TaskVectorR2Result,
    _layer_style,
    plot_task_vector_r2_on_ax,
)
from icl.utils.unified_interface import get_exp_name

apply_paper_style()


def _compute_coin(layers, vocab_size: int = 16):
    from icl.coin.analysis.variance import plot_task_vector_r2_coin
    exp_name = get_exp_name("coin", -1, vocab_size=vocab_size)
    out = plot_task_vector_r2_coin(
        exp_name, layers=layers, batch_size=1024,
        print_summary=False, log_x=False, show=False,
        extraction_point="post_mlp",
    )
    plt.close("all")
    return out["r2_results"]


def _compute_latent(layers):
    from icl.latent_markov.analysis import plot_task_vector_r2_latent
    exp_name = get_exp_name("latent", -1, total_steps=30_000, warmup_steps=15_000)
    out = plot_task_vector_r2_latent(
        exp_name, layers=layers, batch_size=1024,
        print_summary=False, log_x=False, show=False,
        extraction_point="post_mlp",
    )
    plt.close("all")
    return out["r2_results"]


def _compute_linear(layers):
    """Linear uses ANCOVA; convert r2_full → TaskVectorR2Result for compatibility."""
    from icl.linear.analysis import plot_task_vector_r2_linear
    exp_name = get_exp_name(
        "linear", -1,
        n_layer=16, total_steps=30_000,
        warmup_steps=15_000, batch_size=256,
        max_grad_norm=1.0,
    )
    out = plot_task_vector_r2_linear(
        exp_name, layers=layers, n_minor=0, batch_size=1024,
        print_summary=False, log_x=False, show=False,
        extraction_point="post_mlp",
    )
    plt.close("all")

    ancova = out["ancova_results"]
    converted = {}
    for l_num, pos_dict in ancova.items():
        converted[l_num] = {}
        for pos, res in pos_dict.items():
            converted[l_num][pos] = TaskVectorR2Result(
                r2=res.r2_full,
                ss_total=res.ss_total,
                ss_between=res.ss_total - res.ss_res_full,
                ss_within=res.ss_res_full,
                n_tasks=res.n_tasks,
                n_tokens=res.n_covariate_dims,
                n_batch=res.n_samples,
                layer_num=res.layer_num,
                position=res.position,
            )
    return converted


def parse_args():
    p = argparse.ArgumentParser(description="Generate combined task-vector R² figure (E1/E2/E3).")
    p.add_argument("--coin-vocab-size", type=int, default=16, metavar="V",
                   help="Vocabulary size for E1 Coins experiments (default: 16)")
    return p.parse_args()


def main():
    args = parse_args()
    panels = [
        ("E1 (Coins)",   lambda layers, _v=args.coin_vocab_size: _compute_coin(layers, _v),   range(6)),
        ("E2 (Linear)",  _compute_linear, range(4, 16, 2)),
        ("E3 (Latent)",  _compute_latent, range(6)),
    ]

    t_total = time.time()
    results = []
    for label, compute_fn, layers in panels:
        log.info(f"[{label}] computing task-vector R² …")
        t0 = time.time()
        results.append(compute_fn(layers))
        log.info(f"[{label}] done in {time.time() - t0:.1f}s")

    log.info("Composing figure …")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)

    for idx, (ax, r2) in enumerate(zip(axes, results)):
        plot_task_vector_r2_on_ax(
            ax, r2,
            log_x=False,
            show_ylabel=(idx == 0),
        )
        if idx > 0:
            ax.tick_params(labelleft=False)

    fig.tight_layout(w_pad=1.0)
    fig.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
    log.info(f"Saved → {SAVE_PATH}  (total {time.time() - t_total:.1f}s)")
    plt.close(fig)


if __name__ == "__main__":
    main()
