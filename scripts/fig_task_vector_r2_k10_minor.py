#!/usr/bin/env python3
"""Task-vector R² for k=10 experiments with minor tasks included.

Usage (from project root):
    PYTHONPATH=src python scripts/fig_task_vector_r2_k10_minor.py
"""

import time
import logging
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = PROJECT_ROOT / "paper_figs"
SAVE_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from icl.utils.unified_interface import get_exp_name

K = 10
N_MINOR = 30      # include 30 minor tasks (full 1024 would OOM the factorial tensor)
BATCH_SIZE = 64


def run_coin():
    from icl.coin.analysis import plot_task_vector_r2_coin

    exp_name = get_exp_name("coin", K, vocab_size=6)
    log.info(f"[Coin k={K}] exp_name = {exp_name}")
    out = plot_task_vector_r2_coin(
        exp_name,
        layers=range(6),
        batch_size=BATCH_SIZE,
        n_minor=N_MINOR,
        print_summary=True,
        log_x=False,
        show=False,
        extraction_point="post_mlp",
    )
    fig = out["fig"]
    path = SAVE_DIR / f"task_vector_r2_coin_k{K}_minor.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close("all")
    log.info(f"[Coin] saved → {path}")
    return out


def run_latent():
    from icl.latent_markov.analysis import plot_task_vector_r2_latent

    exp_name = get_exp_name("latent", K)
    log.info(f"[Latent k={K}] exp_name = {exp_name}")
    out = plot_task_vector_r2_latent(
        exp_name,
        layers=range(6),
        batch_size=BATCH_SIZE,
        n_minor=N_MINOR,
        print_summary=True,
        log_x=False,
        show=False,
        show_ylabel=False,
        extraction_point="post_mlp",
    )
    fig = out["fig"]
    path = SAVE_DIR / f"task_vector_r2_latent_k{K}_minor.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close("all")
    log.info(f"[Latent] saved → {path}")
    return out


def run_linear():
    from icl.linear.analysis import plot_task_vector_r2_linear

    exp_name = get_exp_name(
        "linear", K,
        n_layer=16,
        total_steps=30_000,
        warmup_steps=15_000,
        batch_size=256,
        max_grad_norm=1.0,
    )
    log.info(f"[Linear k={K}] exp_name = {exp_name}")
    out = plot_task_vector_r2_linear(
        exp_name,
        layers=range(4, 16, 2),
        n_minor=N_MINOR,
        batch_size=BATCH_SIZE,
        print_summary=True,
        log_x=False,
        show=False,
        show_ylabel=False,
        extraction_point="post_mlp",
    )
    fig = out["fig"]
    path = SAVE_DIR / f"task_vector_r2_linear_k{K}_minor.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close("all")
    log.info(f"[Linear] saved → {path}")
    return out


def main():
    t0 = time.time()

    log.info("=" * 60)
    log.info(f"  Task-vector R² with k={K} (minor tasks included)")
    log.info(f"  n_minor={N_MINOR}, batch_size={BATCH_SIZE}")
    log.info("=" * 60)

    log.info("\n>>> Coin <<<")
    run_coin()

    log.info("\n>>> Latent <<<")
    run_latent()

    log.info("\n>>> Linear <<<")
    run_linear()

    log.info(f"\nAll done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
