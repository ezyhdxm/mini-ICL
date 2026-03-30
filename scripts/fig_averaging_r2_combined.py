#!/usr/bin/env python3
"""Generate combined averaging R² figure across E1, E2, E3.

Produces a single 1×3 subplot figure (additive R²: task + token subspace)
instead of three separate PNGs.

Usage (from project root):
    python scripts/fig_averaging_r2_combined.py
"""

import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_PATH = PROJECT_ROOT / "paper_figs" / "averaging_r2_combined.png"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from icl.utils.plot_config import apply_paper_style
from icl.utils.separability import plot_averaging_r2_on_ax
from icl.utils.unified_interface import get_exp_name

apply_paper_style()

if not torch.cuda.is_available():
    log.warning(
        "CUDA not available — running on CPU.  Expect long runtimes for the "
        "linear model forward pass.  On WSL2, install the CUDA toolkit via "
        "https://developer.nvidia.com/cuda/wsl to enable GPU acceleration."
    )


def _compute_coin(layers, vocab_size: int = 6):
    from icl.coin.analysis.probes._plots import plot_averaging_r2_coin
    exp_name = get_exp_name("coin", -1, vocab_size=vocab_size)
    out = plot_averaging_r2_coin(
        exp_name, layers=layers,
        estimation_positions=list(range(100, 120)),
        evaluation_positions=range(120),
        fit_token="anova",
        batch_size=256,
        per_position_mean=True,
        per_position_token_vecs=False,
        log_x=False,
        show=False,
        extraction_point="post_mlp",
        plots="additive",
    )
    plt.close("all")
    return out["results"]


def _compute_linear(layers):
    from icl.linear.analysis.probes._plots import plot_averaging_r2_linear
    exp_name = get_exp_name(
        "linear", -1,
        n_layer=16, total_steps=30_000,
        warmup_steps=15_000, batch_size=256,
        max_grad_norm=1.0,
    )
    out = plot_averaging_r2_linear(
        exp_name, layers=layers,
        estimation_positions=list(range(54, 64)),
        evaluation_positions=range(64),
        fit_token="linear",
        per_position_mean=True,
        post_layernorm=False,
        log_x=False,
        show=False,
        extraction_point="post_mlp",
        plots="additive",
    )
    plt.close("all")
    return out["results"]


def _compute_latent(layers):
    from icl.latent_markov.analysis.probes._plots import plot_averaging_r2_latent
    exp_name = get_exp_name("latent", -1, total_steps=30_000, warmup_steps=15_000)
    out = plot_averaging_r2_latent(
        exp_name, layers=layers,
        estimation_positions=list(range(170, 190)),
        evaluation_positions=range(190),
        fit_token="anova",
        batch_size=256,
        per_position_mean=True,
        per_position_token_vecs=False,
        post_layernorm=True,
        log_x=False,
        show=False,
        extraction_point="post_mlp",
        plots="additive",
    )
    plt.close("all")
    return out["results"]


def _log_last10_r2(label: str, r2: dict) -> None:
    """Log mean additive R² over the last 10 positions for each layer."""
    log.info(f"[{label}] mean R² (last 10 positions) per layer:")
    for layer in sorted(r2.keys()):
        pos_results = r2[layer]
        positions = sorted(pos_results.keys())
        last10 = positions[-10:]
        vals = [pos_results[p].r2_additive for p in last10]
        mean_val = sum(vals) / len(vals)
        log.info(f"  layer {layer:2d}: {mean_val:.4f}  (positions {last10[0]}–{last10[-1]})")


def parse_args():
    p = argparse.ArgumentParser(description="Generate combined averaging R² figure (E1/E2/E3).")
    p.add_argument("--coin-vocab-size", type=int, default=6, metavar="V",
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
        log.info(f"[{label}] computing averaging R² …")
        t0 = time.time()
        r2 = compute_fn(layers)
        results.append(r2)
        log.info(f"[{label}] done in {time.time() - t0:.1f}s")
        _log_last10_r2(label, r2)

    log.info("Composing figure …")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)

    for idx, (ax, r2) in enumerate(zip(axes, results)):
        plot_averaging_r2_on_ax(
            ax, r2,
            log_x=False,
            show_ylabel=False,
        )
        if idx > 0:
            ax.tick_params(labelleft=False)

    fig.tight_layout(w_pad=1.0)
    fig.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
    log.info(f"Saved → {SAVE_PATH}  (total {time.time() - t_total:.1f}s)")
    plt.close(fig)


if __name__ == "__main__":
    main()
