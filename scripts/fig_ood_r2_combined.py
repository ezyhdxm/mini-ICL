#!/usr/bin/env python3
"""Generate combined OOD R² projection figure across E1, E2, E3.

Produces a single 1×3 subplot figure (projection R² of OOD hidden states onto
the major-task subspace) instead of three separate PNGs, with a shared y-axis
and a single legend on the first panel.

Usage (from project root):
    python scripts/fig_ood_r2_combined.py
    python scripts/fig_ood_r2_combined.py --coin-layer 4 --linear-layer 9 --latent-layer 3
"""

import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from icl.utils.plot_config import apply_paper_style
from icl.utils.ood_major_projection_r2 import (
    draw_ood_r2_curves_on_ax,
    plot_maj_r2_ood_across_steps_coin,
    plot_maj_r2_ood_across_steps_linear,
    plot_maj_r2_ood_across_steps_latent,
)

apply_paper_style()

_STEPS_COIN   = [100] + list(range(1_000, 30_001, 1_000))
_STEPS_LINEAR = [200] + list(range(1_000, 30_001, 1_000))
_STEPS_LATENT = [100] + list(range(1_000, 30_001, 1_000))

_COMMON = dict(
    n_ood=128,
    B=1,
    n_gpus=2,
    ema_alpha=0.95,
    logx=True,
    show_title=False,
    position_blocks=((0, -1),),
    show_iqr_band=True,
    band_alpha=0.2,
    extraction_point="post_mlp",
    show=False,
)


def _save_path(coin_layer: int, linear_layer: int, latent_layer: int) -> Path:
    name = f"ood_r2_c{coin_layer}_l{linear_layer}_t{latent_layer}.png"
    return PROJECT_ROOT / "paper_figs" / name


def _compute_coin(layer: int, vocab_size: int = 16, avg_over: int = 128):
    return plot_maj_r2_ood_across_steps_coin(
        k_list=[-1, 0, 2, 4, 10],
        steps=_STEPS_COIN,
        layer=layer,
        vocab_size=vocab_size,
        avg_over=avg_over,
        **_COMMON,
    )


def _compute_linear(layer: int, avg_over: int = 128):
    return plot_maj_r2_ood_across_steps_linear(
        k_list=[-1, 0, 2, 4, 10],
        steps=_STEPS_LINEAR,
        layer=layer,
        avg_over=avg_over,
        exp_name_kwargs=dict(
            n_layer=16,
            total_steps=30_000,
            warmup_steps=15_000,
            batch_size=256,
            max_grad_norm=1.0,
        ),
        **_COMMON,
    )


def _compute_latent(layer: int, avg_over: int = 128):
    latent_kwargs = {**_COMMON, "position_blocks": ((96, 191),)}
    return plot_maj_r2_ood_across_steps_latent(
        k_list=[-1, 0, 2, 4, 10],
        steps=_STEPS_LATENT,
        layer=layer,
        avg_over=avg_over,
        exp_name_kwargs=dict(
            total_steps=30_000,
            warmup_steps=15_000,
        ),
        **latent_kwargs,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Generate combined OOD R² figure.")
    p.add_argument("--coin-layer",      type=int, default=4,  help="Layer index for E1 (Coins) [default: 4]")
    p.add_argument("--linear-layer",    type=int, default=10, help="Layer index for E2 (Linear) [default: 10]")
    p.add_argument("--latent-layer",    type=int, default=4,  help="Layer index for E3 (Latent) [default: 4]")
    p.add_argument("--coin-vocab-size", type=int, default=16, metavar="V",
                   help="Vocabulary size for E1 Coins experiments (default: 16)")
    p.add_argument("--avg-over", type=int, default=128, metavar="A",
                   help="Sequences averaged per OOD task before R² for all experiments (default: 128; use 1 for no averaging)")
    return p.parse_args()


def main():
    args = parse_args()
    save_path = _save_path(args.coin_layer, args.linear_layer, args.latent_layer)

    panels = [
        ("E1 (Coins)",  lambda: _compute_coin(args.coin_layer, args.coin_vocab_size, args.avg_over)),
        ("E2 (Linear)", lambda: _compute_linear(args.linear_layer, args.avg_over)),
        ("E3 (Latent)", lambda: _compute_latent(args.latent_layer, args.avg_over)),
    ]

    log.info(f"Layers: coin={args.coin_layer}, linear={args.linear_layer}, latent={args.latent_layer}")

    t_total = time.time()
    data = []
    for label, compute_fn in panels:
        log.info(f"[{label}] computing OOD R² across steps …")
        t0 = time.time()
        out = compute_fn()
        plt.close("all")
        data.append(out)
        log.info(f"[{label}] done in {time.time() - t0:.1f}s")

    log.info("Composing figure …")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)

    for idx, (ax, out) in enumerate(zip(axes, data)):
        draw_ood_r2_curves_on_ax(
            ax, out["results"],
            logx=True,
            ema_alpha=0.95,
            shadow_alpha=0.1,
            show_iqr_band=True,
            show_ylabel=(idx == 0),
            show_legend=(idx == len(data) - 1),
            legend_title=r"$N_{\mathrm{minor}}$",
        )
        if idx > 0:
            ax.tick_params(labelleft=False)

    fig.tight_layout(w_pad=0.5)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    log.info(f"Saved → {save_path}  (total {time.time() - t_total:.1f}s)")
    plt.close(fig)


if __name__ == "__main__":
    main()
