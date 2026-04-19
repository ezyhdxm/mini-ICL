#!/usr/bin/env python3
"""Generate combined KL-transition heatmap figure across E1, E2, E3.

Produces a single 1×3 subplot figure (phase-transition between Bayesian /
extrapolative inference modes) instead of three separate PNGs, with one
shared colorbar on the right.

Usage (from project root):
    python scripts/fig_kl_transition_combined.py
"""

import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_PATH = PROJECT_ROOT / "paper_figs" / "kl_transition_combined_logx.png"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from icl.utils.plot_config import apply_paper_style
from icl.utils.separability import plot_kl_transition_on_ax

apply_paper_style()


def _compute_coin(vocab_size: int = 6):
    from icl.coin.coin import plot_kl_model_vs_two_bayes_coin_transition_across_k
    return plot_kl_model_vs_two_bayes_coin_transition_across_k(
        k_values=range(11),
        mode="train",
        num_samples=1024,
        show_colorbar=False,
        show=False,
        vocab_size=vocab_size,
    )


def _compute_linear():
    from icl.linear.lr_task import plot_kl_model_vs_two_bayes_linear_transition_across_k
    return plot_kl_model_vs_two_bayes_linear_transition_across_k(
        k_values=range(11),
        mode="train",
        num_samples=1024,
        show_colorbar=False,
        exp_name_kwargs=dict(
            n_layer=16,
            total_steps=30_000,
            warmup_steps=15_000,
            batch_size=256,
            max_grad_norm=1.0,
        ),
        show=False,
    )


def _compute_latent():
    from icl.latent_markov.analysis import plot_kl_model_vs_two_bayes_latent_transition_across_k
    return plot_kl_model_vs_two_bayes_latent_transition_across_k(
        k_values=range(11),
        mode="train",
        num_samples=1024,
        show_colorbar=False,
        show=False,
        exp_name_kwargs=dict(
            total_steps=30_000,
            warmup_steps=15_000,
        ),
    )


def parse_args():
    p = argparse.ArgumentParser(description="Generate combined KL-transition heatmap figure (E1/E2/E3).")
    p.add_argument("--coin-vocab-size", type=int, default=6, metavar="V",
                   help="Vocabulary size for E1 Coins experiments (default: 16)")
    return p.parse_args()


def main():
    args = parse_args()
    panels = [
        ("E1 (Coins)",  lambda _v=args.coin_vocab_size: _compute_coin(_v)),
        ("E2 (Linear)", _compute_linear),
        ("E3 (Latent)", _compute_latent),
    ]

    t_total = time.time()
    data = []
    for label, compute_fn in panels:
        log.info(f"[{label}] computing KL transition heatmap …")
        t0 = time.time()
        out = compute_fn()
        plt.close("all")
        data.append(out)
        log.info(f"[{label}] done in {time.time() - t0:.1f}s")

    import numpy as np
    x_starts = [np.asarray(d["step_grid"], dtype=float).min() for d in data]
    x_ends   = [np.asarray(d["step_grid"], dtype=float).max() for d in data]
    x_lo = 10 ** np.floor(np.log10(max(x_starts)))
    x_hi = max(x_ends)
    common_xlim = (x_lo, x_hi)

    log.info("Composing figure …")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))

    last_mesh = None
    for idx, (ax, out) in enumerate(zip(axes, data)):
        mesh = plot_kl_transition_on_ax(
            ax, out,
            show_ylabel=(idx == 0),
            show_xlabel=True,
            show_colorbar=False,
            fig=fig,
        )
        ax.set_xscale("log")
        ax.set_xlim(common_xlim)
        last_mesh = mesh
        if idx > 0:
            ax.tick_params(labelleft=False)

    # Single shared colorbar on the right of the last panel.
    cbar = fig.colorbar(last_mesh, ax=axes[-1], pad=0.02, fraction=0.046)
    cbar.set_label(
        r"$\log(\mathrm{KL}_{\mathtt{M1}} / \mathrm{KL}_{\mathtt{M2}})$",
        fontsize=12,
    )
    cbar.ax.tick_params(labelsize=11)

    fig.tight_layout(w_pad=0.5)
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
    log.info(f"Saved → {SAVE_PATH}  (total {time.time() - t_total:.1f}s)")
    plt.close(fig)


if __name__ == "__main__":
    main()
