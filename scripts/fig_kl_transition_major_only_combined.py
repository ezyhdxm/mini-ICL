#!/usr/bin/env python3
"""Combined KL-transition heatmaps for major-task-only sweeps (E1/E2/E3).

Uses the same experiments as ``fig_id_ood_loss_major_only_combined.py``
(``run_major_only_sweep.py`` style: ``n_tasks`` powers of two, ``n_minor_tasks=1``,
``p_minor=1e-12``).

Layout mirrors ``fig_kl_transition_combined.py`` exactly (1×3, one shared
colorbar, ratio metric for all panels).  The hybrid/extrapolation baseline is
constructed with the **opposite** prior from training: p_major ≈ 0, p_minor ≈ 1,
so it represents the ideal predictor for a minor-task-dominated distribution.
Y-axis shows log₂(N_major) with N_major tick labels so cells are evenly spaced.

Usage (from ``mini-ICL`` root)::

    uv run python scripts/fig_kl_transition_major_only_combined.py
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_PATH = PROJECT_ROOT / "paper_figs" / "kl_transition_major_only_combined.png"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from icl.utils.plot_config import apply_paper_style
from icl.utils.separability import plot_kl_transition_on_ax

apply_paper_style()

MAJOR_COIN_KW   = dict(total_steps=30_000, warmup_steps=15_000)
MAJOR_LATENT_KW = dict(total_steps=30_000, warmup_steps=15_000)
MAJOR_LINEAR_KW = dict(
    n_layer=16,
    total_steps=30_000,
    warmup_steps=15_000,
    batch_size=256,
    max_grad_norm=1.0,
    noise_scale=0.5,
)

Y_LABEL_MAJOR = r"$\log_2(N_{\mathrm{major}})$"


def _remap_to_log2(out: dict) -> dict:
    """Replace k_values_loaded with log2(N_major) so the y-axis is evenly spaced."""
    out = copy.copy(out)
    ks = np.asarray(out["k_values_loaded"], dtype=float)
    out["k_values_loaded"] = np.log2(ks)
    return out


def _n_major_yticks(ax, n_major_list: list[int]) -> None:
    """Label yticks with log2(N_major) integer values."""
    log2_vals = [int(np.log2(n)) for n in n_major_list]
    ax.set_yticks(log2_vals)
    ax.set_yticklabels([str(v) for v in log2_vals])


def _compute_coin(n_major_list: list[int], vocab_size: int):
    from icl.coin.coin_kl import plot_kl_model_vs_two_bayes_coin_transition_across_k
    return plot_kl_model_vs_two_bayes_coin_transition_across_k(
        n_major_values=n_major_list,
        mode="train",
        num_samples=1024,
        show_colorbar=False,
        show=False,
        vocab_size=vocab_size,
        major_only_exp_kwargs=MAJOR_COIN_KW,
    )


def _compute_linear(n_major_list: list[int]):
    from icl.linear.lr_task import plot_kl_model_vs_two_bayes_linear_transition_across_k
    return plot_kl_model_vs_two_bayes_linear_transition_across_k(
        n_major_values=n_major_list,
        mode="train",
        num_samples=1024,
        show_colorbar=False,
        exp_name_kwargs=dict(MAJOR_LINEAR_KW),
        show=False,
    )


def _compute_latent(n_major_list: list[int]):
    from icl.latent_markov.analysis import plot_kl_model_vs_two_bayes_latent_transition_across_k
    return plot_kl_model_vs_two_bayes_latent_transition_across_k(
        n_major_values=n_major_list,
        mode="train",
        num_samples=1024,
        show_colorbar=False,
        show=False,
        exp_name_kwargs=dict(MAJOR_LATENT_KW),
        major_only_exp_kwargs={},
    )


def parse_args():
    p = argparse.ArgumentParser(description="KL transition heatmaps for major-only sweeps.")
    p.add_argument("--exp-min", type=int, default=2)
    p.add_argument("--exp-max", type=int, default=10)
    p.add_argument("--coin-vocab-size", type=int, default=8)
    p.add_argument("--out", type=Path, default=SAVE_PATH)
    return p.parse_args()


def main():
    args = parse_args()
    n_major_list = [2**e for e in range(args.exp_min, args.exp_max + 1)]

    panels = [
        ("E1 (Coins)",  lambda: _compute_coin(n_major_list, args.coin_vocab_size)),
        ("E2 (Linear)", lambda: _compute_linear(n_major_list)),
        ("E3 (Latent)", lambda: _compute_latent(n_major_list)),
    ]

    t_total = time.time()
    data = []
    for label, compute_fn in panels:
        log.info(f"[{label}] computing …")
        t0 = time.time()
        try:
            out = compute_fn()
        except Exception as e:
            log.error(f"[{label}] failed: {e}")
            raise
        plt.close("all")
        # Remap y-axis: N_major → log2(N_major) for even cell spacing.
        data.append(_remap_to_log2(out))
        log.info(f"[{label}] done in {time.time() - t0:.1f}s")

    log.info("Composing figure …")
    # Extra column width for the two colorbars (ratio + absolute KL).
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    last_mesh = None
    for idx, (ax, out) in enumerate(zip(axes, data)):
        mesh = plot_kl_transition_on_ax(
            ax, out,
            show_ylabel=(idx == 0),
            show_xlabel=True,
            show_colorbar=False,
            fig=fig,
            y_axis_label=Y_LABEL_MAJOR,
            metric="ratio",
        )
        ax.set_xscale("log")
        _n_major_yticks(ax, n_major_list)
        last_mesh = mesh
        if idx > 0:
            ax.tick_params(labelleft=False)

    # Single shared colorbar — mirrors fig_kl_transition_combined.py.
    cbar = fig.colorbar(last_mesh, ax=axes[-1], pad=0.04, fraction=0.046)
    cbar.set_label(
        r"$\log(\mathrm{KL}_{\mathrm{Bayesian}} / \mathrm{KL}_{\mathrm{extrapolation}})$",
    )

    fig.tight_layout(w_pad=0.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    log.info(f"Saved → {args.out}  (total {time.time() - t_total:.1f}s)")
    plt.close(fig)


if __name__ == "__main__":
    main()
