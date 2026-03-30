#!/usr/bin/env python3
"""Generate combined ID/OOD training-loss figure across E1 (Coins), E2 (Linear), E3 (Latent).

Produces a 3×2 subplot figure: rows = experiments (E1, E2, E3),
columns = ID loss (left) and OOD loss (right).  One shared legend
placed to the right of the figure.

Usage (from project root):
    python scripts/fig_id_ood_loss_combined.py
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_PATH = PROJECT_ROOT / "paper_figs" / "id_ood_loss_combined.png"

# Ensure the project src is on the path when running as a plain script.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_log_json(results_subdir: str, exp_name: str) -> dict:
    """Try a few candidate paths for a results log.json and return parsed JSON."""
    import json
    candidates = [
        PROJECT_ROOT / "results" / results_subdir / exp_name / "log.json",
        Path("results") / results_subdir / exp_name / "log.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(f"log.json not found for {exp_name} in {results_subdir}/")


def _extract_id_ood(data: dict, k: int) -> dict:
    """Pull train_steps, id_loss, ood_loss arrays out of a log.json dict."""
    train_steps = np.asarray(data.get("eval/step", data.get("train/step", [])), dtype=float)
    id_loss = data.get("eval/IDLoss")
    ood_loss = data.get("eval/OODLoss")
    if id_loss is None or ood_loss is None:
        raise KeyError("Missing eval/IDLoss or eval/OODLoss")
    id_loss = np.asarray(id_loss, dtype=float)
    ood_loss = np.asarray(ood_loss, dtype=float)
    if train_steps.size == 0:
        L = min(len(id_loss), len(ood_loss))
        train_steps = np.arange(1, L + 1, dtype=float)
    L = min(len(train_steps), len(id_loss), len(ood_loss))
    return dict(
        n_minor=2 ** k if k >= 0 else 0,
        train_steps=train_steps[:L],
        id_loss=id_loss[:L],
        ood_loss=ood_loss[:L],
    )


def _load_coin(k_list, vocab_size: int = 6) -> dict:
    from icl.utils.unified_interface import get_exp_name
    results = {}
    for k in k_list:
        exp_name = get_exp_name("coin", k, vocab_size=vocab_size)
        try:
            data = _load_log_json("coin", exp_name)
            results[k] = _extract_id_ood(data, k)
        except Exception as e:
            log.warning(f"[E1] Could not load k={k}: {e}")
    return results


def _load_linear(k_list, exp_name_kwargs: dict | None = None) -> dict:
    from icl.linear.analysis.posterior._analysis_plots import plot_id_ood_loss
    out = plot_id_ood_loss(
        k_list=k_list,
        logx=False,
        show=False,
        exp_name_kwargs=exp_name_kwargs or dict(
            n_layer=16,
            total_steps=30_000,
            warmup_steps=15_000,
            batch_size=256,
            max_grad_norm=1.0,
        ),
    )
    plt.close("all")
    return out.get("results", {})


def _load_latent(k_list) -> dict:
    from icl.utils.unified_interface import get_exp_name
    results = {}
    for k in k_list:
        exp_name = get_exp_name("latent", k, total_steps=30_000, warmup_steps=15_000)
        try:
            data = _load_log_json("latent", exp_name)
            results[k] = _extract_id_ood(data, k)
        except Exception as e:
            log.warning(f"[E3] Could not load k={k}: {e}")
    return results


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

LEGEND_TITLE = r"$\log_2(N_{\mathrm{minor}})$"
LW = 1.5
ALPHA = 0.85
FS_LABEL = 13
FS_TICK = 11
FS_TITLE = 12


def _color_map(ks_sorted: list) -> dict:
    cmap = plt.get_cmap("viridis")
    nk = len(ks_sorted)
    return {k: cmap(0.15 + 0.75 * (i / max(1, nk - 1))) for i, k in enumerate(ks_sorted)}


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", alpha=0.15, linestyle="-")
    ax.grid(True, which="minor", alpha=0.06, linestyle=":")
    ax.tick_params(labelsize=FS_TICK)


def _plot_row(
    ax_id: plt.Axes,
    ax_ood: plt.Axes,
    results: dict,
    ylabel_id: str,
    row_label: str,
    noise_floor: float | None = None,
) -> list:
    """Populate one row's ID and OOD axes from a results dict.

    Returns the legend handles from the ID panel (for the shared legend).
    """
    ks_sorted = sorted(results.keys())
    cmap = _color_map(ks_sorted)

    for ax in (ax_id, ax_ood):
        _style_ax(ax)
        ax.set_xscale("log")
        if noise_floor is not None:
            ax.axhline(
                noise_floor,
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.6,
                label="_nolegend_",
            )

    for k in ks_sorted:
        d = results[k]
        c = cmap[k]
        xs = d["train_steps"]
        mask = xs > 0
        ax_id.plot(xs[mask], d["id_loss"][mask], color=c, lw=LW, alpha=ALPHA, label=str(k))

    for k in ks_sorted:
        d = results[k]
        c = cmap[k]
        xs = d["train_steps"]
        mask = xs > 0
        ax_ood.plot(xs[mask], d["ood_loss"][mask], color=c, lw=LW, alpha=ALPHA)

    # Row label on the left panel
    ax_id.set_ylabel(f"{row_label}\n{ylabel_id}", fontsize=FS_LABEL)

    return ax_id.get_legend_handles_labels()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate combined ID/OOD loss figure (E1/E2/E3).")
    p.add_argument(
        "--k-max", type=int, default=10, metavar="K",
        help="Maximum k value (inclusive).  k ranges from 0 to K (default: 10).",
    )
    p.add_argument(
        "--coin-vocab-size", type=int, default=6, metavar="V",
        help="Vocabulary size for E1 Coins experiments (default: 16).",
    )
    p.add_argument(
        "--dpi", type=int, default=300,
        help="Output DPI (default: 300).",
    )
    p.add_argument(
        "--out", type=Path, default=SAVE_PATH,
        help=f"Output path (default: {SAVE_PATH}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    k_list = range(args.k_max + 1)
    os.chdir(PROJECT_ROOT)

    # ------------------------------------------------------------------
    # Load data for each experiment
    # ------------------------------------------------------------------
    panels = [
        ("E1", "Coins",  lambda: _load_coin(k_list, vocab_size=args.coin_vocab_size)),
        ("E2", "Linear", lambda: _load_linear(k_list)),
        ("E3", "Latent", lambda: _load_latent(k_list)),
    ]

    loaded = []
    t_total = time.time()
    for exp_id, exp_name, loader in panels:
        label = f"{exp_id} ({exp_name})"
        log.info(f"[{label}] loading data …")
        t0 = time.time()
        results = loader()
        if results:
            log.info(f"[{label}] loaded {len(results)} experiments in {time.time()-t0:.1f}s")
        else:
            log.warning(f"[{label}] no data loaded — row will be empty")
        loaded.append((exp_id, exp_name, results))

    # ------------------------------------------------------------------
    # Build the combined 3×2 figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        3, 2,
        figsize=(10, 9),
        sharex=False,   # x-ranges differ per experiment
    )
    fig.patch.set_facecolor("white")

    ylabels = ["Loss (KL)", "Loss (RMSE)", "Loss (KL)"]
    noise_floors = [None, 0.5, None]   # noise floor only for linear (E2)

    shared_handles, shared_labels = [], []

    for row_idx, ((exp_id, exp_name, results), ylabel, nf) in enumerate(
        zip(loaded, ylabels, noise_floors)
    ):
        ax_id, ax_ood = axes[row_idx]

        # Column titles on the first row only
        if row_idx == 0:
            ax_id.set_title("In-distribution", fontsize=FS_TITLE, pad=6)
            ax_ood.set_title("Out-of-distribution", fontsize=FS_TITLE, pad=6)

        if results:
            handles, labels = _plot_row(
                ax_id, ax_ood,
                results=results,
                ylabel_id=ylabel,
                row_label=exp_id,
                noise_floor=nf,
            )
            # Collect handles from the first populated row for the shared legend
            if not shared_handles:
                shared_handles, shared_labels = handles, labels
        else:
            for ax in (ax_id, ax_ood):
                _style_ax(ax)
                ax.set_xscale("log")
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="gray")
            ax_id.set_ylabel(f"{exp_id}\n{ylabel}", fontsize=FS_LABEL)

        # x-label only on the bottom row
        if row_idx == 2:
            ax_id.set_xlabel("Training Step", fontsize=FS_LABEL)
            ax_ood.set_xlabel("Training Step", fontsize=FS_LABEL)

    fig.tight_layout(w_pad=0.2, h_pad=1.2)

    # Single shared legend to the right of the right column
    if shared_handles:
        fig.legend(
            shared_handles,
            shared_labels,
            title=LEGEND_TITLE,
            fontsize=FS_TICK,
            title_fontsize=FS_LABEL,
            frameon=True,
            framealpha=0.95,
            loc="center left",
            bbox_to_anchor=(0.92, 0.5),
        )

    fig.subplots_adjust(right=0.88)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    log.info(f"Saved → {out_path}  (total {time.time()-t_total:.1f}s)")
    plt.close(fig)


if __name__ == "__main__":
    main()
