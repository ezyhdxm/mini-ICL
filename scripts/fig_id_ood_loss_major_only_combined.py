#!/usr/bin/env python3
"""Combined ID/OOD training-loss figure for major-task-only sweeps (E1/E2/E3).

Expects experiments trained with ``run_major_only_sweep.py`` (or equivalent):
``n_tasks`` swept over powers of two, ``n_minor_tasks=1``, ``p_minor=1e-12``,
same step/warmup settings as ``run_pipeline.py`` / ``run_major_only_sweep.py``.

Usage (from ``mini-ICL`` root)::

    uv run python scripts/fig_id_ood_loss_major_only_combined.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_PATH = PROJECT_ROOT / "paper_figs" / "id_ood_loss_major_only_combined.png"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Match ``run_major_only_sweep.py`` + ``get_exp_name`` / training config.
MAJOR_COIN_KW = dict(total_steps=30_000, warmup_steps=15_000)
MAJOR_LATENT_KW = dict(total_steps=30_000, warmup_steps=15_000)
MAJOR_LINEAR_KW = dict(
    n_layer=16,
    total_steps=30_000,
    warmup_steps=15_000,
    batch_size=256,
    max_grad_norm=1.0,
    noise_scale=0.5,
)

LEGEND_TITLE = r"$N_{\mathrm{major}}$"
LW = 1.5
ALPHA = 0.85
FS_LABEL = 13
FS_TICK = 11
FS_TITLE = 12


def _load_log_json(results_subdir: str, exp_name: str) -> dict:
    candidates = [
        PROJECT_ROOT / "results" / results_subdir / exp_name / "log.json",
        Path("results") / results_subdir / exp_name / "log.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(f"log.json not found for {exp_name} in {results_subdir}/")


def _extract_id_ood_major(data: dict, n_major: int) -> dict:
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
        n_major=int(n_major),
        train_steps=train_steps[:L],
        id_loss=id_loss[:L],
        ood_loss=ood_loss[:L],
    )


def _load_coin_major(n_major_list: list[int], vocab_size: int) -> dict:
    from icl.utils.unified_interface import get_exp_name

    results = {}
    for n_major in n_major_list:
        exp_name = get_exp_name(
            "coin",
            k=0,
            vocab_size=vocab_size,
            n_tasks=n_major,
            n_minor_tasks=1,
            p_minor=1e-12,
            **MAJOR_COIN_KW,
        )
        try:
            data = _load_log_json("coin", exp_name)
            results[n_major] = _extract_id_ood_major(data, n_major)
        except Exception as e:
            log.warning(f"[E1] Could not load N_major={n_major}: {e}")
    return results


def _load_linear_major(n_major_list: list[int]) -> dict:
    from icl.linear.analysis.posterior._analysis_plots import plot_id_ood_loss

    out = plot_id_ood_loss(
        n_major_list=n_major_list,
        logx=False,
        show=False,
        exp_name_kwargs=dict(MAJOR_LINEAR_KW),
    )
    plt.close("all")
    return out.get("results", {})


def _load_latent_major(n_major_list: list[int]) -> dict:
    from icl.utils.unified_interface import get_exp_name

    results = {}
    for n_major in n_major_list:
        exp_name = get_exp_name(
            "latent",
            k=0,
            n_tasks=n_major,
            n_minor_tasks=1,
            p_minor=1e-12,
            **MAJOR_LATENT_KW,
        )
        try:
            data = _load_log_json("latent", exp_name)
            results[n_major] = _extract_id_ood_major(data, n_major)
        except Exception as e:
            log.warning(f"[E3] Could not load N_major={n_major}: {e}")
    return results


def _color_map(keys_sorted: list) -> dict:
    cmap = plt.get_cmap("viridis")
    nk = len(keys_sorted)
    return {k: cmap(0.15 + 0.75 * (i / max(1, nk - 1))) for i, k in enumerate(keys_sorted)}


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
):
    keys_sorted = sorted(results.keys())
    cmap = _color_map(keys_sorted)

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

    for key in keys_sorted:
        d = results[key]
        c = cmap[key]
        xs = d["train_steps"]
        mask = xs > 0
        lbl = str(key)
        ax_id.plot(xs[mask], d["id_loss"][mask], color=c, lw=LW, alpha=ALPHA, label=lbl)

    for key in keys_sorted:
        d = results[key]
        c = cmap[key]
        xs = d["train_steps"]
        mask = xs > 0
        ax_ood.plot(xs[mask], d["ood_loss"][mask], color=c, lw=LW, alpha=ALPHA)

    ax_id.set_ylabel(f"{row_label}\n{ylabel_id}", fontsize=FS_LABEL)
    return ax_id.get_legend_handles_labels()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ID/OOD loss figure for major-only sweeps.")
    p.add_argument(
        "--exp-min", type=int, default=2,
        help="Smallest major count is 2**exp_min (default: 2 → 4).",
    )
    p.add_argument(
        "--exp-max", type=int, default=10,
        help="Largest major count is 2**exp_max (default: 10 → 1024).",
    )
    p.add_argument(
        "--coin-vocab-size", type=int, default=8,
        help="Vocab size for coin runs (default: 8, training sweep default).",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--out", type=Path, default=SAVE_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_major_list = [2**e for e in range(args.exp_min, args.exp_max + 1)]
    os.chdir(PROJECT_ROOT)

    panels = [
        ("E1", "Coins", lambda: _load_coin_major(n_major_list, args.coin_vocab_size)),
        ("E2", "Linear", lambda: _load_linear_major(n_major_list)),
        ("E3", "Latent", lambda: _load_latent_major(n_major_list)),
    ]

    loaded = []
    t_total = time.time()
    for exp_id, exp_name, loader in panels:
        label = f"{exp_id} ({exp_name})"
        log.info(f"[{label}] loading …")
        t0 = time.time()
        results = loader()
        if results:
            log.info(f"[{label}] loaded {len(results)} runs in {time.time()-t0:.1f}s")
        else:
            log.warning(f"[{label}] no data")
        loaded.append((exp_id, exp_name, results))

    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.patch.set_facecolor("white")

    ylabels = ["Loss (KL)", "Loss (RMSE)", "Loss (KL)"]
    noise_floors = [None, 0.5, None]
    shared_handles, shared_labels = [], []

    for row_idx, ((exp_id, exp_name, results), ylabel, nf) in enumerate(
        zip(loaded, ylabels, noise_floors)
    ):
        ax_id, ax_ood = axes[row_idx]
        if row_idx == 0:
            ax_id.set_title("In-distribution", fontsize=FS_TITLE, pad=6)
            ax_ood.set_title("Out-of-distribution", fontsize=FS_TITLE, pad=6)

        if results:
            handles, labels = _plot_row(
                ax_id, ax_ood, results, ylabel_id=ylabel, row_label=exp_id, noise_floor=nf,
            )
            if not shared_handles:
                shared_handles, shared_labels = handles, labels
        else:
            for ax in (ax_id, ax_ood):
                _style_ax(ax)
                ax.set_xscale("log")
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="gray")
            ax_id.set_ylabel(f"{exp_id}\n{ylabel}", fontsize=FS_LABEL)

        if row_idx == 2:
            ax_id.set_xlabel("Training Step", fontsize=FS_LABEL)
            ax_ood.set_xlabel("Training Step", fontsize=FS_LABEL)

    fig.tight_layout(w_pad=0.2, h_pad=1.2)
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
    log.info(f"Saved → {out_path}  ({time.time()-t_total:.1f}s)")
    plt.close(fig)


if __name__ == "__main__":
    main()
