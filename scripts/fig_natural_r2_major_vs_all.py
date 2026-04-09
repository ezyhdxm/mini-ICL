#!/usr/bin/env python3
"""Natural-sequence R²: major-only vs all tasks (major+minor).

Compares whether major tasks have higher R² (lower within-cell variance)
because the model converges faster on them.

Runs on both:
  - V=8 k=10: train_1afaf1150ad2494d2e61b031ea3d54b5
  - V=6 k=10 long: train_6e9df91166eb0ad0e4efbd1fb42b4854

Usage:
    PYTHONPATH=src python scripts/fig_natural_r2_major_vs_all.py
"""

import gc
import time
import logging
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = PROJECT_ROOT / "paper_figs"
SAVE_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

import icl.utils.notebook_utils as nu
from icl.latent_markov.analysis.ood import get_latent_sampler
from icl.coin.analysis._helpers import extract_hidden_multi_coin_latent
from icl.utils.separability._task_vector_r2 import _layer_style

SAMPLES_PER_TASK = 256
TASK_BATCH = 32
N_MINOR = 30


def collect_natural(exp_name, layers, extraction_point="post_mlp"):
    """Return hiddens (L, N, T, D), tokens (N, T), task_ids (N,), metadata."""
    _, _, config = nu.load_everything("latent", exp_name)
    step = config.training.num_epochs
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval().to(config.device)

    sampler, _, _ = get_latent_sampler(exp_name, N_MINOR, n_ood=0)
    device = config.device
    n_major = sampler.n_major_tasks
    n_tasks = n_major + sampler.n_minor_tasks
    T = sampler.seq_len - 1
    V = int(sampler.num_states)

    task_pos = torch.arange(T, device=device)
    all_h, all_tok, all_tid = [], [], []

    for k in range(n_tasks):
        if k % 10 == 0:
            log.info(f"  task {k}/{n_tasks}")
        ph, pt = [], []
        for b0 in range(0, SAMPLES_PER_TASK, TASK_BATCH):
            bs = min(b0 + TASK_BATCH, SAMPLES_PER_TASK) - b0
            if k < n_major:
                gen = sampler.generate(mode="testing", task=k, num_samples=bs)
            else:
                gen = sampler.generate(mode="minor", task=k - n_major, num_samples=bs)
            samples = gen[0] if isinstance(gen, (tuple, list)) else gen
            if samples.dim() == 3:
                samples = samples.squeeze(0)
            samples = samples.to(device)
            pt.append(samples[:, :T].cpu())
            h = extract_hidden_multi_coin_latent(model, samples, list(layers), task_pos, extraction_point=extraction_point)
            ph.append(h.cpu())
            del samples, h
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        all_h.append(torch.cat(ph, dim=1))
        all_tok.append(torch.cat(pt, dim=0))
        all_tid.append(torch.full((SAMPLES_PER_TASK,), k, dtype=torch.long))

    model.cpu(); del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return (torch.cat(all_h, dim=1), torch.cat(all_tok, dim=0),
            torch.cat(all_tid, dim=0), T, V, n_major)


def compute_r2(hiddens, tokens, task_ids, T, V, task_mask=None, eps=1e-10):
    """Cell-means R² per (layer, position). If task_mask given, restrict to those samples."""
    L = hiddens.shape[0]
    r2 = np.zeros((L, T))

    if task_mask is not None:
        hiddens = hiddens[:, task_mask]
        tokens = tokens[task_mask]
        task_ids = task_ids[task_mask]

    for l in range(L):
        for t in range(T):
            h = hiddens[l, :, t, :].float()
            cell_key = task_ids.long() * V + tokens[:, t].long()
            gm = h.mean(dim=0)
            ss_total = ((h - gm) ** 2).sum().item()
            ss_within = 0.0
            for c in cell_key.unique():
                hc = h[cell_key == c]
                if hc.shape[0] < 1:
                    continue
                ss_within += ((hc - hc.mean(dim=0)) ** 2).sum().item()
            r2[l, t] = 1.0 - ss_within / (ss_total + eps)
    return r2


def run_one_experiment(exp_name, label, layers):
    log.info(f"\n{'='*50}")
    log.info(f"  {label}: {exp_name}")
    log.info(f"{'='*50}")

    hiddens, tokens, task_ids, T, V, n_major = collect_natural(exp_name, layers)
    N = hiddens.shape[1]
    log.info(f"  N={N}, T={T}, V={V}, n_major={n_major}")

    major_mask = task_ids < n_major
    minor_mask = task_ids >= n_major

    log.info("  Computing R² (all tasks) ...")
    r2_all = compute_r2(hiddens, tokens, task_ids, T, V)
    log.info("  Computing R² (major only) ...")
    r2_major = compute_r2(hiddens, tokens, task_ids, T, V, task_mask=major_mask)
    log.info("  Computing R² (minor only) ...")
    r2_minor = compute_r2(hiddens, tokens, task_ids, T, V, task_mask=minor_mask)

    return r2_all, r2_major, r2_minor, T, V, n_major, list(layers)


def plot_comparison(r2_all, r2_major, r2_minor, T, layers, label, save_name):
    positions = np.arange(T)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)

    for ax, r2, title in zip(axes, [r2_major, r2_minor, r2_all],
                              ["Major only", "Minor only", "All tasks"]):
        for l_idx, l_num in enumerate(layers):
            ax.plot(positions, r2[l_idx], label=str(l_num),
                    **_layer_style(l_num, len(positions)))
        ax.set_xlabel("Position", fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=12)
        ax.legend(title="Layer", fontsize=9, title_fontsize=9, framealpha=0.9, loc="lower right")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("$R^2$", fontsize=13)
    fig.suptitle(f"Natural-sequence R²: {label}", fontsize=15, y=1.02)
    fig.tight_layout()
    path = SAVE_DIR / save_name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved → {path}")


def print_summary(r2_all, r2_major, r2_minor, T, layers, label, n_major):
    summary_pos = [0, 5, 10, 20, 50, 100, 200, T - 1]
    summary_pos = [p for p in summary_pos if p < T]

    for tag, r2 in [("Major", r2_major), ("Minor", r2_minor), ("All", r2_all)]:
        print(f"\n  {label} — {tag}:")
        header = "  Layer  " + "  ".join(f"t={p:>3d}" for p in summary_pos)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for l_idx, l_num in enumerate(layers):
            row = f"    {l_num:>3d}  " + "  ".join(f"{r2[l_idx, p]:.4f}" for p in summary_pos)
            print(row)


def main():
    t0 = time.time()

    experiments = [
        ("train_1afaf1150ad2494d2e61b031ea3d54b5", "V=8, T=192, k=10", range(6), "natural_r2_major_vs_all_v8.png"),
        ("train_6e9df91166eb0ad0e4efbd1fb42b4854", "V=6, T=384, k=10", range(6), "natural_r2_major_vs_all_v6.png"),
    ]

    for exp_name, label, layers, save_name in experiments:
        r2_all, r2_major, r2_minor, T, V, n_major, layers_used = run_one_experiment(
            exp_name, label, layers,
        )
        plot_comparison(r2_all, r2_major, r2_minor, T, layers_used, label, save_name)

        print(f"\n{'='*60}")
        print(f"  {label} (n_major={n_major})")
        print(f"{'='*60}")
        print_summary(r2_all, r2_major, r2_minor, T, layers_used, label, n_major)
        print(f"{'='*60}")

    log.info(f"\nAll done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
