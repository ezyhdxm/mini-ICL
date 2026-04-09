#!/usr/bin/env python3
"""Task-vector R² (ANOVA) for the latent k=10 vocab=6 long experiment.

Uses the newly trained model: train_6e9df91166eb0ad0e4efbd1fb42b4854

Usage (from project root):
    PYTHONPATH=src python scripts/fig_anova_latent_k10_v6.py
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

EXP_NAME = "train_6e9df91166eb0ad0e4efbd1fb42b4854"
N_MINOR = 30
BATCH_SIZE = 64


def main():
    t0 = time.time()

    from icl.latent_markov.analysis import plot_task_vector_r2_latent

    log.info(f"exp_name = {EXP_NAME}")
    log.info(f"n_minor={N_MINOR}, batch_size={BATCH_SIZE}")

    out = plot_task_vector_r2_latent(
        EXP_NAME,
        layers=range(6),
        batch_size=BATCH_SIZE,
        n_minor=N_MINOR,
        print_summary=True,
        log_x=False,
        show=False,
        show_ylabel=True,
        extraction_point="post_mlp",
    )
    fig = out["fig"]
    path = SAVE_DIR / "task_vector_r2_latent_k10_v6_long.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close("all")
    log.info(f"[interventional] saved → {path}")

    # Also run natural-sequence R² for comparison
    log.info("Running natural-sequence R² ...")
    import gc
    import torch
    import numpy as np
    import icl.utils.notebook_utils as nu
    from icl.latent_markov.analysis.ood import get_latent_sampler
    from icl.coin.analysis._helpers import extract_hidden_multi_coin_latent
    from icl.utils.separability._task_vector_r2 import _layer_style

    _, _, config = nu.load_everything("latent", EXP_NAME)
    step = config.training.num_epochs
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=EXP_NAME, return_actual_step=True,
    )
    model.eval().to(config.device)

    sampler, _, _ = get_latent_sampler(EXP_NAME, N_MINOR, n_ood=0)
    device = config.device
    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    seq_len = sampler.seq_len
    T = seq_len - 1
    V = int(sampler.num_states)
    layers = list(range(6))
    SAMPLES_PER_TASK = 256
    TASK_BATCH = 32

    task_pos = torch.arange(T, device=device)
    all_hiddens = []
    all_tokens = []
    all_task_ids = []

    for k in range(n_tasks):
        if k % 10 == 0:
            log.info(f"  natural: task {k}/{n_tasks}")
        parts_h, parts_t = [], []
        for b0 in range(0, SAMPLES_PER_TASK, TASK_BATCH):
            bs = min(b0 + TASK_BATCH, SAMPLES_PER_TASK) - b0
            if k < sampler.n_major_tasks:
                gen = sampler.generate(mode="testing", task=k, num_samples=bs)
            else:
                gen = sampler.generate(mode="minor", task=k - sampler.n_major_tasks, num_samples=bs)
            samples = gen[0] if isinstance(gen, (tuple, list)) else gen
            if samples.dim() == 3:
                samples = samples.squeeze(0)
            samples = samples.to(device)
            parts_t.append(samples[:, :T].cpu())
            h = extract_hidden_multi_coin_latent(
                model, samples, layers, task_pos, extraction_point="post_mlp",
            )
            parts_h.append(h.cpu())
            del samples, h
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        all_hiddens.append(torch.cat(parts_h, dim=1))
        all_tokens.append(torch.cat(parts_t, dim=0))
        all_task_ids.append(torch.full((SAMPLES_PER_TASK,), k, dtype=torch.long))

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    hiddens = torch.cat(all_hiddens, dim=1)
    tokens = torch.cat(all_tokens, dim=0)
    task_ids = torch.cat(all_task_ids, dim=0)

    L = hiddens.shape[0]
    r2 = np.zeros((L, T))
    for l in range(L):
        for t in range(T):
            h_lt = hiddens[l, :, t, :].float()
            cell_key = task_ids.long() * V + tokens[:, t].long()
            grand_mean = h_lt.mean(dim=0)
            ss_total = ((h_lt - grand_mean) ** 2).sum().item()
            ss_within = 0.0
            for c in cell_key.unique():
                h_cell = h_lt[cell_key == c]
                if h_cell.shape[0] < 1:
                    continue
                ss_within += ((h_cell - h_cell.mean(dim=0)) ** 2).sum().item()
            r2[l, t] = 1.0 - ss_within / (ss_total + 1e-10)

    positions = np.arange(T)
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    ax = axes[0]
    for l_idx, l_num in enumerate(layers):
        ax.plot(positions, 1.0 - r2[l_idx], label=str(l_num),
                **_layer_style(l_num, len(positions)))
    ax.set_xlabel("Position", fontsize=13)
    ax.set_ylabel("Residual variance ratio $(1-R^2)$", fontsize=13)
    ax.set_title("Natural sequences", fontsize=14)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=12)
    ax.legend(title="Layer", fontsize=10, title_fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    for l_idx, l_num in enumerate(layers):
        ax2.plot(positions, r2[l_idx], label=str(l_num),
                 **_layer_style(l_num, len(positions)))
    ax2.set_xlabel("Position", fontsize=13)
    ax2.set_ylabel("$R^2$", fontsize=13)
    ax2.set_title("Natural sequences — $R^2$", fontsize=14)
    ax2.set_ylim(-0.02, 1.02)
    ax2.tick_params(labelsize=12)
    ax2.legend(title="Layer", fontsize=10, title_fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.25)

    fig2.tight_layout()
    path2 = SAVE_DIR / "task_vector_r2_latent_k10_v6_long_natural.png"
    fig2.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    log.info(f"[natural] saved → {path2}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Natural-sequence R² (V=6, seq_len={seq_len}, {n_tasks} tasks)")
    print(f"{'='*60}")
    summary_pos = [0, 5, 10, 20, 50, 100, 200, T - 1]
    summary_pos = [p for p in summary_pos if p < T]
    header = "Layer  " + "  ".join(f"t={p:>3d}" for p in summary_pos)
    print(header)
    print("-" * len(header))
    for l_idx, l_num in enumerate(layers):
        row = f"  {l_num:>3d}  " + "  ".join(f"{r2[l_idx, p]:.4f}" for p in summary_pos)
        print(row)
    print(f"{'='*60}\n")

    log.info(f"All done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
