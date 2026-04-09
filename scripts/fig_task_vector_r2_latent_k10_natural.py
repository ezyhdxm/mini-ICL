#!/usr/bin/env python3
"""Task-vector R² for latent k=10 using NATURAL sequences (no intervention).

Instead of fixing the token at each position (interventional), we generate
natural Markov chain sequences per task and compute the cell-means R²
using the naturally occurring (task, token_at_t) as the cell identity.

Usage (from project root):
    PYTHONPATH=src python scripts/fig_task_vector_r2_latent_k10_natural.py
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

import icl.utils.notebook_utils as nu
from icl.utils.unified_interface import get_exp_name
from icl.latent_markov.analysis.ood import get_latent_sampler
from icl.coin.analysis._helpers import extract_hidden_multi_coin_latent


K_EXP = 10
N_MINOR = 30
SAMPLES_PER_TASK = 256
TASK_BATCH = 32


def collect_natural_hiddens(exp_name, layers, extraction_point="post_mlp"):
    """Generate natural sequences per task, extract hiddens at all positions."""
    _, _, config = nu.load_everything("latent", exp_name)
    step = config.training.num_epochs
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)

    sampler, k_minor, _ = get_latent_sampler(exp_name, N_MINOR, n_ood=0)
    device = config.device

    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    seq_len = sampler.seq_len
    T = seq_len - 1  # extractable positions: 0 .. seq_len-2
    V = int(sampler.num_states)
    L = len(layers)
    D = config.model.emb_dim

    log.info(
        f"  n_tasks={n_tasks}, seq_len={seq_len}, T={T}, V={V}, "
        f"samples_per_task={SAMPLES_PER_TASK}, layers={list(layers)}"
    )

    task_pos = torch.arange(T, device=device)

    all_hiddens = []  # will be list of (L, B, T, D) tensors
    all_tokens = []   # will be list of (B, T) tensors
    all_task_ids = []  # will be list of (B,) tensors

    for k in range(n_tasks):
        log.info(f"  task {k}/{n_tasks} ...")
        task_hiddens_parts = []
        task_tokens_parts = []

        for b_start in range(0, SAMPLES_PER_TASK, TASK_BATCH):
            b_end = min(b_start + TASK_BATCH, SAMPLES_PER_TASK)
            b_size = b_end - b_start

            if k < sampler.n_major_tasks:
                gen_out = sampler.generate(
                    mode="testing", task=k, num_samples=b_size,
                )
            else:
                gen_out = sampler.generate(
                    mode="minor", task=k - sampler.n_major_tasks,
                    num_samples=b_size,
                )
            samples = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
            if samples.dim() == 3:
                samples = samples.squeeze(0)
            samples = samples.to(device)

            tokens_at_pos = samples[:, :T].cpu()  # (b_size, T)
            task_tokens_parts.append(tokens_at_pos)

            h = extract_hidden_multi_coin_latent(
                model=model,
                batch_data=samples,
                layers=list(layers),
                task_pos=task_pos,
                extraction_point=extraction_point,
            )  # (L, b_size, T, D)
            task_hiddens_parts.append(h.cpu())

            del samples, h
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        task_h = torch.cat(task_hiddens_parts, dim=1)  # (L, SAMPLES_PER_TASK, T, D)
        task_tok = torch.cat(task_tokens_parts, dim=0)  # (SAMPLES_PER_TASK, T)

        all_hiddens.append(task_h)
        all_tokens.append(task_tok)
        all_task_ids.append(torch.full((task_h.shape[1],), k, dtype=torch.long))

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Stack: hiddens (L, n_tasks*B, T, D), tokens (n_tasks*B, T), ids (n_tasks*B,)
    hiddens = torch.cat(all_hiddens, dim=1)
    tokens = torch.cat(all_tokens, dim=0)
    task_ids = torch.cat(all_task_ids, dim=0)

    return hiddens, tokens, task_ids, T, V, list(layers)


def compute_natural_r2(hiddens, tokens, task_ids, T, V, n_tasks, eps=1e-10):
    """Compute cell-means R² using naturally occurring (task, token) cells.

    For each (layer, position), group samples by (task_id, token_value),
    compute cell means, and compute 1 - SS_within / SS_total.
    """
    L = hiddens.shape[0]
    N = hiddens.shape[1]
    D = hiddens.shape[3]

    r2_array = np.zeros((L, T))

    for l in range(L):
        for t in range(T):
            h_lt = hiddens[l, :, t, :].float()  # (N, D)
            tok_t = tokens[:, t].long()          # (N,)
            tid = task_ids.long()                # (N,)

            grand_mean = h_lt.mean(dim=0)
            ss_total = ((h_lt - grand_mean) ** 2).sum().item()

            # Cell key = task_id * V + token_value
            cell_key = tid * V + tok_t  # (N,)
            unique_cells = cell_key.unique()

            ss_within = 0.0
            for c in unique_cells:
                mask = (cell_key == c)
                h_cell = h_lt[mask]
                if h_cell.shape[0] < 1:
                    continue
                cell_mean = h_cell.mean(dim=0)
                ss_within += ((h_cell - cell_mean) ** 2).sum().item()

            r2_array[l, t] = 1.0 - ss_within / (ss_total + eps)

    return r2_array


def main():
    t0 = time.time()
    exp_name = get_exp_name("latent", K_EXP)
    log.info(f"exp_name = {exp_name}")

    layers = list(range(6))

    log.info("Collecting natural-sequence hiddens ...")
    hiddens, tokens, task_ids, T, V, layers_used = collect_natural_hiddens(
        exp_name, layers, extraction_point="post_mlp",
    )
    n_tasks = int(task_ids.max().item()) + 1
    log.info(
        f"hiddens shape: {hiddens.shape}, tokens shape: {tokens.shape}, "
        f"n_tasks={n_tasks}, V={V}, T={T}"
    )

    log.info("Computing natural-sequence R² ...")
    r2 = compute_natural_r2(hiddens, tokens, task_ids, T, V, n_tasks)

    # Also compute the interventional R² data for comparison if cached
    log.info("Plotting ...")
    from icl.utils.separability._task_vector_r2 import _layer_style

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    positions = np.arange(T)

    # Left panel: natural R²  (plot 1 - R²)
    ax = axes[0]
    for l_idx, l_num in enumerate(layers_used):
        vals = 1.0 - r2[l_idx, :]
        ax.plot(positions, vals, label=str(l_num),
                **_layer_style(l_num, len(positions)))
    ax.set_xlabel("Position", fontsize=13)
    ax.set_ylabel("Residual variance ratio  $(1 - R^2)$", fontsize=13)
    ax.set_title("Natural sequences", fontsize=14)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=12)
    ax.legend(title="Layer", fontsize=10, title_fontsize=10,
              framealpha=0.9, loc="best")
    ax.grid(True, alpha=0.25)

    # Right panel: natural R² (plot R² directly)
    ax2 = axes[1]
    for l_idx, l_num in enumerate(layers_used):
        ax2.plot(positions, r2[l_idx, :], label=str(l_num),
                 **_layer_style(l_num, len(positions)))
    ax2.set_xlabel("Position", fontsize=13)
    ax2.set_ylabel("$R^2$", fontsize=13)
    ax2.set_title("Natural sequences — $R^2$", fontsize=14)
    ax2.set_ylim(-0.02, 1.02)
    ax2.tick_params(labelsize=12)
    ax2.legend(title="Layer", fontsize=10, title_fontsize=10,
               framealpha=0.9, loc="best")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    save_path = SAVE_DIR / f"task_vector_r2_latent_k{K_EXP}_natural.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {save_path}")

    # Print summary at a few positions
    print(f"\n{'='*60}")
    print(f"  Natural-sequence R² summary (k={K_EXP}, {n_tasks} tasks, V={V})")
    print(f"  {SAMPLES_PER_TASK} samples/task = {n_tasks * SAMPLES_PER_TASK} total")
    print(f"{'='*60}")
    summary_pos = [0, 5, 10, 20, 50, 100, T - 1]
    summary_pos = [p for p in summary_pos if p < T]
    header = "Layer  " + "  ".join(f"t={p:>3d}" for p in summary_pos)
    print(header)
    print("-" * len(header))
    for l_idx, l_num in enumerate(layers_used):
        row = f"  {l_num:>3d}  " + "  ".join(
            f"{r2[l_idx, p]:.4f}" for p in summary_pos
        )
        print(row)
    print(f"{'='*60}")

    log.info(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
