#!/usr/bin/env python3
"""Major-vs-minor ANOVA R² for all three experiments (E1, E2, E3) with k=10.

For E1 (coin) and E3 (latent): cell-means R² with factors (task_id, token).
For E2 (linear): ANCOVA R² with task-specific slopes on continuous covariates.

Produces:
  - paper_figs/anova_major_vs_minor_{coin,latent,linear}.png  (3 plots)
  - paper_figs/anova_major_vs_minor_table.tex                 (LaTeX table)
  - Prints summary table to stdout

Usage:
    PYTHONPATH=src python scripts/fig_major_vs_minor_anova.py
"""

import gc
import time
import logging
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = PROJECT_ROOT / "paper_figs"
SAVE_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

import icl.utils.notebook_utils as nu
from icl.coin.analysis._helpers import extract_hidden_multi_coin_latent
from icl.utils.separability._task_vector_r2 import _layer_style
from icl.utils.unified_interface import get_exp_name

K = 10
SAMPLES_PER_TASK = 256
TASK_BATCH = 32
N_MINOR = 30


# ──────────────────────────────────────────────────────────────────────
#  Discrete tasks (E1 coin, E3 latent): natural-sequence cell-means R²
# ──────────────────────────────────────────────────────────────────────

def collect_discrete(task_name, exp_name, layers):
    """Collect natural-sequence hidden states for coin or latent tasks."""
    if task_name == "coin":
        from icl.coin.coin_ood_analysis import get_new_sampler
        _, sampler_orig, config = nu.load_everything("coin", exp_name)
        sampler, _ = get_new_sampler(exp_name, n_minor=N_MINOR, n_ood=0, sampler=sampler_orig)
    else:
        from icl.latent_markov.analysis.ood import get_latent_sampler
        _, _, config = nu.load_everything("latent", exp_name)
        sampler, _, _ = get_latent_sampler(exp_name, N_MINOR, n_ood=0)

    step = config.training.num_epochs
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval().to(config.device)

    device = config.device
    n_major = sampler.n_major_tasks
    n_tasks = n_major + sampler.n_minor_tasks
    T = sampler.seq_len - 1
    V = int(sampler.num_states)

    task_pos = torch.arange(T, device=device)
    all_h, all_tok, all_tid = [], [], []

    for k in range(n_tasks):
        if k % 10 == 0:
            log.info(f"  [{task_name}] task {k}/{n_tasks}")
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
            h = extract_hidden_multi_coin_latent(
                model, samples, list(layers), task_pos, extraction_point="post_mlp",
            )
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


def compute_cellmeans_r2(hiddens, tokens, task_ids, T, V, task_mask=None, eps=1e-10):
    """Cell-means R² per (layer, position) for discrete-token tasks."""
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


# ──────────────────────────────────────────────────────────────────────
#  E2 (Linear) — continuous tokens, ANCOVA R² via existing infrastructure
# ──────────────────────────────────────────────────────────────────────

def run_linear(exp_name, layers):
    """Compute ANCOVA R² for major/minor subsets of the linear task."""
    from icl.linear.analysis.probes._cache import _get_linear_hiddens_cached
    from icl.utils.separability._ancova import ancova_separability

    log.info("  [linear] Loading cached hiddens ...")
    all_hiddens, demo_data, layers_used, n_major, n_ood, k_minor = \
        _get_linear_hiddens_cached(
            exp_name, list(layers), batch_size=64, chunk_size=16,
            step=None, n_minor=N_MINOR, n_ood=0, verbose=True,
            post_layernorm=False, extraction_point="post_mlp",
        )

    L, n_tasks, n_points, B, D = all_hiddens.shape
    n_dims = demo_data.shape[-1]

    log.info(f"  [linear] hiddens shape: {all_hiddens.shape}, "
             f"n_major={n_major}, n_ood={n_ood}, k_minor={k_minor}")

    major_range = (0, n_major)
    minor_start = n_major + n_ood
    minor_range = (minor_start, minor_start + k_minor)

    def _compute_ancova_subset(task_start, task_end):
        n_sub = task_end - task_start
        r2 = np.zeros((L, n_points))
        task_labels = (
            torch.arange(n_sub).unsqueeze(1).expand(n_sub, B).reshape(-1)
        )
        for l_idx in range(L):
            for pos in range(n_points):
                h = all_hiddens[l_idx, task_start:task_end, pos, :, :]  # (n_sub, B, D)
                h_flat = h.reshape(n_sub * B, D)
                x = demo_data[:, pos, :]  # (B, n_dims)
                x_flat = x.unsqueeze(0).expand(n_sub, B, n_dims).reshape(n_sub * B, n_dims)
                res = ancova_separability(h_flat, x_flat, task_labels, eps=1e-10)
                r2[l_idx, pos] = res.r2_full
        return r2

    log.info("  [linear] Computing ANCOVA R² (major) ...")
    r2_major = _compute_ancova_subset(*major_range)
    log.info("  [linear] Computing ANCOVA R² (minor) ...")
    r2_minor = _compute_ancova_subset(*minor_range)

    del all_hiddens, demo_data
    gc.collect()

    return r2_major, r2_minor, n_points, layers_used


# ──────────────────────────────────────────────────────────────────────
#  Plotting and table generation
# ──────────────────────────────────────────────────────────────────────

def plot_major_vs_minor(r2_major, r2_minor, T, layers, exp_label, save_name):
    positions = np.arange(T)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, r2, title in zip(axes, [r2_major, r2_minor], ["Major tasks", "Minor tasks"]):
        for l_idx, l_num in enumerate(layers):
            ax.plot(positions, r2[l_idx], label=str(l_num),
                    **_layer_style(l_num, len(positions)))
        ax.set_xlabel("Position", fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=12)
        ax.legend(title="Layer", fontsize=9, title_fontsize=9,
                  framealpha=0.9, loc="lower right")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("$R^2$", fontsize=13)
    fig.suptitle(f"ANOVA $R^2$: {exp_label}", fontsize=15, y=1.02)
    fig.tight_layout()
    path = SAVE_DIR / save_name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved plot → {path}")


def generate_latex_table(results):
    """Generate LaTeX table of R² at last position for major/minor."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\caption{ANOVA $R^2$ at the final context position for major (memorized) vs.\ minor (unmemorized) tasks across experiments E1--E3 with $K{=}10$.")
    lines.append(r"Higher $R^2$ indicates that the $(z, s_t)$ cell-means decomposition (E1, E3) or the ANCOVA model with task-specific slopes (E2) explains more hidden-state variance.}")
    lines.append(r"\label{tab:major_minor_r2}")

    for r in results:
        exp_label = r["label"]
        exp_desc = r["desc"]
        layers = r["layers"]
        r2_major_last = r["r2_major_last"]
        r2_minor_last = r["r2_minor_last"]

        n_cols = len(layers)
        col_spec = "l " + "c" * n_cols
        lines.append("")
        lines.append(r"\begin{subtable}{\linewidth}")
        lines.append(r"\centering")
        lines.append(r"\caption{" + exp_desc + "}")
        lines.append(r"\begin{tabular}{@{}" + col_spec + r"@{}}")
        lines.append(r"\toprule")
        lines.append(r"& \multicolumn{" + str(n_cols) + r"}{c}{\textbf{Layer}} \\")
        lines.append(r"\cmidrule(lr){2-" + str(n_cols + 1) + "}")
        lines.append("Split & " + " & ".join(str(l) for l in layers) + r" \\")
        lines.append(r"\midrule")

        maj_vals = " & ".join(f"{v:.3f}" for v in r2_major_last)
        min_vals = " & ".join(f"{v:.3f}" for v in r2_minor_last)
        lines.append(f"Major & {maj_vals} \\\\")
        lines.append(f"Minor & {min_vals} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{subtable}")

    lines.append("")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    results = []

    # ── E1 (Coin) ──
    log.info("\n" + "=" * 60)
    log.info("  E1 (Coin) k=10")
    log.info("=" * 60)

    coin_exp = get_exp_name("coin", K, vocab_size=6)
    log.info(f"  exp_name = {coin_exp}")
    layers_coin = list(range(6))

    h, tok, tid, T, V, n_maj = collect_discrete("coin", coin_exp, layers_coin)
    major_mask = tid < n_maj
    minor_mask = tid >= n_maj

    log.info("  Computing R² (major) ...")
    r2_coin_major = compute_cellmeans_r2(h, tok, tid, T, V, task_mask=major_mask)
    log.info("  Computing R² (minor) ...")
    r2_coin_minor = compute_cellmeans_r2(h, tok, tid, T, V, task_mask=minor_mask)

    plot_major_vs_minor(r2_coin_major, r2_coin_minor, T, layers_coin,
                        "E1 (Dice)", "anova_major_vs_minor_coin.png")

    results.append({
        "label": "E1",
        "desc": "E1 (Dice): cell-means $R^2$",
        "layers": layers_coin,
        "r2_major_last": [r2_coin_major[l, -1] for l in range(len(layers_coin))],
        "r2_minor_last": [r2_coin_minor[l, -1] for l in range(len(layers_coin))],
    })
    del h, tok, tid
    gc.collect()

    # ── E3 (Latent Markov) ──
    log.info("\n" + "=" * 60)
    log.info("  E3 (Latent Markov) k=10")
    log.info("=" * 60)

    latent_exp = get_exp_name("latent", K)
    log.info(f"  exp_name = {latent_exp}")
    layers_latent = list(range(6))

    h, tok, tid, T, V, n_maj = collect_discrete("latent", latent_exp, layers_latent)
    major_mask = tid < n_maj
    minor_mask = tid >= n_maj

    log.info("  Computing R² (major) ...")
    r2_latent_major = compute_cellmeans_r2(h, tok, tid, T, V, task_mask=major_mask)
    log.info("  Computing R² (minor) ...")
    r2_latent_minor = compute_cellmeans_r2(h, tok, tid, T, V, task_mask=minor_mask)

    plot_major_vs_minor(r2_latent_major, r2_latent_minor, T, layers_latent,
                        "E3 (Markov)", "anova_major_vs_minor_latent.png")

    results.append({
        "label": "E3",
        "desc": "E3 (Markov): cell-means $R^2$",
        "layers": layers_latent,
        "r2_major_last": [r2_latent_major[l, -1] for l in range(len(layers_latent))],
        "r2_minor_last": [r2_latent_minor[l, -1] for l in range(len(layers_latent))],
    })
    del h, tok, tid
    gc.collect()

    # ── E2 (Linear) ──
    log.info("\n" + "=" * 60)
    log.info("  E2 (Linear) k=10")
    log.info("=" * 60)

    linear_exp = get_exp_name(
        "linear", K,
        n_layer=16, total_steps=30_000,
        warmup_steps=15_000, batch_size=256, max_grad_norm=1.0,
    )
    log.info(f"  exp_name = {linear_exp}")
    layers_linear = list(range(4, 16, 2))

    r2_linear_major, r2_linear_minor, n_points, layers_used = \
        run_linear(linear_exp, layers_linear)

    plot_major_vs_minor(r2_linear_major, r2_linear_minor, n_points, layers_used,
                        "E2 (Linear)", "anova_major_vs_minor_linear.png")

    results.append({
        "label": "E2",
        "desc": "E2 (Linear): ANCOVA $R^2$",
        "layers": layers_used,
        "r2_major_last": [r2_linear_major[l, -1] for l in range(len(layers_used))],
        "r2_minor_last": [r2_linear_minor[l, -1] for l in range(len(layers_used))],
    })
    gc.collect()

    # ── Print and save table ──
    print("\n" + "=" * 70)
    print("  ANOVA R² at last position: Major vs Minor")
    print("=" * 70)
    for r in results:
        print(f"\n  {r['label']} ({r['desc']}):")
        print(f"    {'Layer':>6s}  {'Major':>8s}  {'Minor':>8s}  {'Gap':>8s}")
        print(f"    {'-' * 34}")
        for i, l in enumerate(r["layers"]):
            maj = r["r2_major_last"][i]
            mnr = r["r2_minor_last"][i]
            print(f"    {l:>6d}  {maj:>8.4f}  {mnr:>8.4f}  {maj - mnr:>+8.4f}")

    latex = generate_latex_table(results)
    table_path = SAVE_DIR / "anova_major_vs_minor_table.tex"
    with open(table_path, "w") as f:
        f.write(latex)
    log.info(f"\n  Saved LaTeX table → {table_path}")

    print("\n" + latex)

    log.info(f"\nAll done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
