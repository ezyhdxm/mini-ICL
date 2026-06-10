"""Combine per-architecture analysis curves into side-by-side comparison figures.

Reads the per-cell .npz written by analyze_arch_grid.py
(results/arch_grid_analysis/<arch>_K<K>/<analysis>.npz) and overlays
transformer / lstm / rnn for each analysis and each K, writing to
results/arch_grid_analysis/combined/:

  combined_residual_variance_K{K}.png  -- 1-R^2 vs position (last layer)
  combined_p1_variance_K{K}.png        -- conditional variance vs position (last layer)
  combined_kl_two_bayes_K{K}.png       -- KL(model||baseline) vs position (mode x baseline)
  combined_beta_alpha_K4.png           -- beta_0 (per arch) vs the shared Bayesian alpha_0

Re-plots from the saved metric arrays (no model needed). Cells missing a given
analysis (e.g. beta_alpha skipped at K=1024) are simply omitted from that figure.
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARCHS = ["transformer", "lstm", "rnn"]
ARCH_COLORS = {"transformer": "#1f77b4", "lstm": "#ff7f0e", "rnn": "#2ca02c"}


def _npz(base, arch, K, name):
    p = os.path.join(base, f"{arch}_K{K}", f"{name}.npz")
    return np.load(p) if os.path.exists(p) else None


def _last_layer_r2(d):
    """task_vector_r2: return (positions, 1-R^2 at the last layer)."""
    pat = re.compile(r"r2_results_(\d+)_(\d+)_r2$")
    by_layer = {}
    for k in d.files:
        m = pat.match(k)
        if m:
            by_layer.setdefault(int(m.group(1)), {})[int(m.group(2))] = float(d[k])
    if not by_layer:
        return None
    L = max(by_layer)
    pos = sorted(by_layer[L])
    return np.array(pos), np.array([1.0 - by_layer[L][p] for p in pos])


def _last_layer_p1(d):
    """p1_variance: return (positions, var curve at the last layer)."""
    if "plotting_data_positions" not in d.files:
        return None
    layer_keys = [k for k in d.files if re.fullmatch(r"plotting_data_var_pos_\d+", k)]
    if not layer_keys:
        return None
    L = max(int(k.split("_")[-1]) for k in layer_keys)
    return d["plotting_data_positions"], d[f"plotting_data_var_pos_{L}"]


def _overlay_simple(base, out_dir, K, name, extractor, ylabel, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    any_curve = False
    for arch in ARCHS:
        d = _npz(base, arch, K, name)
        if d is None:
            continue
        res = extractor(d)
        if res is None:
            continue
        x, y = res
        ax.plot(x, y, marker="o", ms=3, lw=1.5, color=ARCH_COLORS[arch], label=arch)
        any_curve = True
    if not any_curve:
        plt.close(fig)
        return None
    ax.set_xlabel("Position $t$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  (K={K})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = os.path.join(out_dir, f"combined_{name}_K{K}.png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return path


def _combine_kl(base, out_dir, K):
    modes, baselines = ["major", "ood"], ["exact", "hybrid"]
    titles = {"exact": "KL(model || exact Bayes)", "hybrid": "KL(model || 3-known+Dirichlet)"}
    fig, axes = plt.subplots(len(modes), len(baselines), figsize=(11, 7), squeeze=False)
    any_curve = False
    for r, mode in enumerate(modes):
        for c, bl in enumerate(baselines):
            ax = axes[r][c]
            for arch in ARCHS:
                d = _npz(base, arch, K, "kl_two_bayes")
                if d is None:
                    continue
                pk, mk = f"results_{mode}_positions", f"results_{mode}_kl_{bl}_mean"
                if pk not in d.files or mk not in d.files:
                    continue
                ax.plot(d[pk], d[mk], lw=1.6, color=ARCH_COLORS[arch], label=arch)
                any_curve = True
            ax.set_title(f"{mode} mode -- {titles[bl]}", fontsize=10)
            ax.set_xlabel("Position $t$")
            ax.set_ylabel("KL")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    if not any_curve:
        plt.close(fig)
        return None
    fig.suptitle(f"Model vs. two Bayesian solutions  (K={K})")
    fig.tight_layout()
    path = os.path.join(out_dir, f"combined_kl_two_bayes_K{K}.png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return path


def _combine_beta_alpha(base, out_dir, K=4, comp=0):
    """beta_0 per arch vs the shared Bayesian posterior alpha_0 (the true task's
    component). alpha is model-independent, so it is drawn once."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    drawn_alpha = False
    any_curve = False
    for arch in ARCHS:
        d = _npz(base, arch, K, "beta_alpha_fig3")
        if d is None or "task0_beta" not in d.files:
            continue
        beta = d["task0_beta"].mean(axis=0)   # (T, Kcomp)
        post = d["task0_post"].mean(axis=0)    # (T, Kcomp)
        T = beta.shape[0]
        x = np.arange(T)
        if not drawn_alpha:
            ax.plot(x, post[:, comp], "k--", lw=2, label=r"$\alpha_0$ (Bayesian posterior)")
            drawn_alpha = True
        ax.plot(x, beta[:, comp], marker="o", ms=3, lw=1.2,
                color=ARCH_COLORS[arch], label=fr"$\beta_0$ {arch}")
        any_curve = True
    if not any_curve:
        plt.close(fig)
        return None
    ax.set_xlabel("Position $t$")
    ax.set_ylabel("Coefficient on true task")
    ax.set_title(f"Task-vector coefficient vs. Bayesian posterior  (K={K})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    path = os.path.join(out_dir, f"combined_beta_alpha_K{K}.png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/arch_grid_analysis")
    ap.add_argument("--ks", nargs="+", type=int, default=[4, 1024])
    args = ap.parse_args()

    out_dir = os.path.join(args.dir, "combined")
    os.makedirs(out_dir, exist_ok=True)
    written = []

    for K in args.ks:
        r = _overlay_simple(args.dir, out_dir, K, "task_vector_r2", _last_layer_r2,
                            "Residual variance ratio ($1-R^2$)",
                            "Residual variance vs context")
        if r:
            written.append(r)
        r = _overlay_simple(args.dir, out_dir, K, "p1_variance", _last_layer_p1,
                            "Conditional variance", "Conditional variance vs context")
        if r:
            written.append(r)
        r = _combine_kl(args.dir, out_dir, K)
        if r:
            written.append(r)

    r = _combine_beta_alpha(args.dir, out_dir, K=4)
    if r:
        written.append(r)

    print(f"Wrote {len(written)} combined figures to {out_dir}:")
    for p in written:
        print("  ", os.path.basename(p))


if __name__ == "__main__":
    main()
