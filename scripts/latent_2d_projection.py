"""2D projection of latent-conditioned mean hidden states (empirical-unigram view).

Idea: for each underlying latent (task), average the hidden state (layer L) over
many sequences drawn from that latent, at a set of selected positions. Stack all
those latent-conditioned means and project them onto a 2D plane (PCA). Plotting the
selected positions *together* shows how the model's representation of the latent
moves with context and how the latents are arranged.

The colouring tests the "empirical unigram" hypothesis: each point is coloured by
the latent's unigram statistic (the coin bias for vocab=2; the leading PC of the
per-latent unigram distribution otherwise). If the model infers the latent by
tracking token frequencies, the latent-means lie on a smooth manifold ordered by
that unigram colour, sharpening (spreading out) as more context (later positions)
accumulates.

Usage:
  uv run python scripts/latent_2d_projection.py --manifest results/coin_grid_manifest.json \
      --task coin --positions 1 4 16 64 127 --max-k 64
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def collect(exp_name, task, layer, positions, B, max_k):
    import icl.utils.notebook_utils as nu
    from icl.models.hidden_extractor import extract_layer_hiddens

    _, sampler, config = nu.load_everything(task, exp_name)
    dev = config.device
    model, _ = nu.load_checkpoint(config, exp_name=exp_name, return_actual_step=True)
    model.eval().to(dev)
    if hasattr(sampler, "to"):
        sampler.to(dev)
    K = min(int(sampler.n_major_tasks), max_k)
    L = layer if layer is not None else config.model.num_layers - 1
    V = int(config.vocab_size)

    means = np.empty((K, len(positions), model_hidden_dim(model, config)), dtype=np.float32)
    unigram = np.zeros((K, V), dtype=np.float64)
    for k in range(K):
        out = sampler.generate(mode="major", task=k, num_samples=B)
        x = (out[0] if isinstance(out, (tuple, list)) else out)
        if x.dim() == 3:
            x = x.squeeze(0)
        x = x.to(dev)
        h = extract_layer_hiddens(model, x)[L]                 # (B, T, D)
        means[k] = h[:, positions, :].mean(0).cpu().numpy()    # (P, D)
        unigram[k] = torch.bincount(x.reshape(-1), minlength=V).float().cpu().numpy()
    unigram /= unigram.sum(1, keepdims=True)
    return means, unigram, K, V


def model_hidden_dim(model, config):
    return int(getattr(model, "hidden_size", None) or config.model.emb_dim)


def unigram_color(unigram, V):
    """Scalar colour per latent from its unigram: the bias for V=2, else leading PC."""
    if V == 2:
        return unigram[:, 0]                                   # P(token 0) = coin bias
    u = unigram - unigram.mean(0)
    _, _, Vt = np.linalg.svd(u, full_matrices=False)
    return u @ Vt[0]


def project_and_plot(means, color, positions, out_path, title):
    K, P, D = means.shape
    flat = means.reshape(K * P, D)
    flat_c = flat - flat.mean(0)
    _, _, Vt = np.linalg.svd(flat_c, full_matrices=False)      # PCA
    proj = (flat_c @ Vt[:2].T).reshape(K, P, 2)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    cmap = plt.get_cmap("viridis")
    cn = plt.Normalize(color.min(), color.max())
    # marker size grows with position so the context direction is visible.
    sizes = 20 + 130 * (np.arange(P) / max(1, P - 1))
    for k in range(K):
        c = cmap(cn(color[k]))
        ax.plot(proj[k, :, 0], proj[k, :, 1], "-", color=c, lw=0.8, alpha=0.5)
        ax.scatter(proj[k, :, 0], proj[k, :, 1], s=sizes, color=c,
                   edgecolors="k", linewidths=0.3, zorder=3)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_title(title)
    ax.grid(True, alpha=0.3)
    sm = plt.cm.ScalarMappable(norm=cn, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=ax, label="latent unigram (bias / leading PC)")
    ax.text(0.02, 0.98, f"positions {positions}\n(marker size grows with position)",
            transform=ax.transAxes, va="top", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--exp", default=None, help="single exp_name (instead of a manifest)")
    ap.add_argument("--task", default="coin", choices=["latent", "coin"])
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--positions", nargs="+", type=int, default=[1, 4, 16, 64, 127])
    ap.add_argument("--B", type=int, default=512)
    ap.add_argument("--max-k", type=int, default=1024)
    ap.add_argument("--out-dir", default="results/latent_2d_projection")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.exp:
        cells = [{"arch": "model", "exp_name": args.exp, "n_tasks": "?"}]
    else:
        cells = [r for r in json.load(open(args.manifest))["runs"] if r.get("trained", True)]

    for r in cells:
        arch, K_, exp = r["arch"], r.get("n_tasks", "?"), r["exp_name"]
        means, unigram, K, V = collect(exp, args.task, args.layer, args.positions, args.B, args.max_k)
        color = unigram_color(unigram, V)
        out = os.path.join(args.out_dir, f"{arch}_K{K_}_proj2d.png")
        project_and_plot(means, color, args.positions, out,
                         f"{args.task} {arch} K={K_}: latent-conditioned means (layer)")
        np.savez(os.path.join(args.out_dir, f"{arch}_K{K_}_proj2d.npz"),
                 means=means, unigram=unigram, positions=np.array(args.positions))
        print(f"[done] {arch} K={K_}  K_latents={K}  -> {out}")


if __name__ == "__main__":
    main()
