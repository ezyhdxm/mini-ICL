"""Out-of-distribution performance: in-context next-token loss on OOD sequences.

OOD sequences are generated from fresh order-1 transition matrices drawn from the
Dirichlet(alpha) prior -- tasks the model never trained on. We compare, per
context position:

  - model NLL            : -log p_model(x_{t+1} | x_{0:t})
  - Dirichlet-optimal NLL: the Bayes-optimal predictor for this generative
                           process -- online Dirichlet posterior predictive
                           p(j | i, counts) = (counts[i,j] + alpha) / (n_i + V*alpha)
  - uniform baseline     : log V

A model that generalises (extrapolative mode) tracks the Dirichlet-optimal curve;
one that only retrieves trained tasks plateaus above it. The K=4 vs K=1024
contrast is the two-modes test.

Writes per-cell npz/png and a combined figure (model curves grouped by K, vs the
shared Dirichlet-optimal) to <out-dir>.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ARCH_STYLE = {"transformer": "-", "lstm": "--", "rnn": ":"}
K_COLOR = {4: "#d62728", 1024: "#1f77b4"}


@torch.no_grad()
def _dirichlet_optimal_nll(samples, V, alpha):
    """Per-position NLL of the online Dirichlet posterior-predictive (order-1)."""
    B, T = samples.shape
    dev = samples.device
    counts = torch.zeros(B, V, V, device=dev)
    bidx = torch.arange(B, device=dev)
    nll = torch.empty(T - 1, device=dev)
    for t in range(T - 1):
        cur, nxt = samples[:, t], samples[:, t + 1]
        row = counts[bidx, cur]                                  # (B, V) counts seen so far
        p = (row + alpha) / (row.sum(-1, keepdim=True) + V * alpha)
        nll[t] = -torch.log(p[bidx, nxt].clamp_min(1e-12)).mean()
        counts[bidx, cur, nxt] += 1.0                            # observe the transition
    return nll.cpu().numpy()


@torch.no_grad()
def _model_nll(model, samples, V):
    logits = model(samples)
    p = torch.softmax(logits, dim=-1)[..., :V]
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
    B, T = samples.shape
    bidx = torch.arange(B, device=samples.device)
    nll = torch.empty(T - 1, device=samples.device)
    for t in range(T - 1):
        nll[t] = -torch.log(p[bidx, t, samples[:, t + 1]].clamp_min(1e-12)).mean()
    return nll.cpu().numpy()


def _ood_for_cell(exp_name, num_samples, seed_tag):
    import icl.utils.notebook_utils as nu
    _, sampler, config = nu.load_everything("latent", exp_name)
    dev = config.device
    model, _ = nu.load_checkpoint(config, exp_name=exp_name, return_actual_step=True)
    model.eval().to(dev)
    if hasattr(sampler, "to"):
        sampler.to(dev)
    V = int(sampler.num_states)
    alpha = float(getattr(sampler, "alpha", 1.0))  # Dirichlet prior used by the OOD generator
    out = sampler.generate(mode="ood", num_samples=num_samples, epochs=1)
    samples = (out[0] if isinstance(out, (tuple, list)) else out)
    if samples.dim() == 3:
        samples = samples.squeeze(0)
    samples = samples.to(dev)
    model_nll = _model_nll(model, samples, V)
    opt_nll = _dirichlet_optimal_nll(samples, V, alpha)
    uniform = float(np.log(V))
    return {"model_nll": model_nll, "optimal_nll": opt_nll,
            "uniform": uniform, "V": V, "alpha": alpha}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="results/arch_grid_manifest.json")
    ap.add_argument("--out-dir", default="results/arch_grid_analysis/ood_loss")
    ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--ks", nargs="+", type=int, default=[4, 1024])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.manifest) as f:
        runs = [r for r in json.load(f)["runs"]
                if r.get("trained", True) and r["n_tasks"] in args.ks]

    results = {}
    for r in runs:
        arch, K, exp = r["arch"], r["n_tasks"], r["exp_name"]
        print(f"[ood] {arch} K={K} ...", flush=True)
        res = _ood_for_cell(exp, args.num_samples, f"{arch}_{K}")
        results[(arch, K)] = res
        np.savez(os.path.join(args.out_dir, f"{arch}_K{K}_ood_loss.npz"), **res)

    # Combined figure: one subplot per K; model curves per arch vs Dirichlet-optimal.
    fig, axes = plt.subplots(1, len(args.ks), figsize=(6 * len(args.ks), 4.2), squeeze=False)
    for ci, K in enumerate(args.ks):
        ax = axes[0][ci]
        opt_drawn = False
        for arch in ["transformer", "lstm", "rnn"]:
            res = results.get((arch, K))
            if res is None:
                continue
            x = np.arange(len(res["model_nll"]))
            if not opt_drawn:
                ax.plot(x, res["optimal_nll"], color="k", lw=2.2,
                        label="Dirichlet-optimal (Bayes)")
                ax.axhline(res["uniform"], color="gray", ls=":", lw=1, label="uniform")
                opt_drawn = True
            ax.plot(x, res["model_nll"], ARCH_STYLE[arch], color=K_COLOR[K], lw=1.6,
                    label=f"{arch}")
        ax.set_title(f"OOD next-token loss  (K={K})")
        ax.set_xlabel("Position $t$")
        ax.set_ylabel("NLL (nats)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(args.out_dir, "combined_ood_loss.png")
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"\nWrote {out} and per-cell npz to {args.out_dir}")


if __name__ == "__main__":
    main()
