"""Coin (real-coin, vocab=2) hidden-state task-vector projection across architectures.

For each trained coin cell, run the averaging-based task-subspace projection
(beta(t) = hidden state projected on the task vectors, vs the Bayesian posterior
alpha(t)) and save:
  <out-dir>/<arch>_coin/projection.png + .npz   (per architecture, all layers)
  <out-dir>/combined_coin_projection.png         (each arch's beta_0 vs shared alpha_0)

Train the cells first with, e.g.:
  uv run python scripts/train_arch_grid.py --task coin --vocab-size 2 \
      --arch transformer rnn lstm mamba --n-tasks 3 \
      --manifest results/coin_grid_manifest.json
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARCH_COLORS = {"transformer": "#1f77b4", "lstm": "#ff7f0e",
               "rnn": "#2ca02c", "mamba": "#9467bd"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="results/coin_grid_manifest.json")
    ap.add_argument("--out-dir", default="results/coin_analysis")
    ap.add_argument("--layer", type=int, default=None, help="layer index (default: last)")
    ap.add_argument("--B", type=int, default=512, help="eval batch size (tighter error bars)")
    args = ap.parse_args()

    from icl.coin.analysis.trajectory import traj_averaging_projection_plot_coin

    os.makedirs(args.out_dir, exist_ok=True)
    runs = [r for r in json.load(open(args.manifest))["runs"] if r.get("trained", True)]
    print(f"Coin task-vector projection for {len(runs)} cells")

    data = {}
    for r in runs:
        arch, exp = r["arch"], r["exp_name"]
        out = traj_averaging_projection_plot_coin(
            exp, task_ids=[0], B=args.B, layer_index=args.layer,
            project_beta_simplex=True, extraction_point="post_mlp", show=False,
        )
        res = out["results_by_task"][0]
        beta, post = np.asarray(res["beta"]), np.asarray(res["post"])
        cell = os.path.join(args.out_dir, f"{arch}_coin")
        os.makedirs(cell, exist_ok=True)
        if out.get("fig") is not None:
            out["fig"].savefig(os.path.join(cell, "projection.png"), bbox_inches="tight", dpi=130)
            plt.close(out["fig"])
        np.savez(os.path.join(cell, "projection.npz"), beta=beta, post=post)
        data[arch] = (beta, post)
        print(f"[done] {arch}  beta {beta.shape}")

    # Combined: each arch's beta_0 vs the shared Bayesian alpha_0.
    fig, ax = plt.subplots(figsize=(6.5, 4))
    drawn = False
    for arch, (beta, post) in data.items():
        bm, am = beta.mean(0), post.mean(0)
        x = np.arange(bm.shape[0])
        if not drawn:
            ax.plot(x, am[:, 0], "k--", lw=2, label=r"$\alpha_0$ (Bayes)")
            drawn = True
        ax.plot(x, bm[:, 0], marker="o", ms=2, lw=1.0,
                color=ARCH_COLORS.get(arch, "k"), label=arch)
    ax.set_xlabel("Position $t$")
    ax.set_ylabel(r"$\beta_0$ (markers) / $\alpha_0$ (dashed)")
    ax.set_title("Coin (vocab=2): task-vector coefficient vs. Bayesian posterior")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    out_path = os.path.join(args.out_dir, "combined_coin_projection.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
