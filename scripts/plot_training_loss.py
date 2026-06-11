"""Plot training and ID/OOD eval loss across the architecture grid.

Reads each cell's results/latent/<exp>/log.json (written by the trainer:
train/loss per step, eval/IDLoss & eval/OODLoss per eval step) and writes
combined figures grouped by (arch, K) to <out-dir>:

  combined_train_loss.png   -- smoothed train loss vs step
  combined_id_ood_loss.png  -- ID and OOD eval loss vs step (two panels)
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARCH_STYLE = {"transformer": "-", "lstm": "--", "rnn": ":"}
K_COLOR = {4: "#d62728", 1024: "#1f77b4"}


def _load_log(work_dir, exp):
    p = os.path.join(work_dir, exp, "log.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _smooth(y, w):
    y = np.asarray(y, dtype=float)
    if len(y) < w or w <= 1:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="valid")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="results/arch_grid_manifest.json")
    ap.add_argument("--out-dir", default="results/arch_grid_analysis/training")
    ap.add_argument("--smooth", type=int, default=200, help="moving-average window for train loss")
    args = ap.parse_args()

    with open(args.manifest) as f:
        man = json.load(f)
    work_dir, runs = man["work_dir"], man["runs"]
    os.makedirs(args.out_dir, exist_ok=True)
    written = []

    # --- training loss ---
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for r in runs:
        log = _load_log(work_dir, r["exp_name"])
        if not log or not log.get("train/loss"):
            continue
        y = _smooth(log["train/loss"], args.smooth)
        x = np.arange(len(y))
        ax.plot(x, y, ARCH_STYLE[r["arch"]], color=K_COLOR[r["n_tasks"]], lw=1.2,
                label=f"{r['arch']} K{r['n_tasks']}")
    ax.set_xlabel("step")
    ax.set_ylabel("train loss (NLL, nats)")
    ax.set_title(f"Training loss (moving avg, w={args.smooth})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    p = os.path.join(args.out_dir, "combined_train_loss.png")
    fig.savefig(p, bbox_inches="tight", dpi=130)
    plt.close(fig)
    written.append(p)

    # --- ID / OOD eval loss ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, key, title in [(axes[0], "eval/IDLoss", "ID eval loss"),
                           (axes[1], "eval/OODLoss", "OOD eval loss")]:
        for r in runs:
            log = _load_log(work_dir, r["exp_name"])
            if not log or not log.get(key):
                continue
            step = log.get("eval/step") or list(range(len(log[key])))
            n = min(len(step), len(log[key]))
            ax.plot(step[:n], log[key][:n], ARCH_STYLE[r["arch"]], color=K_COLOR[r["n_tasks"]],
                    lw=1.3, label=f"{r['arch']} K{r['n_tasks']}")
        ax.set_xlabel("step")
        ax.set_ylabel("loss (NLL, nats)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("In-distribution vs out-of-distribution eval loss over training")
    fig.tight_layout()
    p = os.path.join(args.out_dir, "combined_id_ood_loss.png")
    fig.savefig(p, bbox_inches="tight", dpi=130)
    plt.close(fig)
    written.append(p)

    print(f"Wrote {len(written)} figures to {args.out_dir}:")
    for p in written:
        print("  ", os.path.basename(p))


if __name__ == "__main__":
    main()
