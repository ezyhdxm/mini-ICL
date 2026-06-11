"""Launch the architecture x num-latents training grid for the latent Markov task.

Experiment design: uniform prior over K latents (equal probability over all
tasks). This maps to the "major-only" setup -- k=-1 gives a negligible minor
pool (n_minor_tasks=1, p_minor=1e-12) and n_tasks=K major tasks sampled
uniformly. We sweep arch in {transformer, lstm, rnn} x K in {4, 1024}.

Writes a manifest (default results/arch_grid_manifest.json) mapping each
(arch, K) cell to its checkpoint exp_name, so the analysis pipeline can locate
every run. exp_name is computed with get_exp_name using `total_steps` (the name
get_exp_name uses for num_epochs); after each run we verify the checkpoint
directory actually exists and record it, so any train/exp_name drift fails loud.

Usage:
  uv run python scripts/train_arch_grid.py                 # full grid
  uv run python scripts/train_arch_grid.py --dry-run       # print exp_names only
  uv run python scripts/train_arch_grid.py --arch lstm --n-tasks 4 --num-epochs 2
"""

import argparse
import json
import os
from itertools import product

ARCHS_DEFAULT = ["transformer", "lstm", "rnn"]
NTASKS_DEFAULT = [4, 1024]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", nargs="+", default=ARCHS_DEFAULT,
                    choices=["transformer", "lstm", "rnn", "mamba"])
    ap.add_argument("--task", default="latent", choices=["latent", "coin"],
                    help="which task to train (default: latent)")
    ap.add_argument("--vocab-size", type=int, default=None,
                    help="vocab size; e.g. 2 for a real coin (default: task default, 8)")
    ap.add_argument("--major-pool-type", default=None,
                    help="coin: e.g. 'maxent' for well-separated coins from --major-means")
    ap.add_argument("--major-means", nargs="+", type=float, default=None,
                    help="coin: target mean (bias) per coin, e.g. 0.1 0.4 0.6 0.9 (avoids "
                         "degenerate near-duplicate coins from random Dirichlet draws)")
    ap.add_argument("--n-tasks", nargs="+", type=int, default=NTASKS_DEFAULT)
    ap.add_argument("--num-epochs", type=int, default=30_000)
    ap.add_argument("--warmup-steps", type=int, default=15_000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--manifest", default="results/arch_grid_manifest.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="print exp_names without training")
    args = ap.parse_args()

    from icl.utils.unified_interface import unified_train
    from icl.utils.unified_path_finder import unified_get_config, get_exp_name

    vocab = args.vocab_size if args.vocab_size is not None else 8
    work_dir = unified_get_config(args.task).work_dir  # e.g. results/latent or results/coin
    cells = list(product(args.arch, args.n_tasks))

    # Merge into an existing manifest so adding an arch doesn't drop earlier cells.
    merged = {}
    if os.path.exists(args.manifest):
        try:
            for e in json.load(open(args.manifest)).get("runs", []):
                merged[(e["arch"], e["n_tasks"])] = e
        except Exception:
            merged = {}

    for i, (arch, K) in enumerate(cells, 1):
        # NOTE: get_exp_name uses `total_steps` where unified_train uses
        # `num_epochs`; both set config.training.num_epochs for the latent task.
        exp_name = get_exp_name(
            args.task, k=-1, n_tasks=K, arch=arch, vocab_size=vocab,
            major_pool_type=args.major_pool_type, major_means=args.major_means,
            total_steps=args.num_epochs, warmup_steps=args.warmup_steps,
        )
        exp_dir = os.path.join(work_dir, exp_name)
        print(f"[{i}/{len(cells)}] arch={arch} K={K} -> {exp_name}", flush=True)
        entry = {"arch": arch, "n_tasks": K, "exp_name": exp_name, "exp_dir": exp_dir}

        if not args.dry_run:
            unified_train(
                args.task, k=-1, n_tasks=K, arch=arch, vocab_size=vocab,
                major_pool_type=args.major_pool_type, major_means=args.major_means,
                num_epochs=args.num_epochs, warmup_steps=args.warmup_steps,
                device=args.device,
            )
            entry["trained"] = os.path.isdir(exp_dir)
            if not entry["trained"]:
                print(f"  WARNING: expected checkpoint dir missing: {exp_dir}\n"
                      f"  (train/get_exp_name params may have drifted)", flush=True)

        merged[(arch, K)] = entry
        # Write the manifest incrementally so a crash mid-grid keeps progress.
        os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
        with open(args.manifest, "w") as f:
            json.dump({"work_dir": work_dir, "runs": list(merged.values())}, f, indent=2)

    print(f"\nManifest written: {args.manifest}  ({len(merged)} cells)", flush=True)


if __name__ == "__main__":
    main()
