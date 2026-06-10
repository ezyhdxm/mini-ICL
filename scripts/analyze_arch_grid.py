"""Run the analysis pipeline over a trained architecture grid and collect results.

Reads the manifest written by train_arch_grid.py and, for each trained (arch, K)
cell, runs the three latent-Markov analyses headlessly and saves their outputs:

  1.   conditional variance vs context      -> plot_p1_variance
  2.   Figure 3 (Bayesian posterior alignment): beta(t) task-vector coefficients
       vs Bayesian posterior alpha(t)        -> traj_averaging_projection_plot
  3.   token-conditioned task-vector R²      -> plot_task_vector_r2_latent

Per cell we write figures (.png) and numeric arrays (.npz) under
<out-dir>/<arch>_K<K>/, plus a combined summary.json. Each analysis is wrapped
so one failure doesn't abort the sweep; errors are recorded in the summary.

Usage:
  uv run python scripts/analyze_arch_grid.py
  uv run python scripts/analyze_arch_grid.py --positions 0 32 64 128   # subsample
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np


def _to_np(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _save_numeric(result: dict, npz_path: str):
    """Save the numeric (non-figure) entries of a result dict to a .npz."""
    arrays = {}
    for k, v in result.items():
        if k in ("fig", "ax") or v is None:
            continue
        try:
            arr = _to_np(v)
            if arr.dtype != object:
                arrays[k] = arr
        except Exception:
            pass  # skip anything that isn't cleanly array-able
    np.savez(npz_path, **arrays)
    return sorted(arrays)


def _save_fig(result: dict, png_path: str):
    fig = result.get("fig")
    if fig is not None:
        fig.savefig(png_path, bbox_inches="tight", dpi=120)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="results/arch_grid_manifest.json")
    ap.add_argument("--out-dir", default="results/arch_grid_analysis")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--positions", nargs="+", type=int, default=None,
                    help="position indices to analyse (default: all)")
    ap.add_argument("--max-lambda-k", type=int, default=64,
                    help="skip the lambda/interpolation analysis when n_tasks exceeds "
                         "this (it is a small-K analysis; dense over K is infeasible at K=1024)")
    args = ap.parse_args()

    from icl.latent_markov.analysis.variance import (
        plot_p1_variance, plot_task_vector_r2_latent,
    )
    from icl.latent_markov.analysis import traj_averaging_projection_plot

    pos = args.positions

    def _beta_alpha_fig3(exp):
        """Figure 3 (Bayesian posterior alignment): beta(t) task-vector coefficients
        vs the Bayesian posterior alpha(t). extraction_point='post_mlp' is the
        cross-architecture (full-layer-output) extraction point."""
        out = traj_averaging_projection_plot(
            exp, task_ids=[0], B=args.batch_size, plot_positions=pos,
            per_position_mean=True, project_beta_simplex=True,
            beta_errbar="quantile", extraction_point="post_mlp", show=False,
        )
        # Flatten results_by_task (beta = model coeffs, post = Bayesian posterior).
        flat = {k: out.get(k) for k in ("fig", "task_vecs", "grand_mean")}
        for tid, r in out.get("results_by_task", {}).items():
            flat[f"task{tid}_beta"] = r["beta"]
            flat[f"task{tid}_post"] = r["post"]
        return flat

    # (name, fn, max_k): max_k=None means run for any K; otherwise skip when n_tasks > max_k.
    analyses = [
        ("p1_variance",
         lambda e: plot_p1_variance(e, batch_size=args.batch_size,
                                    positions_of_interest=pos, show=False), None),
        ("beta_alpha_fig3", _beta_alpha_fig3, args.max_lambda_k),
        ("task_vector_r2",
         lambda e: plot_task_vector_r2_latent(e, batch_size=args.batch_size,
                                              positions_of_interest=pos, show=False), None),
    ]

    with open(args.manifest) as f:
        manifest = json.load(f)
    runs = [r for r in manifest["runs"] if r.get("trained", True)]
    print(f"Analyzing {len(runs)} trained cells from {args.manifest}")

    summary = []
    for run in runs:
        arch, K, exp = run["arch"], run["n_tasks"], run["exp_name"]
        cell_dir = os.path.join(args.out_dir, f"{arch}_K{K}")
        os.makedirs(cell_dir, exist_ok=True)
        cell = {"arch": arch, "n_tasks": K, "exp_name": exp}

        for name, fn, max_k in analyses:
            if max_k is not None and K > max_k:
                cell[name] = {"ok": False, "skipped": f"n_tasks={K} > max_lambda_k={max_k}"}
                continue
            try:
                res = fn(exp)
                keys = _save_numeric(res, os.path.join(cell_dir, f"{name}.npz"))
                _save_fig(res, os.path.join(cell_dir, f"{name}.png"))
                cell[name] = {"ok": True, "saved_keys": keys}
            except Exception as e:  # noqa: BLE001 - record and continue
                cell[name] = {"ok": False, "error": repr(e)[:300]}
                print(f"  [{arch} K={K}] {name} FAILED: {e}")

        summary.append(cell)
        print(f"[done] {arch} K={K} -> {cell_dir}")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nAnalysis complete: {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
