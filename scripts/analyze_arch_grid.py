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


# Keys to never serialize: figures, and raw hidden-state tensors (gigabytes).
_SKIP_KEYS = {"fig", "ax", "axes", "all_hiddens", "token_info"}
_MAX_ELEMS = 5_000_000  # backstop so a stray raw-hidden tensor never lands in the npz


def _collect_arrays(obj, prefix: str, out: dict):
    """Recursively collect array-able leaves (the computed metrics live in nested
    dicts like results_dict / r2_results / results[mode])."""
    if obj is None:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _collect_arrays(v, f"{prefix}{k}_", out)
        return
    if hasattr(obj, "_asdict"):  # namedtuple
        _collect_arrays(obj._asdict(), prefix, out)
        return
    if hasattr(obj, "__dataclass_fields__"):  # dataclass (e.g. TaskVectorR2Result)
        import dataclasses
        _collect_arrays(dataclasses.asdict(obj), prefix, out)
        return
    try:
        arr = _to_np(obj)
    except Exception:
        return
    if arr.dtype == object or arr.size == 0 or arr.size > _MAX_ELEMS:
        return
    out[prefix.rstrip("_")] = arr


def _save_numeric(result: dict, npz_path: str):
    """Save the computed metric arrays (not the raw hiddens) to a .npz."""
    arrays = {}
    for k, v in result.items():
        if k in _SKIP_KEYS:
            continue
        _collect_arrays(v, f"{k}_", arrays)
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
    ap.add_argument("--max-tasks", type=int, default=None,
                    help="cap tasks analysed in variance/R2 (e.g. 32) so large-K cells "
                         "(K=1024) are tractable; statistics estimate fine from a subsample")
    ap.add_argument("--beta-layers", nargs="+", type=int, default=None,
                    help="layers for the alpha-beta (Figure 3) panels; default: ~4 spread "
                         "across depth")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only these analyses by name (e.g. beta_alpha_fig3)")
    ap.add_argument("--archs", nargs="+", default=None,
                    help="process only cells of these architectures (e.g. mamba)")
    args = ap.parse_args()

    from icl.latent_markov.analysis.variance import (
        plot_p1_variance, plot_task_vector_r2_latent,
    )
    from icl.latent_markov.analysis import traj_averaging_projection_plot
    from icl.latent_markov.analysis.two_modes_kl import run_two_modes_kl

    pos = args.positions

    def _kl_two_bayes(exp):
        """Model next-token distribution vs the two CORRECT Bayes solutions for the
        uniform-over-K DGP: the known-pool (uniform 1/K over the K majors) and the
        Dirichlet-new (online add-alpha) predictors. KL(model||ref) vs position, for
        major (ID) and ood modes. Arch-agnostic (softmax(model(x)) + predictors)."""
        return run_two_modes_kl(exp, modes=("major", "ood"), num_samples=256, show=False)

    def _beta_alpha_fig3(exp):
        """Figure 3 (Bayesian posterior alignment) at SEVERAL layers: beta(t)
        task-vector coefficients (markers) vs the Bayesian posterior alpha(t)
        (dashed), one panel per layer. extraction_point='post_mlp' is the
        cross-architecture (full-layer-output) point."""
        import icl.utils.notebook_utils as nu
        _, _, cfg = nu.load_everything("latent", exp)
        nL = int(cfg.model.num_layers)
        if args.beta_layers:
            layers = sorted({L for L in args.beta_layers if 0 <= L < nL})
        else:  # ~4 layers spread across depth
            layers = sorted({int(round(f * (nL - 1))) for f in (0.0, 0.34, 0.67, 1.0)})

        per_layer = {}
        for L in layers:
            o = traj_averaging_projection_plot(
                exp, task_ids=[0], B=args.batch_size, plot_positions=pos,
                per_position_mean=True, project_beta_simplex=True,
                beta_errbar="quantile", extraction_point="post_mlp",
                layer_index=L, show=False,
            )
            r = o["results_by_task"][0]
            per_layer[L] = (np.asarray(r["beta"]), np.asarray(r["post"]))

        fig, axes = plt.subplots(1, len(layers), figsize=(3.6 * len(layers), 3.4),
                                 squeeze=False, sharey=True)
        flat = {"fig": fig}
        for i, L in enumerate(layers):
            ax = axes[0][i]
            beta, post = per_layer[L]
            bmean, amean = beta.mean(axis=0), post.mean(axis=0)   # (T, Kc)
            x = np.arange(bmean.shape[0])
            for k in range(bmean.shape[1]):
                col = plt.get_cmap("tab10")(k % 10)
                ax.plot(x, amean[:, k], "--", lw=1.4, color=col)
                ax.plot(x, bmean[:, k], marker="o", ms=2, lw=0.7, color=col)
            ax.set_title(f"layer {L}", fontsize=10)
            ax.set_xlabel("Position $t$")
            ax.grid(True, alpha=0.3)
            flat[f"layer{L}_beta"] = beta
            flat[f"layer{L}_post"] = post
        axes[0][0].set_ylabel(r"$\beta$ (markers) / $\alpha$ (dashed)")
        fig.suptitle("Task-vector coefficients vs. Bayesian posterior, across layers")
        fig.tight_layout()
        return flat

    # (name, fn, max_k): max_k=None means run for any K; otherwise skip when n_tasks > max_k.
    analyses = [
        ("p1_variance",
         lambda e: plot_p1_variance(e, batch_size=args.batch_size,
                                    positions_of_interest=pos, max_tasks=args.max_tasks,
                                    show=False), None),
        ("beta_alpha_fig3", _beta_alpha_fig3, args.max_lambda_k),
        ("task_vector_r2",
         lambda e: plot_task_vector_r2_latent(e, batch_size=args.batch_size,
                                              positions_of_interest=pos, max_tasks=args.max_tasks,
                                              show=False), None),
        ("kl_two_bayes", _kl_two_bayes, None),
    ]
    if args.only:
        analyses = [a for a in analyses if a[0] in set(args.only)]

    with open(args.manifest) as f:
        manifest = json.load(f)
    runs = [r for r in manifest["runs"] if r.get("trained", True)]
    if args.archs:
        runs = [r for r in runs if r["arch"] in set(args.archs)]
    print(f"Analyzing {len(runs)} trained cells from {args.manifest}")

    # Merge into an existing summary so a partial (--only) run doesn't drop entries.
    summary_path = os.path.join(args.out_dir, "summary.json")
    prior = {}
    if os.path.exists(summary_path):
        try:
            for c in json.load(open(summary_path)):
                prior[(c["arch"], c["n_tasks"])] = c
        except Exception:
            prior = {}

    summary = []
    for run in runs:
        arch, K, exp = run["arch"], run["n_tasks"], run["exp_name"]
        cell_dir = os.path.join(args.out_dir, f"{arch}_K{K}")
        os.makedirs(cell_dir, exist_ok=True)
        cell = dict(prior.get((arch, K), {}))
        cell.update({"arch": arch, "n_tasks": K, "exp_name": exp})

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
