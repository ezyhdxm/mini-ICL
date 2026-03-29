"""KL divergence plots for linear regression analysis."""

import logging
import os
import pickle
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)

import icl.utils.notebook_utils as nu


def _pickle_load_cpu(f):
    """Load a pickle file, remapping any CUDA tensors to CPU."""
    import io

    class _CPUUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
            return super().find_class(module, name)

    return _CPUUnpickler(f).load()
from icl.linear.analysis.posterior._compute import (
    _gauss_kl_surrogate,
    _get_linear_kl_over_steps_cache_path,
    _sample_linear_mode_fixed,
)


@torch.no_grad()
def plot_kl_model_vs_two_bayes_linear_over_steps(
    exp_name: str,
    mode: str = "train",
    num_samples: int = 1024,
    steps=None,
    eps: float = 1e-12,
    add_std_band: bool = False,
    log_y: bool = False,
    use_cache: bool = True,
    force_recompute: bool = False,
    figsize: tuple = (8, 5),
    show: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Plot average KL-surrogate(model || baseline) across exact checkpoint steps
    for one linear experiment.
    """
    from icl.linear.linear_path_utils import load_model_task_config
    from icl.linear.lr_models import DiscreteMMSE, MixedRidge, UnbalancedMMSE

    _, train_task, config = load_model_task_config(exp_name)

    exp_dir = os.path.join(config.work_dir, exp_name)
    if os.getcwd().endswith("notebooks"):
        exp_dir = os.path.join("..", exp_dir)
    checkpoint_dir = exp_dir if config.task.name == "noisy_linear_regression" else os.path.join(exp_dir, "checkpoints")
    available_steps = nu.list_checkpoints(checkpoint_dir)["all_steps"]
    if len(available_steps) == 0:
        raise RuntimeError(f"No checkpoints found for exp_name={exp_name}.")

    if steps is None:
        steps_to_use = list(available_steps)
    else:
        requested_steps = [int(s) for s in steps]
        missing_steps = [s for s in requested_steps if s not in available_steps]
        if missing_steps:
            raise ValueError(
                f"Requested steps not found exactly: {missing_steps}. "
                f"Available steps: {available_steps}"
            )
        steps_to_use = requested_steps

    cache_path = _get_linear_kl_over_steps_cache_path(
        exp_dir=exp_dir,
        mode=mode,
        num_samples=num_samples,
        steps_to_use=steps_to_use,
        eps=eps,
    )
    cached = None
    if use_cache and not force_recompute and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = _pickle_load_cpu(f)
        if verbose:
            print(f"[cache] loaded KL-over-steps from {cache_path}")

    p_minor_orig = float(getattr(train_task, "p_minor", 0.0))
    sigma2 = max(float(train_task.noise_scale) ** 2, eps)

    if cached is None:
        if (
            int(getattr(train_task, "n_minor_tasks", 0)) > 0
            and getattr(train_task, "minor_pool", None) is not None
        ):
            exact_bayes = UnbalancedMMSE(
                scale=float(train_task.noise_scale),
                task_pool=train_task.task_pool.clone(),
                p0=p_minor_orig,
                minor_task_pool=train_task.minor_pool.clone(),
                dtype=train_task.dtype,
            ).to(config.device)
        else:
            exact_bayes = DiscreteMMSE(
                scale=float(train_task.noise_scale),
                task_pool=train_task.task_pool.clone(),
                dtype=train_task.dtype,
            ).to(config.device)

        tau = float(train_task.noise_scale) / max(float(train_task.minor_scale), eps)
        approx_bayes = MixedRidge(
            tau=tau,
            task_pool=train_task.task_pool.clone(),
            p0=p_minor_orig,
            noise_scale=float(train_task.noise_scale),
            dtype=train_task.dtype,
        ).to(config.device)

        sample_step = 0
        data, targets = _sample_linear_mode_fixed(
            train_task,
            mode=mode,
            batch_size=int(num_samples),
            step_idx=sample_step,
        )
        data = data.to(config.device)
        targets = targets.to(config.device)

        preds_exact = exact_bayes(data, targets)
        preds_approx = approx_bayes(data, targets)

        exact_means = []
        approx_means = []
        exact_stds = []
        approx_stds = []
        actual_steps = []

        for step in steps_to_use:
            model, actual_step = nu.load_checkpoint(
                config,
                step=step,
                exp_name=exp_name,
                return_actual_step=True,
            )
            model.eval().to(config.device)

            preds_model = model(data, targets)
            kl_exact = _gauss_kl_surrogate(preds_model, preds_exact, sigma2)
            kl_approx = _gauss_kl_surrogate(preds_model, preds_approx, sigma2)

            per_sample_exact = kl_exact.mean(dim=1)
            per_sample_approx = kl_approx.mean(dim=1)

            exact_means.append(float(per_sample_exact.mean().item()))
            approx_means.append(float(per_sample_approx.mean().item()))
            exact_stds.append(float(per_sample_exact.std().item()))
            approx_stds.append(float(per_sample_approx.std().item()))
            actual_steps.append(int(actual_step))

            if verbose:
                print(
                    f"[step {actual_step}] "
                    f"KL_exact={exact_means[-1]:.6f}, "
                    f"KL_approx={approx_means[-1]:.6f}"
                )

        cache_payload = {
            "exp_name": exp_name,
            "mode": mode,
            "steps": [int(s) for s in actual_steps],
            "num_samples": int(num_samples),
            "p_minor_bayes_used": float(p_minor_orig),
            "sigma2_used": float(sigma2),
            "kl_exact_mean_by_step": exact_means,
            "kl_exact_std_by_step": exact_stds,
            "kl_hybrid_mean_by_step": approx_means,
            "kl_hybrid_std_by_step": approx_stds,
        }
        if use_cache:
            with open(cache_path, "wb") as f:
                pickle.dump(cache_payload, f)
            if verbose:
                print(f"[cache] saved KL-over-steps to {cache_path}")
    else:
        cache_payload = cached

    x = np.asarray(cache_payload["steps"], dtype=float)
    exact_mean_arr = np.asarray(cache_payload["kl_exact_mean_by_step"], dtype=float)
    hybrid_mean_arr = np.asarray(cache_payload["kl_hybrid_mean_by_step"], dtype=float)
    exact_std_arr = np.asarray(cache_payload["kl_exact_std_by_step"], dtype=float)
    hybrid_std_arr = np.asarray(cache_payload["kl_hybrid_std_by_step"], dtype=float)
    band_floor = float(max(eps, 1e-30))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(x, exact_mean_arr, color="#1f77b4", lw=2.0, label="Exact")
    if add_std_band:
        ax.fill_between(
            x,
            np.maximum(exact_mean_arr - exact_std_arr, band_floor if log_y else 0.0),
            exact_mean_arr + exact_std_arr,
            color="#1f77b4",
            alpha=0.18,
        )

    ax.plot(x, hybrid_mean_arr, color="#d62728", lw=2.0, label="Approx")
    if add_std_band:
        ax.fill_between(
            x,
            np.maximum(hybrid_mean_arr - hybrid_std_arr, band_floor if log_y else 0.0),
            hybrid_mean_arr + hybrid_std_arr,
            color="#d62728",
            alpha=0.18,
        )

    ax.set_title("", fontsize=18)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Average KL(model || baseline)", fontsize=11)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "fig": fig,
        "ax": ax,
        "exp_name": exp_name,
        "mode": mode,
        "steps": [int(s) for s in cache_payload["steps"]],
        "num_samples": int(num_samples),
        "log_y": bool(log_y),
        "use_cache": bool(use_cache),
        "cache_path": cache_path if use_cache else None,
        "p_minor_bayes_used": float(cache_payload["p_minor_bayes_used"]),
        "sigma2_used": float(cache_payload["sigma2_used"]),
        "kl_exact_mean_by_step": exact_mean_arr,
        "kl_exact_std_by_step": exact_std_arr,
        "kl_hybrid_mean_by_step": hybrid_mean_arr,
        "kl_hybrid_std_by_step": hybrid_std_arr,
    }


def plot_kl_model_vs_two_bayes_linear_transition_across_k(
    k_values,
    mode: str = "train",
    num_samples: int = 1024,
    steps=None,
    eps: float = 1e-12,
    exp_name_kwargs: Optional[dict] = None,
    show_colorbar: bool = True,
    show_ylabel: bool = True,
    use_cache: bool = True,
    force_recompute: bool = False,
    figsize: tuple = (9, 4),
    show: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Visualize the transition from exact-Bayes-like to Gaussian-new-like behavior across k.

    Relative plot:
      hue = log10(KL_exact / KL_approx)
      alpha = absolute closeness based on min(KL_exact, KL_approx)

    Absolute plot:
      log10(min(KL_exact, KL_approx))
    """
    from matplotlib import colors
    from icl.utils.unified_interface import get_exp_name

    if exp_name_kwargs is None:
        exp_name_kwargs = {}

    def _centers_to_edges(vals: np.ndarray) -> np.ndarray:
        vals = np.asarray(vals, dtype=float)
        if vals.ndim != 1 or vals.size == 0:
            raise ValueError("Need a non-empty 1D array of centers.")
        if vals.size == 1:
            width = 1.0 if vals[0] == 0 else max(1.0, abs(vals[0]) * 0.05)
            return np.asarray([vals[0] - width, vals[0] + width], dtype=float)
        mids = 0.5 * (vals[:-1] + vals[1:])
        left = vals[0] - (mids[0] - vals[0])
        right = vals[-1] + (vals[-1] - mids[-1])
        return np.concatenate(([left], mids, [right]))

    curves = {}
    exp_names = {}
    all_steps = set()
    for k in k_values:
        exp_name = get_exp_name("linear", k, **exp_name_kwargs)
        exp_names[k] = exp_name
        try:
            out = plot_kl_model_vs_two_bayes_linear_over_steps(
                exp_name=exp_name,
                mode=mode,
                num_samples=num_samples,
                steps=steps,
                eps=eps,
                use_cache=use_cache,
                force_recompute=force_recompute,
                show=False,
                verbose=verbose,
            )
            steps_k = np.asarray(out["steps"], dtype=int)
            exact_k = np.asarray(out["kl_exact_mean_by_step"], dtype=float)
            hybrid_k = np.asarray(out["kl_hybrid_mean_by_step"], dtype=float)
            curves[k] = {
                "steps": steps_k,
                "kl_exact_mean_by_step": exact_k,
                "kl_hybrid_mean_by_step": hybrid_k,
                "log10_ratio": np.log(np.clip(exact_k, eps, None) / np.clip(hybrid_k, eps, None)),
            }
            all_steps.update(steps_k.tolist())
        except Exception as e:
            logger.warning(f"k={k} ({exp_names[k]}): {e}")

    ks = sorted(curves.keys())
    if len(ks) == 0:
        raise RuntimeError("No k succeeded. Check exp_name availability/checkpoints.")

    step_grid = np.asarray(sorted(all_steps), dtype=float)
    n_k = len(ks)
    n_s = len(step_grid)
    log_ratio = np.full((n_k, n_s), np.nan, dtype=float)
    exact_mat = np.full((n_k, n_s), np.nan, dtype=float)
    hybrid_mat = np.full((n_k, n_s), np.nan, dtype=float)
    step_to_idx = {int(s): i for i, s in enumerate(step_grid)}

    for row, k in enumerate(ks):
        d = curves[k]
        for step_val, exact_val, hybrid_val, ratio_val in zip(
            d["steps"],
            d["kl_exact_mean_by_step"],
            d["kl_hybrid_mean_by_step"],
            d["log10_ratio"],
        ):
            col = step_to_idx[int(step_val)]
            exact_mat[row, col] = exact_val
            hybrid_mat[row, col] = hybrid_val
            log_ratio[row, col] = ratio_val

    rel_vmax = 2.0
    rel_norm = colors.TwoSlopeNorm(vmin=-rel_vmax, vcenter=0.0, vmax=rel_vmax)

    best_kl = np.minimum(exact_mat, hybrid_mat)
    log10_best = np.full_like(best_kl, np.nan, dtype=float)
    valid_best = np.isfinite(best_kl)
    log10_best[valid_best] = np.log10(np.clip(best_kl[valid_best], eps, None))
    finite_best = log10_best[np.isfinite(log10_best)]
    if finite_best.size > 0:
        abs_vmin = float(np.min(finite_best))
        abs_vmax = float(np.max(finite_best))
    else:
        abs_vmin, abs_vmax = -6.0, 0.0
    if abs_vmax - abs_vmin < 1e-12:
        abs_vmax = abs_vmin + 1e-12
    abs_norm = colors.Normalize(vmin=abs_vmin, vmax=abs_vmax)

    closeness = np.full_like(log10_best, np.nan, dtype=float)
    closeness[valid_best] = 1.0 - (
        (log10_best[valid_best] - abs_vmin) / (abs_vmax - abs_vmin)
    )
    closeness = np.clip(closeness, 0.0, 1.0)
    closeness_gamma = np.full_like(closeness, np.nan, dtype=float)
    closeness_gamma[valid_best] = np.power(closeness[valid_best], 0.6)
    alpha_matrix = np.full_like(log10_best, 0.0, dtype=float)
    alpha_matrix[valid_best] = 0.42 + 0.58 * closeness_gamma[valid_best]

    y_centers = np.asarray(ks, dtype=float)
    x_edges = _centers_to_edges(step_grid)
    y_edges = _centers_to_edges(y_centers)
    log_ratio_ma = np.ma.masked_invalid(log_ratio)
    log10_best_ma = np.ma.masked_invalid(log10_best)

    fig_rel, ax_rel = plt.subplots(1, 1, figsize=figsize)
    cmap_rel = plt.get_cmap("RdBu_r").copy()
    cmap_rel.set_bad(color="#f0f0f0")
    mesh_rel = ax_rel.pcolormesh(
        x_edges,
        y_edges,
        log_ratio_ma,
        cmap=cmap_rel,
        norm=rel_norm,
        shading="auto",
    )
    rel_facecolors = cmap_rel(rel_norm(log_ratio_ma.filled(0.0)))
    rel_facecolors[..., -1] = alpha_matrix
    mesh_rel.set_facecolor(rel_facecolors.reshape(-1, 4))
    if show_colorbar:
        cbar_rel = fig_rel.colorbar(mesh_rel, ax=ax_rel, pad=0.02)
        cbar_rel.set_label(r"$\log(\mathrm{KL}_{\mathrm{exact}} / \mathrm{KL}_{\mathrm{approx}})$", fontsize=13)
        cbar_rel.ax.tick_params(labelsize=12)
    ax_rel.set_xlabel("Training Step", fontsize=11)
    if show_ylabel:
        ax_rel.set_ylabel(r"$\log_2(\mathrm{Number\ of\ Minority\ Tasks})$", fontsize=11)
    ax_rel.set_yticks(y_centers)
    ax_rel.grid(False)
    fig_rel.tight_layout()

    fig_abs, ax_abs = plt.subplots(1, 1, figsize=figsize)
    cmap_abs = plt.get_cmap("viridis_r").copy()
    cmap_abs.set_bad(color="#f0f0f0")
    mesh_abs = ax_abs.pcolormesh(
        x_edges,
        y_edges,
        log10_best_ma,
        cmap=cmap_abs,
        norm=abs_norm,
        shading="auto",
    )
    if show_colorbar:
        cbar_abs = fig_abs.colorbar(mesh_abs, ax=ax_abs, pad=0.02)
        cbar_abs.set_label(r"$\log_{10}(\min(\mathrm{KL}_{\mathrm{exact}}, \mathrm{KL}_{\mathrm{approx}}))$", fontsize=13)
        cbar_abs.ax.tick_params(labelsize=12)
    ax_abs.set_title("Best Absolute KL", fontsize=12)
    ax_abs.set_xlabel("Training Step", fontsize=11)
    if show_ylabel:
        ax_abs.set_ylabel(r"$\log_2(\mathrm{Number\ of\ Minority\ Tasks})$", fontsize=11)
    ax_abs.set_yticks(y_centers)
    ax_abs.grid(False)
    fig_abs.tight_layout()

    if show:
        plt.figure(fig_rel.number)
        plt.show()
        plt.figure(fig_abs.number)
        plt.show()
    else:
        plt.close(fig_rel)
        plt.close(fig_abs)

    return {
        "fig": fig_rel,
        "fig_rel": fig_rel,
        "fig_abs": fig_abs,
        "ax": ax_rel,
        "ax_rel": ax_rel,
        "ax_abs": ax_abs,
        "axes": (ax_rel, ax_abs),
        "mode": mode,
        "show_colorbar": bool(show_colorbar),
        "k_values_loaded": ks,
        "exp_names_by_k": exp_names,
        "step_grid": step_grid.astype(int),
        "log10_ratio_matrix": log_ratio,
        "best_kl_matrix": best_kl,
        "log10_best_kl_matrix": log10_best,
        "alpha_matrix": alpha_matrix,
        "kl_exact_matrix": exact_mat,
        "kl_hybrid_matrix": hybrid_mat,
        "curves_by_k": curves,
    }
