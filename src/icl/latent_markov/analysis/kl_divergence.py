"""
KL divergence analysis between model predictions and GroupUniformKnownBayes.

This module provides functions to compute and visualize KL divergence between
a trained model's predictions and the Bayesian optimal predictor (GroupUniformKnownBayes).
"""

import hashlib
import json
import os
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

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


def _json_ready(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return repr(value)


def _get_latent_kl_over_steps_cache_path(
    *,
    exp_dir: str,
    mode: str,
    num_samples: int,
    steps_to_use,
    p_common_total: Optional[float],
    alpha_new: float,
    eps: float,
):
    cache_dir = os.path.join(exp_dir, "analysis_cache")
    os.makedirs(cache_dir, exist_ok=True)
    payload = {
        "mode": mode,
        "num_samples": int(num_samples),
        "steps_to_use": [int(s) for s in steps_to_use],
        "p_common_total": p_common_total,
        "alpha_new": float(alpha_new),
        "eps": float(eps),
        "version": 1,
    }
    payload_str = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    payload_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"latent_kl_over_steps_{payload_hash}.pkl")


# ---------------------------------------------------------------------------
# KL plots vs two Bayesian baselines (moved from markov_latent.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_kl_model_vs_two_bayes_latent(
    exp_name: str,
    modes: tuple = ("train", "major", "minor", "ood"),
    num_samples: int = 512,
    step: Optional[int] = None,
    p_common_total: Optional[float] = None,
    alpha_new: float = 1.0,
    eps: float = 1e-12,
    figsize: Optional[tuple] = None,
    show: bool = True,
):
    """
    Compare transformer next-token distributions against two Bayesian latent baselines.

    Baselines:
      1) Exact known-pool Bayesian baseline (ThreeKnownUniformBayes or GroupUniformKnownBayes)
      2) ThreeKnownPlusNewDirichletBayes: 3 known majors + one aggregated "new" Dirichlet bucket.
    """
    from icl.latent_markov.analysis.bayes import (
        GroupUniformKnownBayes,
        ThreeKnownPlusNewDirichletBayes,
        ThreeKnownUniformBayes,
    )

    _, sampler, config = nu.load_everything("latent", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval().to(config.device)
    # Ensure the sampler's transition matrices live on the same device as the
    # model (else the Bayesian-baseline torch.cat mixes cpu/cuda tensors).
    if hasattr(sampler, "to"):
        sampler.to(config.device)

    if int(getattr(sampler, "order", 1)) != 1:
        raise ValueError("plot_kl_model_vs_two_bayes_latent currently supports order=1 only.")

    p_minor_orig = float(getattr(sampler, "p_minor", 0.0))
    n_major = int(getattr(sampler, "n_major_tasks", 0))
    n_minor = int(getattr(sampler, "n_minor_tasks", 0))
    if n_major < 3:
        raise ValueError("Need at least 3 major tasks to use three-known latent baselines.")

    trans_major_3 = sampler.major_trans_mat[:3].to(config.device)

    if p_common_total is None:
        p_common_total = float(max(0.0, min(1.0, 1.0 - p_minor_orig)))

    rest_parts = []
    if sampler.major_trans_mat.shape[0] > 3:
        rest_parts.append(sampler.major_trans_mat[3:])
    if n_minor > 0:
        rest_parts.append(sampler.minor_trans_mat)
    if len(rest_parts) == 0:
        exact_bayes = ThreeKnownUniformBayes(
            trans_mat_3=trans_major_3, device=config.device, eps=max(eps, 1e-30),
        )
    else:
        trans_exact = torch.cat([trans_major_3] + rest_parts, dim=0).to(config.device)
        p_common_exact = float((1.0 - p_minor_orig) * (3.0 / max(1.0, float(n_major))))
        p_common_exact = float(max(0.0, min(1.0, p_common_exact)))
        exact_bayes = GroupUniformKnownBayes(
            trans_mat=trans_exact, p_common=p_common_exact, device=config.device,
        )

    hybrid_bayes = ThreeKnownPlusNewDirichletBayes(
        trans_mat_3=trans_major_3, p_common_total=p_common_total,
        alpha=alpha_new, device=config.device,
    )

    def _extract_samples_from_generate_output(out):
        if torch.is_tensor(out):
            s = out
        elif isinstance(out, (tuple, list)):
            s = out[0]
        else:
            raise ValueError(f"Unsupported generate output type: {type(out)}")
        if s.dim() == 3:
            s = s.squeeze(0)
        return s

    def _aligned_model_probs(samples_batch: torch.Tensor):
        logits = model(samples_batch)
        probs_full = torch.softmax(logits, dim=-1)
        V_real = int(sampler.num_states)
        probs_real = probs_full[..., :V_real]
        probs_real = probs_real / probs_real.sum(dim=-1, keepdim=True).clamp_min(eps)
        x_real = samples_batch
        T_real = x_real.size(1)
        if T_real < 2:
            raise ValueError("Need at least 2 tokens for KL-vs-position.")
        probs_cmp = probs_real[:, : T_real - 1, :]
        return probs_cmp, x_real

    def _kl_model_to_ref(p_model: torch.Tensor, p_ref: torch.Tensor):
        p = p_model.clamp_min(eps)
        q = p_ref.clamp_min(eps)
        return (p * (torch.log(p) - torch.log(q))).sum(dim=-1)

    results = {}
    if isinstance(modes, str):
        modes = (modes,)
    else:
        modes = tuple(modes)

    ncols = 1 if len(modes) == 1 else 2
    nrows = (len(modes) + ncols - 1) // ncols
    if figsize is None:
        figsize = (7 * ncols, 4.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for mi, mode in enumerate(modes):
        out = sampler.generate(mode=mode, num_samples=int(num_samples), epochs=1)
        samples = _extract_samples_from_generate_output(out).to(config.device)

        p_model, x_real = _aligned_model_probs(samples)
        p_exact = exact_bayes.predict(x_real)[:, : p_model.size(1), :]
        p_hybrid = hybrid_bayes.predict(x_real)[:, : p_model.size(1), :]

        kl_exact = _kl_model_to_ref(p_model, p_exact)
        kl_hybrid = _kl_model_to_ref(p_model, p_hybrid)

        mean_exact = kl_exact.mean(dim=0).detach().cpu().numpy()
        std_exact = kl_exact.std(dim=0).detach().cpu().numpy()
        mean_hybrid = kl_hybrid.mean(dim=0).detach().cpu().numpy()
        std_hybrid = kl_hybrid.std(dim=0).detach().cpu().numpy()
        pos = torch.arange(p_model.size(1)).cpu().numpy()

        ax = axes_flat[mi]
        ax.plot(pos, mean_exact, color="#1f77b4", lw=2.0, label="Exact known-pool Bayes")
        ax.fill_between(pos, np.maximum(mean_exact - std_exact, 0.0), mean_exact + std_exact, color="#1f77b4", alpha=0.2)
        ax.plot(pos, mean_hybrid, color="#d62728", lw=2.0, label="3-known + Dirichlet-new")
        ax.fill_between(pos, np.maximum(mean_hybrid - std_hybrid, 0.0), mean_hybrid + std_hybrid, color="#d62728", alpha=0.2)
        ax.set_title("", fontsize=18)
        ax.set_xlabel("Position", fontsize=11)
        ax.set_ylabel("KL(model || baseline)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        results[mode] = {
            "positions": pos, "p_minor_bayes_used": float(p_minor_orig),
            "p_common_total_used": float(p_common_total),
            "kl_exact_mean": mean_exact, "kl_exact_std": std_exact,
            "kl_hybrid_mean": mean_hybrid, "kl_hybrid_std": std_hybrid,
            "kl_exact": kl_exact.detach().cpu(), "kl_hybrid": kl_hybrid.detach().cpu(),
        }

    for j in range(len(modes), len(axes_flat)):
        axes_flat[j].axis("off")
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"fig": fig, "axes": axes, "results": results, "modes": modes,
            "step": int(step)}


@torch.no_grad()
def plot_kl_model_vs_two_bayes_latent_across_k(
    k_values,
    mode: str = "ood",
    num_samples: int = 1024,
    p_common_total: Optional[float] = None,
    alpha_new: float = 1.0,
    eps: float = 1e-12,
    add_std_band: bool = False,
    same_y_axis: bool = False,
    figsize: tuple = (14, 5),
    show: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Visualize KL(model || baseline) across multiple k values for the latent task.
    """
    from icl.utils.unified_interface import get_exp_name

    curves = {}
    for k in k_values:
        exp_name = get_exp_name("latent", k)
        try:
            out = plot_kl_model_vs_two_bayes_latent(
                exp_name=exp_name, modes=mode, num_samples=num_samples,
                p_common_total=p_common_total,
                alpha_new=alpha_new, eps=eps, show=False,
            )
            d = out["results"][mode]
            curves[k] = {
                "positions": np.asarray(d["positions"], dtype=float),
                "exact_mean": np.asarray(d["kl_exact_mean"], dtype=float),
                "exact_std": np.asarray(d["kl_exact_std"], dtype=float),
                "hybrid_mean": np.asarray(d["kl_hybrid_mean"], dtype=float),
                "hybrid_std": np.asarray(d["kl_hybrid_std"], dtype=float),
                "p_minor_bayes_used": float(d["p_minor_bayes_used"]),
                "p_common_total_used": float(d["p_common_total_used"]),
            }
        except Exception as e:
            if verbose:
                print(f"[warn] k={k} failed: {e}")

    ks = sorted(curves.keys())
    if len(ks) == 0:
        raise RuntimeError("No k succeeded. Check exp_name availability/checkpoints.")

    if len(ks) <= 10:
        palette = list(plt.get_cmap("tab10").colors)
    elif len(ks) <= 20:
        palette = list(plt.get_cmap("tab20").colors)
    else:
        base = list(plt.get_cmap("tab20").colors)
        reps = (len(ks) + len(base) - 1) // len(base)
        palette = (base * reps)[: len(ks)]
    color_map = {k: palette[i] for i, k in enumerate(ks)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=False, sharey=same_y_axis)

    for k in ks:
        c = color_map[k]
        d = curves[k]
        x = d["positions"]
        ax1.plot(x, d["exact_mean"], color=c, lw=2.0, label=f"k={k}")
        if add_std_band:
            ax1.fill_between(x, np.maximum(d["exact_mean"] - d["exact_std"], 0.0),
                             d["exact_mean"] + d["exact_std"], color=c, alpha=0.15)
        ax2.plot(x, d["hybrid_mean"], color=c, lw=2.0, label=f"k={k}")
        if add_std_band:
            ax2.fill_between(x, np.maximum(d["hybrid_mean"] - d["hybrid_std"], 0.0),
                             d["hybrid_mean"] + d["hybrid_std"], color=c, alpha=0.15)

    ax1.set_title("", fontsize=18)
    ax1.set_xlabel("Position", fontsize=11)
    ax1.set_ylabel("KL(model || exact)", fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("", fontsize=18)
    ax2.set_xlabel("Position", fontsize=11)
    ax2.set_ylabel("KL(model || hybrid)", fontsize=11)
    ax2.grid(True, alpha=0.3)

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=9, frameon=True)
    fig.tight_layout(rect=(0.0, 0.0, 0.9, 1.0))

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"fig": fig, "axes": (ax1, ax2), "mode": mode,
            "k_values_loaded": ks, "curves_by_k": curves}


@torch.no_grad()
def plot_kl_model_vs_two_bayes_latent_over_steps(
    exp_name: str,
    mode: str = "train",
    num_samples: int = 1024,
    steps=None,
    p_common_total: Optional[float] = None,
    alpha_new: float = 1.0,
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
    Plot average KL(model || baseline) across exact checkpoint steps for one latent experiment.

    The evaluation batch is generated once and reused for every checkpoint so the
    KL-vs-step curve reflects model changes rather than sampling noise.
    """
    _, sampler, config = nu.load_everything("latent", exp_name)

    if int(getattr(sampler, "order", 1)) != 1:
        raise ValueError("plot_kl_model_vs_two_bayes_latent_over_steps currently supports order=1 only.")

    exp_dir = os.path.join(config.work_dir, exp_name)
    if os.getcwd().endswith("notebooks"):
        exp_dir = os.path.join("..", exp_dir)
    checkpoint_dir = (
        exp_dir if config.task.name == "noisy_linear_regression"
        else os.path.join(exp_dir, "checkpoints")
    )
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

    cache_path = _get_latent_kl_over_steps_cache_path(
        exp_dir=exp_dir,
        mode=mode,
        num_samples=num_samples,
        steps_to_use=steps_to_use,
        p_common_total=p_common_total,
        alpha_new=alpha_new,
        eps=eps,
    )
    cached = None
    if use_cache and not force_recompute and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = _pickle_load_cpu(f)
        if verbose:
            print(f"[cache] loaded KL-over-steps from {cache_path}")

    p_minor_orig = float(getattr(sampler, "p_minor", 0.0))
    n_major = int(getattr(sampler, "n_major_tasks", 0))
    n_minor = int(getattr(sampler, "n_minor_tasks", 0))
    if n_major < 3:
        raise ValueError("Need at least 3 major tasks to use three-known latent baselines.")

    def _extract_samples_from_generate_output(out):
        if torch.is_tensor(out):
            s = out
        elif isinstance(out, (tuple, list)):
            s = out[0]
        else:
            raise ValueError(f"Unsupported generate output type: {type(out)}")
        if s.dim() == 3:
            s = s.squeeze(0)
        return s

    def _kl_model_to_ref(p_model: torch.Tensor, p_ref: torch.Tensor):
        p = p_model.clamp_min(eps)
        q = p_ref.clamp_min(eps)
        return (p * (torch.log(p) - torch.log(q))).sum(dim=-1)

    if cached is None:
        trans_major_3 = sampler.major_trans_mat[:3].to(config.device)

        p_common_eff = p_common_total
        if p_common_eff is None:
            p_common_eff = float(max(0.0, min(1.0, 1.0 - p_minor_orig)))

        rest_parts = []
        if sampler.major_trans_mat.shape[0] > 3:
            rest_parts.append(sampler.major_trans_mat[3:].to(config.device))
        if n_minor > 0:
            rest_parts.append(sampler.minor_trans_mat.to(config.device))

        if len(rest_parts) == 0:
            from icl.latent_markov.analysis.bayes import ThreeKnownUniformBayes

            exact_bayes = ThreeKnownUniformBayes(
                trans_mat_3=trans_major_3, device=config.device, eps=max(eps, 1e-30),
            )
        else:
            from icl.latent_markov.analysis.bayes import GroupUniformKnownBayes

            trans_exact = torch.cat([trans_major_3] + rest_parts, dim=0)
            p_common_exact = float((1.0 - p_minor_orig) * (3.0 / max(1.0, float(n_major))))
            p_common_exact = float(max(0.0, min(1.0, p_common_exact)))
            exact_bayes = GroupUniformKnownBayes(
                trans_mat=trans_exact, p_common=p_common_exact, device=config.device,
            )

        from icl.latent_markov.analysis.bayes import ThreeKnownPlusNewDirichletBayes

        approx_bayes = ThreeKnownPlusNewDirichletBayes(
            trans_mat_3=trans_major_3, p_common_total=p_common_eff,
            alpha=alpha_new, device=config.device,
        )

        out = sampler.generate(mode=mode, num_samples=int(num_samples), epochs=1)
        samples = _extract_samples_from_generate_output(out).to(config.device)

        x_real = samples
        if x_real.size(1) < 2:
            raise ValueError("Need at least 2 tokens for KL-vs-position.")
        p_exact = exact_bayes.predict(x_real)[:, : x_real.size(1) - 1, :]
        p_approx = approx_bayes.predict(x_real)[:, : x_real.size(1) - 1, :]

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

            logits = model(samples)
            probs_full = torch.softmax(logits, dim=-1)
            probs_real = probs_full[..., : int(sampler.num_states)]
            probs_real = probs_real / probs_real.sum(dim=-1, keepdim=True).clamp_min(eps)
            p_model = probs_real[:, : x_real.size(1) - 1, :]

            kl_exact = _kl_model_to_ref(p_model, p_exact)
            kl_approx = _kl_model_to_ref(p_model, p_approx)

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
            "p_common_total_used": float(p_common_eff),
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
        "p_common_total_used": float(cache_payload["p_common_total_used"]),
        "kl_exact_mean_by_step": exact_mean_arr,
        "kl_exact_std_by_step": exact_std_arr,
        "kl_hybrid_mean_by_step": hybrid_mean_arr,
        "kl_hybrid_std_by_step": hybrid_std_arr,
    }


def plot_kl_model_vs_two_bayes_latent_transition_across_k(
    k_values=None,
    mode: str = "train",
    num_samples: int = 1024,
    steps=None,
    p_common_total: Optional[float] = None,
    alpha_new: float = 1.0,
    eps: float = 1e-12,
    show_colorbar: bool = True,
    show_ylabel: bool = True,
    use_cache: bool = True,
    force_recompute: bool = False,
    figsize: tuple = (9, 4),
    show: bool = True,
    verbose: bool = False,
    exp_name_kwargs: Optional[dict] = None,
    n_major_values=None,
    major_only_exp_kwargs: Optional[dict] = None,
) -> dict:
    """
    Visualize the transition from exact-Bayes-like to Dirichlet-like behavior across k.

    Relative plot:
      hue = log10(KL_exact / KL_approx)
      alpha = absolute closeness based on min(KL_exact, KL_approx)

    Absolute plot:
      log10(min(KL_exact, KL_approx))
    """
    from matplotlib import colors
    from icl.utils.unified_interface import get_exp_name

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

    import logging as _logging
    _logger = _logging.getLogger(__name__)

    if exp_name_kwargs is None:
        exp_name_kwargs = {}

    if n_major_values is not None:
        loop_keys = list(n_major_values)
        use_major_only = True
        # Default hybrid prior for major-only: new-task bucket dominates (p_common ≈ 0).
        if p_common_total is None:
            p_common_total = 1e-12
    elif k_values is not None:
        loop_keys = list(k_values)
        use_major_only = False
    else:
        raise ValueError("Provide k_values or n_major_values.")

    curves = {}
    exp_names = {}
    all_steps = set()
    for key in loop_keys:
        base_kw = dict(exp_name_kwargs)
        if use_major_only:
            base_kw.update(major_only_exp_kwargs or {})
            exp_name = get_exp_name(
                "latent",
                0,
                n_tasks=int(key),
                n_minor_tasks=1,
                p_minor=1e-12,
                **base_kw,
            )
        else:
            exp_name = get_exp_name("latent", key, **base_kw)
        exp_names[key] = exp_name
        try:
            out = plot_kl_model_vs_two_bayes_latent_over_steps(
                exp_name=exp_name,
                mode=mode,
                num_samples=num_samples,
                steps=steps,
                p_common_total=p_common_total,
                alpha_new=alpha_new,
                eps=eps,
                use_cache=use_cache,
                force_recompute=force_recompute,
                show=False,
                verbose=verbose,
            )
            steps_k = np.asarray(out["steps"], dtype=int)
            exact_k = np.asarray(out["kl_exact_mean_by_step"], dtype=float)
            hybrid_k = np.asarray(out["kl_hybrid_mean_by_step"], dtype=float)
            curves[key] = {
                "steps": steps_k,
                "kl_exact_mean_by_step": exact_k,
                "kl_hybrid_mean_by_step": hybrid_k,
                "log10_ratio": np.log(np.clip(exact_k, eps, None) / np.clip(hybrid_k, eps, None)),
            }
            all_steps.update(steps_k.tolist())
        except Exception as e:
            _logger.warning(f"key={key} ({exp_names[key]}): {e}")

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
        cbar_rel.set_label(r"$\log(\mathrm{KL}_{\mathrm{bayesian}} / \mathrm{KL}_{\mathrm{extrapolative}})$", fontsize=13)
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
        cbar_abs.set_label(r"$\log_{10}(\min(\mathrm{KL}_{\mathrm{bayesian}}, \mathrm{KL}_{\mathrm{extrapolative}}))$", fontsize=13)
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
