"""
KL divergence analysis between model predictions and GroupUniformKnownBayes.

This module provides functions to compute and visualize KL divergence between
a trained model's predictions and the Bayesian optimal predictor (GroupUniformKnownBayes).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

import icl.utils.notebook_utils as nu


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


# Legacy KL divergence functions (moved to icl.latent_markov.legacy.kl_divergence_legacy)
from icl.latent_markov.legacy.kl_divergence_legacy import (  # noqa: F401, E402
    compute_kl_divergence_vs_bayes,
    plot_kl_divergence,
    analyze_kl_divergence,
    compute_kl_divergence_vs_dirichlet_bayes,
    analyze_kl_divergence_dirichlet,
    compare_bayes_predictors,
)
