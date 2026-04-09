import hashlib
import json
import os
import pickle
import torch
from typing import Optional

from icl.coin.coin_bayes import KnownPoolBayesCoins, ThreeKnownPlusNewDirichletCoinBayes


def _extract_samples(out):
    """Unwrap sampler.generate() output to a 2-D token tensor."""
    s = out[0] if isinstance(out, (tuple, list)) else out
    if not torch.is_tensor(s):
        raise ValueError(f"Unsupported generate output type: {type(out)}")
    if s.dim() == 3:
        s = s.squeeze(0)
    return s


def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12):
    """KL(p || q) per sample and position.  p, q: (B, T, V) → (B, T)."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (torch.log(p) - torch.log(q))).sum(dim=-1)


def _json_ready(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return repr(value)


def _get_coin_kl_over_steps_cache_path(
    *,
    exp_dir: str,
    mode: str,
    num_samples: int,
    steps_to_use,
    uniform_train_tasks: bool,
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
        "uniform_train_tasks": bool(uniform_train_tasks),
        "p_common_total": p_common_total,
        "alpha_new": float(alpha_new),
        "eps": float(eps),
        "version": 1,
    }
    payload_str = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    payload_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"coin_kl_over_steps_{payload_hash}.pkl")



@torch.no_grad()
def plot_kl_model_vs_two_bayes_coin_over_steps(
    exp_name: str,
    mode: str = "train",
    num_samples: int = 1024,
    steps=None,
    uniform_train_tasks: bool = False,
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
    Plot average KL(model || baseline) across exact checkpoint steps for one experiment.

    The evaluation batch is generated once and reused for every checkpoint so the
    KL-vs-step curve reflects model changes rather than sampling noise.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import icl.utils.notebook_utils as nu

    _, sampler, config = nu.load_everything("coin", exp_name)

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

    cache_path = _get_coin_kl_over_steps_cache_path(
        exp_dir=exp_dir,
        mode=mode,
        num_samples=num_samples,
        steps_to_use=steps_to_use,
        uniform_train_tasks=uniform_train_tasks,
        p_common_total=p_common_total,
        alpha_new=alpha_new,
        eps=eps,
    )

    cached = None
    if use_cache and not force_recompute and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if verbose:
            print(f"[cache] loaded KL-over-steps from {cache_path}")

    p_minor_orig = float(getattr(sampler, "p_minor", 0.0))

    if (
        mode == "train"
        and uniform_train_tasks
        and int(getattr(sampler, "n_major_tasks", 0)) > 0
        and int(getattr(sampler, "n_minor_tasks", 0)) > 0
    ):
        n_major = int(getattr(sampler, "n_major_tasks", 0))
        n_minor = int(getattr(sampler, "n_minor_tasks", 0))
        p_minor_sample = float(n_minor / (n_major + n_minor))
    else:
        p_minor_sample = p_minor_orig

    if cached is None:
        sampler.p_minor = p_minor_sample
        out = sampler.generate(mode=mode, num_samples=int(num_samples), epochs=1)
        samples = _extract_samples(out).to(config.device)

        p_common_eff = p_common_total
        if p_common_eff is None:
            p_common_eff = float(max(0.0, min(1.0, 1.0 - p_minor_orig)))

        exact_bayes = KnownPoolBayesCoins(
            probs_major=sampler.major_p,
            probs_minor=sampler.minor_p if sampler.n_minor_tasks > 0 else None,
            p_minor=p_minor_orig,
            include_minor=True,
            device=config.device,
        )
        hybrid_bayes = ThreeKnownPlusNewDirichletCoinBayes(
            probs_major_3=sampler.major_p[:3],
            p_common_total=p_common_eff,
            alpha=alpha_new,
            device=config.device,
        )

        x_real = samples
        if x_real.size(1) < 2:
            raise ValueError("Need at least 2 tokens for KL-vs-position.")
        p_exact = exact_bayes.predict(x_real)[:, : x_real.size(1) - 1, :]
        p_hybrid = hybrid_bayes.predict(x_real)[:, : x_real.size(1) - 1, :]

        exact_means = []
        hybrid_means = []
        exact_stds = []
        hybrid_stds = []
        actual_steps = []

        try:
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

                kl_exact = _kl_divergence(p_model, p_exact, eps)
                kl_hybrid = _kl_divergence(p_model, p_hybrid, eps)

                per_sample_exact = kl_exact.mean(dim=1)
                per_sample_hybrid = kl_hybrid.mean(dim=1)

                exact_means.append(float(per_sample_exact.mean().item()))
                hybrid_means.append(float(per_sample_hybrid.mean().item()))
                exact_stds.append(float(per_sample_exact.std().item()))
                hybrid_stds.append(float(per_sample_hybrid.std().item()))
                actual_steps.append(int(actual_step))

                if verbose:
                    print(
                        f"[step {actual_step}] "
                        f"KL_exact={exact_means[-1]:.6f}, "
                        f"KL_hybrid={hybrid_means[-1]:.6f}"
                    )
        finally:
            sampler.p_minor = p_minor_orig

        cache_payload = {
            "exp_name": exp_name,
            "mode": mode,
            "steps": [int(s) for s in actual_steps],
            "num_samples": int(num_samples),
            "p_minor_sampling_used": float(p_minor_sample),
            "p_minor_bayes_used": float(p_minor_orig),
            "kl_exact_mean_by_step": exact_means,
            "kl_exact_std_by_step": exact_stds,
            "kl_hybrid_mean_by_step": hybrid_means,
            "kl_hybrid_std_by_step": hybrid_stds,
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
        "p_minor_sampling_used": float(cache_payload["p_minor_sampling_used"]),
        "p_minor_bayes_used": float(cache_payload["p_minor_bayes_used"]),
        "kl_exact_mean_by_step": exact_mean_arr,
        "kl_exact_std_by_step": exact_std_arr,
        "kl_hybrid_mean_by_step": hybrid_mean_arr,
        "kl_hybrid_std_by_step": hybrid_std_arr,
    }


def plot_kl_model_vs_two_bayes_coin_transition_across_k(
    k_values=None,
    mode: str = "train",
    num_samples: int = 1024,
    steps=None,
    uniform_train_tasks: bool = False,
    p_common_total: Optional[float] = None,
    alpha_new: float = 1.0,
    eps: float = 1e-12,
    show_colorbar: bool = True,
    show_ylabel: bool = True,
    figsize: tuple = (9, 4),
    show: bool = True,
    verbose: bool = False,
    vocab_size: Optional[int] = None,
    n_major_values=None,
    major_only_exp_kwargs: Optional[dict] = None,
    use_cache: bool = True,
    force_recompute: bool = False,
) -> dict:
    """
    Visualize the transition from exact-Bayes-like to Dirichlet-like behavior across k.

    Relative plot:
      hue = log10(KL_exact / KL_approx)
      alpha = absolute closeness based on min(KL_exact, KL_approx)

    Absolute plot:
      log10(min(KL_exact, KL_approx))
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors

    from icl.utils.unified_interface import get_exp_name

    if n_major_values is not None:
        loop_keys = list(n_major_values)
        major_kw = major_only_exp_kwargs if major_only_exp_kwargs is not None else {}
        use_major_only = True
        # Default hybrid prior for major-only: new-task bucket dominates (p_common ≈ 0).
        if p_common_total is None:
            p_common_total = 1e-12
    elif k_values is not None:
        loop_keys = list(k_values)
        major_kw = {}
        use_major_only = False
    else:
        raise ValueError("Provide k_values or n_major_values.")

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
    for key in loop_keys:
        if use_major_only:
            exp_name = get_exp_name(
                "coin",
                k=0,
                vocab_size=vocab_size,
                n_tasks=int(key),
                n_minor_tasks=1,
                p_minor=1e-12,
                **major_kw,
            )
        else:
            exp_name = get_exp_name("coin", k=key, vocab_size=vocab_size)
        exp_names[key] = exp_name
        try:
            out = plot_kl_model_vs_two_bayes_coin_over_steps(
                exp_name=exp_name,
                mode=mode,
                num_samples=num_samples,
                steps=steps,
                uniform_train_tasks=uniform_train_tasks,
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
            if verbose:
                print(f"[warn] key={key} failed: {e}")

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

    k_centers = np.asarray(ks, dtype=float)
    y_centers = k_centers
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
    ax_abs.set_yticks(y_centers)
    if show_ylabel:
        ax_abs.set_ylabel(r"$\log_2(\mathrm{Number\ of\ Minority\ Tasks})$", fontsize=11)
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
