"""
Posterior agreement and ID/OOD loss plotting for latent Markov
(non-padded sequences).

Extracted from ``icl.utils.latent_nonpadded``.
"""

import numpy as np
import torch
from typing import Optional

import icl.utils.notebook_utils as nu
from icl.utils.unified_interface import get_exp_name, _get_hiddens_at_real_positions
from icl.latent_markov.analysis.bayes import task_posterior_over_time
from icl.latent_markov.analysis.probes import train_linear_hidden_predictor
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
#  plot_lambda_posterior_agreement
# ---------------------------------------------------------------------------

def plot_lambda_posterior_agreement(
    exp_name: str,
    layer_index: Optional[int] = None,
    n_ood: int = 30,
    B: int = 64,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    window: int = 20,
    figsize: tuple = (18, 5),
    show: bool = True,
    title: str = "",
    eps: float = 1e-12,
) -> dict:
    """
    Agreement between projected coefficients and latent Bayesian posterior,
    measured by TV distance, cosine similarity, and rolling Pearson
    correlation.

    Latent counterpart of
    ``plot_lambda_posterior_agreement_coin_nonpadded``.

    Notes
    -----
    For OOD tasks, posterior is computed under a *major-only* model
    (``include_minor=False``), i.e. misspecified inference over the 3
    anchor tasks.

    This implementation avoids ``Mean of empty slice`` warnings by using
    safe averaging helpers for:
      - empty OOD groups (e.g. ``n_ood=0``),
      - all-NaN rolling-correlation windows.
    """
    import matplotlib.pyplot as plt
    from icl.linear.linear_utils import estimate_lambda_with_r2

    _, _, config = nu.load_everything("latent", exp_name)
    n_layers = config.model.num_layers
    seq_len = config.seq_len
    layer: int = n_layers - 1 if layer_index is None else layer_index

    if fit_positions is None:
        fit_positions = list(range(100, seq_len))

    fit_res = train_linear_hidden_predictor(
        exp_name=exp_name,
        layer=layer,
        n_samples=fit_n_samples,
        positions=fit_positions,
        sample_mode="major",
        n_minor=0,
        step=step,
        print_summary=False,
        skip_baselines=True,
    )
    W = fit_res["model_weight"]
    b_vec = fit_res["model_bias"]
    final_task_vecs = W - W.mean(dim=0, keepdim=True)

    hiddens, _k_minor, demo_data, sampler_clone = _get_hiddens_at_real_positions(
        task_name="latent", exp_name=exp_name,
        n_minor=0, n_ood=n_ood, B=B, step=step,
        return_data=True,
    )

    hiddens_layer = hiddens[layer].to(torch.float32)
    K, T, B_actual, _D = hiddens_layer.shape
    k_major = 3

    task_mean = hiddens_layer[:k_major].mean(dim=(0, 2)).unsqueeze(0)

    actual_endpoints = (
        hiddens_layer[:k_major].mean(dim=-2) - task_mean
    )[:, -1, :]
    ftv_norm = final_task_vecs.float().norm()
    ep_norm = actual_endpoints.float().norm()
    if ftv_norm > 0:
        final_task_vecs = final_task_vecs * (ep_norm / ftv_norm).item()

    def _project_onto_simplex(v):
        v = np.asarray(v, dtype=float)
        if v.ndim == 1:
            v = v[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False
        n = v.shape[1]
        u = np.sort(v, axis=1)[:, ::-1]
        cssv = np.cumsum(u, axis=1) - 1.0
        rho = np.zeros(v.shape[0], dtype=int)
        for i in range(v.shape[0]):
            conds = u[i] - cssv[i] / np.arange(1, n + 1) > 0
            rho[i] = np.where(conds)[0][-1]
        theta = cssv[np.arange(v.shape[0]), rho] / (rho + 1.0)
        w_out = np.maximum(v - theta[:, np.newaxis], 0.0)
        return w_out[0] if squeeze else w_out

    def _cosine_sim(a, b_arr):
        """Row-wise cosine similarity. a, b: (T, C) -> (T,)"""
        dot = (a * b_arr).sum(axis=-1)
        na = np.sqrt((a ** 2).sum(axis=-1)).clip(eps)
        nb = np.sqrt((b_arr ** 2).sum(axis=-1)).clip(eps)
        return dot / (na * nb)

    def _rolling_pearson(a, b_arr, w):
        """Per-component rolling Pearson r, averaged over components."""
        T_len, C = a.shape
        out = np.full(T_len, np.nan)
        for t in range(w - 1, T_len):
            corrs = []
            for c in range(C):
                x = a[t - w + 1:t + 1, c]
                y = b_arr[t - w + 1:t + 1, c]
                sx, sy = x.std(), y.std()
                if sx < eps or sy < eps:
                    continue
                mx, my = x.mean(), y.mean()
                dx, dy = x - mx, y - my
                denom = np.sqrt((dx ** 2).sum() * (dy ** 2).sum())
                corrs.append(float(np.dot(dx, dy) / denom))
            if corrs:
                out[t] = np.mean(corrs)
        return out

    def _mean_axis0_or_nan(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        if arr.shape[0] == 0:
            return np.full(arr.shape[1:], np.nan, dtype=float)
        return arr.mean(axis=0)

    def _nanmean_axis0_no_warning(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        valid = ~np.isnan(arr)
        count = valid.sum(axis=0)
        sumv = np.where(valid, arr, 0.0).sum(axis=0)
        out = np.full(sumv.shape, np.nan, dtype=float)
        np.divide(sumv, count, out=out, where=count > 0)
        return out

    post_all = task_posterior_over_time(
        sampler_clone,
        demo_data,
        include_minor=False,
    )  # (K, B, seq_len, 3)

    n_components = 3
    all_lam = np.zeros((K, B_actual, T, n_components), dtype=float)
    all_post = np.zeros((K, B_actual, T, n_components), dtype=float)

    tv_per_task = np.zeros((K, T), dtype=float)
    cos_per_task = np.zeros((K, T), dtype=float)

    for k in range(K):
        for bi in range(B_actual):
            h = hiddens_layer[k:k + 1, :, bi:bi + 1, :].squeeze(2)
            tv = h - task_mean

            lam_raw, _, _, _ = estimate_lambda_with_r2(
                final_task_vecs, tv, is_zero_mean=True,
            )
            lam_np = lam_raw if isinstance(lam_raw, np.ndarray) else np.asarray(lam_raw)
            lam_proj = _project_onto_simplex(lam_np[0])

            p = post_all[k, bi, :T, :]
            p_np = p.cpu().numpy() if torch.is_tensor(p) else np.asarray(p)

            all_lam[k, bi] = lam_proj
            all_post[k, bi] = p_np

            tv_per_task[k] += 0.5 * np.abs(p_np - lam_proj).sum(axis=-1)
            cos_per_task[k] += _cosine_sim(lam_proj, p_np)

        tv_per_task[k] /= B_actual
        cos_per_task[k] /= B_actual

    tv_major = _mean_axis0_or_nan(tv_per_task[:k_major])
    tv_ood = _mean_axis0_or_nan(tv_per_task[k_major:])
    cos_major = _mean_axis0_or_nan(cos_per_task[:k_major])
    cos_ood = _mean_axis0_or_nan(cos_per_task[k_major:])

    rcorr_per_task = np.full((K, T), np.nan)
    for k in range(K):
        sample_corrs = np.full((B_actual, T), np.nan)
        for bi in range(B_actual):
            sample_corrs[bi] = _rolling_pearson(
                all_lam[k, bi], all_post[k, bi], window,
            )
        rcorr_per_task[k] = _nanmean_axis0_no_warning(sample_corrs)

    rcorr_major = _nanmean_axis0_no_warning(rcorr_per_task[:k_major])
    rcorr_ood = _nanmean_axis0_no_warning(rcorr_per_task[k_major:])
    positions = np.arange(T)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    ax1 = axes[0]
    ax1.plot(positions, tv_major, linewidth=2.5, label="Major tasks", color="#1f77b4")
    ax1.plot(positions, tv_ood, linewidth=2.5, label="OOD tasks", color="#d62728")
    ax1.set_xlabel("Position", fontsize=18)
    ax1.set_ylabel("TV distance", fontsize=18)
    ax1.tick_params(labelsize=16)
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2 = axes[1]
    ax2.plot(positions, cos_major, linewidth=2.5, label="Major tasks", color="#1f77b4")
    ax2.plot(positions, cos_ood, linewidth=2.5, label="OOD tasks", color="#d62728")
    ax2.set_xlabel("Position", fontsize=18)
    ax2.set_ylabel("Cosine similarity", fontsize=18)
    ax2.tick_params(labelsize=16)
    ax2.legend(fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 1.02)

    ax3 = axes[2]
    vm = ~np.isnan(rcorr_major)
    vo = ~np.isnan(rcorr_ood)
    ax3.plot(positions[vm], rcorr_major[vm], linewidth=2.5,
             label="Major tasks", color="#1f77b4")
    ax3.plot(positions[vo], rcorr_ood[vo], linewidth=2.5,
             label="OOD tasks", color="#d62728")
    ax3.set_xlabel("Position", fontsize=18)
    ax3.set_ylabel(f"Rolling correlation (w={window})", fontsize=18)
    ax3.tick_params(labelsize=16)
    ax3.legend(fontsize=14)
    ax3.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=18)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close()

    return {
        'fig': fig,
        'axes': axes,
        'positions': positions,
        'tv_major': tv_major,
        'tv_ood': tv_ood,
        'cos_major': cos_major,
        'cos_ood': cos_ood,
        'rcorr_major': rcorr_major,
        'rcorr_ood': rcorr_ood,
        'tv_per_task': tv_per_task,
        'cos_per_task': cos_per_task,
        'rcorr_per_task': rcorr_per_task,
        'all_lam': all_lam,
        'all_post': all_post,
        'W': W,
        'b': b_vec,
    }


# ---------------------------------------------------------------------------
#  plot_id_ood_loss
# ---------------------------------------------------------------------------

def plot_id_ood_loss(
    k_list,
    vocab_size: Optional[int] = None,
    logx: bool = True,
    figsize: tuple = (12, 5),
    show: bool = True,
) -> dict:
    """
    Plot ID loss and OOD loss side by side for multiple latent experiments.

    Loads ``log.json`` from each latent experiment directory and plots
    ``eval/IDLoss`` (left) and ``eval/OODLoss`` (right) vs ``eval/step``.

    Parameters
    ----------
    k_list : sequence of int
        Values of k passed to ``get_exp_name("latent", k, vocab_size=...)``.
    vocab_size : int, optional
    logx : bool
        Use log scale for x-axis.
    figsize : tuple
    show : bool

    Returns
    -------
    dict
        ``{'fig', 'ax1', 'ax2', 'results'}``.
    """
    import json
    import os
    import matplotlib.pyplot as plt

    def _resolve_latent_exp_dir(exp_name: str) -> str:
        candidates = [
            os.path.join("results", "latent", exp_name),
            os.path.join("..", "results", "latent", exp_name),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0]

    results = {}
    for k in k_list:
        exp_name = get_exp_name("latent", k, vocab_size=vocab_size)
        n_minor_tasks = 2 ** k if k >= 0 else 0

        try:
            exp_dir = _resolve_latent_exp_dir(exp_name)
            log_path = os.path.join(exp_dir, "log.json")
            with open(log_path, "r") as f:
                data = json.load(f)

            train_steps = np.asarray(data.get("eval/step", data.get("train/step", [])), dtype=float)
            id_loss = data.get("eval/IDLoss", None)
            ood_loss = data.get("eval/OODLoss", None)
            if id_loss is None or ood_loss is None:
                raise KeyError("Missing eval/IDLoss or eval/OODLoss in latent log.json")

            id_loss = np.asarray(id_loss, dtype=float)
            ood_loss = np.asarray(ood_loss, dtype=float)

            if train_steps.size == 0:
                L = min(len(id_loss), len(ood_loss))
                train_steps = np.arange(1, L + 1, dtype=float)
            L = min(len(train_steps), len(id_loss), len(ood_loss))
            train_steps = train_steps[:L]
            id_loss = id_loss[:L]
            ood_loss = ood_loss[:L]

            results[k] = {
                "n_minor": n_minor_tasks,
                "train_steps": train_steps,
                "id_loss": id_loss,
                "ood_loss": ood_loss,
            }
        except Exception as e:
            logger.warning(f"Could not load k={k}: {e}")

    ks_sorted = sorted(results.keys())
    if not ks_sorted:
        logger.warning("No latent experiments loaded successfully.")
        return {}

    k_min, k_max = min(ks_sorted), max(ks_sorted)
    cmap = plt.get_cmap("viridis")
    color_map = {}
    for k in ks_sorted:
        if k_max > k_min:
            color_map[k] = cmap((k - k_min) / (k_max - k_min))
        else:
            color_map[k] = cmap(0.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for k in ks_sorted:
        d = results[k]
        c = color_map[k]
        xs, ys = d["train_steps"], d["id_loss"]
        if logx:
            mask = xs > 0
            xs, ys = xs[mask], ys[mask]
        if xs.size == 0:
            continue
        ax1.plot(xs, ys, color=c, linewidth=2.0)

    if logx:
        ax1.set_xscale("log")
    ax1.set_xlabel("Training Step", fontsize=16)
    ax1.set_ylabel("ID Loss", fontsize=16)
    ax1.tick_params(labelsize=14)
    ax1.grid(True, which="both", alpha=0.25)

    for k in ks_sorted:
        d = results[k]
        c = color_map[k]
        xs, ys = d["train_steps"], d["ood_loss"]
        if logx:
            mask = xs > 0
            xs, ys = xs[mask], ys[mask]
        if xs.size == 0:
            continue
        ax2.plot(xs, ys, color=c, linewidth=2.0)

    if logx:
        ax2.set_xscale("log")
    ax2.set_xlabel("Training Step", fontsize=16)
    ax2.set_ylabel("OOD Loss", fontsize=16)
    ax2.tick_params(labelsize=14)
    ax2.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    if show:
        plt.show()

    return {"fig": fig, "ax1": ax1, "ax2": ax2, "results": results}


# ---------------------------------------------------------------------------
#  plot_latent_task_posterior
# ---------------------------------------------------------------------------

def plot_latent_task_posterior(
    exp_name: str,
    n_plots: int = 3,
    major_only: bool = False,
    max_positions: Optional[int] = None,
    figsize: tuple = (12, 4),
    title: Optional[str] = None,
    show: bool = True,
) -> dict:
    """
    Generate random latent Markov samples and plot the Bayesian task
    posterior P(Z=k | X_{0:t}) over real-token positions.

    Non-padded counterpart of the legacy ``plot_latent_task_posterior``
    (in ``icl.latent_markov.legacy.unified_latent``).

    The posterior is always computed with the **original training prior**
    (same ``p_minor`` as was used during model training).

    Parameters
    ----------
    exp_name : str
        Experiment name (folder under ``results/latent/``).
    n_plots : int, default=3
        Number of independent samples to plot (one subplot each).
    major_only : bool, default=False
        If True, only major-task sequences are generated.  The posterior
        still tracks all tasks (major + minor) as hypotheses.
    max_positions : int, optional
        Cap on real-token positions to show per subplot.
    figsize : tuple, default=(12, 4)
        Figure size per subplot row (total height = ``figsize[1] * n_plots``).
    title : str, optional
        Custom suptitle.
    show : bool, default=True
        Whether to call ``plt.show()``.

    Returns
    -------
    info : dict
        ``{'posteriors': [...], 'fig': Figure, 'axes': [Axes, ...]}``.
    """
    import matplotlib.pyplot as plt

    _, sampler, config = nu.load_everything("latent", exp_name)
    T = sampler.total_trans
    device = config.device

    original_p_minor = getattr(sampler, "p_minor", 0.0)

    # For sampling: suppress minor tasks when major_only is requested
    if major_only:
        sampler.p_minor = 0.0

    samples_raw = sampler.generate(mode="train", num_samples=n_plots, epochs=1)
    if isinstance(samples_raw, tuple):
        samples_raw = samples_raw[0]
    if samples_raw.dim() == 3:
        samples_raw = samples_raw.squeeze(0)
    samples = samples_raw.to(device)

    # Restore original p_minor for posterior computation so the prior
    # matches the training distribution.
    sampler.p_minor = original_p_minor

    posterior_all = task_posterior_over_time(sampler, samples)  # (n_plots, L_real, T)

    n_major = sampler.n_major_tasks
    n_minor_tasks = sampler.n_minor_tasks
    major_cmap = plt.cm.Blues
    minor_cmap = plt.cm.Reds
    major_colors = [
        major_cmap(0.3 + 0.6 * i / max(n_major - 1, 1)) for i in range(n_major)
    ]
    minor_colors = [
        minor_cmap(0.3 + 0.6 * i / max(n_minor_tasks - 1, 1))
        for i in range(n_minor_tasks)
    ]
    T_plot = T

    fig, axes = plt.subplots(
        n_plots, 1,
        figsize=(figsize[0], figsize[1] * n_plots),
        squeeze=False,
    )
    axes = [axes[i, 0] for i in range(n_plots)]
    posteriors_out = []

    for idx, ax in enumerate(axes):
        posterior = posterior_all[idx].cpu()
        if max_positions is not None:
            posterior = posterior[:max_positions]
        L_real = posterior.shape[0]
        posteriors_out.append(posterior)

        x_axis = torch.arange(L_real)
        major_labeled = False
        minor_labeled = False
        for k in range(T_plot):
            if k < n_major:
                color = major_colors[k]
                label = ("Major" if not major_labeled else None) if idx == 0 else None
                major_labeled = True
            else:
                color = minor_colors[k - n_major]
                label = ("Minor" if not minor_labeled else None) if idx == 0 else None
                minor_labeled = True
            ax.plot(
                x_axis.numpy(), posterior[:, k].numpy(),
                label=label, color=color, alpha=0.8, linewidth=1.5,
            )

        ax.set_ylabel("P(Z=k | obs)")
        ax.set_xlim(0, max(L_real - 1, 1))
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Sample {idx + 1}", fontsize=10)
        if idx == 0:
            ax.legend(
                bbox_to_anchor=(1.05, 1), loc="upper left",
                fontsize="small", ncol=max(1, T_plot // 20),
            )

    axes[-1].set_xlabel("Real-token position")
    if title is None:
        suffix = " (major only)" if major_only else ""
        title = f"Latent task posterior over time — {exp_name}{suffix}"
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)

    sampler.p_minor = original_p_minor

    return {
        "posteriors": posteriors_out,
        "fig": fig,
        "axes": axes,
    }
