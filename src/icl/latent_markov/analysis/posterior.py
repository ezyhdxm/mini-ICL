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
    fit_include_position_bias: bool = True,
    fit_include_logit: bool = True,
    window: int = 20,
    show_random_baseline: bool = True,
    random_baseline_draws: int = 32,
    random_seed: int = 0,
    show_task_basis_baseline: bool = True,
    major_only: bool = False,
    min_position: Optional[int] = None,
    max_position: Optional[int] = None,
    figsize: tuple = (10, 4),
    show: bool = True,
    title: str = "",
    eps: float = 1e-12,
) -> dict:
    """
    Agreement between projected coefficients and latent Bayesian posterior,
    measured by Hellinger distance, with optional baselines.

    Latent counterpart of
    ``plot_lambda_posterior_agreement_coin_nonpadded``.

    Notes
    -----
    For OOD tasks, posterior is computed under a *major-only* model
    (``include_minor=False``), i.e. misspecified inference over the 3
    anchor tasks.

    This implementation avoids ``Mean of empty slice`` warnings for empty/NaN
    OOD baseline blocks.
    """
    import matplotlib.pyplot as plt
    from icl.linear.linear_utils import estimate_lambda_with_r2

    if major_only:
        n_ood = 0

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
        include_position_bias=fit_include_position_bias,
        include_logit=fit_include_logit,
        sample_mode="major",
        n_minor=0,
        step=step,
        print_summary=False,
        skip_baselines=True,
    )
    W = fit_res["model_weight"]
    b_vec = fit_res["model_bias"]
    W_mean = W.mean(dim=0, keepdim=True)

    hiddens, _k_minor, demo_data, sampler_clone = _get_hiddens_at_real_positions(
        task_name="latent", exp_name=exp_name,
        n_minor=0, n_ood=n_ood, B=B, step=step,
        return_data=True,
    )

    hiddens_layer = hiddens[layer].to(torch.float32)
    K, T, B_actual, D = hiddens_layer.shape
    k_major = 3
    dev = hiddens_layer.device

    b_f = b_vec.float().to(dev)
    W_mean_f = W_mean.float().to(dev)
    W_c = (W - W_mean).float().to(dev)

    def _to_float_device(t):
        if isinstance(t, torch.Tensor) and t.numel() > 0:
            return t.float().to(dev)
        return torch.zeros((0, D), dtype=torch.float32, device=dev)

    W_tok_f = _to_float_device(fit_res.get("token_weight", None))
    W_logit_f = _to_float_device(fit_res.get("logit_weight", None))
    W_pos_f = _to_float_device(fit_res.get("position_weight", None))
    fit_pos_list = [int(p) for p in fit_res.get("positions", fit_positions)]
    fit_pos_to_col = {p: i for i, p in enumerate(fit_pos_list)}

    model_for_logits = None
    model_device = None
    if W_logit_f.shape[0] > 0:
        _, _, config, *_ = nu.load_everything("latent", exp_name)
        _step = step if step is not None else config.training.num_epochs
        model_for_logits, _ = nu.load_checkpoint(
            config, step=_step, exp_name=exp_name, return_actual_step=True,
        )
        model_device = next(model_for_logits.parameters()).device

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

    def _safe_mean(arr, axis=0):
        arr = np.asarray(arr, dtype=float)
        if arr.shape[axis] == 0:
            shape = list(arr.shape)
            shape.pop(axis)
            return np.full(shape, np.nan, dtype=float)
        with np.errstate(all="ignore"):
            return np.nanmean(arr, axis=axis)

    post_all = task_posterior_over_time(
        sampler_clone,
        demo_data,
        include_minor=False,
    )  # (K, B, seq_len, 3)

    n_components = 3
    all_lam = np.zeros((K, B_actual, T, n_components), dtype=float)
    all_post = np.zeros((K, B_actual, T, n_components), dtype=float)

    tv_per_task = np.zeros((K, T), dtype=float)
    tv_random_per_task = np.zeros((K, T), dtype=float)
    tv_task_basis_per_task = np.full((K, T), np.nan, dtype=float)
    rng = np.random.default_rng(random_seed)

    for k in range(K):
        h_tid = hiddens_layer[k].permute(1, 0, 2).contiguous()  # (B, T, D)
        nuisance = torch.zeros_like(h_tid)

        if W_tok_f.shape[0] > 0:
            tok_k = demo_data[k, :B_actual, :T].long().to(dev)
            nuisance = nuisance + W_tok_f[tok_k]

        if W_logit_f.shape[0] > 0 and model_for_logits is not None:
            with torch.no_grad():
                samples_k = demo_data[k, :B_actual].to(device=model_device)
                logits_k = model_for_logits(samples_k).float()[:, :T, :W_logit_f.shape[0]]
            nuisance = nuisance + torch.einsum(
                "btd,df->btf", logits_k.to(dev), W_logit_f,
            )

        if W_pos_f.shape[0] > 0:
            pos_effect = torch.zeros((T, D), dtype=torch.float32, device=dev)
            for t in range(T):
                j = fit_pos_to_col.get(t, None)
                if j is not None and j < W_pos_f.shape[0]:
                    pos_effect[t] = W_pos_f[j]
            nuisance = nuisance + pos_effect.unsqueeze(0)

        h_adj = h_tid - b_f.unsqueeze(0).unsqueeze(0) - nuisance

        lam_raw, _, _, _ = estimate_lambda_with_r2(
            W_c, h_adj - W_mean_f.unsqueeze(0), is_zero_mean=True,
        )
        lam_np = lam_raw if isinstance(lam_raw, np.ndarray) else np.asarray(lam_raw)
        lam_proj = _project_onto_simplex(lam_np.reshape(-1, lam_np.shape[-1])).reshape(lam_np.shape)

        for bi in range(B_actual):
            p = post_all[k, bi, :T, :]
            p_np = p.cpu().numpy() if torch.is_tensor(p) else np.asarray(p)

            all_lam[k, bi] = lam_proj[bi]
            all_post[k, bi] = p_np

            tv_per_task[k] += np.sqrt(0.5 * ((np.sqrt(np.maximum(p_np, 0)) - np.sqrt(np.maximum(lam_proj[bi], 0))) ** 2).sum(axis=-1))
            if show_random_baseline and random_baseline_draws > 0:
                rand_lam = rng.dirichlet(
                    np.ones(n_components, dtype=float),
                    size=(random_baseline_draws, T),
                )
                tv_rand = np.sqrt(0.5 * ((np.sqrt(rand_lam) - np.sqrt(np.maximum(p_np[None, :, :], 0))) ** 2).sum(axis=-1)).mean(axis=0)
                tv_random_per_task[k] += tv_rand
            if show_task_basis_baseline and k < k_major:
                basis = np.zeros((T, n_components), dtype=float)
                basis[:, k] = 1.0
                tv_basis = np.sqrt(0.5 * ((np.sqrt(np.maximum(p_np, 0)) - np.sqrt(basis)) ** 2).sum(axis=-1))
                if np.isnan(tv_task_basis_per_task[k]).all():
                    tv_task_basis_per_task[k] = 0.0
                tv_task_basis_per_task[k] += tv_basis

        tv_per_task[k] /= B_actual
        if show_random_baseline and random_baseline_draws > 0:
            tv_random_per_task[k] /= B_actual
        if show_task_basis_baseline and k < k_major:
            tv_task_basis_per_task[k] /= B_actual

    tv_major = _safe_mean(tv_per_task[:k_major])
    tv_ood = _safe_mean(tv_per_task[k_major:])
    if show_random_baseline and random_baseline_draws > 0:
        tv_random_major = _safe_mean(tv_random_per_task[:k_major])
        tv_random_ood = _safe_mean(tv_random_per_task[k_major:])
    else:
        tv_random_major = None
        tv_random_ood = None
    if show_task_basis_baseline:
        major_block = tv_task_basis_per_task[:k_major]
        ood_block = tv_task_basis_per_task[k_major:]
        tv_task_basis_major = (
            np.nanmean(major_block, axis=0)
            if major_block.size > 0 and np.isfinite(major_block).any()
            else None
        )
        tv_task_basis_ood = (
            np.nanmean(ood_block, axis=0)
            if ood_block.size > 0 and np.isfinite(ood_block).any()
            else None
        )
    else:
        tv_task_basis_major = None
        tv_task_basis_ood = None
    positions = np.arange(T)
    _sfx = "" if major_only else " major"

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(positions, tv_major, linewidth=2.5,
            label=r"$\lambda(t)$ vs $\alpha_t$" + (f" {_sfx}" if _sfx else ""),
            color="#1f77b4", marker="o", markersize=4)
    if not major_only:
        ax.plot(positions, tv_ood, linewidth=2.5,
                label=r"$\lambda(t)$ vs $\alpha_t$ OOD",
                color="#d62728", marker="s", markersize=4)
    if tv_random_major is not None:
        ax.plot(
            positions, tv_random_major, linewidth=2.0, ls="--",
            label="uniform-random" + (f" {_sfx}" if _sfx else ""),
            color="#7eb8da", marker="^", markersize=3,
        )
    if not major_only and tv_random_ood is not None:
        ax.plot(
            positions, tv_random_ood, linewidth=2.0, ls="--",
            label="uniform-random OOD", color="#e8836b",
            marker="v", markersize=3,
        )
    if tv_task_basis_major is not None:
        ax.plot(
            positions, tv_task_basis_major, linewidth=2.0, ls=":",
            label="task-basis" + (f" {_sfx}" if _sfx else ""),
            color="#2ca02c", marker="D", markersize=3,
        )
    if not major_only and tv_task_basis_ood is not None and np.isfinite(tv_task_basis_ood).any():
        ax.plot(
            positions, tv_task_basis_ood, linewidth=2.0, ls=":",
            label="task-basis OOD", color="#9467bd",
            marker="d", markersize=3,
        )
    ax.set_xlabel("Position", fontsize=18)
    ax.set_ylabel("Hellinger distance", fontsize=18)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    _xlim_lo = min_position if min_position is not None else 0
    _xlim_hi = max_position if max_position is not None else None
    ax.set_xlim(_xlim_lo, _xlim_hi)

    if title:
        fig.suptitle("", fontsize=18)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close()

    return {
        'fig': fig,
        'ax': ax,
        'positions': positions,
        'tv_major': tv_major,
        'tv_ood': tv_ood,
        'tv_random_major': tv_random_major,
        'tv_random_ood': tv_random_ood,
        'tv_task_basis_major': tv_task_basis_major,
        'tv_task_basis_ood': tv_task_basis_ood,
        'tv_per_task': tv_per_task,
        'tv_random_per_task': tv_random_per_task if show_random_baseline else None,
        'tv_task_basis_per_task': tv_task_basis_per_task if show_task_basis_baseline else None,
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

    # Evenly sample colormap (avoid dark purples): use 15%--90% of viridis
    nk = len(ks_sorted)
    cmap = plt.get_cmap("viridis")
    color_map = {}
    for i, k in enumerate(ks_sorted):
        t = 0.15 + 0.75 * (i / max(1, nk - 1))
        color_map[k] = cmap(t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, which="major", alpha=0.15, linestyle="-")
        ax.grid(True, which="minor", alpha=0.06, linestyle=":")

    lw, alpha = 1.5, 0.85
    fs_label, fs_tick = 14, 12

    for k in ks_sorted:
        d = results[k]
        c = color_map[k]
        xs, ys = d["train_steps"], d["id_loss"]
        if logx:
            mask = xs > 0
            xs, ys = xs[mask], ys[mask]
        if xs.size == 0:
            continue
        ax1.plot(xs, ys, color=c, linewidth=lw, alpha=alpha, label=str(k))

    ax1.set_title("In-distribution", fontsize=fs_label)
    if logx:
        ax1.set_xscale("log")
    ax1.set_xlabel("Training Step", fontsize=fs_label)
    ax1.set_ylabel("Loss (KL)", fontsize=fs_label)
    ax1.tick_params(labelsize=fs_tick)

    handles, labels = ax1.get_legend_handles_labels()
    ax2.legend(
        handles,
        labels,
        title=r"$\log_2(n_{\mathrm{minor}})$",
        fontsize=fs_tick,
        title_fontsize=fs_label,
        frameon=True,
        framealpha=0.95,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    for k in ks_sorted:
        d = results[k]
        c = color_map[k]
        xs, ys = d["train_steps"], d["ood_loss"]
        if logx:
            mask = xs > 0
            xs, ys = xs[mask], ys[mask]
        if xs.size == 0:
            continue
        ax2.plot(xs, ys, color=c, linewidth=lw, alpha=alpha)

    ax2.set_title("Out-of-distribution", fontsize=fs_label)
    if logx:
        ax2.set_xscale("log")
    ax2.set_xlabel("Training Step", fontsize=fs_label)
    ax2.set_ylabel("")
    ax2.tick_params(labelsize=fs_tick)

    fig.subplots_adjust(wspace=0.05, right=0.78)
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

    Non-padded counterpart of the original ``plot_latent_task_posterior``.

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
    fig.suptitle("", fontsize=18, y=1.01)
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
