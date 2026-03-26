"""Posterior plotting and agreement analysis for the Coin task."""

import os
from typing import Optional, Sequence

import numpy as np
import torch

import icl.utils.notebook_utils as nu
from icl.coin.analysis.probes import train_linear_hidden_predictor_coin
from icl.utils.logger import setup_logger
from icl.utils.unified_path_finder import _get_exp_dir

logger = setup_logger(__name__)


def plot_coin_task_posterior(
    exp_name: str,
    n_plots: int = 3,
    uniform_prior: bool = True,
    major_only: bool = False,
    max_positions: Optional[int] = None,
    figsize: tuple = (12, 4),
    title: Optional[str] = None,
) -> dict:
    """
    Generate random coin samples and plot the task posterior over time.

    Plot task posterior for coin sequences.

    All tokens are real — ``samples.long()`` is used directly.

    Parameters
    ----------
    exp_name : str
        Experiment name (folder under ``results/coin/``).
    n_plots : int
        Number of independent samples to plot.
    uniform_prior : bool
        If True, sets ``p_minor`` for uniform *sampling* so all tasks appear
        equally often.  The posterior prior always uses the training distribution.
    major_only : bool
        If True, sample only from major tasks.
    max_positions : int, optional
        Cap on real-token positions to show per subplot.
    figsize : tuple
        Figure size per subplot row.
    title : str, optional
        Custom suptitle.

    Returns
    -------
    info : dict
        ``{'posteriors': [...], 'fig': Figure, 'axes': [Axes, ...]}``.
    """
    import matplotlib.pyplot as plt

    _, sampler, config = nu.load_everything("coin", exp_name)
    device = config.device

    # Save training distribution for posterior computation.
    train_p_minor = getattr(sampler, 'p_minor', 0.0)

    if uniform_prior and sampler.n_minor_tasks > 0:
        sampler.p_minor = sampler.n_minor_tasks / (sampler.n_major_tasks + sampler.n_minor_tasks)

    # Generate samples — non-padded, so shape is (B, seq_len)
    sample_mode = "major" if major_only else "train"
    samples_raw = sampler.generate(mode=sample_mode, num_samples=n_plots, epochs=1)
    if isinstance(samples_raw, tuple):
        samples_raw = samples_raw[0]
    if samples_raw.dim() == 3:
        samples_raw = samples_raw.squeeze(0)
    samples = samples_raw.to(device)

    # Restore training p_minor so posterior uses the training prior.
    sampler.p_minor = train_p_minor

    # Build coin probability pools and prior from *training* distribution.
    K = sampler.num_states
    maj = sampler.major_p.to(device=device, dtype=torch.float32)
    Kmaj = maj.shape[0]
    use_minor = sampler.n_minor_tasks > 0 and sampler.minor_p is not None
    if use_minor:
        minp = sampler.minor_p.to(device=device, dtype=torch.float32)
        Kmin = minp.shape[0]
        P = torch.cat([maj, minp], dim=0)
    else:
        Kmin = 0
        P = maj
    Ktot = P.shape[0]

    eps = 1e-30
    if Kmin == 0:
        prior = torch.full((Ktot,), 1.0 / max(1, Kmaj), device=device, dtype=torch.float32)
    else:
        p0 = float(train_p_minor)
        prior_major = (1.0 - p0) / max(1, Kmaj)
        prior_minor = p0 / max(1, Kmin)
        prior = torch.cat([
            torch.full((Kmaj,), prior_major, device=device, dtype=torch.float32),
            torch.full((Kmin,), prior_minor, device=device, dtype=torch.float32),
        ])
    prior = torch.clamp(prior, min=eps)
    log_prior = prior.log()
    logP = torch.log(torch.clamp(P, min=eps))  # (Ktot, K)

    # All tokens are real — no ::2 slicing needed
    x = samples.long()  # (B, seq_len)
    B_actual, L_real_full = x.shape
    posterior_time = torch.empty((B_actual, L_real_full, Ktot), device=device, dtype=torch.float32)
    counts = torch.zeros((B_actual, K), device=device, dtype=torch.float32)

    for t in range(L_real_full):
        counts.scatter_add_(
            dim=-1, index=x[:, t:t + 1],
            src=torch.ones((B_actual, 1), device=device, dtype=torch.float32),
        )
        loglik = counts @ logP.T
        unnorm = loglik + log_prior.unsqueeze(0)
        log_post = unnorm - torch.logsumexp(unnorm, dim=-1, keepdim=True)
        posterior_time[:, t, :] = torch.exp(log_post)

    # ---- plotting ----
    n_major = sampler.n_major_tasks
    n_minor_tasks = sampler.n_minor_tasks
    major_cmap = plt.cm.Blues
    minor_cmap = plt.cm.Reds
    major_colors = [major_cmap(0.3 + 0.6 * i / max(n_major - 1, 1)) for i in range(n_major)]
    minor_colors = [minor_cmap(0.3 + 0.6 * i / max(n_minor_tasks - 1, 1)) for i in range(n_minor_tasks)]
    T_plot = Ktot

    fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], figsize[1] * n_plots), squeeze=False)
    axes = [axes[i, 0] for i in range(n_plots)]
    posteriors_out = []

    for idx, ax in enumerate(axes):
        posterior = posterior_time[idx].cpu()
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
            ax.plot(x_axis.numpy(), posterior[:, k].numpy(), label=label,
                    color=color, alpha=0.8, linewidth=1.5)

        ax.set_ylabel("P(Z=k | obs)")
        ax.set_xlim(0, max(L_real - 1, 1))
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("", fontsize=14)
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small',
                      ncol=max(1, T_plot // 20))

    axes[-1].set_xlabel("Real-token position")
    if title is None:
        title = f"Coin task posterior over time — {exp_name}"
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    plt.show()

    return {
        'posteriors': posteriors_out,
        'fig': fig,
        'axes': axes,
    }


def plot_id_ood_loss_coin(
    k_list: Sequence[int],
    vocab_size: int = 8,
    logx: bool = True,
    figsize: tuple = (12, 5),
    show: bool = True,
) -> dict:
    """
    Plot ID loss and OOD loss side by side for multiple coin experiments.

    Loads ``log.json`` from each experiment directory and plots
    ``eval/IDLoss`` (left panel) and ``eval/OODLoss`` (right panel) vs
    ``eval/step``. One line per k value.

    Parameters
    ----------
    k_list : sequence of int
        Values of k passed to ``get_exp_name("coin", k)``.
    vocab_size : int
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
    import matplotlib.pyplot as plt
    from icl.utils.unified_interface import get_exp_name

    results = {}
    for k in k_list:
        exp_name = get_exp_name("coin", k, vocab_size=vocab_size)
        n_minor_tasks = 2 ** k if k >= 0 else 0

        try:
            _, _, config = nu.load_everything("coin", exp_name)
            exp_dir = _get_exp_dir(config, exp_name)
            log_path = os.path.join(exp_dir, "log.json")
            with open(log_path, "r") as f:
                data = json.load(f)
            train_steps = np.asarray(data["eval/step"], dtype=float)
            id_loss = np.asarray(data["eval/IDLoss"], dtype=float)
            ood_loss = np.asarray(data["eval/OODLoss"], dtype=float)

            results[k] = dict(
                n_minor=n_minor_tasks,
                train_steps=train_steps,
                id_loss=id_loss,
                ood_loss=ood_loss,
            )
        except Exception as e:
            logger.warning(f"Could not load k={k}: {e}")

    ks_sorted = sorted(results.keys())
    if not ks_sorted:
        logger.warning("No experiments loaded successfully.")
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
        xs, y_id = d["train_steps"], d["id_loss"]
        if logx:
            mask = xs > 0
            xs, y_id = xs[mask], y_id[mask]
        if xs.size == 0:
            continue
        ax1.plot(xs, y_id, color=c, linewidth=lw, alpha=alpha, label=str(k))

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
        xs, y_ood = d["train_steps"], d["ood_loss"]
        if logx:
            mask = xs > 0
            xs, y_ood = xs[mask], y_ood[mask]
        if xs.size == 0:
            continue
        ax2.plot(xs, y_ood, color=c, linewidth=lw, alpha=alpha)

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


def plot_kl_lambda_vs_posterior_coin(
    exp_name: str,
    layer_index: Optional[int] = None,
    n_ood: int = 30,
    B: int = 64,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    figsize: tuple = (10, 6),
    show: bool = True,
    title: str = "",
    eps: float = 1e-12,
) -> dict:
    """
    KL divergence between projected coefficients and true Bayesian posterior,
    averaged over samples, plotted versus position for major and OOD tasks.

    For each task (3 major + ``n_ood`` OOD) and each of the ``B`` batch
    samples, computes the simplex-projected coefficients
    ``lambda(t)`` at every position and the KL divergence
    ``KL(posterior || lambda)`` from the true (misspecified for OOD) Bayesian
    posterior.  The per-position KL is averaged over samples and then over
    tasks within each group (major / OOD), yielding two curves.

    Parameters
    ----------
    exp_name : str
    layer_index : int, optional
        ``None`` -> last layer.
    n_ood : int
    B : int
    step : int, optional
    fit_n_samples : int
    fit_positions : list, optional
        ``None`` -> ``range(100, seq_len)``.
    figsize : tuple
    show : bool
    title : str
    eps : float

    Returns
    -------
    dict
        ``{'fig', 'ax', 'positions', 'kl_major', 'kl_ood',
        'kl_major_all', 'kl_ood_all', 'W', 'b'}``.
    """
    import matplotlib.pyplot as plt
    from icl.utils.unified_interface import _get_hiddens_at_real_positions
    from icl.linear.linear_utils import estimate_lambda_with_r2
    from icl.utils.unified_plot import posterior_over_models_over_time_per_sample

    _, _, config = nu.load_everything("coin", exp_name)
    n_layers = config.model.num_layers
    seq_len = config.seq_len
    if layer_index is None:
        layer_index = n_layers - 1

    if fit_positions is None:
        fit_positions = list(range(100, seq_len))

    fit_res = train_linear_hidden_predictor_coin(
        exp_name=exp_name,
        layer=layer_index,
        n_samples=fit_n_samples,
        positions=fit_positions,
        sample_mode="major",
        step=step,
        print_summary=False,
        skip_baselines=True,
    )
    W = fit_res["model_weight"]  # (3, D)
    b_vec = fit_res["model_bias"]

    final_task_vecs = W - W.mean(dim=0, keepdim=True)  # (3, D)

    hiddens, k_minor, demo_data, sampler_clone = _get_hiddens_at_real_positions(
        task_name="coin", exp_name=exp_name,
        n_minor=0, n_ood=n_ood, B=B, step=step,
        return_data=True,
    )

    hiddens_layer = hiddens[layer_index].to(torch.float32)  # (K, T, B, D)
    K, T, B_actual, D = hiddens_layer.shape
    k_major = 3

    task_mean = hiddens_layer[:k_major].mean(dim=(0, 2)).unsqueeze(0)  # (1, T, D)

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
        w = np.maximum(v - theta[:, np.newaxis], 0.0)
        return w[0] if squeeze else w

    def _kl(p, q):
        """KL(p || q) for rows of p, q."""
        p = np.clip(p, eps, None)
        q = np.clip(q, eps, None)
        return np.sum(p * np.log(p / q), axis=-1)

    post_all = posterior_over_models_over_time_per_sample(
        demo_data, sampler_clone.major_p,
    )  # (K, B, seq_len, 3)

    # kl_per_task[k] = (T,) averaged over B samples
    kl_per_task = np.zeros((K, T), dtype=float)

    for k in range(K):
        kl_samples = np.zeros((B_actual, T), dtype=float)
        for bi in range(B_actual):
            h = hiddens_layer[k:k+1, :, bi:bi+1, :]  # (1, T, 1, D)
            h = h.squeeze(2)  # (1, T, D)
            tv = h - task_mean  # (1, T, D)

            lam_raw, _, _, _ = estimate_lambda_with_r2(
                final_task_vecs, tv, is_zero_mean=True,
            )
            lam_np = lam_raw if isinstance(lam_raw, np.ndarray) else np.asarray(lam_raw)
            lam_proj = _project_onto_simplex(lam_np[0])  # (T, 3)

            p = post_all[k, bi, :T, :]  # (T, 3)
            p_np = p.cpu().numpy() if torch.is_tensor(p) else np.asarray(p)

            kl_samples[bi] = _kl(p_np, lam_proj)

        kl_per_task[k] = kl_samples.mean(axis=0)

    kl_major = kl_per_task[:k_major].mean(axis=0)  # (T,)
    kl_ood = kl_per_task[k_major:].mean(axis=0)    # (T,)
    positions = np.arange(T)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(positions, kl_major, linewidth=2.5, label="Major tasks", color="#1f77b4")
    ax.plot(positions, kl_ood, linewidth=2.5, label="OOD tasks", color="#d62728")
    ax.set_xlabel("Position", fontsize=18)
    ax.set_ylabel(r"KL( posterior $\|$ $\lambda$ )", fontsize=18)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title("", fontsize=18)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close()

    return {
        'fig': fig,
        'ax': ax,
        'positions': positions,
        'kl_major': kl_major,
        'kl_ood': kl_ood,
        'kl_per_task': kl_per_task,
        'W': W,
        'b': b_vec,
    }


def plot_lambda_posterior_agreement_coin(
    exp_name: str,
    layer_index: Optional[int] = None,
    n_ood: int = 30,
    B: int = 64,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    fit_include_position_bias: bool = True,
    fit_include_logit: bool = False,
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
    Agreement between projected coefficients and Bayesian posterior,
    measured by Hellinger distance.

    Plots:
    ``H(t) = sqrt(0.5 * sum_k (sqrt(lambda_k(t)) - sqrt(P(Z=k|x_{1:t})))^2)``
    averaged over samples and tasks, with optional random-simplex baseline.

    Parameters
    ----------
    exp_name : str
    layer_index : int, optional
        ``None`` -> last layer.
    n_ood : int
    B : int
    step : int, optional
    fit_n_samples : int
    fit_positions : list, optional
        ``None`` -> ``range(100, seq_len)``.
    fit_include_position_bias : bool
        Whether to include one-hot position nuisance features in probe fitting.
    fit_include_logit : bool
        Reserved for API consistency. Coin hidden probe has no logit block,
        so this must stay ``False``.
    window : int
        Deprecated (kept for backward compatibility; unused).
    show_random_baseline : bool
        If True, overlays random-simplex Hellinger baselines.
    random_baseline_draws : int
        Number of random lambda draws per sample/time for baseline TV.
    random_seed : int
        Seed for random baseline generation.
    show_task_basis_baseline : bool
        If True, overlays a baseline that always outputs the one-hot
        standard basis vector of the underlying task (major tasks only).
    max_position : int, optional
        If set, restrict the x-axis display to ``[0, max_position]``.
    figsize : tuple
    show : bool
    title : str
    eps : float

    Returns
    -------
    dict
        ``{'fig', 'ax', 'positions', 'tv_major', 'tv_ood',
        'tv_random_major', 'tv_random_ood',
        'tv_task_basis_major', 'tv_task_basis_ood', ...}``.
    """
    import matplotlib.pyplot as plt
    from icl.utils.unified_interface import _get_hiddens_at_real_positions
    from icl.linear.linear_utils import estimate_lambda_with_r2
    from icl.utils.unified_plot import posterior_over_models_over_time_per_sample

    if major_only:
        n_ood = 0

    _, _, config = nu.load_everything("coin", exp_name)
    n_layers = config.model.num_layers
    seq_len = config.seq_len
    if layer_index is None:
        layer_index = n_layers - 1

    if fit_positions is None:
        fit_positions = list(range(100, seq_len))
    if fit_include_logit:
        raise ValueError(
            "Coin hidden probe does not include a logit feature block; "
            "use fit_include_logit=False."
        )

    fit_res = train_linear_hidden_predictor_coin(
        exp_name=exp_name,
        layer=layer_index,
        n_samples=fit_n_samples,
        positions=fit_positions,
        include_position_bias=fit_include_position_bias,
        sample_mode="major",
        step=step,
        print_summary=False,
        skip_baselines=True,
    )
    W = fit_res["model_weight"]
    b_vec = fit_res["model_bias"]
    final_task_vecs = W - W.mean(dim=0, keepdim=True)

    hiddens, k_minor, demo_data, sampler_clone = _get_hiddens_at_real_positions(
        task_name="coin", exp_name=exp_name,
        n_minor=0, n_ood=n_ood, B=B, step=step,
        return_data=True,
    )

    hiddens_layer = hiddens[layer_index].to(torch.float32)
    K, T, B_actual, D = hiddens_layer.shape
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

    post_all = posterior_over_models_over_time_per_sample(
        demo_data, sampler_clone.major_p,
    )

    n_components = 3
    all_lam = np.zeros((K, B_actual, T, n_components), dtype=float)
    all_post = np.zeros((K, B_actual, T, n_components), dtype=float)

    tv_per_task = np.zeros((K, T), dtype=float)
    tv_random_per_task = np.zeros((K, T), dtype=float)
    tv_task_basis_per_task = np.full((K, T), np.nan, dtype=float)
    rng = np.random.default_rng(random_seed)

    for k in range(K):
        for bi in range(B_actual):
            h = hiddens_layer[k:k+1, :, bi:bi+1, :].squeeze(2)
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

            tv_per_task[k] += np.sqrt(0.5 * ((np.sqrt(np.maximum(p_np, 0)) - np.sqrt(np.maximum(lam_proj, 0))) ** 2).sum(axis=-1))
            if show_random_baseline and random_baseline_draws > 0:
                rand_lam = rng.dirichlet(
                    np.ones(n_components, dtype=float),
                    size=(random_baseline_draws, T),
                )  # (R, T, C)
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

    tv_major = tv_per_task[:k_major].mean(axis=0)
    tv_ood = tv_per_task[k_major:].mean(axis=0)
    if show_random_baseline and random_baseline_draws > 0:
        tv_random_major = tv_random_per_task[:k_major].mean(axis=0)
        tv_random_ood = tv_random_per_task[k_major:].mean(axis=0)
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

    # ---- plot ----
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
