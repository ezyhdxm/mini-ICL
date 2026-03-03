"""Trajectory projection plots for the Coin task."""

import numpy as np
import torch
from typing import Optional

import icl.utils.notebook_utils as nu
from icl.coin.analysis.probes import train_linear_hidden_predictor_coin


def traj_projection_plot_coin(
    task_name: str,
    exp_name: str,
    n_minor: int = 64,
    n_ood: int = 30,
    B: int = 64,
    b_maj: int = 2,
    b_minor: int = 0,
    b_ood: int = 2,
    layer_index: Optional[int] = None,
    minor_projection: bool = False,
    use_mean: bool = True,
    step: Optional[int] = None,
    show_legend: bool = True,
    figsize: tuple = (9, 7),
    show: bool = True,
    title: str = "",
    **kwargs,
) -> dict:
    """
    2-D trajectory projection plot on **non-padded** hidden representations.

    Non-padded counterpart of ``traj_projection_plot`` (in ``unified_plot.py``).
    Supports ``"coin"``, ``"latent"``, and ``"linear"`` tasks.

    Steps:
      1. Extract hiddens via ``_get_hiddens_at_real_positions``.
      2. Select the requested layer → ``(K, T, B, D)``.
      3. Compute task vectors (batch-mean minus global mean of first 3 tasks).
      4. Estimate λ / R² via ``estimate_lambda_with_r2``.
      5. Project onto 2-D plane defined by the 3 major-task vectors and plot
         trajectories with group colours.

    Compared with the padded version:
      - Latent hiddens are ``(K, T, B, D)`` (no ``V`` / vocab axis), so the
        ``voc`` parameter is not needed.
      - Coin / linear hiddens have the same shape as padded, just extracted at
        real-token positions instead of padding positions.

    Parameters
    ----------
    task_name : str
        ``"coin"``, ``"latent"``, or ``"linear"``.
    exp_name : str
    n_minor : int
        Capped at ``sampler.n_minor_tasks`` for coin/latent.
    n_ood : int
    B : int
    b_maj, b_minor, b_ood : int
        Per-task random batch trajectories when ``use_mean=False``.
    layer_index : int, optional
        ``None`` → last layer.
    minor_projection : bool
    use_mean : bool
        ``True``:  batch-averaged task vectors.
        ``False``: individual batch trajectories.
    step : int, optional
    show_legend : bool
    figsize : tuple
    show : bool
    title : str
    **kwargs
        ``return_p=True`` — return coin probability pools (coin only).
        ``return_data=True`` — return demo data; enables posterior colouring
        for individual OOD trajectories (coin only).

    Returns
    -------
    dict
        ``{'fig', 'ax', 'task_vecs_over_all_time', 'final_task_vecs',
        'r2_scores', 'lambdas', 'hiddens_layer'}``.
    """
    import matplotlib.pyplot as plt
    from icl.utils.unified_interface import _get_hiddens_at_real_positions
    from icl.linear.linear_utils import estimate_lambda_with_r2
    from icl.utils.traj_plot import project_with_r2_trajectories_group_colors_mpl
    from icl.utils.unified_ood_analysis import _get_minor_final_task_vecs

    if task_name not in ("coin", "latent", "linear"):
        raise ValueError(
            f"traj_projection_plot_coin supports coin/latent/linear, "
            f"got {task_name!r}"
        )

    # ---- 1. get hiddens ----
    demo_data = None
    sampler_clone = None

    get_kwargs = dict(
        task_name=task_name, exp_name=exp_name,
        n_minor=n_minor, n_ood=n_ood, B=B, step=step,
    )

    if task_name == "coin":
        if kwargs.get("return_p", False):
            hiddens, k_minor, _hover_p = _get_hiddens_at_real_positions(
                **get_kwargs, return_p=True,
            )
        elif kwargs.get("return_data", False):
            hiddens, k_minor, demo_data, sampler_clone = _get_hiddens_at_real_positions(
                **get_kwargs, return_data=True,
            )
        else:
            hiddens, k_minor = _get_hiddens_at_real_positions(**get_kwargs)
    else:
        # latent / linear — always 2-value return
        hiddens, k_minor = _get_hiddens_at_real_positions(**get_kwargs)

    # ---- 2. select layer ----
    if task_name == "linear":
        from icl.linear.linear_path_utils import load_model_task_config
        _, _, config = load_model_task_config(exp_name)
        n_layers = config.model.n_layer
    else:
        _, _, config = nu.load_everything(task_name, exp_name)
        n_layers = config.model.num_layers

    if layer_index is None:
        layer_index = n_layers - 1

    hiddens_layer = hiddens[layer_index].to(torch.float32)  # (K, T, B, D)

    K, T, B_actual, D = hiddens_layer.shape
    k_major = 3

    # ---- 3. task vectors ----
    task_mean = hiddens_layer[:k_major].mean(dim=(0, 2)).unsqueeze(0)  # (1, T, D)
    task_vecs_over_all_time = hiddens_layer.mean(dim=-2) - task_mean   # (K, T, D)

    if minor_projection:
        is_zero_mean = False
        final_task_vecs = _get_minor_final_task_vecs(
            task_vecs_over_all_time, k_minor,
        )
    else:
        is_zero_mean = True
        final_task_vecs = task_vecs_over_all_time[:k_major, -1]  # (3, D)

    if not use_mean:
        s_major = slice(0, k_major)
        s_mid = slice(k_major, K - k_minor)
        s_minor = slice(K - k_minor, K)

        def take_group(h_group, b, d_group=None):
            Kg = h_group.shape[0]
            idx = torch.randint(0, B_actual, (b,), device=h_group.device)
            x = h_group[:, :, idx, :]  # (Kg, T, b, D)
            x = x.permute(0, 2, 1, 3).contiguous().view(Kg * b, T, D)
            if d_group is not None:
                d = d_group[:, idx.to(d_group.device), :]
                return x, d
            return x

        if kwargs.get("return_data", False) and demo_data is not None:
            from icl.utils.unified_plot import posterior_over_models_over_time_per_sample
            out_major, _d_major = take_group(hiddens_layer[s_major], b_maj, demo_data[s_major])
            out_mid, d_mid = take_group(hiddens_layer[s_mid], b_ood, demo_data[s_mid])
            out_minor, _d_minor = take_group(hiddens_layer[s_minor], b_minor, demo_data[s_minor])
            _ood_post = posterior_over_models_over_time_per_sample(
                d_mid, sampler_clone.major_p,
            )
        else:
            out_major = take_group(hiddens_layer[s_major], b_maj)
            out_mid = take_group(hiddens_layer[s_mid], b_ood)
            out_minor = take_group(hiddens_layer[s_minor], b_minor)

        task_vecs_over_all_time = (
            torch.cat([out_major, out_mid, out_minor], dim=0) - task_mean
        )

    # ---- 4. lambda / R² ----
    lambdas, r2_scores, _, _ = estimate_lambda_with_r2(
        final_task_vecs,
        task_vecs_over_all_time,
        is_zero_mean=is_zero_mean,
    )

    # ---- 5. plot ----
    fig, ax, *_ = project_with_r2_trajectories_group_colors_mpl(
        task_vecs_over_all_time,
        final_task_vecs,
        r2_scores,
        n_minor=k_minor,
        n_ood=n_ood,
        use_mean=use_mean,
        b_major=b_maj,
        b_minor=b_minor,
        b_ood=b_ood,
        step=step,
        show_legend=show_legend,
        title=title,
        figsize=figsize,
    )

    if show:
        plt.show()

    return {
        'fig': fig,
        'ax': ax,
        'task_vecs_over_all_time': task_vecs_over_all_time,
        'final_task_vecs': final_task_vecs,
        'r2_scores': r2_scores,
        'lambdas': lambdas,
        'hiddens_layer': hiddens_layer,
    }


def traj_posterior_projection_plot_coin(
    exp_name: str,
    layer_index: Optional[int] = None,
    n_minor: int = 64,
    n_ood: int = 30,
    B: int = 64,
    b_maj: int = 2,
    b_minor: int = 0,
    b_ood: int = 2,
    use_mean: bool = True,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    show_legend: bool = True,
    figsize: tuple = (9, 7),
    show: bool = True,
    title: str = "",
    annotate: bool = True,
    major_colors: tuple = ("#1a5276", "#2471a3", "#5dade2"),
    ood_base_color: str = "#e74c3c",
) -> dict:
    """
    2-D trajectory projection for coin (non-padded) using **posterior-derived**
    task vectors.

    Instead of defining the projection plane from the raw batch-averaged
    hiddens of the 3 major tasks (as ``traj_projection_plot_coin`` does),
    this function:

      1. Fits a linear model ``posterior_logits @ W + b ≈ hiddens`` in
         major mode at late positions via
         ``train_linear_hidden_predictor_coin``.
      2. Uses the centered rows of ``W`` as the 3 final task vectors:
         ``final_task_vecs[i] = W[i] - W.mean(0)``.
         This isolates the posterior-encoding component of the hidden
         representation.
      3. Extracts OOD hiddens via ``_get_hiddens_at_real_positions``, computes
         batch-averaged (or per-sample) trajectories, and projects them
         onto the posterior plane.

    Parameters
    ----------
    exp_name : str
    layer_index : int, optional
        ``None`` → last layer.
    n_minor, n_ood, B : int
        Passed to ``_get_hiddens_at_real_positions``.
    b_maj, b_minor, b_ood : int
        Per-task random batch trajectories when ``use_mean=False``.
    use_mean : bool
    step : int, optional
    fit_n_samples : int
        Number of samples for the linear fit (passed to
        ``train_linear_hidden_predictor_coin``).
    fit_positions : list, optional
        Positions used for the linear fit.  ``None`` → ``range(40, 80)``.
    show_legend : bool
    figsize : tuple
    show : bool
    title : str

    Returns
    -------
    dict
        ``{'fig', 'ax', 'task_vecs_over_all_time', 'final_task_vecs',
        'r2_scores', 'lambdas', 'hiddens_layer', 'W', 'b',
        'fit_results'}``.
    """
    import matplotlib.pyplot as plt
    from icl.utils.unified_interface import _get_hiddens_at_real_positions
    from icl.linear.linear_utils import estimate_lambda_with_r2
    from icl.utils.traj_plot import project_with_r2_trajectories_group_colors_mpl

    # ---- 0. resolve layer & config ----
    _, _, config = nu.load_everything("coin", exp_name)
    n_layers = config.model.num_layers
    seq_len = config.seq_len
    if layer_index is None:
        layer_index = n_layers - 1

    # ---- 1. fit posterior → hiddens (major mode, late positions) ----
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
    b = fit_res["model_bias"]    # (D,)

    # Final task vectors: centered rows of W (bias cancels)
    final_task_vecs = W - W.mean(dim=0, keepdim=True)  # (3, D)

    # ---- 2. extract OOD hiddens ----
    hiddens, k_minor = _get_hiddens_at_real_positions(
        task_name="coin", exp_name=exp_name,
        n_minor=n_minor, n_ood=n_ood, B=B, step=step,
    )

    hiddens_layer = hiddens[layer_index].to(torch.float32)  # (K, T, B, D)
    K, T, B_actual, D = hiddens_layer.shape
    k_major = 3

    # ---- 3. task_vecs_over_all_time ----
    task_mean = hiddens_layer[:k_major].mean(dim=(0, 2)).unsqueeze(0)  # (1, T, D)

    if use_mean:
        task_vecs_over_all_time = hiddens_layer.mean(dim=-2) - task_mean  # (K, T, D)
    else:
        s_major = slice(0, k_major)
        s_mid = slice(k_major, K - k_minor)
        s_minor = slice(K - k_minor, K)

        def take_group(h_group, b_count):
            Kg = h_group.shape[0]
            idx = torch.randint(0, B_actual, (b_count,), device=h_group.device)
            x = h_group[:, :, idx, :]  # (Kg, T, b_count, D)
            return x.permute(0, 2, 1, 3).contiguous().view(Kg * b_count, T, D)

        out_major = take_group(hiddens_layer[s_major], b_maj)
        out_mid = take_group(hiddens_layer[s_mid], b_ood)
        out_minor = take_group(hiddens_layer[s_minor], b_minor)
        task_vecs_over_all_time = (
            torch.cat([out_major, out_mid, out_minor], dim=0) - task_mean
        )

    # ---- 3b. rescale final_task_vecs to match actual major endpoints ----
    actual_endpoints = task_vecs_over_all_time[:k_major, -1]  # (3, D)
    ftv_norm = final_task_vecs.float().norm()
    ep_norm = actual_endpoints.float().norm()
    if ftv_norm > 0:
        final_task_vecs = final_task_vecs * (ep_norm / ftv_norm).item()

    # ---- 4. lambda / R² ----
    lambdas, r2_scores, _, _ = estimate_lambda_with_r2(
        final_task_vecs,
        task_vecs_over_all_time,
        is_zero_mean=True,
    )

    # ---- 4b. pick informative time indices via farthest-point sampling ----
    n_show = 4
    lam_np = lambdas if isinstance(lambdas, np.ndarray) else np.asarray(lambdas)
    coords_all = lam_np[:, :, :2]  # (K_traj, T, 2)
    K_traj, T_len = coords_all.shape[0], coords_all.shape[1]
    selected = [0, T_len - 1]
    for _ in range(min(n_show, T_len) - 2):
        best_t, best_score = -1, -1.0
        for t in range(T_len):
            if t in selected:
                continue
            per_traj = np.array([
                min(np.linalg.norm(coords_all[k, t] - coords_all[k, s]) for s in selected)
                for k in range(K_traj)
            ])
            score = np.median(per_traj)
            if score > best_score:
                best_score, best_t = score, t
        if best_t >= 0:
            selected.append(best_t)
    t_selected = sorted(selected)

    # ---- 5. plot ----
    fig, ax, *_ = project_with_r2_trajectories_group_colors_mpl(
        task_vecs_over_all_time,
        final_task_vecs,
        r2_scores,
        n_minor=k_minor,
        n_ood=n_ood,
        use_mean=use_mean,
        b_major=b_maj,
        b_minor=b_minor,
        b_ood=b_ood,
        step=step,
        show_legend=show_legend,
        title=title,
        figsize=figsize,
        major_colors=major_colors,
        ood_base_color=ood_base_color,
        ood_hue_jitter=0.02,
        ood_sat_jitter=0.05,
        ood_val_jitter=0.05,
        annotate=annotate,
        annotate_k=2,
        annotate_start=False,
        t_show_override=t_selected,
        show_pow2_anchors=False,
        size_min=5,
        size_max=16,
        mid_size_factor=0.5,
        line_width=1.4,
    )

    if show:
        plt.show()

    return {
        'fig': fig,
        'ax': ax,
        'task_vecs_over_all_time': task_vecs_over_all_time,
        'final_task_vecs': final_task_vecs,
        'r2_scores': r2_scores,
        'lambdas': lambdas,
        'hiddens_layer': hiddens_layer,
        'W': W,
        'b': b,
        'fit_results': fit_res,
    }


def traj_post_posterior_projection_plot_coin(
    exp_name: str,
    layer_index: Optional[int] = None,
    task_ids: Optional[list] = None,
    n_individual: int = 10,
    B: int = 64,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    n_ood: Optional[int] = None,
    center_lambda: bool = True,
    figsize: Optional[tuple] = None,
    show: bool = True,
    corner_colors: tuple = ("#1f77b4", "#2ca02c", "#d62728"),
    # deprecated — use task_ids instead
    task_id: Optional[int] = None,
    n_rows: Optional[int] = None,
) -> dict:
    """Compare λ(t) from hidden projections with Bayesian P(Z=k|x_{1:t}).

    For each task k, extracts hidden trajectories h_l(t), estimates
    λ(t) via least-squares projection onto centred task vectors, and
    overlays the oracle posterior P(Z=k | x_{1:t}).

    Each row = one task.  Shows mean ± std band (clamped [0,1]) plus
    individual traces.

    Parameters
    ----------
    exp_name : str
    layer_index : int, optional
        ``None`` → last layer.
    task_ids : list of int, optional
        Which tasks to show as rows. 0, 1, 2 = major; ≥3 = OOD.
        Default ``[0, 1, 2]``.
    n_individual : int
        Number of individual sample traces shown behind the mean.
    B : int
    step : int, optional
    fit_n_samples : int
    fit_positions : list, optional
    n_ood : int, optional
        Number of OOD tasks to generate. Auto-determined from *task_ids*
        if not given.
    center_lambda : bool
        If True, project λ onto the probability simplex. Otherwise clip
        to [0, 1].
    figsize : tuple
    show : bool
    corner_colors : tuple of 3 hex strings

    Returns
    -------
    dict
        ``{'fig', 'axes', 'results_by_task', 'W', 'b', 'fit_results',
        'task_ids', 'center_lambda'}``.
    """
    import gc
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from icl.utils.unified_interface import _get_hiddens_at_real_positions
    from icl.linear.linear_utils import estimate_lambda_with_r2
    from icl.utils.unified_plot import posterior_over_models_over_time_per_sample

    # Backward compat: old task_id / n_rows API
    if task_ids is None:
        if task_id is not None:
            task_ids = [task_id]
        else:
            task_ids = [0, 1, 2]

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
    W = fit_res["model_weight"]        # (K_major, D)
    b = fit_res["model_bias"]          # (D,)
    W_mean = W.mean(dim=0, keepdim=True)  # (1, D)

    # From the probe:  h ≈ π W + x_t W_tok + b + ε
    # Centred form:    h − b − w̄ ≈ π (W − w̄) + x_t W_tok + ε  =  π W̃ + noise
    b_vec = b.float()              # (D,)
    W_mean_f = W_mean.float()      # (1, D)
    W_c = (W - W_mean).float()     # (K_major, D)  — centred task vectors W̃
    W_f = W.float()                # (K_major, D)  — uncentred

    max_tid = max(task_ids)
    if n_ood is None:
        n_ood = max(1, max_tid - 2)
    hiddens, _k_minor, demo_data, sampler_clone = _get_hiddens_at_real_positions(
        task_name="coin", exp_name=exp_name,
        n_minor=0, n_ood=n_ood, B=B, step=step,
        return_data=True,
    )

    hiddens_layer = hiddens[layer_index].to(torch.float32)  # (K, T, B, D)
    K, T, B_actual, D = hiddens_layer.shape
    k_major = W.shape[0]
    dev = hiddens_layer.device
    b_vec = b_vec.to(dev)
    W_mean_f = W_mean_f.to(dev)
    W_c = W_c.to(dev)
    W_f = W_f.to(dev)

    for tid in task_ids:
        if tid >= K:
            raise ValueError(f"task_id={tid} out of range (K={K} tasks available)")

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

    n_indiv = min(n_individual, B_actual)
    indiv_indices = torch.randperm(B_actual)[:n_indiv]

    results_by_task = {}
    for tid in task_ids:
        lam_list = []
        for bi in range(B_actual):
            h = hiddens_layer[tid:tid + 1, :, bi:bi + 1, :].squeeze(2)  # (1, T, D)

            if center_lambda:
                lam_raw, *_ = estimate_lambda_with_r2(
                    W_c, h - b_vec - W_mean_f, is_zero_mean=True,
                )
            else:
                lam_raw, *_ = estimate_lambda_with_r2(
                    W_f, h - b_vec, is_zero_mean=False,
                )
            lam_np = np.asarray(lam_raw)[0]  # (T, K_major)
            lam_list.append(
                _project_onto_simplex(lam_np) if center_lambda
                else np.clip(lam_np, 0.0, 1.0)
            )
        lam_all = np.stack(lam_list, axis=0)  # (B, T, K_major)

        post_tensor = posterior_over_models_over_time_per_sample(
            demo_data[tid:tid + 1], sampler_clone.major_p,
        ).float()  # (1, B, seq_len, K_major)
        post_all = post_tensor[0, :, :T, :k_major].cpu().numpy()  # (B, T, K_major)

        results_by_task[tid] = {"lam": lam_all, "post": post_all}

    # ---- plot ----
    corner_rgb = [mcolors.to_rgb(c) for c in corner_colors]
    ts = np.arange(T)
    n_plot_rows = len(task_ids)
    if figsize is None:
        figsize = (14, 3.5 * n_plot_rows)
    fig, axes = plt.subplots(
        n_plot_rows, k_major, figsize=figsize, sharey=True, squeeze=False,
    )

    mode_str = "simplex" if center_lambda else "clipped"
    for row, tid in enumerate(task_ids):
        lam_all = results_by_task[tid]["lam"]    # (B, T, K_major)
        post_all = results_by_task[tid]["post"]  # (B, T, K_major)
        lam_mean = lam_all.mean(axis=0)          # (T, K_major)
        lam_std = lam_all.std(axis=0)
        post_mean = post_all.mean(axis=0)
        post_std = post_all.std(axis=0)

        for col in range(k_major):
            ax = axes[row, col]
            c = corner_rgb[col]

            for idx in indiv_indices:
                ii = int(idx)
                ax.plot(ts, lam_all[ii, :, col], color=c, lw=0.5, alpha=0.12)

            ax.plot(ts, lam_mean[:, col], color=c, lw=2.5, alpha=0.95,
                    label=rf"$\lambda_{col+1}$ mean ({mode_str})")
            lo = np.clip(lam_mean[:, col] - lam_std[:, col], 0.0, 1.0)
            hi = np.clip(lam_mean[:, col] + lam_std[:, col], 0.0, 1.0)
            ax.fill_between(ts, lo, hi, color=c, alpha=0.15)

            ax.plot(ts, post_mean[:, col], color=c, lw=2, ls="--", alpha=0.7,
                    label=rf"$P(Z\!={col+1})$ mean")
            p_lo = np.clip(post_mean[:, col] - post_std[:, col], 0.0, 1.0)
            p_hi = np.clip(post_mean[:, col] + post_std[:, col], 0.0, 1.0)
            ax.fill_between(ts, p_lo, p_hi, color=c, alpha=0.08, hatch="//")

            ax.set_ylim(-0.05, 1.05)
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(labelsize=14)
            if row == n_plot_rows - 1:
                ax.set_xlabel("Position $t$", fontsize=16)
            task_label = f"Task {tid}" if tid < k_major else f"OOD {tid}"
            if col == 0:
                ax.set_ylabel(task_label, fontsize=16)
            if row == 0:
                ax.set_title(
                    rf"$\lambda_{col+1}$ / $\mathbb{{P}}(Z\!={col+1}|\mathrm{{data}})$",
                    fontsize=15,
                )
            if row == 0 and col == 0:
                ax.legend(fontsize=10, loc="best", framealpha=0.7)

    fig.suptitle(
        f"Layer {layer_index}  (B={B_actual}, {n_indiv} traces shown)",
        fontsize=15, y=1.01,
    )
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close()

    del hiddens
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "fig": fig,
        "axes": axes,
        "results_by_task": results_by_task,
        "W": W,
        "b": b,
        "fit_results": fit_res,
        "task_ids": task_ids,
        "center_lambda": center_lambda,
    }
