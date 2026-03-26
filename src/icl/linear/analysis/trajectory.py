"""Trajectory projection plots for linear regression analysis."""

import gc
from typing import Optional

import numpy as np
import torch

import icl.utils.notebook_utils as nu
from icl.utils.logger import setup_logger
from icl.utils.unified_interface import _get_hiddens_at_real_positions
from icl.linear.analysis._helpers import (
    _show_or_close,
    _task_positions,
    _project_onto_simplex,
)
from icl.linear.analysis.probes import train_linear_hidden_predictor
from icl.linear.analysis.interventions._helpers import _extract_hiddens_for_pool

logger = setup_logger(__name__)


def traj_posterior_projection_plot(
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
    estimation_positions: Optional[list] = None,
    extraction_point: str = "post_attn",
    show_legend: bool = True,
    figsize: tuple = (4, 4),
    show: bool = True,
    title: str = "",
    annotate: bool = True,
    major_colors: tuple = ("#1a5276", "#2471a3", "#5dade2"),
    ood_base_color: str = "#e74c3c",
    ood_t_show_override: Optional[list] = None,
    major_n_show: int = 7,
    major_line_width: Optional[float] = 2.0,
    ood_line_alpha_factor: float = 0.5,
) -> dict:
    """2-D trajectory projection using averaging-based task vectors.

    Task vectors are estimated by averaging hidden states of major tasks at
    late positions (``estimation_positions``), then centering.  The projection
    plane is spanned by these averaged task vectors.

    Steps:
      1. Extract hiddens for all task groups via ``_get_hiddens_at_real_positions``.
      2. Estimate task vectors by averaging major-task hiddens at late positions.
      3. Subtract per-position grand mean to get centered trajectories.
      4. SVD → pick 2 leading directions → project onto that plane.
      5. Plot ID + OOD trajectories over in-context positions.

    Parameters
    ----------
    exp_name : str
    layer_index : int, optional — defaults to last layer
    n_minor, n_ood, B : int
    b_maj, b_minor, b_ood : int — per-task trajectories when ``use_mean=False``
    use_mean : bool
    step : int, optional
    estimation_positions : list, optional
        Positions used to estimate task vectors by averaging.
        ``None`` → last 10 positions of the sequence.
    extraction_point : str
        ``"post_attn"`` (default) — after the attention block, before MLP.
        ``"post_mlp"`` — after the full block (residual stream).
    """
    import matplotlib.pyplot as plt
    from icl.linear.linear_utils import estimate_lambda_with_r2
    from icl.linear.linear_path_utils import load_model_task_config
    from icl.utils.traj_plot import project_with_r2_trajectories_group_colors_mpl
    from icl.utils.separability import estimate_task_vectors_by_averaging

    _, _, config = load_model_task_config(exp_name)
    n_points = int(config.task.n_points)
    if layer_index is None:
        layer_index = config.model.n_layer - 1

    # ---- extract hiddens for all task groups ----
    hiddens, k_minor = _get_hiddens_at_real_positions(
        task_name="linear", exp_name=exp_name,
        n_minor=n_minor, n_ood=n_ood, B=B, step=step,
        extraction_point=extraction_point,
    )
    hiddens_layer = hiddens[layer_index].to(torch.float32)  # (K, T, B, D)
    K, T, B_actual, D = hiddens_layer.shape
    k_major = 3
    n_ood_actual = K - k_major - k_minor

    # ---- estimate task vectors from major tasks by averaging ----
    hiddens_major = hiddens_layer[:k_major]  # (k_major, T, B, D)
    if estimation_positions is None:
        estimation_positions = list(range(max(0, n_points - 10), n_points))
    task_vecs, grand_mean = estimate_task_vectors_by_averaging(
        hiddens_major, estimation_positions,
    )
    final_task_vecs = task_vecs.float()  # (k_major, D), already centered
    grand_mean_dev = grand_mean.to(hiddens_layer.device)

    # ── Build task-vector trajectories (subtract grand mean) ────────
    if use_mean:
        tvecs = hiddens_layer.mean(dim=-2) - grand_mean_dev  # (K, T, D)
    else:
        def _sample_group(h, n):
            idx = torch.randint(0, B_actual, (n,), device=h.device)
            return h[:, :, idx, :].permute(0, 2, 1, 3).contiguous().view(-1, T, D)

        tvecs = torch.cat([
            _sample_group(hiddens_layer[:k_major], b_maj),
            _sample_group(hiddens_layer[k_major:K - k_minor], b_ood),
            _sample_group(hiddens_layer[K - k_minor:], b_minor),
        ], dim=0) - grand_mean_dev

    lambdas, r2_scores, *_ = estimate_lambda_with_r2(
        final_task_vecs, tvecs, is_zero_mean=True,
    )

    # ---- pick time indices: first (major_n_show-1) steps + end ----
    lam_np = np.asarray(lambdas)
    T_len = lam_np.shape[1]
    n_initial = min(max(2, int(major_n_show)) - 1, T_len - 1)
    t_selected = sorted(set(range(n_initial)) | {T_len - 1})

    _ood_ts = [0, -1] if ood_t_show_override is None else ood_t_show_override
    fig, ax, *_ = project_with_r2_trajectories_group_colors_mpl(
        tvecs, final_task_vecs, r2_scores,
        n_minor=k_minor, n_ood=n_ood_actual,
        use_mean=use_mean, b_major=b_maj, b_minor=b_minor, b_ood=b_ood,
        step=step, show_legend=show_legend, title=title, figsize=figsize,
        major_colors=major_colors, ood_base_color=ood_base_color,
        ood_hue_jitter=0.02, ood_sat_jitter=0.05, ood_val_jitter=0.05,
        annotate=annotate, annotate_k=2, annotate_start=False,
        t_show_override=t_selected, ood_t_show_override=_ood_ts,
        show_pow2_anchors=False,
        size_min=5, size_max=16, mid_size_factor=0.5, line_width=1.4,
        major_line_width=major_line_width,
        ood_line_alpha_factor=ood_line_alpha_factor,
    )

    if show:
        plt.show()

    return {
        "fig": fig, "ax": ax,
        "task_vecs_over_all_time": tvecs, "final_task_vecs": final_task_vecs,
        "r2_scores": r2_scores, "lambdas": lambdas,
        "hiddens_layer": hiddens_layer,
        "task_vecs": task_vecs, "grand_mean": grand_mean,
    }


def traj_post_posterior_projection_plot(
    exp_name: str,
    layer_index: Optional[int] = None,
    task_ids: Optional[list] = None,
    n_individual: int = 10,
    B: int = 64,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    fit_include_position_bias: bool = False,
    fit_include_logit: bool = False,
    n_ood: Optional[int] = None,
    center_lambda: bool = True,
    figsize: Optional[tuple] = None,
    show: bool = True,
    corner_colors: tuple = ("#0072B2", "#E69F00", "#009E73"),
    # deprecated — use task_ids instead
    task_id: Optional[int] = None,
    n_rows: Optional[int] = None,
) -> dict:
    """Compare λ(t) from hidden projections with Bayesian P(z=k|data).

    For each task k, extracts hidden trajectories h_l(t), estimates
    λ(t) = (h − b) @ W̃ᵀ (or simplex-projected), and overlays the
    oracle posterior P(z=k | x₁:t, y₁:t₋₁).

    Each row = one task.  Shows mean ± std band (clamped [0,1]) plus
    individual traces.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from icl.linear.linear_utils import estimate_lambda_with_r2
    from icl.linear.analysis.posterior import task_posterior_over_time_linear_regression
    from icl.linear.linear_path_utils import load_model_task_config
    from icl.linear.linear_ood_analysis import _create_eval_task_pool, _setup_eval_task, setup_device
    from icl.linear.task_vecs import extract_hidden_multi

    # Backward compat: old task_id / n_rows API
    if task_ids is None:
        if task_id is not None:
            task_ids = [task_id]
        else:
            task_ids = [0, 1, 2]

    _, train_task, config = load_model_task_config(exp_name)
    n_points = int(config.task.n_points)
    if layer_index is None:
        layer_index = config.model.n_layer - 1
    if fit_positions is None:
        fit_positions = list(range(max(0, n_points - 20), n_points))

    # ---- fit probe: posterior -> hiddens (3 major rows in W) ----
    fit_res = train_linear_hidden_predictor(
        exp_name=exp_name, layer=layer_index, n_samples=fit_n_samples,
        positions=fit_positions, sample_mode="major", n_minor=None,
        include_position_bias=fit_include_position_bias,
        include_logit=fit_include_logit,
        step=step, print_summary=False, skip_baselines=True,
    )
    W = fit_res["model_weight"]          # (3, D)
    b = fit_res["model_bias"]            # (D,)
    W_mean = W.mean(dim=0, keepdim=True) # (1, D)

    # ---- build eval pool: [3 major, OOD…] ----
    max_tid = max(task_ids)
    if n_ood is None:
        n_ood = max(1, max_tid - 2)
    device = setup_device(None)
    major_pool = train_task.task_pool.squeeze(-1).to(device)
    if major_pool.shape[0] < 3:
        raise ValueError("Expected at least 3 major tasks for projection.")
    ood_pool_raw, _ = _create_eval_task_pool(
        train_task, K=n_ood, include_minor=False, device=device, n_minor=0,
    )
    n_anchor = int(train_task.task_pool.shape[0])
    ood_pool = ood_pool_raw[n_anchor:] if ood_pool_raw.shape[0] >= n_anchor \
        else ood_pool_raw.new_empty((0, ood_pool_raw.shape[1]))
    if ood_pool.numel() > 0:
        keep = torch.ones(ood_pool.shape[0], dtype=torch.bool, device=device)
        for a in major_pool[:3]:
            keep &= ~(ood_pool == a.unsqueeze(0)).all(dim=1)
        ood_pool = ood_pool[keep][:n_ood]
    eval_task = _setup_eval_task(
        config, torch.cat([major_pool[:3], ood_pool], dim=0), B, device,
    )
    eval_task.batch_size = B

    if step is None:
        step = config.training.total_steps
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval().to(config.device)

    pad_mode = getattr(model, "pad", "mapsto")
    task_pos = _task_positions(pad_mode, n_points, device)

    n_tasks = int(eval_task.task_pool.shape[0])
    for tid in task_ids:
        if tid >= n_tasks:
            raise ValueError(f"task_id={tid} out of range (K={n_tasks} tasks available)")

    # ---- extract hidden trajectories in chunks ----
    demo_data = eval_task.sample_data(step=step).to(device)
    D = int(config.model.n_embd)

    hiddens, all_targets = _extract_hiddens_for_pool(
        model, eval_task, demo_data,
        step=step, layer=layer_index, task_pos=task_pos, D=D, chunk=16,
    )

    task_targets = {}
    for tid in task_ids:
        if tid >= all_targets.shape[0]:
            raise RuntimeError(f"task_id={tid} out of range (K={all_targets.shape[0]})")
        task_targets[tid] = all_targets[tid]  # (B, T)

    # ── Subtract nuisance terms and estimate λ(t) ─────────────────
    hiddens = hiddens.float()
    k_major = 3
    T, B_actual = hiddens.shape[1], hiddens.shape[2]
    dev = hiddens.device
    b_vec = b.to(dev).float()
    W_c = (W - W_mean).to(dev).float()
    W_f = W.to(dev).float()
    W_mean_f = W_mean.to(dev).float()

    def _to_float_device(t):
        if isinstance(t, torch.Tensor) and t.numel() > 0:
            return t.float().to(dev)
        return torch.zeros((0, D), dtype=torch.float32, device=dev)

    W_tok_f = _to_float_device(fit_res.get("token_weight", None))
    W_logit_f = _to_float_device(fit_res.get("logit_weight", None))
    W_pos_f = _to_float_device(fit_res.get("position_weight", None))
    fit_pos_list = [int(p) for p in fit_res.get("positions", fit_positions)]
    fit_pos_to_col = {p: i for i, p in enumerate(fit_pos_list)}

    # Token nuisance is shared across tasks (same x-values).
    tok_nuisance = torch.zeros((B_actual, T, D), dtype=torch.float32, device=dev)
    if W_tok_f.shape[0] > 0:
        x_vals = demo_data[:B_actual, :T].float().to(dev)  # (B, T, n_dims)
        tok_nuisance = torch.einsum("btn,nd->btd", x_vals, W_tok_f)

    pos_nuisance = torch.zeros((T, D), dtype=torch.float32, device=dev)
    if W_pos_f.shape[0] > 0:
        for t in range(T):
            j = fit_pos_to_col.get(t, None)
            if j is not None and j < W_pos_f.shape[0]:
                pos_nuisance[t] = W_pos_f[j]

    n_indiv = min(n_individual, B_actual)
    indiv_indices = torch.randperm(B_actual)[:n_indiv]

    results_by_task = {}
    for tid in task_ids:
        h_tid = hiddens[tid].permute(1, 0, 2).contiguous()  # (B, T, D)
        nuisance = tok_nuisance.clone()

        if W_logit_f.shape[0] > 0:
            with torch.no_grad():
                preds = model(
                    demo_data[:B_actual].to(device),
                    all_targets[tid, :B_actual].to(device),
                ).float()[:, :T]  # (B, T)
            logit_vals = preds.unsqueeze(-1).to(dev)  # (B, T, 1)
            nuisance = nuisance + torch.einsum(
                "btl,ld->btd", logit_vals, W_logit_f,
            )

        nuisance = nuisance + pos_nuisance.unsqueeze(0)

        h_adj = h_tid - b_vec.unsqueeze(0).unsqueeze(0) - nuisance
        if center_lambda:
            lam_raw, *_ = estimate_lambda_with_r2(
                W_c, h_adj - W_mean_f.unsqueeze(0), is_zero_mean=True,
            )
        else:
            lam_raw, *_ = estimate_lambda_with_r2(
                W_f, h_adj, is_zero_mean=False,
            )
        lam_np = np.asarray(lam_raw)  # (B, T, K_major)
        if center_lambda:
            lam_all = _project_onto_simplex(lam_np.reshape(-1, lam_np.shape[-1])).reshape(lam_np.shape)
        else:
            lam_all = np.clip(lam_np, 0.0, 1.0)

        post_tensor = task_posterior_over_time_linear_regression(
            train_task, demo_data, task_targets[tid].to(demo_data.device),
            include_minor=False,
        )  # (B, T, K)
        post_all = post_tensor[:, :, :k_major].detach().cpu().numpy()  # (B, T, 3)

        results_by_task[tid] = {"lam": lam_all, "post": post_all}

    # ---- plot (split into major and OOD figures) ----
    corner_rgb = [mcolors.to_rgb(c) for c in corner_colors]
    ts = np.arange(T)
    mode_str = "simplex" if center_lambda else "clipped"
    k_major = 3
    major_task_ids = [tid for tid in task_ids if tid < k_major]
    ood_task_ids = [tid for tid in task_ids if tid >= k_major]

    def _group_figsize(n_rows: int):
        if figsize is not None:
            return figsize
        return (14, 3.5 * max(n_rows, 1))

    def _plot_group(group_task_ids):
        if not group_task_ids:
            return None, None
        n_rows_local = len(group_task_ids)
        fig_local, axes_local = plt.subplots(
            n_rows_local, 3, figsize=_group_figsize(n_rows_local), sharey=True, squeeze=False,
        )
        for row, tid in enumerate(group_task_ids):
            lam_all = results_by_task[tid]["lam"]
            post_all = results_by_task[tid]["post"]
            lam_mean = lam_all.mean(axis=0)
            lam_std = lam_all.std(axis=0)
            post_mean = post_all.mean(axis=0)
            post_std = post_all.std(axis=0)

            for col in range(3):
                ax = axes_local[row, col]
                c = corner_rgb[col]
                for idx in indiv_indices:
                    ii = int(idx)
                    ax.plot(ts, lam_all[ii, :, col], color=c, lw=0.5, alpha=0.12)

                ax.plot(ts, lam_mean[:, col], color=c, lw=2.5, alpha=0.95)
                lo = np.clip(lam_mean[:, col] - lam_std[:, col], 0.0, 1.0)
                hi = np.clip(lam_mean[:, col] + lam_std[:, col], 0.0, 1.0)
                ax.fill_between(ts, lo, hi, color=c, alpha=0.18, zorder=1)

                post_color = "#111111"
                post_band_color = "#9E9E9E"
                ax.plot(ts, post_mean[:, col], color=post_color, lw=2.2, ls="--", alpha=0.95)
                p_lo = np.clip(post_mean[:, col] - post_std[:, col], 0.0, 1.0)
                p_hi = np.clip(post_mean[:, col] + post_std[:, col], 0.0, 1.0)
                ax.fill_between(
                    ts, p_lo, p_hi,
                    facecolor="none",
                    edgecolor=post_band_color,
                    hatch="////",
                    linewidth=0.0,
                    alpha=0.9,
                    zorder=2,
                )

                ax.set_ylim(-0.05, 1.05)
                ax.grid(axis="y", alpha=0.3)
                ax.tick_params(labelsize=20)
                if row == n_rows_local - 1:
                    ax.set_xlabel("Position $t$", fontsize=22)
                if col == 0:
                    task_label = f"Task {tid}" if tid < k_major else f"OOD {tid}"
                    ax.set_ylabel(task_label, fontsize=22)
                if row == 0:
                    handles = [
                        Line2D([0], [0], color=c, lw=2.5, ls="-",
                               label=rf"$\lambda_{{{col+1}}}$ mean ({mode_str})"),
                        Patch(facecolor=c, alpha=0.18, label=rf"$\lambda_{{{col+1}}}$ std band"),
                        Line2D([0], [0], color=post_color, lw=2.2, ls="--",
                               label=rf"$P(Z\!={col+1})$ mean"),
                        Patch(facecolor="white", edgecolor=post_band_color, hatch="////",
                              label=rf"$P(Z\!={col+1})$ std band"),
                    ]
                    ax.legend(handles=handles, fontsize=16, loc="best", framealpha=0.75)

        fig_local.suptitle("", fontsize=18, y=1.01)
        _show_or_close(fig_local, show)
        return fig_local, axes_local

    fig_major, axes_major = _plot_group(major_task_ids)
    fig_ood, axes_ood = _plot_group(ood_task_ids)
    fig = fig_major if fig_major is not None else fig_ood
    axes = axes_major if axes_major is not None else axes_ood

    model.cpu(); del model, eval_task
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "fig": fig, "axes": axes,
        "fig_major": fig_major, "axes_major": axes_major,
        "fig_ood": fig_ood, "axes_ood": axes_ood,
        "results_by_task": results_by_task,
        "W": W, "b": b, "fit_results": fit_res,
        "task_ids": task_ids, "pad_mode": pad_mode, "center_lambda": center_lambda,
    }


def plot_lambda_posterior_agreement(
    exp_name: str,
    layer_index: Optional[int] = None,
    n_ood: int = 30,
    B: int = 64,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    fit_include_position_bias: bool = False,
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
    """Quantify agreement between λ(t) and P(z|data) using Hellinger distance."""
    import matplotlib.pyplot as plt
    from icl.linear.linear_utils import estimate_lambda_with_r2
    from icl.linear.analysis.posterior import task_posterior_over_time_linear_regression
    from icl.linear.linear_path_utils import load_model_task_config
    from icl.linear.linear_ood_analysis import _create_eval_task_pool, _setup_eval_task, setup_device

    if major_only:
        n_ood = 0

    _, train_task, config = load_model_task_config(exp_name)
    n_points = int(config.task.n_points)
    if layer_index is None:
        layer_index = config.model.n_layer - 1
    if fit_positions is None:
        fit_positions = list(range(max(0, n_points - 20), n_points))

    # ---- fit probe ----
    fit_res = train_linear_hidden_predictor(
        exp_name=exp_name, layer=layer_index, n_samples=fit_n_samples,
        positions=fit_positions, sample_mode="major", n_minor=None,
        include_position_bias=fit_include_position_bias,
        include_logit=fit_include_logit,
        step=step, print_summary=False, skip_baselines=True,
    )
    W = fit_res["model_weight"]          # (3, D)
    b_vec = fit_res["model_bias"]        # (D,)
    W_mean = W.mean(dim=0, keepdim=True) # (1, D)

    # ---- build eval pool ----
    device = setup_device(None)
    major_pool = train_task.task_pool.squeeze(-1).to(device)
    if major_pool.shape[0] < 3:
        raise ValueError("Expected at least 3 major tasks for projection.")
    if n_ood > 0:
        ood_pool, _ = _create_eval_task_pool(
            train_task, K=n_ood, include_minor=False, device=device, n_minor=0,
        )
        full_pool = torch.cat([major_pool[:3], ood_pool], dim=0)
    else:
        full_pool = major_pool[:3]
    eval_task = _setup_eval_task(config, full_pool, B, device)
    eval_task.batch_size = B

    if step is None:
        step = config.training.total_steps
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval().to(config.device)

    pad_mode = getattr(model, "pad", "mapsto")
    task_pos = _task_positions(pad_mode, n_points, device)

    K = int(eval_task.task_pool.shape[0])
    D = int(config.model.n_embd)
    demo_data = eval_task.sample_data(step=step).to(device)

    hiddens, task_targets = _extract_hiddens_for_pool(
        model, eval_task, demo_data,
        step=step, layer=layer_index, task_pos=task_pos, D=D, chunk=16,
    )

    # ---- compute lambda & posterior per task ----
    hiddens = hiddens.float()
    k_major, T, B_actual = 3, hiddens.shape[1], hiddens.shape[2]
    dev = hiddens.device
    b_dev = b_vec.to(dev).float()
    W_c = (W - W_mean).to(dev).float()
    W_mean_dev = W_mean.to(dev).float()

    def _safe_mean(arr, axis=0):
        """Mean along *axis*; returns NaN array if axis has size 0."""
        arr = np.asarray(arr, dtype=float)
        if arr.shape[axis] == 0:
            shape = list(arr.shape)
            shape.pop(axis)
            return np.full(shape, np.nan)
        with np.errstate(all="ignore"):
            return np.nanmean(arr, axis=axis)

    # ── Compute agreement metrics between λ and posterior ────────────
    # Hellinger distance:  H(λ, P) = sqrt(½ Σₖ (√λₖ − √Pₖ)²)  ∈ [0, 1]
    # Cosine similarity between the two K-vectors at each time step.
    all_lam = np.zeros((K, B_actual, T, 3), dtype=float)
    all_post = np.zeros((K, B_actual, T, 3), dtype=float)
    tv_per_task = np.zeros((K, T))
    tv_random_per_task = np.zeros((K, T))
    tv_task_basis_per_task = np.full((K, T), np.nan, dtype=float)
    rng = np.random.default_rng(random_seed)

    for k in range(K):
        post_k = task_posterior_over_time_linear_regression(
            train_task, demo_data, task_targets[k].to(demo_data.device),
            include_minor=False,
        ).detach().cpu().numpy()  # (B, T, 3)

        for bi in range(B_actual):
            h = hiddens[k:k + 1, :, bi, :]  # (1, T, D)
            lam_raw, *_ = estimate_lambda_with_r2(W_c, h - b_dev - W_mean_dev, is_zero_mean=True)
            lam = _project_onto_simplex(np.asarray(lam_raw)[0])  # (T, 3)
            p = post_k[bi]

            all_lam[k, bi] = lam
            all_post[k, bi] = p
            tv_per_task[k] += np.sqrt(0.5 * ((np.sqrt(np.maximum(p, 0)) - np.sqrt(np.maximum(lam, 0))) ** 2).sum(axis=-1))
            if show_random_baseline and random_baseline_draws > 0:
                rand_lam = rng.dirichlet(
                    np.ones(3, dtype=float),
                    size=(random_baseline_draws, T),
                )
                tv_rand = np.sqrt(0.5 * ((np.sqrt(rand_lam) - np.sqrt(np.maximum(p[None, :, :], 0))) ** 2).sum(axis=-1)).mean(axis=0)
                tv_random_per_task[k] += tv_rand
            if show_task_basis_baseline and k < k_major:
                basis = np.zeros((T, 3), dtype=float)
                basis[:, k] = 1.0
                tv_basis = np.sqrt(0.5 * ((np.sqrt(np.maximum(p, 0)) - np.sqrt(basis)) ** 2).sum(axis=-1))
                if np.isnan(tv_task_basis_per_task[k]).all():
                    tv_task_basis_per_task[k] = 0.0
                tv_task_basis_per_task[k] += tv_basis

        tv_per_task[k] /= B_actual
        if show_random_baseline and random_baseline_draws > 0:
            tv_random_per_task[k] /= B_actual
        if show_task_basis_baseline and k < k_major:
            tv_task_basis_per_task[k] /= B_actual

    # ---- aggregate major / OOD ----
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
    ts = np.arange(T)
    _sfx = "" if major_only else " major"

    # ---- plot ----
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(ts, tv_major, lw=2.5,
            label=r"$\lambda(t)$ vs $\alpha_t$" + (f" {_sfx}" if _sfx else ""),
            color="#1f77b4", marker="o", markersize=4)
    if not major_only:
        ax.plot(ts, tv_ood, lw=2.5,
                label=r"$\lambda(t)$ vs $\alpha_t$ OOD",
                color="#d62728", marker="s", markersize=4)
    if tv_random_major is not None:
        ax.plot(
            ts, tv_random_major, lw=2.0, ls="--",
            label="uniform-random" + (f" {_sfx}" if _sfx else ""),
            color="#7eb8da", marker="^", markersize=3,
        )
    if not major_only and tv_random_ood is not None:
        ax.plot(
            ts, tv_random_ood, lw=2.0, ls="--",
            label="uniform-random OOD", color="#e8836b",
            marker="v", markersize=3,
        )
    if tv_task_basis_major is not None:
        ax.plot(
            ts, tv_task_basis_major, lw=2.0, ls=":",
            label="task-basis" + (f" {_sfx}" if _sfx else ""),
            color="#2ca02c", marker="D", markersize=3,
        )
    if not major_only and tv_task_basis_ood is not None and np.isfinite(tv_task_basis_ood).any():
        ax.plot(
            ts, tv_task_basis_ood, lw=2.0, ls=":",
            label="task-basis OOD", color="#9467bd",
            marker="d", markersize=3,
        )
    ax.set(xlabel="Position", ylabel="Hellinger distance", ylim=(0, None))
    ax.xaxis.label.set_fontsize(18)
    ax.yaxis.label.set_fontsize(18)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)
    _xlim_lo = min_position if min_position is not None else 0
    _xlim_hi = max_position if max_position is not None else None
    ax.set_xlim(_xlim_lo, _xlim_hi)

    fig.suptitle("", fontsize=18)
    _show_or_close(fig, show)

    model.cpu(); del model, eval_task
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "fig": fig, "ax": ax, "positions": ts,
        "tv_major": tv_major, "tv_ood": tv_ood,
        "tv_random_major": tv_random_major, "tv_random_ood": tv_random_ood,
        "tv_task_basis_major": tv_task_basis_major,
        "tv_task_basis_ood": tv_task_basis_ood,
        "tv_per_task": tv_per_task,
        "tv_random_per_task": tv_random_per_task if show_random_baseline else None,
        "tv_task_basis_per_task": tv_task_basis_per_task if show_task_basis_baseline else None,
        "all_lam": all_lam, "all_post": all_post,
        "W": W, "b": b_vec, "pad_mode": pad_mode,
    }


# ---------------------------------------------------------------------------
# Averaging-based trajectory projection (unified with plot_averaging_r2_linear)
# ---------------------------------------------------------------------------


def traj_averaging_projection_plot(
    exp_name: str,
    layer_index: Optional[int] = None,
    task_ids: Optional[list] = None,
    B: int = 64,
    step: Optional[int] = None,
    estimation_positions: Optional[list] = None,
    plot_positions: Optional[list] = None,
    per_position_mean: bool = True,
    n_ood: Optional[int] = None,
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
    figsize: Optional[tuple] = None,
    show: bool = True,
    show_legend: bool = True,
    show_ylabel: bool = True,
    beta_errbar: str = "std",
    project_beta_simplex: bool = False,
    corner_colors: tuple = ("#0072B2", "#E69F00", "#009E73"),
) -> dict:
    r"""Compare beta(t) from averaging-based task-subspace projection with
    the Bayesian posterior alpha(t).

    Task vectors are estimated by averaging hidden states at late positions
    (same method as ``plot_averaging_r2_linear``).  For each position *t*,
    hidden states are demeaned and projected onto the task subspace to
    obtain coefficients beta_{t,k}.

    The resulting 1x3 figure shows beta (points with error bars) overlaid
    with the Bayesian posterior alpha (dashed lines) for each major task.

    Parameters
    ----------
    exp_name : str
    layer_index : int, optional — defaults to last layer
    task_ids : list of int, optional — which generating tasks to show
        (default ``[0, 1, 2]``, i.e. the 3 major tasks)
    B : int — batch size for evaluation
    step : int, optional — checkpoint step
    estimation_positions : list of int, optional
        Positions used to estimate task vectors (default: last 10).
    plot_positions : list of int, optional
        Positions to include in the plot.  ``None`` → all positions.
    per_position_mean : bool
        If True, subtract per-position grand mean; if False, subtract
        the global grand mean from the estimation step.
    n_ood : int, optional — number of OOD tasks in eval pool
    figsize : tuple, optional
    show : bool
    corner_colors : tuple of 3 colour strings

    Returns
    -------
    dict with 'fig', 'axes', 'results_by_task', 'task_vecs', 'grand_mean'.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from icl.linear.analysis.posterior import task_posterior_over_time_linear_regression
    from icl.linear.linear_path_utils import load_model_task_config
    from icl.linear.linear_ood_analysis import (
        _create_eval_task_pool, _setup_eval_task, setup_device,
    )
    from icl.linear.task_vecs import extract_hidden_multi
    from icl.utils.separability import estimate_task_vectors_by_averaging

    if task_ids is None:
        task_ids = [0, 1, 2]

    _, train_task, config = load_model_task_config(exp_name)
    n_points = int(config.task.n_points)
    k_major = int(train_task.task_pool.shape[0])

    if layer_index is None:
        layer_index = config.model.n_layer - 1
    if estimation_positions is None:
        estimation_positions = list(range(max(0, n_points - 10), n_points))

    # ---- build eval pool: [major, OOD…] ----
    max_tid = max(task_ids)
    if n_ood is None:
        n_ood = max(1, max_tid - k_major + 1) if max_tid >= k_major else 0
    device = setup_device(None)
    major_pool = train_task.task_pool.squeeze(-1).to(device)
    ood_pool_raw, _ = _create_eval_task_pool(
        train_task, K=n_ood, include_minor=False, device=device, n_minor=0,
    )
    n_anchor = int(major_pool.shape[0])
    ood_pool = ood_pool_raw[n_anchor:] if ood_pool_raw.shape[0] > n_anchor \
        else ood_pool_raw.new_empty((0, ood_pool_raw.shape[1]))
    eval_pool = torch.cat([major_pool[:k_major], ood_pool], dim=0)
    eval_task = _setup_eval_task(config, eval_pool, B, device)
    eval_task.batch_size = B

    if step is None:
        step = config.training.total_steps
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)

    pad_mode = getattr(model, "pad", "mapsto")
    task_pos = _task_positions(pad_mode, n_points, device)
    D = int(config.model.n_embd)

    # ---- extract hidden trajectories ----
    demo_data = eval_task.sample_data(step=step).to(device)
    hiddens, all_targets = _extract_hiddens_for_pool(
        model, eval_task, demo_data,
        step=step, layer=layer_index, task_pos=task_pos, D=D, chunk=16,
        post_layernorm=post_layernorm,
        extraction_point=extraction_point,
    )
    # hiddens: (n_eval_tasks, T, B, D)
    T = hiddens.shape[1]
    B_actual = hiddens.shape[2]
    hiddens = hiddens.float()

    # ---- estimate task vectors from major tasks by averaging ----
    hiddens_major = hiddens[:k_major]  # (K, T, B, D)
    task_vecs, grand_mean = estimate_task_vectors_by_averaging(
        hiddens_major, estimation_positions,
    )
    # task_vecs: (K, D), grand_mean: (D,)

    # ---- build decode matrix: drop last row, enforce sum(beta)=1 ----
    V = task_vecs.float()            # (K, D)
    V_basis = V[:-1]                 # (K-1, D)
    # gamma = (V_basis V_basis^T)^{-1} V_basis h^T  gives gamma_k = beta_k - beta_K
    decode_gamma = torch.linalg.solve(
        V_basis @ V_basis.T, V_basis,
    )  # (K-1, D)

    # ---- compute beta(t) for each task ----
    results_by_task = {}
    for tid in task_ids:
        if tid >= hiddens.shape[0]:
            raise ValueError(
                f"task_id={tid} out of range ({hiddens.shape[0]} tasks available)"
            )
        h_tid = hiddens[tid]  # (T, B, D)

        beta_all = np.empty((B_actual, T, k_major), dtype=np.float32)
        for t in range(T):
            h_t = h_tid[t].float()  # (B, D)
            if per_position_mean:
                mu_t = hiddens_major[:, t, :, :].reshape(-1, D).float().mean(dim=0)
            else:
                mu_t = grand_mean.float()
            h_centered = h_t - mu_t.unsqueeze(0)  # (B, D)
            # gamma: (K-1, B), where gamma_k = beta_k - beta_K
            gamma = (decode_gamma @ h_centered.T)  # (K-1, B)
            # Recover beta with sum(beta) = 1
            beta_K = (1.0 - gamma.sum(dim=0)) / k_major  # (B,)
            beta_t = torch.empty(k_major, B_actual, device=V.device)
            beta_t[:k_major - 1] = gamma + beta_K.unsqueeze(0)
            beta_t[k_major - 1] = beta_K
            beta_all[:, t, :] = beta_t.T.cpu().numpy()

        # Bayesian posterior
        post_tensor = task_posterior_over_time_linear_regression(
            train_task, demo_data, all_targets[tid].to(demo_data.device),
            include_minor=False,
        )  # (B, T, K)
        post_all = post_tensor[:, :, :k_major].detach().cpu().numpy()

        results_by_task[tid] = {"beta": beta_all, "post": post_all}

    # ---- select positions to plot ----
    if plot_positions is not None:
        pidx = np.array([t for t in plot_positions if 0 <= t < T])
    else:
        pidx = np.arange(T)
    T_plot = len(pidx)

    # ---- plot: n_rows x 1, all K components on each axes ----
    corner_rgb = [mcolors.to_rgb(c) for c in corner_colors]
    ts = pidx.astype(float)
    n_rows = len(task_ids)

    if figsize is None:
        figsize = (8, 3.5 * n_rows)
    fig, axes_arr = plt.subplots(
        n_rows, 1, figsize=figsize, sharey=True, squeeze=False,
    )

    markers = ("o", "s", "^", "D", "v", "P", "*", "X")
    jitter_width = 1.0
    jitter_offsets = np.linspace(
        -jitter_width / 2, jitter_width / 2, k_major,
    )

    dense_cutoff = 5
    thin_step = 2
    show_masks = []
    for col in range(k_major):
        m = np.zeros(T_plot, dtype=bool)
        m[:dense_cutoff] = True
        for i in range(dense_cutoff, T_plot):
            if (i - dense_cutoff + col) % thin_step == 0:
                m[i] = True
        show_masks.append(m)

    y_lo, y_hi = np.inf, -np.inf
    for row, tid in enumerate(task_ids):
        ax = axes_arr[row, 0]
        beta_all = results_by_task[tid]["beta"][:, pidx, :]  # (B, T_plot, K)
        post_all = results_by_task[tid]["post"][:, pidx, :]  # (B, T_plot, K)
        beta_mean = beta_all.mean(axis=0)
        beta_std = beta_all.std(axis=0)

        if project_beta_simplex:
            for t in range(beta_mean.shape[0]):
                v = beta_mean[t]
                u = np.sort(v)[::-1]
                cssv = np.cumsum(u) - 1.0
                rho = np.nonzero(u > cssv / np.arange(1, len(u) + 1))[0][-1]
                theta = cssv[rho] / (rho + 1.0)
                beta_mean[t] = np.maximum(v - theta, 0.0)

        if beta_errbar == "quantile":
            beta_q_lo = np.percentile(beta_all, 25, axis=0)
            beta_q_hi = np.percentile(beta_all, 75, axis=0)
        post_mean = post_all.mean(axis=0)
        post_q10 = np.percentile(post_all, 10, axis=0)
        post_q90 = np.percentile(post_all, 90, axis=0)

        for col in range(k_major):
            c = corner_rgb[col]
            mk = markers[col % len(markers)]
            ts_jittered = ts + jitter_offsets[col]
            sm = show_masks[col]

            if beta_errbar == "quantile":
                yerr_lo = np.clip(beta_mean[sm, col] - beta_q_lo[sm, col], 0, None)
                yerr_hi = np.clip(beta_q_hi[sm, col] - beta_mean[sm, col], 0, None)
                yerr = [yerr_lo, yerr_hi]
            else:
                yerr = beta_std[sm, col]
            if project_beta_simplex:
                bm = beta_mean[sm, col]
                if isinstance(yerr, list):
                    yerr[0] = np.minimum(yerr[0], bm)
                    yerr[1] = np.minimum(yerr[1], 1.0 - bm)
                else:
                    yerr = [np.minimum(yerr, bm),
                            np.minimum(yerr, 1.0 - bm)]
            ax.errorbar(
                ts_jittered[sm],
                beta_mean[sm, col],
                yerr=yerr,
                fmt=mk, color=c, markersize=3.5, linewidth=1.2,
                capsize=2, capthick=0.8, elinewidth=0.8,
                label=rf"$\beta_{{{col+1}}}$" if row == 0 else None,
                zorder=3,
            )

            ax.plot(
                ts, post_mean[:, col], color=c, lw=2.2, ls="--", alpha=0.9,
                label=rf"$\alpha_{{{col+1}}}$" if row == 0 else None,
                zorder=2,
            )
            ax.fill_between(
                ts, post_q10[:, col], post_q90[:, col],
                color=c, alpha=0.15, linewidth=0, zorder=1,
            )

        if beta_errbar == "quantile":
            beta_lo_arr = beta_q_lo.ravel()
            beta_hi_arr = beta_q_hi.ravel()
        else:
            beta_lo_arr = (beta_mean - beta_std).ravel()
            beta_hi_arr = (beta_mean + beta_std).ravel()
        all_vals = np.concatenate([
            beta_lo_arr, beta_hi_arr,
            post_q10.ravel(),
            post_q90.ravel(),
        ])
        y_lo = min(y_lo, float(np.nanmin(all_vals)))
        y_hi = max(y_hi, float(np.nanmax(all_vals)))

        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(labelsize=14)
        if show_ylabel:
            task_label = f"Task {tid + 1}" if tid < k_major else f"OOD {tid + 1}"
            ax.set_ylabel(task_label, fontsize=16)
        if row == n_rows - 1:
            ax.set_xlabel("Position $t$", fontsize=16)

    if project_beta_simplex:
        y_lo, y_hi, pad = -0.05, 1.05, 0.0
    else:
        pad = 0.08 * (y_hi - y_lo) if y_hi > y_lo else 0.1
    for row in range(n_rows):
        axes_arr[row, 0].set_ylim(y_lo - pad, y_hi + pad)
    if show_legend:
        axes_arr[0, 0].legend(
            fontsize=14, loc="upper left", bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0, framealpha=0.8,
        )
    fig.tight_layout()
    if show_legend:
        fig.subplots_adjust(right=0.78)
    _show_or_close(fig, show)
    axes = axes_arr

    model.cpu(); del model, eval_task
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "fig": fig, "axes": axes,
        "results_by_task": results_by_task,
        "task_vecs": task_vecs, "grand_mean": grand_mean,
        "task_ids": task_ids, "pad_mode": pad_mode,
    }


# ──────────────────────────────────────────────────────────────────────
#  2-D trajectory plot using averaging-based task vectors
# ──────────────────────────────────────────────────────────────────────


def traj_averaging_2d_projection_plot(
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
    estimation_positions: Optional[list] = None,
    estimation_B: int = 128,
    per_position_mean: bool = True,
    show_legend: bool = True,
    figsize: tuple = (9, 7),
    show: bool = True,
    title: str = "",
    annotate: bool = True,
    major_colors: tuple = ("#1a5276", "#2471a3", "#5dade2"),
    ood_base_color: str = "#e74c3c",
    ood_t_show_override: Optional[list] = None,
    major_n_show: int = 7,
    major_line_width: Optional[float] = 2.0,
    ood_line_alpha_factor: float = 0.5,
) -> dict:
    """2-D trajectory projection using averaging-based task vectors.

    Like ``traj_posterior_projection_plot`` but estimates task vectors
    by averaging hidden states at late positions instead of fitting an
    OLS probe.

    Steps:
      1. Extract hiddens for major (+ OOD + minor) tasks at *layer_index*.
      2. ``estimate_task_vectors_by_averaging`` on major-task hiddens.
      3. Subtract per-position or global grand mean to get trajectories.
      4. SVD on centered task vectors → 2-D projection plane.
      5. Plot trajectories coloured by group, sized by R².
    """
    import matplotlib.pyplot as plt
    from icl.linear.linear_utils import estimate_lambda_with_r2
    from icl.linear.linear_path_utils import load_model_task_config
    from icl.linear.linear_ood_analysis import (
        _create_eval_task_pool, _setup_eval_task, setup_device,
    )
    from icl.utils.traj_plot import project_with_r2_trajectories_group_colors_mpl
    from icl.utils.separability import estimate_task_vectors_by_averaging

    _, train_task, config = load_model_task_config(exp_name)
    n_points = int(config.task.n_points)
    k_major = int(train_task.n_tasks)
    D = int(config.model.n_embd)

    if layer_index is None:
        layer_index = config.model.n_layer - 1
    if estimation_positions is None:
        estimation_positions = list(range(max(0, n_points - 10), n_points))
    if step is None:
        step = config.training.total_steps

    # ── Build eval pool: [major, OOD, minor] ─────────────────────────
    device = setup_device(None)
    major_pool = train_task.task_pool.squeeze(-1).to(device)  # (K, D_x)

    ood_pool_raw, _ = _create_eval_task_pool(
        train_task, K=n_ood, include_minor=False, device=device, n_minor=0,
    )
    n_anchor = int(major_pool.shape[0])
    ood_pool = ood_pool_raw[n_anchor:] if ood_pool_raw.shape[0] > n_anchor \
        else ood_pool_raw.new_empty((0, ood_pool_raw.shape[1]))
    n_ood_actual = ood_pool.shape[0]

    k_minor_actual = 0
    minor_pool = ood_pool.new_empty((0, ood_pool.shape[1]))
    if n_minor > 0 and hasattr(train_task, "minor_pool") and train_task.minor_pool is not None:
        mp = train_task.minor_pool.squeeze(-1).to(device)
        k_minor_actual = min(n_minor, mp.shape[0])
        if k_minor_actual > 0:
            minor_pool = mp[:k_minor_actual]

    eval_pool = torch.cat(
        [major_pool[:k_major], ood_pool, minor_pool], dim=0,
    )
    K_total = eval_pool.shape[0]

    eval_task = _setup_eval_task(config, eval_pool, max(B, estimation_B), device)
    eval_task.batch_size = max(B, estimation_B)

    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)

    pad_mode = getattr(model, "pad", "mapsto")
    task_pos = _task_positions(pad_mode, n_points, device)

    # ── Extract hidden states ────────────────────────────────────────
    demo_data = eval_task.sample_data(step=step).to(device)
    hiddens_all, _ = _extract_hiddens_for_pool(
        model, eval_task, demo_data,
        step=step, layer=layer_index, task_pos=task_pos, D=D,
        n_tasks=K_total, chunk=8,
    )
    # hiddens_all: (K_total, T, B_pool, D)
    hiddens_all = hiddens_all.float()
    T = hiddens_all.shape[1]

    # ── Estimate task vectors from major tasks by averaging ──────────
    hiddens_major = hiddens_all[:k_major]  # (K_major, T, B_pool, D)
    task_vecs, grand_mean = estimate_task_vectors_by_averaging(
        hiddens_major, estimation_positions,
    )
    # task_vecs: (K_major, D_model), centered, sum to zero

    # ── Build per-position or global mean ────────────────────────────
    if per_position_mean:
        mu = hiddens_major.mean(dim=(0, 2))  # (T, D)
    else:
        mu = grand_mean.unsqueeze(0).expand(T, -1)  # (T, D)

    # ── Compute trajectory vectors: h(t) − μ(t) ─────────────────────
    if use_mean:
        # Mean across batch: (K_total, T, D)
        tvecs = hiddens_all.mean(dim=2) - mu.unsqueeze(0)
    else:
        B_actual = hiddens_all.shape[2]
        def _sample_group(h, n):
            idx = torch.randint(0, B_actual, (n,))
            return (h[:, :, idx, :].permute(0, 2, 1, 3)
                    .contiguous().view(-1, T, D))

        tvecs = torch.cat([
            _sample_group(hiddens_all[:k_major], b_maj),
            _sample_group(hiddens_all[k_major:k_major + n_ood_actual], b_ood),
            _sample_group(hiddens_all[k_major + n_ood_actual:], b_minor),
        ], dim=0) - mu.unsqueeze(0)

    # ── Project onto task-vector plane and compute R² ────────────────
    W_ref = task_vecs.float()  # (K_major, D) — already centered
    lambdas, r2_scores, *_ = estimate_lambda_with_r2(
        W_ref, tvecs, is_zero_mean=True,
    )

    # ── Time-step selection: first (major_n_show-1) steps + end ─────
    lam_np = np.asarray(lambdas)
    T_len = lam_np.shape[1]
    n_initial = min(max(2, int(major_n_show)) - 1, T_len - 1)
    t_selected = sorted(set(range(n_initial)) | {T_len - 1})
    _ood_ts = [0, -1] if ood_t_show_override is None else ood_t_show_override

    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax, *_ = project_with_r2_trajectories_group_colors_mpl(
        tvecs, W_ref, r2_scores,
        n_minor=k_minor_actual, n_ood=n_ood_actual,
        use_mean=use_mean, b_major=b_maj, b_minor=b_minor, b_ood=b_ood,
        step=step, show_legend=show_legend, title=title, figsize=figsize,
        major_colors=major_colors, ood_base_color=ood_base_color,
        ood_hue_jitter=0.02, ood_sat_jitter=0.05, ood_val_jitter=0.05,
        annotate=annotate, annotate_k=2, annotate_start=False,
        t_show_override=t_selected, ood_t_show_override=_ood_ts,
        show_pow2_anchors=False,
        size_min=5, size_max=16, mid_size_factor=0.5, line_width=1.4,
        major_line_width=major_line_width,
        ood_line_alpha_factor=ood_line_alpha_factor,
    )

    if show:
        plt.show()

    model.cpu(); del model, eval_task
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "fig": fig, "ax": ax,
        "task_vecs_over_all_time": tvecs, "task_vecs": task_vecs,
        "grand_mean": grand_mean, "r2_scores": r2_scores,
        "lambdas": lambdas, "hiddens_all": hiddens_all,
    }
