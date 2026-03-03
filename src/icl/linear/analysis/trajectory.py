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
    """2-D trajectory projection using posterior-derived task vectors.

    Steps:
      1. OLS:  h ≈ πW + b   (W ∈ ℝ^{K×D})
      2. Centre: W̃ = W − w̄1ᵀ
      3. SVD → pick 2 leading directions → project h onto that plane
      4. Plot ID + OOD trajectories over in-context positions
    """
    import matplotlib.pyplot as plt
    from icl.linear.linear_utils import estimate_lambda_with_r2
    from icl.linear.linear_path_utils import load_model_task_config
    from icl.utils.traj_plot import project_with_r2_trajectories_group_colors_mpl

    _, _, config = load_model_task_config(exp_name)
    n_points = int(config.task.n_points)
    if layer_index is None:
        layer_index = config.model.n_layer - 1
    if fit_positions is None:
        fit_positions = list(range(max(0, n_points - 20), n_points))

    # ---- fit probe: posterior -> hiddens (3 major rows in W) ----
    fit_res = train_linear_hidden_predictor(
        exp_name=exp_name, layer=layer_index, n_samples=fit_n_samples,
        positions=fit_positions, sample_mode="major", n_minor=0,
        step=step, print_summary=False, skip_baselines=True,
    )
    W = fit_res["model_weight"]          # (3, D)
    b = fit_res["model_bias"]            # (D,)
    W_mean = W.mean(dim=0, keepdim=True) # (1, D)

    # ---- extract hiddens for all task groups ----
    hiddens, k_minor = _get_hiddens_at_real_positions(
        task_name="linear", exp_name=exp_name,
        n_minor=n_minor, n_ood=n_ood, B=B, step=step,
    )
    hiddens_layer = hiddens[layer_index].to(torch.float32)  # (K, T, B, D)
    K, T, B_actual, D = hiddens_layer.shape
    k_major = 3
    n_ood_actual = K - k_major - k_minor

    b_vec = b.to(hiddens_layer.device).float()

    # ── Build task-vector trajectories ──────────────────────────────
    # The "task vector" at time t is  τ(t) = h(t) − b,  where b is
    # the probe bias.  This removes the input-independent offset,
    # leaving the task-dependent signal.
    if use_mean:
        tvecs = hiddens_layer.mean(dim=-2) - b_vec  # (K, T, D)
    else:
        def _sample_group(h, n):
            idx = torch.randint(0, B_actual, (n,), device=h.device)
            return h[:, :, idx, :].permute(0, 2, 1, 3).contiguous().view(-1, T, D)

        tvecs = torch.cat([
            _sample_group(hiddens_layer[:k_major], b_maj),
            _sample_group(hiddens_layer[k_major:K - k_minor], b_ood),
            _sample_group(hiddens_layer[K - k_minor:], b_minor),
        ], dim=0) - b_vec

    # ── Project onto centred task-weight matrix ──────────────────
    # W̃ = W − w̄1ᵀ  (centre rows).  Then solve the least-squares
    # problem  τ(t) − w̄ ≈ λ(t)ᵀ W̃  for barycentric coordinates
    # λ(t) ∈ ℝ^K.  λₖ(t) ≈ 1 means the hidden state at time t
    # looks like task k; λₖ(t) ≈ 0 means no resemblance.
    dev = tvecs.device
    W_c = (W - W_mean).to(dev)
    lambdas, r2_scores, *_ = estimate_lambda_with_r2(
        W_c, tvecs - W_mean.to(dev), is_zero_mean=True,
    )

    # ---- pick annotation time-steps that spread trajectories apart ----
    lam_np = np.asarray(lambdas)[:, :, :2]  # (K_traj, T, 2)
    K_traj, T_len = lam_np.shape[:2]
    selected = [0, T_len - 1]
    for _ in range(min(4, T_len) - 2):
        best_t, best_score = -1, -1.0
        for t in range(T_len):
            if t in selected:
                continue
            dists = [min(np.linalg.norm(lam_np[k, t] - lam_np[k, s]) for s in selected)
                     for k in range(K_traj)]
            score = np.median(dists)
            if score > best_score:
                best_score, best_t = score, t
        if best_t >= 0:
            selected.append(best_t)

    fig, ax, *_ = project_with_r2_trajectories_group_colors_mpl(
        tvecs, W, r2_scores,
        n_minor=k_minor, n_ood=n_ood_actual,
        use_mean=use_mean, b_major=b_maj, b_minor=b_minor, b_ood=b_ood,
        step=step, show_legend=show_legend, title=title, figsize=figsize,
        major_colors=major_colors, ood_base_color=ood_base_color,
        ood_hue_jitter=0.02, ood_sat_jitter=0.05, ood_val_jitter=0.05,
        annotate=annotate, annotate_k=2, annotate_start=False,
        t_show_override=sorted(selected), show_pow2_anchors=False,
        size_min=5, size_max=16, mid_size_factor=0.5, line_width=1.4,
    )

    if show:
        plt.show()

    return {
        "fig": fig, "ax": ax,
        "task_vecs_over_all_time": tvecs, "final_task_vecs": W,
        "r2_scores": r2_scores, "lambdas": lambdas,
        "hiddens_layer": hiddens_layer, "W": W, "b": b,
        "fit_results": fit_res,
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
    n_ood: Optional[int] = None,
    center_lambda: bool = True,
    figsize: Optional[tuple] = None,
    show: bool = True,
    corner_colors: tuple = ("#1f77b4", "#2ca02c", "#d62728"),
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
        positions=fit_positions, sample_mode="major", n_minor=0,
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

    # ── Estimate λ(t) and oracle posterior for each task ────────────
    # λ(t) = least-squares coefficients solving  (h−b) − w̄ ≈ λᵀW̃.
    # When center_lambda is True, λ is projected onto the probability
    # simplex Δ = {λ ≥ 0, Σλ = 1} for interpretability.
    # Posterior P(z=k|data) is the Bayesian gold standard.
    hiddens = hiddens.float()
    k_major = 3
    T, B_actual = hiddens.shape[1], hiddens.shape[2]
    dev = hiddens.device
    b_vec = b.to(dev).float()
    W_c = (W - W_mean).to(dev).float()
    W_f = W.to(dev).float()
    W_mean_f = W_mean.to(dev).float()

    n_indiv = min(n_individual, B_actual)
    indiv_indices = torch.randperm(B_actual)[:n_indiv]

    results_by_task = {}
    for tid in task_ids:
        h_all = hiddens[tid]  # (T, B, D)

        lam_list = []
        for bi in range(B_actual):
            h = h_all[:, bi, :].unsqueeze(0)  # (1, T, D)
            if center_lambda:
                lam_raw, *_ = estimate_lambda_with_r2(W_c, h - b_vec - W_mean_f, is_zero_mean=True)
            else:
                lam_raw, *_ = estimate_lambda_with_r2(W_f, h - b_vec, is_zero_mean=False)
            lam_np = np.asarray(lam_raw)[0]  # (T, 3)
            lam_list.append(
                _project_onto_simplex(lam_np) if center_lambda
                else np.clip(lam_np, 0.0, 1.0)
            )
        lam_all = np.stack(lam_list, axis=0)  # (B, T, 3)

        post_tensor = task_posterior_over_time_linear_regression(
            train_task, demo_data, task_targets[tid].to(demo_data.device),
            include_minor=False,
        )  # (B, T, K)
        post_all = post_tensor[:, :, :k_major].detach().cpu().numpy()  # (B, T, 3)

        results_by_task[tid] = {"lam": lam_all, "post": post_all}

    # ---- plot ----
    corner_rgb = [mcolors.to_rgb(c) for c in corner_colors]
    ts = np.arange(T)
    n_plot_rows = len(task_ids)
    if figsize is None:
        figsize = (14, 3.5 * n_plot_rows)
    fig, axes = plt.subplots(n_plot_rows, 3, figsize=figsize, sharey=True, squeeze=False)

    mode_str = "simplex" if center_lambda else "clipped"
    for row, tid in enumerate(task_ids):
        lam_all = results_by_task[tid]["lam"]   # (B, T, 3)
        post_all = results_by_task[tid]["post"] # (B, T, 3)
        lam_mean = lam_all.mean(axis=0)         # (T, 3)
        lam_std = lam_all.std(axis=0)           # (T, 3)
        post_mean = post_all.mean(axis=0)       # (T, 3)
        post_std = post_all.std(axis=0)         # (T, 3)

        for col in range(3):
            ax = axes[row, col]
            c = corner_rgb[col]

            # individual traces
            for idx in indiv_indices:
                ii = int(idx)
                ax.plot(ts, lam_all[ii, :, col], color=c, lw=0.5, alpha=0.12)

            # mean lambda + std band
            ax.plot(ts, lam_mean[:, col], color=c, lw=2.5, alpha=0.95,
                    label=rf"$\lambda_{col+1}$ mean ({mode_str})")
            lo = np.clip(lam_mean[:, col] - lam_std[:, col], 0.0, 1.0)
            hi = np.clip(lam_mean[:, col] + lam_std[:, col], 0.0, 1.0)
            ax.fill_between(ts, lo, hi, color=c, alpha=0.15)

            # mean posterior + std band
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
            task_label = f"Task {tid}" if tid < 3 else f"OOD {tid}"
            if col == 0:
                ax.set_ylabel(task_label, fontsize=16)
            if row == 0:
                ax.set_title(rf"$\lambda_{col+1}$ / $\mathbb{{P}}(Z\!={col+1}|\mathrm{{data}})$", fontsize=15)
            if row == 0 and col == 0:
                ax.legend(fontsize=10, loc="best", framealpha=0.7)

    fig.suptitle(f"Layer {layer_index}  (B={B_actual}, {n_indiv} traces shown)",
                 fontsize=15, y=1.01)
    _show_or_close(fig, show)

    model.cpu(); del model, eval_task
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "fig": fig, "axes": axes,
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
    window: int = 20,
    figsize: tuple = (18, 5),
    show: bool = True,
    title: str = "",
    eps: float = 1e-12,
) -> dict:
    """Quantify agreement between λ(t) and P(z|data) across positions.

    λ(t) = simplex_proj((h−b) @ W̃ᵀ)  from the linear probe;
    P(z=k|data) from the oracle posterior.

    Three panels per group (major / OOD): TV distance, cosine similarity,
    rolling Pearson correlation.
    """
    import matplotlib.pyplot as plt
    from icl.linear.linear_utils import estimate_lambda_with_r2
    from icl.linear.analysis.posterior import task_posterior_over_time_linear_regression
    from icl.linear.linear_path_utils import load_model_task_config
    from icl.linear.linear_ood_analysis import _create_eval_task_pool, _setup_eval_task, setup_device

    _, train_task, config = load_model_task_config(exp_name)
    n_points = int(config.task.n_points)
    if layer_index is None:
        layer_index = config.model.n_layer - 1
    if fit_positions is None:
        fit_positions = list(range(max(0, n_points - 20), n_points))

    # ---- fit probe ----
    fit_res = train_linear_hidden_predictor(
        exp_name=exp_name, layer=layer_index, n_samples=fit_n_samples,
        positions=fit_positions, sample_mode="major", n_minor=0,
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
    ood_pool, _ = _create_eval_task_pool(
        train_task, K=n_ood, include_minor=False, device=device, n_minor=0,
    )
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

    def _cosine_sim(a, b_arr):
        """cos(a, b) = ⟨a, b⟩ / (‖a‖ ‖b‖),  computed row-wise."""
        dot = (a * b_arr).sum(axis=-1)
        return dot / (np.linalg.norm(a, axis=-1).clip(eps) * np.linalg.norm(b_arr, axis=-1).clip(eps))

    def _rolling_pearson(a, b_arr, w):
        """Pearson correlation between a and b in a sliding window of size w.

        At each time t, computes  ρ(a[t-w+1:t+1, c], b[t-w+1:t+1, c])
        for each coordinate c, then averages over c.
        """
        T_len, C = a.shape
        out = np.full(T_len, np.nan)
        for t in range(w - 1, T_len):
            corrs = []
            for c in range(C):
                xw = a[t - w + 1:t + 1, c]
                yw = b_arr[t - w + 1:t + 1, c]
                if xw.std() < eps or yw.std() < eps:
                    continue
                dx, dy = xw - xw.mean(), yw - yw.mean()
                corrs.append(float(np.dot(dx, dy) / np.sqrt((dx**2).sum() * (dy**2).sum())))
            if corrs:
                out[t] = np.mean(corrs)
        return out

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
    # TV distance:  TV(λ, P) = ½ Σₖ |λₖ − Pₖ|   ∈ [0, 1]
    #   (0 = perfect agreement; 1 = maximally different distributions)
    # Cosine similarity between the two K-vectors at each time step.
    all_lam = np.zeros((K, B_actual, T, 3), dtype=float)
    all_post = np.zeros((K, B_actual, T, 3), dtype=float)
    tv_per_task = np.zeros((K, T))
    cos_per_task = np.zeros((K, T))

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
            tv_per_task[k] += 0.5 * np.abs(p - lam).sum(axis=-1)  # TV distance
            cos_per_task[k] += _cosine_sim(lam, p)

        tv_per_task[k] /= B_actual
        cos_per_task[k] /= B_actual

    # ---- aggregate major / OOD ----
    tv_major = _safe_mean(tv_per_task[:k_major])
    tv_ood = _safe_mean(tv_per_task[k_major:])
    cos_major = _safe_mean(cos_per_task[:k_major])
    cos_ood = _safe_mean(cos_per_task[k_major:])

    rcorr_per_task = np.full((K, T), np.nan)
    for k in range(K):
        sample_corrs = np.stack([
            _rolling_pearson(all_lam[k, bi], all_post[k, bi], window)
            for bi in range(B_actual)
        ])
        rcorr_per_task[k] = _safe_mean(sample_corrs)

    rcorr_major = _safe_mean(rcorr_per_task[:k_major])
    rcorr_ood = _safe_mean(rcorr_per_task[k_major:])
    ts = np.arange(T)

    # ---- plot ----
    panels = [
        ("TV distance",                     tv_major, tv_ood, dict(ylim=(0, None))),
        ("Cosine similarity",               cos_major, cos_ood, dict(ylim=(0.5, 1.02))),
        (f"Rolling correlation (w={window})", rcorr_major, rcorr_ood, {}),
    ]
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (ylabel, maj, ood, kw) in zip(axes, panels):
        vm, vo = ~np.isnan(maj), ~np.isnan(ood)
        ax.plot(ts[vm], maj[vm], lw=2.5, label="Major tasks", color="#1f77b4")
        ax.plot(ts[vo], ood[vo], lw=2.5, label="OOD tasks", color="#d62728")
        ax.set(xlabel="Position", ylabel=ylabel, **kw)
        ax.xaxis.label.set_fontsize(18); ax.yaxis.label.set_fontsize(18)
        ax.tick_params(labelsize=16); ax.legend(fontsize=14); ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=18)
    _show_or_close(fig, show)

    model.cpu(); del model, eval_task
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "fig": fig, "axes": axes, "positions": ts,
        "tv_major": tv_major, "tv_ood": tv_ood,
        "cos_major": cos_major, "cos_ood": cos_ood,
        "rcorr_major": rcorr_major, "rcorr_ood": rcorr_ood,
        "tv_per_task": tv_per_task, "cos_per_task": cos_per_task,
        "rcorr_per_task": rcorr_per_task,
        "all_lam": all_lam, "all_post": all_post,
        "W": W, "b": b_vec, "pad_mode": pad_mode,
    }
