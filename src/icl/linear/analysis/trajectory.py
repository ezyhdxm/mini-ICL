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
