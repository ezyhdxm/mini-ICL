"""Trajectory projection plots for the Coin task."""

import numpy as np
import torch
from typing import Optional, List

import icl.utils.notebook_utils as nu
from icl.coin.analysis.probes import train_linear_hidden_predictor_coin
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)


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
    figsize: tuple = (7, 5.5),
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
    """
    2-D trajectory projection for coin (non-padded) using **averaging-based**
    task vectors.

    Task vectors are estimated by averaging hidden states of major tasks at
    late positions (``estimation_positions``), then centering.  The projection
    plane is spanned by these averaged task vectors.

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
    estimation_positions : list, optional
        Positions used to estimate task vectors by averaging.
        ``None`` → last 30 positions of the sequence.
    extraction_point : str
        ``"post_attn"`` (default) — after the attention block, before MLP.
        ``"post_mlp"`` — after the full block (residual stream).
    show_legend : bool
    figsize : tuple
    show : bool
    title : str
    ood_t_show_override : list, optional
        Time indices shown for OOD trajectories (absolute or python-negative).
        Default ``[0, -1]``: show only the start and end point, connected by a
        single line — much cleaner when many OOD tasks are plotted.
        Pass ``None`` to use the same dense time grid as the major trajectories.
    major_n_show : int
        Total number of time points shown along each major trajectory
        (including the end point).  Shows the first ``major_n_show - 1``
        time steps (t = 0, 1, …, major_n_show-2) plus the final step,
        so the trajectory captures early dynamics and the converged state.
        Default 6 (first 5 steps + end).

    Returns
    -------
    dict
        ``{'fig', 'ax', 'task_vecs_over_all_time', 'final_task_vecs',
        'r2_scores', 'lambdas', 'hiddens_layer', 'task_vecs', 'grand_mean'}``.
    """
    import matplotlib.pyplot as plt
    from icl.utils.unified_interface import _get_hiddens_at_real_positions
    from icl.linear.linear_utils import estimate_lambda_with_r2
    from icl.utils.traj_plot import project_with_r2_trajectories_group_colors_mpl
    from icl.utils.separability import estimate_task_vectors_by_averaging

    # ---- 0. resolve layer & config ----
    _, _, config = nu.load_everything("coin", exp_name)
    n_layers = config.model.num_layers
    if layer_index is None:
        layer_index = n_layers - 1

    # ---- 1. extract hiddens ----
    hiddens, k_minor = _get_hiddens_at_real_positions(
        task_name="coin", exp_name=exp_name,
        n_minor=n_minor, n_ood=n_ood, B=B, step=step,
        extraction_point=extraction_point,
    )

    hiddens_layer = hiddens[layer_index].to(torch.float32)  # (K, T, B, D)
    K, T, B_actual, D = hiddens_layer.shape
    k_major = 3

    # ---- 2. estimate task vectors from major tasks by averaging ----
    hiddens_major = hiddens_layer[:k_major]  # (k_major, T, B, D)
    if estimation_positions is None:
        estimation_positions = list(range(max(0, T - 30), T))
    task_vecs, grand_mean = estimate_task_vectors_by_averaging(
        hiddens_major, estimation_positions,
    )
    final_task_vecs = task_vecs.float()  # (k_major, D), already centered
    grand_mean_dev = grand_mean.to(hiddens_layer.device)

    # ---- 3. task_vecs_over_all_time ----
    if use_mean:
        task_vecs_over_all_time = hiddens_layer.mean(dim=-2) - grand_mean_dev  # (K, T, D)
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
            torch.cat([out_major, out_mid, out_minor], dim=0) - grand_mean_dev
        )

    # ---- 4. lambda / R² ----
    lambdas, r2_scores, _, _ = estimate_lambda_with_r2(
        final_task_vecs,
        task_vecs_over_all_time,
        is_zero_mean=True,
    )

    # ---- 4b. pick time indices for major: first (major_n_show-1) steps + end ----
    n_show = max(2, int(major_n_show))
    lam_np = lambdas if isinstance(lambdas, np.ndarray) else np.asarray(lambdas)
    T_len = lam_np.shape[1]
    n_initial = min(n_show - 1, T_len - 1)  # leave room for the end point
    t_selected = sorted(set(range(n_initial)) | {T_len - 1})

    # ---- 5. plot ----
    # Default: OOD shows start + midpoint + end (adds one intermediate to fill empty space).
    _ood_ts = [0, -1] if ood_t_show_override is None else ood_t_show_override
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
        ood_t_show_override=_ood_ts,
        show_pow2_anchors=False,
        size_min=5,
        size_max=16,
        mid_size_factor=0.5,
        line_width=1.4,
        major_line_width=major_line_width,
        ood_line_alpha_factor=ood_line_alpha_factor,
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
        'task_vecs': task_vecs,
        'grand_mean': grand_mean,
    }


def traj_post_posterior_projection_plot_coin(
    exp_name: str,
    layer_index: Optional[int] = None,
    task_ids: Optional[list] = None,
    n_individual: int = 10,
    show_individual_traces: bool = False,
    B: int = 64,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    fit_include_position_bias: bool = True,
    fit_include_logit: bool = False,
    n_ood: Optional[int] = None,
    center_lambda: bool = True,
    show_std_curves: bool = False,
    annotate_agreement: bool = True,
    show_std_panel: bool = True,
    std_figsize: Optional[tuple] = None,
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
    show_individual_traces : bool
        If True, draw individual sample traces behind means.
    B : int
    step : int, optional
    fit_n_samples : int
    fit_positions : list, optional
    fit_include_position_bias : bool
        Whether to include one-hot position nuisance features in probe fitting.
    fit_include_logit : bool
        Reserved for API consistency. Coin hidden probe has no logit block,
        so this must stay ``False``.
    n_ood : int, optional
        Number of OOD tasks to generate. Auto-determined from *task_ids*
        if not given.
    center_lambda : bool
        If True, project λ onto the probability simplex. Otherwise clip
        to [0, 1].
    show_std_curves : bool
        Deprecated. Std is shown via shaded bands for both λ and posterior.
    annotate_agreement : bool
        If True, annotate each panel with correlation of means and stds.
    show_std_panel : bool
        If True, create a second figure with std-only curves.
    std_figsize : tuple, optional
        Figure size for std-only panel grid. ``None`` uses ``figsize``.
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
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
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
    W = fit_res["model_weight"]        # (K_major, D)
    b = fit_res["model_bias"]          # (D,)
    W_mean = W.mean(dim=0, keepdim=True)  # (1, D)

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

    # h ≈ π W_task + x_t W_tok + p W_pos + b + ε
    # Subtract all available nuisance terms before projecting onto task basis.
    b_vec = b.float().to(dev)
    W_mean_f = W_mean.float().to(dev)
    W_c = (W - W_mean).float().to(dev)
    W_f = W.float().to(dev)

    def _to_float_device(t):
        if isinstance(t, torch.Tensor) and t.numel() > 0:
            return t.float().to(dev)
        return torch.zeros((0, D), dtype=torch.float32, device=dev)

    W_tok_f = _to_float_device(fit_res.get("token_weight", None))
    W_pos_f = _to_float_device(fit_res.get("position_weight", None))
    fit_pos_list = [int(p) for p in fit_res.get("positions", fit_positions)]
    fit_pos_to_col = {p: i for i, p in enumerate(fit_pos_list)}

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

    def _safe_corr(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.size == 0 or b.size == 0:
            return float("nan")
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    n_indiv = min(n_individual, B_actual)
    indiv_indices = torch.randperm(B_actual)[:n_indiv] if show_individual_traces else torch.tensor([], dtype=torch.long)

    results_by_task = {}
    agreement_metrics = {}
    for tid in task_ids:
        h_tid = hiddens_layer[tid].permute(1, 0, 2).contiguous()  # (B, T, D)
        nuisance = torch.zeros_like(h_tid)

        if W_tok_f.shape[0] > 0:
            tok_tid = demo_data[tid, :B_actual, :T].long().to(dev)
            max_tok = int(tok_tid.max().item()) if tok_tid.numel() > 0 else -1
            if max_tok >= W_tok_f.shape[0]:
                raise ValueError(
                    f"Token id {max_tok} out of range for token_weight rows={W_tok_f.shape[0]}",
                )
            nuisance = nuisance + W_tok_f[tok_tid]

        if W_pos_f.shape[0] > 0:
            pos_effect = torch.zeros((T, D), dtype=torch.float32, device=dev)
            for t in range(T):
                j = fit_pos_to_col.get(t, None)
                if j is not None and j < W_pos_f.shape[0]:
                    pos_effect[t] = W_pos_f[j]
            nuisance = nuisance + pos_effect.unsqueeze(0)

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

        post_tensor = posterior_over_models_over_time_per_sample(
            demo_data[tid:tid + 1], sampler_clone.major_p,
        ).float()  # (1, B, seq_len, K_major)
        post_all = post_tensor[0, :, :T, :k_major].cpu().numpy()  # (B, T, K_major)

        results_by_task[tid] = {"lam": lam_all, "post": post_all}
        lam_mean_t = lam_all.mean(axis=0)   # (T, K_major)
        lam_std_t = lam_all.std(axis=0)
        post_mean_t = post_all.mean(axis=0)
        post_std_t = post_all.std(axis=0)
        agreement_metrics[tid] = {
            "mean_corr": [_safe_corr(lam_mean_t[:, j], post_mean_t[:, j]) for j in range(k_major)],
            "std_corr": [_safe_corr(lam_std_t[:, j], post_std_t[:, j]) for j in range(k_major)],
        }

    # ---- plot (split into major and OOD figures) ----
    corner_rgb = [mcolors.to_rgb(c) for c in corner_colors]
    ts = np.arange(T)
    mode_str = "simplex" if center_lambda else "clipped"

    major_task_ids = [tid for tid in task_ids if tid < k_major]
    ood_task_ids = [tid for tid in task_ids if tid >= k_major]

    def _group_figsize(n_rows: int):
        if figsize is not None:
            return figsize
        return (14, 3.5 * max(n_rows, 1))

    def _plot_group(group_task_ids, group_name):
        if not group_task_ids:
            return None, None
        n_rows_local = len(group_task_ids)
        fig_local, axes_local = plt.subplots(
            n_rows_local, k_major,
            figsize=_group_figsize(n_rows_local),
            sharey=True, squeeze=False,
        )
        for row, tid in enumerate(group_task_ids):
            lam_all = results_by_task[tid]["lam"]
            post_all = results_by_task[tid]["post"]
            lam_mean = lam_all.mean(axis=0)
            lam_std = lam_all.std(axis=0)
            post_mean = post_all.mean(axis=0)
            post_std = post_all.std(axis=0)

            for col in range(k_major):
                ax = axes_local[row, col]
                c = corner_rgb[col]
                if show_individual_traces:
                    for idx in indiv_indices:
                        ii = int(idx)
                        ax.plot(ts, lam_all[ii, :, col], color=c, lw=0.5, alpha=0.10)

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
                ax.tick_params(labelsize=14)
                if row == n_rows_local - 1:
                    ax.set_xlabel("Position $t$", fontsize=16)
                if col == 0:
                    task_label = f"Task {tid}" if tid < k_major else f"OOD {tid}"
                    ax.set_ylabel(task_label, fontsize=16)
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
                    ax.legend(handles=handles, fontsize=12, loc="best", framealpha=0.75)
                if annotate_agreement:
                    m_corr = agreement_metrics[tid]["mean_corr"][col]
                    s_corr = agreement_metrics[tid]["std_corr"][col]
                    ax.text(
                        0.03, 0.97,
                        f"r_mean={m_corr:.2f}\nr_std={s_corr:.2f}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.5, edgecolor="none"),
                    )
        fig_local.suptitle("", fontsize=18, y=1.01)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig_local)
        return fig_local, axes_local

    fig_major, axes_major = _plot_group(major_task_ids, "major")
    fig_ood, axes_ood = _plot_group(ood_task_ids, "ood")
    fig = fig_major if fig_major is not None else fig_ood
    axes = axes_major if axes_major is not None else axes_ood

    std_fig, std_axes = None, None
    if show_std_panel:
        if std_figsize is None:
            std_figsize = figsize

        def _plot_std_group(group_task_ids):
            if not group_task_ids:
                return None, None
            n_rows_local = len(group_task_ids)
            fig_local, axes_local = plt.subplots(
                n_rows_local, k_major,
                figsize=(std_figsize if std_figsize is not None else _group_figsize(n_rows_local)),
                sharey=True, squeeze=False,
            )
            for row, tid in enumerate(group_task_ids):
                lam_all = results_by_task[tid]["lam"]
                post_all = results_by_task[tid]["post"]
                lam_std = lam_all.std(axis=0)
                post_std = post_all.std(axis=0)
                for col in range(k_major):
                    ax = axes_local[row, col]
                    c = corner_rgb[col]
                    post_color = "#111111"
                    ax.plot(ts, lam_std[:, col], color=c, lw=2.2, ls="-",
                            label=rf"$\lambda_{col+1}$ std")
                    ax.plot(ts, post_std[:, col], color=post_color, lw=2.2, ls="--",
                            label=rf"$P(Z\!={col+1})$ std")
                    ax.set_ylim(bottom=0.0)
                    ax.grid(axis="y", alpha=0.3)
                    ax.tick_params(labelsize=13)
                    if row == n_rows_local - 1:
                        ax.set_xlabel("Position $t$", fontsize=14)
                    if col == 0:
                        task_label = f"Task {tid}" if tid < k_major else f"OOD {tid}"
                        ax.set_ylabel(task_label, fontsize=14)
                    if row == 0:
                        ax.legend(fontsize=12, loc="best", framealpha=0.75)
            fig_local.suptitle("", fontsize=16, y=1.01)
            plt.tight_layout()
            if show:
                plt.show()
            else:
                plt.close(fig_local)
            return fig_local, axes_local

        std_fig_major, std_axes_major = _plot_std_group(major_task_ids)
        std_fig_ood, std_axes_ood = _plot_std_group(ood_task_ids)
        std_fig = std_fig_major if std_fig_major is not None else std_fig_ood
        std_axes = std_axes_major if std_axes_major is not None else std_axes_ood
    else:
        std_fig_major = std_axes_major = std_fig_ood = std_axes_ood = None

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
        "agreement_metrics": agreement_metrics,
        "fig_major": fig_major,
        "axes_major": axes_major,
        "fig_ood": fig_ood,
        "axes_ood": axes_ood,
        "std_fig": std_fig,
        "std_axes": std_axes,
        "std_fig_major": std_fig_major,
        "std_axes_major": std_axes_major,
        "std_fig_ood": std_fig_ood,
        "std_axes_ood": std_axes_ood,
    }


# ---------------------------------------------------------------------------
#  traj_cellmean_projection_plot_coin
# ---------------------------------------------------------------------------

def traj_cellmean_projection_plot_coin(
    exp_name: str,
    layer_index: Optional[int] = None,
    task_ids: Optional[List[int]] = None,
    B: int = 64,
    step: Optional[int] = None,
    estimation_positions: Optional[List[int]] = None,
    estimation_B: int = 256,
    n_ood: Optional[int] = None,
    project_simplex: bool = True,
    show_individual_traces: bool = False,
    n_individual: int = 10,
    annotate_agreement: bool = True,
    figsize: Optional[tuple] = None,
    show: bool = True,
    corner_colors: tuple = ("#1f77b4", "#2ca02c", "#d62728"),
) -> dict:
    r"""Compare lambda(t) from cell-mean projection with Bayesian posterior.

    Instead of fitting a joint OLS probe (posterior + token -> hiddens),
    this function:

    1. Estimates task-token cell means  theta_{k,a} = E[h_t | z=k, s_t=a]
       by averaging hidden states from natural sequences at late-context
       positions, grouped by the observed token.
    2. For each evaluation sample at position t with token s_t = a,
       decomposes h_t onto {theta_{k,a}}_k under the affine constraint
       sum_k lambda_k = 1, yielding lambda(t).
    3. Compares lambda(t) to the Bayesian posterior pi_t.

    Only major tasks are used for cell-mean estimation and posterior
    computation.

    Parameters
    ----------
    exp_name : str
    layer_index : int, optional
        ``None`` -> last layer.
    task_ids : list of int, optional
        Which tasks to plot. Default ``[0, 1, 2]`` (major only).
    B : int
        Batch size for evaluation trajectories.
    step : int, optional
    estimation_positions : list of int, optional
        Positions used to estimate cell means. ``None`` -> last half of
        sequence.
    estimation_B : int
        Batch size for cell-mean estimation sequences (larger = more
        stable cell means). Default 256.
    n_ood : int, optional
        Number of OOD tasks for evaluation. Auto-determined from task_ids.
    project_simplex : bool
        If True, project lambda onto the probability simplex.
    show_individual_traces : bool
    n_individual : int
    annotate_agreement : bool
    figsize : tuple, optional
    show : bool
    corner_colors : tuple of 3 hex strings

    Returns
    -------
    dict with keys 'fig', 'axes', 'results_by_task', 'cell_means',
    'task_ids', 'agreement_metrics'.
    """
    import gc
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from icl.utils.unified_interface import _get_hiddens_at_real_positions
    from icl.utils.unified_plot import posterior_over_models_over_time_per_sample
    from icl.utils.linear_algebra_utils import project_onto_cellmeans

    if task_ids is None:
        task_ids = [0, 1, 2]

    _, _, config = nu.load_everything("coin", exp_name)
    n_layers = config.model.num_layers
    seq_len = config.seq_len
    k_major = 3

    if layer_index is None:
        layer_index = n_layers - 1

    if estimation_positions is None:
        estimation_positions = list(range(seq_len // 2, seq_len - 1))

    # ---- 1. Get natural trajectories for cell-mean estimation (major only) ----
    logger.info(
        f"[cellmean coin] generating {estimation_B} natural sequences per "
        f"major task for cell-mean estimation, layer {layer_index}"
    )
    est_hiddens, _, est_demo_data, _ = _get_hiddens_at_real_positions(
        task_name="coin", exp_name=exp_name,
        n_minor=0, n_ood=0, B=estimation_B, step=step,
        return_data=True,
    )
    # est_hiddens: (L, K_major, T, B_est, D)
    est_h = est_hiddens[layer_index].to(torch.float32)  # (K_major, T, B_est, D)
    est_tokens = est_demo_data[:k_major]  # (K_major, B_est, seq_len)

    # Collect unique token values from the estimation data at est positions
    est_pos = estimation_positions
    tok_vals = set()
    for k_idx in range(k_major):
        for p in est_pos:
            if p < est_tokens.shape[2]:
                tok_vals.update(est_tokens[k_idx, :, p].unique().tolist())
    unique_tokens = sorted(tok_vals)
    n_unique = len(unique_tokens)
    tok_to_idx = {int(v): i for i, v in enumerate(unique_tokens)}

    # Compute cell means: for each (task k, token a), average h over
    # (batch, position) where s_t == a at late positions.
    D = est_h.shape[-1]
    cell_sums = torch.zeros(n_unique, k_major, D, dtype=torch.float32)
    cell_counts = torch.zeros(n_unique, k_major, dtype=torch.long)

    for k_idx in range(k_major):
        for p in est_pos:
            if p >= est_h.shape[1]:
                continue
            h_p = est_h[k_idx, p]             # (B_est, D)
            tok_p = est_tokens[k_idx, :, p]   # (B_est,)
            for a_val, a_idx in tok_to_idx.items():
                mask = (tok_p == a_val)
                if mask.any():
                    cell_sums[a_idx, k_idx] += h_p[mask].sum(dim=0).cpu()
                    cell_counts[a_idx, k_idx] += mask.sum().item()

    # Guard against empty cells
    safe_counts = cell_counts.clamp(min=1).unsqueeze(-1).float()
    cell_means = cell_sums / safe_counts  # (n_unique, K_major, D)

    min_count = int(cell_counts.min().item())
    logger.info(
        f"[cellmean coin] cell means computed: {n_unique} tokens x {k_major} tasks, "
        f"min cell count = {min_count}, "
        f"total samples per task = {len(est_pos) * estimation_B}"
    )

    del est_hiddens, est_h, est_tokens, est_demo_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ---- 2. Get evaluation trajectories ----
    max_tid = max(task_ids)
    if n_ood is None:
        n_ood = max(1, max_tid - 2) if max_tid > 2 else 0
    hiddens, _k_minor, demo_data, sampler_clone = _get_hiddens_at_real_positions(
        task_name="coin", exp_name=exp_name,
        n_minor=0, n_ood=n_ood, B=B, step=step,
        return_data=True,
    )

    hiddens_layer = hiddens[layer_index].to(torch.float32)  # (K, T, B, D)
    K, T, B_actual, _ = hiddens_layer.shape
    dev = hiddens_layer.device

    for tid in task_ids:
        if tid >= K:
            raise ValueError(f"task_id={tid} out of range (K={K} tasks available)")

    cell_means_dev = cell_means.to(dev)

    # Remap raw token values to cell_means indices
    remap_size = int(max(unique_tokens)) + 1
    tok_remap = torch.full((remap_size,), -1, dtype=torch.long, device=dev)
    for val, idx in tok_to_idx.items():
        tok_remap[int(val)] = idx

    # ---- 3. Project each task's trajectories ----
    def _safe_corr(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.size == 0 or b.size == 0 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    results_by_task = {}
    agreement_metrics = {}
    for tid in task_ids:
        h_tid = hiddens_layer[tid].permute(1, 0, 2).contiguous()  # (B, T, D)
        raw_tok = demo_data[tid, :B_actual, :T].long().to(dev)    # (B, T)
        tok_tid = tok_remap[raw_tok]  # remap to cell_means indices

        lam_all = project_onto_cellmeans(
            cell_means_dev, h_tid, tok_tid,
            project_simplex=project_simplex,
        )  # (B, T, K_major) numpy

        post_tensor = posterior_over_models_over_time_per_sample(
            demo_data[tid:tid + 1], sampler_clone.major_p,
        ).float()  # (1, B, seq_len, K_major)
        post_all = post_tensor[0, :, :T, :k_major].cpu().numpy()

        results_by_task[tid] = {"lam": lam_all, "post": post_all}
        lam_mean = lam_all.mean(axis=0)
        lam_std = lam_all.std(axis=0)
        post_mean = post_all.mean(axis=0)
        post_std = post_all.std(axis=0)
        agreement_metrics[tid] = {
            "mean_corr": [_safe_corr(lam_mean[:, j], post_mean[:, j]) for j in range(k_major)],
            "std_corr": [_safe_corr(lam_std[:, j], post_std[:, j]) for j in range(k_major)],
        }

    # ---- 4. Plot ----
    corner_rgb = [mcolors.to_rgb(c) for c in corner_colors]
    ts = np.arange(T)
    mode_str = "simplex" if project_simplex else "raw"

    major_task_ids = [tid for tid in task_ids if tid < k_major]
    ood_task_ids = [tid for tid in task_ids if tid >= k_major]

    n_indiv = min(n_individual, B_actual)
    indiv_indices = (
        torch.randperm(B_actual)[:n_indiv]
        if show_individual_traces
        else torch.tensor([], dtype=torch.long)
    )

    def _group_figsize(n_rows: int):
        if figsize is not None:
            return figsize
        return (14, 3.5 * max(n_rows, 1))

    def _plot_group(group_task_ids):
        if not group_task_ids:
            return None, None
        n_rows_local = len(group_task_ids)
        fig_local, axes_local = plt.subplots(
            n_rows_local, k_major,
            figsize=_group_figsize(n_rows_local),
            sharey=True, squeeze=False,
        )
        for row, tid in enumerate(group_task_ids):
            lam_all = results_by_task[tid]["lam"]
            post_all = results_by_task[tid]["post"]
            lam_mean = lam_all.mean(axis=0)
            lam_std = lam_all.std(axis=0)
            post_mean = post_all.mean(axis=0)
            post_std = post_all.std(axis=0)

            for col in range(k_major):
                ax = axes_local[row, col]
                c = corner_rgb[col]
                if show_individual_traces:
                    for idx in indiv_indices:
                        ii = int(idx)
                        ax.plot(ts, lam_all[ii, :, col], color=c, lw=0.5, alpha=0.10)

                ax.plot(ts, lam_mean[:, col], color=c, lw=2.5, alpha=0.95)
                lo = np.clip(lam_mean[:, col] - lam_std[:, col], 0.0, 1.0)
                hi = np.clip(lam_mean[:, col] + lam_std[:, col], 0.0, 1.0)
                ax.fill_between(ts, lo, hi, color=c, alpha=0.18, zorder=1)

                post_color = "#111111"
                post_band_color = "#9E9E9E"
                ax.plot(ts, post_mean[:, col], color=post_color,
                        lw=2.2, ls="--", alpha=0.95)
                p_lo = np.clip(post_mean[:, col] - post_std[:, col], 0.0, 1.0)
                p_hi = np.clip(post_mean[:, col] + post_std[:, col], 0.0, 1.0)
                ax.fill_between(
                    ts, p_lo, p_hi,
                    facecolor="none", edgecolor=post_band_color,
                    hatch="////", linewidth=0.0, alpha=0.9, zorder=2,
                )

                ax.set_ylim(-0.05, 1.05)
                ax.grid(axis="y", alpha=0.3)
                ax.tick_params(labelsize=14)
                if row == n_rows_local - 1:
                    ax.set_xlabel("Position $t$", fontsize=16)
                if col == 0:
                    label = f"Task {tid}" if tid < k_major else f"OOD {tid}"
                    ax.set_ylabel(label, fontsize=16)
                if row == 0:
                    handles = [
                        Line2D([0], [0], color=c, lw=2.5, ls="-",
                               label=rf"$\lambda_{{{col+1}}}$ ({mode_str})"),
                        Patch(facecolor=c, alpha=0.18,
                              label=rf"$\lambda_{{{col+1}}}$ std"),
                        Line2D([0], [0], color=post_color, lw=2.2, ls="--",
                               label=rf"$P(Z\!={col+1})$ mean"),
                        Patch(facecolor="white", edgecolor=post_band_color,
                              hatch="////",
                              label=rf"$P(Z\!={col+1})$ std"),
                    ]
                    ax.legend(handles=handles, fontsize=12, loc="best",
                              framealpha=0.75)
                if annotate_agreement:
                    m_corr = agreement_metrics[tid]["mean_corr"][col]
                    s_corr = agreement_metrics[tid]["std_corr"][col]
                    ax.text(
                        0.03, 0.97,
                        f"r_mean={m_corr:.2f}\nr_std={s_corr:.2f}",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", alpha=0.5,
                                  edgecolor="none"),
                    )
        fig_local.suptitle("", fontsize=18, y=1.01)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig_local)
        return fig_local, axes_local

    fig_major, axes_major = _plot_group(major_task_ids)
    fig_ood, axes_ood = _plot_group(ood_task_ids)
    fig = fig_major if fig_major is not None else fig_ood
    axes = axes_major if axes_major is not None else axes_ood

    del hiddens
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "fig": fig,
        "axes": axes,
        "results_by_task": results_by_task,
        "cell_means": cell_means.cpu(),
        "task_ids": task_ids,
        "agreement_metrics": agreement_metrics,
        "fig_major": fig_major,
        "axes_major": axes_major,
        "fig_ood": fig_ood,
        "axes_ood": axes_ood,
    }


# ---------------------------------------------------------------------------
#  traj_averaging_projection_plot_coin  (averaging-based task vectors, coin)
# ---------------------------------------------------------------------------

def traj_averaging_projection_plot_coin(
    exp_name: str,
    layer_index: Optional[int] = None,
    task_ids: Optional[List[int]] = None,
    B: int = 64,
    step: Optional[int] = None,
    estimation_positions: Optional[List[int]] = None,
    plot_positions: Optional[List[int]] = None,
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
    the Bayesian posterior alpha(t), for the coin task.

    Task vectors are estimated by averaging hidden states of major tasks
    at late positions (per-position demeaned).  For each position *t*,
    hidden states are centred and projected onto the task subspace to
    obtain coefficients beta_{t,k} under the sum-to-one constraint.

    The figure shows beta (points with error bars) overlaid with the
    Bayesian posterior alpha (dashed lines with std bands) for each major
    task, all on a single axis per evaluation task.

    Parameters
    ----------
    exp_name : str
    layer_index : int, optional — defaults to last layer
    task_ids : list of int, optional
        Which generating tasks to show (default ``[0, 1, 2]``).
    B : int — batch size for evaluation
    step : int, optional — checkpoint step
    estimation_positions : list of int, optional
        Positions used to estimate task vectors (default: last 30).
    plot_positions : list of int, optional
        Positions to include in the plot.  ``None`` → all positions.
    per_position_mean : bool
        If True, subtract per-position grand mean; if False, subtract
        the global grand mean from the estimation step.
    n_ood : int, optional — number of OOD tasks in eval pool
    figsize : tuple, optional
    show : bool
    corner_colors : tuple of colour strings (one per major task)

    Returns
    -------
    dict with 'fig', 'axes', 'results_by_task', 'task_vecs', 'grand_mean'.
    """
    import gc
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from icl.utils.unified_interface import _get_hiddens_at_real_positions
    from icl.utils.unified_plot import posterior_over_models_over_time_per_sample
    from icl.utils.separability import estimate_task_vectors_by_averaging

    if task_ids is None:
        task_ids = [0, 1, 2]

    _, sampler_orig, config = nu.load_everything("coin", exp_name)
    n_layers = config.model.num_layers
    seq_len = sampler_orig.seq_len - 1
    k_major = sampler_orig.n_major_tasks

    if layer_index is None:
        layer_index = n_layers - 1
    if estimation_positions is None:
        estimation_positions = list(range(max(0, seq_len - 30), seq_len))

    max_tid = max(task_ids)
    if n_ood is None:
        n_ood = max(1, max_tid - k_major + 1) if max_tid >= k_major else 0

    hiddens, _k_minor, demo_data, sampler_clone = _get_hiddens_at_real_positions(
        task_name="coin", exp_name=exp_name,
        n_minor=0, n_ood=n_ood, B=B, step=step,
        return_data=True, post_layernorm=post_layernorm,
        extraction_point=extraction_point,
    )

    hiddens_layer = hiddens[layer_index].float()  # (K, T, B, D)
    K, T, B_actual, D = hiddens_layer.shape

    # ---- estimate task vectors from major tasks by averaging ----
    hiddens_major = hiddens_layer[:k_major]  # (K_major, T, B, D)
    task_vecs, grand_mean = estimate_task_vectors_by_averaging(
        hiddens_major, estimation_positions,
    )

    # ---- build decode matrix: drop last row, enforce sum(beta)=1 ----
    V = task_vecs.float()
    V_basis = V[:-1]  # (K_major-1, D)
    decode_gamma = torch.linalg.solve(
        V_basis @ V_basis.T, V_basis,
    )  # (K_major-1, D)

    # ---- compute beta(t) for each task ----
    results_by_task = {}
    for tid in task_ids:
        if tid >= K:
            raise ValueError(
                f"task_id={tid} out of range ({K} tasks available)"
            )
        h_tid = hiddens_layer[tid]  # (T, B, D)

        beta_all = np.empty((B_actual, T, k_major), dtype=np.float32)
        for t in range(T):
            h_t = h_tid[t].float()  # (B, D)
            if per_position_mean:
                mu_t = hiddens_major[:, t, :, :].reshape(-1, D).float().mean(dim=0)
            else:
                mu_t = grand_mean.float()
            h_centered = (h_t - mu_t.unsqueeze(0).to(h_t.device)).to(decode_gamma.device)  # (B, D)
            gamma = decode_gamma @ h_centered.T  # (K_major-1, B)
            beta_K = (1.0 - gamma.sum(dim=0)) / k_major
            beta_t = torch.empty(k_major, B_actual, device=V.device)
            beta_t[:k_major - 1] = gamma + beta_K.unsqueeze(0)
            beta_t[k_major - 1] = beta_K
            beta_all[:, t, :] = beta_t.T.cpu().numpy()

        # Bayesian posterior
        post_tensor = posterior_over_models_over_time_per_sample(
            demo_data[tid:tid + 1], sampler_clone.major_p,
        ).float()  # (1, B, seq_len, K_major)
        post_all = post_tensor[0, :, :T, :k_major].cpu().numpy()

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
    if show:
        plt.show()
    else:
        plt.close(fig)

    del hiddens
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "fig": fig, "axes": axes_arr,
        "results_by_task": results_by_task,
        "task_vecs": task_vecs, "grand_mean": grand_mean,
        "task_ids": task_ids,
    }
