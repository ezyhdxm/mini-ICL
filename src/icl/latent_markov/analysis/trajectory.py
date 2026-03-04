"""
Trajectory projection and weight-row orthogonality analysis for latent Markov
(non-padded sequences).

Extracted from ``icl.utils.latent_nonpadded``.
"""

import numpy as np
import torch
from typing import Optional

import icl.utils.notebook_utils as nu
from icl.utils.unified_interface import _get_hiddens_at_real_positions
from icl.latent_markov.analysis.bayes import task_posterior_over_time
from icl.latent_markov.analysis.probes import train_linear_hidden_predictor
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
#  traj_posterior_projection_plot
# ---------------------------------------------------------------------------

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
    """
    2-D trajectory projection for latent Markov (non-padded) using
    **posterior-derived** task vectors.

    Instead of defining the projection plane from raw batch-averaged hiddens,
    this function:

      1. Fits a linear model ``posterior @ W + b ≈ hiddens`` in major mode
         at late positions via ``train_linear_hidden_predictor``.
      2. Uses the centered rows of ``W`` as the 3 final task vectors.
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
        Number of samples for the linear fit.
    fit_positions : list, optional
        Positions used for the linear fit. ``None`` → ``range(100, seq_len)``.
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
    from icl.linear.linear_utils import estimate_lambda_with_r2
    from icl.utils.traj_plot import project_with_r2_trajectories_group_colors_mpl

    _, _, config, *_ = nu.load_everything("latent", exp_name)
    n_layers = config.model.num_layers
    seq_len = config.seq_len
    if layer_index is None:
        layer_index = n_layers - 1

    if fit_positions is None:
        fit_positions = list(range(100, seq_len))

    fit_res = train_linear_hidden_predictor(
        exp_name=exp_name,
        layer=layer_index,
        n_samples=fit_n_samples,
        positions=fit_positions,
        sample_mode="major",
        n_minor=0,
        step=step,
        print_summary=False,
        skip_baselines=True,
    )
    W = fit_res["model_weight"]          # (K_major, D)
    b = fit_res["model_bias"]            # (D,)
    W_mean = W.mean(dim=0, keepdim=True) # (1, D)

    # W̃ = W − w̄  (centred task vectors)
    final_task_vecs = (W - W_mean).float()  # (K_major, D)

    hiddens, k_minor = _get_hiddens_at_real_positions(
        task_name="latent", exp_name=exp_name,
        n_minor=n_minor, n_ood=n_ood, B=B, step=step,
    )

    hiddens_layer = hiddens[layer_index].to(torch.float32)  # (K, T, B, D)
    K, T, B_actual, D = hiddens_layer.shape
    k_major = 3

    # From the probe:  h ≈ π W + b + ε
    # Since Σπ_k = 1:  h − b − w̄ ≈ π W̃ + ε
    # So we subtract b and w̄ from the hidden trajectories, then
    # project onto W̃ to recover barycentric coordinates λ ≈ π.
    b_vec = b.float()

    if use_mean:
        tvecs = hiddens_layer.mean(dim=-2) - b_vec  # (K, T, D)
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
        tvecs = torch.cat([out_major, out_mid, out_minor], dim=0) - b_vec

    task_vecs_over_all_time = tvecs - W_mean.float()  # (K, T, D)

    lambdas, r2_scores, _, _ = estimate_lambda_with_r2(
        final_task_vecs,
        task_vecs_over_all_time,
        is_zero_mean=True,
    )

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


# ---------------------------------------------------------------------------
#  plot_weight_row_cosine_heatmap
# ---------------------------------------------------------------------------

def plot_weight_row_cosine_heatmap(
    exp_name: str,
    layer_index: Optional[int] = None,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    sample_mode: str = "major",
    n_minor: int = 0,
    center_rows: bool = True,
    figsize: tuple = (6, 5),
    cmap: str = "coolwarm",
    vmin: float = -1.0,
    vmax: float = 1.0,
    annotate: bool = True,
    show_axes: bool = True,
    show: bool = True,
    title: Optional[str] = None,
) -> dict:
    """
    Plot cosine-similarity heatmap between rows of reverse-fit weight matrix W.

    This follows the same reverse-fit step used by
    ``traj_posterior_projection_plot``:
    fit ``posterior -> hiddens`` and treat rows of ``W`` as task vectors.

    Parameters
    ----------
    exp_name : str
    layer_index : int, optional
        ``None`` -> last layer.
    step : int, optional
    fit_n_samples : int
    fit_positions : list, optional
        ``None`` -> ``range(100, seq_len)``.
    sample_mode : str
        Forwarded to ``train_linear_hidden_predictor``.
        Use ``"major"`` with ``n_minor=0`` to match the 3-anchor setup.
    n_minor : int
    center_rows : bool
        If True, use ``W - mean(W, dim=0)`` before cosine similarity
        (matches task-vector construction in traj projection).
    figsize, cmap, vmin, vmax, annotate, show, title

    Returns
    -------
    dict
        ``{'fig', 'ax', 'cosine_matrix', 'W', 'W_used', 'fit_results'}``.
    """
    import matplotlib.pyplot as plt

    _, _, config, *_ = nu.load_everything("latent", exp_name)
    n_layers = config.model.num_layers
    seq_len = config.seq_len
    if layer_index is None:
        layer_index = n_layers - 1

    if fit_positions is None:
        fit_positions = list(range(100, seq_len))

    fit_res = train_linear_hidden_predictor(
        exp_name=exp_name,
        layer=layer_index,
        n_samples=fit_n_samples,
        positions=fit_positions,
        sample_mode=sample_mode,
        n_minor=n_minor,
        step=step,
        print_summary=False,
        skip_baselines=True,
    )
    W = fit_res["model_weight"].float().cpu()  # (K_fit, D)
    W_used = W - W.mean(dim=0, keepdim=True) if center_rows else W

    norms = W_used.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Wn = W_used / norms
    cosine_matrix = (Wn @ Wn.T).cpu().numpy()

    K_fit = cosine_matrix.shape[0]
    labels = [f"row {i}" for i in range(K_fit)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cosine_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cosine similarity", fontsize=12)

    if show_axes:
        ax.set_xticks(np.arange(K_fit))
        ax.set_yticks(np.arange(K_fit))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Weight row index", fontsize=12)
        ax.set_ylabel("Weight row index", fontsize=12)
        ax.tick_params(labelsize=11)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    if annotate:
        for i in range(K_fit):
            for j in range(K_fit):
                val = cosine_matrix[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10)

    ax.set_title("", fontsize=18)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "fig": fig,
        "ax": ax,
        "cosine_matrix": cosine_matrix,
        "W": W,
        "W_used": W_used,
        "fit_results": fit_res,
    }


# ---------------------------------------------------------------------------
#  compute_weight_row_orthogonality_metrics
# ---------------------------------------------------------------------------

def compute_weight_row_orthogonality_metrics(
    exp_name: str,
    layer_index: Optional[int] = None,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    sample_mode: str = "major",
    n_minor: int = 0,
    center_rows: bool = False,
    print_summary: bool = True,
) -> dict:
    """
    Compute row-orthogonality diagnostics for reverse-fit weight matrix ``W``.

    Fits ``posterior -> hiddens`` (same as trajectory posterior projection),
    then computes metrics on rows of ``W`` (or centered rows).

    Metrics
    -------
    - ``mean_abs_offdiag_cosine``: mean |cos| over off-diagonal pairs
    - ``max_abs_offdiag_cosine``: max |cos| (mutual coherence)
    - ``fro_norm_G_minus_I``: ||G - I||_F, where G = row-normalized Gram
    - ``fro_norm_G_minus_I_per_row``: ||G - I||_F / K
    - ``condition_number``: s_max / s_min of row matrix
    - ``effective_rank``: exp(entropy of normalized singular values)
    """
    _, _, config, *_ = nu.load_everything("latent", exp_name)
    n_layers = config.model.num_layers
    seq_len = config.seq_len
    if layer_index is None:
        layer_index = n_layers - 1

    if fit_positions is None:
        fit_positions = list(range(100, seq_len))

    fit_res = train_linear_hidden_predictor(
        exp_name=exp_name,
        layer=layer_index,
        n_samples=fit_n_samples,
        positions=fit_positions,
        sample_mode=sample_mode,
        n_minor=n_minor,
        step=step,
        print_summary=False,
        skip_baselines=True,
    )
    W = fit_res["model_weight"].float().cpu()  # (K, D)
    W_used = W - W.mean(dim=0, keepdim=True) if center_rows else W

    K = int(W_used.shape[0])
    row_norms = W_used.norm(dim=1, keepdim=True).clamp_min(1e-12)
    U = W_used / row_norms
    G = U @ U.T  # cosine Gram matrix
    I = torch.eye(K, dtype=G.dtype)

    off_mask = ~torch.eye(K, dtype=torch.bool)
    off_vals = G[off_mask]
    mean_abs_offdiag = off_vals.abs().mean().item() if off_vals.numel() > 0 else 0.0
    max_abs_offdiag = off_vals.abs().max().item() if off_vals.numel() > 0 else 0.0

    diff = G - I
    fro_norm = torch.linalg.norm(diff, ord="fro").item()
    fro_norm_per_row = fro_norm / max(K, 1)

    svals = torch.linalg.svdvals(W_used)
    smax = svals.max().item() if svals.numel() > 0 else float("nan")
    smin = svals.min().item() if svals.numel() > 0 else float("nan")
    condition_number = (smax / smin) if (svals.numel() > 0 and smin > 0) else float("inf")

    if svals.numel() > 0:
        p = (svals / svals.sum().clamp_min(1e-12)).clamp_min(1e-12)
        eff_rank = torch.exp(-(p * p.log()).sum()).item()
    else:
        eff_rank = float("nan")

    metrics = {
        "mean_abs_offdiag_cosine": mean_abs_offdiag,
        "max_abs_offdiag_cosine": max_abs_offdiag,
        "fro_norm_G_minus_I": fro_norm,
        "fro_norm_G_minus_I_per_row": fro_norm_per_row,
        "condition_number": condition_number,
        "effective_rank": eff_rank,
        "n_rows": K,
        "center_rows": center_rows,
        "layer_index": layer_index,
        "sample_mode": sample_mode,
    }

    if print_summary:
        print("=== W-row Orthogonality Metrics (latent non-padded) ===")
        print(f"Layer: {layer_index}, n_rows: {K}, centered: {center_rows}")
        print(f"mean |offdiag cosine| : {mean_abs_offdiag:.6f}")
        print(f"max  |offdiag cosine| : {max_abs_offdiag:.6f}")
        print(f"||G - I||_F           : {fro_norm:.6f}")
        print(f"||G - I||_F / K       : {fro_norm_per_row:.6f}")
        print(f"condition number      : {condition_number:.6f}")
        print(f"effective rank        : {eff_rank:.6f}")

    return {
        "metrics": metrics,
        "cosine_matrix": G.cpu().numpy(),
        "W": W,
        "W_used": W_used,
        "singular_values": svals.cpu().numpy(),
        "fit_results": fit_res,
    }


# ---------------------------------------------------------------------------
#  traj_post_posterior_projection_plot
# ---------------------------------------------------------------------------

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
    from icl.linear.linear_utils import estimate_lambda_with_r2

    # Backward compat: old task_id / n_rows API
    if task_ids is None:
        if task_id is not None:
            task_ids = [task_id]
        else:
            task_ids = [0, 1, 2]

    _, _, config, *_ = nu.load_everything("latent", exp_name)
    n_layers = config.model.num_layers
    seq_len = config.seq_len
    if layer_index is None:
        layer_index = n_layers - 1

    if fit_positions is None:
        fit_positions = list(range(100, seq_len))

    fit_res = train_linear_hidden_predictor(
        exp_name=exp_name,
        layer=layer_index,
        n_samples=fit_n_samples,
        positions=fit_positions,
        sample_mode="major",
        n_minor=0,
        step=step,
        print_summary=False,
        skip_baselines=True,
    )
    W = fit_res["model_weight"]        # (K_major, D)
    b = fit_res["model_bias"]          # (D,)
    W_mean = W.mean(dim=0, keepdim=True)  # (1, D)

    # From the probe:  h ≈ π W + b + ε
    # Centred form:    h − b − w̄ ≈ π (W − w̄) + ε  =  π W̃ + ε
    # So we subtract b and w̄ from h, then project onto W̃.
    b_vec = b.float()              # (D,)
    W_mean_f = W_mean.float()      # (1, D)
    W_c = (W - W_mean).float()     # (K_major, D)  — centred task vectors W̃
    W_f = W.float()                # (K_major, D)  — uncentred

    max_tid = max(task_ids)
    if n_ood is None:
        n_ood = max(1, max_tid - 2)
    hiddens, _k_minor, demo_data, sampler_clone = _get_hiddens_at_real_positions(
        task_name="latent", exp_name=exp_name,
        n_minor=0, n_ood=n_ood, B=B, step=step,
        return_data=True,
    )

    hiddens_layer = hiddens[layer_index].to(torch.float32)  # (K, T, B, D)
    K, T, B_actual, D = hiddens_layer.shape
    k_major = 3

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
                # (h − b − w̄) ≈ π W̃ + ε  →  project onto W̃
                lam_raw, *_ = estimate_lambda_with_r2(
                    W_c, h - b_vec - W_mean_f, is_zero_mean=True,
                )
            else:
                # (h − b) ≈ π W + ε  →  project onto W (unconstrained)
                lam_raw, *_ = estimate_lambda_with_r2(
                    W_f, h - b_vec, is_zero_mean=False,
                )
            lam_np = np.asarray(lam_raw)[0]  # (T, K_major)
            lam_list.append(
                _project_onto_simplex(lam_np) if center_lambda
                else np.clip(lam_np, 0.0, 1.0)
            )
        lam_all = np.stack(lam_list, axis=0)  # (B, T, K_major)

        post_tensor = task_posterior_over_time(
            sampler_clone,
            demo_data[tid],
            include_minor=False,
        ).float()  # (B, seq_len, K_major)
        post_all = post_tensor[:, :T, :k_major].cpu().numpy()  # (B, T, K_major)

        results_by_task[tid] = {"lam": lam_all, "post": post_all}

    # ---- plot ----
    corner_rgb = [mcolors.to_rgb(c) for c in corner_colors]
    ts = np.arange(T)
    n_plot_rows = len(task_ids)
    if figsize is None:
        figsize = (14, 3.5 * n_plot_rows)
    fig, axes = plt.subplots(
        n_plot_rows, 3, figsize=figsize, sharey=True, squeeze=False,
    )

    mode_str = "simplex" if center_lambda else "clipped"
    for row, tid in enumerate(task_ids):
        lam_all = results_by_task[tid]["lam"]    # (B, T, 3)
        post_all = results_by_task[tid]["post"]  # (B, T, 3)
        lam_mean = lam_all.mean(axis=0)          # (T, 3)
        lam_std = lam_all.std(axis=0)            # (T, 3)
        post_mean = post_all.mean(axis=0)        # (T, 3)
        post_std = post_all.std(axis=0)          # (T, 3)

        for col in range(3):
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
            task_label = f"Task {tid}" if tid < 3 else f"OOD {tid}"
            if col == 0:
                ax.set_ylabel(task_label, fontsize=16)
            if row == 0:
                ax.set_title("", fontsize=18)
            if row == 0 and col == 0:
                ax.legend(fontsize=10, loc="best", framealpha=0.7)

    fig.suptitle("", fontsize=18, y=1.01)
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
