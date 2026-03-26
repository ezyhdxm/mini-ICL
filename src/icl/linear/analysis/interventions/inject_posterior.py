"""Posterior-weighted task-vector injection for the Linear Regression task.

For OOD samples, replaces the task-subspace component with the
posterior-weighted combination of task vectors and compares the model
output to the misspecified Bayesian prediction.

Also provides a *direct* variant that skips OOD generation entirely:
sample α ~ Dirichlet, inject α @ W_task, and compare to α-weighted
prediction ŷ = Σ_k α_k (x^T w_k).
"""

import gc
from typing import Optional

import numpy as np
import torch

import icl.utils.notebook_utils as nu
from icl.linear.analysis.probes import train_linear_hidden_predictor
from icl.linear.analysis.interventions._helpers import (
    _load_and_prepare_model,
    _create_ood_task,
    _cleanup_model,
)
from icl.linear.analysis.posterior import task_posterior_over_time_linear_regression
from icl.linear.analysis._helpers import _show_or_close
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
#  Direct injection (no OOD, no posterior computation)
# ──────────────────────────────────────────────────────────────────────

def intervene_direct_injection(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples: int = 2000,
    eval_positions: Optional[list] = None,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    center_task_vecs: bool = True,
    include_position_bias: bool = False,
    include_logit: bool = False,
    dirichlet_alpha: float = 1.0,
    verbose: bool = False,
    print_summary: bool = True,
) -> dict:
    """Direct α-injection test for the linear-interpolation hypothesis.

    Instead of generating OOD sequences and computing posteriors, this
    function:

    1. Samples input sequences from major tasks.
    2. Samples α ~ Dirichlet(dirichlet_alpha · 1_K) independently per
       sample (constant across all point positions).
    3. Injects  h'[2p] = h[2p] − h[2p] P_task + (α W_task)  at each
       prediction position.
    4. Compares model output to  ŷ = Σ_k α_k (x_p^T w_k).

    Returns a dict compatible with ``plot_inject_posterior_per_position``.
    """
    model, train_task, config, device = _load_and_prepare_model(
        exp_name, step=step,
    )
    if step is None:
        step = config.training.total_steps

    n_points = int(config.task.n_points)
    n_dims = int(config.task.n_dims)
    K_major = int(train_task.n_tasks)
    task_pool = train_task.task_pool.to(device)    # (K, D, 1)
    W_major = task_pool[:K_major].squeeze(-1)       # (K, D)

    # ── Fit task subspace ────────────────────────────────────────────
    if fit_positions is None:
        fit_positions = list(range(min(10, n_points), n_points))

    fit_res = train_linear_hidden_predictor(
        exp_name=exp_name, layer=layer, n_samples=fit_n_samples,
        positions=fit_positions, sample_mode="major", step=step,
        n_minor=None, print_summary=False, skip_baselines=True,
        include_position_bias=include_position_bias,
        include_logit=include_logit,
    )
    W_fit = fit_res["model_weight"].float()          # (K, D_model)

    if center_task_vecs:
        tv = W_fit - W_fit.mean(dim=0, keepdim=True)
    else:
        tv = W_fit.clone()
    _, S_tv, Vt_tv = torch.linalg.svd(tv, full_matrices=False)
    rank = int((S_tv > 1e-6 * S_tv[0]).sum().item())
    P_task = (Vt_tv[:rank].T @ Vt_tv[:rank]).to(device)

    ref_vecs = (W_fit - W_fit.mean(dim=0) if center_task_vecs
                else W_fit.clone()).to(device)        # (K, D_model)

    if verbose:
        logger.info(
            f"[direct inj linear] rank={rank}, "
            f"R\u00b2={fit_res['val_r2']:.4f}"
        )

    # ── Evaluate ─────────────────────────────────────────────────────
    if eval_positions is None:
        eval_positions = list(range(min(10, n_points), n_points))

    inject_seq_pos = torch.arange(n_points, device=device) * 2

    dirichlet = torch.distributions.Dirichlet(
        torch.ones(K_major, device=device) * dirichlet_alpha,
    )

    acc_base = {p: [] for p in eval_positions}
    acc_inj = {p: [] for p in eval_positions}
    acc_mode = {p: [] for p in eval_positions}
    acc_post_std = {p: [] for p in eval_positions}
    acc_min_dist = {p: [] for p in eval_positions}

    eye_K = torch.eye(K_major, device=device)

    orig_bs = int(train_task.batch_size)
    train_task.batch_size = B
    n_done = 0
    bi = 0

    while n_done < n_samples:
        demo_data, _, demo_target = train_task.sample_batch(
            step=bi + 99999, is_eval=False,
        )
        demo_data = demo_data.to(device)
        demo_target = demo_target.to(device)
        cur_B = demo_data.shape[0]
        bi += 1

        alpha = dirichlet.sample((cur_B,))              # (B, K)

        pred_per_task = torch.einsum(
            "bpd,kd->bpk", demo_data, W_major,
        )                                                # (B, P, K)
        y_mis = (alpha.unsqueeze(1) * pred_per_task).sum(-1)  # (B, P)
        mode_idx = alpha.argmax(dim=-1)                        # (B,)
        y_mode = pred_per_task[
            torch.arange(cur_B, device=device), :, mode_idx
        ]                                                      # (B, P)

        with torch.no_grad():
            preds_base = model(demo_data, demo_target)

        def _hook(mod, inp, out,
                  _a=alpha, _W=ref_vecs, _P=P_task, _sp=inject_seq_pos):
            h = out if torch.is_tensor(out) else out[0]
            h_new = h.clone()
            B_h = h.shape[0]
            h_at = h[:, _sp, :]
            tc = (_a[:B_h] @ _W).unsqueeze(1)            # (B, 1, D)
            h_new[:, _sp, :] = h_at - (h_at @ _P) + tc
            return h_new if torch.is_tensor(out) else (h_new,) + out[1:]

        handle = model.transformer.blocks[layer].attn_block.register_forward_hook(
            _hook,
        )
        try:
            with torch.no_grad():
                preds_inj = model(demo_data, demo_target)
        finally:
            handle.remove()

        for p in eval_positions:
            if p >= preds_base.shape[1]:
                continue
            y_m = y_mis[:, p]
            mse_b = ((preds_base[:, p] - y_m) ** 2).mean().item()
            mse_i = ((preds_inj[:, p] - y_m) ** 2).mean().item()
            mse_md = ((y_mode[:, p] - y_m) ** 2).mean().item()
            acc_base[p].append(mse_b)
            acc_inj[p].append(mse_i)
            acc_mode[p].append(mse_md)
            acc_post_std[p].append(alpha.std(dim=-1).mean().item())
            dists = (alpha.unsqueeze(1) - eye_K.unsqueeze(0)).abs().sum(-1)
            acc_min_dist[p].append(dists.min(dim=-1).values.mean().item())

        n_done += cur_B
        del demo_data, demo_target, preds_base, preds_inj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    train_task.batch_size = orig_bs

    # ── Aggregate ────────────────────────────────────────────────────
    positions, mse_base, mse_inj, mse_mode_list = [], [], [], []
    post_std, min_dist = [], []
    for p in eval_positions:
        if acc_base[p]:
            positions.append(p)
            mse_base.append(float(np.mean(acc_base[p])))
            mse_inj.append(float(np.mean(acc_inj[p])))
            mse_mode_list.append(float(np.mean(acc_mode[p])))
            post_std.append(float(np.mean(acc_post_std[p])))
            min_dist.append(float(np.mean(acc_min_dist[p])))

    _cleanup_model(model)

    if print_summary:
        print(f"\n{'=' * 70}")
        print(f"Direct Injection  (linear, layer {layer})")
        print(f"{'=' * 70}")
        print(f"  rank={rank}  K={K_major}  "
              f"dirichlet_alpha={dirichlet_alpha}")
        print(f"  Mean MSE(baseline, y_\u03b1) = {np.mean(mse_base):.4f}")
        print(f"  Mean MSE(injected, y_\u03b1) = {np.mean(mse_inj):.4f}")
        print(f"  Mean MSE(mode, y_\u03b1)     = {np.mean(mse_mode_list):.4f}")
        print()

    return dict(
        layer=layer, rank=rank, K_major=K_major,
        positions=positions,
        mse_baseline=np.array(mse_base),
        mse_injected=np.array(mse_inj),
        mse_mode=np.array(mse_mode_list),
        posterior_std=np.array(post_std),
        min_dist_basis=np.array(min_dist),
    )


# ──────────────────────────────────────────────────────────────────────
#  Direct injection using averaging-based task vectors
# ──────────────────────────────────────────────────────────────────────


def intervene_averaging_injection(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples: int = 2000,
    eval_positions: Optional[list] = None,
    step: Optional[int] = None,
    estimation_positions: Optional[list] = None,
    estimation_B: int = 128,
    per_position_mean: bool = True,
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
    dirichlet_alpha: float = 1.0,
    figsize: tuple = (10, 3.5),
    show: bool = True,
    verbose: bool = False,
    print_summary: bool = True,
    color_vmin: Optional[float] = None,
    color_vmax: Optional[float] = None,
    use_per_position_vecs: bool = False,
) -> dict:
    """Direct α-injection using averaging-based task vectors.

    Like ``intervene_direct_injection`` but estimates task vectors by
    averaging hidden states at late positions (``estimation_positions``)
    rather than fitting an OLS probe.

    Steps:
      1. Extract hidden states for all major tasks at *layer*.
      2. Estimate task vectors via ``estimate_task_vectors_by_averaging``.
      3. Build orthogonal projector P_task from the centered task vectors.
      4. For each eval position p, run a *separate* forward pass that
         injects only at position p:  h'[p] = h[p] − (h[p]−μ_p)P + α·θ.
         Compare output at p to ŷ_p = Σ_k α_k (x_p^T w_k).
      5. Plot RMSE (baseline vs injected vs mode) directly.

    Parameters
    ----------
    per_position_mean : bool
        If True, subtract per-position grand mean when computing the
        task-subspace component to remove (handles RoPE rotations).
        If False, use the global grand mean from the estimation step.
    estimation_B : int
        Batch size used for the hidden-state extraction step (task vector
        estimation).  Separate from *B* which is used in the injection
        loop.
    use_per_position_vecs : bool
        If True, estimate task vectors independently at each position and
        perform direct subtraction in the hook:
        ``h'[p] = h[p] − θ_k(p) + α·θ(p)`` where k is the true task.
        Data is generated per-task so the true task label is known.
    """
    import matplotlib.pyplot as plt
    from icl.linear.linear_ood_analysis import (
        _create_eval_task_pool, _setup_eval_task, setup_device,
    )
    from icl.linear.task_vecs import extract_hidden_multi
    from icl.utils.separability import (
        estimate_task_vectors_by_averaging,
        per_position_task_vectors,
    )
    from icl.linear.analysis._helpers import _task_positions
    from icl.linear.analysis.interventions._helpers import (
        _extract_hiddens_for_pool,
    )

    # ── Load model & config ──────────────────────────────────────────
    model, train_task, config, device = _load_and_prepare_model(
        exp_name, step=step,
    )
    if step is None:
        step = config.training.total_steps

    n_points = int(config.task.n_points)
    K_major = int(train_task.n_tasks)
    D = int(config.model.n_embd)
    task_pool = train_task.task_pool.to(device)       # (K, D_x, 1)
    W_major = task_pool[:K_major].squeeze(-1)          # (K, D_x)

    if estimation_positions is None:
        estimation_positions = list(range(max(0, n_points - 10), n_points))
    if eval_positions is None:
        eval_positions = list(range(min(10, n_points), n_points))

    # ── Extract hidden states for major tasks ────────────────────────
    major_pool = train_task.task_pool.squeeze(-1).to(device)  # (K, D_x)
    eval_pool = major_pool[:K_major]
    eval_task = _setup_eval_task(config, eval_pool, estimation_B, device)
    eval_task.batch_size = estimation_B

    pad_mode = getattr(model, "pad", "mapsto")
    task_pos = _task_positions(pad_mode, n_points, device)

    demo_data_est = eval_task.sample_data(step=step).to(device)
    hiddens_major, _ = _extract_hiddens_for_pool(
        model, eval_task, demo_data_est,
        step=step, layer=layer, task_pos=task_pos, D=D,
        n_tasks=K_major, chunk=8, post_layernorm=post_layernorm,
        extraction_point=extraction_point,
    )
    # hiddens_major: (K, T, B_est, D)  float32 CPU
    hiddens_major = hiddens_major.float()
    T_full = hiddens_major.shape[1]

    # ── Estimate task vectors ─────────────────────────────────────────
    if use_per_position_vecs:
        tv_pos_all, _grand_means = per_position_task_vectors(
            hiddens_major, per_position_mean=per_position_mean,
        )
        tv_pos_all = tv_pos_all.float().to(device)  # (K, T, D)

        if verbose:
            logger.info(
                f"[avg inj linear per-pos] K={K_major}, "
                f"direct subtraction with per-position task vectors"
            )
    else:
        task_vecs, grand_mean = estimate_task_vectors_by_averaging(
            hiddens_major, estimation_positions,
        )

        tv = task_vecs.float()
        _, S_tv, Vt_tv = torch.linalg.svd(tv, full_matrices=False)
        rank = int((S_tv > 1e-6 * S_tv[0]).sum().item())
        P_task = (Vt_tv[:rank].T @ Vt_tv[:rank]).to(device)
        ref_vecs = task_vecs.to(device)

        tv_pos_all = None
        P_task_per_pos = None

        if per_position_mean:
            mu_per_point = hiddens_major.mean(dim=(0, 2)).to(device)
        else:
            mu_global = grand_mean.to(device)

        if verbose:
            logger.info(
                f"[avg inj linear] rank={rank}, K={K_major}, "
                f"estimation_positions={estimation_positions[:3]}..{estimation_positions[-1]}"
            )

    del hiddens_major, demo_data_est
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Injection loop (per-position) ────────────────────────────────
    dirichlet = torch.distributions.Dirichlet(
        torch.ones(K_major, device=device) * dirichlet_alpha,
    )

    all_alphas = []
    all_mse_base = []
    all_mse_inj = []
    all_mse_mode = []

    def _forward_chunked(model_fn, data, target, chunk_size):
        """Run model forward in sub-batches to avoid OOM."""
        outs = []
        for i in range(0, data.shape[0], chunk_size):
            outs.append(model_fn(data[i:i+chunk_size], target[i:i+chunk_size]))
        return torch.cat(outs, dim=0)

    def _make_single_pos_hook(alpha_full, point_idx, task_labels_full=None):
        """Hook that injects at a single position only."""
        seq_pos = int(task_pos[point_idx].item())

        _offset = [0]
        def _hook(mod, inp, out):
            h = out if torch.is_tensor(out) else out[0]
            h_new = h.clone()
            B_h = h.shape[0]
            o = _offset[0]
            a_sub = alpha_full[o:o+B_h]
            _offset[0] += B_h
            h_at = h_new[:B_h, seq_pos, :]
            if use_per_position_vecs:
                _rv = tv_pos_all[:, point_idx, :]
                tl_sub = task_labels_full[o:o+B_h]
                theta_k = _rv[tl_sub]
                tc = a_sub @ _rv
                h_new[:B_h, seq_pos, :] = h_at - theta_k + tc
            else:
                _mu = (mu_per_point[point_idx] if per_position_mean
                       else mu_global)
                _rv = ref_vecs
                h_centered = h_at - _mu.unsqueeze(0)
                tc = a_sub @ _rv
                h_new[:B_h, seq_pos, :] = h_at - (h_centered @ P_task) + tc
            return h_new if torch.is_tensor(out) else (h_new,) + out[1:]
        return _hook

    orig_bs = int(train_task.batch_size)
    fwd_chunk = B
    n_done = 0
    bi = 0

    while n_done < n_samples:
        # ── Generate samples ──────────────────────────────────────────
        if use_per_position_vecs:
            B_per_task = max(1, B // K_major)
            train_task.batch_size = B_per_task
            all_data, all_target, all_tl = [], [], []
            for k in range(K_major):
                w_k = task_pool[k]  # (D_x, 1)
                d_k, t_k = train_task.sample_from_task(
                    w_k, step=bi * K_major + k + 99999,
                )
                all_data.append(d_k)
                all_target.append(t_k)
                all_tl.append(
                    torch.full((d_k.shape[0],), k, dtype=torch.long),
                )
            demo_data = torch.cat(all_data, dim=0).to(device)
            demo_target = torch.cat(all_target, dim=0).to(device)
            task_labels = torch.cat(all_tl).to(device)
        else:
            train_task.batch_size = B
            demo_data, _, demo_target = train_task.sample_batch(
                step=bi + 99999, is_eval=False,
            )
            demo_data = demo_data.to(device)
            demo_target = demo_target.to(device)
            task_labels = None

        cur_B = demo_data.shape[0]
        bi += 1

        alpha = dirichlet.sample((cur_B,))              # (B, K)

        pred_per_task = torch.einsum(
            "bpd,kd->bpk", demo_data, W_major,
        )                                                # (B, P, K)
        y_mis = (alpha.unsqueeze(1) * pred_per_task).sum(-1)  # (B, P)
        mode_idx = alpha.argmax(dim=-1)                        # (B,)
        y_mode = pred_per_task[
            torch.arange(cur_B, device=device), :, mode_idx
        ]                                                      # (B, P)

        # Baseline forward (with OOM retry)
        while True:
            try:
                with torch.no_grad():
                    preds_base = _forward_chunked(
                        model, demo_data, demo_target, fwd_chunk,
                    )
                break
            except torch.cuda.OutOfMemoryError:
                fwd_chunk = max(1, fwd_chunk // 2)
                logger.warning(
                    f"OOM during baseline forward, reducing chunk to {fwd_chunk}"
                )
                torch.cuda.empty_cache()

        # Per-position injected forwards
        se_inj_per_pos = []
        for p in eval_positions:
            if p >= preds_base.shape[1]:
                continue
            while True:
                try:
                    hook_fn = _make_single_pos_hook(alpha, p, task_labels)
                    handle = model.transformer.blocks[layer].attn_block.register_forward_hook(
                        hook_fn,
                    )
                    try:
                        with torch.no_grad():
                            preds_inj_p = _forward_chunked(
                                model, demo_data, demo_target, fwd_chunk,
                            )
                    finally:
                        handle.remove()
                    y_m_p = y_mis[:, p]
                    se_inj_per_pos.append(
                        ((preds_inj_p[:, p] - y_m_p) ** 2).cpu()
                    )
                    del preds_inj_p
                    break
                except torch.cuda.OutOfMemoryError:
                    fwd_chunk = max(1, fwd_chunk // 2)
                    logger.warning(
                        f"OOM during injected forward (pos {p}), "
                        f"reducing chunk to {fwd_chunk}"
                    )
                    torch.cuda.empty_cache()

        if se_inj_per_pos:
            valid_pos = [p for p in eval_positions if p < preds_base.shape[1]]
            valid_pos_t = torch.tensor(valid_pos, device=device)
            y_m = y_mis[:, valid_pos_t]
            se_b = (preds_base[:, valid_pos_t] - y_m) ** 2
            se_md = (y_mode[:, valid_pos_t] - y_m) ** 2
            se_i = torch.stack(se_inj_per_pos, dim=1)         # (B, n_eval)
            all_alphas.append(alpha.cpu())
            all_mse_base.append(se_b.mean(dim=1).cpu())
            all_mse_inj.append(se_i.mean(dim=1))              # already CPU
            all_mse_mode.append(se_md.mean(dim=1).cpu())

        n_done += cur_B
        del demo_data, demo_target, preds_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    train_task.batch_size = orig_bs

    # ── Aggregate per-sample data ────────────────────────────────────
    all_alphas = torch.cat(all_alphas, dim=0).numpy()         # (N, K)
    rmse_base = torch.cat(all_mse_base).sqrt().numpy()        # (N,)
    rmse_inj = torch.cat(all_mse_inj).sqrt().numpy()
    rmse_mode = torch.cat(all_mse_mode).sqrt().numpy()

    # ── Simplex plots ────────────────────────────────────────────────
    from icl.utils.plot_config import plot_simplex_panel_pairs

    fig1, fig2 = plot_simplex_panel_pairs(
        all_alphas, rmse_base, rmse_inj, rmse_mode,
        metric_label="RMSE",
        mean_fmt="mean RMSE = {:.4f}",
        color_vmin=color_vmin, color_vmax=color_vmax,
        figsize=figsize,
    )
    _show_or_close(fig1, show, tight=False)
    _show_or_close(fig2, show, tight=False)

    _cleanup_model(model)

    _rank = rank if not use_per_position_vecs else None
    result = dict(
        layer=layer, rank=_rank, K_major=K_major,
        alphas=all_alphas,
        rmse_baseline=rmse_base,
        rmse_injected=rmse_inj,
        rmse_mode=rmse_mode,
        fig=fig1, fig_mode=fig2,
    )

    if print_summary:
        print(f"\n{'=' * 70}")
        print(f"Averaging Injection  (linear, layer {layer})")
        print(f"{'=' * 70}")
        print(f"  rank={_rank}  K={K_major}  "
              f"dirichlet_alpha={dirichlet_alpha}")
        print(f"  Mean RMSE(baseline, y_α) = {rmse_base.mean():.4f}")
        print(f"  Mean RMSE(injected, y_α) = {rmse_inj.mean():.4f}")
        print(f"  Mean RMSE(mode, y_α)     = {rmse_mode.mean():.4f}")
        print()

    return result


# ──────────────────────────────────────────────────────────────────────
#  OOD-based posterior injection (original approach)
# ──────────────────────────────────────────────────────────────────────


def intervene_inject_posterior(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples_eval: int = 500,
    n_ood: int = 30,
    eval_positions: Optional[list] = None,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    center_task_vecs: bool = True,
    include_position_bias: bool = False,
    include_logit: bool = False,
    filter_convergent: bool = True,
    convergence_threshold: float = 0.75,
    filter_positions: Optional[list] = None,
    verbose: bool = False,
    print_summary: bool = True,
) -> dict:
    """
    Posterior-weighted task-vector injection on OOD data (linear regression).

    For each OOD sample at each point position *p*:

      1. Compute misspecified posterior α_{p,k} = P(Z=k | x_{≤p}, y_{<p})
         using only major task weights.
      2. Replace task component at every prediction position::

             h'[2p] = h[2p] − h[2p] P_task + α_p W_task

      3. Compare intervened model prediction to the misspecified Bayesian
         prediction  ŷ_mis = Σ_k α_{p,k} (x_p^T w_k).

    Returns
    -------
    dict with keys:
        positions    : evaluated point positions
        mse_baseline : MSE(baseline_pred, ŷ_misspecified) per position
        mse_injected : MSE(injected_pred, ŷ_misspecified) per position
    """
    model, train_task, config, device = _load_and_prepare_model(
        exp_name, step=step,
    )
    if step is None:
        step = config.training.total_steps

    n_points = int(config.task.n_points)
    n_dims = int(config.task.n_dims)
    K_major = int(train_task.n_tasks)
    task_pool = train_task.task_pool.to(device)    # (K, D, 1)
    W_major = task_pool[:K_major].squeeze(-1)       # (K, D)

    ood_task = _create_ood_task(train_task, config, B, n_ood, device)

    # ── Fit task subspace ────────────────────────────────────────────
    if fit_positions is None:
        fit_positions = list(range(min(10, n_points), n_points))

    fit_res = train_linear_hidden_predictor(
        exp_name=exp_name, layer=layer, n_samples=fit_n_samples,
        positions=fit_positions, sample_mode="major", step=step,
        n_minor=None, print_summary=False, skip_baselines=True,
        include_position_bias=include_position_bias,
        include_logit=include_logit,
    )
    W_fit = fit_res["model_weight"].float()          # (K, D_model)

    if center_task_vecs:
        tv = W_fit - W_fit.mean(dim=0, keepdim=True)
    else:
        tv = W_fit.clone()
    _, S_tv, Vt_tv = torch.linalg.svd(tv, full_matrices=False)
    rank = int((S_tv > 1e-6 * S_tv[0]).sum().item())
    P_task = (Vt_tv[:rank].T @ Vt_tv[:rank]).to(device)

    ref_vecs = (W_fit - W_fit.mean(dim=0) if center_task_vecs
                else W_fit.clone()).to(device)        # (K, D_model)

    if verbose:
        logger.info(
            f"[posterior inj linear] rank={rank}, "
            f"R\u00b2={fit_res['val_r2']:.4f}"
        )

    # ── Evaluate ─────────────────────────────────────────────────────
    if eval_positions is None:
        eval_positions = list(range(min(10, n_points), n_points))

    inject_seq_pos = torch.arange(n_points, device=device) * 2

    # ── Collect OOD samples (optionally filtering convergent ones) ─
    all_data_list, all_target_list, all_alpha_list = [], [], []
    n_generated = 0
    max_generate = n_samples_eval * 10

    orig_bs = int(ood_task.batch_size)
    ood_task.batch_size = B
    bi = 0

    while (sum(d.shape[0] for d in all_data_list) < n_samples_eval
           and n_generated < max_generate):
        demo_data, _, demo_target = ood_task.sample_batch(
            step=bi + 99999, is_eval=True,
        )
        bi += 1
        demo_data = demo_data.to(device)
        demo_target = demo_target.to(device)
        n_generated += demo_data.shape[0]

        alpha = task_posterior_over_time_linear_regression(
            train_task, demo_data, demo_target, include_minor=False,
        ).to(device)

        if filter_convergent:
            if filter_positions is not None:
                fp = [min(p, alpha.shape[1] - 1) for p in filter_positions]
            else:
                fp = [max(eval_positions[0] - 10, 0),
                      min(eval_positions[-1] + 10, alpha.shape[1] - 1)]
            keep = torch.ones(alpha.shape[0], dtype=torch.bool, device=device)
            for fp_i in fp:
                keep &= alpha[:, fp_i, :].max(dim=-1).values < convergence_threshold
            if keep.any():
                all_data_list.append(demo_data[keep])
                all_target_list.append(demo_target[keep])
                all_alpha_list.append(alpha[keep])
        else:
            all_data_list.append(demo_data)
            all_target_list.append(demo_target)
            all_alpha_list.append(alpha)

        del demo_data, demo_target
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ood_task.batch_size = orig_bs

    if not all_data_list:
        if print_summary:
            print(f"  WARNING: No non-convergent OOD samples found "
                  f"(threshold={convergence_threshold})")
        _cleanup_model(model)
        return dict(
            layer=layer, rank=rank, K_major=K_major,
            positions=[], mse_baseline=np.array([]),
            mse_injected=np.array([]), mse_mode=np.array([]),
            posterior_std=np.array([]), min_dist_basis=np.array([]),
        )

    all_data = torch.cat(all_data_list)[:n_samples_eval]
    all_target = torch.cat(all_target_list)[:n_samples_eval]
    all_alpha = torch.cat(all_alpha_list)[:n_samples_eval]

    if print_summary and filter_convergent:
        kept = all_data.shape[0]
        print(f"  Filtered: kept {kept}/{n_generated} OOD samples "
              f"({kept / n_generated * 100:.0f}%, "
              f"threshold={convergence_threshold})")

    # ── Run intervention in batches ───────────────────────────────
    acc_base = {p: [] for p in eval_positions}
    acc_inj = {p: [] for p in eval_positions}
    acc_mode = {p: [] for p in eval_positions}
    acc_post_std = {p: [] for p in eval_positions}
    acc_min_dist = {p: [] for p in eval_positions}

    eye_K = torch.eye(K_major, device=device)

    for start in range(0, all_data.shape[0], B):
        demo_data = all_data[start:start + B]
        demo_target = all_target[start:start + B]
        alpha = all_alpha[start:start + B]

        pred_per_task = torch.einsum(
            "bpd,kd->bpk", demo_data, W_major,
        )
        y_mis = (alpha * pred_per_task).sum(-1)
        mode_idx = alpha.argmax(dim=-1)                       # (B, T)
        y_mode = pred_per_task.gather(2, mode_idx.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            preds_base = model(demo_data, demo_target)

        def _hook(mod, inp, out,
                  _a=alpha, _W=ref_vecs, _P=P_task, _sp=inject_seq_pos):
            h = out if torch.is_tensor(out) else out[0]
            h_new = h.clone()
            B_h = h.shape[0]
            h_at = h[:, _sp, :]
            tc = _a[:B_h] @ _W
            h_new[:, _sp, :] = h_at - (h_at @ _P) + tc
            return h_new if torch.is_tensor(out) else (h_new,) + out[1:]

        handle = model.transformer.blocks[layer].attn_block.register_forward_hook(
            _hook,
        )
        try:
            with torch.no_grad():
                preds_inj = model(demo_data, demo_target)
        finally:
            handle.remove()

        for p in eval_positions:
            if p >= preds_base.shape[1]:
                continue
            y_m = y_mis[:, p]
            mse_b = ((preds_base[:, p] - y_m) ** 2).mean().item()
            mse_i = ((preds_inj[:, p] - y_m) ** 2).mean().item()
            mse_md = ((y_mode[:, p] - y_m) ** 2).mean().item()
            acc_base[p].append(mse_b)
            acc_inj[p].append(mse_i)
            acc_mode[p].append(mse_md)
            a_p = alpha[:, p, :]
            acc_post_std[p].append(a_p.std(dim=-1).mean().item())
            dists = (a_p.unsqueeze(1) - eye_K.unsqueeze(0)).abs().sum(dim=-1)
            acc_min_dist[p].append(dists.min(dim=-1).values.mean().item())

        del demo_data, demo_target, preds_base, preds_inj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Aggregate ────────────────────────────────────────────────────
    positions, mse_base, mse_inj, mse_mode_list = [], [], [], []
    post_std, min_dist = [], []
    for p in eval_positions:
        if acc_base[p]:
            positions.append(p)
            mse_base.append(float(np.mean(acc_base[p])))
            mse_inj.append(float(np.mean(acc_inj[p])))
            mse_mode_list.append(float(np.mean(acc_mode[p])))
            post_std.append(float(np.mean(acc_post_std[p])))
            min_dist.append(float(np.mean(acc_min_dist[p])))

    _cleanup_model(model)

    if print_summary:
        print(f"\n{'=' * 70}")
        print(f"Posterior-Weighted Injection  (linear, layer {layer})")
        print(f"{'=' * 70}")
        print(f"  rank={rank}  K={K_major}  n_ood={n_ood}")
        print(f"  Mean MSE(baseline, misBayes) = {np.mean(mse_base):.4f}")
        print(f"  Mean MSE(injected, misBayes) = {np.mean(mse_inj):.4f}")
        print(f"  Mean MSE(mode, misBayes)      = {np.mean(mse_mode_list):.4f}")
        print()

    return dict(
        layer=layer, rank=rank, K_major=K_major,
        positions=positions,
        mse_baseline=np.array(mse_base),
        mse_injected=np.array(mse_inj),
        mse_mode=np.array(mse_mode_list),
        posterior_std=np.array(post_std),
        min_dist_basis=np.array(min_dist),
    )


# ──────────────────────────────────────────────────────────────────────
#  Layer sweep + plotting
# ──────────────────────────────────────────────────────────────────────

def plot_inject_posterior_across_layers(
    exp_name: str,
    layers: Optional[list] = None,
    show: bool = True,
    **kwargs,
):
    """Sweep ``intervene_inject_posterior`` over layers and plot.

    Returns ``(fig, all_results)``.
    """
    import matplotlib.pyplot as plt
    from icl.linear.linear_path_utils import load_model_task_config

    if layers is None:
        _, _, config = load_model_task_config(exp_name)
        layers = list(range(config.model.n_layer))

    all_res = {}
    for l in layers:
        logger.info(f"[posterior inj sweep linear] layer {l}")
        all_res[l] = intervene_inject_posterior(
            exp_name=exp_name, layer=l, print_summary=True, **kwargs,
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    last = layers[-1]
    r = all_res[last]
    ax1.plot(r["positions"], r["mse_baseline"] ** 0.5, "o-",
             label="baseline", lw=2, ms=4)
    ax1.plot(r["positions"], r["mse_injected"] ** 0.5, "s-",
             label="posterior-injected", lw=2, ms=4)
    ax1.set_xlabel("Point position", fontsize=14)
    ax1.set_ylabel(r"RMSE(output, $\hat y_{\mathrm{misBayes}}$)", fontsize=14)
    ax1.set_title("", fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)

    x = np.arange(len(layers))
    bw = 0.35
    avg_base = [np.mean(all_res[l]["mse_baseline"]) ** 0.5 for l in layers]
    avg_inj = [np.mean(all_res[l]["mse_injected"]) ** 0.5 for l in layers]
    ax2.bar(x - bw / 2, avg_base, bw,
            label="baseline", color="#F44336", alpha=0.85)
    ax2.bar(x + bw / 2, avg_inj, bw,
            label="posterior-injected", color="#4CAF50", alpha=0.85)
    for i, (vb, vi) in enumerate(zip(avg_base, avg_inj)):
        ax2.text(x[i] - bw / 2, vb, f"{vb:.3f}",
                 ha="center", va="bottom", fontsize=8)
        ax2.text(x[i] + bw / 2, vi, f"{vi:.3f}",
                 ha="center", va="bottom", fontsize=8)
    ax2.set_xlabel("Layer", fontsize=14)
    ax2.set_ylabel(r"Mean RMSE(output, $\hat y_{\mathrm{misBayes}}$)",
                   fontsize=14)
    ax2.set_xticks(x, [str(l) for l in layers])
    ax2.legend(fontsize=12)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _show_or_close(fig, show)

    return fig, all_res


def plot_inject_posterior_per_position(
    result: dict,
    *,
    title: Optional[str] = None,
    figsize: tuple = (6, 4),
    show: bool = True,
):
    """Detailed per-position visualisation of a single-layer result.

    *result* is the dict returned by ``intervene_inject_posterior``
    or ``intervene_direct_injection``.

    Produces two separate figures:

    1. MSE(output, ŷ_misBayes) vs position for baseline, injected, and mode.
    2. Posterior / alpha concentration metrics vs position.

    Returns ``(fig_mse, fig_conc)``.
    """
    import matplotlib.pyplot as plt

    pos = np.array(result["positions"])
    rmse_b = np.array(result["mse_baseline"]) ** 0.5
    rmse_i = np.array(result["mse_injected"]) ** 0.5
    _mse_md_raw = np.array(result.get("mse_mode", []))
    rmse_md = _mse_md_raw ** 0.5 if len(_mse_md_raw) > 0 else _mse_md_raw
    post_std = np.array(result.get("posterior_std", []))
    layer = result["layer"]

    sup = title if title is not None else f"Posterior-weighted injection  (layer {layer})"

    # ── Figure 1: RMSE vs position ───────────────────────────────────
    fig_mse, ax1 = plt.subplots(figsize=figsize)
    ax1.fill_between(pos, rmse_i, rmse_b, alpha=0.15, color="#F44336",
                     label="gap closed by injection")
    ax1.plot(pos, rmse_b, "o-", color="#F44336", lw=2, ms=4,
             label="unmodified")
    ax1.plot(pos, rmse_i, "s-", color="#4CAF50", lw=2, ms=4,
             label=r"$\alpha$-injected")
    if len(rmse_md) == len(pos):
        ax1.plot(pos, rmse_md, "^--", color="#2196F3", lw=1.5, ms=4,
                 label=r"mode task $q_{k^\star}$")
    ax1.set_xlabel("Point position", fontsize=13)
    ax1.set_ylabel(r"RMSE(output, $\sum_k \alpha_k q_k$)", fontsize=13)
    if sup:
        ax1.set_title(sup, fontsize=14)
    ax1.legend(fontsize=10, loc="best")
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    _show_or_close(fig_mse, show)

    # ── Figure 2: posterior concentration ─────────────────────────────
    min_d = np.array(result.get("min_dist_basis", []))
    has_std = len(post_std) == len(pos)
    has_dist = len(min_d) == len(pos)

    fig_conc, ax2 = plt.subplots(figsize=figsize)
    if has_std or has_dist:
        if has_std:
            ax2.plot(pos, post_std, "D-", color="#9C27B0", lw=2, ms=4,
                     label="posterior std")
        if has_dist:
            ax2.plot(pos, min_d, "^-", color="#FF9800", lw=2, ms=4,
                     label=r"min $\|\alpha - e_k\|_1$")
            ax2.axhline(0.0, color="#FF9800", ls="--", lw=1, alpha=0.4,
                        label=r"$\delta$ (concentrated)")
        ax2.set_xlabel("Point position", fontsize=13)
        ax2.set_ylabel("Posterior concentration", fontsize=13)
        ax2.set_title("", fontsize=13)
        ax2.legend(fontsize=9, loc="best")
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "metrics not available\n(re-run intervention)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=11)

    plt.tight_layout()
    _show_or_close(fig_conc, show)

    return fig_mse, fig_conc
