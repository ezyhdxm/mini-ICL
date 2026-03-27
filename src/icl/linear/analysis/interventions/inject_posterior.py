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
