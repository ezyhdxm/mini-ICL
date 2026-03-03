"""Shared infrastructure for causal intervention experiments."""
import gc
from typing import Optional

import numpy as np
import torch

import icl.utils.notebook_utils as nu
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Model loading / cleanup
# ---------------------------------------------------------------------------

def _load_and_prepare_model(exp_name, step=None):
    """Load model, train_task, config; put model in eval mode.

    Returns ``(model, train_task, config, device)``.
    """
    from icl.linear.linear_path_utils import load_model_task_config

    _, train_task, config = load_model_task_config(exp_name)
    if step is None:
        step = config.training.total_steps
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    model.requires_grad_(False)
    return model, train_task, config, config.device


def _create_ood_task(train_task, config, B, n_ood, device):
    """Create an OOD evaluation task with the same noise scale as *train_task*."""
    from icl.linear.linear_ood_analysis import (
        _create_eval_task_pool,
        _setup_eval_task,
    )
    eval_task_pool, _ = _create_eval_task_pool(
        train_task, K=n_ood, include_minor=False,
        device=device, n_minor=0,
    )
    ood_task = _setup_eval_task(config, eval_task_pool, B, device)
    ood_task.noise_scale = float(train_task.noise_scale)
    return ood_task


def _cleanup_model(model):
    """Move model to CPU and free GPU memory."""
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_hiddens_for_pool(
    model, eval_task, demo_data, *, step, layer, task_pos, D,
    n_tasks=None, chunk=8,
):
    """Chunked extraction of h_l(x | task_k) for every task k in the pool.

    For each task k with weight wₖ, generates targets yₜ = wₖᵀxₜ + ε,
    runs the model forward, and collects the hidden state at *layer*.

    Returns
    -------
    hiddens : (K, T, B, D) float32 CPU tensor
    targets : (K, B, T) float32 CPU tensor
    """
    from icl.linear.task_vecs import extract_hidden_multi

    B, n_points = demo_data.shape[:2]
    if n_tasks is None:
        n_tasks = eval_task.task_pool.shape[0]

    hiddens = torch.empty(n_tasks, n_points, B, D, dtype=torch.float32)
    all_targets = torch.empty(n_tasks, B, n_points, dtype=torch.float32)

    for i in range(0, n_tasks, chunk):
        ce = min(i + chunk, n_tasks)
        ca = ce - i
        dr = (demo_data.unsqueeze(0).expand(ca, B, n_points, -1)
              .reshape(-1, n_points, demo_data.size(-1)))
        dt = eval_task.evaluate(
            demo_data, eval_task.task_pool[i:ce].squeeze(-1).T, step=step,
        )
        if dt.ndim == 3:
            all_targets[i:ce] = dt.permute(2, 0, 1).detach().cpu()
            dt = dt.permute(2, 0, 1).reshape(-1, n_points)
        else:
            all_targets[i:ce] = dt.unsqueeze(0).detach().cpu()
            dt = dt.unsqueeze(0).expand(ca, -1, -1).reshape(-1, n_points)

        ch = extract_hidden_multi(
            model=model, demo_data=dr, demo_target=dt,
            layers=[layer], task_pos=task_pos,
        )
        hiddens[i:ce] = ch[0].reshape(ca, B, n_points, D).permute(0, 2, 1, 3).cpu()
        del dr, dt, ch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return hiddens, all_targets


@torch.no_grad()
def _extract_baseline_per_position(
    model, demo_data, dummy_targets, *, layer, n_points, D,
):
    """h_l at seq-pos 0, with x₀ replaced by xₜ for each t.

    This gives a baseline representation where only the current input
    varies but no in-context examples are present.

    Returns h_baseline: (T, B, D) float32 CPU tensor.
    """
    from icl.linear.task_vecs import extract_hidden_multi

    seq_pos_0 = [0]
    h_baseline = torch.empty(n_points, demo_data.shape[0], D, dtype=torch.float32)
    for t in range(n_points):
        d_t = demo_data.clone()
        d_t[:, 0, :] = demo_data[:, t, :]
        h_t = extract_hidden_multi(
            model=model, demo_data=d_t, demo_target=dummy_targets,
            layers=[layer], task_pos=seq_pos_0,
        )
        h_baseline[t] = h_t[0, :, 0, :].cpu()
        del d_t, h_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return h_baseline


# ---------------------------------------------------------------------------
# Task subspace fitting
# ---------------------------------------------------------------------------

def _fit_task_subspace(
    exp_name, layer, step, fit_n_samples, fit_positions, n_points,
    center_task_vecs, *, n_minor=0,
):
    """Fit a linear probe h ≈ π W + b and extract the task-subspace projector.

    Steps:
      1. Train OLS probe: h ≈ [π, x_t, ŷ_t] · [W; W_tok; W_logit] + b
      2. Take W ∈ ℝ^{K×D}  (optionally centre rows: W̃ = W − w̄1ᵀ)
      3. SVD:  W̃ = UΣVᵀ,  keep r columns with σᵢ > 10⁻⁶·σ₁
      4. Projector:  P_S = V_r V_rᵀ  ∈ ℝ^{D×D}

    Returns ``(W_fit, b_fit, basis, rank, P_task, fit_res)`` where

    - *W_fit* ``(K, D)`` — raw probe weights,
    - *b_fit* ``(D,)``   — probe bias,
    - *basis* ``(D, rank)`` — orthonormal basis for the task subspace,
    - *P_task* ``(D, D)``  — orthogonal projector onto *basis*.
    """
    from icl.linear.analysis.probes import train_linear_hidden_predictor

    if fit_positions is None:
        fit_positions = list(range(min(10, n_points), n_points))

    fit_res = train_linear_hidden_predictor(
        exp_name=exp_name,
        layer=layer,
        n_samples=fit_n_samples,
        positions=fit_positions,
        sample_mode="major",
        step=step,
        n_minor=n_minor,
        print_summary=False,
        skip_baselines=True,
    )
    W_fit = fit_res["model_weight"].float()
    b_fit = fit_res["model_bias"].float()

    if center_task_vecs:
        task_vecs = W_fit - W_fit.mean(dim=0, keepdim=True)
    else:
        task_vecs = W_fit.clone()
    _, S_tv, Vt_tv = torch.linalg.svd(task_vecs, full_matrices=False)
    rank = int((S_tv > 1e-6 * S_tv[0]).sum().item())
    basis = Vt_tv[:rank].T  # (D, rank)
    P_task = basis @ basis.T  # (D, D)
    return W_fit, b_fit, basis, rank, P_task, fit_res


# ---------------------------------------------------------------------------
# Projection-removal experiment (shared by 4 interventions)
# ---------------------------------------------------------------------------

def _run_projection_removal(
    model, layer, P_proj, scale, task_obj, eval_positions,
    n_samples, B, device, *,
    minor_only=False, track_oracle=False,
):
    """Causal intervention: h′ = h − s · h P  at layer l.

    Compares baseline MSE vs intervened MSE over *eval_positions*.
    P is an orthogonal projector (D×D), s is *scale* (default 1).

    Returns dict with baseline/intervened MSE (overall + per-position).
    """
    baseline_losses_by_pos = {p: [] for p in eval_positions}
    intervened_losses_by_pos = {p: [] for p in eval_positions}
    base_oracle_by_pos = {p: [] for p in eval_positions} if track_oracle else None
    int_oracle_by_pos = {p: [] for p in eval_positions} if track_oracle else None

    n_batches = max(1, (n_samples + B - 1) // B)
    orig_bs = int(task_obj.batch_size)
    task_obj.batch_size = B

    for bi in range(n_batches):
        if minor_only:
            demo_data, demo_tasks, demo_target = task_obj.sample_batch(
                step=bi + 33333, minor_only=True,
            )
        else:
            demo_data, demo_tasks, demo_target = task_obj.sample_batch(
                step=bi + 33333, is_eval=True,
            )
        demo_data = demo_data.to(device)
        demo_target = demo_target.to(device)
        if track_oracle:
            demo_tasks = demo_tasks.to(device)

        with torch.no_grad():
            preds_base = model(demo_data, demo_target)

        # Intervention hook:  h → h − s · hP  =  (I − sP) h
        # When s=1 and P is an orthogonal projector (P² = P = Pᵀ),
        # this removes the component of h in the column span of P,
        # leaving only the orthogonal complement.
        def hook_fn(module, inp, out, _P=P_proj, _s=scale):
            h = out if torch.is_tensor(out) else out[0]
            h_mod = h - _s * (h @ _P)
            if torch.is_tensor(out):
                return h_mod
            return (h_mod,) + out[1:]

        handle = model.transformer.blocks[layer].attn_block.register_forward_hook(
            hook_fn,
        )
        try:
            with torch.no_grad():
                preds_int = model(demo_data, demo_target)
        finally:
            handle.remove()

        if track_oracle:
            oracle_preds = (demo_data @ demo_tasks).squeeze(-1)

        for p in eval_positions:
            if p >= preds_base.shape[1]:
                continue
            mse_b = ((preds_base[:, p] - demo_target[:, p]) ** 2).mean().item()
            mse_i = ((preds_int[:, p] - demo_target[:, p]) ** 2).mean().item()
            baseline_losses_by_pos[p].append(mse_b)
            intervened_losses_by_pos[p].append(mse_i)
            if track_oracle:
                base_oracle_by_pos[p].append(
                    ((preds_base[:, p] - oracle_preds[:, p]) ** 2).mean().item()
                )
                int_oracle_by_pos[p].append(
                    ((preds_int[:, p] - oracle_preds[:, p]) ** 2).mean().item()
                )

        del demo_data, demo_target, preds_base, preds_int
        if track_oracle:
            del demo_tasks, oracle_preds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    task_obj.batch_size = orig_bs

    bp, ip, vp = [], [], []
    bop, iop = [], []
    for p in eval_positions:
        if baseline_losses_by_pos[p]:
            bp.append(np.mean(baseline_losses_by_pos[p]))
            ip.append(np.mean(intervened_losses_by_pos[p]))
            vp.append(p)
            if track_oracle:
                bop.append(np.mean(base_oracle_by_pos[p]))
                iop.append(np.mean(int_oracle_by_pos[p]))

    baseline_avg = float(np.mean(bp)) if bp else float("nan")
    intervened_avg = float(np.mean(ip)) if ip else float("nan")

    result = {
        "baseline": baseline_avg,
        "intervened": intervened_avg,
        "delta": intervened_avg - baseline_avg,
        "baseline_per_pos": bp,
        "intervened_per_pos": ip,
        "positions": vp,
    }
    if track_oracle:
        base_oracle_avg = float(np.mean(bop)) if bop else float("nan")
        int_oracle_avg = float(np.mean(iop)) if iop else float("nan")
        result.update({
            "baseline_to_oracle": base_oracle_avg,
            "intervened_to_oracle": int_oracle_avg,
            "delta_to_oracle": int_oracle_avg - base_oracle_avg,
        })
    return result


# ---------------------------------------------------------------------------
# Layer sweep
# ---------------------------------------------------------------------------

def _sweep_layers(exp_name, intervene_fn, layers, log_prefix, **kwargs):
    """Call *intervene_fn* for each layer, return ``{layer: result_dict}``."""
    if layers is None:
        from icl.linear.linear_path_utils import load_model_task_config
        _, _, config = load_model_task_config(exp_name)
        layers = list(range(config.model.n_layer))

    all_results = {}
    for l in layers:
        logger.info(f"[{log_prefix}] layer {l} ...")
        all_results[l] = intervene_fn(
            exp_name=exp_name, layer=l,
            verbose=False, print_summary=False, **kwargs,
        )
    return layers, all_results


# ---------------------------------------------------------------------------
# Joint task + token OLS fit
# ---------------------------------------------------------------------------

def _build_token_features(
    demo_data, demo_target, fit_point_pos,
    *, device, use_y, use_suff, n_dims,
):
    """Build token features [xₜ, y_{t-1}, XᵀY/t, triu(XᵀX/t)] at given positions.

    Always includes xₜ.  Optionally adds y_{t-1} (use_y) and
    running sufficient statistics (use_suff).  Returns (B, P, F).
    """
    x_at = demo_data.index_select(1, fit_point_pos)
    parts = [x_at]

    if use_y:
        B_cur = demo_data.shape[0]
        P_fit = fit_point_pos.shape[0]
        y_prev = torch.zeros(B_cur, P_fit, 1, device=device, dtype=demo_data.dtype)
        for i in range(P_fit):
            p = int(fit_point_pos[i].item())
            if p > 0:
                y_prev[:, i, 0] = demo_target[:, p - 1]
        parts.append(y_prev)

    if use_suff:
        # Running sufficient statistics for Bayesian linear regression:
        #   XᵀY(t) = Σ_{s≤t} xₛ yₛ   ∈ ℝ^d
        #   XᵀX(t) = Σ_{s≤t} xₛ xₛᵀ  ∈ ℝ^{d×d}   (symmetric)
        # Normalise by (t+1) for scale stability.  Store the upper
        # triangle of XᵀX/(t+1) to avoid redundant symmetric entries.
        # Final feature vector at position t:  [XᵀY/(t+1), vech(XᵀX/(t+1))]
        B_cur, T_cur, D_x = demo_data.shape
        triu_idx = torch.triu_indices(D_x, D_x, offset=0, device=device)
        suff_dim = D_x + D_x * (D_x + 1) // 2
        XtY = torch.zeros(B_cur, D_x, device=device, dtype=demo_data.dtype)
        XtX = torch.zeros(B_cur, D_x, D_x, device=device, dtype=demo_data.dtype)
        suff_all = torch.zeros(B_cur, T_cur, suff_dim, device=device, dtype=demo_data.dtype)
        for t in range(T_cur):
            xt = demo_data[:, t, :]
            yt = demo_target[:, t]
            XtY += torch.einsum("bd,b->bd", xt, yt)      # XᵀY += xₜ yₜ
            XtX += torch.einsum("bi,bj->bij", xt, xt)    # XᵀX += xₜ xₜᵀ
            denom = float(t + 1)
            xtx_upper = (XtX / denom)[:, triu_idx[0], triu_idx[1]]
            suff_all[:, t, :] = torch.cat([XtY / denom, xtx_upper], dim=-1)
        parts.append(suff_all.index_select(1, fit_point_pos))

    return torch.cat(parts, dim=-1)


def _joint_fit_task_token(
    model, layer, train_task,
    *, device,
    fit_positions, fit_n_samples, B, n_dims, n_embd,
    input_protection_features="x_y",
    center_task_vecs=True,
):
    """Joint OLS:  h = [π_major, tok] · [W_task; W_tok] + b.

    *tok* includes xₜ (always) and optionally y_{t-1} and XᵀY/t + XᵀX/t.
    After solving OLS, centres W_task rows and takes SVD to produce an
    orthonormal task-subspace basis.

    Returns dict: W_task, W_tok, basis (D, rank), rank, joint_r2.
    """
    from icl.linear.analysis.posterior import task_posterior_over_time_linear_regression

    valid = {"x", "x_y", "x_y_suff"}
    if input_protection_features not in valid:
        raise ValueError(
            f"input_protection_features={input_protection_features!r} not in {valid}"
        )
    use_y = input_protection_features in {"x_y", "x_y_suff"}
    use_suff = input_protection_features == "x_y_suff"

    n_major = train_task.n_tasks
    fit_seq_pos = torch.tensor(
        [2 * p for p in fit_positions], device=device, dtype=torch.long,
    )
    fit_point_pos = torch.tensor(fit_positions, device=device, dtype=torch.long)

    all_h, all_post, all_tok = [], [], []
    n_batches = (fit_n_samples + B - 1) // B
    saved_bs = int(train_task.batch_size)
    train_task.batch_size = B

    for bi in range(n_batches):
        data, _, target = train_task.sample_batch(step=bi, is_eval=False)
        data, target = data.to(device), target.to(device)

        post = task_posterior_over_time_linear_regression(
            train_task, data, target, include_minor=True,
        )
        post_at = post[:, fit_point_pos, :n_major]

        cache = {}
        def _hook(mod, inp, out, _c=cache):
            h = out if torch.is_tensor(out) else out[0]
            _c["h"] = h.index_select(1, fit_seq_pos).detach()

        handle = model.transformer.blocks[layer].attn_block.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                model(data, target)
        finally:
            handle.remove()

        tok_feat = _build_token_features(
            data, target, fit_point_pos,
            device=device, use_y=use_y, use_suff=use_suff, n_dims=n_dims,
        )

        all_h.append(cache["h"].cpu())
        all_post.append(post_at.cpu())
        all_tok.append(tok_feat.cpu())
        del data, target, post
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    train_task.batch_size = saved_bs

    H = torch.cat(all_h, 0).reshape(-1, n_embd).float()
    Pi = torch.cat(all_post, 0).reshape(-1, n_major).float()
    Tok = torch.cat(all_tok, 0)
    tok_feat_dim = Tok.shape[-1]
    Tok = Tok.reshape(-1, tok_feat_dim).float()
    del all_h, all_post, all_tok

    # ── Solve joint OLS: [π, tok, 1] → h ──────────────────────────
    # Stack features X = [π, tok] ∈ ℝ^{n × (K+F)} and solve
    #   min_{W,b}  ‖H − XW − 1bᵀ‖²_F
    # via pseudoinverse of the augmented matrix [X, 1].
    # W_task (first K rows) and W_tok (next F rows) give the
    # fitted directions for posterior and token features.
    X = torch.cat([Pi, Tok], dim=1)
    X_aug = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)
    W_aug = torch.linalg.pinv(X_aug) @ H
    W_task = W_aug[:n_major]    # (K, D)
    W_tok = W_aug[n_major:-1]   # (F, D)

    pred = X @ W_aug[:-1] + W_aug[-1]
    ss_res = ((H - pred) ** 2).sum().item()
    ss_tot = ((H - H.mean(0)) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    del X, X_aug, W_aug, H, Pi, Tok, pred
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Centre task vectors and extract an orthonormal basis via SVD.
    # W̃ = W_task − w̄1ᵀ  removes the mean direction (which is shared
    # by all tasks and thus uninteresting for discrimination).
    # SVD:  W̃ = UΣVᵀ  →  basis = V_{:rank}  (columns of V for σ > tol).
    if center_task_vecs:
        tv = W_task - W_task.mean(dim=0, keepdim=True)
    else:
        tv = W_task.clone()
    _, S, Vt = torch.linalg.svd(tv, full_matrices=False)
    rank = int((S > 1e-6 * S[0]).sum().item())
    basis = Vt[:rank].T  # (D, rank)

    return {
        "W_task": W_task,
        "W_tok": W_tok,
        "basis": basis,
        "rank": rank,
        "joint_r2": r2,
        "tok_feat_dim": tok_feat_dim,
    }


# ---------------------------------------------------------------------------
# Protected subspace construction
# ---------------------------------------------------------------------------

def _build_protected_subspace(
    basis, W_tok_fit, D_dim,
    *, device,
    token_protection_mode,
    token_protection_rank,
    token_protection_var_explained,
):
    """Build orthogonal complement of (task + token) protected subspace.

    S_prot = span(V_task)  ∪  span(V_tok)   (optionally PCA-reduced)
    P_orth = I − S_prot S_protᵀ

    Token protection modes:
      "full"          — protect all W_tok directions
      "residual_pca"  — PCA on (I − P_task) W_tok, keep top r components
      "none"          — protect task subspace only

    Returns dict: P_orth (D×D), U_orth, orth_dim, protected_rank.
    """
    if token_protection_mode not in {"residual_pca", "full", "none"}:
        raise ValueError(
            f"Unsupported token_protection_mode={token_protection_mode!r}. "
            "Use one of {'residual_pca', 'full', 'none'}."
        )

    tok_feat_dim = W_tok_fit.shape[0]
    W_tok_dir = W_tok_fit.T  # (D, tok_feat_dim)
    tok_basis = torch.empty(D_dim, 0, dtype=W_tok_dir.dtype)
    tok_rank = 0

    if token_protection_mode == "full":
        tok_basis = W_tok_dir
        tok_rank = int(tok_basis.shape[1])

    elif token_protection_mode == "residual_pca":
        # Project token directions onto the orthogonal complement of the
        # task subspace:  W_resid = (I − P_task) W_tok.
        # Then PCA the residual to find the most important token-specific
        # directions that are NOT already protected by the task basis.
        W_resid = W_tok_dir - basis @ (basis.T @ W_tok_dir)
        U_r, S_r, _ = torch.linalg.svd(W_resid, full_matrices=False)
        if S_r.numel() > 0 and S_r[0] > 0:
            rank_svd = int((S_r > 1e-6 * S_r[0]).sum().item())
            if token_protection_rank is None:
                # Auto-select rank to explain a target fraction of residual variance
                s2 = S_r[:rank_svd] ** 2
                cum = torch.cumsum(s2, dim=0) / s2.sum().clamp_min(1e-12)
                k_var = int(torch.searchsorted(
                    cum, torch.tensor(token_protection_var_explained),
                ).item()) + 1
                tok_rank = max(1, min(rank_svd, k_var))
            else:
                tok_rank = max(0, min(int(token_protection_rank), rank_svd))
            if tok_rank > 0:
                tok_basis = U_r[:, :tok_rank]

    # ── Build the orthogonal complement projector ────────────────────
    # Combine task basis and token basis into one protected subspace,
    # re-orthogonalise via SVD, then compute:
    #   P_prot = U_prot U_protᵀ       (projector onto protected space)
    #   P_orth = I − P_prot           (projector onto its complement)
    # The eigendecomposition of P_orth extracts an explicit orthonormal
    # basis U_orth for the unprotected subspace (eigenvalues ≈ 1).
    parts = [basis]
    if tok_basis.shape[1] > 0:
        parts.append(tok_basis)
    combined = torch.cat(parts, dim=1)
    U_c, S_c, _ = torch.linalg.svd(combined, full_matrices=False)
    prot_rank = int((S_c > 1e-6 * S_c[0]).sum().item())
    P_prot = U_c[:, :prot_rank] @ U_c[:, :prot_rank].T
    P_orth = (torch.eye(D_dim) - P_prot).to(device)

    eig_vals, eig_vecs = torch.linalg.eigh(P_orth)
    U_orth = eig_vecs[:, eig_vals > 0.5].to(device)

    return {
        "P_orth": P_orth,
        "U_orth": U_orth,
        "orth_dim": U_orth.shape[1],
        "protected_rank": prot_rank,
        "token_basis_rank": tok_rank,
        "token_basis_max_rank": tok_feat_dim,
    }
