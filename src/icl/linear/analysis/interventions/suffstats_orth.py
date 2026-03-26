"""Causal intervention: remove sufficient-statistics component from orth complement."""
import gc
from typing import Optional

import numpy as np
import torch

import icl.utils.notebook_utils as nu
from icl.utils.logger import setup_logger
from icl.linear.analysis._helpers import _show_or_close
from icl.linear.analysis.interventions._helpers import (
    _cleanup_model,
    _joint_fit_task_token,
    _build_protected_subspace,
)

logger = setup_logger(__name__)


def intervene_remove_suffstats_orth(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples_probe: int = 2000,
    n_samples_eval: int = 500,
    n_ood: int = 30,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    eval_positions: Optional[list] = None,
    center_task_vecs: bool = True,
    scale: float = 1.0,
    features: str = "suff_y",
    probe_source: str = "major",
    probe_type: str = "linear",
    probe_ridge_alpha: float = 0.0,
    mlp_hidden: int = 128,
    mlp_epochs: int = 300,
    mlp_lr: float = 1e-3,
    token_protection_mode: str = "residual_pca",
    token_protection_rank: Optional[int] = None,
    token_protection_var_explained: float = 0.9,
    input_protection_features: str = "x_y",
    target_subspace: Optional["torch.Tensor"] = None,
    verbose: bool = False,
    print_summary: bool = True,
) -> dict:
    """
    Linear regression counterpart of
    ``intervene_remove_unigram_orth_coin_nonpadded``.

    Pipeline:

    1. Joint fit ``[pi_major, tok_feat] -> hiddens`` to get unbiased task
       and token subspaces, then compute the orthogonal projector
       ``P_orth = I - P_protected``.  Token features in the joint fit are
       controlled by ``input_protection_features``.
    2. Collect probe features at each eval position.  Controlled by
       ``features``:

       - ``"suff"`` — running X^T Y / t, vech(X^T X / t)
       - ``"suff_y"`` — suff stats + y_t  (default)
       - ``"pred"`` — x_t @ beta_hat  (ridge prediction at position t)
       - ``"betahat"`` — full ridge beta_hat vector (n_dims dimensions)
       - ``"pgauss"`` — P(Z=K+1), the Bayesian posterior probability of
         being OOD, computed per position using data up to that point
       - Combine freely with ``_``, e.g. ``"suff_y_pred_pgauss"``
    3. Fit a probe ``features -> H_orth``.  ``probe_type="linear"`` uses
       pseudo-inverse; ``"mlp"`` trains a 2-layer MLP.
    4. For each mode (major, OOD): subtract the probe-predicted orth
       component and measure MSE change.

    Parameters
    ----------
    target_subspace : Tensor, optional
        If provided (shape ``(D, k)``), skip the task-subspace / token-
        protection computation entirely and use ``target_subspace @
        target_subspace.T`` as the projector.  Typically set to ``V_opt``
        from ``intervene_optimal_orth_direction_linear_nonpadded`` so the
        probe targets only the OOD-specific direction.  This makes the
        intervention selective: OOD samples have large projections onto
        V_opt (large removal), Major samples have small projections
        (minimal removal).

    Returns
    -------
    dict with baseline/intervened/delta MSE for major and OOD, probe R², etc.
    """
    from icl.linear.linear_path_utils import load_model_task_config
    from icl.linear.analysis.posterior import task_posterior_with_gaussian_linear_regression
    from icl.linear.linear_ood_analysis import (
        _create_eval_task_pool,
        _setup_eval_task,
    )

    valid_tokens = {"suff", "y", "pred", "pgauss", "betahat"}
    feat_tokens = set(features.split("_"))
    unknown = feat_tokens - valid_tokens
    if unknown:
        raise ValueError(
            f"Unknown feature tokens {unknown} in features={features!r}. "
            f"Valid tokens: {valid_tokens}. Combine with '_', e.g. 'suff_y_pred_pgauss'."
        )
    if probe_type not in {"linear", "mlp"}:
        raise ValueError(f"probe_type={probe_type!r} not in {{'linear', 'mlp'}}")

    use_suff = "suff" in feat_tokens
    use_y = "y" in feat_tokens
    use_pred = "pred" in feat_tokens
    use_pgauss = "pgauss" in feat_tokens
    use_betahat = "betahat" in feat_tokens

    _, train_task, config = load_model_task_config(exp_name)
    if step is None:
        step = config.training.total_steps
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    model.requires_grad_(False)
    device = config.device

    n_points = int(config.task.n_points)
    n_dims = int(config.task.n_dims)
    n_embd = int(config.model.n_embd)
    noise_scale = float(train_task.noise_scale)

    eval_task_pool, _ = _create_eval_task_pool(
        train_task, K=n_ood, include_minor=False,
        device=device, n_minor=0,
    )
    ood_task = _setup_eval_task(config, eval_task_pool, B, device)
    ood_task.noise_scale = noise_scale

    D_dim = int(n_embd)

    if target_subspace is not None:
        V_ext = target_subspace.float()
        if V_ext.dim() == 1:
            V_ext = V_ext.unsqueeze(1)
        P_orth = (V_ext @ V_ext.T).to(device)
        rank = 0
        protected_rank = D_dim - V_ext.shape[1]
        token_basis_rank = 0

        joint_r2 = float("nan")

        if verbose:
            logger.info(
                f"[suffstats-orth linear] Using external target_subspace "
                f"(rank {V_ext.shape[1]})"
            )
    else:
        # ---- 1. Joint fit + protected subspace ----
        if fit_positions is None:
            fit_positions = list(range(min(10, n_points), n_points))

        fit = _joint_fit_task_token(
            model, layer, train_task,
            device=device, fit_positions=fit_positions,
            fit_n_samples=fit_n_samples, B=B,
            n_dims=n_dims, n_embd=D_dim,
            input_protection_features=input_protection_features,
            center_task_vecs=center_task_vecs,
        )
        joint_r2 = fit["joint_r2"]
        rank = fit["rank"]

        prot = _build_protected_subspace(
            fit["basis"], fit["W_tok"], D_dim,
            device=device,
            token_protection_mode=token_protection_mode,
            token_protection_rank=token_protection_rank,
            token_protection_var_explained=token_protection_var_explained,
        )
        P_orth = prot["P_orth"]
        protected_rank = prot["protected_rank"]
        token_basis_rank = prot["token_basis_rank"]

        if verbose:
            logger.info(
                f"[suffstats-orth linear] Joint fit R\u00b2={joint_r2:.4f}, "
                f"task rank={rank}, protected rank={protected_rank} "
                f"(token-mode={token_protection_mode}, token-rank={token_basis_rank}, "
                f"input-features={input_protection_features})"
            )

    # ---- 2. Feature / hidden collection ----
    if eval_positions is None:
        eval_positions = list(range(n_points))
    eval_seq_positions = [2 * p for p in eval_positions]
    eval_seq_pos_tensor = torch.tensor(eval_seq_positions, device=device, dtype=torch.long)
    eval_point_tensor = torch.tensor(eval_positions, device=device, dtype=torch.long)

    suff_dim = n_dims

    def _compute_suff_stats(demo_data, demo_target):
        """Running average XᵀY/t at each position.  Returns (B, T, d)."""
        B_s, T, D_x = demo_data.shape
        XtY = torch.zeros(B_s, D_x, device=demo_data.device, dtype=demo_data.dtype)
        out = torch.zeros(B_s, T, D_x, device=demo_data.device, dtype=demo_data.dtype)
        for t in range(T):
            x_t = demo_data[:, t, :]
            y_t = demo_target[:, t]
            XtY = XtY + torch.einsum("bd,b->bd", x_t, y_t)   # XᵀY += xₜyₜ
            out[:, t, :] = XtY / float(t + 1)
        return out

    def _compute_ridge_pred(demo_data, demo_target, lam=None):
        """Running ridge prediction:  ŷₜ = ŵₜᵀxₜ  where
        ŵₜ = (XᵀX + λI)⁻¹ XᵀY  uses data up to position t.
        Returns (B, T, 1)."""
        if lam is None:
            lam = noise_scale ** 2
        B_s, T, D_x = demo_data.shape
        XtY = torch.zeros(B_s, D_x, device=demo_data.device, dtype=demo_data.dtype)
        XtX = torch.zeros(B_s, D_x, D_x, device=demo_data.device, dtype=demo_data.dtype)
        reg = lam * torch.eye(D_x, device=demo_data.device, dtype=demo_data.dtype)
        out = torch.zeros(B_s, T, 1, device=demo_data.device, dtype=demo_data.dtype)
        for t in range(T):
            x_t = demo_data[:, t, :]
            y_t = demo_target[:, t]
            XtY = XtY + torch.einsum("bd,b->bd", x_t, y_t)
            XtX = XtX + torch.einsum("bi,bj->bij", x_t, x_t)
            beta = torch.linalg.solve(XtX + reg, XtY.unsqueeze(-1)).squeeze(-1)
            out[:, t, 0] = (x_t * beta).sum(-1)   # ŵₜᵀxₜ
        return out

    def _compute_betahat(demo_data, demo_target, lam=None):
        """Running ridge estimate  ŵₜ = (XᵀX + λI)⁻¹ XᵀY.
        Returns (B, T, d)."""
        if lam is None:
            lam = noise_scale ** 2
        B_s, T, D_x = demo_data.shape
        XtY = torch.zeros(B_s, D_x, device=demo_data.device, dtype=demo_data.dtype)
        XtX = torch.zeros(B_s, D_x, D_x, device=demo_data.device, dtype=demo_data.dtype)
        reg = lam * torch.eye(D_x, device=demo_data.device, dtype=demo_data.dtype)
        out = torch.zeros(B_s, T, D_x, device=demo_data.device, dtype=demo_data.dtype)
        for t in range(T):
            x_t = demo_data[:, t, :]
            y_t = demo_target[:, t]
            XtY = XtY + torch.einsum("bd,b->bd", x_t, y_t)
            XtX = XtX + torch.einsum("bi,bj->bij", x_t, x_t)
            beta = torch.linalg.solve(XtX + reg, XtY.unsqueeze(-1)).squeeze(-1)
            out[:, t, :] = beta
        return out

    def _compute_pgauss_per_pos(demo_data, demo_target, eval_pos_list):
        """P(Z=K+1) at each eval position using data up to that position.
        Returns (B, len(eval_pos_list), 1)."""
        import math
        B_s, T, D_x = demo_data.shape
        P_ev = len(eval_pos_list)
        out = torch.zeros(B_s, P_ev, 1, device=demo_data.device, dtype=torch.float32)

        W_pool = train_task.task_pool.to(demo_data.device).squeeze(-1).float()  # (K, D)
        K_pool = W_pool.shape[0]
        sigma2 = float(train_task.noise_scale) ** 2
        tau2 = float(train_task.task_scale) ** 2
        p_gauss_prior = float(getattr(train_task, "p_minor", 0.0))
        if p_gauss_prior <= 0:
            p_gauss_prior = 1.0 / (K_pool + 1)
        log_prior_k = math.log(max((1.0 - p_gauss_prior) / K_pool, 1e-30))
        log_prior_g = math.log(max(p_gauss_prior, 1e-30))

        eps = 1e-30
        eye_D = torch.eye(D_x, device=demo_data.device, dtype=torch.float32)
        data_f = demo_data.float()
        tgt_f = demo_target.float()

        XtX = torch.zeros(B_s, D_x, D_x, device=demo_data.device, dtype=torch.float32)
        Xty = torch.zeros(B_s, D_x, device=demo_data.device, dtype=torch.float32)
        yty = torch.zeros(B_s, device=demo_data.device, dtype=torch.float32)
        sse_k = torch.zeros(B_s, K_pool, device=demo_data.device, dtype=torch.float32)

        eval_pos_set = {p: idx for idx, p in enumerate(eval_pos_list)}
        for t in range(T):
            xt = data_f[:, t, :]
            yt = tgt_f[:, t]
            XtX = XtX + torch.einsum("bi,bj->bij", xt, xt)
            Xty = Xty + xt * yt.unsqueeze(-1)
            yty = yty + yt ** 2
            pred_k = torch.einsum("bd,kd->bk", xt, W_pool)
            sse_k = sse_k + (yt.unsqueeze(-1) - pred_k) ** 2

            if t in eval_pos_set:
                n_obs = t + 1
                log_norm = -0.5 * n_obs * math.log(2 * math.pi * sigma2)
                loglik_k = log_norm + (-0.5 * sse_k / sigma2)  # (B, K)

                Lambda_N = (1.0 / tau2) * eye_D.unsqueeze(0) + (1.0 / sigma2) * XtX
                L_N = torch.linalg.cholesky(Lambda_N)
                log_det_LN = 2.0 * L_N.diagonal(dim1=-2, dim2=-1).log().sum(-1)
                log_det_prior = D_x * math.log(1.0 / tau2)
                m_N = torch.cholesky_solve(Xty.unsqueeze(-1) / sigma2, L_N).squeeze(-1)
                quad = (m_N * (Lambda_N @ m_N.unsqueeze(-1)).squeeze(-1)).sum(-1)
                logml_gauss = (
                    -0.5 * n_obs * math.log(2 * math.pi)
                    - 0.5 * n_obs * math.log(sigma2)
                    + 0.5 * log_det_prior
                    - 0.5 * log_det_LN
                    - 0.5 / sigma2 * yty
                    + 0.5 * quad
                )

                log_joint_k = loglik_k + log_prior_k      # (B, K)
                log_joint_g = logml_gauss + log_prior_g    # (B,)
                log_all = torch.cat([log_joint_k, log_joint_g.unsqueeze(-1)], dim=-1)
                log_Z = torch.logsumexp(log_all, dim=-1)
                pgauss_val = torch.exp(log_joint_g - log_Z)
                out[:, eval_pos_set[t], 0] = pgauss_val

        return out

    def _build_features(suff_at_pos, y_at_pos, pred_at_pos=None,
                        pgauss_at_pos=None, betahat_at_pos=None):
        """Concatenate selected probe features.  All inputs: (B, P, *)."""
        parts = []
        if use_suff:
            parts.append(suff_at_pos)
        if use_y:
            parts.append(y_at_pos.unsqueeze(-1) if y_at_pos.dim() == 2 else y_at_pos)
        if use_pred:
            parts.append(pred_at_pos)
        if use_pgauss:
            parts.append(pgauss_at_pos)
        if use_betahat:
            parts.append(betahat_at_pos)
        return torch.cat(parts, dim=-1)

    if probe_source not in {"major", "ood", "minor"}:
        raise ValueError(f"probe_source={probe_source!r} not in {{'major', 'ood', 'minor'}}")

    if probe_source == "minor":
        if not (int(getattr(train_task, "n_minor_tasks", 0)) > 0
                and getattr(train_task, "minor_pool", None) is not None):
            raise ValueError(
                "probe_source='minor' requires a train_task with a non-empty minor_pool "
                f"(n_minor_tasks={getattr(train_task, 'n_minor_tasks', 0)})"
            )

    if verbose:
        logger.info(f"[suffstats-orth linear] Collecting probe training data (source={probe_source}) ...")

    if probe_source == "ood":
        probe_task = ood_task
    else:
        probe_task = train_task

    probe_minor_only = (probe_source == "minor")

    all_H_probe, all_F_probe = [], []
    n_probe_batches = (n_samples_probe + B - 1) // B
    original_bs = int(probe_task.batch_size)
    probe_task.batch_size = B

    for bi in range(n_probe_batches):
        demo_data, _, demo_target = probe_task.sample_batch(
            step=bi + 11111, is_eval=True, minor_only=probe_minor_only,
        )
        demo_data, demo_target = demo_data.to(device), demo_target.to(device)

        cache = {}

        def hook_fn(module, inp, out, _cache=cache):
            h = out if torch.is_tensor(out) else out[0]
            _cache["h"] = h.index_select(1, eval_seq_pos_tensor).detach()

        handle = model.transformer.blocks[layer].attn_block.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                model(demo_data, demo_target)
            hiddens = cache["h"]  # (B, P, D)
        finally:
            handle.remove()

        suff = _compute_suff_stats(demo_data, demo_target) if use_suff else None
        suff_at_pos = suff.index_select(1, eval_point_tensor) if suff is not None else torch.empty(demo_data.shape[0], len(eval_positions), 0, device=device)
        y_at_pos = demo_target.index_select(1, eval_point_tensor)    # (B, P)
        pred = _compute_ridge_pred(demo_data, demo_target) if use_pred else None
        pred_at_pos = pred.index_select(1, eval_point_tensor) if pred is not None else None
        pgauss = _compute_pgauss_per_pos(demo_data, demo_target, eval_positions) if use_pgauss else None
        bhat = _compute_betahat(demo_data, demo_target) if use_betahat else None
        bhat_at_pos = bhat.index_select(1, eval_point_tensor) if bhat is not None else None
        feat = _build_features(suff_at_pos, y_at_pos, pred_at_pos, pgauss, bhat_at_pos)

        all_H_probe.append(hiddens.cpu())
        all_F_probe.append(feat.cpu())
        del demo_data, demo_target, hiddens, suff, suff_at_pos, y_at_pos, pred, pred_at_pos, pgauss, bhat, bhat_at_pos, feat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    probe_task.batch_size = original_bs

    H_probe = torch.cat(all_H_probe, 0)  # (N, P, D)
    F_probe = torch.cat(all_F_probe, 0)  # (N, P, feat_dim)
    n_flat = H_probe.shape[0] * H_probe.shape[1]
    feat_dim = F_probe.shape[-1]
    H_flat = H_probe.reshape(n_flat, -1).float()
    F_flat = F_probe.reshape(n_flat, -1).float()

    H_orth_flat = H_flat @ P_orth.cpu()

    if verbose:
        logger.info(
            f"[suffstats-orth linear] Feature dim={feat_dim} "
            f"(features={features!r}), probe_type={probe_type!r}"
        )

    # ---- 3. Fit probe ----
    n_probe_train = int(0.8 * n_flat)
    probe_perm = torch.randperm(n_flat)
    pr_tr, pr_va = probe_perm[:n_probe_train], probe_perm[n_probe_train:]

    if probe_type == "linear":
        def _ridge_fit(F_in, Y_in, alpha):
            """Ridge regression: W = (F^T F + alpha I)^{-1} F^T Y."""
            FtF = F_in.T @ F_in
            if alpha > 0:
                FtF = FtF + alpha * torch.eye(FtF.shape[0], dtype=FtF.dtype)
            return torch.linalg.solve(FtF, F_in.T @ Y_in)

        ones_tr = torch.ones(n_probe_train, 1)
        F_aug_tr = torch.cat([F_flat[pr_tr], ones_tr], dim=1)
        W_aug_val = _ridge_fit(F_aug_tr, H_orth_flat[pr_tr], probe_ridge_alpha)
        pred_va = torch.cat([F_flat[pr_va], torch.ones(pr_va.shape[0], 1)], dim=1) @ W_aug_val
        ss_res = ((H_orth_flat[pr_va] - pred_va) ** 2).sum().item()
        ss_tot = ((H_orth_flat[pr_va] - H_orth_flat[pr_va].mean(0)) ** 2).sum().item()
        probe_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        ones_col = torch.ones(n_flat, 1)
        F_aug = torch.cat([F_flat, ones_col], dim=1)
        W_aug = _ridge_fit(F_aug, H_orth_flat, probe_ridge_alpha)
        W_probe = W_aug[:-1].to(device)  # (feat_dim, D)
        b_probe = W_aug[-1].to(device)   # (D,)
        mlp_probe = None

    else:
        import torch.nn as tnn

        out_dim = H_orth_flat.shape[1]
        mlp = tnn.Sequential(
            tnn.Linear(feat_dim, mlp_hidden),
            tnn.ReLU(),
            tnn.Linear(mlp_hidden, out_dim),
        )
        opt_mlp = torch.optim.Adam(mlp.parameters(), lr=mlp_lr, weight_decay=probe_ridge_alpha)
        F_tr, H_tr = F_flat[pr_tr], H_orth_flat[pr_tr]
        F_va, H_va = F_flat[pr_va], H_orth_flat[pr_va]

        for ep in range(mlp_epochs):
            pred = mlp(F_tr)
            loss = ((H_tr - pred) ** 2).mean()
            opt_mlp.zero_grad()
            loss.backward()
            opt_mlp.step()

        with torch.no_grad():
            pred_va = mlp(F_va)
            ss_res = ((H_va - pred_va) ** 2).sum().item()
            ss_tot = ((H_va - H_va.mean(0)) ** 2).sum().item()
        probe_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        mlp.eval()
        mlp_probe = mlp.to(device)
        W_probe = None
        b_probe = None

    if verbose:
        logger.info(
            f"[suffstats-orth linear] Probe R² ({probe_type}, {features}): "
            f"{probe_r2:.4f}"
        )

    del H_probe, F_probe, H_flat, F_flat, H_orth_flat, all_H_probe, all_F_probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ---- 4. Intervention experiment ----
    def _run_experiment(task_obj, n_samples):
        baseline_losses_by_pos = {p: [] for p in eval_positions}
        intervened_losses_by_pos = {p: [] for p in eval_positions}
        base_oracle_by_pos = {p: [] for p in eval_positions}
        int_oracle_by_pos = {p: [] for p in eval_positions}

        orig_bs = int(task_obj.batch_size)
        task_obj.batch_size = B
        n_batches = (n_samples + B - 1) // B

        for bi in range(n_batches):
            demo_data, demo_tasks, demo_target = task_obj.sample_batch(step=bi + 22222, is_eval=True)
            demo_data, demo_target = demo_data.to(device), demo_target.to(device)
            demo_tasks = demo_tasks.to(device)

            suff = _compute_suff_stats(demo_data, demo_target) if use_suff else None
            suff_at_pos = suff.index_select(1, eval_point_tensor) if suff is not None else torch.empty(demo_data.shape[0], len(eval_positions), 0, device=device)
            y_at_pos = demo_target.index_select(1, eval_point_tensor)
            pred = _compute_ridge_pred(demo_data, demo_target) if use_pred else None
            pred_at_pos = pred.index_select(1, eval_point_tensor) if pred is not None else None
            pgauss = _compute_pgauss_per_pos(demo_data, demo_target, eval_positions) if use_pgauss else None
            bhat = _compute_betahat(demo_data, demo_target) if use_betahat else None
            bhat_at_pos = bhat.index_select(1, eval_point_tensor) if bhat is not None else None
            feat = _build_features(suff_at_pos, y_at_pos, pred_at_pos, pgauss, bhat_at_pos)  # (B, P, feat_dim)

            with torch.no_grad():
                preds_base = model(demo_data, demo_target)

            def intervention_hook(module, inp, out,
                                  _feat=feat, _W=W_probe, _b=b_probe,
                                  _mlp=mlp_probe, _P=P_orth):
                h = out if torch.is_tensor(out) else out[0]
                B_h, L_h, D_h = h.shape
                n_pts = _feat.shape[1]
                pred_full = torch.zeros(B_h, L_h, D_h, device=h.device, dtype=h.dtype)
                if _mlp is not None:
                    with torch.no_grad():
                        pred_at_pos = _mlp(_feat)  # (B, P, D)
                else:
                    pred_at_pos = _feat @ _W + _b  # (B, P, D)
                for p_idx, p in enumerate(eval_positions):
                    seq_p = eval_seq_positions[p_idx]
                    if seq_p < L_h and p_idx < n_pts:
                        pred_full[:, seq_p, :] = pred_at_pos[:, p_idx, :]
                pred_projected = pred_full @ _P
                h_modified = h - scale * pred_projected
                if torch.is_tensor(out):
                    return h_modified
                return (h_modified,) + out[1:]

            handle = model.transformer.blocks[layer].attn_block.register_forward_hook(
                intervention_hook,
            )
            try:
                with torch.no_grad():
                    preds_int = model(demo_data, demo_target)
            finally:
                handle.remove()

            oracle_preds = (demo_data @ demo_tasks).squeeze(-1)  # (B, n_points)

            for p in eval_positions:
                if p >= preds_base.shape[1]:
                    continue
                mse_base = ((preds_base[:, p] - demo_target[:, p]) ** 2).mean().item()
                mse_int = ((preds_int[:, p] - demo_target[:, p]) ** 2).mean().item()
                baseline_losses_by_pos[p].append(mse_base)
                intervened_losses_by_pos[p].append(mse_int)
                mse_base_oracle = ((preds_base[:, p] - oracle_preds[:, p]) ** 2).mean().item()
                mse_int_oracle = ((preds_int[:, p] - oracle_preds[:, p]) ** 2).mean().item()
                base_oracle_by_pos[p].append(mse_base_oracle)
                int_oracle_by_pos[p].append(mse_int_oracle)

            del demo_data, demo_tasks, demo_target, preds_base, preds_int, oracle_preds, suff, feat
            del suff_at_pos, y_at_pos, pred, pred_at_pos, pgauss, bhat, bhat_at_pos
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        task_obj.batch_size = orig_bs

        baseline_per_pos, intervened_per_pos, valid_positions = [], [], []
        base_oracle_per_pos, int_oracle_per_pos = [], []
        for p in eval_positions:
            if baseline_losses_by_pos[p]:
                baseline_per_pos.append(np.mean(baseline_losses_by_pos[p]))
                intervened_per_pos.append(np.mean(intervened_losses_by_pos[p]))
                base_oracle_per_pos.append(np.mean(base_oracle_by_pos[p]))
                int_oracle_per_pos.append(np.mean(int_oracle_by_pos[p]))
                valid_positions.append(p)

        baseline_avg = float(np.mean(baseline_per_pos)) if baseline_per_pos else float("nan")
        intervened_avg = float(np.mean(intervened_per_pos)) if intervened_per_pos else float("nan")
        base_oracle_avg = float(np.mean(base_oracle_per_pos)) if base_oracle_per_pos else float("nan")
        int_oracle_avg = float(np.mean(int_oracle_per_pos)) if int_oracle_per_pos else float("nan")

        return {
            "baseline": baseline_avg,
            "intervened": intervened_avg,
            "delta": intervened_avg - baseline_avg,
            "baseline_to_oracle": base_oracle_avg,
            "intervened_to_oracle": int_oracle_avg,
            "delta_to_oracle": int_oracle_avg - base_oracle_avg,
            "baseline_per_pos": baseline_per_pos,
            "intervened_per_pos": intervened_per_pos,
            "positions": valid_positions,
        }

    def _eval_per_major_task(n_samples):
        """Evaluate the intervention separately for each major task."""
        task_pool = train_task.task_pool.to(device)
        n_major = task_pool.shape[0]

        bl = [{p: [] for p in eval_positions} for _ in range(n_major)]
        it = [{p: [] for p in eval_positions} for _ in range(n_major)]

        n_batches = (n_samples + B - 1) // B
        orig_bs = int(train_task.batch_size)
        train_task.batch_size = B

        for bi in range(n_batches):
            demo_data, demo_tasks, demo_target = train_task.sample_batch(
                step=bi + 33333, is_eval=True,
            )
            demo_data, demo_target = demo_data.to(device), demo_target.to(device)
            demo_tasks = demo_tasks.to(device)

            dists = (demo_tasks.unsqueeze(1) - task_pool.unsqueeze(0)).norm(dim=2).squeeze(-1)
            task_idx = dists.argmin(dim=1)

            suff = _compute_suff_stats(demo_data, demo_target) if use_suff else None
            suff_at_pos = (suff.index_select(1, eval_point_tensor) if suff is not None
                           else torch.empty(demo_data.shape[0], len(eval_positions), 0, device=device))
            y_at_pos = demo_target.index_select(1, eval_point_tensor)
            pred = _compute_ridge_pred(demo_data, demo_target) if use_pred else None
            pred_at_pos = pred.index_select(1, eval_point_tensor) if pred is not None else None
            pgauss = _compute_pgauss_per_pos(demo_data, demo_target, eval_positions) if use_pgauss else None
            bhat = _compute_betahat(demo_data, demo_target) if use_betahat else None
            bhat_at_pos = bhat.index_select(1, eval_point_tensor) if bhat is not None else None
            feat = _build_features(suff_at_pos, y_at_pos, pred_at_pos, pgauss, bhat_at_pos)

            with torch.no_grad():
                preds_base = model(demo_data, demo_target)

            def _hook(module, inp, out,
                      _feat=feat, _W=W_probe, _b=b_probe,
                      _mlp=mlp_probe, _P=P_orth):
                h = out if torch.is_tensor(out) else out[0]
                B_h, L_h, D_h = h.shape
                n_pts = _feat.shape[1]
                pf = torch.zeros(B_h, L_h, D_h, device=h.device, dtype=h.dtype)
                if _mlp is not None:
                    with torch.no_grad():
                        pa = _mlp(_feat)
                else:
                    pa = _feat @ _W + _b
                for p_idx, p in enumerate(eval_positions):
                    seq_p = eval_seq_positions[p_idx]
                    if seq_p < L_h and p_idx < n_pts:
                        pf[:, seq_p, :] = pa[:, p_idx, :]
                h_mod = h - scale * (pf @ _P)
                if torch.is_tensor(out):
                    return h_mod
                return (h_mod,) + out[1:]

            handle = model.transformer.blocks[layer].attn_block.register_forward_hook(_hook)
            try:
                with torch.no_grad():
                    preds_int = model(demo_data, demo_target)
            finally:
                handle.remove()

            oracle_preds = (demo_data @ demo_tasks).squeeze(-1)

            for p in eval_positions:
                if p >= preds_base.shape[1]:
                    continue
                bl_mse = (preds_base[:, p] - oracle_preds[:, p]) ** 2
                it_mse = (preds_int[:, p] - oracle_preds[:, p]) ** 2
                for k in range(n_major):
                    mask = task_idx == k
                    if mask.any():
                        bl[k][p].append(bl_mse[mask].mean().item())
                        it[k][p].append(it_mse[mask].mean().item())

            del demo_data, demo_tasks, demo_target, preds_base, preds_int, oracle_preds
            del suff, suff_at_pos, y_at_pos, pred, pred_at_pos, pgauss, bhat, bhat_at_pos, feat

        train_task.batch_size = orig_bs

        results = []
        for k in range(n_major):
            valid = [p for p in eval_positions if bl[k][p]]
            if valid:
                bl_avg = float(np.mean([np.mean(bl[k][p]) for p in valid]))
                it_avg = float(np.mean([np.mean(it[k][p]) for p in valid]))
            else:
                bl_avg, it_avg = float("nan"), float("nan")
            delta = it_avg - bl_avg
            pct = 100.0 * delta / bl_avg if bl_avg > 0 else float("nan")
            results.append({
                "task_idx": k,
                "baseline": bl_avg,
                "intervened": it_avg,
                "delta": delta,
                "pct": pct,
            })
        return results

    if verbose:
        logger.info("[suffstats-orth linear] Running major experiment ...")
    res_major = _run_experiment(train_task, n_samples_eval)

    if verbose:
        logger.info("[suffstats-orth linear] Running OOD experiment ...")
    res_ood = _run_experiment(ood_task, n_samples_eval)

    if verbose:
        logger.info("[suffstats-orth linear] Running per-task evaluation ...")
    per_task = _eval_per_major_task(n_samples_eval)

    _cleanup_model(model)

    pct_major = (
        100.0 * res_major["delta"] / res_major["baseline"]
        if res_major["baseline"] > 0 else float("nan")
    )
    pct_ood = (
        100.0 * res_ood["delta"] / res_ood["baseline"]
        if res_ood["baseline"] > 0 else float("nan")
    )

    results = {
        "baseline_loss_major": res_major["baseline"],
        "intervened_loss_major": res_major["intervened"],
        "delta_loss_major": res_major["delta"],
        "pct_increase_major": pct_major,
        "baseline_loss_ood": res_ood["baseline"],
        "intervened_loss_ood": res_ood["intervened"],
        "delta_loss_ood": res_ood["delta"],
        "pct_increase_ood": pct_ood,
        "baseline_to_oracle_major": res_major["baseline_to_oracle"],
        "intervened_to_oracle_major": res_major["intervened_to_oracle"],
        "delta_to_oracle_major": res_major["delta_to_oracle"],
        "baseline_to_oracle_ood": res_ood["baseline_to_oracle"],
        "intervened_to_oracle_ood": res_ood["intervened_to_oracle"],
        "delta_to_oracle_ood": res_ood["delta_to_oracle"],
        "baseline_per_pos_major": res_major["baseline_per_pos"],
        "intervened_per_pos_major": res_major["intervened_per_pos"],
        "baseline_per_pos_ood": res_ood["baseline_per_pos"],
        "intervened_per_pos_ood": res_ood["intervened_per_pos"],
        "eval_positions": res_major["positions"],
        "layer": layer,
        "scale": scale,
        "probe_r2": probe_r2,
        "joint_r2": joint_r2,
        "task_subspace_rank": rank,
        "protected_rank": protected_rank,
        "token_protection_mode": token_protection_mode,
        "token_protection_rank": token_basis_rank,
        "uses_target_subspace": target_subspace is not None,
        "features": features,
        "probe_type": probe_type,
        "feat_dim": feat_dim,
        "oracle_mse": noise_scale ** 2,
        "per_task": per_task,
    }

    if print_summary:
        oracle_mse = noise_scale ** 2
        print(f"\n{'=' * 65}")
        mode_label = "V_opt" if target_subspace is not None else "Orth"
        print(
            f"Causal Intervention: Remove Features from {mode_label}  "
            f"(layer {layer}, scale={scale})"
        )
        print(f"{'=' * 65}")
        if target_subspace is not None:
            ts_rank = 1 if target_subspace.dim() == 1 else target_subspace.shape[1]
            print(
                f"  Target: external subspace (rank {ts_rank})  |  "
                f"Features: {features} (dim={feat_dim})  |  "
                f"Probe: {probe_type}  |  R²: {probe_r2:.4f}"
            )
        else:
            _r2 = "\u00b2"
            print(
                f"  Joint fit R{_r2}: {joint_r2:.4f}  |  "
                f"Task rank: {rank}  |  Protected rank: {protected_rank}"
            )
            print(
                f"  Features: {features} (dim={feat_dim})  |  "
                f"Probe: {probe_type}  |  Probe R{_r2}: {probe_r2:.4f}"
            )
        print(f"  Eval positions: {len(res_major['positions'])} positions\n")
        print(f"{'Metric':<30} {'Major':>12} {'OOD':>12}")
        print("-" * 54)
        print(
            f"{'Baseline MSE (→ target)':<30} "
            f"{res_major['baseline']:>12.4f} "
            f"{res_ood['baseline']:>12.4f}"
        )
        print(
            f"{'Intervened MSE (→ target)':<30} "
            f"{res_major['intervened']:>12.4f} "
            f"{res_ood['intervened']:>12.4f}"
        )
        print(
            f"{'Δ MSE (→ target)':<30} "
            f"{res_major['delta']:>12.4f} "
            f"{res_ood['delta']:>12.4f}"
        )
        print(
            f"{'% increase (→ target)':<30} "
            f"{pct_major:>11.1f}% "
            f"{pct_ood:>11.1f}%"
        )
        print()
        print(
            f"{'Baseline MSE (→ oracle)':<30} "
            f"{res_major['baseline_to_oracle']:>12.4f} "
            f"{res_ood['baseline_to_oracle']:>12.4f}"
        )
        print(
            f"{'Intervened MSE (→ oracle)':<30} "
            f"{res_major['intervened_to_oracle']:>12.4f} "
            f"{res_ood['intervened_to_oracle']:>12.4f}"
        )
        print(
            f"{'Δ MSE (→ oracle)':<30} "
            f"{res_major['delta_to_oracle']:>12.4f} "
            f"{res_ood['delta_to_oracle']:>12.4f}"
        )

        if per_task:
            _delta_hdr = "\u0394 MSE"
            print(f"\n  Per-task intervention effect (MSE \u2192 oracle):")
            print(f"    {'Task':<8} {'Baseline':>10} {'Intervened':>11} "
                  f"{_delta_hdr:>10} {'% increase':>11}")
            print(f"    {'-' * 51}")
            for pt in per_task:
                print(f"    {'Task ' + str(pt['task_idx']):<8} "
                      f"{pt['baseline']:>10.4f} {pt['intervened']:>11.4f} "
                      f"{pt['delta']:>10.4f} {pt['pct']:>10.1f}%")

    return results


def plot_remove_suffstats_orth_across_layers(
    exp_name: str,
    layers: Optional[list] = None,
    target_subspaces: Optional[dict] = None,
    figsize: tuple = (14, 6),
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs,
):
    """
    Sweep ``intervene_remove_suffstats_orth`` across layers.

    Parameters
    ----------
    target_subspaces : dict, optional
        Mapping ``{layer: V_opt_tensor}`` from
        ``intervene_optimal_orth_direction_linear_nonpadded`` results.
        When provided, the probe targets only V_opt directions (OOD-
        selective intervention).  Typically built as::

            target_subspaces = {l: vopt_results[l]["directions"] for l in layers}

    **kwargs
        Forwarded to ``intervene_remove_suffstats_orth``
        (e.g. ``B``, ``n_samples_probe``, ``scale``, ``features``, ``probe_type``, etc.).

    Returns ``(fig, fig_per_task, all_results)``.
    """
    import matplotlib.pyplot as plt
    from icl.linear.linear_path_utils import load_model_task_config

    if layers is None:
        _, _, config = load_model_task_config(exp_name)
        layers = list(range(config.model.n_layer))

    all_results = {}
    for l in layers:
        logger.info(f"[suffstats-orth sweep] layer {l} ...")
        ts = target_subspaces.get(l) if target_subspaces is not None else None
        all_results[l] = intervene_remove_suffstats_orth(
            exp_name=exp_name, layer=l, target_subspace=ts,
            verbose=False, print_summary=True, **kwargs,
        )

    # ---- plotting ----
    def _extract(key):
        return [all_results[l][key] for l in layers]

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(layers)), 6))
    x = np.arange(len(layers))
    bar_w = 0.35

    int_maj = np.sqrt(_extract("intervened_to_oracle_major"))
    int_ood = np.sqrt(_extract("intervened_to_oracle_ood"))

    ax.bar(x - bar_w / 2, int_maj, bar_w, label="Major (intervened)", color="#2196F3", alpha=0.85)
    ax.bar(x + bar_w / 2, int_ood, bar_w, label="OOD (intervened)", color="#FF9800", alpha=0.85)
    for i, (vm, vo) in enumerate(zip(int_maj, int_ood)):
        ax.text(x[i] - bar_w / 2, vm, f"{vm:.4f}", ha="center", va="bottom", fontsize=9)
        ax.text(x[i] + bar_w / 2, vo, f"{vo:.4f}", ha="center", va="bottom", fontsize=9)

    base_maj = np.sqrt(np.mean(_extract("baseline_to_oracle_major")))
    base_ood = np.sqrt(np.mean(_extract("baseline_to_oracle_ood")))
    ax.axhline(base_maj, color="#2196F3", ls="--", lw=1.5, alpha=0.7,
               label=f"Major baseline ({base_maj:.4f})")
    ax.axhline(base_ood, color="#FF9800", ls="--", lw=1.5, alpha=0.7,
               label=f"OOD baseline ({base_ood:.4f})")

    ax.set(xlabel="Layer", ylabel="RMSE (model \u2192 oracle)", title="")
    ax.set_xticks(x, [str(l) for l in layers])
    ax.tick_params(labelsize=14)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(15)
    ax.title.set_size(15)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    features = kwargs.get("features", "suff_y")
    probe_type = kwargs.get("probe_type", "linear")
    scale = kwargs.get("scale", 1.0)
    feat_label = features.replace("_", "+")

    r2_str = "  ".join(f"L{l}: {all_results[l]['probe_r2']:.3f}" for l in layers)
    fig.text(0.5, -0.02,
             f"Probe R\u00b2 ({probe_type}, {feat_label} \u2192 H_orth):  {r2_str}",
             ha="center", fontsize=11, style="italic")

    fig.suptitle("", fontsize=18, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    _show_or_close(fig, show)

    # ---- per-task subplot ----
    sample_pt = all_results[layers[0]].get("per_task")
    if sample_pt:
        n_major = len(sample_pt)
        colors = plt.cm.Set2(np.linspace(0, 1, max(n_major, 3)))

        fig_pt, ax_pt = plt.subplots(figsize=(max(8, 1.4 * len(layers)), 6))
        bar_w_pt = 0.8 / n_major
        for k in range(n_major):
            deltas = []
            for l in layers:
                pt = all_results[l].get("per_task", [])
                if k < len(pt):
                    deltas.append(np.sqrt(pt[k]["intervened"]) - np.sqrt(pt[k]["baseline"]))
                else:
                    deltas.append(float("nan"))
            offset = (k - (n_major - 1) / 2) * bar_w_pt
            bars = ax_pt.bar(x + offset, deltas, bar_w_pt,
                             label=f"Task {k}", color=colors[k], alpha=0.85)
            for i, v in enumerate(deltas):
                if not np.isnan(v):
                    ax_pt.text(x[i] + offset, v, f"{v:.4f}",
                               ha="center", va="bottom", fontsize=8)

        ax_pt.set(xlabel="Layer",
                  ylabel="\u0394 RMSE (model \u2192 oracle)",
                  title="")
        ax_pt.set_xticks(x, [str(l) for l in layers])
        ax_pt.tick_params(labelsize=13)
        ax_pt.xaxis.label.set_size(15)
        ax_pt.yaxis.label.set_size(14)
        ax_pt.title.set_size(15)
        ax_pt.legend(fontsize=11)
        ax_pt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        _show_or_close(fig_pt, show)
    else:
        fig_pt = None

    return fig, fig_pt, all_results
