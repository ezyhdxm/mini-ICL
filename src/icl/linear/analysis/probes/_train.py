"""OLS probe training for linear regression analysis."""

import gc
from typing import Optional

import torch

import icl.utils.notebook_utils as nu
from icl.linear.analysis._helpers import (
    _temporary_linear_minor_task_setup,
    _temporary_task_attributes,
)
from icl.linear.linear_path_utils import load_model_task_config
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)


def train_linear_hidden_predictor(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples: int = 1000,
    step: Optional[int] = None,
    n_minor: Optional[int] = None,
    verbose: bool = False,
    positions: Optional[list] = None,
    validation_split: float = 0.2,
    include_position_bias: bool = False,
    include_logit: bool = False,
    uniform_sampling: bool = True,
    sample_mode: str = "train",
    skip_baselines: bool = False,
    print_summary: bool = True,
    use_log_posterior: bool = False,
    anchor_minor_samples: Optional[int] = None,
    extraction_point: str = "post_attn",
) -> dict:
    """Joint OLS: h = [φ(π), xₜ, (optional) ŷₜ] · W + b.

    φ is log or identity (controlled by *use_log_posterior*).
    Joint fitting ensures W_task directions are orthogonal to input/logit
    confounds (Frisch–Waugh–Lovell).

    Returns dict with fitted weights, R², partial R², F-tests, and
    design-matrix collinearity diagnostics (VIF, condition number).
    When multiple positions are fit jointly, includes one-hot position
    nuisance features (enabled by ``include_position_bias``) so the
    intercept can vary with position.
    """
    from icl.linear.linear_path_utils import load_model_task_config
    from icl.linear.analysis.posterior import task_posterior_over_time_linear_regression

    model_loaded, train_task, config = load_model_task_config(exp_name)
    del model_loaded

    if step is None:
        step = config.training.total_steps

    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True
    )
    model.eval().to(config.device)

    n_points = int(config.task.n_points)
    if positions is None:
        positions = list(range(min(10, n_points)))
    else:
        positions = list(positions)
        if not all(0 <= p < n_points for p in positions):
            raise ValueError(f"All positions must be in [0, {n_points - 1}], got {positions}")

    with _temporary_linear_minor_task_setup(
        train_task,
        n_minor=n_minor,
        uniform_sampling=uniform_sampling,
        sample_mode=sample_mode,
    ) as (include_minor, n_tasks_total, _original_p_minor):
        x_seq_positions = [2 * p for p in positions]

        device = config.device
        seq_pos = torch.tensor(x_seq_positions, device=device, dtype=torch.long)
        point_pos = torch.tensor(positions, device=device, dtype=torch.long)

        if verbose:
            logger.info(
                f"Linear non-padded hidden predictor: layer={layer}, B={B}, n_samples={n_samples}, "
                f"positions={positions}, sample_mode={sample_mode}, "
                f"n_tasks(total)={n_tasks_total}"
            )

        all_hiddens, all_posteriors, all_x_tokens, all_logits = [], [], [], []
        n_batches = (n_samples + B - 1) // B

        def _sample_batch_for_mode(batch_idx: int):
            if sample_mode == "train":
                demo_data, _, demo_target = train_task.sample_batch(step=batch_idx, is_eval=False)
                return demo_data, demo_target
            demo_data = train_task.sample_data(step=batch_idx)
            if sample_mode == "major":
                idx = torch.randint(0, train_task.n_tasks, (B,), device=demo_data.device)
                tasks = train_task.task_pool[idx]
            elif sample_mode == "minor":
                if not include_minor:
                    raise ValueError("sample_mode='minor' requested but no minor tasks are available.")
                idx = torch.randint(0, train_task.n_minor_tasks, (B,), device=demo_data.device)
                tasks = train_task.minor_pool[idx]
            else:
                raise ValueError(f"Unknown sample_mode={sample_mode!r}. Use 'train', 'major', or 'minor'.")
            demo_target = train_task.evaluate(demo_data, tasks, step=batch_idx)
            return demo_data, demo_target

        with _temporary_task_attributes(train_task, batch_size=B):
            for batch_idx in range(n_batches):
                demo_data, demo_target = _sample_batch_for_mode(batch_idx)
                demo_data = demo_data.to(device)
                demo_target = demo_target.to(device)

                post = task_posterior_over_time_linear_regression(
                    train_task,
                    demo_data,
                    demo_target,
                    include_minor=include_minor,
                )  # (B, T, K)

                cache = {}
                if extraction_point == "post_mlp":
                    layer_module = model.transformer.blocks[layer]
                else:
                    layer_module = model.transformer.blocks[layer].attn_block

                def hook_fn(module, inp, out):
                    cache["hidden"] = out.index_select(dim=1, index=seq_pos).detach()  # (B, P, D)

                handle = layer_module.register_forward_hook(hook_fn)
                try:
                    with torch.no_grad():
                        preds = model(demo_data, demo_target)  # yhat: (B, T)
                    h_batch = cache["hidden"]  # (B, P, D)
                finally:
                    handle.remove()

                x_batch = demo_data.index_select(dim=1, index=point_pos)  # x_{p_j}: (B, P, d)
                post_batch = post[:, point_pos, :]  # pi_{p_j}: (B, P, K)
                logit_batch = preds.index_select(dim=1, index=point_pos).unsqueeze(-1)  # (B, P, 1)

                all_hiddens.append(h_batch.cpu())
                all_posteriors.append(post_batch.cpu())
                all_x_tokens.append(x_batch.cpu())
                all_logits.append(logit_batch.cpu())

                del demo_data, demo_target, post, preds, h_batch, x_batch, post_batch, logit_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            _do_anchor = (
                sample_mode == "major"
                and include_minor
                and _original_p_minor > 1e-6
                and (anchor_minor_samples is None or anchor_minor_samples > 0)
            )
            if _do_anchor:
                n_anchor = anchor_minor_samples if anchor_minor_samples is not None else max(B, n_samples // 5)
                n_anchor_batches = (n_anchor + B - 1) // B
                if verbose:
                    logger.info(f"[linear] anchoring with {n_anchor} train-mode samples ({n_anchor_batches} batches)")
                for batch_idx in range(n_batches, n_batches + n_anchor_batches):
                    demo_data = train_task.sample_data(step=batch_idx)
                    demo_data = demo_data.to(device)
                    idx_maj = torch.randint(0, train_task.n_tasks, (B // 2,), device=demo_data.device)
                    idx_min = torch.randint(0, train_task.n_minor_tasks, (B - B // 2,), device=demo_data.device)
                    tasks = torch.cat([train_task.task_pool[idx_maj], train_task.minor_pool[idx_min]], dim=0)
                    demo_target = train_task.evaluate(demo_data, tasks, step=batch_idx)

                    post = task_posterior_over_time_linear_regression(
                        train_task, demo_data, demo_target, include_minor=include_minor,
                    )

                    cache = {}
                    if extraction_point == "post_mlp":
                        layer_module = model.transformer.blocks[layer]
                    else:
                        layer_module = model.transformer.blocks[layer].attn_block

                    def hook_fn(module, inp, out):
                        cache["hidden"] = out.index_select(dim=1, index=seq_pos).detach()

                    handle = layer_module.register_forward_hook(hook_fn)
                    try:
                        with torch.no_grad():
                            preds = model(demo_data, demo_target)
                        h_batch = cache["hidden"]
                    finally:
                        handle.remove()

                    x_batch = demo_data.index_select(dim=1, index=point_pos)
                    post_batch = post[:, point_pos, :]
                    logit_batch = preds.index_select(dim=1, index=point_pos).unsqueeze(-1)

                    all_hiddens.append(h_batch.cpu())
                    all_posteriors.append(post_batch.cpu())
                    all_x_tokens.append(x_batch.cpu())
                    all_logits.append(logit_batch.cpu())

                    del demo_data, demo_target, post, preds, h_batch, x_batch, post_batch, logit_batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    hiddens_all = torch.cat(all_hiddens, dim=0)      # (N, P, D)
    post_all = torch.cat(all_posteriors, dim=0)      # (N, P, K_full)
    x_all = torch.cat(all_x_tokens, dim=0)           # (N, P, n_dims)
    logit_all = torch.cat(all_logits, dim=0)         # (N, P, 1)

    n_major = train_task.n_tasks
    post_all = post_all[:, :, :n_major]  # (N, P, K_major)

    N_seq = hiddens_all.shape[0]
    n_seq_train = int(N_seq * (1 - validation_split))
    seq_perm = torch.randperm(N_seq)
    seq_tr, seq_va = seq_perm[:n_seq_train], seq_perm[n_seq_train:]

    def _flatten(tensor, indices):
        return tensor[indices].reshape(-1, tensor.shape[-1]).float()

    Ytr = _flatten(hiddens_all, seq_tr)
    Yva = _flatten(hiddens_all, seq_va)
    post_tr = _flatten(post_all, seq_tr)
    post_va = _flatten(post_all, seq_va)
    x_tr = _flatten(x_all, seq_tr)
    x_va = _flatten(x_all, seq_va)
    logit_tr = _flatten(logit_all, seq_tr)
    logit_va = _flatten(logit_all, seq_va)

    n_pos = hiddens_all.shape[1]
    use_pos_bias = include_position_bias and n_pos > 1
    if use_pos_bias:
        pos_tr = torch.arange(n_pos).unsqueeze(0).expand(seq_tr.shape[0], n_pos).reshape(-1)
        pos_va = torch.arange(n_pos).unsqueeze(0).expand(seq_va.shape[0], n_pos).reshape(-1)
        X_pos_full_tr = torch.zeros(pos_tr.shape[0], n_pos, dtype=torch.float32)
        X_pos_full_tr.scatter_(1, pos_tr.unsqueeze(1), 1.0)
        X_pos_full_va = torch.zeros(pos_va.shape[0], n_pos, dtype=torch.float32)
        X_pos_full_va.scatter_(1, pos_va.unsqueeze(1), 1.0)
        X_pos_tr = X_pos_full_tr[:, :-1]
        X_pos_va = X_pos_full_va[:, :-1]
    else:
        X_pos_tr = None
        X_pos_va = None

    if use_log_posterior:
        X_main_tr = torch.log(post_tr + 1e-10)
        X_main_va = torch.log(post_va + 1e-10)
    else:
        X_main_tr, X_main_va = post_tr, post_va

    _n_posterior_orig = X_main_tr.shape[1]
    _post_sum_tr = X_main_tr.sum(dim=1)
    _sum_std = _post_sum_tr.std().item()
    _drop_last_col = (
        not use_log_posterior
        and _n_posterior_orig > 1
        and _sum_std < 1e-3
    )
    if _drop_last_col:
        X_main_tr = X_main_tr[:, :-1]
        X_main_va = X_main_va[:, :-1]

    X_tok_tr, X_tok_va = x_tr, x_va
    if include_logit:
        X_logit_tr, X_logit_va = logit_tr, logit_va
    else:
        X_logit_tr = torch.zeros(X_tok_tr.shape[0], 0, dtype=torch.float32)
        X_logit_va = torch.zeros(X_tok_va.shape[0], 0, dtype=torch.float32)

    n_total = hiddens_all.shape[0] * hiddens_all.shape[1]
    n_train = Ytr.shape[0]

    def _fit_ols(Xtr, Ytr, Xva, Yva):
        """OLS with intercept.  Returns ``(W, b, stats)``.

        Solves  min_W,b  ‖Y − XW − 1bᵀ‖²_F  via pseudoinverse of [X, 1].
        R² = 1 − SS_res / SS_tot   where SS_tot = ‖Y − Ȳ‖²_F.
        """
        ones_tr = torch.ones(Xtr.shape[0], 1, dtype=Xtr.dtype, device=Xtr.device)
        Xtr_aug = torch.cat([Xtr, ones_tr], dim=1)  # [X, 1]: (n, p+1)
        W_aug = torch.linalg.pinv(Xtr_aug) @ Ytr    # (p+1, D)
        W = W_aug[:-1, :]  # (p, D)
        b = W_aug[-1, :]   # (D,)

        pred_tr = Xtr @ W + b
        pred_va = Xva @ W + b
        tr_res = Ytr - pred_tr
        va_res = Yva - pred_va

        tr_ss_res = (tr_res ** 2).sum().item()
        va_ss_res = (va_res ** 2).sum().item()
        tr_ss_tot = ((Ytr - Ytr.mean(dim=0)) ** 2).sum().item()
        va_ss_tot = ((Yva - Yva.mean(dim=0)) ** 2).sum().item()
        n_dim = Ytr.shape[1]
        return W, b, {
            "tr_mse": tr_ss_res / (Ytr.shape[0] * n_dim),
            "va_mse": va_ss_res / (Yva.shape[0] * n_dim),
            "tr_r2": 1.0 - tr_ss_res / tr_ss_tot if tr_ss_tot > 0 else float("nan"),
            "va_r2": 1.0 - va_ss_res / va_ss_tot if va_ss_tot > 0 else float("nan"),
            "tr_ss_res": tr_ss_res,
            "va_ss_res": va_ss_res,
            "n_features": Xtr.shape[1],
        }

    def _fit_mlp_r2(Xtr, Ytr, Xva, Yva,
                    hidden_dim=128, epochs=200, lr=1e-3, batch_size=4096):
        """Fit a 2-layer MLP and return validation R² (linearity check)."""
        from torch import nn
        d_in, d_out = Xtr.shape[1], Ytr.shape[1]
        mlp = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_out),
        )
        opt = torch.optim.Adam(mlp.parameters(), lr=lr)
        for _ in range(epochs):
            perm = torch.randperm(Xtr.shape[0])
            for i in range(0, Xtr.shape[0], batch_size):
                batch = perm[i:i + batch_size]
                loss = ((mlp(Xtr[batch]) - Ytr[batch]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        mlp.eval()
        with torch.no_grad():
            pred = mlp(Xva)
            ss_res = ((Yva - pred) ** 2).sum().item()
            ss_tot = ((Yva - Yva.mean(dim=0)) ** 2).sum().item()
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    X_joint_tr_parts = [X_main_tr, X_tok_tr, X_logit_tr]
    X_joint_va_parts = [X_main_va, X_tok_va, X_logit_va]
    if X_pos_tr is not None:
        X_joint_tr_parts.append(X_pos_tr)
        X_joint_va_parts.append(X_pos_va)
    X_joint_tr = torch.cat(X_joint_tr_parts, dim=1)
    X_joint_va = torch.cat(X_joint_va_parts, dim=1)
    W_joint, b_joint, joint_s = _fit_ols(X_joint_tr, Ytr, X_joint_va, Yva)

    d_main = X_main_tr.shape[1]
    d_tok = X_tok_tr.shape[1]
    d_logit = X_logit_tr.shape[1]
    d_pos = X_pos_tr.shape[1] if X_pos_tr is not None else 0
    W_task_raw = W_joint[:d_main, :]
    W_tok_block = W_joint[d_main:d_main + d_tok, :]
    W_logit_block = W_joint[d_main + d_tok:d_main + d_tok + d_logit, :]
    W_pos_raw = W_joint[d_main + d_tok + d_logit:, :] if d_pos > 0 else None

    if _drop_last_col:
        W_task = torch.zeros((_n_posterior_orig, W_task_raw.shape[1]),
                             dtype=W_task_raw.dtype)
        W_task[:_n_posterior_orig - 1, :] = W_task_raw
    else:
        W_task = W_task_raw

    if W_pos_raw is not None:
        W_pos_block = torch.zeros((n_pos, W_pos_raw.shape[1]),
                                  dtype=W_pos_raw.dtype)
        W_pos_block[:n_pos - 1, :] = W_pos_raw
    else:
        W_pos_block = None

    if X_pos_tr is not None:
        X_main_marg_tr = torch.cat([X_main_tr, X_pos_tr], dim=1)
        X_main_marg_va = torch.cat([X_main_va, X_pos_va], dim=1)
        X_tok_marg_tr = torch.cat([X_tok_tr, X_pos_tr], dim=1)
        X_tok_marg_va = torch.cat([X_tok_va, X_pos_va], dim=1)
        X_logit_marg_tr = torch.cat([X_logit_tr, X_pos_tr], dim=1)
        X_logit_marg_va = torch.cat([X_logit_va, X_pos_va], dim=1)
    else:
        X_main_marg_tr, X_main_marg_va = X_main_tr, X_main_va
        X_tok_marg_tr, X_tok_marg_va = X_tok_tr, X_tok_va
        X_logit_marg_tr, X_logit_marg_va = X_logit_tr, X_logit_va

    _, _, pi_s = _fit_ols(X_main_marg_tr, Ytr, X_main_marg_va, Yva)
    _, _, tok_s = _fit_ols(X_tok_marg_tr, Ytr, X_tok_marg_va, Yva)
    _, _, logit_s = _fit_ols(X_logit_marg_tr, Ytr, X_logit_marg_va, Yva)

    X_post_tok_parts_tr = [X_main_tr, X_tok_tr]
    X_post_tok_parts_va = [X_main_va, X_tok_va]
    if X_pos_tr is not None:
        X_post_tok_parts_tr.append(X_pos_tr)
        X_post_tok_parts_va.append(X_pos_va)
    X_post_tok_tr = torch.cat(X_post_tok_parts_tr, dim=1)
    X_post_tok_va = torch.cat(X_post_tok_parts_va, dim=1)
    _, _, post_tok_s = _fit_ols(X_post_tok_tr, Ytr, X_post_tok_va, Yva)

    X_post_logit_parts_tr = [X_main_tr, X_logit_tr]
    X_post_logit_parts_va = [X_main_va, X_logit_va]
    if X_pos_tr is not None:
        X_post_logit_parts_tr.append(X_pos_tr)
        X_post_logit_parts_va.append(X_pos_va)
    X_post_logit_tr = torch.cat(X_post_logit_parts_tr, dim=1)
    X_post_logit_va = torch.cat(X_post_logit_parts_va, dim=1)
    _, _, post_logit_s = _fit_ols(X_post_logit_tr, Ytr, X_post_logit_va, Yva)

    X_tok_logit_parts_tr = [X_tok_tr, X_logit_tr]
    X_tok_logit_parts_va = [X_tok_va, X_logit_va]
    if X_pos_tr is not None:
        X_tok_logit_parts_tr.append(X_pos_tr)
        X_tok_logit_parts_va.append(X_pos_va)
    X_tok_logit_tr = torch.cat(X_tok_logit_parts_tr, dim=1)
    X_tok_logit_va = torch.cat(X_tok_logit_parts_va, dim=1)
    _, _, tok_logit_s = _fit_ols(X_tok_logit_tr, Ytr, X_tok_logit_va, Yva)

    _eps = 1e-10
    partial_r2_post = (
        (joint_s["va_r2"] - tok_logit_s["va_r2"])
        / max(1.0 - tok_logit_s["va_r2"], _eps)
    )
    partial_r2_tok = (
        (joint_s["va_r2"] - post_logit_s["va_r2"])
        / max(1.0 - post_logit_s["va_r2"], _eps)
    )
    partial_r2_logit = (
        (joint_s["va_r2"] - post_tok_s["va_r2"])
        / max(1.0 - post_tok_s["va_r2"], _eps)
    )

    n_tr = Ytr.shape[0]
    p_full = d_main + d_tok + d_logit + d_pos
    df_den = n_tr - p_full - 1

    def _f_test(ss_reduced, ss_full, q, df_d):
        if q <= 0 or df_d <= 0 or ss_full <= 0:
            return {"F": float("nan"), "p": float("nan"),
                    "df_num": q, "df_den": df_d}
        f_val = ((ss_reduced - ss_full) / q) / (ss_full / df_d)
        try:
            from scipy.stats import f as f_dist
            p_val = float(f_dist.sf(max(f_val, 0.0), q, df_d))
        except ImportError:
            p_val = float("nan")
        return {"F": f_val, "p": p_val, "df_num": q, "df_den": df_d}

    f_test_post = _f_test(
        tok_logit_s["tr_ss_res"], joint_s["tr_ss_res"], d_main, df_den,
    )
    f_test_tok = _f_test(
        post_logit_s["tr_ss_res"], joint_s["tr_ss_res"], d_tok, df_den,
    )
    f_test_logit = _f_test(
        post_tok_s["tr_ss_res"], joint_s["tr_ss_res"], d_logit, df_den,
    )

    cond_num = float(torch.linalg.cond(
        torch.cat([X_joint_tr, torch.ones(n_tr, 1, dtype=X_joint_tr.dtype, device=X_joint_tr.device)], dim=1)
    ).item())

    def _group_vif(X_group, X_rest):
        """Generalised VIF for a multi-column feature group.

        Regress X_group on X_rest → multivariate R².
        VIF = 1 / (1 − R²).  VIF = 1 means no collinearity;
        VIF > 10 is a common threshold for concern.
        """
        ones = torch.ones(X_rest.shape[0], 1, dtype=X_rest.dtype, device=X_rest.device)
        X_aug = torch.cat([X_rest, ones], dim=1)
        W = torch.linalg.pinv(X_aug) @ X_group
        pred = X_aug @ W
        ss_res = ((X_group - pred) ** 2).sum().item()
        ss_tot = ((X_group - X_group.mean(0)) ** 2).sum().item()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return 1.0 / max(1.0 - r2, 1e-10), r2

    if X_pos_tr is not None and d_logit > 0:
        vif_post, r2_post_from_rest = _group_vif(
            X_main_tr, torch.cat([X_tok_tr, X_logit_tr, X_pos_tr], dim=1))
        vif_tok, r2_tok_from_rest = _group_vif(
            X_tok_tr, torch.cat([X_main_tr, X_logit_tr, X_pos_tr], dim=1))
        vif_logit, r2_logit_from_rest = _group_vif(
            X_logit_tr, torch.cat([X_main_tr, X_tok_tr, X_pos_tr], dim=1))
    elif d_logit > 0:
        vif_post, r2_post_from_rest = _group_vif(
            X_main_tr, torch.cat([X_tok_tr, X_logit_tr], dim=1))
        vif_tok, r2_tok_from_rest = _group_vif(
            X_tok_tr, torch.cat([X_main_tr, X_logit_tr], dim=1))
        vif_logit, r2_logit_from_rest = _group_vif(
            X_logit_tr, torch.cat([X_main_tr, X_tok_tr], dim=1))
    else:
        if X_pos_tr is not None:
            vif_post, r2_post_from_rest = _group_vif(
                X_main_tr, torch.cat([X_tok_tr, X_pos_tr], dim=1))
            vif_tok, r2_tok_from_rest = _group_vif(
                X_tok_tr, torch.cat([X_main_tr, X_pos_tr], dim=1))
        else:
            vif_post, r2_post_from_rest = _group_vif(
                X_main_tr, X_tok_tr)
            vif_tok, r2_tok_from_rest = _group_vif(
                X_tok_tr, X_main_tr)
        vif_logit, r2_logit_from_rest = float("nan"), float("nan")

    gvif_post = vif_post ** (1.0 / (2 * d_main)) if d_main > 0 else float("nan")
    gvif_tok = vif_tok ** (1.0 / (2 * d_tok)) if d_tok > 0 else float("nan")
    gvif_logit = vif_logit ** (1.0 / (2 * d_logit)) if d_logit > 0 else float("nan")

    def _pairwise_r2(Xa, Xb):
        """R² from regressing Xa on Xb (with intercept)."""
        ones = torch.ones(Xb.shape[0], 1, dtype=Xb.dtype, device=Xb.device)
        X_aug = torch.cat([Xb, ones], dim=1)
        W = torch.linalg.pinv(X_aug) @ Xa
        pred = X_aug @ W
        ss_res = ((Xa - pred) ** 2).sum().item()
        ss_tot = ((Xa - Xa.mean(0)) ** 2).sum().item()
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    pairwise_r2_post_tok = _pairwise_r2(X_main_tr, X_tok_tr)
    pairwise_r2_post_logit = _pairwise_r2(X_main_tr, X_logit_tr)
    pairwise_r2_tok_logit = _pairwise_r2(X_tok_tr, X_logit_tr)

    design_diagnostics = {
        "condition_number": cond_num,
        "n_features": {"posterior": d_main, "token": d_tok, "logit": d_logit,
                        "position": d_pos,
                        "total": d_main + d_tok + d_logit + d_pos},
        "vif": {"posterior": vif_post, "token": vif_tok, "logit": vif_logit},
        "gvif_adj": {"posterior": gvif_post, "token": gvif_tok, "logit": gvif_logit},
        "r2_from_rest": {"posterior": r2_post_from_rest, "token": r2_tok_from_rest,
                         "logit": r2_logit_from_rest},
        "pairwise_r2": {"post_tok": pairwise_r2_post_tok,
                         "post_logit": pairwise_r2_post_logit,
                         "tok_logit": pairwise_r2_tok_logit},
    }

    diagnostics = {
        "r2_posterior_only": pi_s["va_r2"],
        "r2_token_only": tok_s["va_r2"],
        "r2_logit_only": logit_s["va_r2"],
        "r2_post_tok": post_tok_s["va_r2"],
        "r2_post_logit": post_logit_s["va_r2"],
        "r2_tok_logit": tok_logit_s["va_r2"],
        "r2_joint": joint_s["va_r2"],
        "partial_r2_posterior": partial_r2_post,
        "partial_r2_token": partial_r2_tok,
        "partial_r2_logit": partial_r2_logit,
        "f_test_posterior": f_test_post,
        "f_test_token": f_test_tok,
        "f_test_logit": f_test_logit,
        "condition_number": cond_num,
        "design_diagnostics": design_diagnostics,
        "mlp_val_r2": None,
        "position_bias_included": bool(use_pos_bias),
        "posterior_column_dropped": bool(_drop_last_col),
    }

    geometry = None

    if not skip_baselines:
        diagnostics["mlp_val_r2"] = _fit_mlp_r2(
            X_joint_tr, Ytr, X_joint_va, Yva,
        )

        eps = 1e-10
        rank_tol = 1e-5
        Wt_f = W_task_raw.T.float()   # (D, K)
        Wx_f = W_tok_block.T.float()  # (D, d)

        def _rank_basis(M: torch.Tensor) -> torch.Tensor:
            """Orthonormal basis for col(M), truncated to numerical rank."""
            U, S, _ = torch.linalg.svd(M, full_matrices=False)
            r = (S > S[0] * rank_tol).sum().item()
            return U[:, :r]

        Qt = _rank_basis(Wt_f)  # (D, r_task)
        Qx = _rank_basis(Wx_f)  # (D, r_tok)

        cos_angles = torch.linalg.svdvals(Qt.T @ Qx).clamp(0.0, 1.0)
        angles_deg = torch.rad2deg(torch.acos(cos_angles))

        subspace_angles = {
            "principal_angles_deg": angles_deg.tolist(),
            "mean_angle_deg": angles_deg.mean().item(),
            "min_angle_deg": angles_deg.min().item(),
            "max_cos": cos_angles.max().item(),
            "mean_cos2": cos_angles.pow(2).mean().item(),
            "rank_task": Qt.shape[1],
            "rank_token": Qx.shape[1],
        }

        c_task_va = (X_main_va @ W_task_raw).float()  # (n_val, D)
        c_tok_va = (X_tok_va @ W_tok_block).float()  # (n_val, D)
        dot = (c_task_va * c_tok_va).sum(dim=1)  # (n_val,)
        norms = c_task_va.norm(dim=1) * c_tok_va.norm(dim=1) + eps
        per_sample_cos = dot / norms  # (n_val,)

        component_cosine = {
            "mean": per_sample_cos.mean().item(),
            "std": per_sample_cos.std().item(),
            "median": per_sample_cos.median().item(),
            "abs_mean": per_sample_cos.abs().mean().item(),
        }

        Wt_rows = W_task_raw.float()
        Wx_rows = W_tok_block.float()
        Wt_norm = Wt_rows / (Wt_rows.norm(dim=1, keepdim=True) + eps)
        Wx_norm = Wx_rows / (Wx_rows.norm(dim=1, keepdim=True) + eps)
        row_cos_matrix = Wt_norm @ Wx_norm.T
        row_cosine = {
            "matrix": row_cos_matrix.cpu(),
            "max_abs": row_cos_matrix.abs().max().item(),
            "mean_abs": row_cos_matrix.abs().mean().item(),
        }

        geometry = {
            "joint_train_mse": joint_s["tr_mse"],
            "joint_val_mse": joint_s["va_mse"],
            "joint_train_r2": joint_s["tr_r2"],
            "joint_val_r2": joint_s["va_r2"],
            "subspace_angles": subspace_angles,
            "component_cosine": component_cosine,
            "row_cosine": row_cosine,
            "component_weight": {
                "main": W_task.cpu(),
                "token": W_tok_block.cpu(),
                "logit": W_logit_block.cpu(),
                "bias": b_joint.cpu(),
            },
        }

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    results = {
        "train_mse": joint_s["tr_mse"],
        "val_mse": joint_s["va_mse"],
        "train_r2": joint_s["tr_r2"],
        "val_r2": joint_s["va_r2"],
        "model_weight": W_task.cpu(),
        "model_bias": b_joint.cpu(),
        "token_weight": W_tok_block.cpu(),
        "logit_weight": W_logit_block.cpu(),
        "position_weight": W_pos_block.cpu() if W_pos_block is not None else None,
        "diagnostics": diagnostics,
        "geometry": geometry,
        "layer": layer,
        "n_tasks": n_tasks_total,
        "hidden_dim": hiddens_all.shape[-1],
        "n_samples": n_total,
        "n_train": n_train,
        "n_val": Yva.shape[0],
        "positions": positions,
        "pad_mode": "none",
    }

    if print_summary:
        diag = diagnostics
        has_logit = d_logit > 0
        has_pos = diag.get("position_bias_included", False)
        pos_tag = "+pos" if has_pos else ""
        _r2 = "\u00b2"
        dropped = diag.get("posterior_column_dropped", False)
        print(f"\n=== Fit Summary (Linear, layer {layer}, mode={sample_mode!r}) ===")
        if dropped:
            print("  [posterior col dropped: sum(pi)=1 detected, last column removed to avoid dummy-variable trap]")
        if use_pos_bias:
            print("  [position one-hot: last column dropped to avoid dummy-variable trap]")
        print(f"{'Model':<30} {'Train R' + _r2:>10} {'Val R' + _r2:>10} {'Val MSE':>12}")
        print("-" * 64)

        if has_logit:
            rows = [
                (f"Joint (task+tok+logit{pos_tag})", joint_s),
                (f"Post+tok{pos_tag} (no logit)", post_tok_s),
                (f"Posterior{pos_tag}", pi_s),
                (f"Token{pos_tag}", tok_s),
                (f"Logit{pos_tag}", logit_s),
            ]
        else:
            rows = [
                (f"Joint (task+tok{pos_tag})", joint_s),
                (f"Posterior{pos_tag}", pi_s),
                (f"Token{pos_tag}", tok_s),
            ]

        for label, s in rows:
            if s:
                print(f"{label:<30} {s['tr_r2']:>10.4f} {s['va_r2']:>10.4f} {s['va_mse']:>12.6f}")
        print()
        _pr2 = "Partial R" + _r2
        if has_logit:
            print(f"{_pr2}  posterior|rest = {diag['partial_r2_posterior']:.4f}"
                  f"    token|rest = {diag['partial_r2_token']:.4f}"
                  f"    logit|rest = {diag['partial_r2_logit']:.4f}")
        else:
            print(f"{_pr2}  posterior|rest = {diag['partial_r2_posterior']:.4f}"
                  f"    token|rest = {diag['partial_r2_token']:.4f}")
        fp, ft, fl = diag["f_test_posterior"], diag["f_test_token"], diag["f_test_logit"]
        if has_logit:
            print(f"F-test  posterior: F={fp['F']:.1f} p={fp['p']:.2e}"
                  f"   token: F={ft['F']:.1f} p={ft['p']:.2e}"
                  f"   logit: F={fl['F']:.1f} p={fl['p']:.2e}")
        else:
            print(f"F-test  posterior: F={fp['F']:.1f} p={fp['p']:.2e}"
                  f"   token: F={ft['F']:.1f} p={ft['p']:.2e}")
        print(f"Condition number: {diag['condition_number']:.1f}")

        dd = diag["design_diagnostics"]
        _arrow = "\u2194"
        _r2_rest_hdr = "R" + _r2 + " from rest"
        _pw_hdr = "Pairwise R" + _r2 + " between groups:"
        print(f"\n  Design matrix collinearity:")
        print(f"    {'Group':<12} {'dims':>5} {'VIF':>10} {'GVIF^(1/2p)':>12} {_r2_rest_hdr:>14}")
        print(f"    {'-' * 55}")
        groups = ["posterior", "token"] + (["logit"] if has_logit else [])
        for grp in groups:
            ndim = dd["n_features"][grp]
            vif_val = dd["vif"][grp]
            gvif_val = dd["gvif_adj"][grp]
            r2_rest = dd["r2_from_rest"][grp]
            print(f"    {grp:<12} {ndim:>5d} {vif_val:>10.2f} {gvif_val:>12.4f} {r2_rest:>14.4f}")
        if has_pos:
            d_pos_val = dd["n_features"].get("position", 0)
            print(f"    {'position':<12} {d_pos_val:>5d}       (nuisance — VIF not computed)")
        print(f"\n    {_pw_hdr}")
        print(f"      post{_arrow}tok  = {dd['pairwise_r2']['post_tok']:.4f}")
        if has_logit:
            print(f"      post{_arrow}logit= {dd['pairwise_r2']['post_logit']:.4f}")
            print(f"      tok{_arrow}logit = {dd['pairwise_r2']['tok_logit']:.4f}")

        if diag["mlp_val_r2"] is not None:
            gap = diag["mlp_val_r2"] - joint_s["va_r2"]
            _mlp = "MLP val R" + _r2
            print(f"{_mlp}: {diag['mlp_val_r2']:.4f}  (linear gap = {gap:.4f})")
        if geometry is not None:
            sa = geometry["subspace_angles"]
            cc = geometry["component_cosine"]
            _deg = "\u00b0"
            print(f"Subspace angles (task vs token): "
                  f"mean={sa['mean_angle_deg']:.1f}{_deg}  "
                  f"min={sa['min_angle_deg']:.1f}{_deg}  "
                  f"max_cos={sa['max_cos']:.4f}  "
                  f"(rank {sa['rank_task']} vs {sa['rank_token']})")
            print(f"Component cos(task, token):      "
                  f"mean={cc['mean']:.4f}  "
                  f"std={cc['std']:.4f}  "
                  f"|mean|={cc['abs_mean']:.4f}")
            rc = geometry["row_cosine"]
            print(f"Row cos(W_task, W_tok):          "
                  f"max|cos|={rc['max_abs']:.4f}  "
                  f"mean|cos|={rc['mean_abs']:.4f}  "
                  f"({rc['matrix'].shape[0]}x{rc['matrix'].shape[1]})")

    return results
