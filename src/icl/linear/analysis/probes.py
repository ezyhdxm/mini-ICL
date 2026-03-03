"""Probe training for linear regression analysis."""

import gc
from typing import Optional, Union

import numpy as np
import torch

import icl.utils.notebook_utils as nu
from icl.linear.analysis._helpers import (
    _temporary_linear_minor_task_setup,
    _temporary_task_attributes,
)
from icl.linear.linear_path_utils import load_model_task_config
from icl.linear.task_vecs import extract_hidden_multi
from icl.linear.task_variance import compute_task_variance_multi_layer, extract_plotting_data_multi_layer
from icl.linear.linear_ood_analysis import _create_eval_task_pool, _setup_eval_task, setup_device
from icl.linear.p1_variance import compute_p1_variance_multi_layer
from icl.linear.p1_variance import extract_plotting_data_multi_layer as extract_p1_plotting_data_multi_layer
from icl.utils.unified_path_finder import get_exp_name
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
    uniform_sampling: bool = True,
    sample_mode: str = "train",
    skip_baselines: bool = False,
    print_summary: bool = True,
    use_log_posterior: bool = False,
) -> dict:
    """Joint OLS: h = [φ(π), xₜ, ŷₜ] · [W_task; W_tok; W_logit] + b.

    φ is log or identity (controlled by *use_log_posterior*).
    Joint fitting ensures W_task directions are orthogonal to input/logit
    confounds (Frisch–Waugh–Lovell).

    Returns dict with fitted weights, R², partial R², F-tests, and
    design-matrix collinearity diagnostics (VIF, condition number).
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
    ) as (include_minor, n_tasks_total):
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

        # For each batch b = 1, ..., n_batches, sample sequences
        #   X_b in R^{B x T x d},  Y_b in R^{B x T}
        # and collect three quantities at selected positions p_1, ..., p_P:
        #
        #   1) Oracle filtering posterior (from task_posterior_over_time):
        #        pi_{b,t,k} = p(z=k | X_{1:t}, Y_{1:t-1})    shape (B, T, K)
        #      then select positions:  pi_b[p_j, :]             -> (B, P, K)
        #
        #   2) Hidden states from layer l at the x_t token positions:
        #        h_{b,j} = h^{(l)}_{s(p_j)}                    -> (B, P, D)
        #      where s(p) = 2p maps point index to sequence index.
        #
        #   3) Input vectors at selected positions:
        #        x_{b,j} = X_{b, p_j}                          -> (B, P, d)
        #
        # These are accumulated across batches and later used to fit
        # the joint model:
        #   h  ~  [phi(pi), x, logit] @ [W_task; W_tok; W_logit] + b
        # where phi is identity or log.  The joint fit controls for
        # token and logit information, yielding unbiased task directions.
        with _temporary_task_attributes(train_task, batch_size=B):
            for batch_idx in range(n_batches):
                demo_data, demo_target = _sample_batch_for_mode(batch_idx)
                demo_data = demo_data.to(device)
                demo_target = demo_target.to(device)

                # pi_{b,t,k} = p(z=k | X_{1:t}, Y_{1:t-1})
                post = task_posterior_over_time_linear_regression(
                    train_task,
                    demo_data,
                    demo_target,
                    include_minor=include_minor,
                )  # (B, T, K)

                # h_{b,j} = hidden state at sequence position s(p_j)
                cache = {}
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

    hiddens_all = torch.cat(all_hiddens, dim=0)      # (N, P, D)
    post_all = torch.cat(all_posteriors, dim=0)      # (N, P, K_full)
    x_all = torch.cat(all_x_tokens, dim=0)           # (N, P, n_dims)
    logit_all = torch.cat(all_logits, dim=0)         # (N, P, 1)

    # Keep only the major-task posterior columns for fitting.
    # The full posterior is computed correctly (Bayesian over all tasks),
    # but minor-task columns add noise and overparameterize the design matrix.
    n_major = train_task.n_tasks
    post_all = post_all[:, :, :n_major]  # (N, P, K_major)

    # Split on sequence index (dim-0) BEFORE flattening so that all
    # positions from a given sequence stay together.  This avoids
    # leaking correlated positions across the train/val boundary.
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

    if use_log_posterior:
        X_main_tr = torch.log(post_tr + 1e-10)
        X_main_va = torch.log(post_va + 1e-10)
    else:
        X_main_tr, X_main_va = post_tr, post_va
    X_tok_tr, X_tok_va = x_tr, x_va
    X_logit_tr, X_logit_va = logit_tr, logit_va

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

    # ---- Primary fit: joint [posterior, x_t, logit_t] -> hidden ----
    # h = [phi(pi), x_t, logit_t] @ [W_task; W_tok; W_logit] + b
    # By Frisch-Waugh-Lovell, W_task controls for token and logit info,
    # giving unbiased task directions.
    X_joint_tr = torch.cat([X_main_tr, X_tok_tr, X_logit_tr], dim=1)
    X_joint_va = torch.cat([X_main_va, X_tok_va, X_logit_va], dim=1)
    W_joint, b_joint, joint_s = _fit_ols(X_joint_tr, Ytr, X_joint_va, Yva)

    d_main = X_main_tr.shape[1]
    d_tok = X_tok_tr.shape[1]
    d_logit = X_logit_tr.shape[1]
    W_task = W_joint[:d_main, :]
    W_tok_block = W_joint[d_main:d_main + d_tok, :]
    W_logit_block = W_joint[d_main + d_tok:, :]

    # ---- Marginal and pairwise fits (for partial R² and F-test) ----
    _, _, pi_s = _fit_ols(X_main_tr, Ytr, X_main_va, Yva)
    _, _, tok_s = _fit_ols(X_tok_tr, Ytr, X_tok_va, Yva)
    _, _, logit_s = _fit_ols(X_logit_tr, Ytr, X_logit_va, Yva)

    X_post_tok_tr = torch.cat([X_main_tr, X_tok_tr], dim=1)
    X_post_tok_va = torch.cat([X_main_va, X_tok_va], dim=1)
    _, _, post_tok_s = _fit_ols(X_post_tok_tr, Ytr, X_post_tok_va, Yva)

    X_post_logit_tr = torch.cat([X_main_tr, X_logit_tr], dim=1)
    X_post_logit_va = torch.cat([X_main_va, X_logit_va], dim=1)
    _, _, post_logit_s = _fit_ols(X_post_logit_tr, Ytr, X_post_logit_va, Yva)

    X_tok_logit_tr = torch.cat([X_tok_tr, X_logit_tr], dim=1)
    X_tok_logit_va = torch.cat([X_tok_va, X_logit_va], dim=1)
    _, _, tok_logit_s = _fit_ols(X_tok_logit_tr, Ytr, X_tok_logit_va, Yva)

    # ── Partial R² (validation) ──────────────────────────────────────
    # Partial R² isolates the *unique* variance explained by a feature
    # group after controlling for all other groups:
    #
    #   ΔR²(A | rest) = (R²_full − R²_{rest only}) / (1 − R²_{rest only})
    #
    # Numerator = variance uniquely attributable to A.
    # Denominator = residual variance available for A to explain.
    # Values near 0 → A is redundant given the others; near 1 → essential.
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

    # ── Incremental F-test (training data) ──────────────────────────
    # Tests H₀: the q extra features in group A have zero coefficients.
    #
    #   F = [(SS_reduced − SS_full) / q] / [SS_full / (n − p − 1)]
    #
    # SS_reduced = residual sum of squares without group A.
    # SS_full    = residual sum of squares with all groups.
    # q          = number of features in group A (numerator df).
    # n − p − 1  = residual df in the full model (denominator df).
    # Under H₀, F ~ F(q, n−p−1).
    n_tr = Ytr.shape[0]
    p_full = d_main + d_tok + d_logit
    df_den = n_tr - p_full - 1

    def _f_test(ss_reduced, ss_full, q, df_d):
        if df_d <= 0 or ss_full <= 0:
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

    # ── Design matrix collinearity diagnostics ──────────────────────
    # Condition number κ([X,1]) = σ_max / σ_min of the augmented design
    # matrix.  κ >> 1 indicates near-collinearity; rule of thumb: κ > 30
    # is concerning, κ > 1000 is severe.
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

    vif_post, r2_post_from_rest = _group_vif(
        X_main_tr, torch.cat([X_tok_tr, X_logit_tr], dim=1))
    vif_tok, r2_tok_from_rest = _group_vif(
        X_tok_tr, torch.cat([X_main_tr, X_logit_tr], dim=1))
    vif_logit, r2_logit_from_rest = _group_vif(
        X_logit_tr, torch.cat([X_main_tr, X_tok_tr], dim=1))

    # GVIF^(1/(2p)):  When a group has p > 1 columns, the raw GVIF
    # grows with p.  Raising to 1/(2p) makes it comparable to a
    # scalar √VIF, so the same threshold (≈ √10 ≈ 3.2) applies.
    gvif_post = vif_post ** (1.0 / (2 * d_main)) if d_main > 0 else float("nan")
    gvif_tok = vif_tok ** (1.0 / (2 * d_tok)) if d_tok > 0 else float("nan")
    gvif_logit = vif_logit ** (1.0 / (2 * d_logit)) if d_logit > 0 else float("nan")

    # Pairwise correlation between group predictions (how much each group
    # is linearly related to another, ignoring the third)
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
                        "total": d_main + d_tok + d_logit},
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
    }

    # ---- Optional heavier diagnostics (MLP + geometry) ----
    geometry = None

    if not skip_baselines:
        diagnostics["mlp_val_r2"] = _fit_mlp_r2(
            X_joint_tr, Ytr, X_joint_va, Yva,
        )

        # ── Principal angles between task and token subspaces ────────
        # Given two subspaces S_task = col(W_task^T) and S_tok = col(W_tok^T)
        # in ℝ^D, the principal angles θ₁ ≤ θ₂ ≤ … measure their mutual
        # alignment.  They satisfy:
        #
        #   cos θᵢ = σᵢ(Q_taskᵀ Q_tok)
        #
        # where Q_task, Q_tok are orthonormal bases (from SVD).
        # If all angles ≈ 90° the subspaces are nearly orthogonal;
        # small angles mean the subspaces share similar directions.
        #
        # Note: posterior features sum to 1, so W_task has rank ≤ K−1.
        # Rank truncation avoids a noise column that corrupts the angles.
        eps = 1e-10
        rank_tol = 1e-5
        Wt_f = W_task.T.float()   # (D, K)
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

        # ── Per-sample cosine between task and token contributions ───
        # For sample i, the fitted model decomposes h ≈ πW + xW_tok + …
        # Task contribution:  c_task(i) = πᵢ W_task   ∈ ℝ^D
        # Token contribution: c_tok(i)  = xᵢ W_tok    ∈ ℝ^D
        # cos(i) = ⟨c_task, c_tok⟩ / (‖c_task‖ ‖c_tok‖)
        # Values near 0 → the two components point in different directions.
        c_task_va = (X_main_va @ W_task).float()  # (n_val, D)
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

        geometry = {
            "joint_train_mse": joint_s["tr_mse"],
            "joint_val_mse": joint_s["va_mse"],
            "joint_train_r2": joint_s["tr_r2"],
            "joint_val_r2": joint_s["va_r2"],
            "subspace_angles": subspace_angles,
            "component_cosine": component_cosine,
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
        _r2 = "\u00b2"
        print(f"\n=== Fit Summary (layer {layer}, mode={sample_mode!r}) ===")
        print(f"{'Model':<25} {'Train R' + _r2:>10} {'Val R' + _r2:>10} {'Val MSE':>12}")
        print("-" * 59)
        print(f"{'Joint (task+tok+logit)':<25} {joint_s['tr_r2']:>10.4f} {joint_s['va_r2']:>10.4f} {joint_s['va_mse']:>12.6f}")
        print(f"{'Post+tok (no logit)':<25} {post_tok_s['tr_r2']:>10.4f} {post_tok_s['va_r2']:>10.4f} {post_tok_s['va_mse']:>12.6f}")
        print(f"{'Posterior only':<25} {pi_s['tr_r2']:>10.4f} {pi_s['va_r2']:>10.4f} {pi_s['va_mse']:>12.6f}")
        print(f"{'Token only':<25} {tok_s['tr_r2']:>10.4f} {tok_s['va_r2']:>10.4f} {tok_s['va_mse']:>12.6f}")
        print(f"{'Logit only':<25} {logit_s['tr_r2']:>10.4f} {logit_s['va_r2']:>10.4f} {logit_s['va_mse']:>12.6f}")
        print()
        _pr2 = "Partial R" + _r2
        print(f"{_pr2}  posterior|rest = {diag['partial_r2_posterior']:.4f}"
              f"    token|rest = {diag['partial_r2_token']:.4f}"
              f"    logit|rest = {diag['partial_r2_logit']:.4f}")
        fp, ft, fl = diag["f_test_posterior"], diag["f_test_token"], diag["f_test_logit"]
        print(f"F-test  posterior: F={fp['F']:.1f} p={fp['p']:.2e}"
              f"   token: F={ft['F']:.1f} p={ft['p']:.2e}"
              f"   logit: F={fl['F']:.1f} p={fl['p']:.2e}")
        print(f"Condition number: {diag['condition_number']:.1f}")

        dd = diag["design_diagnostics"]
        _arrow = "\u2194"
        _r2_rest_hdr = "R" + _r2 + " from rest"
        _pw_hdr = "Pairwise R" + _r2 + " between groups:"
        print(f"\n  Design matrix collinearity (layer-independent):")
        print(f"    {'Group':<12} {'dims':>5} {'VIF':>10} {'GVIF^(1/2p)':>12} {_r2_rest_hdr:>14}")
        print(f"    {'-' * 55}")
        for grp in ("posterior", "token", "logit"):
            ndim = dd["n_features"][grp]
            vif_val = dd["vif"][grp]
            gvif_val = dd["gvif_adj"][grp]
            r2_rest = dd["r2_from_rest"][grp]
            print(f"    {grp:<12} {ndim:>5d} {vif_val:>10.2f} {gvif_val:>12.4f} {r2_rest:>14.4f}")
        print(f"\n    {_pw_hdr}")
        print(f"      post{_arrow}tok  = {dd['pairwise_r2']['post_tok']:.4f}")
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

    return results


def probe_gaussian_posterior(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples_major: int = 1000,
    n_samples_ood: int = 1000,
    n_ood: int = 30,
    step: Optional[int] = None,
    positions: Optional[list] = None,
    p_gaussian: Optional[float] = None,
    learning_rate: float = 0.01,
    num_epochs: int = 100,
    validation_split: float = 0.2,
    verbose: bool = False,
) -> dict:
    """Probe whether h encodes the (K+1)-way posterior with a Gaussian hypothesis.

    Prior:  p(z) = Σₖ πₖ δ(z−wₖ)  +  π_{K+1} N(z; 0, τ²I)
    Oracle: P(z=k|X,Y) ∝ πₖ ∏ₜ N(yₜ; wₖᵀxₜ, σ²)        for k ≤ K
            P(z=K+1|X,Y) ∝ π_{K+1} N(Y; 0, σ²I + τ²XXᵀ)

    Trains a softmax probe  p̂(z=k|hₜ) = softmax(Whₜ + b)ₖ  by minimising
    KL[p_oracle ‖ p̂] on in-distribution + OOD sequences.

    Returns per-split KL, oracle vs predicted P(z=K+1), and trained probe.
    """
    from torch import nn
    import torch.optim as optim
    from icl.linear.linear_path_utils import load_model_task_config
    from icl.linear.analysis.posterior import task_posterior_with_gaussian_linear_regression
    from icl.linear.linear_ood_analysis import (
        _create_eval_task_pool,
        _setup_eval_task,
    )
    from icl.linear.analysis._helpers import _temporary_task_attributes

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
    n_embd = int(config.model.n_embd)

    eval_task_pool, _ = _create_eval_task_pool(
        train_task, K=n_ood, include_minor=False,
        device=device, n_minor=0,
    )
    ood_task = _setup_eval_task(config, eval_task_pool, B, device)
    ood_task.noise_scale = float(train_task.noise_scale)

    if positions is None:
        positions = list(range(n_points))
    seq_pos_tensor = torch.tensor([2 * p for p in positions], device=device, dtype=torch.long)

    Kmaj = int(train_task.n_tasks)
    K_plus_1 = Kmaj + 1

    def _collect(task_obj, n_samples, is_eval):
        """Collect hidden states and oracle (K+1)-way posteriors."""
        all_h, all_post = [], []
        n_batches = max(1, (n_samples + B - 1) // B)
        with _temporary_task_attributes(task_obj, batch_size=B):
            for bi in range(n_batches):
                demo_data, _, demo_target = task_obj.sample_batch(
                    step=bi + 55555, is_eval=is_eval,
                )
                demo_data = demo_data.to(device)
                demo_target = demo_target.to(device)

                cache = {}
                def hook_fn(module, inp, out, _cache=cache):
                    h = out if torch.is_tensor(out) else out[0]
                    _cache["h"] = h.index_select(1, seq_pos_tensor).detach()

                handle = model.transformer.blocks[layer].attn_block.register_forward_hook(hook_fn)
                try:
                    with torch.no_grad():
                        model(demo_data, demo_target)
                    h = cache["h"]  # (B_cur, P, D)
                finally:
                    handle.remove()

                post = task_posterior_with_gaussian_linear_regression(
                    train_task, demo_data, demo_target,
                    include_minor=False, p_gaussian=p_gaussian,
                )  # (B_cur, K+1)
                post_expanded = post.unsqueeze(1).expand(-1, len(positions), -1)

                all_h.append(h.cpu())
                all_post.append(post_expanded.cpu())
                del demo_data, demo_target, h, post, post_expanded
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        H = torch.cat(all_h, 0).reshape(-1, n_embd).float()
        P = torch.cat(all_post, 0).reshape(-1, K_plus_1).float()
        return H, P

    if verbose:
        logger.info(f"[gauss-probe] Collecting major hiddens (layer {layer}) ...")
    h_maj, post_maj = _collect(train_task, n_samples_major, is_eval=False)
    if verbose:
        logger.info("[gauss-probe] Collecting OOD hiddens ...")
    h_ood, post_ood = _collect(ood_task, n_samples_ood, is_eval=True)

    if verbose:
        logger.info(
            f"[gauss-probe] Major: {h_maj.shape[0]} samples, "
            f"mean P(gauss)={post_maj[:, -1].mean():.4f}  |  "
            f"OOD: {h_ood.shape[0]} samples, "
            f"mean P(gauss)={post_ood[:, -1].mean():.4f}"
        )

    H_all = torch.cat([h_maj, h_ood], dim=0)
    P_all = torch.cat([post_maj, post_ood], dim=0)
    is_ood_mask = torch.cat([
        torch.zeros(h_maj.shape[0], dtype=torch.bool),
        torch.ones(h_ood.shape[0], dtype=torch.bool),
    ], dim=0)
    N = H_all.shape[0]

    perm = torch.randperm(N)
    n_train = int(N * (1 - validation_split))
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    H_train, P_train = H_all[train_idx], P_all[train_idx]
    H_val, P_val = H_all[val_idx], P_all[val_idx]
    ood_mask_val = is_ood_mask[val_idx]

    probe = nn.Sequential(
        nn.Linear(n_embd, K_plus_1, bias=True),
        nn.Softmax(dim=-1),
    ).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=learning_rate)
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    train_bs = min(4096, n_train)

    def _batched_kl(probe_model, H, P_target):
        """Compute mean KL over batches."""
        losses = []
        for i in range(0, H.shape[0], train_bs):
            end = min(i + train_bs, H.shape[0])
            pred = probe_model(H[i:end].to(device))
            losses.append(kl_loss_fn(torch.log(pred + 1e-10), P_target[i:end].to(device)).item())
        return sum(losses) / len(losses)

    loss_history, val_loss_history = [], []

    for epoch in range(num_epochs):
        probe.train()
        optimizer.zero_grad()
        for i in range(0, n_train, train_bs):
            end = min(i + train_bs, n_train)
            pred = probe(H_train[i:end].to(device))
            loss = kl_loss_fn(torch.log(pred + 1e-10), P_train[i:end].to(device)) * (end - i) / n_train
            loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            loss_history.append(_batched_kl(probe, H_train, P_train))
            val_loss_history.append(_batched_kl(probe, H_val, P_val))

        if verbose and (epoch + 1) % 20 == 0:
            logger.info(
                f"[gauss-probe] epoch {epoch+1}/{num_epochs}  "
                f"train KL={loss_history[-1]:.4f}  val KL={val_loss_history[-1]:.4f}"
            )

    probe.eval()
    with torch.no_grad():
        pred_val = torch.cat([
            probe(H_val[i:min(i + train_bs, len(val_idx))].to(device)).cpu()
            for i in range(0, len(val_idx), train_bs)
        ], dim=0)

    maj_mask_val = ~ood_mask_val
    pred_maj_val, pred_ood_val = pred_val[maj_mask_val], pred_val[ood_mask_val]
    true_maj_val, true_ood_val = P_val[maj_mask_val], P_val[ood_mask_val]

    def _safe_mean(t, col=-1):
        return t[:, col].mean().item() if t.shape[0] > 0 else float("nan")

    def _safe_numpy(t, col=-1):
        return t[:, col].numpy() if t.shape[0] > 0 else np.array([])

    def _kl_subset(pred_sub, true_sub):
        if pred_sub.shape[0] == 0:
            return float("nan")
        with torch.no_grad():
            return kl_loss_fn(
                torch.log(pred_sub.to(device) + 1e-10),
                true_sub.to(device),
            ).item()

    probe = probe.cpu()

    results = {
        "layer": layer,
        "n_tasks": Kmaj,
        "loss_history": loss_history,
        "val_loss_history": val_loss_history,
        "final_train_kl": loss_history[-1],
        "final_val_kl": val_loss_history[-1],
        "kl_major_val": _kl_subset(pred_maj_val, true_maj_val),
        "kl_ood_val": _kl_subset(pred_ood_val, true_ood_val),
        "mean_pred_pgauss_major": _safe_mean(pred_maj_val),
        "mean_pred_pgauss_ood": _safe_mean(pred_ood_val),
        "mean_true_pgauss_major": _safe_mean(true_maj_val),
        "mean_true_pgauss_ood": _safe_mean(true_ood_val),
        "pred_pgauss_major": _safe_numpy(pred_maj_val),
        "pred_pgauss_ood": _safe_numpy(pred_ood_val),
        "true_pgauss_major": _safe_numpy(true_maj_val),
        "true_pgauss_ood": _safe_numpy(true_ood_val),
        "probe": probe,
    }

    if verbose:
        logger.info(
            f"[gauss-probe] layer {layer} done.  "
            f"val KL={results['final_val_kl']:.4f}  |  "
            f"P(gauss) pred major={results['mean_pred_pgauss_major']:.4f}  "
            f"OOD={results['mean_pred_pgauss_ood']:.4f}  |  "
            f"P(gauss) true major={results['mean_true_pgauss_major']:.4f}  "
            f"OOD={results['mean_true_pgauss_ood']:.4f}"
        )

    return results


# ---------------------------------------------------------------------------
# Padded-sequence probe functions (from unified_interface.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_token_conditioned_hiddens(
    exp_name: str,
    layers: Optional[list] = None,
    chunk_size: int = 16,
    step: Optional[int] = None,
    positions_of_interest: Optional[list] = None,
    max_unique_tokens: Optional[int] = None,
    batch_size: int = 16,
    n_minor: int = 64,
    n_ood: int = 30,
    verbose: bool = False,
) -> tuple:
    """Get token-conditioned hidden representations (pad-mode aware).

    For each position of interest, fixes every observed data token at that
    position and extracts the hidden at the **data-token** sequence position
    (not the PAD position).  Works with any ``pad`` mode.

    Returns
    -------
    all_hiddens : torch.Tensor
        ``(L, n_positions, n_unique_tokens, n_tasks, batch_size, n_embd)``
    demo_data : torch.Tensor
        ``(batch_size, n_points, n_dims)``
    token_info : dict
    """
    from icl.linear.analysis._helpers import _point_to_seq_pos

    _, train_task, config = load_model_task_config(exp_name)
    if step is None:
        step = config.training.total_steps

    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )

    device = setup_device(None)
    eval_task_pool, k_minor = _create_eval_task_pool(
        train_task, K=n_ood, include_minor=True,
        device=device, n_minor=n_minor,
    )
    eval_task = _setup_eval_task(config, eval_task_pool, batch_size, device)
    eval_task.batch_size = batch_size

    if layers is None:
        layers = list(range(config.model.n_layer))

    n_tasks = eval_task.task_pool.shape[0]
    n_points = config.task.n_points
    n_dims = config.task.n_dims
    n_embd = config.model.n_embd
    L = len(layers)
    pad_mode = getattr(model, "pad", "mapsto")

    if positions_of_interest is None:
        positions_of_interest = list(range(n_points))
    else:
        positions_of_interest = list(positions_of_interest)
        if not all(0 <= p < n_points for p in positions_of_interest):
            raise ValueError(f"All positions must be in [0, {n_points - 1}]")
    n_positions = len(positions_of_interest)

    if verbose:
        logger.info(
            f"get_token_conditioned_hiddens: layers={layers}, B={batch_size}, "
            f"n_tasks={n_tasks}, pad={pad_mode}, positions={positions_of_interest}"
        )

    demo_data = eval_task.sample_data(step=step).to(device)

    unique_tokens_by_pos = {}
    for p in positions_of_interest:
        tokens = demo_data[:, p, :]
        if max_unique_tokens is not None and tokens.shape[0] > max_unique_tokens:
            idx = torch.randperm(tokens.shape[0], device=device)[:max_unique_tokens]
            tokens = tokens[idx]
        unique_tokens_by_pos[p] = tokens

    max_n_unique = max(t.shape[0] for t in unique_tokens_by_pos.values())

    results_by_layer = {l: {p: {} for p in positions_of_interest} for l in layers}

    for pos_idx in positions_of_interest:
        unique_tokens = unique_tokens_by_pos[pos_idx]
        extract_seq_pos = _point_to_seq_pos(pad_mode, pos_idx)

        for token_idx, fixed_token in enumerate(unique_tokens):
            for i in range(0, n_tasks, chunk_size):
                chunk_end = min(i + chunk_size, n_tasks)
                chunk_n = chunk_end - i

                mod_data = demo_data.clone()
                mod_data[:, pos_idx, :] = fixed_token.unsqueeze(0).expand(batch_size, -1)

                mod_target = eval_task.evaluate(
                    mod_data,
                    eval_task.task_pool[i:chunk_end].squeeze(-1).T,
                    step=step,
                )

                mod_data_rep = (
                    mod_data.unsqueeze(0)
                    .expand(chunk_n, batch_size, n_points, -1)
                    .reshape(-1, n_points, n_dims)
                )
                if mod_target.ndim == 3:
                    mod_target = mod_target.permute(2, 0, 1).reshape(-1, n_points)
                else:
                    mod_target = (
                        mod_target.unsqueeze(0)
                        .expand(chunk_n, -1, -1)
                        .reshape(-1, n_points)
                    )

                h = extract_hidden_multi(
                    model=model, demo_data=mod_data_rep,
                    demo_target=mod_target, layers=layers,
                    task_pos=extract_seq_pos,
                )

                for l_idx, l in enumerate(layers):
                    results_by_layer[l][pos_idx].setdefault(token_idx, []).append(
                        h[l_idx].reshape(chunk_n, batch_size, n_embd)
                    )

    all_hiddens = torch.zeros(
        (L, n_positions, max_n_unique, n_tasks, batch_size, n_embd),
        dtype=demo_data.dtype, device=device,
    )
    for l_idx, l in enumerate(layers):
        for p_idx, p in enumerate(positions_of_interest):
            n_unique = unique_tokens_by_pos[p].shape[0]
            for t_idx in range(min(n_unique, max_n_unique)):
                chunks = results_by_layer[l][p].get(t_idx, [])
                if chunks:
                    combined = torch.cat(chunks, dim=0)[:n_tasks]
                    all_hiddens[l_idx, p_idx, t_idx, :combined.shape[0]] = combined

    token_info = {
        "positions": positions_of_interest,
        "unique_tokens": {
            p: tok.cpu().numpy().tolist()
            for p, tok in unique_tokens_by_pos.items()
        },
        "token_type": "data",
        "n_unique_tokens": {
            p: tok.shape[0] for p, tok in unique_tokens_by_pos.items()
        },
    }

    model.cpu(); del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return all_hiddens.detach().cpu(), demo_data, token_info


def get_task_variance(
    exp_name: str,
    layers: Optional[list] = None,
    chunk_size: int = 16,
    step: Optional[int] = None,
    positions_of_interest: Optional[list] = None,
    batch_size: int = 16,
    n_minor: int = 64,
    n_ood: int = 30,
    verbose: bool = False,
    eps: float = 1e-8,
) -> tuple:
    """Compute task variance (P2) for the linear regression task.

    Extracts hiddens at **data-token** positions (pad-mode aware) and
    computes the variance of batch-averaged hiddens across tasks.

    Returns
    -------
    all_hiddens : torch.Tensor
        ``(L, n_tasks, n_points, batch_size, n_embd)``
    demo_data : torch.Tensor
        ``(batch_size, n_points, n_dims)``
    results_dict : dict
        Layer number -> ``TaskVarianceResults``
    plotting_data : dict
    """
    from icl.linear.task_vecs import extract_hidden_multi
    from icl.linear.analysis._helpers import _task_positions

    _, train_task, config = load_model_task_config(exp_name)

    if step is None:
        step = config.training.total_steps

    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )

    device = setup_device(None)
    eval_task_pool, k_minor = _create_eval_task_pool(
        train_task, K=n_ood, include_minor=True,
        device=device, n_minor=n_minor,
    )
    eval_task = _setup_eval_task(config, eval_task_pool, batch_size, device)
    eval_task.batch_size = batch_size

    if layers is None:
        layers = list(range(config.model.n_layer))

    n_points = config.task.n_points
    n_embd = config.model.n_embd
    n_tasks = eval_task.task_pool.shape[0]
    L = len(layers)
    pad_mode = getattr(model, "pad", "mapsto")
    task_pos = _task_positions(pad_mode, n_points, device=device)

    if verbose:
        logger.info(
            f"get_task_variance: layers={layers}, B={batch_size}, "
            f"n_tasks={n_tasks}, pad={pad_mode}"
        )

    all_hiddens = torch.empty(
        (L, n_tasks, n_points, batch_size, n_embd),
        dtype=torch.float32, device="cpu",
    )
    demo_data = eval_task.sample_data(step=step)

    for i in range(0, n_tasks, chunk_size):
        chunk_end = min(i + chunk_size, n_tasks)
        chunk_n = chunk_end - i

        demo_data_rep = (
            demo_data.unsqueeze(0)
            .expand(chunk_n, batch_size, n_points, -1)
            .reshape(-1, n_points, demo_data.size(-1))
        )

        demo_target = eval_task.evaluate(
            demo_data,
            eval_task.task_pool[i:chunk_end].squeeze(-1).T,
            step=step,
        )
        if demo_target.ndim == 3:
            demo_target = demo_target.permute(2, 0, 1).reshape(-1, n_points)

        h = extract_hidden_multi(
            model=model, demo_data=demo_data_rep,
            demo_target=demo_target, layers=layers, task_pos=task_pos,
        )
        h = h.reshape(L, chunk_n, batch_size, n_points, n_embd)
        h = h.permute(0, 1, 3, 2, 4)
        all_hiddens[:, i:chunk_end] = h.cpu()

        del h
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_hiddens = all_hiddens.detach()

    results_dict = compute_task_variance_multi_layer(
        all_hiddens=all_hiddens,
        positions_of_interest=positions_of_interest,
        layers=layers,
        eps=eps,
    )
    plotting_data = extract_plotting_data_multi_layer(results_dict)

    model.cpu(); del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return all_hiddens, demo_data, results_dict, plotting_data


def train_linear_softmax_posterior_predictor_linear(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples: int = 1000,
    step: Optional[int] = None,
    learning_rate: float = 0.01,
    num_epochs: int = 100,
    verbose: bool = False,
    position: Union[int, list] = -1,
    validation_split: float = 0.2,
    uniform_sampling: bool = True,
    skip_baselines: bool = False,
    sample_mode: str = "train",
) -> dict:
    """
    Train a linear softmax model to predict task posteriors from hidden representations.
    
    For linear regression task:
    1. Gets samples from the task (data and targets)
    2. Computes cumulative task posteriors using task_posterior_over_time_linear_regression
    3. Extracts hidden representations at the specified layer and position(s)
    4. Trains a linear softmax model to map hidden representations to posteriors
    5. Reports training loss
    
    Parameters:
    -----------
    exp_name : str
        Experiment name (e.g., "train_...")
    layer : int
        Layer index to extract hidden representations from
    B : int, default=64
        Batch size for sampling
    n_samples : int, default=1000
        Total number of samples to use for training
    step : int, optional
        Step for checkpoint loading. If None, uses final checkpoint.
    learning_rate : float, default=0.01
        Learning rate for training the linear model
    num_epochs : int, default=100
        Number of training epochs
    verbose : bool, default=False
        Whether to print progress messages
    position : int or list, default=-1
        Position index(ices) to extract hidden representations from.
        - If int: single position index. -1 means the final position (after all data points).
        - If list: multiple position indices. Each position's hidden will be paired with the same posterior.
        For linear regression, positions are typically at 3*i+1 for padded sequences.
    validation_split : float, default=0.2
        Fraction of data to use for validation (between 0 and 1).
        The remaining fraction is used for training.
    uniform_sampling : bool, default=True
        If True, modifies p_minor to achieve uniform sampling across all tasks
        (each task has equal probability). If False, uses the original sampler's p_minor.
    skip_baselines : bool, default=False
        If True, skips training permutation and logits baselines to save time.
        Only trains the main model (hiddens -> posteriors).
    
    sample_mode : str, default="train"
        - ``"train"``: sample from train distribution (major + minor), K-way posterior target.
        - ``"major"``: sample from major tasks only, 3-way posterior target.
        - ``"major_gaussian"``: sample from train distribution (major + minor for diversity),
          (K_major+1)-way posterior target (3 major + 1 Gaussian "new task" hypothesis).
          Use ``uniform_sampling=True`` for uniform coverage over major + minor tasks.
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'final_loss': float, final training loss
        - 'final_val_loss': float, final validation loss
        - 'loss_history': list of training losses during training
        - 'val_loss_history': list of validation losses during training
        - 'model': the trained linear model
        - 'baseline_final_loss': float, final training loss for permutation baseline
        - 'baseline_final_val_loss': float, final validation loss for permutation baseline
        - 'baseline_loss_history': list of training losses for baseline (shuffled data)
        - 'baseline_val_loss_history': list of validation losses for baseline (shuffled data)
        - 'baseline_model': the trained baseline model (trained on shuffled data)
        - 'logits_baseline_final_loss': float, final training loss for logits baseline
        - 'logits_baseline_final_val_loss': float, final validation loss for logits baseline
        - 'logits_baseline_loss_history': list of training losses for logits baseline
        - 'logits_baseline_val_loss_history': list of validation losses for logits baseline
        - 'logits_baseline_model': the trained logits baseline model (trained on logits)
        - 'layer': int, the layer index used
        - 'n_tasks': int, number of tasks
    """
    from icl.linear.analysis.posterior import task_posterior_over_time_linear_regression
    from torch import nn
    import torch.optim as optim
    
    use_gaussian = sample_mode == "major_gaussian"
    if use_gaussian:
        from icl.linear.analysis.posterior import task_posterior_with_gaussian_linear_regression
    
    # Load config/model/task
    model_loaded, train_task, config = load_model_task_config(exp_name)
    
    if step is None:
        step = config.training.total_steps
    
    # Load model from checkpoint (may be different step than default)
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval()
    model.to(config.device)
    
    n_tasks = train_task.n_tasks
    if n_tasks <= 0:
        raise ValueError("This function requires a finite task pool (n_tasks > 0)")
    
    # Determine total number of tasks (major + minor if included)
    if sample_mode in ("major", "major_gaussian"):
        include_minor = False
    else:
        include_minor = train_task.n_minor_tasks > 0 and train_task.minor_pool is not None
    if use_gaussian:
        n_total_tasks = n_tasks + 1  # 3 major + 1 Gaussian
    elif include_minor:
        n_total_tasks = n_tasks + train_task.n_minor_tasks
    else:
        n_total_tasks = n_tasks
    
    # Optionally modify p_minor to achieve uniform sampling across all tasks.
    # For major_gaussian we still sample from the full train distribution
    # (major + minor) so the probe sees diversity in P(Z=K+1).
    original_p_minor = train_task.p_minor
    has_minor_pool = train_task.n_minor_tasks > 0 and train_task.minor_pool is not None
    if use_gaussian and has_minor_pool:
        train_task.p_minor = 0.25
        if verbose:
            logger.info(f"Modified p_minor from {original_p_minor:.6f} to {train_task.p_minor:.6f} for major_gaussian mode")
    elif uniform_sampling and include_minor and has_minor_pool:
        train_task.p_minor = train_task.n_minor_tasks / (train_task.n_tasks + train_task.n_minor_tasks)
        if verbose:
            logger.info(f"Modified p_minor from {original_p_minor:.6f} to {train_task.p_minor:.6f} for uniform task sampling")
    elif verbose and not uniform_sampling:
        logger.info(f"Using original p_minor: {original_p_minor:.6f} (not modified for uniform sampling)")
    
    # Normalize position to a list
    if isinstance(position, int):
        positions = [position]
    else:
        positions = list(position)
    
    if verbose:
        logger.info(f"Training linear softmax model to predict posteriors from layer {layer} hidden representations")
        logger.info(f"Number of tasks: {n_tasks} (major), {train_task.n_minor_tasks} (minor), Total: {n_total_tasks}")
        logger.info(f"Batch size: {B}, Total samples: {n_samples}, Positions: {positions}")
        if uniform_sampling:
            logger.info("Using uniform task sampling (modified p_minor)")
        else:
            logger.info("Using original sampler's p_minor (not modified)")
    
    # Collect samples, hiddens, logits, and posteriors
    device = config.device
    all_hiddens = []
    all_logits = []
    all_posteriors = []
    
    n_batches = (n_samples + B - 1) // B  # Ceiling division
    
    if verbose:
        logger.info(f"Collecting {n_batches} batches of data...")
    
    # Store original batch size and restore later
    original_batch_size = train_task.batch_size
    train_task.batch_size = B
    
    try:
        for batch_idx in range(n_batches):
            if sample_mode == "major":
                demo_data = train_task.sample_data(step=batch_idx).to(device)
                idx = torch.randint(0, train_task.n_tasks, (B,), device=demo_data.device)
                tasks = train_task.task_pool[idx]
                demo_target = train_task.evaluate(demo_data, tasks, step=batch_idx).to(device)
            else:
                # "train" and "major_gaussian" both sample from the full distribution
                demo_data, _, demo_target = train_task.sample_batch(step=batch_idx, is_eval=False)
                demo_data = demo_data.to(device)
                demo_target = demo_target.to(device)
            
            if use_gaussian:
                # (K+1)-way cumulative posterior: 3 major + Gaussian at each position
                n_pts_batch = demo_data.shape[1]
                point_pos_batch = [
                    (n_pts_batch - 1 if p == -1 else p) for p in positions
                ]
                post_list = []
                for pp in point_pos_batch:
                    data_t = demo_data[:, : pp + 1, :]
                    tgt_t = demo_target[:, : pp + 1]
                    post_t = task_posterior_with_gaussian_linear_regression(
                        train_task, data_t, tgt_t, include_minor=False,
                    )  # (B, K+1)
                    post_list.append(post_t)
                posteriors_expanded = torch.stack(post_list, dim=1)  # (B, len(positions), K+1)
            else:
                # K-way cumulative posterior
                posteriors_over_time = task_posterior_over_time_linear_regression(
                    train_task,
                    demo_data,
                    demo_target,
                    include_minor=include_minor,
                )  # (B, n_points, K)
            
            # Extract hidden representations at specified layer and positions
            cache = {}
            layer_module = model.transformer.blocks[layer].attn_block
            
            # Helper function to convert positions to indices
            def get_pos_indices(seq_length):
                pos_indices = []
                for pos in positions:
                    if pos == -1:
                        pos_indices.append(seq_length - 1)
                    else:
                        if pos >= seq_length:
                            raise ValueError(f"Position {pos} >= sequence length {seq_length}")
                        pos_indices.append(pos)
                return torch.tensor(pos_indices, device=device, dtype=torch.long)
            
            # Forward pass to get sequence length first
            with torch.no_grad():
                logits_full = model(demo_data, demo_target)  # (B, L, vocab_size)
                seq_len = logits_full.size(1)
            
            # Now we know seq_len, compute position indices
            pos_indices = get_pos_indices(seq_len)
            
            def hook_fn(module, inp, out):
                if torch.is_tensor(out):
                    # out: (B, L, D)
                    cache["hidden"] = out.index_select(dim=1, index=pos_indices).detach()  # (B, len(positions), D)
                elif isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                    cache["hidden"] = out[0].index_select(dim=1, index=pos_indices).detach()  # (B, len(positions), D)
                else:
                    raise RuntimeError(f"Unsupported hook output type: {type(out)}")
            
            handle = layer_module.register_forward_hook(hook_fn)
            try:
                with torch.no_grad():
                    # Forward pass again to extract hiddens (we already have logits)
                    _ = model(demo_data, demo_target)
                    # Extract logits at the same positions
                    logits_batch = logits_full.index_select(dim=1, index=pos_indices)  # (B, len(positions), vocab_size)
                hiddens_batch = cache["hidden"]  # (B, len(positions), D)
            finally:
                handle.remove()
            
            if not use_gaussian:
                # Select cumulative posteriors at requested point positions
                n_pts = demo_data.shape[1]
                point_pos = [
                    (n_pts - 1 if p == -1 else p) for p in positions
                ]
                point_pos_t = torch.tensor(point_pos, device=device, dtype=torch.long)
                posteriors_expanded = posteriors_over_time[:, point_pos_t, :]  # (B, len(positions), K)
            # else: posteriors_expanded already set in the Gaussian branch above
            
            # Move to CPU to save GPU memory
            all_hiddens.append(hiddens_batch.cpu())
            all_logits.append(logits_batch.cpu())
            all_posteriors.append(posteriors_expanded.cpu())
            
            # Clear GPU memory
            if use_gaussian:
                del demo_data, demo_target, posteriors_expanded, hiddens_batch, logits_batch, logits_full
            else:
                del demo_data, demo_target, posteriors_over_time, hiddens_batch, logits_batch, logits_full
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        # Restore original batch size and p_minor
        train_task.batch_size = original_batch_size
        if 'original_p_minor' in locals():
            train_task.p_minor = original_p_minor
    
    # Concatenate all batches
    hiddens_all = torch.cat(all_hiddens, dim=0)  # (n_samples, len(positions), D)
    logits_all = torch.cat(all_logits, dim=0)  # (n_samples, len(positions), vocab_size)
    posteriors_all = torch.cat(all_posteriors, dim=0)  # (n_samples, len(positions), K)
    
    # Reshape: flatten position dimension
    # hiddens: (n_samples * len(positions), D)
    # logits: (n_samples * len(positions), vocab_size)
    # posteriors: (n_samples * len(positions), K)
    n_samples_actual = hiddens_all.shape[0]
    n_positions = hiddens_all.shape[1]
    n_total = n_samples_actual * n_positions
    
    hiddens_flat = hiddens_all.reshape(n_total, -1)  # (n_total, D)
    logits_flat = logits_all.reshape(n_total, -1)  # (n_total, vocab_size)
    posteriors_flat = posteriors_all.reshape(n_total, -1)  # (n_total, K)
    
    D = hiddens_flat.shape[1]
    vocab_size = logits_flat.shape[1]
    T = posteriors_flat.shape[1]  # Number of tasks
    
    # Split into training and validation sets
    n_train = int(n_total * (1 - validation_split))
    indices = torch.randperm(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Use flattened tensors for indexing - keep on CPU to save GPU memory
    hiddens_train = hiddens_flat[train_indices].cpu()
    logits_train = logits_flat[train_indices].cpu()
    posteriors_train = posteriors_flat[train_indices].cpu()
    hiddens_val = hiddens_flat[val_indices].cpu()
    logits_val = logits_flat[val_indices].cpu()
    posteriors_val = posteriors_flat[val_indices].cpu()
    
    # Clean up intermediate tensors
    del hiddens_all, logits_all, posteriors_all, hiddens_flat, logits_flat, posteriors_flat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    if verbose:
        logger.info(f"Total data: {n_total} samples")
        logger.info(f"Training data: {n_train} samples ({100*(1-validation_split):.1f}%)")
        logger.info(f"Validation data: {len(val_indices)} samples ({100*validation_split:.1f}%)")
        logger.info(f"Training data shape: hiddens {hiddens_train.shape}, logits {logits_train.shape}, posteriors {posteriors_train.shape}")
    
    # Use mini-batch training to reduce GPU memory usage
    train_batch_size = min(4096, n_train)  # Use smaller batches for training
    val_batch_size = min(4096, len(val_indices))  # Use smaller batches for validation
    
    if verbose:
        logger.info(f"Using mini-batch training: train_batch_size={train_batch_size}, val_batch_size={val_batch_size}")
    
    # Create linear softmax model: hidden (D) -> logits (T) -> softmax -> posterior (T)
    linear_model = nn.Sequential(
        nn.Linear(D, T, bias=True),
        nn.Softmax(dim=-1)
    ).to(device)
    
    # Training setup
    optimizer = optim.Adam(linear_model.parameters(), lr=learning_rate)
    criterion = nn.KLDivLoss(reduction='batchmean')  # KL divergence for probability distributions
    
    loss_history = []
    val_loss_history = []
    
    if verbose:
        logger.info(f"Training linear model for {num_epochs} epochs...")
    
    # Training loop with mini-batches
    for epoch in range(num_epochs):
        # Training phase - use mini-batches
        linear_model.train()
        optimizer.zero_grad()
        
        train_losses = []
        # Process training data in mini-batches
        for i in range(0, n_train, train_batch_size):
            end_idx = min(i + train_batch_size, n_train)
            h_batch = hiddens_train[i:end_idx].to(device)
            p_batch = posteriors_train[i:end_idx].to(device)
            
            pred_posteriors_batch = linear_model(h_batch)
            log_pred_batch = torch.log(pred_posteriors_batch + 1e-10)
            batch_loss = criterion(log_pred_batch, p_batch) * (end_idx - i) / n_train  # Weight by batch size
            
            batch_loss.backward()
            train_losses.append(batch_loss.item() * n_train / (end_idx - i))  # Unweight for logging
            
            del h_batch, p_batch, pred_posteriors_batch, log_pred_batch, batch_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        optimizer.step()
        train_loss = sum(train_losses) / len(train_losses)
        
        # Validation phase - use mini-batches
        linear_model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(0, len(val_indices), val_batch_size):
                end_idx = min(i + val_batch_size, len(val_indices))
                h_batch = hiddens_val[i:end_idx].to(device)
                p_batch = posteriors_val[i:end_idx].to(device)
                
                pred_posteriors_batch = linear_model(h_batch)
                log_pred_batch = torch.log(pred_posteriors_batch + 1e-10)
                batch_loss = criterion(log_pred_batch, p_batch)
                val_losses.append(batch_loss.item())
                
                del h_batch, p_batch, pred_posteriors_batch, log_pred_batch, batch_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        val_loss = sum(val_losses) / len(val_losses)
        
        loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        if verbose and (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    final_loss = loss_history[-1]
    final_val_loss = val_loss_history[-1]
    
    if verbose:
        logger.info(f"Training completed. Final train loss: {final_loss:.6f}, Final val loss: {final_val_loss:.6f}")
    
    # Initialize baseline results with None/NaN if skipping
    final_baseline_loss = float('nan')
    final_baseline_val_loss = float('nan')
    baseline_loss_history = []
    baseline_val_loss_history = []
    baseline_model = None
    
    final_logits_baseline_loss = float('nan')
    final_logits_baseline_val_loss = float('nan')
    logits_baseline_loss_history = []
    logits_baseline_val_loss_history = []
    logits_baseline_model = None
    
    if not skip_baselines:
        # Permutation baseline: shuffle posteriors to break the pairing with hiddens
        if verbose:
            logger.info("Training permutation baseline (shuffled posteriors)...")
        
        # Create shuffled posteriors (shuffle independently for train and val) - keep on CPU
        posteriors_train_shuffled = posteriors_train[torch.randperm(n_train)].cpu()
        posteriors_val_shuffled = posteriors_val[torch.randperm(len(val_indices))].cpu()
        
        # Clear GPU cache before baseline training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Create a new model for the baseline
        baseline_model = nn.Sequential(
            nn.Linear(D, T, bias=True),
            nn.Softmax(dim=-1)
        ).to(device)
        
        # Training setup for baseline
        baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=learning_rate)
        
        baseline_loss_history = []
        baseline_val_loss_history = []
        
        # Training loop for baseline - use mini-batches
        for epoch in range(num_epochs):
            # Training phase
            baseline_model.train()
            baseline_optimizer.zero_grad()
            
            baseline_train_losses = []
            for i in range(0, n_train, train_batch_size):
                end_idx = min(i + train_batch_size, n_train)
                h_batch = hiddens_train[i:end_idx].to(device)
                p_batch = posteriors_train_shuffled[i:end_idx].to(device)
                
                pred_posteriors_batch = baseline_model(h_batch)
                log_pred_batch = torch.log(pred_posteriors_batch + 1e-10)
                batch_loss = criterion(log_pred_batch, p_batch) * (end_idx - i) / n_train
                
                batch_loss.backward()
                baseline_train_losses.append(batch_loss.item() * n_train / (end_idx - i))
                
                del h_batch, p_batch, pred_posteriors_batch, log_pred_batch, batch_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            baseline_optimizer.step()
            train_loss_baseline = sum(baseline_train_losses) / len(baseline_train_losses)
            
            # Validation phase
            baseline_model.eval()
            baseline_val_losses = []
            with torch.no_grad():
                for i in range(0, len(val_indices), val_batch_size):
                    end_idx = min(i + val_batch_size, len(val_indices))
                    h_batch = hiddens_val[i:end_idx].to(device)
                    p_batch = posteriors_val_shuffled[i:end_idx].to(device)
                    
                    pred_posteriors_batch = baseline_model(h_batch)
                    log_pred_batch = torch.log(pred_posteriors_batch + 1e-10)
                    batch_loss = criterion(log_pred_batch, p_batch)
                    baseline_val_losses.append(batch_loss.item())
                    
                    del h_batch, p_batch, pred_posteriors_batch, log_pred_batch, batch_loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            val_loss_baseline = sum(baseline_val_losses) / len(baseline_val_losses)
            
            baseline_loss_history.append(train_loss_baseline)
            baseline_val_loss_history.append(val_loss_baseline)
            
            if verbose and (epoch + 1) % 20 == 0:
                logger.info(f"  Baseline Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_baseline:.6f}, Val Loss: {val_loss_baseline:.6f}")
        
        final_baseline_loss = baseline_loss_history[-1]
        final_baseline_val_loss = baseline_val_loss_history[-1]
        
        if verbose:
            logger.info(f"Baseline completed. Final train loss: {final_baseline_loss:.6f}, Final val loss: {final_baseline_val_loss:.6f}")
            logger.info(f"Improvement over baseline - Train: {final_loss - final_baseline_loss:.6f}, Val: {final_val_loss - final_baseline_val_loss:.6f}")
        
        # Logits baseline: train a model to predict posteriors from logits
        if verbose:
            logger.info("Training logits baseline (predicting posteriors from logits)...")
        
        # Clear GPU cache before logits baseline training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Create a new model for logits baseline
        logits_baseline_model = nn.Sequential(
            nn.Linear(vocab_size, T, bias=True),
            nn.Softmax(dim=-1)
        ).to(device)
        
        # Training setup for logits baseline
        logits_baseline_optimizer = optim.Adam(logits_baseline_model.parameters(), lr=learning_rate)
        
        logits_baseline_loss_history = []
        logits_baseline_val_loss_history = []
        
        # Training loop for logits baseline - use mini-batches
        for epoch in range(num_epochs):
            # Training phase
            logits_baseline_model.train()
            logits_baseline_optimizer.zero_grad()
            
            logits_train_losses = []
            for i in range(0, n_train, train_batch_size):
                end_idx = min(i + train_batch_size, n_train)
                l_batch = logits_train[i:end_idx].to(device)
                p_batch = posteriors_train[i:end_idx].to(device)
                
                pred_posteriors_batch = logits_baseline_model(l_batch)
                log_pred_batch = torch.log(pred_posteriors_batch + 1e-10)
                batch_loss = criterion(log_pred_batch, p_batch) * (end_idx - i) / n_train
                
                batch_loss.backward()
                logits_train_losses.append(batch_loss.item() * n_train / (end_idx - i))
                
                del l_batch, p_batch, pred_posteriors_batch, log_pred_batch, batch_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            logits_baseline_optimizer.step()
            train_loss_logits = sum(logits_train_losses) / len(logits_train_losses)
            
            # Validation phase
            logits_baseline_model.eval()
            logits_val_losses = []
            with torch.no_grad():
                for i in range(0, len(val_indices), val_batch_size):
                    end_idx = min(i + val_batch_size, len(val_indices))
                    l_batch = logits_val[i:end_idx].to(device)
                    p_batch = posteriors_val[i:end_idx].to(device)
                    
                    pred_posteriors_batch = logits_baseline_model(l_batch)
                    log_pred_batch = torch.log(pred_posteriors_batch + 1e-10)
                    batch_loss = criterion(log_pred_batch, p_batch)
                    logits_val_losses.append(batch_loss.item())
                    
                    del l_batch, p_batch, pred_posteriors_batch, log_pred_batch, batch_loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            val_loss_logits = sum(logits_val_losses) / len(logits_val_losses)
            
            logits_baseline_loss_history.append(train_loss_logits)
            logits_baseline_val_loss_history.append(val_loss_logits)
            
            if verbose and (epoch + 1) % 20 == 0:
                logger.info(f"  Logits Baseline Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_logits:.6f}, Val Loss: {val_loss_logits:.6f}")
        
        final_logits_baseline_loss = logits_baseline_loss_history[-1]
        final_logits_baseline_val_loss = logits_baseline_val_loss_history[-1]
        
        if verbose:
            logger.info(f"Logits baseline completed. Final train loss: {final_logits_baseline_loss:.6f}, Final val loss: {final_logits_baseline_val_loss:.6f}")
            logger.info(f"Comparison - Hiddens vs Logits - Train: {final_loss - final_logits_baseline_loss:.6f}, Val: {final_val_loss - final_logits_baseline_val_loss:.6f}")
    
    # Move models back to CPU
    linear_model = linear_model.cpu()
    if baseline_model is not None:
        baseline_model = baseline_model.cpu()
    if logits_baseline_model is not None:
        logits_baseline_model = logits_baseline_model.cpu()
    
    return {
        'final_loss': final_loss,
        'final_val_loss': final_val_loss,
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'model': linear_model,
        'baseline_final_loss': final_baseline_loss,
        'baseline_final_val_loss': final_baseline_val_loss,
        'baseline_loss_history': baseline_loss_history,
        'baseline_val_loss_history': baseline_val_loss_history,
        'baseline_model': baseline_model,
        'logits_baseline_final_loss': final_logits_baseline_loss,
        'logits_baseline_final_val_loss': final_logits_baseline_val_loss,
        'logits_baseline_loss_history': logits_baseline_loss_history,
        'logits_baseline_val_loss_history': logits_baseline_val_loss_history,
        'logits_baseline_model': logits_baseline_model,
        'layer': layer,
        'n_tasks': n_total_tasks,
        'hidden_dim': D,
        'vocab_size': vocab_size,
        'n_samples': n_total,
        'n_train': n_train,
        'n_val': len(val_indices),
    }


def plot_posterior_predictor_loss_vs_k_linear(
    k_values: list,
    layer: int,
    B: int = 64,
    n_samples: int = 1000,
    step: Optional[int] = None,
    learning_rate: float = 0.01,
    num_epochs: int = 100,
    verbose: bool = False,
    position: Union[int, list] = -1,
    positions: Optional[Union[int, list]] = None,
    validation_split: float = 0.2,
    uniform_sampling: bool = True,
    backend: str = "matplotlib",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> dict:
    """
    Train posterior predictors for different k values (log2 of number of minor tasks)
    and plot training and validation losses against k.
    
    Parameters are similar to plot_posterior_predictor_loss_vs_k_latent, but for linear task.
    Note: linear task doesn't use vocab_size.
    
    Parameters:
    -----------
    positions : int, list, or None, optional
        Alias for 'position' parameter. If provided, overrides 'position'.
        If None, uses 'position' parameter value.
    """
    # Handle positions alias for consistency with other plotting functions
    if positions is not None:
        position = positions
    
    train_losses = []
    val_losses = []
    
    for k in k_values:
        if verbose:
            logger.info(f"Processing k={k} (n_minor_tasks = {2**k})...")
        
        exp_name = get_exp_name("linear", k)
        
        try:
            results = train_linear_softmax_posterior_predictor_linear(
                exp_name=exp_name,
                layer=layer,
                B=B,
                n_samples=n_samples,
                step=step,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                verbose=verbose,
                position=position,
                validation_split=validation_split,
                uniform_sampling=uniform_sampling,
                skip_baselines=True,  # Skip baselines for plotting
            )
            
            train_losses.append(results['final_loss'])
            val_losses.append(results['final_val_loss'])
            
            if verbose:
                logger.info(f"  k={k}: Train Loss: {results['final_loss']:.6f}, Val Loss: {results['final_val_loss']:.6f}")
            
            # Clean up
            del results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error processing k={k}: {e}")
            train_losses.append(float('nan'))
            val_losses.append(float('nan'))
    
    # Plot results
    if backend == "matplotlib":
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(k_values, train_losses, 'o-', label='Training Loss', linewidth=2, markersize=8)
        ax.plot(k_values, val_losses, 's-', label='Validation Loss', linewidth=2, markersize=8)
        ax.set_xlabel('k (log2 of number of minor tasks)', fontsize=12)
        ax.set_ylabel('KL Divergence Loss', fontsize=12)
        ax.set_title(f'Posterior Predictor Loss vs k (Linear Task, Layer {layer})', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
            
    elif backend == "plotly":
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_values, y=train_losses,
            mode='lines+markers',
            name='Training Loss',
            line=dict(width=2),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=k_values, y=val_losses,
            mode='lines+markers',
            name='Validation Loss',
            line=dict(width=2),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title=f'Posterior Predictor Loss vs k (Linear Task, Layer {layer})',
            xaxis_title='k (log2 of number of minor tasks)',
            yaxis_title='KL Divergence Loss',
            width=figsize[0]*100,
            height=figsize[1]*100,
        )
        
        if save_path:
            fig.write_image(save_path)
        if show:
            fig.show()
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    return {
        'k_values': k_values,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'fig': fig,
    }
