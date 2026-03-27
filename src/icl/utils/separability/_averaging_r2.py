"""Averaging-based R² and simplex-constrained task-subspace projection."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AveragingR2Result:
    """R² from the averaging-based task-vector test at one (layer, position)."""

    r2_task: float
    r2_additive: float

    ss_total: float
    ss_task: float
    ss_token: float

    n_tasks: int
    n_samples: int

    layer_num: Optional[int] = None
    position: Optional[int] = None


def _simplex_project_coeffs(
    task_vecs: torch.Tensor,
    h_centered: torch.Tensor,
) -> torch.Tensor:
    """Per-sample simplex-constrained coefficients and reconstruction.

    Solves  min_{β ≥ 0, Σβ=1} ‖h − Θᵀβ‖²  approximately by first
    computing the unconstrained affine solution (Σβ=1) then projecting
    onto the probability simplex.

    Parameters
    ----------
    task_vecs : (K, D)
    h_centered : (N, D)

    Returns
    -------
    h_hat : (N, D)  reconstructed hidden states
    """
    from icl.utils.linear_algebra_utils import _project_onto_simplex_np
    import numpy as np

    K, D = task_vecs.shape
    Theta = task_vecs.float()  # (K, D)

    G = Theta @ Theta.T  # (K, K)
    rhs = h_centered @ Theta.T  # (N, K)

    ones = torch.ones(K, 1, dtype=G.dtype, device=G.device)
    G_aug = torch.cat([
        torch.cat([G, ones], dim=1),
        torch.cat([ones.T, torch.zeros(1, 1, dtype=G.dtype, device=G.device)], dim=1),
    ], dim=0)

    rhs_aug = torch.cat([rhs, torch.ones(rhs.shape[0], 1, dtype=rhs.dtype, device=rhs.device)], dim=1)

    sol = torch.linalg.solve(G_aug, rhs_aug.T).T  # (N, K+1)
    beta_unconstrained = sol[:, :K]  # (N, K)

    beta_np = beta_unconstrained.detach().cpu().numpy()
    beta_proj = np.stack([_project_onto_simplex_np(b) for b in beta_np])
    beta_simplex = torch.from_numpy(beta_proj).to(dtype=Theta.dtype, device=Theta.device)

    h_hat = beta_simplex @ Theta  # (N, D)
    return h_hat


def task_subspace_r2_at_position(
    task_vecs: torch.Tensor,
    hiddens: torch.Tensor,
    covariates: Optional[torch.Tensor] = None,
    fit_token: str = "linear",
    grand_mean: Optional[torch.Tensor] = None,
    token_vecs: Optional[torch.Tensor] = None,
    token_ids: Optional[torch.Tensor] = None,
    simplex: bool = True,
    eps: float = 1e-10,
) -> "AveragingR2Result":
    """Fraction of hidden-state variance explained by task subspace (+ token).

    Subtracts a mean vector and projects onto span(task_vecs), optionally
    fitting a linear token model on the residual.

    Parameters
    ----------
    task_vecs : (K, D) centred task vectors
    hiddens : (n_tasks, B, D) hidden states at one position
    covariates : (B, d) or (n_tasks, B, d)
        Covariates x_t.  Used when ``fit_token="linear"``.
        Shape ``(B, d)`` is broadcast identically to every task
        (e.g. shared input in linear regression).  Shape
        ``(n_tasks, B, d)`` provides per-task covariates (e.g. one-hot
        tokens that differ across tasks).
    fit_token : "none" | "linear" | "anova"
        ``"linear"``: OLS on ``covariates`` (continuous or one-hot).
        ``"anova"``: subtract the known token vector (looked up via
        ``token_ids``), then project the remainder onto the task
        subspace via drop-one.
        ``"none"``: no token model.
    grand_mean : (D,), optional
        If provided, subtract this fixed mean instead of the per-position
        sample mean.  Useful for testing what happens without per-position
        demeaning.
    token_vecs : (V, D), optional
        Pre-estimated centred token vectors.  Required when
        ``fit_token="anova"``.
    token_ids : (n_tasks, B), optional
        Integer token identity at this position for each (task, sample).
        Required when ``fit_token="anova"``.
    simplex : bool
        If True (default), constrain task coefficients β to the
        probability simplex (β ≥ 0, Σβ = 1).  If False, use
        unconstrained orthogonal projection.
    eps : float

    Returns
    -------
    AveragingR2Result
    """
    K, D = task_vecs.shape
    n_tasks, B, _ = hiddens.shape
    device = hiddens.device

    h_flat = hiddens.reshape(n_tasks * B, D).float()
    N = h_flat.shape[0]

    if grand_mean is not None:
        mu_t = grand_mean.to(device).float()
    else:
        mu_t = h_flat.mean(dim=0)
    h_centered = h_flat - mu_t.unsqueeze(0)

    ss_total = (h_centered ** 2).sum().item()

    V = task_vecs.to(device).float()  # (K, D)

    if simplex:
        # ---- Task-only R² (simplex-constrained) ----
        h_task_hat = _simplex_project_coeffs(V, h_centered)
        ss_task_residual = ((h_centered - h_task_hat) ** 2).sum().item()
        ss_task = ss_total - ss_task_residual
        r2_task = ss_task / (ss_total + eps)

        # ---- Task + token R² ----
        if fit_token == "none" or (covariates is None and token_vecs is None):
            r2_additive = r2_task
            ss_token = 0.0
        elif fit_token == "linear":
            residual = h_centered - h_task_hat  # (N, D)

            if covariates.ndim == 3:
                d = covariates.shape[2]
                x_flat = covariates.to(device).float().reshape(N, d)
            else:
                d = covariates.shape[1]
                x_flat = covariates.to(device).float()
                x_flat = x_flat.unsqueeze(0).expand(n_tasks, B, d).reshape(N, d)

            XtX = x_flat.T @ x_flat
            XtX_reg = XtX + eps * torch.eye(d, device=XtX.device)
            W_x = torch.linalg.solve(XtX_reg, x_flat.T @ residual)
            h_token = x_flat @ W_x

            ss_token = (h_token ** 2).sum().item()
            ss_residual = ((residual - h_token) ** 2).sum().item()
            r2_additive = 1.0 - ss_residual / (ss_total + eps)
        elif fit_token == "anova":
            if token_vecs is None or token_ids is None:
                raise ValueError(
                    "fit_token='anova' requires token_vecs and token_ids"
                )
            Vt = token_vecs.to(device).float()
            ids_flat = token_ids.reshape(N).long()
            h_tok_effect = Vt[ids_flat]                    # (N, D)
            h_no_tok = h_centered - h_tok_effect
            h_task_hat_nt = _simplex_project_coeffs(V, h_no_tok)
            h_additive_hat = h_task_hat_nt + h_tok_effect
            ss_residual = ((h_centered - h_additive_hat) ** 2).sum().item()
            ss_additive = ss_total - ss_residual
            r2_additive = ss_additive / (ss_total + eps)
            ss_token = ss_additive - ss_task
        else:
            raise ValueError(
                f"fit_token must be 'none', 'linear', or 'anova', "
                f"got {fit_token!r}"
            )
    else:
        # ---- Task-only R² (unconstrained projection) ----
        V_basis = V[:-1]  # (K-1, D), full rank
        P = V_basis.T @ torch.linalg.solve(V_basis @ V_basis.T, V_basis)
        h_task = (h_centered @ P)  # (N, D)

        ss_task = (h_task ** 2).sum().item()
        r2_task = ss_task / (ss_total + eps)

        # ---- Task + token R² ----
        if fit_token == "none" or covariates is None:
            r2_additive = r2_task
            ss_token = 0.0
        elif fit_token == "linear":
            residual = h_centered - h_task  # (N, D)

            if covariates.ndim == 3:
                d = covariates.shape[2]
                x_flat = covariates.to(device).float().reshape(N, d)
            else:
                d = covariates.shape[1]
                x_flat = covariates.to(device).float()
                x_flat = x_flat.unsqueeze(0).expand(n_tasks, B, d).reshape(N, d)

            XtX = x_flat.T @ x_flat
            XtX_reg = XtX + eps * torch.eye(d, device=XtX.device)
            W_x = torch.linalg.solve(XtX_reg, x_flat.T @ residual)
            h_token = x_flat @ W_x

            ss_token = (h_token ** 2).sum().item()
            ss_residual = ((residual - h_token) ** 2).sum().item()
            r2_additive = 1.0 - ss_residual / (ss_total + eps)
        elif fit_token == "anova":
            if token_vecs is None or token_ids is None:
                raise ValueError(
                    "fit_token='anova' requires token_vecs and token_ids"
                )
            Vt = token_vecs.to(device).float()
            Vt_basis = Vt[:-1]
            combined = torch.cat([V_basis, Vt_basis], dim=0)
            Q, _ = torch.linalg.qr(combined.T)
            P_comb = Q @ Q.T
            h_additive = h_centered @ P_comb

            ss_additive = (h_additive ** 2).sum().item()
            r2_additive = ss_additive / (ss_total + eps)
            ss_token = ss_additive - ss_task
        else:
            raise ValueError(
                f"fit_token must be 'none', 'linear', or 'anova', "
                f"got {fit_token!r}"
            )

    return AveragingR2Result(
        r2_task=r2_task,
        r2_additive=r2_additive,
        ss_total=ss_total,
        ss_task=ss_task,
        ss_token=ss_token,
        n_tasks=n_tasks,
        n_samples=N,
    )
