"""Internal helpers and posterior computation for linear regression."""

import hashlib
import json
import math
import os
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _json_ready(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return repr(value)


def _get_linear_kl_over_steps_cache_path(
    *,
    exp_dir: str,
    mode: str,
    num_samples: int,
    steps_to_use,
    eps: float,
    p_minor_hybrid: float | None = None,
):
    cache_dir = os.path.join(exp_dir, "analysis_cache")
    os.makedirs(cache_dir, exist_ok=True)
    payload = {
        "mode": mode,
        "num_samples": int(num_samples),
        "steps_to_use": [int(s) for s in steps_to_use],
        "eps": float(eps),
        "p_minor_hybrid": float(p_minor_hybrid) if p_minor_hybrid is not None else None,
        "version": 1,
    }
    payload_str = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    payload_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"linear_kl_over_steps_{payload_hash}.pkl")


def _sample_linear_mode_fixed(task, mode: str, batch_size: int, step_idx: int):
    bs_old = int(task.batch_size)
    task.batch_size = int(batch_size)
    try:
        data = task.sample_data(step_idx)
        B = data.shape[0]
        dev = data.device

        if mode == "major":
            if task.task_pool is None or int(task.n_tasks) <= 0:
                raise ValueError("No major task pool available.")
            idx = torch.randint(0, int(task.n_tasks), (B,), device=dev, generator=task.task_gen)
            tasks = task.task_pool[idx]

        elif mode == "minor":
            if int(task.n_minor_tasks) > 0 and task.minor_pool is not None:
                idx = torch.randint(0, int(task.n_minor_tasks), (B,), device=dev, generator=task.task_gen)
                tasks = task.minor_pool[idx]
            else:
                tasks = torch.randn(
                    (B, int(task.n_dims), 1),
                    generator=task.task_gen,
                    dtype=task.dtype,
                    device=dev,
                ) * float(task.task_scale)

        elif mode == "ood":
            tasks = torch.randn(
                (B, int(task.n_dims), 1),
                generator=task.task_gen,
                dtype=task.dtype,
                device=dev,
            ) * float(task.task_scale)

        elif mode in ("train", "test", "testing", "eval"):
            tasks = task.sample_tasks(step_idx, is_eval=False)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        targets = task.evaluate(data, tasks, step_idx)
        return data, targets
    finally:
        task.batch_size = bs_old


def _gauss_kl_surrogate(mu_model: torch.Tensor, mu_ref: torch.Tensor, sigma2: float):
    return 0.5 * (mu_model - mu_ref).pow(2) / sigma2

def _validate_and_build_pool(task, data, targets, include_minor):
    """Validate inputs and concatenate major (+optional minor) task vectors.

    Returns (targets, W_all, Kmaj, Kmin, B, T, D).
    """
    device = data.device
    if targets.dim() == 3 and targets.size(-1) == 1:
        targets = targets.squeeze(-1)
    assert data.dim() == 3 and targets.dim() == 2
    B, T, D = data.shape
    assert targets.shape == (B, T)

    if task.n_tasks <= 0 or task.task_pool is None:
        raise ValueError("Posterior needs a finite task_pool (n_tasks > 0).")

    W_major = task.task_pool.to(device)
    if include_minor and (task.n_minor_tasks > 0) and (task.minor_pool is not None):
        W_minor = task.minor_pool.to(device)
        W_all = torch.cat([W_major, W_minor], dim=0)
        Kmaj, Kmin = W_major.shape[0], W_minor.shape[0]
    else:
        W_all = W_major
        Kmaj, Kmin = W_major.shape[0], 0

    K = W_all.shape[0]
    assert W_all.shape == (K, D, 1)
    return targets, W_all, Kmaj, Kmin, B, T, D


def _uniform_log_prior(task, Kmaj, Kmin, device, eps=1e-30):
    """Build log π  with  π_k ∝ 1/K_maj  (or weighted by p_minor)."""
    K = Kmaj + Kmin
    dtype = torch.float32
    if Kmin == 0:
        prior = torch.full((K,), 1.0 / Kmaj, device=device, dtype=dtype)
    else:
        p0 = float(task.p_minor)
        prior = torch.cat([
            torch.full((Kmaj,), (1.0 - p0) / Kmaj, device=device, dtype=dtype),
            torch.full((Kmin,), p0 / Kmin, device=device, dtype=dtype),
        ], dim=0)
    return torch.clamp(prior, min=eps).log()


# ---------------------------------------------------------------------------
# Posterior computation
# ---------------------------------------------------------------------------


@torch.no_grad()
def task_posterior_linear_regression(
    task,  # NoisyLinearRegression instance
    data: torch.Tensor,      # (B, T, D)
    targets: torch.Tensor,   # (B, T)  (or (B,T,1))
    *,
    include_minor: bool = True,
    return_log: bool = False,
    eps: float = 1e-30,
) -> torch.Tensor:
    """Compute posterior P(z=k | X,Y) over discrete task pool.

    log P(z=k | X,Y) ∝ log πₖ − ‖Y − Xwₖ‖² / (2σ²)

    Returns (B, K) posterior (or log-posterior).
    """
    targets, W_all, Kmaj, Kmin, B, T, D = _validate_and_build_pool(
        task, data, targets, include_minor,
    )
    K = W_all.shape[0]
    log_prior = _uniform_log_prior(task, Kmaj, Kmin, data.device, eps)

    W2 = W_all.squeeze(-1)               # (K, D)
    preds = torch.einsum("btd,kd->btk", data, W2)  # (B, T, K)

    resid = targets.unsqueeze(-1) - preds         # (B, T, K)
    sse = (resid ** 2).sum(dim=1)                 # (B, K)

    sigma2 = max(float(task.noise_scale) ** 2, eps)

    loglik = -0.5 * sse / sigma2                  # (B, K)

    unnorm = loglik + log_prior.view(1, K)        # (B, K)
    log_post = unnorm - torch.logsumexp(unnorm, dim=-1, keepdim=True)

    return log_post if return_log else torch.exp(log_post)


def task_posterior_over_time_linear_regression(
    task,
    data: torch.Tensor,      # (B, T, D)
    targets: torch.Tensor,   # (B, T)  (or (B,T,1))
    *,
    include_minor: bool = True,
    return_log: bool = False,
    eps: float = 1e-30,
) -> torch.Tensor:
    """Filtering posterior P(z=k | x₀:t, y₀:t₋₁) at every position t.

    At position t the model has seen (x₀,y₀)…(x_{t-1},y_{t-1}),xₜ and
    must predict yₜ, so yₜ is *not* included.

    Returns (B, T, K) posterior (or log-posterior).
    """
    targets, W_all, Kmaj, Kmin, B, T, D = _validate_and_build_pool(
        task, data, targets, include_minor,
    )
    K = W_all.shape[0]
    log_prior = _uniform_log_prior(task, Kmaj, Kmin, data.device, eps)

    W2 = W_all.squeeze(-1)                           # (K, D)
    preds = torch.einsum("btd,kd->btk", data, W2)    # (B, T, K)
    resid = targets.unsqueeze(-1) - preds             # (B, T, K)

    sigma2 = max(float(task.noise_scale) ** 2, eps)
    resid_sq = resid ** 2                              # (B, T, K)
    cum_sse = resid_sq.cumsum(dim=1)                   # (B, T, K)
    # Shift cumSSE by one position: at position t the model predicts yₜ
    # using only (x₀,y₀)…(x_{t-1},y_{t-1}), so the log-likelihood must
    # exclude the current residual.  cum_sse_shifted[t] = Σ_{s<t} (yₛ − wₖᵀxₛ)².
    cum_sse_shifted = torch.zeros_like(cum_sse)
    cum_sse_shifted[:, 1:, :] = cum_sse[:, :-1, :]    # (B, T, K)
    loglik = -0.5 * cum_sse_shifted / sigma2           # (B, T, K)

    unnorm = loglik + log_prior.view(1, 1, K)         # (B, T, K)
    log_post = unnorm - torch.logsumexp(unnorm, dim=-1, keepdim=True)

    return log_post if return_log else torch.exp(log_post)


@torch.no_grad()
def task_posterior_with_gaussian_linear_regression(
    task,  # NoisyLinearRegression instance
    data: torch.Tensor,      # (B, T, D)
    targets: torch.Tensor,   # (B, T)  (or (B,T,1))
    *,
    include_minor: bool = False,
    p_gaussian: Optional[float] = None,
    return_log: bool = False,
    eps: float = 1e-30,
) -> torch.Tensor:
    """(K+1)-way posterior with a Gaussian "new task" hypothesis.

    Discrete:  log P(z=k | X,Y) ∝ log πₖ − ‖Y − Xwₖ‖²/(2σ²)
    Gaussian:  P(Y | X, z=K+1)  = N(Y; 0, σ²I + τ²XXᵀ)  (Woodbury, O(D³))

    Returns (B, K+1) posterior (or log-posterior).
    """
    targets, W_all, Kmaj, Kmin, B, T, D = _validate_and_build_pool(
        task, data, targets, include_minor,
    )
    K = W_all.shape[0]
    device = data.device
    dtype = torch.float32

    if p_gaussian is None:
        p_gaussian = float(getattr(task, "p_minor", 0.0))
        if p_gaussian <= 0:
            p_gaussian = 1.0 / (K + 1)
    p_discrete = 1.0 - p_gaussian

    if Kmin == 0:
        pi_k = p_discrete / Kmaj
        log_prior_discrete = torch.full((K,), math.log(max(pi_k, eps)),
                                        device=device, dtype=dtype)
    else:
        p0 = float(task.p_minor)
        pi_major = (1.0 - p0) * p_discrete
        pi_minor = p0 * p_discrete
        log_prior_discrete = torch.cat([
            torch.full((Kmaj,), math.log(max(pi_major / Kmaj, eps)),
                        device=device, dtype=dtype),
            torch.full((Kmin,), math.log(max(pi_minor / Kmin, eps)),
                        device=device, dtype=dtype),
        ], dim=0)

    log_prior_gauss = math.log(max(p_gaussian, eps))

    sigma2 = float(task.noise_scale) ** 2
    sigma2 = max(sigma2, eps)

    W2 = W_all.squeeze(-1)                         # (K, D)
    preds = torch.einsum("btd,kd->btk", data.float(), W2.float())  # (B, T, K)
    resid = targets.float().unsqueeze(-1) - preds   # (B, T, K)
    sse = (resid ** 2).sum(dim=1)                   # (B, K)

    log_norm_const = -0.5 * T * math.log(2 * math.pi * sigma2)
    loglik_known = log_norm_const + (-0.5 * sse / sigma2)  # (B, K)

    # ── Marginal likelihood for the Gaussian hypothesis ──────────────────
    # Model:  w ~ N(0, τ²I),  yₜ | w,xₜ ~ N(wᵀxₜ, σ²)
    # Marginal:  p(Y|X, z=K+1) = N(Y; 0, σ²I + τ²XXᵀ)
    #
    # Direct evaluation of the T×T covariance is expensive.  Instead use
    # the Bayesian linear regression identity:
    #
    #   log p(Y|X) = −T/2 log(2π) − ½ log|Σ_Y| − ½ Yᵀ Σ_Y⁻¹ Y
    #
    # where Σ_Y = σ²I_T + τ²XXᵀ.  By the matrix determinant lemma and
    # Woodbury identity, everything reduces to D×D operations:
    #
    #   Posterior precision:  Λ_N = (1/τ²)I_D + (1/σ²) XᵀX
    #   log|Σ_Y| = T log σ² + D log τ² + log|Λ_N|
    #   Yᵀ Σ_Y⁻¹ Y = (1/σ²)[ yᵀy − (1/σ²)(Xᵀy)ᵀ Λ_N⁻¹ (Xᵀy) ]
    #
    # Λ_N is factored via Cholesky for stable inversion and log-determinant.

    tau2 = float(task.task_scale) ** 2
    tau2 = max(tau2, eps)

    data_f = data.float()       # X: (B, T, D)
    tgt_f = targets.float()     # Y: (B, T)

    XtX = torch.bmm(data_f.transpose(1, 2), data_f)  # XᵀX: (B, D, D)
    Xty = torch.bmm(data_f.transpose(1, 2), tgt_f.unsqueeze(-1)).squeeze(-1)  # Xᵀy: (B, D)
    yty = (tgt_f ** 2).sum(dim=1)  # yᵀy: (B,)

    eye_D = torch.eye(D, device=device, dtype=torch.float32)
    Lambda_N = (1.0 / tau2) * eye_D.unsqueeze(0) + (1.0 / sigma2) * XtX  # Λ_N: (B, D, D)

    L_N = torch.linalg.cholesky(Lambda_N)  # Cholesky: Λ_N = L_N L_Nᵀ

    # log|Λ_N| = 2 Σᵢ log L_N[i,i]
    log_det_Lambda_N = 2.0 * L_N.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)  # (B,)

    # log|Σ_Y| = T log σ² + D log τ² + log|Λ_N|
    log_det_Sigma_y = T * math.log(sigma2) + D * math.log(tau2) + log_det_Lambda_N

    # v = Λ_N⁻¹ Xᵀy   (solved via Cholesky)
    v = torch.cholesky_solve(Xty.unsqueeze(-1), L_N).squeeze(-1)  # (B, D)
    # (Xᵀy)ᵀ Λ_N⁻¹ (Xᵀy)
    quad_correction = (Xty * v).sum(dim=-1)  # (B,)

    # Yᵀ Σ_Y⁻¹ Y  via Woodbury
    quad_form = (1.0 / sigma2) * (yty - (1.0 / sigma2) * quad_correction)  # (B,)

    loglik_gauss = (
        -0.5 * T * math.log(2 * math.pi)
        - 0.5 * log_det_Sigma_y
        - 0.5 * quad_form
    )  # (B,)

    unnorm_known = loglik_known + log_prior_discrete.unsqueeze(0)  # (B, K)
    unnorm_gauss = (loglik_gauss + log_prior_gauss).unsqueeze(-1)  # (B, 1)
    unnorm = torch.cat([unnorm_known, unnorm_gauss], dim=-1)       # (B, K+1)

    log_post = unnorm - torch.logsumexp(unnorm, dim=-1, keepdim=True)

    return log_post if return_log else torch.exp(log_post)
