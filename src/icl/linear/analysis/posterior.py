"""Posterior computation and plotting for linear regression analysis."""

import gc
import hashlib
import json
import math
import os
import pickle
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

import icl.utils.notebook_utils as nu
from icl.linear.analysis._helpers import _show_or_close, _temporary_task_attributes
from icl.linear.analysis.probes import train_linear_hidden_predictor
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)


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
):
    cache_dir = os.path.join(exp_dir, "analysis_cache")
    os.makedirs(cache_dir, exist_ok=True)
    payload = {
        "mode": mode,
        "num_samples": int(num_samples),
        "steps_to_use": [int(s) for s in steps_to_use],
        "eps": float(eps),
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


@torch.no_grad()
def plot_kl_model_vs_two_bayes_linear(
    exp_name: str,
    modes: tuple = ("train", "major", "minor", "ood"),
    num_samples: int = 512,
    step: Optional[int] = None,
    eps: float = 1e-12,
    figsize: Optional[tuple] = None,
    show: bool = True,
):
    """Compare transformer ŷ against two Bayesian baselines via KL surrogate.

    Baselines:
      1. Exact known-pool Bayes  (DiscreteMMSE or UnbalancedMMSE)
      2. 3-known + Gaussian-new  (MixedRidge)

    KL surrogate (equal-variance Gaussian):
        KL ≈ (μ_model − μ_baseline)² / (2σ²)
    """
    import matplotlib.pyplot as plt
    from icl.linear.linear_path_utils import load_model_task_config
    from icl.linear.lr_models import DiscreteMMSE, UnbalancedMMSE, MixedRidge

    _, train_task, config = load_model_task_config(exp_name)
    if step is None:
        step = config.training.total_steps
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval().to(config.device)

    p_minor_orig = float(getattr(train_task, "p_minor", 0.0))
    sigma2 = max(float(train_task.noise_scale) ** 2, eps)

    if (
        int(getattr(train_task, "n_minor_tasks", 0)) > 0
        and getattr(train_task, "minor_pool", None) is not None
    ):
        exact_bayes = UnbalancedMMSE(
            scale=float(train_task.noise_scale),
            task_pool=train_task.task_pool.clone(),
            p0=p_minor_orig,
            minor_task_pool=train_task.minor_pool.clone(),
            dtype=train_task.dtype,
        ).to(config.device)
    else:
        exact_bayes = DiscreteMMSE(
            scale=float(train_task.noise_scale),
            task_pool=train_task.task_pool.clone(),
            dtype=train_task.dtype,
        ).to(config.device)

    tau = float(train_task.noise_scale) / max(float(train_task.minor_scale), eps)
    hybrid_bayes = MixedRidge(
        tau=tau,
        task_pool=train_task.task_pool.clone(),
        p0=p_minor_orig,
        noise_scale=float(train_task.noise_scale),
        dtype=train_task.dtype,
    ).to(config.device)

    def _sample_linear_mode(task, mode: str, batch_size: int, step_idx: int):
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

    def _gauss_kl_surrogate(mu_model: torch.Tensor, mu_ref: torch.Tensor):
        return 0.5 * (mu_model - mu_ref).pow(2) / sigma2  # (B, T)

    results = {}
    if isinstance(modes, str):
        modes = (modes,)
    else:
        modes = tuple(modes)

    ncols = 1 if len(modes) == 1 else 2
    nrows = (len(modes) + ncols - 1) // ncols
    if figsize is None:
        figsize = (7 * ncols, 4.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for mi, mode in enumerate(modes):
        data, targets = _sample_linear_mode(train_task, mode=mode, batch_size=int(num_samples), step_idx=int(step) + mi)
        data = data.to(config.device)
        targets = targets.to(config.device)

        preds_model = model(data, targets)         # (B, T)
        preds_exact = exact_bayes(data, targets)   # (B, T)
        preds_hybrid = hybrid_bayes(data, targets) # (B, T)

        kl_exact = _gauss_kl_surrogate(preds_model, preds_exact)
        kl_hybrid = _gauss_kl_surrogate(preds_model, preds_hybrid)

        mean_exact = kl_exact.mean(dim=0).detach().cpu().numpy()
        std_exact = kl_exact.std(dim=0).detach().cpu().numpy()
        mean_hybrid = kl_hybrid.mean(dim=0).detach().cpu().numpy()
        std_hybrid = kl_hybrid.std(dim=0).detach().cpu().numpy()
        pos = torch.arange(preds_model.size(1)).cpu().numpy()

        ax = axes_flat[mi]
        ax.plot(pos, mean_exact, color="#1f77b4", lw=2.0, label="Exact known-pool Bayes")
        exact_lo = np.maximum(mean_exact - std_exact, 0.0)
        exact_hi = mean_exact + std_exact
        ax.fill_between(pos, exact_lo, exact_hi, color="#1f77b4", alpha=0.2)

        ax.plot(pos, mean_hybrid, color="#d62728", lw=2.0, label="3-known + Gaussian-new")
        hybrid_lo = np.maximum(mean_hybrid - std_hybrid, 0.0)
        hybrid_hi = mean_hybrid + std_hybrid
        ax.fill_between(pos, hybrid_lo, hybrid_hi, color="#d62728", alpha=0.2)

        ax.set_title("", fontsize=18)
        ax.set_xlabel("Position", fontsize=11)
        ax.set_ylabel("KL(model || baseline)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        results[mode] = {
            "positions": pos,
            "p_minor_bayes_used": float(p_minor_orig),
            "sigma2_used": float(sigma2),
            "kl_exact_mean": mean_exact,
            "kl_exact_std": std_exact,
            "kl_hybrid_mean": mean_hybrid,
            "kl_hybrid_std": std_hybrid,
            "kl_exact": kl_exact.detach().cpu(),
            "kl_hybrid": kl_hybrid.detach().cpu(),
        }

    for j in range(len(modes), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "fig": fig,
        "axes": axes,
        "results": results,
        "modes": modes,
        "step": int(step),
    }


@torch.no_grad()
def plot_kl_model_vs_two_bayes_linear_across_k(
    k_values,
    mode: str = "ood",
    num_samples: int = 1024,
    eps: float = 1e-12,
    exp_name_kwargs: Optional[dict] = None,
    add_std_band: bool = False,
    same_y_axis: bool = False,
    figsize: tuple = (14, 5),
    show: bool = True,
    verbose: bool = False,
) -> dict:
    """Sweep k → KL(model‖baseline) for the linear task.

    For each k: loads experiment, computes KL surrogate vs both Bayesian
    baselines, and plots mean curves over positions.
    """
    import matplotlib.pyplot as plt
    from icl.utils.unified_interface import get_exp_name

    if exp_name_kwargs is None:
        exp_name_kwargs = {}

    curves = {}
    for k in k_values:
        exp_name = get_exp_name("linear", k, **exp_name_kwargs)
        try:
            out = plot_kl_model_vs_two_bayes_linear(
                exp_name=exp_name,
                modes=mode,
                num_samples=num_samples,
                eps=eps,
                show=False,
            )
            d = out["results"][mode]
            curves[k] = {
                "positions": np.asarray(d["positions"], dtype=float),
                "exact_mean": np.asarray(d["kl_exact_mean"], dtype=float),
                "exact_std": np.asarray(d["kl_exact_std"], dtype=float),
                "hybrid_mean": np.asarray(d["kl_hybrid_mean"], dtype=float),
                "hybrid_std": np.asarray(d["kl_hybrid_std"], dtype=float),
            }
        except Exception as e:
            if verbose:
                print(f"[warn] k={k} failed: {e}")

    ks = sorted(curves.keys())
    if len(ks) == 0:
        raise RuntimeError("No k succeeded. Check exp_name availability/checkpoints.")

    if len(ks) <= 10:
        palette = list(plt.get_cmap("tab10").colors)
    elif len(ks) <= 20:
        palette = list(plt.get_cmap("tab20").colors)
    else:
        base = list(plt.get_cmap("tab20").colors)
        reps = (len(ks) + len(base) - 1) // len(base)
        palette = (base * reps)[: len(ks)]
    color_map = {k: palette[i] for i, k in enumerate(ks)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=False, sharey=same_y_axis)

    for k in ks:
        c = color_map[k]
        d = curves[k]
        x = d["positions"]

        ax1.plot(x, d["exact_mean"], color=c, lw=2.0, label=f"k={k}")
        if add_std_band:
            exact_lo = np.maximum(d["exact_mean"] - d["exact_std"], 0.0)
            exact_hi = d["exact_mean"] + d["exact_std"]
            ax1.fill_between(x, exact_lo, exact_hi, color=c, alpha=0.15)

        ax2.plot(x, d["hybrid_mean"], color=c, lw=2.0, label=f"k={k}")
        if add_std_band:
            hybrid_lo = np.maximum(d["hybrid_mean"] - d["hybrid_std"], 0.0)
            hybrid_hi = d["hybrid_mean"] + d["hybrid_std"]
            ax2.fill_between(x, hybrid_lo, hybrid_hi, color=c, alpha=0.15)

    ax1.set_title("", fontsize=18)
    ax1.set_xlabel("Position", fontsize=11)
    ax1.set_ylabel("KL(model || exact)", fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("", fontsize=18)
    ax2.set_xlabel("Position", fontsize=11)
    ax2.set_ylabel("KL(model || hybrid)", fontsize=11)
    ax2.grid(True, alpha=0.3)

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=9, frameon=True)
    fig.tight_layout(rect=(0.0, 0.0, 0.9, 1.0))

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "fig": fig,
        "axes": (ax1, ax2),
        "mode": mode,
        "k_values_loaded": ks,
        "curves_by_k": curves,
    }


@torch.no_grad()
def plot_kl_model_vs_two_bayes_linear_over_steps(
    exp_name: str,
    mode: str = "train",
    num_samples: int = 1024,
    steps=None,
    eps: float = 1e-12,
    add_std_band: bool = False,
    log_y: bool = False,
    use_cache: bool = True,
    force_recompute: bool = False,
    figsize: tuple = (8, 5),
    show: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Plot average KL-surrogate(model || baseline) across exact checkpoint steps
    for one linear experiment.
    """
    from icl.linear.linear_path_utils import load_model_task_config
    from icl.linear.lr_models import DiscreteMMSE, MixedRidge, UnbalancedMMSE

    _, train_task, config = load_model_task_config(exp_name)

    exp_dir = os.path.join(config.work_dir, exp_name)
    if os.getcwd().endswith("notebooks"):
        exp_dir = os.path.join("..", exp_dir)
    checkpoint_dir = exp_dir if config.task.name == "noisy_linear_regression" else os.path.join(exp_dir, "checkpoints")
    available_steps = nu.list_checkpoints(checkpoint_dir)["all_steps"]
    if len(available_steps) == 0:
        raise RuntimeError(f"No checkpoints found for exp_name={exp_name}.")

    if steps is None:
        steps_to_use = list(available_steps)
    else:
        requested_steps = [int(s) for s in steps]
        missing_steps = [s for s in requested_steps if s not in available_steps]
        if missing_steps:
            raise ValueError(
                f"Requested steps not found exactly: {missing_steps}. "
                f"Available steps: {available_steps}"
            )
        steps_to_use = requested_steps

    cache_path = _get_linear_kl_over_steps_cache_path(
        exp_dir=exp_dir,
        mode=mode,
        num_samples=num_samples,
        steps_to_use=steps_to_use,
        eps=eps,
    )
    cached = None
    if use_cache and not force_recompute and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if verbose:
            print(f"[cache] loaded KL-over-steps from {cache_path}")

    p_minor_orig = float(getattr(train_task, "p_minor", 0.0))
    sigma2 = max(float(train_task.noise_scale) ** 2, eps)

    if cached is None:
        if (
            int(getattr(train_task, "n_minor_tasks", 0)) > 0
            and getattr(train_task, "minor_pool", None) is not None
        ):
            exact_bayes = UnbalancedMMSE(
                scale=float(train_task.noise_scale),
                task_pool=train_task.task_pool.clone(),
                p0=p_minor_orig,
                minor_task_pool=train_task.minor_pool.clone(),
                dtype=train_task.dtype,
            ).to(config.device)
        else:
            exact_bayes = DiscreteMMSE(
                scale=float(train_task.noise_scale),
                task_pool=train_task.task_pool.clone(),
                dtype=train_task.dtype,
            ).to(config.device)

        tau = float(train_task.noise_scale) / max(float(train_task.minor_scale), eps)
        approx_bayes = MixedRidge(
            tau=tau,
            task_pool=train_task.task_pool.clone(),
            p0=p_minor_orig,
            noise_scale=float(train_task.noise_scale),
            dtype=train_task.dtype,
        ).to(config.device)

        sample_step = 0
        data, targets = _sample_linear_mode_fixed(
            train_task,
            mode=mode,
            batch_size=int(num_samples),
            step_idx=sample_step,
        )
        data = data.to(config.device)
        targets = targets.to(config.device)

        preds_exact = exact_bayes(data, targets)
        preds_approx = approx_bayes(data, targets)

        exact_means = []
        approx_means = []
        exact_stds = []
        approx_stds = []
        actual_steps = []

        for step in steps_to_use:
            model, actual_step = nu.load_checkpoint(
                config,
                step=step,
                exp_name=exp_name,
                return_actual_step=True,
            )
            model.eval().to(config.device)

            preds_model = model(data, targets)
            kl_exact = _gauss_kl_surrogate(preds_model, preds_exact, sigma2)
            kl_approx = _gauss_kl_surrogate(preds_model, preds_approx, sigma2)

            per_sample_exact = kl_exact.mean(dim=1)
            per_sample_approx = kl_approx.mean(dim=1)

            exact_means.append(float(per_sample_exact.mean().item()))
            approx_means.append(float(per_sample_approx.mean().item()))
            exact_stds.append(float(per_sample_exact.std().item()))
            approx_stds.append(float(per_sample_approx.std().item()))
            actual_steps.append(int(actual_step))

            if verbose:
                print(
                    f"[step {actual_step}] "
                    f"KL_exact={exact_means[-1]:.6f}, "
                    f"KL_approx={approx_means[-1]:.6f}"
                )

        cache_payload = {
            "exp_name": exp_name,
            "mode": mode,
            "steps": [int(s) for s in actual_steps],
            "num_samples": int(num_samples),
            "p_minor_bayes_used": float(p_minor_orig),
            "sigma2_used": float(sigma2),
            "kl_exact_mean_by_step": exact_means,
            "kl_exact_std_by_step": exact_stds,
            "kl_hybrid_mean_by_step": approx_means,
            "kl_hybrid_std_by_step": approx_stds,
        }
        if use_cache:
            with open(cache_path, "wb") as f:
                pickle.dump(cache_payload, f)
            if verbose:
                print(f"[cache] saved KL-over-steps to {cache_path}")
    else:
        cache_payload = cached

    x = np.asarray(cache_payload["steps"], dtype=float)
    exact_mean_arr = np.asarray(cache_payload["kl_exact_mean_by_step"], dtype=float)
    hybrid_mean_arr = np.asarray(cache_payload["kl_hybrid_mean_by_step"], dtype=float)
    exact_std_arr = np.asarray(cache_payload["kl_exact_std_by_step"], dtype=float)
    hybrid_std_arr = np.asarray(cache_payload["kl_hybrid_std_by_step"], dtype=float)
    band_floor = float(max(eps, 1e-30))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(x, exact_mean_arr, color="#1f77b4", lw=2.0, label="Exact")
    if add_std_band:
        ax.fill_between(
            x,
            np.maximum(exact_mean_arr - exact_std_arr, band_floor if log_y else 0.0),
            exact_mean_arr + exact_std_arr,
            color="#1f77b4",
            alpha=0.18,
        )

    ax.plot(x, hybrid_mean_arr, color="#d62728", lw=2.0, label="Approx")
    if add_std_band:
        ax.fill_between(
            x,
            np.maximum(hybrid_mean_arr - hybrid_std_arr, band_floor if log_y else 0.0),
            hybrid_mean_arr + hybrid_std_arr,
            color="#d62728",
            alpha=0.18,
        )

    ax.set_title("", fontsize=18)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Average KL(model || baseline)", fontsize=11)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "fig": fig,
        "ax": ax,
        "exp_name": exp_name,
        "mode": mode,
        "steps": [int(s) for s in cache_payload["steps"]],
        "num_samples": int(num_samples),
        "log_y": bool(log_y),
        "use_cache": bool(use_cache),
        "cache_path": cache_path if use_cache else None,
        "p_minor_bayes_used": float(cache_payload["p_minor_bayes_used"]),
        "sigma2_used": float(cache_payload["sigma2_used"]),
        "kl_exact_mean_by_step": exact_mean_arr,
        "kl_exact_std_by_step": exact_std_arr,
        "kl_hybrid_mean_by_step": hybrid_mean_arr,
        "kl_hybrid_std_by_step": hybrid_std_arr,
    }


def plot_kl_model_vs_two_bayes_linear_transition_across_k(
    k_values,
    mode: str = "train",
    num_samples: int = 1024,
    steps=None,
    eps: float = 1e-12,
    exp_name_kwargs: Optional[dict] = None,
    show_colorbar: bool = True,
    show_ylabel: bool = True,
    use_cache: bool = True,
    force_recompute: bool = False,
    figsize: tuple = (10, 5),
    show: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Visualize the transition from exact-Bayes-like to Gaussian-new-like behavior across k.

    Relative plot:
      hue = log10(KL_exact / KL_approx)
      alpha = absolute closeness based on min(KL_exact, KL_approx)

    Absolute plot:
      log10(min(KL_exact, KL_approx))
    """
    from matplotlib import colors
    from icl.utils.unified_interface import get_exp_name

    if exp_name_kwargs is None:
        exp_name_kwargs = {}

    def _centers_to_edges(vals: np.ndarray) -> np.ndarray:
        vals = np.asarray(vals, dtype=float)
        if vals.ndim != 1 or vals.size == 0:
            raise ValueError("Need a non-empty 1D array of centers.")
        if vals.size == 1:
            width = 1.0 if vals[0] == 0 else max(1.0, abs(vals[0]) * 0.05)
            return np.asarray([vals[0] - width, vals[0] + width], dtype=float)
        mids = 0.5 * (vals[:-1] + vals[1:])
        left = vals[0] - (mids[0] - vals[0])
        right = vals[-1] + (vals[-1] - mids[-1])
        return np.concatenate(([left], mids, [right]))

    curves = {}
    exp_names = {}
    all_steps = set()
    for k in k_values:
        exp_name = get_exp_name("linear", k, **exp_name_kwargs)
        exp_names[k] = exp_name
        try:
            out = plot_kl_model_vs_two_bayes_linear_over_steps(
                exp_name=exp_name,
                mode=mode,
                num_samples=num_samples,
                steps=steps,
                eps=eps,
                use_cache=use_cache,
                force_recompute=force_recompute,
                show=False,
                verbose=verbose,
            )
            steps_k = np.asarray(out["steps"], dtype=int)
            exact_k = np.asarray(out["kl_exact_mean_by_step"], dtype=float)
            hybrid_k = np.asarray(out["kl_hybrid_mean_by_step"], dtype=float)
            curves[k] = {
                "steps": steps_k,
                "kl_exact_mean_by_step": exact_k,
                "kl_hybrid_mean_by_step": hybrid_k,
                "log10_ratio": np.log(np.clip(exact_k, eps, None) / np.clip(hybrid_k, eps, None)),
            }
            all_steps.update(steps_k.tolist())
        except Exception as e:
            if verbose:
                print(f"[warn] k={k} failed: {e}")

    ks = sorted(curves.keys())
    if len(ks) == 0:
        raise RuntimeError("No k succeeded. Check exp_name availability/checkpoints.")

    step_grid = np.asarray(sorted(all_steps), dtype=float)
    n_k = len(ks)
    n_s = len(step_grid)
    log_ratio = np.full((n_k, n_s), np.nan, dtype=float)
    exact_mat = np.full((n_k, n_s), np.nan, dtype=float)
    hybrid_mat = np.full((n_k, n_s), np.nan, dtype=float)
    step_to_idx = {int(s): i for i, s in enumerate(step_grid)}

    for row, k in enumerate(ks):
        d = curves[k]
        for step_val, exact_val, hybrid_val, ratio_val in zip(
            d["steps"],
            d["kl_exact_mean_by_step"],
            d["kl_hybrid_mean_by_step"],
            d["log10_ratio"],
        ):
            col = step_to_idx[int(step_val)]
            exact_mat[row, col] = exact_val
            hybrid_mat[row, col] = hybrid_val
            log_ratio[row, col] = ratio_val

    rel_vmax = 2.0
    rel_norm = colors.TwoSlopeNorm(vmin=-rel_vmax, vcenter=0.0, vmax=rel_vmax)

    best_kl = np.minimum(exact_mat, hybrid_mat)
    log10_best = np.full_like(best_kl, np.nan, dtype=float)
    valid_best = np.isfinite(best_kl)
    log10_best[valid_best] = np.log10(np.clip(best_kl[valid_best], eps, None))
    finite_best = log10_best[np.isfinite(log10_best)]
    if finite_best.size > 0:
        abs_vmin = float(np.min(finite_best))
        abs_vmax = float(np.max(finite_best))
    else:
        abs_vmin, abs_vmax = -6.0, 0.0
    if abs_vmax - abs_vmin < 1e-12:
        abs_vmax = abs_vmin + 1e-12
    abs_norm = colors.Normalize(vmin=abs_vmin, vmax=abs_vmax)

    closeness = np.full_like(log10_best, np.nan, dtype=float)
    closeness[valid_best] = 1.0 - (
        (log10_best[valid_best] - abs_vmin) / (abs_vmax - abs_vmin)
    )
    closeness = np.clip(closeness, 0.0, 1.0)
    closeness_gamma = np.full_like(closeness, np.nan, dtype=float)
    closeness_gamma[valid_best] = np.power(closeness[valid_best], 0.6)
    alpha_matrix = np.full_like(log10_best, 0.0, dtype=float)
    alpha_matrix[valid_best] = 0.42 + 0.58 * closeness_gamma[valid_best]

    y_centers = np.asarray(ks, dtype=float)
    x_edges = _centers_to_edges(step_grid)
    y_edges = _centers_to_edges(y_centers)
    log_ratio_ma = np.ma.masked_invalid(log_ratio)
    log10_best_ma = np.ma.masked_invalid(log10_best)

    fig_rel, ax_rel = plt.subplots(1, 1, figsize=figsize)
    cmap_rel = plt.get_cmap("RdBu_r").copy()
    cmap_rel.set_bad(color="#f0f0f0")
    mesh_rel = ax_rel.pcolormesh(
        x_edges,
        y_edges,
        log_ratio_ma,
        cmap=cmap_rel,
        norm=rel_norm,
        shading="auto",
    )
    rel_facecolors = cmap_rel(rel_norm(log_ratio_ma.filled(0.0)))
    rel_facecolors[..., -1] = alpha_matrix
    mesh_rel.set_facecolor(rel_facecolors.reshape(-1, 4))
    if show_colorbar:
        cbar_rel = fig_rel.colorbar(mesh_rel, ax=ax_rel, pad=0.02)
        cbar_rel.set_label(r"$\log(\mathrm{KL}_{\mathrm{exact}} / \mathrm{KL}_{\mathrm{approx}})$", fontsize=14)
        cbar_rel.ax.tick_params(labelsize=12)
    ax_rel.set_xlabel("Training Step", fontsize=11)
    if show_ylabel:
        ax_rel.set_ylabel(r"$\log_2(\mathrm{Number\ of\ Minority\ Tasks})$", fontsize=11)
    ax_rel.set_yticks(y_centers)
    ax_rel.grid(False)
    fig_rel.tight_layout()

    fig_abs, ax_abs = plt.subplots(1, 1, figsize=figsize)
    cmap_abs = plt.get_cmap("viridis_r").copy()
    cmap_abs.set_bad(color="#f0f0f0")
    mesh_abs = ax_abs.pcolormesh(
        x_edges,
        y_edges,
        log10_best_ma,
        cmap=cmap_abs,
        norm=abs_norm,
        shading="auto",
    )
    if show_colorbar:
        cbar_abs = fig_abs.colorbar(mesh_abs, ax=ax_abs, pad=0.02)
        cbar_abs.set_label(r"$\log_{10}(\min(\mathrm{KL}_{\mathrm{exact}}, \mathrm{KL}_{\mathrm{approx}}))$", fontsize=14)
        cbar_abs.ax.tick_params(labelsize=12)
    ax_abs.set_title("Best Absolute KL", fontsize=12)
    ax_abs.set_xlabel("Training Step", fontsize=11)
    if show_ylabel:
        ax_abs.set_ylabel(r"$\log_2(\mathrm{Number\ of\ Minority\ Tasks})$", fontsize=11)
    ax_abs.set_yticks(y_centers)
    ax_abs.grid(False)
    fig_abs.tight_layout()

    if show:
        plt.figure(fig_rel.number)
        plt.show()
        plt.figure(fig_abs.number)
        plt.show()
    else:
        plt.close(fig_rel)
        plt.close(fig_abs)

    return {
        "fig": fig_rel,
        "fig_rel": fig_rel,
        "fig_abs": fig_abs,
        "ax": ax_rel,
        "ax_rel": ax_rel,
        "ax_abs": ax_abs,
        "axes": (ax_rel, ax_abs),
        "mode": mode,
        "show_colorbar": bool(show_colorbar),
        "k_values_loaded": ks,
        "exp_names_by_k": exp_names,
        "step_grid": step_grid.astype(int),
        "log10_ratio_matrix": log_ratio,
        "best_kl_matrix": best_kl,
        "log10_best_kl_matrix": log10_best,
        "alpha_matrix": alpha_matrix,
        "kl_exact_matrix": exact_mat,
        "kl_hybrid_matrix": hybrid_mat,
        "curves_by_k": curves,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_task_posterior(
    exp_name: str,
    n_plots: int = 3,
    uniform_prior: bool = True,
    include_gaussian: bool = True,
    major_only: bool = False,
    max_positions: Optional[int] = None,
    figsize: tuple = (12, 4),
    title: Optional[str] = None,
    show: bool = True,
) -> dict:
    """Plot P(z=k | x₁:t, y₁:t₋₁) over time for random sequences.

    Optionally includes a Gaussian "new task" hypothesis (K+1-th component).
    """
    import matplotlib.pyplot as plt
    from icl.linear.linear_path_utils import load_model_task_config
    _, train_task, config = load_model_task_config(exp_name)
    device = config.device
    n_points = int(config.task.n_points)
    B = max(n_plots, 4)

    has_minor = (train_task.n_minor_tasks > 0
                 and train_task.minor_pool is not None)
    K_major = int(train_task.n_tasks)
    K_minor = int(train_task.n_minor_tasks) if has_minor else 0
    T_plot = max_positions if max_positions is not None else n_points

    # Temporarily override task state for sampling.
    temp_overrides = {"batch_size": B}
    if uniform_prior and has_minor:
        temp_overrides["p_minor"] = K_minor / (K_major + K_minor)
    with _temporary_task_attributes(train_task, **temp_overrides):
        demo_data, _, demo_target = train_task.sample_batch(step=42, is_eval=major_only)
        demo_data = demo_data.to(device)
        demo_target = demo_target.to(device)

    # ---- compute posterior ----
    if include_gaussian:
        K_total = K_major + K_minor + 1
        posterior_time = torch.zeros(B, T_plot, K_total, device=device)
        for t in range(T_plot):
            posterior_time[:, t, :] = task_posterior_with_gaussian_linear_regression(
                train_task, demo_data[:, :t+1], demo_target[:, :t+1],
                include_minor=has_minor,
            )
    else:
        post_all = task_posterior_over_time_linear_regression(
            train_task, demo_data, demo_target, include_minor=has_minor,
        )
        posterior_time = post_all[:, :T_plot, :]

    # ---- plot ----
    major_colors = [plt.cm.Blues(0.3 + 0.6 * i / max(K_major - 1, 1))
                    for i in range(K_major)]
    minor_colors = [plt.cm.Reds(0.3 + 0.6 * i / max(K_minor - 1, 1))
                    for i in range(K_minor)]
    ts = np.arange(T_plot)

    fig, axes = plt.subplots(n_plots, 1,
                             figsize=(figsize[0], figsize[1] * n_plots),
                             squeeze=False)

    for idx, ax in enumerate(axes.flat):
        post = posterior_time[idx].cpu().numpy()

        for k in range(K_major):
            ax.plot(ts, post[:, k], color=major_colors[k], alpha=0.8, lw=1.5,
                    label="Major" if idx == 0 and k == 0 else None)

        for k in range(K_minor):
            ax.plot(ts, post[:, K_major + k], color=minor_colors[k], alpha=0.6, lw=1.0,
                    label="Minor" if idx == 0 and k == 0 else None)

        if include_gaussian:
            ax.plot(ts, post[:, -1], color="black", ls="--", lw=2.0, alpha=0.9,
                    label="P(Gaussian)" if idx == 0 else None)

        ax.set(ylabel="P(Z=k | X,Y)", xlim=(0, max(T_plot - 1, 1)),
               ylim=(-0.02, 1.02), title=f"Sample {idx + 1}")
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")

    axes.flat[-1].set_xlabel("Position t")
    fig.suptitle("", fontsize=18, y=1.01)
    fig.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)

    posteriors = [posterior_time[i].cpu() for i in range(n_plots)]
    return {
        "posteriors": posteriors,
        "fig": fig,
        "axes": list(axes.flat),
        "demo_data": demo_data.cpu(),
        "demo_target": demo_target.cpu(),
    }


def plot_val_r2_across_layers(
    exp_name: str,
    layers: Optional[list] = None,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Sweep OLS probe h ≈ πW + b across layers; plot R² and partial-R² bars.

    Also prints design-matrix collinearity diagnostics (condition number,
    VIF, GVIF, pairwise R² between feature groups).
    """
    import matplotlib.pyplot as plt
    from icl.linear.linear_path_utils import load_model_task_config

    if layers is None:
        _, _, config = load_model_task_config(exp_name)
        layers = list(range(config.model.n_layer))

    def _diag(r, key, default=float("nan")):
        d = r.get("diagnostics")
        return d[key] if d is not None else default

    all_results = {}
    for layer in layers:
        logger.info(f"[sweep] running layer {layer} ...")
        all_results[layer] = train_linear_hidden_predictor(
            exp_name=exp_name, layer=layer, print_summary=False, **kwargs,
        )

    # Print design matrix diagnostics (layer-independent, use first layer)
    first_res = all_results[layers[0]]
    dd = first_res.get("diagnostics", {}).get("design_diagnostics")
    if dd is not None:
        _r2 = "\u00b2"
        print(f"\n{'=' * 65}")
        print(f"  Design Matrix Collinearity Summary (layer-independent)")
        print(f"{'=' * 65}")
        print(f"  Condition number: {dd['condition_number']:.2e}")
        print(f"  Features: posterior={dd['n_features']['posterior']}  "
              f"token={dd['n_features']['token']}  "
              f"logit={dd['n_features']['logit']}  "
              f"(total={dd['n_features']['total']})")
        print()
        print(f"  {'Group':<12} {'dims':>5} {'VIF':>10} "
              f"{'GVIF^(1/2p)':>12} {'R{r2} from rest':>14}".format(r2=_r2))
        print(f"  {'-' * 55}")
        for grp in ("posterior", "token", "logit"):
            ndim = dd["n_features"][grp]
            vif_val = dd["vif"][grp]
            gvif_val = dd["gvif_adj"][grp]
            r2_rest = dd["r2_from_rest"][grp]
            print(f"  {grp:<12} {ndim:>5d} {vif_val:>10.2f} "
                  f"{gvif_val:>12.4f} {r2_rest:>14.4f}")
        print()
        pw = dd["pairwise_r2"]
        _arrow = "\u2194"
        print(f"  Pairwise R{_r2} between feature groups:")
        print(f"    post{_arrow}tok   = {pw['post_tok']:.4f}")
        print(f"    post{_arrow}logit = {pw['post_logit']:.4f}")
        print(f"    tok{_arrow}logit  = {pw['tok_logit']:.4f}")
        print(f"{'=' * 65}\n")

    x = np.arange(len(layers))
    layer_labels = [str(l) for l in layers]

    marginal_metrics = {
        "Joint": lambda r: r["val_r2"],
        "Posterior only": lambda r: _diag(r, "r2_posterior_only"),
        "Token only": lambda r: _diag(r, "r2_token_only"),
        "Logit only": lambda r: _diag(r, "r2_logit_only"),
    }
    partial_metrics = {
        "Posterior | rest": lambda r: _diag(r, "partial_r2_posterior"),
        "Token | rest": lambda r: _diag(r, "partial_r2_token"),
        "Logit | rest": lambda r: _diag(r, "partial_r2_logit"),
    }

    panels = [marginal_metrics, partial_metrics]
    panel_titles = ["Val R\u00b2 (marginal)", "Partial R\u00b2 (unique contribution)"]
    panel_ylabels = ["R\u00b2", "Partial R\u00b2"]

    fig, axes = plt.subplots(1, 2, figsize=(max(5 * len(layers) / 4, 12), 5))

    for ax, metrics, ptitle, ylabel in zip(
        axes, panels, panel_titles, panel_ylabels,
    ):
        n_m = len(metrics)
        bw = 0.8 / n_m
        colors = plt.cm.Set2(np.linspace(0, 0.8, n_m))
        for i, (name, ext) in enumerate(metrics.items()):
            vals = [ext(all_results[l]) for l in layers]
            offset = (i - (n_m - 1) / 2) * bw
            bars = ax.bar(x + offset, vals, bw, label=name, color=colors[i])
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    txt = f"{v:.2f}".lstrip("0") if 0 < abs(v) < 1 else f"{v:.2f}"
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height(), txt,
                            ha="center", va="bottom", fontsize=9)
        ax.set(xlabel="Layer", ylabel=ylabel, title=ptitle)
        ax.set_xticks(x, layer_labels)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    if title:
        fig.suptitle("", fontsize=18, y=1.02)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    _show_or_close(fig, show)

    return fig, all_results


def plot_id_ood_loss(
    k_list,
    logx: bool = True,
    figsize: tuple = (12, 5),
    show: bool = True,
    pad: Optional[str] = "none",
    exp_name_kwargs: Optional[dict] = None,
    id_source: str = "id",
    ood_source: str = "ood",
    metric_key: str = "Transformer | True",
    noise_scale: Optional[float] = 0.5,
) -> dict:
    """Plot ID and OOD training loss vs step for multiple k values.

    The plotted loss is RMSE(model predictions, noisy targets). With additive
    Gaussian noise ε ~ N(0, σ²), the best achievable RMSE is σ (noise_scale).
    If ``noise_scale`` is set, a horizontal reference line is drawn at that level.

    Parameters
    ----------
    k_list : sequence of int
        Each k → ``get_exp_name("linear", k, pad=pad, **exp_name_kwargs)``.
    exp_name_kwargs : dict, optional
        Extra kwargs passed to ``get_exp_name`` for linear (e.g. ``n_layer``,
        ``total_steps``, ``warmup_steps``, ``batch_size``). Must match the
        config used when training the experiments.
    id_source : str
        Eval block used for ID curve. Common choices:
        ``"id"`` (new format), ``"pretrain_false"``, ``"pretrain_true"``,
        ``"coin_id"``.  ``"id"`` maps to ``eval/IDLoss``.
    ood_source : str
        Eval block used for OOD curve. Common choices:
        ``"ood"`` (new format), ``"latent_false"``, ``"latent_true"``,
        ``"coin_ood"``.  ``"ood"`` maps to ``eval/OODLoss``.
    metric_key : str
        Metric name inside eval blocks (default ``"Transformer | True"``).
    noise_scale : float, optional
        Standard deviation of observation noise (default 0.5). Best achievable
        RMSE equals this value. If provided, a horizontal "noise floor" line is
        drawn. Set to None to disable.

    Returns
    -------
    dict
        ``{'fig', 'ax1', 'ax2', 'results'}``.
    """
    import json
    import os
    import matplotlib.pyplot as plt
    from icl.utils.unified_interface import get_exp_name

    if exp_name_kwargs is None:
        exp_name_kwargs = {}

    def _resolve_linear_exp_dir(exp_name: str) -> str:
        # Try both common contexts: repo root and notebooks/.
        candidates = [
            os.path.join("results", "linear", exp_name),
            os.path.join("..", "results", "linear", exp_name),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        # Fall back to root-style path for clear error reporting.
        return candidates[0]

    def _to_mean_curve(values):
        # values is usually a list over eval checkpoints; each item may be scalar
        # or a list/tensor of per-sample errors.
        curve = []
        for v in values:
            arr = np.asarray(v, dtype=float)
            curve.append(float(arr.mean()) if arr.size > 0 else float("nan"))
        return np.asarray(curve, dtype=float)

    def _resolve_source_key(source: str) -> Optional[str]:
        s = str(source).strip().lower()
        mapping = {
            "pretrain_false": "eval/Pretrain_false",
            "pretrain_true": "eval/Pretrain_true",
            "latent_false": "eval/Latent_false",
            "latent_true": "eval/Latent_true",
            "coin_id": "eval/IDLoss",
            "coin_ood": "eval/OODLoss",
            "id": "eval/IDLoss",
            "ood": "eval/OODLoss",
            "minor": "eval/MinorLoss",
        }
        if s in mapping:
            return mapping[s]
        if source in mapping.values():
            return source
        return None

    def _extract_curve_from_block(eval_block: dict):
        if not isinstance(eval_block, dict):
            return None
        if metric_key in eval_block:
            return _to_mean_curve(eval_block[metric_key])
        if "Transformer | True" in eval_block:
            return _to_mean_curve(eval_block["Transformer | True"])
        # Fallback: first Transformer-like metric.
        for k_metric, vals in eval_block.items():
            if str(k_metric).startswith("Transformer |"):
                return _to_mean_curve(vals)
        return None

    _OLD_FORMAT_FALLBACKS = {
        "eval/IDLoss": ["eval/Pretrain_true", "eval/Pretrain_false"],
        "eval/OODLoss": ["eval/Latent_true", "eval/Latent_false"],
    }

    def _extract_curve(data_dict: dict, source: str):
        src_key = _resolve_source_key(source)
        if src_key is None:
            raise KeyError(
                f"Unknown source {source!r}. Use one of "
                f"id/ood/minor/pretrain_false/pretrain_true/latent_false/latent_true/coin_id/coin_ood."
            )

        # Flat new-format entries (eval/IDLoss, eval/OODLoss, eval/MinorLoss)
        if src_key in ("eval/IDLoss", "eval/OODLoss", "eval/MinorLoss"):
            vals = data_dict.get(src_key, None)
            if vals is not None:
                return np.asarray(vals, dtype=float), src_key
            # Fall back to old nested format
            for fallback_key in _OLD_FORMAT_FALLBACKS.get(src_key, []):
                block = data_dict.get(fallback_key, None)
                curve = _extract_curve_from_block(block)
                if curve is not None:
                    return curve, fallback_key
            return None, src_key

        # Nested linear-style eval block
        block = data_dict.get(src_key, None)
        curve = _extract_curve_from_block(block)
        if curve is not None:
            return curve, src_key

        # Case-insensitive fallback in case key casing drifts.
        src_key_l = src_key.lower()
        for k_data, v_data in data_dict.items():
            if isinstance(k_data, str) and k_data.lower() == src_key_l:
                curve = _extract_curve_from_block(v_data)
                if curve is not None:
                    return curve, k_data

        return None, src_key

    results = {}
    for k in k_list:
        exp_name = get_exp_name("linear", k, pad=pad, **exp_name_kwargs)
        n_minor_tasks = 2 ** k if k >= 0 else 0

        try:
            exp_dir = _resolve_linear_exp_dir(exp_name)
            log_path = os.path.join(exp_dir, "log.json")
            with open(log_path, "r") as f:
                data = json.load(f)

            train_steps = np.asarray(data.get("eval/step", data.get("train/step", [])), dtype=float)
            id_loss, id_key_used = _extract_curve(data, id_source)
            ood_loss, ood_key_used = _extract_curve(data, ood_source)

            if id_loss is None or ood_loss is None:
                raise KeyError(
                    f"Could not find requested ID/OOD curves in log.json "
                    f"(id_source={id_source!r}, ood_source={ood_source!r}, metric={metric_key!r})."
                )

            # Align lengths robustly
            if train_steps.size == 0:
                L = min(len(id_loss), len(ood_loss))
                train_steps = np.arange(1, L + 1, dtype=float)
            L = min(len(train_steps), len(id_loss), len(ood_loss))
            train_steps = np.asarray(train_steps[:L], dtype=float)
            id_loss = np.asarray(id_loss[:L], dtype=float)
            ood_loss = np.asarray(ood_loss[:L], dtype=float)

            results[k] = dict(
                n_minor=n_minor_tasks,
                train_steps=train_steps,
                id_loss=id_loss,
                ood_loss=ood_loss,
                id_source_used=id_key_used,
                ood_source_used=ood_key_used,
                metric_key_used=metric_key,
            )
        except Exception as e:
            logger.warning(f"Could not load k={k}: {e}")

    ks_sorted = sorted(results.keys())
    if not ks_sorted:
        logger.warning("No experiments loaded successfully.")
        return {}

    # Evenly sample colormap (avoid dark purples): use 15%--90% of viridis
    nk = len(ks_sorted)
    cmap = plt.get_cmap("viridis")
    color_map = {}
    for i, k in enumerate(ks_sorted):
        t = 0.15 + 0.75 * (i / max(1, nk - 1))
        color_map[k] = cmap(t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, which="major", alpha=0.15, linestyle="-")
        ax.grid(True, which="minor", alpha=0.06, linestyle=":")
        if noise_scale is not None:
            ax.axhline(noise_scale, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="_nolegend_")

    lw, alpha = 1.5, 0.85
    fs_label, fs_tick = 14, 12

    for k in ks_sorted:
        d = results[k]
        c = color_map[k]
        xs, y_id = d["train_steps"], d["id_loss"]
        if logx:
            mask = xs > 0
            xs, y_id = xs[mask], y_id[mask]
        if xs.size == 0:
            continue
        ax1.plot(xs, y_id, color=c, linewidth=lw, alpha=alpha, label=str(k))

    ax1.set_title("In-distribution", fontsize=fs_label)
    if logx:
        ax1.set_xscale("log")
    ax1.set_xlabel("Training Step", fontsize=fs_label)
    ax1.set_ylabel("Loss (RMSE)", fontsize=fs_label)
    ax1.tick_params(labelsize=fs_tick)

    handles, labels = ax1.get_legend_handles_labels()
    ax2.legend(
        handles,
        labels,
        title=r"$\log_2(n_{\mathrm{minor}})$",
        fontsize=fs_tick,
        title_fontsize=fs_label,
        frameon=True,
        framealpha=0.95,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    for k in ks_sorted:
        d = results[k]
        c = color_map[k]
        xs, y_ood = d["train_steps"], d["ood_loss"]
        if logx:
            mask = xs > 0
            xs, y_ood = xs[mask], y_ood[mask]
        if xs.size == 0:
            continue
        ax2.plot(xs, y_ood, color=c, linewidth=lw, alpha=alpha)

    ax2.set_title("Out-of-distribution", fontsize=fs_label)
    if logx:
        ax2.set_xscale("log")
    ax2.set_xlabel("Training Step", fontsize=fs_label)
    ax2.set_ylabel("")
    ax2.tick_params(labelsize=fs_tick)

    fig.subplots_adjust(wspace=0.05, right=0.78)
    fig.tight_layout()
    if show:
        plt.show()

    return {"fig": fig, "ax1": ax1, "ax2": ax2, "results": results}


def plot_task_variance(
    exp_name: str,
    layers: Optional[list] = None,
    batch_size: int = 16,
    positions_of_interest: Optional[list] = None,
    n_minor: int = 64,
    n_ood: int = 30,
    step: Optional[int] = None,
    verbose: bool = False,
    eps: float = 1e-8,
    chunk_size: int = 16,
    figsize: tuple = (8, 6),
    log_x: bool = True,
    show: bool = True,
    title: Optional[str] = None,
) -> dict:
    """Compute and plot normalised task variance (P2) for the linear regression task.

    Convenience wrapper around :func:`get_task_variance`.

    Parameters
    ----------
    exp_name : str
    layers : list, optional
        ``None`` -> all layers.
    batch_size : int
    positions_of_interest : list, optional
        ``None`` -> all positions.
    n_minor : int
        Number of minor tasks.
    n_ood : int
        Number of OOD tasks.
    step : int, optional
    verbose : bool
    eps : float
    chunk_size : int
    figsize : tuple
    log_x : bool
    show : bool
    title : str, optional

    Returns
    -------
    dict
        ``{'all_hiddens', 'demo_data', 'results_dict',
        'plotting_data', 'fig', 'ax'}``.
    """
    import matplotlib.pyplot as plt
    from icl.linear.analysis.probes import get_task_variance

    all_hiddens, demo_data, results_dict, plotting_data = get_task_variance(
        exp_name=exp_name,
        layers=layers,
        chunk_size=chunk_size,
        step=step,
        positions_of_interest=positions_of_interest,
        batch_size=batch_size,
        n_minor=n_minor,
        n_ood=n_ood,
        verbose=verbose,
        eps=eps,
    )

    fig, ax = plt.subplots(figsize=figsize)

    for layer_idx in plotting_data["layers"]:
        positions = plotting_data["positions"]
        var_pos_norm = plotting_data["var_pos_norm"][layer_idx]
        ax.plot(
            positions, var_pos_norm, "o-",
            label=f"Layer {layer_idx}",
            linewidth=2, markersize=6,
        )

    ax.set_xlabel("Position" + (" (log scale)" if log_x else ""), fontsize=16)
    ax.set_ylabel("Normalized Task Variance", fontsize=16)
    if log_x:
        ax.set_xscale("log")
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title("", fontsize=18)

    _show_or_close(fig, show)

    return {
        "all_hiddens": all_hiddens,
        "demo_data": demo_data,
        "results_dict": results_dict,
        "plotting_data": plotting_data,
        "fig": fig,
        "ax": ax,
    }
