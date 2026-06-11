"""Corrected two-modes Bayesian solutions for the uniform-over-K latent DGP.

The original KL baselines (GroupUniformKnownBayes / ThreeKnownPlusNewDirichletBayes)
assume the paper's DGP: 3 "known" majors + a minor/Dirichlet pool with specific
mixing. That does NOT match the uniform-over-K setup (k=-1, n_tasks=K), where each
sequence picks one of K major transition matrices uniformly at random.

The two correct Bayes-optimal predictors here are:

  - "known" (retrieval): the predictor that knows the K major transition matrices
    and assumes a uniform 1/K prior over them. Posterior-predictive
        p(x_{t+1}=j | x_{0:t}) = sum_k P(Z=k | x_{0:t}) * T_k(x_t, j),
    with P(Z=k|.) from task_posterior_over_time(include_minor=False). Optimal on
    in-distribution sequences.

  - "new" (learning / extrapolation): the predictor that treats the transition
    matrix as unknown, drawn from the Dirichlet(alpha) prior (the OOD generative
    process). Online posterior-predictive (add-alpha running counts). Optimal on
    out-of-distribution sequences, and the K -> infinity limit.

All predictives are returned aligned as (B, T-1, V): entry [b, t] predicts
x_{t+1} given x_{0:t}.
"""

import numpy as np
import torch

from icl.latent_markov.analysis.bayes import task_posterior_over_time


@torch.no_grad()
def known_uniform_predictive(sampler, samples: torch.Tensor) -> torch.Tensor:
    """Uniform-1/K known-pool posterior-predictive, shape (B, T-1, V)."""
    dev = samples.device
    B, T = samples.shape
    trans = sampler.major_trans_mat.to(dev)              # (K, V, V), order-1
    K, V, _ = trans.shape
    # P(Z=k | x_{0:t}) for t = 0 .. T-1 ; uniform 1/K prior over the K majors.
    post = task_posterior_over_time(sampler, samples, include_minor=False)
    post = post.reshape(B, T, K).to(dev)                 # (B, T, K)
    states = samples[:, :-1]                             # x_t for t=0..T-2  -> (B, T-1)
    # T_k(x_t, :) for every k: gather rows by current state.
    rows = trans[:, states, :]                           # (K, B, T-1, V)
    w = post[:, :-1, :]                                  # (B, T-1, K) posterior given x_{0:t}
    pred = torch.einsum("btk,kbtj->btj", w, rows)        # (B, T-1, V)
    return pred.clamp_min(1e-12)


@torch.no_grad()
def dirichlet_new_predictive(samples: torch.Tensor, V: int, alpha: float) -> torch.Tensor:
    """Online Dirichlet posterior-predictive (add-alpha counts), shape (B, T-1, V)."""
    dev = samples.device
    B, T = samples.shape
    counts = torch.zeros(B, V, V, device=dev)
    bidx = torch.arange(B, device=dev)
    pred = torch.empty(B, T - 1, V, device=dev)
    for t in range(T - 1):
        cur = samples[:, t]                               # state x_t
        row = counts[bidx, cur]                           # counts of x_t -> * so far
        pred[:, t, :] = (row + alpha) / (row.sum(-1, keepdim=True) + V * alpha)
        counts[bidx, cur, samples[:, t + 1]] += 1.0       # observe x_t -> x_{t+1}
    return pred.clamp_min(1e-12)


@torch.no_grad()
def model_predictive(model, samples: torch.Tensor, V: int) -> torch.Tensor:
    """Model next-token distribution aligned to (B, T-1, V)."""
    logits = model(samples)
    p = torch.softmax(logits, dim=-1)[..., :V]
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
    return p[:, :-1, :].clamp_min(1e-12)


@torch.no_grad()
def kl_model_to(p_model: torch.Tensor, p_ref: torch.Tensor) -> torch.Tensor:
    """KL(model || ref) per position, averaged over batch -> (T-1,)."""
    kl = (p_model * (p_model.log() - p_ref.log())).sum(-1)   # (B, T-1)
    return kl.mean(0)


@torch.no_grad()
def run_two_modes_kl(exp_name, modes=("major", "ood"), num_samples=256, show=False):
    """KL(model || each correct Bayes solution) vs position, per generation mode.

    Returns {'fig', 'results', 'modes'} where results[mode] has positions and the
    per-position KL to the known-pool (1/K) and Dirichlet-new solutions.
    """
    import matplotlib.pyplot as plt
    import icl.utils.notebook_utils as nu

    _, sampler, config = nu.load_everything("latent", exp_name)
    dev = config.device
    model, _ = nu.load_checkpoint(config, exp_name=exp_name, return_actual_step=True)
    model.eval().to(dev)
    if hasattr(sampler, "to"):
        sampler.to(dev)
    V = int(sampler.num_states)
    alpha = float(getattr(sampler, "alpha", 1.0))

    results = {}
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4.2), squeeze=False)
    for mi, mode in enumerate(modes):
        out = sampler.generate(mode=mode, num_samples=num_samples, epochs=1)
        x = out[0] if isinstance(out, (tuple, list)) else out
        if x.dim() == 3:
            x = x.squeeze(0)
        x = x.to(dev)

        p_model = model_predictive(model, x, V)
        kl_known = kl_model_to(p_model, known_uniform_predictive(sampler, x)).cpu().numpy()
        kl_new = kl_model_to(p_model, dirichlet_new_predictive(x, V, alpha)).cpu().numpy()
        pos = np.arange(len(kl_known))

        ax = axes[0][mi]
        ax.plot(pos, kl_known, color="#1f77b4", lw=1.8, label="KL(model || known-pool $1/K$)")
        ax.plot(pos, kl_new, color="#d62728", lw=1.8, label="KL(model || Dirichlet-new)")
        ax.set_title(f"{mode} mode", fontsize=11)
        ax.set_xlabel("Position $t$")
        ax.set_ylabel("KL")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        results[mode] = {"positions": pos, "kl_known_mean": kl_known, "kl_new_mean": kl_new}

    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {"fig": fig, "results": results, "modes": list(modes)}
