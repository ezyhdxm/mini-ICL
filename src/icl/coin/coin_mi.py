import math

import torch
from typing import Optional


def _entropy_bits(probs, eps=1e-12):
    """Compute entropy in bits from a probability vector (1D tensor or numpy)."""
    p = torch.as_tensor(probs, dtype=torch.float64).clamp(min=eps)
    return -(p * p.log2()).sum().item()


@torch.no_grad()
def estimate_mi_task_vs_history_coin(
    task=None,
    *,
    exp_name: str = None,
    num_samples: int = 8192,
    min_position: int = 0,
    max_position: int = None,
    uniform_prior: bool = True,
    dirichlet_alpha: float = 1.0,
    eps: float = 1e-12,
    seed=None,
):
    """
    Estimate MI = H(next | Dirichlet belief state) - E_{Z|ctx}[H(next | Z)]
    for the coin task, averaged over positions and samples.

    - H(next | Dirichlet belief state): entropy of the predictive distribution
      under a symmetric Dirichlet(alpha) prior, using only the observed counts
      as the sufficient statistic (no task pool knowledge).
      P(next=j | counts) = (count_j + alpha) / (total + K*alpha)
    - E_{Z|ctx}[H(next | Z)]: expected per-task entropy, conditioned on the
      task identity Z via the Bayesian posterior P(Z=k | context). This is the
      residual entropy when you know which task generated the data.

    MI measures how much knowing the task pool (and resolving task identity
    through the posterior) reduces next-token entropy compared to the
    uninformed Dirichlet belief state.

    Parameters
    ----------
    task : Coins, optional
        A Coins sampler instance. If None, loaded from exp_name.
    exp_name : str, optional
        Experiment name (folder under results/coin/).
    num_samples : int, default=8192
        Number of sequences to generate.
    min_position : int, default=0
        Only include real-token positions >= this index.
    max_position : int, optional
        Only include real-token positions < this index. None = no limit.
    uniform_prior : bool, default=True
        If True, sets p_minor for uniform task prior.
    dirichlet_alpha : float, default=1.0
        Concentration parameter for the symmetric Dirichlet prior used in the
        uninformed belief state. alpha=1.0 gives Laplace smoothing (uniform prior).
    eps : float, default=1e-12
        Numerical floor.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict with keys:
        - MI_bits: H(next | Dirichlet) - E_Z[H(next | Z)], in bits
        - H_next_given_dirichlet_bits: avg H using Dirichlet belief state
        - H_next_given_bayes_bits: avg E_{Z|ctx}[H(next | Z)] via posterior
        - n_pairs_used: total (sample, position) pairs
        - uniform_prior: whether uniform prior was used
    """
    if isinstance(task, str):
        if exp_name is not None:
            raise ValueError("Got exp_name both as positional arg and keyword arg.")
        exp_name = task
        task = None

    if task is None and exp_name is None:
        raise ValueError("Must provide either task or exp_name.")

    if task is None:
        import icl.utils.notebook_utils as nu
        _, task, _ = nu.load_everything("coin", exp_name)

    if seed is not None:
        torch.manual_seed(seed)

    original_p_minor = float(getattr(task, 'p_minor', 0.0))
    if uniform_prior and task.n_minor_tasks > 0:
        task.p_minor = task.n_minor_tasks / (task.n_major_tasks + task.n_minor_tasks)

    device = task.device

    K = task.num_states
    maj = task.major_p.to(device=device, dtype=torch.float64)
    Kmaj = maj.shape[0]
    use_minor = task.n_minor_tasks > 0 and task.minor_p is not None
    if use_minor:
        minp = task.minor_p.to(device=device, dtype=torch.float64)
        Kmin = minp.shape[0]
        P = torch.cat([maj, minp], dim=0)  # (Ktot, K)
    else:
        Kmin = 0
        P = maj
    Ktot = P.shape[0]

    if Kmin == 0:
        prior = torch.full((Ktot,), 1.0 / max(1, Kmaj), device=device, dtype=torch.float64)
    else:
        p0 = float(task.p_minor)
        prior_major = (1.0 - p0) / max(1, Kmaj)
        prior_minor = p0 / max(1, Kmin)
        prior = torch.cat([
            torch.full((Kmaj,), prior_major, device=device, dtype=torch.float64),
            torch.full((Kmin,), prior_minor, device=device, dtype=torch.float64),
        ])
    prior = prior.clamp(min=eps)
    log_prior = prior.log()
    logP = P.clamp(min=eps).log()  # (Ktot, K)

    task_entropies = torch.zeros(Ktot, dtype=torch.float64, device=device)
    for k in range(Ktot):
        pk = P[k].clamp(min=eps)
        task_entropies[k] = -(pk * pk.log2()).sum()

    samples_raw = task.generate(mode="train", num_samples=num_samples, epochs=1)
    if isinstance(samples_raw, tuple):
        samples_raw = samples_raw[0]
    if samples_raw.dim() == 3:
        samples_raw = samples_raw.squeeze(0)
    samples = samples_raw.to(device)

    x = samples.long()
    B, L_real = x.shape

    lo = min_position
    hi = max_position if max_position is not None else L_real

    mi_sum = 0.0
    h_dir_sum = 0.0
    h_bayes_sum = 0.0
    n_pairs = 0

    alpha = float(dirichlet_alpha)
    counts = torch.zeros((B, K), device=device, dtype=torch.float64)

    for t in range(L_real):
        counts.scatter_add_(
            dim=-1, index=x[:, t:t + 1].long(),
            src=torch.ones((B, 1), device=device, dtype=torch.float64),
        )

        if t < lo or t >= hi:
            continue

        total = counts.sum(dim=-1, keepdim=True)  # (B, 1)
        p_dir = (counts + alpha) / (total + K * alpha)  # (B, K)
        p_dir_clamped = p_dir.clamp(min=eps)
        H_dir = -(p_dir_clamped * p_dir_clamped.log2()).sum(dim=-1)  # (B,)

        loglik = counts @ logP.T  # (B, Ktot)
        unnorm = loglik + log_prior.unsqueeze(0)
        log_post = unnorm - torch.logsumexp(unnorm, dim=-1, keepdim=True)
        post = log_post.exp()  # (B, Ktot)

        H_bayes = (post * task_entropies.unsqueeze(0)).sum(dim=-1)  # (B,)

        mi_t = H_dir - H_bayes  # (B,)

        mi_sum += mi_t.sum().item()
        h_dir_sum += H_dir.sum().item()
        h_bayes_sum += H_bayes.sum().item()
        n_pairs += B

    task.p_minor = original_p_minor

    if n_pairs == 0:
        raise RuntimeError("No (sample, position) pairs in the specified range.")

    return {
        "MI_bits": mi_sum / n_pairs,
        "H_next_given_dirichlet_bits": h_dir_sum / n_pairs,
        "H_next_given_bayes_bits": h_bayes_sum / n_pairs,
        "n_pairs_used": n_pairs,
        "uniform_prior": uniform_prior,
    }


def plot_mi_vs_k_coin(
    k_values,
    num_samples: int = 8192,
    min_position: int = 0,
    max_position: int = None,
    uniform_prior: bool = True,
    vocab_size=None,
    seed=None,
    figsize: tuple = (10, 6),
    save_path=None,
    show: bool = True,
    verbose: bool = False,
):
    """
    Compute I(Z; next_token | x_{1:t}) for each k and plot MI vs k for the coin task.

    Parameters
    ----------
    k_values : list of int
        k values where number of minor tasks = 2^k.
    num_samples : int, default=8192
        Sequences per experiment.
    min_position : int, default=0
        Only include real-token positions >= this.
    max_position : int, optional
        Only include real-token positions < this. None = no limit.
    uniform_prior : bool, default=True
        Use uniform task prior.
    vocab_size : int, optional
        Vocabulary size for get_exp_name.
    seed : int, optional
        Random seed (incremented per k).
    figsize : tuple, default=(10, 6)
        Figure size.
    save_path : str, optional
        Path to save figure.
    show : bool, default=True
        Display the plot.
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    dict with 'k_values', 'mi_bits', 'H_history', 'H_task', 'fig'.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from icl.utils.unified_interface import get_exp_name

    mi_bits = []
    h_dirichlet = []
    h_bayes = []

    for i, k in enumerate(k_values):
        exp_name = get_exp_name("coin", k, vocab_size=vocab_size)
        if verbose:
            print(f"Processing k={k} (2^k={2**k} minor tasks), exp={exp_name}")

        try:
            s = seed + i if seed is not None else None
            result = estimate_mi_task_vs_history_coin(
                exp_name=exp_name, num_samples=num_samples,
                min_position=min_position, max_position=max_position,
                uniform_prior=uniform_prior, seed=s,
            )
            mi_bits.append(result["MI_bits"])
            h_dirichlet.append(result["H_next_given_dirichlet_bits"])
            h_bayes.append(result["H_next_given_bayes_bits"])

            if verbose:
                print(f"  MI={result['MI_bits']:.4f} bits, "
                      f"H(Dirichlet)={result['H_next_given_dirichlet_bits']:.4f}, "
                      f"H(Bayes)={result['H_next_given_bayes_bits']:.4f}")
        except Exception as e:
            print(f"Warning: k={k} failed: {e}")
            mi_bits.append(float('nan'))
            h_dirichlet.append(float('nan'))
            h_bayes.append(float('nan'))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(k_values, mi_bits, 'o-', linewidth=2, markersize=8, color='blue',
            label='H(Dirichlet) - H(Bayes)')
    ax.plot(k_values, h_dirichlet, 's--', linewidth=1.5, markersize=6, color='gray',
            alpha=0.7, label='H(next | Dirichlet belief state)')
    ax.plot(k_values, h_bayes, '^--', linewidth=1.5, markersize=6, color='orange',
            alpha=0.7, label='H(next | Bayesian predictor)')
    ax.set_xlabel('k (log2 of number of minor tasks)', fontsize=12)
    ax.set_ylabel('Bits', fontsize=12)
    pos_label = ""
    if min_position > 0 or max_position is not None:
        pos_label = f", positions [{min_position}:{max_position}]"
    ax.set_title(f'Benefit of Task Pool: Dirichlet vs Bayesian (Coin Task{pos_label})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        'k_values': k_values,
        'mi_bits': mi_bits,
        'H_dirichlet': h_dirichlet,
        'H_bayes': h_bayes,
        'fig': fig,
    }
