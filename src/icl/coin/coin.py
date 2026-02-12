import torch
from typing import Tuple, Optional


import torch
from typing import Tuple, Optional


class Coins:
    """
    Categorical (multinomial) sampler over `num_states` symbols.
    Optionally pads into an even/odd token pattern with pad id = num_states.
    """

    def __init__(self, config):
        self.seq_len = int(config.seq_len)
        self.pad = bool(config.task.pad)

        if self.pad:
            self.num_states = int(config.vocab_size) - 1
        else:
            self.num_states = int(config.vocab_size)

        self.batch_size = int(config.batch_size)
        self.eval_size = int(config.eval_size)
        self.test_size = int(config.test_size)
        self.device = config.device
        self.seed = int(getattr(config, "seed", 0))

        # Dirichlet concentration parameter for sampling categorical probabilities
        self.alpha = float(getattr(config.task, "alpha", 1.0))

        self.n_major_tasks = int(getattr(config.task, "n_tasks", 0))
        self.n_minor_tasks = int(getattr(config.task, "n_minor_tasks", 0))
        self.p_minor = float(getattr(config.task, "p_minor", 1e-12))

        # Pre-sample major/minor pools (optional)
        self.major_p = self._maybe_sample_task_pool(self.n_major_tasks, pool_name="major")
        self.minor_p = self._maybe_sample_task_pool(self.n_minor_tasks, pool_name="minor")

    # -----------------------------
    # Probability pool generation
    # -----------------------------
    def _dirichlet(self) -> torch.distributions.Dirichlet:
        return torch.distributions.Dirichlet(
            torch.full((self.num_states,), self.alpha, device=self.device)
        )

    def _sample_disjoint_uniform_pool(
        self,
        n_tasks: int,
        *,
        eps_outside: float = 0.001,
    ) -> torch.Tensor:
        """
        Create n_tasks probability vectors with (almost) disjoint supports.
        Each row is ~uniform on its support.

        - eps_outside = 0.0  -> hard disjoint supports (exact zeros)
        - eps_outside > 0.0  -> almost disjoint, numerically safer
        """
        K = self.num_states
        n_tasks = int(n_tasks)

        if n_tasks <= 0:
            return self._dirichlet().sample((1,))

        # If K < n_tasks, disjoint supports are impossible; fallback to Dirichlet.
        if K < n_tasks:
            return self._dirichlet().sample((n_tasks,))

        # Reproducible permutation of states
        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed)
        perm = torch.randperm(K, generator=g, device=self.device)

        # Split perm into n_tasks chunks as evenly as possible
        sizes = [(K // n_tasks) + (1 if i < (K % n_tasks) else 0) for i in range(n_tasks)]
        # Ensure each task gets at least 1 token in its support
        assert all(s > 0 for s in sizes)

        out = torch.full((n_tasks, K), float(eps_outside), device=self.device, dtype=torch.float32)

        start = 0
        for i, sz in enumerate(sizes):
            idx = perm[start : start + sz]
            # Uniform-ish on support by putting constant mass there (then normalize)
            out[i, idx] = 1.0
            start += sz

        out = out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return out

    def _maybe_sample_task_pool(self, n_tasks: int, *, pool_name: str) -> torch.Tensor:
        n_tasks = int(n_tasks)
        if n_tasks <= 0:
            return self._dirichlet().sample((1,))

        # SPECIAL: make 3 major tasks highly distinctive / almost disjoint
        if pool_name == "major" and n_tasks == 3:
            return self._sample_disjoint_uniform_pool(n_tasks, eps_outside=0.005)

        # Default: sample each task from a Dirichlet
        return self._dirichlet().sample((n_tasks,))  # (n_tasks, K)

    def to(self, device):
        self.device = device
        self.major_p = self.major_p.to(device)
        self.minor_p = self.minor_p.to(device)

    @property
    def total_tasks(self) -> int:
        return int(self.n_major_tasks + self.n_minor_tasks)

    # -----------------------------
    # Sampling utilities
    # -----------------------------
    def _sample_categorical_sequence(self, probs: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        probs: (N, K) row-stochastic (or close)
        returns: (N, seq_len) int64 tokens in {0,...,K-1}
        """
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        idx = torch.multinomial(probs, num_samples=seq_len, replacement=True)  # (N, seq_len)
        return idx.to(torch.long)

    def _num_samples_for_mode(self, mode: str, epochs: int, num_samples: Optional[int]) -> int:
        if mode == "train":
            base = self.batch_size if num_samples is None else int(num_samples)
        elif mode == "test":
            base = self.test_size if num_samples is None else int(num_samples)
        elif mode in ["testing", "major", "minor"]:
            base = 1 if num_samples is None else int(num_samples)
        elif mode in ["eval", "ood"]:
            base = self.eval_size if num_samples is None else int(num_samples)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return base * int(epochs)

    def _sample_from_pool(
        self,
        pool: Optional[torch.Tensor],
        n_tasks: int,
        *,
        num_samples: int,
        task: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          probs:   (N, K)
          latent:  (N,) task indices in [0, n_tasks) (meaningful only if n_tasks>0)
        """
        n_tasks = int(n_tasks)
        num_samples = int(num_samples)

        if n_tasks <= 0 or pool is None:
            # Fallback: fresh Dirichlet per sample (OOD-style)
            probs = self._dirichlet().sample((num_samples,))
            latent = torch.full((num_samples,), -1, dtype=torch.long, device=self.device)
            return probs, latent

        if task is None:
            latent = torch.randint(high=n_tasks, size=(num_samples,), device=self.device)
        else:
            if not (0 <= task < n_tasks):
                raise ValueError(f"task id out of range: task={task}, n_tasks={n_tasks}")
            latent = torch.full((num_samples,), task, dtype=torch.long, device=self.device)

        probs = pool[latent]  # (N, K)
        return probs, latent

    # -----------------------------
    # Main API
    # -----------------------------
    def generate(
        self,
        epochs: int = 1,
        mode: str = "train",
        task: Optional[int] = None,
        num_samples: Optional[int] = None,
    ):
        """
        Modes:
          - "major":   sample only from major pool
          - "minor":   sample only from minor pool; if n_minor_tasks==0 -> random Dirichlet per sample
          - "ood":     random Dirichlet per sample
          - "train"/"test"/"testing"/"eval": mixture of major/minor pools using p_minor when both exist.
            If one pool doesn't exist, uses the other.
            If neither exist, falls back to random Dirichlet per sample.
        """
        assert mode in ["train", "test", "testing", "eval", "ood", "major", "minor"], f"Invalid mode: {mode}"

        epochs = int(epochs)
        N = self._num_samples_for_mode(mode, epochs, num_samples)

        # --- choose probs (N,K) and latent (N,) if applicable ---
        if mode == "major":
            probs, latent_major = self._sample_from_pool(
                self.major_p, self.n_major_tasks, num_samples=N, task=task
            )
            latent = latent_major  # (N,) or -1s if no major pool

        elif mode == "minor":
            # if n_minor_tasks == 0, _sample_from_pool falls back to random Dirichlet per sample
            probs, latent_minor = self._sample_from_pool(
                self.minor_p, self.n_minor_tasks, num_samples=N, task=task
            )
            # For consistency with the old indexing scheme:
            latent = torch.where(
                latent_minor >= 0,
                latent_minor + self.n_major_tasks,
                latent_minor,
            )

        elif mode == "ood":
            probs = self._dirichlet().sample((N,))
            latent = None

        else:
            # train/test/testing/eval
            if task is not None:
                # honor explicit task id if provided, with fallback behavior if pools missing
                if task < 0:
                    raise ValueError("task must be nonnegative when provided.")
                if task < self.n_major_tasks:
                    probs, latent_major = self._sample_from_pool(
                        self.major_p, self.n_major_tasks, num_samples=N, task=task
                    )
                    latent = latent_major
                else:
                    minor_id = task - self.n_major_tasks
                    probs, latent_minor = self._sample_from_pool(
                        self.minor_p, self.n_minor_tasks, num_samples=N, task=minor_id
                    )
                    latent = torch.where(
                        latent_minor >= 0,
                        latent_minor + self.n_major_tasks,
                        latent_minor,
                    )
            else:
                # mixture logic
                has_major = self.n_major_tasks > 0
                has_minor = self.n_minor_tasks > 0

                if has_major and has_minor:
                    latent_major = torch.randint(self.n_major_tasks, (N,), device=self.device)
                    latent_minor = torch.randint(self.n_minor_tasks, (N,), device=self.device)
                    trans_major = self.major_p[latent_major]
                    trans_minor = self.minor_p[latent_minor]

                    use_minor = (torch.rand(N, device=self.device) < self.p_minor)
                    probs = torch.where(use_minor[:, None], trans_minor, trans_major)
                    latent = torch.where(use_minor, self.n_major_tasks + latent_minor, latent_major)

                elif has_major:
                    # sample major tasks from the major pool
                    latent_major = torch.randint(self.n_major_tasks, (N,), device=self.device)
                    trans_major = self.major_p[latent_major]  # (N, K)

                    # sample "minor" tasks as fresh Dirichlet draws (OOD-style), since no minor pool exists
                    trans_minor = self._dirichlet().sample((N,))  # (N, K)

                    # mix using p_minor
                    use_minor = (torch.rand(N, device=self.device) < self.p_minor)  # (N,)
                    probs = torch.where(use_minor[:, None], trans_minor, trans_major)

                    # latent: major id if major chosen; -1 if the "minor"/OOD draw was chosen
                    latent = torch.where(
                        use_minor,
                        torch.full((N,), -1, dtype=torch.long, device=self.device),
                        latent_major,
                    )

                elif has_minor:
                    probs, latent_minor = self._sample_from_pool(
                        self.minor_p, self.n_minor_tasks, num_samples=N, task=None
                    )
                    latent = self.n_major_tasks + latent_minor

                else:
                    # no pools at all -> random dirichlet per sample
                    probs = self._dirichlet().sample((N,))
                    latent = None

        # --- sample tokens ---
        samples = self._sample_categorical_sequence(probs, self.seq_len)  # (N, seq_len)

        # --- optional padding pattern ---
        if self.pad:
            padded = torch.zeros((N, 2 * self.seq_len - 1), dtype=torch.long, device=self.device)
            padded[:, 1::2] = self.num_states  # pad token id
            padded[:, ::2] = samples
            samples = padded

        # --- return shapes / extras consistent with your old API ---
        if mode == "train":
            # probs returned as (epochs, batch, K)
            return (
                samples.reshape(epochs, N // epochs, -1),
                probs.reshape(epochs, N // epochs, -1),
            )

        if mode in ["testing", "major", "minor"] and task is None:
            return samples, probs, latent

        return samples, probs



@torch.no_grad()
def task_posterior_coins(
    sampler,                     # Coins instance
    samples: torch.Tensor,        # (N,L) or (E,N,L)
    probs_major: Optional[torch.Tensor] = None,  # optional override: (Kmaj, K)
    probs_minor: Optional[torch.Tensor] = None,  # optional override: (Kmin, K)
    *,
    include_minor: bool = True,
    return_log: bool = False,
    eps: float = 1e-30,
) -> torch.Tensor:
    """
    Compute posterior P(Z=k | x_{1:T}) for Coins (0th-order Markov / i.i.d. categorical).

    Prior matches sampler.generate():
      - If n_minor_tasks == 0: uniform over major tasks.
      - Else: P(minor)=p_minor, P(major)=1-p_minor, then uniform within pool.

    If sampler.pad == True, ignores pad positions (uses samples[..., ::2]).

    Args:
        sampler: Coins instance.
        samples: Long tensor (N,L) or (E,N,L). If padded, L = 2*seq_len-1.
        probs_major/probs_minor: optional probability pools to override sampler.major_p/minor_p.
        include_minor: whether to include minor pool in posterior (if available).
        return_log: if True, return log posterior.
        eps: numerical floor.

    Returns:
        posterior: (..., Ktot) where ... is (N,) or (E,N,),
                   Ktot = n_major_tasks + (n_minor_tasks if included and available).
    """
    device = samples.device
    dtype = torch.float32

    # Accept (N,L) or (E,N,L)
    if samples.dim() == 2:
        samples_ = samples.unsqueeze(0)  # (1,N,L)
        squeeze0 = True
    elif samples.dim() == 3:
        samples_ = samples
        squeeze0 = False
    else:
        raise ValueError(f"samples must have shape (N,L) or (E,N,L), got {samples.shape}")

    E, N, _ = samples_.shape

    # Strip padding: real tokens are at even positions
    if getattr(sampler, "pad", False):
        x = samples_[..., ::2]  # (E,N,T)
    else:
        x = samples_
    E, N, T = x.shape

    K = int(sampler.num_states)  # number of real symbols

    # Fetch pools
    maj = probs_major if probs_major is not None else sampler.major_p
    minp = probs_minor if probs_minor is not None else sampler.minor_p

    if maj is None or int(sampler.n_major_tasks) <= 0:
        raise ValueError("Posterior requires a finite major pool (n_major_tasks > 0).")

    maj = maj.to(device=device, dtype=dtype)  # (Kmaj, K)
    Kmaj = maj.shape[0]
    if maj.shape[1] != K:
        raise ValueError(f"major_p has wrong num_states: {maj.shape[1]} != {K}")

    use_minor_pool = (
        include_minor
        and int(sampler.n_minor_tasks) > 0
        and (minp is not None)
        and (minp.shape[0] == int(sampler.n_minor_tasks))
    )
    if use_minor_pool:
        minp = minp.to(device=device, dtype=dtype)  # (Kmin, K)
        Kmin = minp.shape[0]
        if minp.shape[1] != K:
            raise ValueError(f"minor_p has wrong num_states: {minp.shape[1]} != {K}")
        P = torch.cat([maj, minp], dim=0)  # (Ktot, K)
    else:
        Kmin = 0
        P = maj  # (Kmaj, K)

    Ktot = P.shape[0]

    # Prior over tasks
    if Kmin == 0:
        prior = torch.full((Ktot,), 1.0 / max(1, Kmaj), device=device, dtype=dtype)
    else:
        p0 = float(sampler.p_minor)
        prior_major = (1.0 - p0) / max(1, Kmaj)
        prior_minor = p0 / max(1, Kmin)
        prior = torch.cat([
            torch.full((Kmaj,), prior_major, device=device, dtype=dtype),
            torch.full((Kmin,), prior_minor, device=device, dtype=dtype),
        ], dim=0)

    prior = torch.clamp(prior, min=eps)
    log_prior = prior.log()  # (Ktot,)

    # Compute counts c_j for each (E,N)
    counts = torch.zeros((E, N, K), device=device, dtype=dtype)
    counts.scatter_add_(
        dim=-1,
        index=x.long(),
        src=torch.ones((E, N, T), device=device, dtype=dtype),
    )  # (E,N,K)

    # Log-likelihood: sum_j c_j log p_task[j]
    logP = torch.log(torch.clamp(P, min=eps))         # (Ktot, K)
    loglik = torch.einsum("enk,tk->ent", counts, logP)  # (E,N,Ktot)

    # Posterior
    unnorm = loglik + log_prior.view(1, 1, Ktot)      # (E,N,Ktot)
    log_post = unnorm - torch.logsumexp(unnorm, dim=-1, keepdim=True)

    out = log_post if return_log else torch.exp(log_post)
    return out.squeeze(0) if squeeze0 else out


import math


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
    # Allow passing exp_name as first positional arg
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

    # Build pools and prior
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

    # Precompute per-task entropies: H(p_k) in bits for each task k
    task_entropies = torch.zeros(Ktot, dtype=torch.float64, device=device)
    for k in range(Ktot):
        pk = P[k].clamp(min=eps)
        task_entropies[k] = -(pk * pk.log2()).sum()

    # Generate samples
    samples_raw = task.generate(mode="train", num_samples=num_samples, epochs=1)
    if isinstance(samples_raw, tuple):
        samples_raw = samples_raw[0]
    if samples_raw.dim() == 3:
        samples_raw = samples_raw.squeeze(0)
    samples = samples_raw.to(device)

    # Real tokens
    if getattr(task, "pad", False):
        x = samples[:, ::2].long()
    else:
        x = samples.long()
    B, L_real = x.shape

    # Determine position range
    lo = min_position
    hi = max_position if max_position is not None else L_real

    # Accumulators
    mi_sum = 0.0
    h_dir_sum = 0.0
    h_bayes_sum = 0.0
    n_pairs = 0

    alpha = float(dirichlet_alpha)
    counts = torch.zeros((B, K), device=device, dtype=torch.float64)

    for t in range(L_real):
        # Update counts with token at position t
        counts.scatter_add_(
            dim=-1, index=x[:, t:t + 1].long(),
            src=torch.ones((B, 1), device=device, dtype=torch.float64),
        )

        if t < lo or t >= hi:
            continue

        # --- Dirichlet belief state (no task knowledge) ---
        # P_dir(next=j | counts) = (count_j + alpha) / (total + K*alpha)
        total = counts.sum(dim=-1, keepdim=True)  # (B, 1)
        p_dir = (counts + alpha) / (total + K * alpha)  # (B, K)
        p_dir_clamped = p_dir.clamp(min=eps)
        H_dir = -(p_dir_clamped * p_dir_clamped.log2()).sum(dim=-1)  # (B,)

        # --- Bayesian belief state (with task pool knowledge) ---
        # Posterior: P(Z=k | x_{1:t})
        loglik = counts @ logP.T  # (B, Ktot)
        unnorm = loglik + log_prior.unsqueeze(0)
        log_post = unnorm - torch.logsumexp(unnorm, dim=-1, keepdim=True)
        post = log_post.exp()  # (B, Ktot)

        # Condition on Z via the posterior: E_{Z|ctx}[H(p_k)]
        # = sum_k P(Z=k|ctx) * H(p_k)
        H_bayes = (post * task_entropies.unsqueeze(0)).sum(dim=-1)  # (B,)

        # MI = H(next | Dirichlet) - E_{Z|ctx}[H(next | Z)]
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
