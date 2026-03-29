import torch
from typing import Tuple, Optional, List


def _solve_tilting_parameter(K: int, target_mean: float, tol: float = 1e-12) -> float:
    """
    Find eta such that the max-entropy distribution p_i = exp(eta*i)/Z
    over faces {1, ..., K} has mean equal to target_mean.
    """
    faces = torch.arange(1, K + 1, dtype=torch.float64)
    lo, hi = -50.0, 50.0
    for _ in range(300):
        mid = (lo + hi) / 2.0
        logp = mid * faces
        logp = logp - logp.max()
        p = torch.exp(logp)
        p = p / p.sum()
        current_mean = (faces * p).sum().item()
        if abs(current_mean - target_mean) < tol:
            break
        if current_mean < target_mean:
            lo = mid
        else:
            hi = mid
    return mid


def make_tilted_pool(K: int, target_means: List[float], device="cpu") -> torch.Tensor:
    """
    Build a (len(target_means), K+1) tensor of probability vectors.
    Each row is the max-entropy distribution over K-sided die faces {1,...,K}.
    Index 0 has probability 0; index i (1 <= i <= K) holds P(face = i).
    This way torch.multinomial produces tokens in {1,...,K} directly.
    """
    faces = torch.arange(1, K + 1, dtype=torch.float64)
    rows = []
    for mu in target_means:
        if not (1.0 <= mu <= float(K)):
            raise ValueError(
                f"target mean {mu} is out of range [1, {K}] for a {K}-sided die"
            )
        eta = _solve_tilting_parameter(K, mu)
        logp = eta * faces
        logp = logp - logp.max()
        p = torch.exp(logp)
        p = p / p.sum()
        p_full = torch.zeros(K + 1, dtype=torch.float32)
        p_full[1:] = p.float()
        rows.append(p_full)
    return torch.stack(rows).to(device)


def make_maxent_pool(
    K: int,
    target_means: List[float],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build a (len(target_means), K) tensor of probability vectors.

    Each row is the maximum-entropy distribution over 0-indexed tokens
    {0, ..., K-1} whose expected value equals ``target_means[i]``.
    The max-entropy solution is the exponential-family form
    ``p(x) ∝ exp(η · x)`` with ``η`` chosen to satisfy the mean
    constraint.  This is the distribution closest to uniform (in KL
    sense) that achieves the desired mean.

    Use with ``vocab_size = K``.
    """
    tokens = torch.arange(K, dtype=torch.float64)
    rows = []
    for mu in target_means:
        if not (0.0 <= mu <= float(K - 1)):
            raise ValueError(
                f"target mean {mu} is out of range [0, {K - 1}] "
                f"for tokens {{0, ..., {K - 1}}}"
            )
        lo, hi = -50.0, 50.0
        for _ in range(300):
            mid = (lo + hi) / 2.0
            logp = mid * tokens
            logp = logp - logp.logsumexp(0)
            current_mean = (logp.exp() * tokens).sum().item()
            if abs(current_mean - mu) < 1e-12:
                break
            if current_mean < mu:
                lo = mid
            else:
                hi = mid
        logp = mid * tokens
        logp = logp - logp.logsumexp(0)
        rows.append(logp.exp().float())
    return torch.stack(rows).to(device)


def make_partition_pool(
    K: int,
    target_means: List[float],
    eps_outside: Optional[float] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build a (len(target_means), K) tensor of probability vectors via
    soft contiguous partitioning of tokens {0, ..., K-1}.

    Tokens are split into ``n = len(target_means)`` contiguous groups as
    evenly as possible.  Each token gets a floor probability of
    ``eps_outside``; the remaining mass is concentrated on the partition
    tokens with weights chosen so that the overall expected token value
    matches ``target_means[i]`` as closely as possible.

    If ``eps_outside`` is None (default), it is set automatically to the
    largest feasible value (scaled by 0.9) that still allows all target
    means to be achieved.  This maximises the floor while respecting the
    mean constraints.

    Because the supports are near-disjoint, OOD samples (e.g. from a
    Dirichlet prior) project roughly equidistantly from all tasks.

    Use with ``vocab_size = K``.
    """
    n_tasks = len(target_means)
    if n_tasks == 0:
        raise ValueError("target_means must be non-empty")
    if K < n_tasks:
        raise ValueError(
            f"Cannot partition {K} tokens into {n_tasks} tasks"
        )

    sizes = [(K // n_tasks) + (1 if i < (K % n_tasks) else 0)
             for i in range(n_tasks)]
    partitions: List[List[int]] = []
    start = 0
    for sz in sizes:
        partitions.append(list(range(start, start + sz)))
        start += sz

    if eps_outside is None:
        S = K * (K - 1) / 2.0
        max_eps = 1.0 / K - 1e-9
        for i, mu in enumerate(target_means):
            lo = float(min(partitions[i]))
            hi = float(max(partitions[i]))
            d_lo = S - lo * K
            if d_lo > 1e-12 and mu - lo < d_lo * max_eps:
                max_eps = min(max_eps, (mu - lo) / d_lo)
            d_hi = S - hi * K
            if d_hi < -1e-12 and hi - mu < (-d_hi) * max_eps:
                max_eps = min(max_eps, (hi - mu) / (-d_hi))
        eps_outside = 0.9 * max(max_eps, 1e-6)

    tokens = torch.arange(K, dtype=torch.float64)
    bg_mean = eps_outside * tokens.sum().item()
    extra_mass = 1.0 - K * eps_outside
    if extra_mass <= 0:
        raise ValueError(
            f"eps_outside={eps_outside} too large for K={K} "
            f"(need K * eps_outside < 1)"
        )

    rows = []
    for i, mu in enumerate(target_means):
        part_indices = partitions[i]
        part_tokens = tokens[part_indices]
        lo_part = part_tokens.min().item()
        hi_part = part_tokens.max().item()

        needed = (mu - bg_mean) / extra_mass
        needed = max(lo_part, min(hi_part, needed))

        if len(part_indices) == 1:
            w = torch.ones(1, dtype=torch.float64)
        elif len(part_indices) == 2:
            a, b = part_tokens[0].item(), part_tokens[1].item()
            t = (b - needed) / (b - a)
            w = torch.tensor([t, 1.0 - t], dtype=torch.float64)
        else:
            eta = _solve_tilting_parameter(
                len(part_indices), needed - lo_part + 1.0
            )
            shifted = torch.arange(
                1, len(part_indices) + 1, dtype=torch.float64
            )
            logw = eta * shifted
            logw -= logw.max()
            w = logw.exp()
            w /= w.sum()

        p = torch.full((K,), eps_outside, dtype=torch.float64)
        for j, gi in enumerate(part_indices):
            p[gi] += w[j] * extra_mass
        rows.append(p.float())

    return torch.stack(rows).to(device)


class Coins:
    """
    Categorical (multinomial) sampler over `num_states` symbols.
    """

    def __init__(self, config):
        self.seq_len = int(config.seq_len)
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
        self.major_pool_type = str(getattr(config.task, "major_pool_type", "disjoint"))
        self.major_means = getattr(config.task, "major_means", None)
        self.major_seed = int(getattr(config.task, "major_seed", self.seed))

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

        # Major tasks: behaviour controlled by self.major_pool_type
        if pool_name == "major":
            if self.major_pool_type in ("tilted", "maxent", "partition"):
                if self.major_means is None:
                    raise ValueError(
                        f"major_pool_type='{self.major_pool_type}' requires "
                        "config.task.major_means to be set "
                        "(list of target means, one per major task)"
                    )
                means = list(self.major_means)
                if len(means) != n_tasks:
                    raise ValueError(
                        f"len(major_means)={len(means)} != n_tasks={n_tasks}"
                    )
                if self.major_pool_type == "tilted":
                    n_faces = self.num_states - 1
                    return make_tilted_pool(n_faces, means, device=self.device)
                if self.major_pool_type == "maxent":
                    return make_maxent_pool(
                        self.num_states, means, device=self.device
                    )
                return make_partition_pool(
                    self.num_states, means, device=self.device
                )
            if self.major_pool_type == "disjoint" and n_tasks == 3:
                return self._sample_disjoint_uniform_pool(n_tasks, eps_outside=0.005)
            # "dirichlet" or any other value, or disjoint with n_tasks != 3.
            # Sample on CPU with a seeded generator so major_p is identical across
            # runs that share the same config.seed (e.g. different k values), then
            # move to the target device.  Saving/restoring the CPU RNG state ensures
            # no side-effects on subsequent random calls (batch generation, etc.).
            cpu_rng_state = torch.get_rng_state()
            torch.manual_seed(self.major_seed)
            cpu_dist = torch.distributions.Dirichlet(
                torch.full((self.num_states,), self.alpha)
            )
            samples = cpu_dist.sample((n_tasks,)).to(self.device)
            torch.set_rng_state(cpu_rng_state)
            return samples

        # Minor/other: always sample from Dirichlet
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


# ── backward-compat re-exports ──────────────────────────────────────
from icl.coin.coin_posterior import task_posterior_coins  # noqa: F401,E402
from icl.coin.coin_bayes import KnownPoolBayesCoins, ThreeKnownPlusNewDirichletCoinBayes  # noqa: F401,E402
from icl.coin.coin_kl import (  # noqa: F401,E402
    plot_kl_model_vs_two_bayes_coin_over_steps,
    plot_kl_model_vs_two_bayes_coin_transition_across_k,
)
