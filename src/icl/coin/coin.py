import torch
from typing import Tuple, Optional

# from icl.latent_markov.latent_utils import generate_markov_chains

# config specifies the number of different transitions
# each time, we randomly sample a transition matrix to use

class Coins:
    """
    Categorical (multinomial) sampler over `num_states` symbols.
    Optionally pads into an even/odd token pattern with pad id = num_states.
    """

    def __init__(self, config):
        self.seq_len = config.seq_len
        self.pad = config.task.pad

        if config.task.pad:
            self.num_states = config.vocab_size - 1
        else:
            self.num_states = config.vocab_size

        self.batch_size = config.batch_size
        self.eval_size = config.eval_size
        self.test_size = config.test_size
        self.device = config.device
        self.seed = config.seed

        # Dirichlet concentration parameter for sampling categorical probabilities
        self.alpha = float(config.task.alpha) if hasattr(config.task, "alpha") else 1.0

        self.n_major_tasks = int(config.task.n_tasks)
        self.n_minor_tasks = int(getattr(config.task, "n_minor_tasks", 0))
        self.p_minor = float(getattr(config.task, "p_minor", 0.0))

        if self.n_major_tasks > 0:
            dirichlet = torch.distributions.Dirichlet(
                torch.full((self.num_states,), self.alpha, device=self.device)
            )
            self.major_p = dirichlet.sample((self.n_major_tasks,))  # (n_major_tasks, K)
        else:
            self.major_p = None

        if self.n_minor_tasks > 0:
            dirichlet = torch.distributions.Dirichlet(
                torch.full((self.num_states,), self.alpha, device=self.device)
            )
            self.minor_p = dirichlet.sample((self.n_minor_tasks,))  # (n_minor_tasks, K)
        else:
            self.minor_p = None

    def to(self, device):
        self.device = device
        if self.major_p is not None:
            self.major_p = self.major_p.to(device)
        if self.minor_p is not None:
            self.minor_p = self.minor_p.to(device)

    @property
    def total_tasks(self) -> int:
        return int(self.n_major_tasks + self.n_minor_tasks)

    def _sample_categorical_sequence(self, probs: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        probs: (N, K) row-stochastic
        returns: (N, seq_len) int64 tokens in {0,...,K-1}
        """
        # torch.multinomial expects nonnegative rows; rows need not be perfectly normalized,
        # but it's good practice to ensure sum=1.
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # Sample N * seq_len draws, then reshape
        idx = torch.multinomial(probs, num_samples=seq_len, replacement=True)  # (N, seq_len)
        return idx.to(torch.long)

    def generate(
        self,
        epochs=1,
        mode: str = "train",
        task=None,
        num_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert mode in ["train", "test", "testing", "eval", "ood", "major", "minor"], f"Invalid mode: {mode}"

        if mode == "train":
            num_samples = num_samples if num_samples is not None else self.batch_size
        elif mode == "test":
            num_samples = num_samples if num_samples is not None else self.test_size
        elif mode in ["testing", "major", "minor"]:
            num_samples = num_samples if num_samples is not None else 1
        elif mode in ["eval", "ood"]:
            num_samples = num_samples if num_samples is not None else self.eval_size
        else:
            raise ValueError(f"Invalid mode: {mode}")

        num_samples *= epochs

        # probs will always be (num_samples, K)
        if mode == "major":
            if task is None:
                latent_major = torch.randint(high=self.n_major_tasks, size=(num_samples,), device=self.device)
            else:
                assert 0 <= task < self.n_major_tasks, "task id out of range"
                latent_major = torch.full((num_samples,), task, dtype=torch.long, device=self.device)

            assert self.major_p is not None, "No major tasks available."
            probs = self.major_p[latent_major]  # (N, K)
            latent = latent_major

        elif mode == "minor":
            if self.n_minor_tasks == 0:
                raise ValueError("No minor tasks available.")
            if task is None:
                latent_minor = torch.randint(high=self.n_minor_tasks, size=(num_samples,), device=self.device)
            else:
                assert 0 <= task < self.n_minor_tasks, "task id out of range"
                latent_minor = torch.full((num_samples,), task, dtype=torch.long, device=self.device)

            assert self.minor_p is not None, "No minor tasks available."
            probs = self.minor_p[latent_minor]  # (N, K)
            latent = self.n_major_tasks + latent_minor

        elif mode in ["train", "test", "testing", "eval"]:
            if task is None:
                latent_major = torch.randint(high=self.n_major_tasks, size=(num_samples,), device=self.device) if self.n_major_tasks > 0 else None
                latent_minor = torch.randint(high=self.n_minor_tasks, size=(num_samples,), device=self.device) if self.n_minor_tasks > 0 else None
            else:
                assert 0 <= task < self.n_major_tasks + self.n_minor_tasks, "task id out of range"
                if task < self.n_major_tasks:
                    latent_major = torch.full((num_samples,), task, dtype=torch.long, device=self.device)
                    latent_minor = None
                else:
                    latent_major = None
                    latent_minor = torch.full((num_samples,), task - self.n_major_tasks, dtype=torch.long, device=self.device)

            trans_major = self.major_p[latent_major] if latent_major is not None else None  # (N, K)
            trans_minor = self.minor_p[latent_minor] if latent_minor is not None else None  # (N, K)

            if trans_major is not None and trans_minor is not None:
                use_minor = (torch.rand(num_samples, device=self.device) < self.p_minor)  # (N,)
                probs = torch.where(use_minor[:, None], trans_minor, trans_major)         # (N, K)
                latent = torch.where(use_minor, self.n_major_tasks + latent_minor, latent_major)
            elif trans_major is not None:
                probs = trans_major
                latent = latent_major
            elif trans_minor is not None:
                probs = trans_minor
                latent = self.n_major_tasks + latent_minor
            else:
                raise ValueError("No transition matrices available.")

        elif mode == "ood" or self.n_major_tasks + self.n_minor_tasks == 0:
            # NEW: OOD -> sample a fresh categorical distribution per sample from Dirichlet
            dirichlet = torch.distributions.Dirichlet(
                torch.full((self.num_states,), self.alpha, device=self.device)
            )
            probs = dirichlet.sample((num_samples,))  # (N, K)

        # NEW: categorical sampling (tokens in {0..K-1})
        samples = self._sample_categorical_sequence(probs, self.seq_len)  # (N, seq_len)

        if self.pad:
            padded_samples = torch.zeros((num_samples, 2 * self.seq_len - 1), dtype=torch.long, device=self.device)
            padded_samples[:, 1::2] = self.num_states  # pad token id
            padded_samples[:, ::2] = samples
            samples = padded_samples

        if mode == "train":
            # probs returned as (epochs, batch, K)
            return (
                samples.reshape(epochs, num_samples // epochs, -1),
                probs.reshape(epochs, num_samples // epochs, -1),
            )

        if mode in ["testing", "major", "minor"] and task is None:
            return samples, probs, latent

        return samples, probs
    

