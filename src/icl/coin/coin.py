import torch
from typing import Tuple, Optional

import pandas as pd
from itertools import product
from IPython.display import display

from collections import defaultdict

# from icl.latent_markov.latent_utils import generate_markov_chains

# config specifies the number of different transitions
# each time, we randomly sample a transition matrix to use



class Coins:
    """
    High-order Markov chain sampler with latent task structure.
    
    This class implements a flexible Markov chain data generator where each sequence
    is generated from one of multiple transition matrices (tasks). Supports:
    - High-order Markov chains (order > 1)
    - Major and minor task pools with configurable mixing
    - Banded transition matrices (sparse structure)
    - Padding for even/odd token patterns
    - Various generation modes (train, test, OOD, etc.)
    
    The key idea is that each sequence follows a specific transition matrix, allowing
    the study of in-context learning on multiple related tasks.
    """
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.pad = config.task.pad
        self.num_states = 2
        self.batch_size = config.batch_size
        self.eval_size = config.eval_size
        self.test_size = config.test_size
        self.device = config.device
        self.seed = config.seed # Seed for random number generation
        
        self.n_major_tasks = config.task.n_tasks # Total number of transition matrices
        self.n_minor_tasks = config.task.n_minor_tasks if 'n_minor_tasks' in config.task else 0
        self.p_minor = config.task.p_minor if 'p_minor' in config.task else 0.0 # Probability of tasks from the minor task pool
        self.major_p = torch.rand(self.n_major_tasks, device=self.device) if self.n_major_tasks > 0 else None
        self.minor_p = torch.rand(self.n_minor_tasks, device=self.device) if self.n_minor_tasks > 0 else None

    def to(self, device):
        """
        Move all tensors to the specified device.
        
        Args:
            device: Target device (e.g., 'cuda' or 'cpu')
        """
        self.device = device
        self.major_p = self.major_p.to(device) if self.major_p else None
        self.minor_p = self.minor_p.to(device) if self.minor_p else None 
    
    @property
    def total_tasks(self) -> int:
        return int(self.n_major_tasks + self.n_minor_tasks)
    
    # generate samples from the model
    def generate(self, 
                 epochs=1, 
                 mode:str="train",
                 task=None, 
                 num_samples: Optional[int] = None, 
                 )-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences from the Markov chain sampler.
        
        Supports multiple modes:
        - "train", "test", "eval": Sample from major/minor task pool with probability p_minor
        - "major": Sample only from major tasks
        - "minor": Sample only from minor tasks
        - "ood": Sample from out-of-distribution (random Dirichlet prior)
        - "testing": Like "major" but returns task labels
        
        Args:
            epochs: Number of epochs to generate (for batching)
            mode: Generation mode (see description above)
            task: Specific task ID to use, or None for random sampling
            num_samples: Number of samples to generate per epoch
            return_trans_mat: If True (for OOD mode), also return transition matrices
        
        Returns:
            samples: Generated sequences, shape depends on mode and epochs
            probs: Transition probabilities for the generated sequences (optional)
            Additional returns for certain modes (latent labels, trans_mat, etc.)
        """
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

        if mode == "major":
            if task is None:
                latent_major = torch.randint(high=self.n_major_tasks, size=(num_samples,), device=self.device)
            else:
                assert 0 <= task < self.n_major_tasks, "task id out of range"
                latent_major = torch.full((num_samples,), task, dtype=torch.long, device=self.device)
            assert self.major_p is not None, "No major tasks available."
            probs = self.major_p[latent_major] # Shape: (num_samples, num_states_order, num_states)
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
            probs = self.minor_p[latent_minor] # Shape: (num_samples, num_states_order, num_states)
            latent = self.n_major_tasks + latent_minor

        elif mode in ["train", "test", "testing", "eval"]:
            if task is None:
                latent_major = torch.randint(high=self.n_major_tasks, size=(num_samples,), device=self.device) # Shape: (num_samples,), randomly choose a transition matrix for each sample
                latent_minor = torch.randint(high=self.n_minor_tasks, size=(num_samples,), device=self.device) if self.n_minor_tasks > 0 else None
            else:
                assert 0 <= task < self.n_major_tasks + self.n_minor_tasks, "task id out of range"
                if task < self.n_major_tasks:
                    latent_major = torch.full((num_samples,), task, dtype=torch.long, device=self.device)
                    latent_minor = None
                else:
                    latent_major = None
                    latent_minor = torch.full((num_samples,), task - self.n_major_tasks, dtype=torch.long, device=self.device)
            
            trans_major = self.major_p[latent_major] if latent_major is not None else None  # (num_samples,)
            trans_minor = self.minor_p[latent_minor] if latent_minor is not None else None  # (num_samples,)

            if trans_major is not None and trans_minor is not None:
                use_minor = (torch.rand(num_samples, device=self.device) < self.p_minor)   # (num_samples,) bool

                probs = torch.where(use_minor, trans_minor, trans_major)  # (num_samples,)

                latent = torch.where(
                    use_minor,
                    self.n_major_tasks + latent_minor,   # (num_samples,)
                    latent_major                          # (num_samples,)
                )

            elif trans_major is not None:
                probs = trans_major
                latent = latent_major
            elif trans_minor is not None:
                probs = trans_minor
                latent = self.n_major_tasks + latent_minor
            else:
                raise ValueError("No transition matrices available.")
            
        
        elif mode == "ood" or self.n_major_tasks + self.n_minor_tasks == 0:
            probs = torch.rand((num_samples,), device=self.device) # Shape: (num_samples,)
        
        samples = torch.bernoulli(probs[:, None].expand(num_samples, self.seq_len))

        
        if self.pad:
            padded_samples = torch.zeros((num_samples, 2*self.seq_len-1), dtype=torch.long, device=self.device)
            padded_samples[:, 1::2] = self.num_states
            padded_samples[:, ::2] = samples
            samples = padded_samples

        if mode == "train":
            return samples.reshape(epochs, num_samples//epochs, -1), probs.reshape(epochs, num_samples//epochs, -1)

        if mode in ["testing", "major", "minor"] and task is None:
            return samples, probs, latent

        return samples, probs
    
    

class CoinIDBayes:
    """
    Bayesian predictor for in-distribution tasks using known transition matrices.
    
    Implements Bayesian inference to predict next tokens given a sequence,
    weighting by how well each transition matrix matches the observed sequence.
    Assumes the true transition matrix is one of the known candidates.
    """
    def __init__(self, trans_mat, device="cpu"):
        self.log_trans_mat = trans_mat.log().to(device) # (K, N, N)
        self.total_trans = trans_mat.size(0)
        self.num_states = trans_mat.size(1)
        self.device = device
    
    def predict(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Predict next token probabilities using Bayesian inference.
        
        Computes posterior predictive distribution by weighting each candidate
        transition matrix by its likelihood under the observed sequence.
        
        Args:
            samples: Input sequences of shape (B, T)
        
        Returns:
            Predicted probabilities of shape (B, T, N) for each position
        """
        K, N, _ = self.log_trans_mat.size()
        B, T = samples.size()
        preds = torch.zeros((B, T, self.num_states), dtype=torch.float, device=self.device)
        samples = samples.to(self.device)

        s0 = samples[:, 0]  # (B,)

        # Build index tensors:
        k_idx = torch.arange(K, device=self.device).view(K, 1, 1)      # (K, 1, 1)
        n_idx = torch.arange(N, device=self.device).view(1, 1, N)      # (1, 1, N)

        # Expand s0 to (1, B, 1) for broadcasting
        s0_expand = s0.view(1, B, 1)                                   # (1, B, 1)

        # Advanced indexing:
        log_trans_rows = self.log_trans_mat[k_idx, s0_expand, n_idx]   # (K, B, N)

        preds[:, 0] = log_trans_rows.exp().mean(dim=0) # (B, N)
        cumulative_log_probs = torch.zeros((self.total_trans, B), dtype=torch.float, device=self.device)

        log_trans_flat = self.log_trans_mat.view(K, N * N)  # (K, N*N)

        for t in range(samples.size(1)-1):
            s_t = samples[:, t] # (B,)
            s_tp1 = samples[:, t+1] # (B,)
            flat_indices = s_t * N + s_tp1  # (B,)
            
            cumulative_log_probs += log_trans_flat[k_idx.view(K,1), flat_indices.view(1,B)] # (K, B)
            s_tp1_expand = s_tp1.view(1, B, 1)  # (1, B, 1)
            curr = self.log_trans_mat[k_idx, s_tp1_expand, n_idx] # (K, B, N)
            log_numerator = torch.logsumexp(curr + cumulative_log_probs.unsqueeze(-1), dim=0)  # (B, N)
            log_denominator = torch.logsumexp(cumulative_log_probs.unsqueeze(-1), dim=0) # (B, 1)
            preds[:, t+1] = (log_numerator - log_denominator).exp()  # (B, N)

        return preds


class CoinOODBayes:
    """
    Bayesian predictor for out-of-distribution tasks using online estimation.
    
    Implements Bayesian inference with a Dirichlet prior, learning the transition
    matrix online as more data is observed. Used when the true transition matrix
    is unknown and potentially different from training tasks.
    """
    def __init__(self, num_states, alpha, device="cpu"):
        self.num_states = num_states
        self.device = device
        self.alpha = alpha
    
    def predict(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Predict next token probabilities using online Bayesian estimation.
        
        Accumulates transition counts and computes posterior predictive
        distribution with Dirichlet-multinomial conjugate prior.
        
        Args:
            samples: Input sequences of shape (B, T)
        
        Returns:
            Predicted probabilities of shape (B, T, N) for each position
        """
        samples = samples.to(self.device)
        B, T = samples.size()
        cumsums = torch.zeros((B, self.num_states, self.num_states), 
                              dtype=torch.float, device=self.device)
        
        preds = torch.zeros((B, T, self.num_states), dtype=torch.float, device=self.device)
        preds[:, 0] = 1.0 / self.num_states  # Uniform distribution for the first token
        b_idx = torch.arange(B, device=self.device)

        for t in range(T-1):
            s_t = samples[:, t]  # (B,)
            s_tp1 = samples[:, t+1]
            
            cumsums[b_idx, s_t, s_tp1] += 1
            preds[:, t+1] = (cumsums[b_idx, s_tp1] + self.alpha) / (cumsums[b_idx, s_tp1].sum(dim=-1, keepdim=True) + self.num_states * self.alpha)

        return preds

