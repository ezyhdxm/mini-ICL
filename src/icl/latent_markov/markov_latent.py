import torch
from typing import Tuple, Optional

import pandas as pd
from itertools import product
from IPython.display import display

from collections import defaultdict

# from icl.latent_markov.latent_utils import generate_markov_chains

# config specifies the number of different transitions
# each time, we randomly sample a transition matrix to use



class LatentMarkov:
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
        self.bandwidth = getattr(config.task, "bandwidth", None) 
        self.circular_band = getattr(config.task, "circular_band", False) 
        self.seq_len = config.seq_len
        self.pad = config.task.pad
        if self.pad:
            self.num_states = config.vocab_size - 1
        else:
            self.num_states = config.vocab_size
        self.batch_size = config.batch_size
        self.eval_size = config.eval_size
        self.test_size = config.test_size
        self.device = config.device
        if 'stationary' in config.task: # To be compatible with the old config
            self.random_stationary = config.task.stationary # Whether to use sampled stationary distribution
        else:
            self.random_stationary = False 
        if self.random_stationary: 
            raise NotImplementedError("Random stationary distribution is not implemented yet")
        
        self.alpha = config.task.alpha # Dirichlet prior for the transition matrix
        self.seed = config.seed # Seed for random number generation
        
        self.n_major_tasks = config.task.n_tasks # Total number of transition matrices
        self.n_minor_tasks = config.task.n_minor_tasks if 'n_minor_tasks' in config.task else 0
        self.p_minor = config.task.p_minor if 'p_minor' in config.task else 0.0 # Probability of tasks from the minor task pool
        self.order = config.task.order # Order of the Markov chain
        assert self.order > 0, "Not supported yet for order 0"
        self.num_states_order = self.num_states ** self.order # Number of states in the (high order) Markov chain

        self.dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device) * config.task.alpha)

        self.powers = (self.num_states ** torch.arange(self.order - 1, -1, -1, device=self.device)).long()
        
        if self.n_major_tasks > 0:
            if self.random_stationary is False:
                if self.bandwidth is None:
                    self.major_trans_mat = self.dirichlet_dist.sample((self.n_major_tasks, self.num_states_order,))
                    self.major_trans_mat /= self.major_trans_mat.sum(dim=-1, keepdim=True)
                else:
                    self.major_trans_mat = self._sample_far_banded_trans_mats(self.n_major_tasks)
                self.stationary = None
            else:
                raise NotImplementedError("Random stationary distribution with multiple transition matrices is not implemented yet.")
                self.trans_mat, self.stationary = generate_markov_chains(self.total_trans, 
                                                                         self.num_states, 
                                                                         self.alpha, 
                                                                         device=self.device,
                                                                         seed=self.seed)  # Shape: (topics, num_states_order, num_states)
        if self.n_minor_tasks > 0:
            if self.random_stationary is False:
                if self.bandwidth is None:
                    self.minor_trans_mat = self.dirichlet_dist.sample((self.n_minor_tasks, self.num_states_order,))
                    self.minor_trans_mat /= self.minor_trans_mat.sum(dim=-1, keepdim=True)
                else:
                    self.minor_trans_mat = self._sample_banded_trans_mats(self.n_minor_tasks)

    def _make_banded_mask(self) -> torch.Tensor:
        """
        Create a banded mask for transition matrices.
        
        For high-order Markov chains, each row corresponds to a state history.
        This mask restricts transitions to only states within the bandwidth of
        the last symbol in the history. Supports both regular and circular bands.
        
        Returns:
            Boolean mask of shape (num_states_order, num_states), True where transitions are allowed
        """
        if self.bandwidth is None:
            return torch.ones(self.num_states_order, self.num_states, device=self.device, dtype=torch.bool)
        
        k = int(self.bandwidth)
        rows = torch.arange(self.num_states_order, device=self.device)
        last_sym = rows % self.num_states
        cols = torch.arange(self.num_states, device=self.device)

        last_sym = last_sym.view(-1, 1)  # Shape: (num_states_order, 1)
        cols = cols.view(1, -1)          # Shape: (1, num_states)
        if self.circular_band:
            d = torch.abs(last_sym - cols)
            d = torch.minimum(d, self.num_states - d)
            mask = (d <= k)
        else:
            mask = (torch.abs(last_sym - cols) <= k)
        return mask
    
    def _sample_far_banded_trans_mats(self, n_mats: int) -> torch.Tensor:
        """
        Sample banded transition matrices that are far apart from each other.
        
        Uses Hellinger distance to select diverse transition matrices from a larger pool.
        This ensures the major tasks are well-separated, making them easier to distinguish.
        
        Args:
            n_mats: Number of matrices to sample
        
        Returns:
            Tensor of shape (n_mats, num_states_order, num_states) with diverse transition matrices
        """
        device = self.device
        dtype  = torch.float32 

        pool_factor = 8                 
        beta_low, beta_high = 1.5, 3.0  
        eps = 1e-12                     
        m_pool = max(n_mats, pool_factor * n_mats)

        mask = self._make_banded_mask()  # (num_states_order, num_states) bool
        mask_3d = mask.unsqueeze(0).expand(m_pool, -1, -1)  # (m_pool, L, S)
        L, S = self.num_states_order, self.num_states

        alpha = torch.as_tensor(self.alpha, device=device, dtype=dtype)
        gamma = torch.distributions.Gamma(alpha, torch.ones((), device=device, dtype=dtype))

        W = gamma.sample((m_pool, L, S)).to(device=device, dtype=dtype)
        W = W * mask_3d  # banded 约束

        row_sums = W.sum(dim=-1, keepdim=True)                       # (m_pool, L, 1)
        need_fix = (row_sums.squeeze(-1) <= 0)                       # (m_pool, L)
        if need_fix.any():
            rows = torch.arange(L, device=device)
            diag_col = (rows % S)
            eye_cols = torch.nn.functional.one_hot(diag_col, num_classes=S).to(dtype).to(device)  # (L, S)
            eye_cols = eye_cols.unsqueeze(0).expand(m_pool, -1, -1)                                # (m_pool, L, S)
            W = torch.where(need_fix.unsqueeze(-1), eye_cols, W)
            row_sums = W.sum(dim=-1, keepdim=True)

        P = W / (row_sums + eps)

        betas = (beta_low + (beta_high - beta_low) * torch.rand((m_pool, 1, 1), device=device, dtype=dtype))
        P = torch.clamp(P, min=eps)
        P = P ** betas
        P = P * mask_3d
        P = P / (P.sum(dim=-1, keepdim=True) + eps)  

        def pairwise_hellinger(M):  # M: (M, L, S)
            Z = torch.sqrt(torch.clamp(M, min=eps))  # (M, L, S)
            # ||Zi - Zj||^2 = ||Zi||^2 + ||Zj||^2 - 2 <Zi, Zj>
            norms = (Z**2).sum(-1, keepdim=True)           # (M, L, 1)
            # inner products:
            Zi = Z.unsqueeze(1)                             # (M,1,L,S)
            Zj = Z.unsqueeze(0)                             # (1,M,L,S)
            ip = (Zi * Zj).sum(-1)                          # (M,M,L)
            d2 = norms.squeeze(-1).unsqueeze(1) + norms.squeeze(-1).unsqueeze(0) - 2.0 * ip  # (M,M,L)
            d  = torch.sqrt(torch.clamp(d2, min=0.0)) * (0.5 ** 0.5)  # (M,M,L)
            return d.mean(dim=-1)  

        D = pairwise_hellinger(P)  # (m_pool, m_pool)
        D.fill_diagonal_(0.0)

        chosen = []

        avg_dist = (D.sum(dim=1) / (m_pool - 1 + eps))
        first = torch.argmax(avg_dist).item()
        chosen.append(first)

        while len(chosen) < n_mats:
            
            mask_sel = torch.zeros(m_pool, dtype=torch.bool, device=device)
            mask_sel[chosen] = True
            # shape: (num_unselected, num_selected)
            d_to_S = D[~mask_sel][:, mask_sel]
            
            min_d, _ = d_to_S.min(dim=1)
            
            idx_unselected = torch.nonzero(~mask_sel, as_tuple=False).squeeze(-1)
            pick_rel = torch.argmax(min_d).item()
            pick_abs = idx_unselected[pick_rel].item()
            chosen.append(pick_abs)

        P_out = P[chosen]  # (n_mats, L, S)
        return P_out

    def _sample_banded_trans_mats(self, n_mats: int) -> torch.Tensor:
        """
        Sample simple banded transition matrices from a Dirichlet prior.
        
        Creates banded transition matrices where each row samples from a Dirichlet
        distribution constrained by the banded mask. Used for minor tasks and OOD sampling.
        
        Args:
            n_mats: Number of matrices to sample
        
        Returns:
            Tensor of shape (n_mats, num_states_order, num_states) with random banded transition matrices
        """
        mask = self._make_banded_mask()  # (num_states_order, num_states) bool
        mask_3d = mask.unsqueeze(0).expand(n_mats, -1, -1)  # (n_mats, num_states_order, num_states)
        gamma = torch.distributions.Gamma(self.alpha * torch.ones((), device=self.device), torch.ones((), device=self.device))
        W = gamma.sample((n_mats, self.num_states_order, self.num_states))
        W = W * mask_3d

        row_sums = W.sum(dim=-1, keepdim=True)  # (n_mats, num_states_order, 1)
        need_fix = (row_sums.squeeze(-1) == 0)  # (n_mats, num_states_order)
        if need_fix.any():
            n = self.num_states
            rows = torch.arange(self.num_states_order, device=self.device)
            diag_col = (rows % n) 
            eye_cols = torch.nn.functional.one_hot(diag_col, num_classes=n).to(W.dtype).to(W.device)  # (num_states_order, n)
            eye_cols = eye_cols.unsqueeze(0).expand(n_mats, -1, -1)  # (n_mats, num_states_order, n)
            W = torch.where(need_fix.unsqueeze(-1), eye_cols, W)
            row_sums = W.sum(dim=-1, keepdim=True)
        P = W / row_sums
        return P

    def to(self, device):
        """
        Move all tensors to the specified device.
        
        Args:
            device: Target device (e.g., 'cuda' or 'cpu')
        """
        self.device = device
        if self.n_major_tasks > 0:
            self.major_trans_mat = self.major_trans_mat.to(device)
        if self.n_minor_tasks > 0:
            self.minor_trans_mat = self.minor_trans_mat.to(device)
        self.dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device) * self.alpha)
        if self.random_stationary:
            self.stationary = self.stationary.to(device)
        self.powers = (self.num_states ** torch.arange(self.order - 1, -1, -1, device=self.device)).long()
        return 

    def print_trans_mat(self, task_id):
        """
        Print the transition matrix for a given task_id as a formatted table.
        
        Displays the transition matrix with row labels showing the state history
        and column labels showing the next state.
        
        Args:
            task_id: int, the index of the task to display
        """
        perms = list(product(range(self.num_states), repeat=self.order))
        perms = [''.join(map(str, p)) for p in perms]
        df = pd.DataFrame(self.major_trans_mat[task_id].cpu(), 
                          index=perms, 
                          columns=[f"{i}" for i in range(self.num_states)])
        pd.set_option('display.float_format', '{:.3f}'.format)
        display(df)
    
    @property
    def total_trans(self) -> int:
        return int(self.n_major_tasks + self.n_minor_tasks)
    
    # generate samples from the model
    def generate(self, 
                 epochs=1, mode:str="train",
                 task=None, num_samples: Optional[int] = None, 
                 return_trans_mat=False)-> Tuple[torch.Tensor, torch.Tensor]:
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
        
        num_samples *= epochs

        if mode == "major":
            if task is None:
                latent_major = torch.randint(high=self.n_major_tasks, size=(num_samples,), device=self.device)
            else:
                assert 0 <= task < self.n_major_tasks, "task id out of range"
                latent_major = torch.full((num_samples,), task, dtype=torch.long, device=self.device)
            trans_mat = self.major_trans_mat[latent_major] # Shape: (num_samples, num_states_order, num_states)
            latent = latent_major
        
        elif mode == "minor":
            if self.n_minor_tasks == 0:
                raise ValueError("No minor tasks available.")
            if task is None:
                latent_minor = torch.randint(high=self.n_minor_tasks, size=(num_samples,), device=self.device)
            else:
                assert 0 <= task < self.n_minor_tasks, "task id out of range"
                latent_minor = torch.full((num_samples,), task, dtype=torch.long, device=self.device)
            trans_mat = self.minor_trans_mat[latent_minor] # Shape: (num_samples, num_states_order, num_states)
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
            
            trans_major = self.major_trans_mat[latent_major] if latent_major is not None else None # Shape: (num_samples, num_states_order, num_states)
            trans_minor = self.minor_trans_mat[latent_minor] if latent_minor is not None else None # Shape: (num_samples, num_states_order, num_states)
            # Build final trans_mat
            if trans_major is not None and trans_minor is not None:
                # flip a Bernoulli coin for each sample
                use_minor = (torch.rand(num_samples, device=self.device) < self.p_minor)
                trans_mat = torch.where(use_minor.view(-1, 1, 1), trans_minor, trans_major)
                latent = torch.where(
                    use_minor,
                    self.n_major_tasks + latent_minor,   # minor bank shifted by n_major_tasks
                    latent_major                          # major bank as-is
                )
            elif trans_major is not None:
                trans_mat = trans_major
                latent = latent_major
            elif trans_minor is not None:
                trans_mat = trans_minor
                latent = self.n_major_tasks + latent_minor
            else:
                raise ValueError("No transition matrices available.")
            
        
        elif mode == "ood" or self.n_major_tasks + self.n_minor_tasks == 0:
            if self.random_stationary is False:
                if self.bandwidth is None:
                    trans_mat = self.dirichlet_dist.sample((num_samples, self.num_states_order,))
                    trans_mat /= trans_mat.sum(dim=-1, keepdim=True)
                else:
                    trans_mat = self._sample_banded_trans_mats(num_samples)
                stationary = None
            else:
                raise NotImplementedError("Random stationary distribution is not implemented yet")
                trans_mat, stationary = generate_markov_chains(num_samples,
                                                               self.num_states, 
                                                               self.alpha, 
                                                               device=self.device,
                                                               seed=self.seed)
            
        # Initialize the samples tensor
        samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
        
        # Initialize the state (randomly choose starting states for each sequence)
        state = torch.randint(high=self.num_states, size=(num_samples, self.order), device=self.device)
        samples[:, :self.order] = state

        range_vec = torch.arange(num_samples, device=self.device) # Shape: (num_samples,)
            
        for t in range(self.order, self.seq_len):
            state_indices = torch.sum(state*self.powers, dim=1)

            probs = trans_mat[range_vec, state_indices]  # Shape: (num_samples, num_states)

            # Sample the next states for the entire batch
            next_states = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Update the sequence with the sampled next states
            samples[:, t] = next_states
            
            # Update the state window (shift left and append the new state)
            # state = torch.cat([state[:, 1:], next_states.unsqueeze(1)], dim=1)
            state[:, :-1] = state[:, 1:].clone()  # Shift left
            state[:, -1] = next_states    # Append new state
        
        if self.pad:
            padded_samples = torch.zeros((num_samples, 2*self.seq_len-1), dtype=torch.long, device=self.device)
            padded_samples[:, 1::2] = self.num_states
            padded_samples[:, ::2] = samples
            samples = padded_samples

        if mode == "train":
            return samples.reshape(epochs, num_samples//epochs, -1), probs.reshape(epochs, num_samples//epochs, -1)

        if mode in ["testing", "major", "minor"] and task is None:
            return samples, probs, latent
        
        if mode == "ood" and return_trans_mat:
            return samples, probs, trans_mat, stationary


        return samples, probs

    def get_task_matrix(self, global_task_id: int) -> torch.Tensor:
        """
        Map a global task id [0, total_trans) to a (num_states_order, num_states) matrix.
        
        Global IDs map to major tasks first (0 to n_major_tasks-1), then minor tasks.
        
        Args:
            global_task_id: Global task index in [0, total_trans)
        
        Returns:
            Transition matrix of shape (num_states_order, num_states)
        """
        assert 0 <= global_task_id < self.total_trans
        if global_task_id < self.n_major_tasks:
            return self.major_trans_mat[global_task_id]
        else:
            return self.minor_trans_mat[global_task_id - self.n_major_tasks]
    
    # generate summary statistics of the sampler
    def summary(self) -> defaultdict:
        """
        Generate unigram statistics (marginal token frequencies) for each task.
        
        Samples sequences from each transition matrix and computes the empirical
        unigram distribution. Useful for understanding the stationary distribution
        of each task.
        
        Returns:
            Dictionary mapping task IDs to unigram probability distributions
        """
        unigram_stats = defaultdict(torch.Tensor)
        num_samples = 1000
        device = self.device

        for i in range(self.total_trans):
            trans_i = self.get_task_matrix(i).to(device)  # (S^order, S)

            samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=device)
            state = torch.randint(high=self.num_states, size=(num_samples, self.order), device=device)
            samples[:, :self.order] = state

            # rolling index for speed (see tip below)
            base = self.num_states ** (self.order - 1)
            idx = torch.sum(state * self.powers, dim=1)

            for t in range(self.order, self.seq_len):
                probs = trans_i[idx]  # (num_samples, num_states)
                next_states = torch.multinomial(probs, num_samples=1).squeeze(1)
                samples[:, t] = next_states

                # update rolling index
                if self.order > 0:
                    idx = (idx % base) * self.num_states + next_states

            unigram_stats[i] = torch.bincount(samples.flatten(), minlength=self.num_states).float() / (num_samples * self.seq_len)

        return unigram_stats
    

class LatentIDBayes:
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


class LatentOODBayes:
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




import torch
import torch.nn as nn

class GroupUniformKnownBayes(nn.Module):
    """
    Bayesian predictor when the true transition matrix is one of M known candidates.

    Prior:
        total mass p_common on first 3 chains (indices 0,1,2), uniform within them;
        total mass 1-p_common on remaining K chains (indices 3..M-1), uniform within them.
    """
    def __init__(self, trans_mat: torch.Tensor, p_common: float = 0.9, device: str = "cpu"):
        """
        Args:
            trans_mat: (M, N, N) row-stochastic transition matrices
            p_common: prior mass on first 3 chains
        """
        super().__init__()
        assert trans_mat.dim() == 3
        M, N, N2 = trans_mat.shape
        assert N == N2
        assert M >= 4, "Need M=3+K with K>=1 so M>=4."

        self.M = M
        self.N = N
        self.device = device

        # log transition matrices
        self.register_buffer("log_trans_mat", torch.log(trans_mat.to(device).clamp_min(1e-30)))

        # prior pi over z
        K = M - 3
        pi = torch.empty(M, device=device, dtype=torch.float)
        pi[:3] = p_common / 3.0
        pi[3:] = (1.0 - p_common) / float(K)
        self.register_buffer("log_pi", torch.log(pi.clamp_min(1e-30)))

    @torch.no_grad()
    def predict(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Args:
            samples: (B, T) integer states in {0,...,N-1}

        Returns:
            preds: (B, T, N) posterior predictive distribution for each position.
                   preds[:, t] = P(s_{t+1} | s_{1:t}) for t=0..T-1 (with preds[:,T-1] predicting next beyond sequence).
                   If you prefer preds[:,0] to be uniform, replace the first-step logic accordingly.
        """
        samples = samples.to(self.device)
        B, T = samples.shape
        M, N, _ = self.log_trans_mat.shape

        preds = torch.zeros((B, T, N), device=self.device, dtype=torch.float)

        # indexing helpers
        m_idx = torch.arange(M, device=self.device).view(M, 1)   # (M,1)

        # cumulative log-likelihoods: log w_t(z) up to time t (observing transitions 1->...->t)
        # start with log prior (uniform initial cancels)
        cum_logw = self.log_pi.view(M, 1).expand(M, B).clone()    # (M,B)

        # t=0: predict s2 given s1 (uses posterior after seeing s1 only => just prior)
        s0 = samples[:, 0]                                        # (B,)
        # predictive: sum_z pi(z) P_z(. | s1)
        # compute rows for all z: (M,B,N)
        row0 = self.log_trans_mat[:, s0, :]                       # advanced indexing -> (M,B,N)
        log_num = torch.logsumexp(row0 + cum_logw.unsqueeze(-1), dim=0)    # (B,N)
        log_den = torch.logsumexp(cum_logw, dim=0).unsqueeze(-1)           # (B,1)
        preds[:, 0] = (log_num - log_den).exp()

        # pre-flatten transitions for fast gather of observed prob
        log_trans_flat = self.log_trans_mat.view(M, N * N)        # (M, N*N)

        for t in range(T - 1):
            s_t = samples[:, t]       # (B,)
            s_tp1 = samples[:, t + 1]
            flat_idx = s_t * N + s_tp1                       # (B,)

            # update cum_logw with observed transition likelihood
            # gather log P_z(s_{t+1} | s_t) for all z
            obs_logp = log_trans_flat.gather(dim=1, index=flat_idx.view(1, B).expand(M, B))  # (M,B)
            cum_logw = cum_logw + obs_logp

            # posterior predictive for next token given s_{t+1}
            row = self.log_trans_mat[:, s_tp1, :]                                  # (M,B,N)
            log_num = torch.logsumexp(row + cum_logw.unsqueeze(-1), dim=0)         # (B,N)
            log_den = torch.logsumexp(cum_logw, dim=0).unsqueeze(-1)               # (B,1)
            preds[:, t + 1] = (log_num - log_den).exp()

        return preds


import torch
import torch.nn as nn

class ThreeKnownPlusNewDirichletBayes(nn.Module):
    """
    Bayesian predictor over z in {0,1,2,new}.
      - z=0,1,2: known transition matrices P3[z]
      - z=3: a new (unknown) transition matrix; each row ~ Dirichlet(alpha=1,...,1)
    Prior:
      P(z in {0,1,2}) = p_common_total, uniform over the 3
      P(z=new) = 1 - p_common_total
    """
    def __init__(self, trans_mat_3: torch.Tensor, p_common_total: float = 0.9, alpha: float = 1.0, device: str = "cpu"):
        """
        Args:
            trans_mat_3: (3, N, N) known transitions
            p_common_total: prior mass on known group
            alpha: Dirichlet concentration (scalar; you asked alpha=1)
        """
        super().__init__()
        assert trans_mat_3.dim() == 3 and trans_mat_3.size(0) == 3
        _, N, N2 = trans_mat_3.shape
        assert N == N2

        self.N = N
        self.device = device
        self.alpha = float(alpha)

        self.register_buffer("log_trans_3", torch.log(trans_mat_3.to(device).clamp_min(1e-30)))  # (3,N,N)

        pi = torch.tensor([p_common_total/3.0, p_common_total/3.0, p_common_total/3.0, 1.0 - p_common_total],
                          device=device, dtype=torch.float)
        self.register_buffer("log_pi", torch.log(pi.clamp_min(1e-30)))

    @torch.no_grad()
    def predict(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Args:
            samples: (B, T) states in {0,...,N-1}

        Returns:
            preds: (B, T, N) posterior predictive distribution at each position.
        """
        samples = samples.to(self.device)
        B, T = samples.shape
        N = self.N

        preds = torch.zeros((B, T, N), device=self.device, dtype=torch.float)

        # z dimension is 4: 0,1,2,new
        cum_logw = self.log_pi.view(4, 1).expand(4, B).clone()  # (4,B)

        # counts for the new chain, per batch:
        # counts[b,i,j] = number of observed transitions i->j so far
        counts = torch.zeros((B, N, N), device=self.device, dtype=torch.float)
        row_sums = torch.zeros((B, N), device=self.device, dtype=torch.float)  # row_sums[b,i] = sum_j counts[b,i,j]
        b_idx = torch.arange(B, device=self.device)

        # helper for known chains flatten
        log_trans_3_flat = self.log_trans_3.view(3, N * N)  # (3, N*N)

        # t=0: predictive given s1 only => prior mixture
        s0 = samples[:, 0]  # (B,)
        # known part rows: (3,B,N)
        known_row0 = self.log_trans_3[:, s0, :]
        # new chain predictive row (Dirichlet prior, no data remember): uniform 1/N
        new_row0 = torch.full((B, N), fill_value=-torch.log(torch.tensor(float(N), device=self.device)), device=self.device)

        # stack rows for z=0,1,2,3 -> (4,B,N) in log
        row0 = torch.cat([known_row0, new_row0.unsqueeze(0)], dim=0)
        log_num = torch.logsumexp(row0 + cum_logw.unsqueeze(-1), dim=0)     # (B,N)
        log_den = torch.logsumexp(cum_logw, dim=0).unsqueeze(-1)            # (B,1)
        preds[:, 0] = (log_num - log_den).exp()

        for t in range(T - 1):
            s_t = samples[:, t]
            s_tp1 = samples[:, t + 1]
            flat_idx = s_t * N + s_tp1  # (B,)

            # ---- update known chains likelihood
            obs_logp_known = log_trans_3_flat.gather(
                dim=1, index=flat_idx.view(1, B).expand(3, B)
            )  # (3,B)
            cum_logw[0:3] = cum_logw[0:3] + obs_logp_known

            # ---- update new chain (Dirichlet predictive for observed transition)
            # pred_new = (alpha + counts[b, s_t, s_tp1]) / (N*alpha + row_sums[b, s_t])
            num = self.alpha + counts[b_idx, s_t, s_tp1]                       # (B,)
            den = N * self.alpha + row_sums[b_idx, s_t]                         # (B,)
            pred_new = (num / den).clamp_min(1e-30)
            cum_logw[3] = cum_logw[3] + torch.log(pred_new)

            # ---- update counts after consuming transition
            counts[b_idx, s_t, s_tp1] += 1.0
            row_sums[b_idx, s_t] += 1.0

            # ---- posterior predictive for next token given current state s_{t+1}
            s_curr = s_tp1

            # known rows: (3,B,N)
            known_row = self.log_trans_3[:, s_curr, :]  # (3,B,N)

            # new chain row predictive: (alpha + counts[b, s_curr, :]) / (N*alpha + row_sums[b, s_curr])
            new_num = self.alpha + counts[b_idx, s_curr, :]                     # (B,N)
            new_den = (N * self.alpha + row_sums[b_idx, s_curr]).unsqueeze(-1)   # (B,1)
            new_row = torch.log((new_num / new_den).clamp_min(1e-30))            # (B,N)

            row = torch.cat([known_row, new_row.unsqueeze(0)], dim=0)            # (4,B,N)

            log_num = torch.logsumexp(row + cum_logw.unsqueeze(-1), dim=0)       # (B,N)
            log_den = torch.logsumexp(cum_logw, dim=0).unsqueeze(-1)             # (B,1)
            preds[:, t + 1] = (log_num - log_den).exp()

        return preds

import torch
import torch.nn as nn

class ThreeKnownUniformBayes(nn.Module):
    """
    Bayesian predictor assuming the true transition matrix is one of 3 known candidates,
    with a uniform prior over the 3 tasks/chains.

    Given a sequence s_1:T, maintains posterior over z in {0,1,2}:
        p(z | s_1:t) ∝ p(z) * Π_{τ=1}^{t-1} P_z(s_{τ+1} | s_τ)
    and outputs posterior predictive:
        p(s_{t+1} | s_1:t) = Σ_z p(z | s_1:t) P_z(s_{t+1} | s_t).
    """
    def __init__(self, trans_mat_3: torch.Tensor, device: str = "cpu", eps: float = 1e-30):
        """
        Args:
            trans_mat_3: (3, N, N) tensor, each row sums to 1
            device: cpu/cuda
            eps: clamp to avoid log(0)
        """
        super().__init__()
        assert trans_mat_3.dim() == 3 and trans_mat_3.size(0) == 3, "Need (3, N, N)."
        _, N, N2 = trans_mat_3.shape
        assert N == N2

        self.N = N
        self.device = device
        self.eps = eps

        # store log transitions
        logT = torch.log(trans_mat_3.to(device).clamp_min(eps))
        self.register_buffer("log_trans", logT)  # (3, N, N)

        # uniform prior over 3
        log_pi = torch.log(torch.full((3,), 1.0 / 3.0, device=device))
        self.register_buffer("log_pi", log_pi)   # (3,)

    @torch.no_grad()
    def predict(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Args:
            samples: (B, T) integer tokens in {0,...,N-1}

        Returns:
            preds: (B, T, N) where preds[:, t] = p(s_{t+1} | s_{1:t})
        """
        samples = samples.to(self.device)
        B, T = samples.shape
        N = self.N

        preds = torch.zeros((B, T, N), device=self.device, dtype=torch.float)

        # cum_logw[z,b] = log p(z) + sum_{τ=0}^{t-1} log P_z(s_{τ+1}|s_τ)
        # initialize at t=0 (seeing only s_1): posterior is just prior
        cum_logw = self.log_pi.view(3, 1).expand(3, B).clone()  # (3,B)

        # t=0: predictive of s_2 given s_1 under prior mixture
        s0 = samples[:, 0]                         # (B,)
        row0 = self.log_trans[:, s0, :]            # (3,B,N)

        log_num = torch.logsumexp(row0 + cum_logw.unsqueeze(-1), dim=0)  # (B,N)
        log_den = torch.logsumexp(cum_logw, dim=0).unsqueeze(-1)         # (B,1)
        preds[:, 0] = (log_num - log_den).exp()

        # If you instead want preds[:,0] = uniform over tokens:
        # preds[:, 0] = 1.0 / N

        # Flatten transitions so we can gather observed transition probs fast.
        log_trans_flat = self.log_trans.view(3, N * N)  # (3, N*N)

        for t in range(T - 1):
            s_t = samples[:, t]         # (B,)
            s_tp1 = samples[:, t + 1]   # (B,)
            flat_idx = s_t * N + s_tp1  # (B,)

            # update posterior weights with likelihood of observed transition
            obs_logp = log_trans_flat.gather(
                dim=1, index=flat_idx.view(1, B).expand(3, B)
            )  # (3,B)
            cum_logw = cum_logw + obs_logp

            # posterior predictive for next token given current state s_{t+1}
            curr = s_tp1
            row = self.log_trans[:, curr, :]  # (3,B,N)

            log_num = torch.logsumexp(row + cum_logw.unsqueeze(-1), dim=0)  # (B,N)
            log_den = torch.logsumexp(cum_logw, dim=0).unsqueeze(-1)        # (B,1)
            preds[:, t + 1] = (log_num - log_den).exp()

        return preds



import torch
from typing import Optional, Tuple

@torch.no_grad()
def task_posterior_over_time(
    sampler,
    samples: torch.Tensor,
    *,
    return_log: bool = False,
    eps: float = 1e-30,
) -> torch.Tensor:
    """
    Compute filtering posterior P(Z=k | X_{0:t}) for each t, for sequences generated by LatentMarkov.

    Args:
        sampler: an instance of LatentMarkov (your class).
        samples: Long tensor of shape (B, L_obs) or (E, B, L_obs).
                 If sampler.pad == True, L_obs should be 2*seq_len-1 and
                 we will use samples[..., ::2] as the real tokens.
        return_log: if True, return log posterior instead of posterior.
        eps: numerical floor for probabilities.

    Returns:
        post: Tensor of shape (..., L_real, T) where
              T = sampler.total_trans, L_real = sampler.seq_len,
              and ... is (B,) or (E,B,) matching input prefix dims.
              post[..., t, k] = P(Z=k | X_{0:t}) (or log if return_log).
    """
    device = samples.device
    dtype = torch.float32

    # Allow (E,B,L) or (B,L)
    orig_shape = samples.shape
    if samples.dim() == 2:
        prefix_shape = (samples.shape[0],)  # (B,)
        samples_ = samples.unsqueeze(0)     # (1,B,L)
    elif samples.dim() == 3:
        prefix_shape = samples.shape[:2]    # (E,B)
        samples_ = samples
    else:
        raise ValueError(f"samples must have shape (B,L) or (E,B,L), got {samples.shape}")

    E, B, L_obs = samples_.shape

    # If padded, keep only real token positions (0,2,4,...)
    if sampler.pad:
        x = samples_[..., ::2]  # (E,B,L_real)
    else:
        x = samples_
    E, B, L = x.shape
    assert L == sampler.seq_len, f"Expected real length {sampler.seq_len}, got {L}"

    T = sampler.total_trans
    order = sampler.order
    S = sampler.num_states
    powers = sampler.powers.to(device=device)

    # Stack all transition matrices into (T, S^order, S)
    if sampler.n_minor_tasks > 0:
        trans_all = torch.cat([sampler.major_trans_mat, sampler.minor_trans_mat], dim=0).to(device=device)
    else:
        trans_all = sampler.major_trans_mat.to(device=device)
    assert trans_all.shape[0] == T

    # Build log prior over tasks (T,)
    if sampler.n_minor_tasks == 0:
        prior = torch.full((T,), 1.0 / max(1, sampler.n_major_tasks), device=device, dtype=dtype)
    else:
        prior_major = (1.0 - float(sampler.p_minor)) / max(1, sampler.n_major_tasks)
        prior_minor = float(sampler.p_minor) / max(1, sampler.n_minor_tasks)
        prior = torch.cat([
            torch.full((sampler.n_major_tasks,), prior_major, device=device, dtype=dtype),
            torch.full((sampler.n_minor_tasks,), prior_minor, device=device, dtype=dtype),
        ], dim=0)
    prior = torch.clamp(prior, min=eps)
    log_prior = prior.log()  # (T,)

    # We'll store log posterior for numerical stability
    # Shape: (E,B,L,T)
    log_post = torch.empty((E, B, L, T), device=device, dtype=dtype)

    # Initialize: for t < order, posterior is just the prior (initial tokens are task-independent)
    log_post[:, :, :min(order, L), :] = log_prior.view(1, 1, 1, T).expand(E, B, min(order, L), T)

    # Running unnormalized log-belief: log p(z) + sum_{s=order..t} log P_z( x_s | history )
    running = log_prior.view(1, 1, T).expand(E, B, T).clone()

    if L > order:
        # Maintain rolling history window of length 'order' from observed tokens
        # state_hist at step t represents x_{t-order:t} (length order)
        state_hist = x[:, :, :order].clone()  # (E,B,order)

        base = (S ** (order - 1)) if order > 0 else 1

        for t in range(order, L):
            # Compute index of the current history
            # idx = sum_{j=0..order-1} state_hist[..., j] * powers[j]
            idx = torch.sum(state_hist * powers.view(1, 1, -1), dim=-1).long()  # (E,B)

            xt = x[:, :, t].long()  # (E,B)

            # Get prob under each task for each (E,B) pair: P_k[idx, xt]
            # trans_all: (T, Lidx, S)
            # trans_all[:, idx] -> (T, E, B, S)
            probs_row = trans_all[:, idx]  # advanced indexing, shape (T, E, B, S)
            # gather xt -> (T, E, B)
            probs = probs_row.gather(dim=-1, index=xt.unsqueeze(0).unsqueeze(-1).expand(T, E, B, 1)).squeeze(-1)

            probs = torch.clamp(probs, min=eps)
            running = running + probs.permute(1, 2, 0).log()  # (E,B,T)

            # Normalize to get log posterior at time t
            log_norm = torch.logsumexp(running, dim=-1, keepdim=True)  # (E,B,1)
            log_post[:, :, t, :] = running - log_norm

            # Update history window
            if order > 0:
                # shift left and append xt
                state_hist = torch.roll(state_hist, shifts=-1, dims=-1)
                state_hist[:, :, -1] = xt

    if return_log:
        out = log_post
    else:
        out = torch.exp(log_post)

    # Return shape matching input prefix dims: (B,L,T) or (E,B,L,T)
    if len(prefix_shape) == 1:  # original was (B,L)
        return out.squeeze(0)    # (B,L,T)
    return out                  # (E,B,L,T)
