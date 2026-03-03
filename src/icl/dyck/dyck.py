import torch
from icl.dyck.dyck_utils import sample_binary_mask


class DyckPathTask:
    def __init__(self, config):
        self.num_states = config.vocab_size - 2  
        self.seq_len = config.seq_len
        self.dyck_length = config.task.dyck_length
        
        # Support for major and minor task pools
        self.n_major_tasks = config.task.n_tasks if hasattr(config.task, 'n_tasks') else 0
        self.n_minor_tasks = config.task.n_minor_tasks if hasattr(config.task, 'n_minor_tasks') else 0
        self.p_minor = config.task.p_minor if hasattr(config.task, 'p_minor') else 0.0
        
        self.order = config.task.order
        self.num_states_order = self.num_states ** self.order
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        self.eval_size = config.eval_size
        self.device = config.device
        self.alpha = config.task.alpha
        self.repeat_prob = config.task.repeat_prob
        self.one = self.num_states + 1
        self.neg = self.num_states
        
        # Generate major and minor task pools
        if self.n_major_tasks > 0:
            self.major_task_pool = self._random_dyck_path(self.n_major_tasks)
        if self.n_minor_tasks > 0:
            self.minor_task_pool = self._random_dyck_path(self.n_minor_tasks)
        else:
            self.minor_task_pool = None
        
        if self.order > 0:
            self.powers = (self.num_states ** torch.arange(self.order - 1, -1, -1, device=self.device)).long()

            dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device) * self.alpha)
            self.trans_matrix = dirichlet_dist.sample((self.num_states_order,))
            self.trans_matrix /= self.trans_matrix.sum(dim=1, keepdim=True)
        else:
            # For order=0, sample i.i.d. uniform distribution
            self.powers = None
            self.trans_matrix = torch.full((self.num_states,), 1.0 / self.num_states, device=self.device)
    
    def to(self, device):
        self.device = device
        self.trans_matrix = self.trans_matrix.to(device)
        if self.n_major_tasks > 0:  
            self.major_task_pool = self.major_task_pool.to(device)
        if self.n_minor_tasks > 0:
            self.minor_task_pool = self.minor_task_pool.to(device)
        if self.powers is not None:
            self.powers = self.powers.to(device)
        
        return self

    @staticmethod
    def _dyck_path_probability(r,k):
        """
        Vectorized computation of Arnold & Sleep's probability of placing -1.
        r: Tensor of shape [batch_size], number of unmatched +1 steps
        k: Tensor of shape [batch_size], number of steps remaining
        Returns: Tensor of probabilities, shape [batch_size]
        """
        prob = torch.zeros_like(r, dtype=torch.float32)
        mask = r > 0  # valid positions where -1 is possible
        r_, k_ = r[mask], k[mask]
        prob[mask] = (r_ * (k_ + r_ + 2)) / (2 * k_ * (r_ + 1))
        return prob
    
    def _random_dyck_path(self, num_samples) -> torch.Tensor:
        """
        Generate a batch of Dyck paths of length 2n using PyTorch.
        Returns a tensor of shape [batch_size, 2n] with values +1 or -1.
        """
        L = 2 * self.dyck_length
        path = torch.empty(num_samples, L, device=self.device, dtype=torch.int8)
        r = torch.zeros(num_samples, device=self.device, dtype=torch.int32)  # unmatched +1
        k = torch.full((num_samples,), L, device=self.device, dtype=torch.int32)  # remaining steps

        for t in range(L):
            prob_down = self._dyck_path_probability(r, k)
            rand = torch.rand(num_samples, device=self.device)
            step = torch.where(rand < prob_down, torch.full_like(rand, -1, dtype=torch.int8),
                                                torch.full_like(rand,  1, dtype=torch.int8))
            path[:, t] = step
            r += step
            k -= 1
        
        path[path == 1] = self.one
        path[path == -1] = self.neg
        
        return path

    @property
    def total_trans(self) -> int:
        """Total number of tasks (major + minor)"""
        return int(self.n_major_tasks + self.n_minor_tasks)
    
    def get_task_dyck_path(self, global_task_id: int) -> torch.Tensor:
        """
        Map a global task id [0, total_trans) to a dyck path.
        """
        assert 0 <= global_task_id < self.total_trans, f"Task id {global_task_id} out of range [0, {self.total_trans})"
        if global_task_id < self.n_major_tasks:
            return self.major_task_pool[global_task_id]
        else:
            return self.minor_task_pool[global_task_id - self.n_major_tasks]
    
    def _plant_dyck(self, dyck_str: torch.Tensor, dyck_mask=None) -> torch.Tensor:
        # dyck_str: [B, L]
        batch_size, dyck_len = dyck_str.shape
        seq_len = self.seq_len
        dyck_str = dyck_str.to(self.device)

        assert dyck_len <= seq_len, "Dyck path too long for the target sequence."

        # ---- Build mask as bool ----
        if dyck_mask is not None:
            assert dyck_mask.ndim == 1 and dyck_mask.numel() == seq_len, "dyck_mask length mismatch"
            # (1, seq_len) -> (B, seq_len) broadcasted view is fine for reading
            mask = dyck_mask.to(self.device)
            mask = mask.to(torch.bool).unsqueeze(0).expand(batch_size, seq_len)
        else:
            mask = (torch.rand((batch_size, seq_len), device=self.device) < self.repeat_prob)

        # ---- Cap number of ones per row safely ----
        # compute cumulative count in int64 to avoid uint8 overflow
        cumsum_i64 = mask.to(torch.int64).cumsum(dim=1)
        # cap by BOTH available dyck tokens and desired limit
        dyck_limit = getattr(self, 'dyck_length', dyck_len) * 2
        max_keep = min(dyck_len, dyck_limit)
        # keep only positions where count <= max_keep
        mask = mask & (cumsum_i64 <= max_keep)

        # ---- Build running index (0..k-1 at 1-positions; -1 elsewhere) ----
        running_index = mask.to(torch.int64).cumsum(dim=1) - 1
        valid_pos = mask  # True where we place a token

        # ---- Scatter from dyck_str into planted using advanced indexing ----
        planted = torch.zeros((batch_size, seq_len), dtype=dyck_str.dtype, device=self.device)
        if valid_pos.any():
            batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand_as(running_index)

            planted[valid_pos] = dyck_str[batch_indices[valid_pos], running_index[valid_pos]]

        return planted

    def generate(self, epochs=1, mode="train", num_samples=None, task=None, dyck_mask=None):
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

        seq_len = self.seq_len

        samples = torch.zeros((num_samples, seq_len), dtype=torch.long, device=self.device)

        if self.order > 0:
            state = torch.randint(high=self.num_states, size=(num_samples, self.order), device=self.device)
        else:
            state = None  # No state needed for order=0

        # Handle major/minor task selection
        if mode == "major":
            if task is None:
                latent_major = torch.randint(high=self.n_major_tasks, size=(num_samples,), device=self.device)
            else:
                assert 0 <= task < self.n_major_tasks, "task id out of range"
                latent_major = torch.full((num_samples,), task, dtype=torch.long, device=self.device)
            hidden_values = self.major_task_pool[latent_major]
            latent = latent_major
        
        elif mode == "minor":
            if self.n_minor_tasks == 0:
                raise ValueError("No minor tasks available.")
            if task is None:
                latent_minor = torch.randint(high=self.n_minor_tasks, size=(num_samples,), device=self.device)
            else:
                assert 0 <= task < self.n_minor_tasks, "task id out of range"
                latent_minor = torch.full((num_samples,), task, dtype=torch.long, device=self.device)
            hidden_values = self.minor_task_pool[latent_minor]
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
            
            hidden_major = self.major_task_pool[latent_major] if latent_major is not None else None
            hidden_minor = self.minor_task_pool[latent_minor] if latent_minor is not None else None
            
            # Mix major and minor tasks based on p_minor
            if hidden_major is not None and hidden_minor is not None:
                use_minor = (torch.rand(num_samples, device=self.device) < self.p_minor)
                hidden_values = torch.where(use_minor.unsqueeze(1), hidden_minor, hidden_major)
                latent = torch.where(
                    use_minor,
                    self.n_major_tasks + latent_minor,
                    latent_major
                )
            elif hidden_major is not None:
                hidden_values = hidden_major
                latent = latent_major
            elif hidden_minor is not None:
                hidden_values = hidden_minor
                latent = self.n_major_tasks + latent_minor
            else:
                raise ValueError("No task pools available.")
        
        elif mode == "ood" or self.n_major_tasks + self.n_minor_tasks == 0:
            hidden_values = self._random_dyck_path(num_samples)

        planted_dyck = self._plant_dyck(hidden_values, dyck_mask)
        
        if self.order > 0:
            # Markov chain generation for order > 0
            samples[:, :self.order] = state

            for t in range(self.order, seq_len):
                state_indices = torch.sum(state * self.powers, dim=1)
                probs = self.trans_matrix[state_indices]
                next_states = torch.multinomial(probs, num_samples=1).squeeze(1)
                samples[:, t] = next_states

                state[:, :-1] = state[:, 1:]
                state[:, -1] = next_states
        else:
            # For order=0, sample i.i.d. from uniform distribution
            samples = torch.multinomial(
                self.trans_matrix.unsqueeze(0).expand(num_samples, -1),
                num_samples=seq_len,
                replacement=True
            )
        
        masks = planted_dyck != 0
        samples[planted_dyck != 0] = planted_dyck[planted_dyck != 0].long()

        if mode == "train":
            out_len = samples.shape[-1]
            return samples.reshape(epochs, -1, out_len), masks.reshape(epochs, -1, out_len)

        if mode in ["testing", "major", "minor"] and task is None:
            return samples, masks, latent
        
        return samples, masks


# ── backward-compat re-exports ──────────────────────────────────────
from icl.dyck.dyck_posterior import dyck_task_posterior_over_time  # noqa: F401,E402
from icl.dyck.dyck_bayes import TrieNode, Trie, DyckBayes  # noqa: F401,E402
from icl.dyck.dyck_kl import (  # noqa: F401,E402
    plot_kl_model_vs_two_bayes_dyck,
    plot_kl_model_vs_two_bayes_dyck_across_k,
)
from icl.dyck.dyck_mi import (  # noqa: F401,E402
    WeightedDyckTrie,
    _binary_entropy_bits,
    estimate_mi_prefix_vs_height_train,
)
