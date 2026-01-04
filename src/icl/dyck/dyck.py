import torch

class TrieNode:
    def __init__(self) -> None:
        self.children = {} # key: -1 or 1
        self.count = 0 # number of sequences that pass through this node
        self.count_pos = 0 # number of sequences where next token is 1
    
class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, sequence):
        node = self.root
        for i in range(len(sequence) - 1):
            token = sequence[i]
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
            node.count += 1
            if sequence[i+1] == 1:
                node.count_pos += 1

class DyckPathTask:
    def __init__(self, config):
        self.pad = config.task.pad if hasattr(config.task, 'pad') else False
        if self.pad:
            self.num_states = config.vocab_size - 3
        else: 
            self.num_states = config.vocab_size - 2  
        self.seq_len = config.seq_len
        if self.pad: 
            assert self.seq_len % 2 == 1, "Sequence length must be odd when padding is enabled."
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
            # For order=0, sample i.i.d. distribution from Dirichlet prior
            self.powers = None
            dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device) * self.alpha)
            self.trans_matrix = dirichlet_dist.sample()  # Sample a single probability vector
            self.trans_matrix /= self.trans_matrix.sum()  # Ensure normalization
    
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
        padded = getattr(self, 'pad', False)
        seq_len = self.seq_len if not padded else (self.seq_len + 1) // 2
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
        
        padded = hasattr(self, 'pad')
        if padded:
            padded = self.pad

        seq_len = (self.seq_len + 1) // 2 if padded else self.seq_len

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
        
        padded = hasattr(self, 'pad')
        if padded: padded = self.pad

        if padded:
            padded_samples = torch.full((num_samples, self.seq_len), fill_value=self.num_states+2, dtype=torch.long, device=self.device)
            padded_samples[:, ::2] = samples
            padded_masks = torch.full((num_samples, self.seq_len), fill_value=0, dtype=torch.long, device=self.device)
            padded_masks[:, ::2] = masks
            samples, masks = padded_samples, padded_masks

        if mode == "train":
            if padded: return samples.reshape(epochs, -1, self.seq_len), masks.reshape(epochs, -1, self.seq_len)
            else: return samples.reshape(epochs, -1, seq_len), masks.reshape(epochs, -1, seq_len)

        if mode in ["testing", "major", "minor"] and task is None:
            return samples, masks, latent
        
        return samples, masks















class DyckBayes:
    def __init__(self, config, sampler, flag=False):
        self.pad = config.task.pad if "pad" in config.task else False
        self.num_states = config.vocab_size-1 if self.pad else config.vocab_size
        self.dyck_length = config.task.dyck_length
        self.trans_matrix = sampler.trans_matrix
        self.repeat_prob = config.task.repeat_prob
        self.order = config.task.order
        self.trie = None
        self.one = sampler.one
        self.neg = sampler.neg
        self.flag = flag
        
        # Build Trie from all task pools (major + minor)
        if sampler.total_trans > 0:
            self.trie = Trie()
            # Insert major task pool
            if sampler.n_major_tasks > 0:
                for seq in sampler.major_task_pool:
                    self.trie.insert([1 if s==sampler.one else -1 for s in seq.tolist()])
            # Insert minor task pool
            # if sampler.n_minor_tasks > 0:
            #    for seq in sampler.minor_task_pool:
            #        self.trie.insert([1 if s==sampler.one else -1 for s in seq.tolist()])

    def dyck_pos(self, dyckseq):
        """
        seq: Tensor of shape [2*dyck_length], where each element is -1 or 1.
        return: probs[i] = Pr(seq[i] = 1 | seq[:i])
        """
        eps = 1e-6  # small value to avoid division by zero
        probs = torch.zeros(dyckseq.shape[0], device=dyckseq.device, dtype=torch.float32) + eps # probability of being one at each position
        seq_list = [1 if s==self.one else -1 for s in dyckseq.tolist()]
        
        if (self.trie is None) or self.flag:
            dR, dU = 0, 0
            for i, s in enumerate(seq_list):
                probs[i] = (dR - dU + 2) / (dR - dU + 1) * (self.dyck_length - dR) / (2*self.dyck_length - dR - dU)
                if s == -1:
                    dU += 1
                else:
                    dR += 1
        
        else:
            node = self.trie.root
            probs[0] = 1
            for i, s in enumerate(seq_list[:-1]):
                if s not in node.children:
                    break
                node = node.children[s]
                probs[i+1] = node.count_pos / node.count 
        
        return probs

    def extend_dyck_prob(self, seq):
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)  # [1, T]
        B, T = seq.shape

        mask = (seq == self.one) | (seq == self.neg)

        prob = torch.zeros((B,T), dtype=torch.float32, device=seq.device)

        for b in range(B):
            mask_b = mask[b]
            seq_b = seq[b]
            dyck_probs = self.dyck_pos(seq_b[mask_b])
            
            # Step 1: compute cumulative sum of mask (in int form)
            cumsum = mask_b.int().cumsum(dim=0)
            tot = dyck_probs.shape[0]

            prob_b = torch.zeros(T, dtype=dyck_probs.dtype, device=seq.device)

            # Step 3: assign dyckprob[j] where j = cumsum[i] if j < 2L
            valid = (cumsum < tot) & (cumsum >= 0)
            indices = cumsum[valid]
            prob_b[valid] = dyck_probs[indices]
            prob_b[cumsum == tot] = -1
            prob[b] = prob_b
            
        return prob
    
    def fast_markov_probs(self, seq):
        """
        seq: [B, T] integer tokens
        return: [B, T, num_states - 2] markov part
        """
        B, T = seq.shape
        K = self.num_states - 2

        # Preallocate output
        markov_out = torch.zeros((B, T, K), device=seq.device, dtype=torch.float32)

        if self.order == 0:
            # For order=0, use uniform i.i.d. distribution
            markov_out[:, :, :] = self.trans_matrix[:K].unsqueeze(0).unsqueeze(0)  # Broadcast to [B, T, K]
            return markov_out

        # Initial uniform probs
        prev = torch.ones((B, K), device=seq.device) / K
        chosen_rows = torch.zeros((B, K), device=seq.device, dtype=torch.float32)

        for t in range(T):
            s_t = seq[:, t]  # [B]

            # Mask where s in {self.one, self.neg}
            update_mask = (s_t == self.one) | (s_t == self.neg) # [B]

            # For mask == True: multiply prev @ trans_matrix
            updated = torch.matmul(prev, self.trans_matrix)  # [B, K]

            # For mask == False: use trans_matrix[s_t]
            chosen_rows[~update_mask] = self.trans_matrix[s_t[~update_mask]]  # [B, K]

            # Combine based on mask
            prev = torch.where(update_mask.unsqueeze(1), updated, chosen_rows)

            # Save
            markov_out[:, t] = prev

        return markov_out

    def pos_prob(self, seq):
        # probs[i] : Pr(seq[i+1] | seq[:i+1])
        if seq.dim() == 1: 
            seq = seq.unsqueeze(0)
        B, T = seq.shape
        K = self.num_states - 2

        probs = torch.zeros((B, T, self.num_states), device=seq.device, dtype=torch.float32)
        dyck_probs = self.extend_dyck_prob(seq) # [B, T]
        dyck_mask = dyck_probs >= 0
        batch_idx, time_idx = torch.where(dyck_mask)
        dyck_vals = dyck_probs[batch_idx, time_idx]
        probs[batch_idx, time_idx, self.one] = self.repeat_prob * dyck_vals
        probs[batch_idx, time_idx, self.neg] = self.repeat_prob * (1 - dyck_vals)
        
        markov_part = self.fast_markov_probs(seq)  # shape [B, T, K]
        probs[batch_idx, time_idx, :K] = markov_part[batch_idx, time_idx, :K] * (1 - self.repeat_prob)
        batch_idx, time_idx = torch.where(~dyck_mask)
        probs[batch_idx, time_idx, :K] = markov_part[batch_idx, time_idx, :K]

        if self.pad:
            eps = 1e-8  # or any small constant you need

            # Create a column filled with eps
            eps_column = torch.full((B, T, 1), fill_value=eps, device=probs.device, dtype=probs.dtype)

            # Concatenate along the last dimension
            probs = torch.cat([probs, eps_column], dim=-1)  # shape (B, T, D+1)


        return probs[:,:-1,:]
    
    def predict(self, seq):
        probs = self.pos_prob(seq)
        preds = torch.argmax(probs, dim=-1)
        return preds