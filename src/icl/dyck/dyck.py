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
            # For order=0, sample i.i.d. uniform distribution
            self.powers = None
            # dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device) * self.alpha)
            # self.trans_matrix = dirichlet_dist.sample()  # Sample a single probability vector
            # self.trans_matrix /= self.trans_matrix.sum()  # Ensure normalization
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






import torch

@torch.no_grad()
def dyck_task_posterior_over_time_nonpadded(
    task,                       # DyckPathTask instance
    samples: torch.Tensor,      # (B,L_obs) or (E,B,L_obs)
    masks: torch.Tensor,        # same shape as samples; 1 where Dyck token was planted
    *,
    return_log: bool = False,
    eps: float = 1e-30,
) -> torch.Tensor:
    """
    Compute filtering posterior P(Z=k | s_{0:t}) over Dyck-path identity Z for each t,
    but ONLY at the non-padded (real-token) steps, following the style of task_posterior_over_time.

    If task.pad == True, we use samples[..., ::2] and masks[..., ::2] as the real tokens/masks.

    Args:
        task: DyckPathTask.
        samples: Long tensor (B,L_obs) or (E,B,L_obs).
                 If task.pad==True, L_obs should be task.seq_len (as returned by your DyckPathTask.generate),
                 and we will use samples[..., ::2] as the real tokens.
        masks:   Long/bool tensor same shape as samples.
        return_log: return log posterior if True.
        eps: numerical floor for probabilities (used when normalizing / degenerate rows).

    Returns:
        post: Tensor shape (B,L_real,T) or (E,B,L_real,T),
              where L_real = (task.seq_len+1)//2 if padded else task.seq_len,
              and T = task.total_trans.
              post[..., t, k] = P(Z=k | s_{0:t}) (or log if return_log) on the real-token timeline.
    """
    device = samples.device
    dtype = torch.float32

    # Allow (E,B,L) or (B,L)
    if samples.dim() == 2:
        samples_ = samples.unsqueeze(0)  # (1,B,L_obs)
        masks_ = masks.unsqueeze(0)
        squeeze_E = True
    elif samples.dim() == 3:
        samples_ = samples
        masks_ = masks
        squeeze_E = False
    else:
        raise ValueError(f"samples must have shape (B,L) or (E,B,L), got {samples.shape}")

    if masks_.shape != samples_.shape:
        raise ValueError(f"masks must match samples shape; got masks={masks_.shape}, samples={samples_.shape}")

    # Keep only real token positions when padded (match the user's reference implementation)
    if getattr(task, "pad", False):
        x = samples_[..., ::2]
        m = masks_[..., ::2]
    else:
        x = samples_
        m = masks_

    E, B, L = x.shape
    T = task.total_trans
    assert T > 0, "No tasks available: total_trans == 0"

    # Build task pool of Dyck strings: (T, L_dyck)
    if task.n_minor_tasks > 0:
        dyck_all = torch.cat([task.major_task_pool, task.minor_task_pool], dim=0).to(device=device)
    else:
        dyck_all = task.major_task_pool.to(device=device)
    assert dyck_all.shape[0] == T

    # Prior over tasks (match generate() mixture)
    if task.n_minor_tasks == 0:
        prior = torch.full((T,), 1.0 / max(1, task.n_major_tasks), device=device, dtype=dtype)
    else:
        prior_major = (1.0 - float(task.p_minor)) / max(1, task.n_major_tasks)
        prior_minor = float(task.p_minor) / max(1, task.n_minor_tasks)
        prior = torch.cat([
            torch.full((task.n_major_tasks,), prior_major, device=device, dtype=dtype),
            torch.full((task.n_minor_tasks,), prior_minor, device=device, dtype=dtype),
        ], dim=0)
    prior = torch.clamp(prior, min=eps)
    log_prior = prior.log()  # (T,)

    # Output: (E,B,L,T)
    log_post = torch.empty((E, B, L, T), device=device, dtype=dtype)

    # Running unnormalized log-belief
    running = log_prior.view(1, 1, T).expand(E, B, T).clone()

    # planted index j = cumsum(m==1)-1, computed on the REAL timeline
    m_bool = m.to(torch.bool)
    planted_idx = m_bool.to(torch.long).cumsum(dim=-1) - 1  # (E,B,L)

    neg_inf = torch.tensor(-float("inf"), device=device, dtype=dtype)

    for t in range(L):
        is_planted = m_bool[:, :, t]  # (E,B)
        if is_planted.any():
            eb = torch.nonzero(is_planted, as_tuple=False)  # (N,2)
            e_idx = eb[:, 0]
            b_idx = eb[:, 1]

            j = planted_idx[e_idx, b_idx, t].long()   # (N,)
            obs = x[e_idx, b_idx, t].long()           # (N,)

            # expected tokens for each task: (N,T) where expected[n,k]=dyck_all[k, j[n]]
            expected = dyck_all.transpose(0, 1).index_select(0, j)  # (N,T)

            mismatch = expected != obs.unsqueeze(1)  # (N,T)
            running[e_idx, b_idx, :] = torch.where(mismatch, neg_inf, running[e_idx, b_idx, :])

        # Normalize to posterior at time t
        maxv = torch.max(running, dim=-1, keepdim=True).values  # (E,B,1)
        all_neg_inf = torch.isneginf(maxv)                      # (E,B,1)

        stabilized = running - maxv
        probs = torch.exp(stabilized)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)

        if all_neg_inf.any():
            probs = torch.where(all_neg_inf.expand_as(probs), torch.full_like(probs, 1.0 / T), probs)

        log_post[:, :, t, :] = torch.log(probs.clamp_min(eps))

        # carry forward log posterior
        running = log_post[:, :, t, :].clone()

    out = log_post if return_log else torch.exp(log_post)
    return out.squeeze(0) if squeeze_E else out















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


# ----------------------------------------------------------------
# Mutual-information estimation: prefix vs. (height, remaining)
# ----------------------------------------------------------------
import math
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class _WTrieNode:
    children: dict
    w_total: float
    w_plus: float  # weight mass of paths whose next step is +1

    def __init__(self):
        self.children = {}      # key: +1/-1, value: _WTrieNode
        self.w_total = 0.0
        self.w_plus = 0.0


class WeightedDyckTrie:
    """
    Weighted prefix trie over a finite pool of Dyck paths.
    Supports Bayes-optimal next-step prob under a specified prior over tasks.
    """
    def __init__(self):
        self.root = _WTrieNode()

    def insert(self, steps, weight: float):
        node = self.root
        node.w_total += float(weight)
        for s in steps:
            if s not in node.children:
                node.children[s] = _WTrieNode()
            if s == +1:
                node.w_plus += float(weight)
            node = node.children[s]
            node.w_total += float(weight)

    def p_next_plus(self, prefix_steps):
        node = self.root
        for s in prefix_steps:
            if s not in node.children:
                return None
            node = node.children[s]
        if node.w_total <= 0.0:
            return None
        return node.w_plus / node.w_total


def _binary_entropy_bits(p: float, eps: float = 1e-12) -> float:
    p = float(min(max(p, eps), 1.0 - eps))
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


@torch.no_grad()
def estimate_mi_prefix_vs_height_train(
    task=None,
    *,
    exp_name: str = None,
    num_samples: int = 8192,
    min_dyck_position: int = 0,
    max_dyck_position: int = None,
    uniform_prior: bool = True,
    eps: float = 1e-12,
    seed=None,
):
    """
    Approximate I(prefix; next_step | height, remaining) in bits,
    under the *training-mode* distribution (major/minor mixed by p_minor).

    The mutual information measures how much more the specific Dyck prefix
    tells us about the next step compared to knowing only (height, remaining).

    Parameters
    ----------
    task : DyckPathTask, optional
        A DyckPathTask instance. If None, loaded from exp_name.
    exp_name : str, optional
        Experiment name (folder under results/dyck/). Used to load the sampler
        when task is not provided.
    num_samples : int, default=8192
        Number of sequences to generate for the estimate.
    min_dyck_position : int, default=0
        Only include Dyck positions with index >= this value (0-indexed among
        the planted Dyck tokens). Useful for skipping early uninformative positions.
    max_dyck_position : int, optional
        Only include Dyck positions with index < this value. If None, uses all
        positions from min_dyck_position onward.
    uniform_prior : bool, default=True
        If True, temporarily sets p_minor so that every task (major and minor)
        has equal prior probability for both the trie weights and sampling.
        If False, uses the sampler's original p_minor.
    eps : float, default=1e-12
        Numerical floor for entropy computation.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        - MI_bits: estimated mutual information in bits
        - H_next_given_HR_bits: H(next | height, remaining)
        - H_next_given_prefix_bits: H(next | prefix)
        - n_pairs_used: number of (prefix -> next) pairs used
        - n_groups_HR: number of distinct (height, remaining) groups
        - pad: whether the task uses padding
        - includes_minor: whether minor tasks were included
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
        _, task, _ = nu.load_everything("dyck", exp_name)

    if seed is not None:
        torch.manual_seed(seed)

    one, neg = task.one, task.neg

    # Optionally set uniform prior
    original_p_minor = float(getattr(task, "p_minor", 0.0))
    if uniform_prior and int(getattr(task, "n_minor_tasks", 0)) > 0:
        task.p_minor = task.n_minor_tasks / (task.n_major_tasks + task.n_minor_tasks)

    # ---- Build weighted trie to match current prior over tasks ----
    trie = WeightedDyckTrie()

    n_major = int(getattr(task, "n_major_tasks", 0))
    n_minor = int(getattr(task, "n_minor_tasks", 0))
    p_minor = float(getattr(task, "p_minor", 0.0))

    if n_major <= 0:
        raise ValueError("Need n_major_tasks > 0 for train-mode task prior.")

    w_major = (1.0 - p_minor) / max(1, n_major)
    w_minor = (p_minor / max(1, n_minor)) if n_minor > 0 else 0.0

    # Insert major pool
    for seq in task.major_task_pool:
        steps = [(+1 if int(s) == int(one) else -1) for s in seq.tolist()]
        trie.insert(steps, w_major)

    # Insert minor pool (if present)
    if n_minor > 0 and task.minor_task_pool is not None:
        for seq in task.minor_task_pool:
            steps = [(+1 if int(s) == int(one) else -1) for s in seq.tolist()]
            trie.insert(steps, w_minor)

    L_dyck = int(task.major_task_pool.shape[1])  # = 2 * dyck_length

    # ---- Sample sequences in training mode ----
    samples, masks = task.generate(mode="train", num_samples=num_samples)

    # generate(mode="train") returns (1, B, L) — squeeze the epoch dimension
    if samples.dim() == 3:
        samples = samples.squeeze(0)
        masks = masks.squeeze(0)

    # Use non-padded timeline
    if getattr(task, "pad", False):
        x = samples[:, ::2]
        m = masks[:, ::2]
    else:
        x = samples
        m = masks

    B, _ = x.shape
    m_bool = m.to(torch.bool)

    # ---- Collect prefix-based probs and group by (height, remaining) ----
    p_list = []
    group_ps = defaultdict(list)

    for b in range(B):
        pos = torch.nonzero(m_bool[b], as_tuple=False).squeeze(1)
        if pos.numel() < 2:
            continue

        dyck_tokens = x[b, pos]
        steps = [(+1 if int(tok) == int(one) else -1) for tok in dyck_tokens.tolist()]

        height = 0
        for i in range(len(steps) - 1):
            height += steps[i]
            remaining = L_dyck - (i + 1)

            # i is the 0-indexed Dyck position of the prefix end;
            # we predict step i+1, so filter on position i
            if i < min_dyck_position:
                continue
            if max_dyck_position is not None and i >= max_dyck_position:
                continue

            prefix = steps[: i + 1]
            p = trie.p_next_plus(prefix)
            if p is None:
                continue

            p_list.append(p)
            group_ps[(height, remaining)].append(p)

    n = len(p_list)
    if n == 0:
        raise RuntimeError("No usable (prefix -> next) pairs found in train samples.")

    # H(next | prefix)
    H_given_P = sum(_binary_entropy_bits(p, eps=eps) for p in p_list) / n

    # H(next | height, remaining): average p within each (h,r) group, then entropy
    H_given_HR = 0.0
    for _, ps in group_ps.items():
        w = len(ps) / n
        q = sum(ps) / len(ps)
        H_given_HR += w * _binary_entropy_bits(q, eps=eps)

    MI = H_given_HR - H_given_P

    # Restore original p_minor
    task.p_minor = original_p_minor

    return {
        "MI_bits": MI,
        "H_next_given_HR_bits": H_given_HR,
        "H_next_given_prefix_bits": H_given_P,
        "n_pairs_used": n,
        "n_groups_HR": len(group_ps),
        "pad": bool(getattr(task, "pad", False)),
        "includes_minor": bool(n_minor > 0 and p_minor > 0),
        "uniform_prior": uniform_prior,
    }


def plot_mi_vs_k_dyck(
    k_values,
    num_samples: int = 8192,
    min_dyck_position: int = 0,
    max_dyck_position: int = None,
    uniform_prior: bool = True,
    seed=None,
    figsize: tuple = (10, 6),
    save_path=None,
    show: bool = True,
    verbose: bool = False,
):
    """
    Compute I(prefix; next_step | height, remaining) for each k and plot MI vs k.

    Parameters
    ----------
    k_values : list of int
        List of k values where number of minor tasks = 2^k.
    num_samples : int, default=8192
        Number of sequences per experiment for MI estimation.
    min_dyck_position : int, default=0
        Only include Dyck positions >= this index in the MI computation.
    max_dyck_position : int, optional
        Only include Dyck positions < this index. None = no upper bound.
    seed : int, optional
        Random seed (incremented per k for independence).
    figsize : tuple, default=(10, 6)
        Figure size.
    save_path : str, optional
        Path to save the figure.
    show : bool, default=True
        Whether to display the plot.
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    dict with keys:
        - 'k_values': list of k values
        - 'mi_bits': list of MI values in bits
        - 'H_prefix': list of H(next | prefix) values
        - 'H_hr': list of H(next | height, remaining) values
        - 'fig': matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from icl.utils.unified_interface import get_exp_name

    mi_bits = []
    h_prefix = []
    h_hr = []

    for i, k in enumerate(k_values):
        exp_name = get_exp_name("dyck", k)
        if verbose:
            print(f"Processing k={k} (2^k={2**k} minor tasks), exp={exp_name}")

        try:
            s = seed + i if seed is not None else None
            result = estimate_mi_prefix_vs_height_train(
                exp_name=exp_name, num_samples=num_samples,
                min_dyck_position=min_dyck_position,
                max_dyck_position=max_dyck_position,
                uniform_prior=uniform_prior,
                seed=s,
            )
            mi_bits.append(result["MI_bits"])
            h_prefix.append(result["H_next_given_prefix_bits"])
            h_hr.append(result["H_next_given_HR_bits"])

            if verbose:
                print(f"  MI={result['MI_bits']:.4f} bits, "
                      f"H(next|prefix)={result['H_next_given_prefix_bits']:.4f}, "
                      f"H(next|h,r)={result['H_next_given_HR_bits']:.4f}")
        except Exception as e:
            print(f"Warning: k={k} failed: {e}")
            mi_bits.append(float('nan'))
            h_prefix.append(float('nan'))
            h_hr.append(float('nan'))

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(k_values, mi_bits, 'o-', linewidth=2, markersize=8, color='blue',
            label='I(prefix; next | h, r)')
    ax.plot(k_values, h_hr, 's--', linewidth=1.5, markersize=6, color='gray',
            alpha=0.7, label='H(next | h, r)')
    ax.plot(k_values, h_prefix, '^--', linewidth=1.5, markersize=6, color='orange',
            alpha=0.7, label='H(next | prefix)')
    ax.set_xlabel('k (log2 of number of minor tasks)', fontsize=12)
    ax.set_ylabel('Bits', fontsize=12)
    ax.set_title('Mutual Information: Prefix vs (Height, Remaining)\nDyck Task', fontsize=14)
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
        'H_prefix': h_prefix,
        'H_hr': h_hr,
        'fig': fig,
    }