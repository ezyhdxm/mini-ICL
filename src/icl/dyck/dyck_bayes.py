import torch


class TrieNode:
    def __init__(self) -> None:
        self.children = {} # key: -1 or 1
        self.count = 0 # number of sequences that pass through this node
        self.count_pos = 0 # number of sequences where next token is 1


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, sequence, weight=1.0):
        w = float(weight)
        node = self.root
        for i in range(len(sequence) - 1):
            token = sequence[i]
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
            node.count += w
            if sequence[i+1] == 1:
                node.count_pos += w


class DyckBayes:
    def __init__(self, config, sampler, flag=False):
        self.num_states = config.vocab_size
        self.dyck_length = config.task.dyck_length
        self.trans_matrix = sampler.trans_matrix
        self.repeat_prob = config.task.repeat_prob
        self.order = config.task.order
        self.trie = None
        self.major_trie = None
        self.one = sampler.one
        self.neg = sampler.neg
        self.flag = flag
        self.p_minor = float(getattr(sampler, "p_minor", 0.0))
        self.n_major_tasks = int(getattr(sampler, "n_major_tasks", 0))
        self.n_minor_tasks = int(getattr(sampler, "n_minor_tasks", 0))

        # Build Trie from all task pools (major + minor)
        if sampler.total_trans > 0:
            self.trie = Trie()
            n_major = int(getattr(sampler, "n_major_tasks", 0))
            n_minor = int(getattr(sampler, "n_minor_tasks", 0))
            p_minor = float(getattr(sampler, "p_minor", 0.0))

            if n_major > 0 and n_minor > 0:
                w_major = (1.0 - p_minor) / n_major
                w_minor = p_minor / n_minor
            elif n_major > 0:
                w_major = 1.0 / n_major
                w_minor = 0.0
            elif n_minor > 0:
                w_major = 0.0
                w_minor = 1.0 / n_minor
            else:
                w_major = 0.0
                w_minor = 0.0

            # Insert major task pool
            if sampler.n_major_tasks > 0:
                self.major_trie = Trie()
                for seq in sampler.major_task_pool:
                    seq_steps = [1 if s == sampler.one else -1 for s in seq.tolist()]
                    self.trie.insert(
                        seq_steps,
                        weight=w_major,
                    )
                    # Major-only trie for the hybrid baseline (major-aware, minor-agnostic).
                    self.major_trie.insert(
                        seq_steps,
                        weight=(1.0 / n_major) if n_major > 0 else 0.0,
                    )
            # Insert minor task pool
            if sampler.n_minor_tasks > 0 and sampler.minor_task_pool is not None:
                for seq in sampler.minor_task_pool:
                    self.trie.insert(
                        [1 if s == sampler.one else -1 for s in seq.tolist()],
                        weight=w_minor,
                    )

    def dyck_pos(self, dyckseq):
        """
        seq: Tensor of shape [2*dyck_length], where each element is -1 or 1.
        return: probs[i] = Pr(seq[i] = 1 | seq[:i])
        """
        eps = 1e-6
        probs = torch.zeros(dyckseq.shape[0], device=dyckseq.device, dtype=torch.float32) + eps
        seq_list = [1 if s==self.one else -1 for s in dyckseq.tolist()]

        if (self.trie is None) or self.flag:
            # combinatorial Dyck prior predictor ("new/minor unknown" component)
            dR, dU = 0, 0
            for i, s in enumerate(seq_list):
                p_new = (dR - dU + 2) / (dR - dU + 1) * (self.dyck_length - dR) / (2*self.dyck_length - dR - dU)

                if self.flag and (self.major_trie is not None):
                    # flag=True: major-aware but minor-agnostic
                    node = self.major_trie.root
                    valid_prefix = True
                    for tok in seq_list[:i]:
                        if tok not in node.children:
                            valid_prefix = False
                            break
                        node = node.children[tok]
                    if valid_prefix and node.count > 0:
                        p_major = node.count_pos / node.count
                        probs[i] = (1.0 - self.p_minor) * p_major + self.p_minor * p_new
                    else:
                        probs[i] = p_new
                else:
                    probs[i] = p_new

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

            cumsum = mask_b.int().cumsum(dim=0)
            tot = dyck_probs.shape[0]

            prob_b = torch.zeros(T, dtype=dyck_probs.dtype, device=seq.device)

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

        markov_out = torch.zeros((B, T, K), device=seq.device, dtype=torch.float32)

        if self.order == 0:
            markov_out[:, :, :] = self.trans_matrix[:K].unsqueeze(0).unsqueeze(0)
            return markov_out

        prev = torch.ones((B, K), device=seq.device) / K
        chosen_rows = torch.zeros((B, K), device=seq.device, dtype=torch.float32)

        for t in range(T):
            s_t = seq[:, t]  # [B]

            update_mask = (s_t == self.one) | (s_t == self.neg)

            updated = torch.matmul(prev, self.trans_matrix)  # [B, K]

            chosen_rows[~update_mask] = self.trans_matrix[s_t[~update_mask]]

            prev = torch.where(update_mask.unsqueeze(1), updated, chosen_rows)

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

        return probs[:,:-1,:]

    def predict(self, seq):
        probs = self.pos_prob(seq)
        preds = torch.argmax(probs, dim=-1)
        return preds
