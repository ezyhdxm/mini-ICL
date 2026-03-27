import math
from dataclasses import dataclass
from collections import defaultdict

import torch


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
        - includes_minor: whether minor tasks were included
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
        _, task, _ = nu.load_everything("dyck", exp_name)

    if seed is not None:
        torch.manual_seed(seed)

    one, neg = task.one, task.neg

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

    for seq in task.major_task_pool:
        steps = [(+1 if int(s) == int(one) else -1) for s in seq.tolist()]
        trie.insert(steps, w_major)

    if n_minor > 0 and task.minor_task_pool is not None:
        for seq in task.minor_task_pool:
            steps = [(+1 if int(s) == int(one) else -1) for s in seq.tolist()]
            trie.insert(steps, w_minor)

    L_dyck = int(task.major_task_pool.shape[1])  # = 2 * dyck_length

    # ---- Sample sequences in training mode ----
    samples, masks = task.generate(mode="train", num_samples=num_samples)

    if samples.dim() == 3:
        samples = samples.squeeze(0)
        masks = masks.squeeze(0)

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

    H_given_P = sum(_binary_entropy_bits(p, eps=eps) for p in p_list) / n

    H_given_HR = 0.0
    for _, ps in group_ps.items():
        w = len(ps) / n
        q = sum(ps) / len(ps)
        H_given_HR += w * _binary_entropy_bits(q, eps=eps)

    MI = H_given_HR - H_given_P

    task.p_minor = original_p_minor

    return {
        "MI_bits": MI,
        "H_next_given_HR_bits": H_given_HR,
        "H_next_given_prefix_bits": H_given_P,
        "n_pairs_used": n,
        "n_groups_HR": len(group_ps),
        "includes_minor": bool(n_minor > 0 and p_minor > 0),
        "uniform_prior": uniform_prior,
    }
