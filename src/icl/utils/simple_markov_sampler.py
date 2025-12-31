import numpy as np
from typing import Optional
import torch


def sample_markov_trajectories(
    P: np.ndarray,
    T: int,
    num_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample base Markov trajectories X of shape (num_samples, T+1) from transition matrix P.
    X[:,0] ~ Uniform({0,...,n-1}) and X[:,t+1] ~ P[X[:,t], :].

    Returns: X (num_samples, T+1) int64
    """
    if rng is None:
        rng = np.random.default_rng()

    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"P must be square (n,n), got {P.shape}")
    if T < 0:
        raise ValueError("T must be >= 0")
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    n = P.shape[0]
    row_sums = P.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Each row of P must have positive sum.")
    P = P / row_sums

    # CDF for inverse-CDF sampling
    cdf = np.cumsum(P, axis=1)
    cdf[:, -1] = 1.0

    X = np.empty((num_samples, T + 1), dtype=np.int64)
    X[:, 0] = rng.integers(0, n, size=num_samples)

    for t in range(T):
        prev = X[:, t]
        u = rng.random(num_samples)

        nxt = np.empty(num_samples, dtype=np.int64)
        # group by prev-state (same idea as your naive version)
        for s in np.unique(prev):
            mask = (prev == s)
            nxt[mask] = np.searchsorted(cdf[s], u[mask], side="right")

        X[:, t + 1] = nxt

    return X


def trajectories_to_interleaved_tokens(
    X: np.ndarray,
    *,
    sep_id: int,
    dtype=np.int16,
) -> np.ndarray:
    """
    Convert base trajectories X (B, T+1) into token sequences (B, Sfull)
    where Sfull = 2*(T+1)-1 and pattern is:
      [X0, sep, X1, sep, ..., X_{T-1}, sep, X_T]
    If you want to DROP the last state (to match your earlier sample[:-1]),
    pass X[:, :-1] into this function.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (B, L), got {X.shape}")

    B, L = X.shape
    Sfull = 2 * L - 1

    out = np.empty((B, Sfull), dtype=dtype)
    out[:, 0::2] = X.astype(dtype, copy=False)        # even positions: states
    out[:, 1::2] = np.array(sep_id, dtype=dtype)      # odd positions: sep
    return out


def get_all_samples_base_only(
    n_tasks: int,
    sampler_clone0,
    num_samples: int,
    rng: Optional[np.random.Generator] = None,
    dtype=np.int16,
) -> torch.Tensor:
    """
    Returns samples of shape (n_tasks+3, B, Sfull) on CPU as a torch.LongTensor,
    where B=num_samples and Sfull=2*seq_len-1.

    IMPORTANT: matches your earlier convention:
      - you previously used sample[:-1] (drop the final state) when filling ::2.
      - so here we sample X of length seq_len = T+1, then use X[:, :-1]
        so the number of state tokens equals seq_len-1 (=Tm1), consistent with
        extracting at sep positions for t in [0..Tm1-1].

    If you actually want to keep X_T too, remove the '[:-1]' drop.
    """
    if rng is None:
        rng = np.random.default_rng()

    seq_len = int(sampler_clone0.seq_len)          # = T+1
    T = seq_len - 1                               # Markov steps
    V = int(sampler_clone0.num_states)
    sep_id = V

    n_total = n_tasks + 3
    B = int(num_samples)

    # We'll build (n_total, B, Sfull) as numpy then convert to torch.
    # We drop the last state so L = seq_len-1 = T, hence Sfull = 2*T-1 = 2*(seq_len-1)-1
    # BUT your KV code expects Sfull = 2*seq_len - 1 when seq_len=(Sfull+1)//2.
    # So we should NOT change seq_len. Instead: keep interleaving length based on (seq_len),
    # but fill the last state position with something consistent.
    #
    # To match your earlier samples shape ( ... , 2*seq_len - 1 ),
    # we will interleave L = seq_len (so Sfull=2*seq_len-1) and then overwrite
    # the final state token with sep or keep X_T.
    #
    # Easiest: keep full X (length seq_len), interleave to (B, 2*seq_len-1).
    # Then, if you want to "ignore" X_T, you can just not use it later.
    #
    # HOWEVER: your earlier fill used sample[:-1] into even slots, leaving the last even slot
    # (corresponding to X_T) not used. In that layout, the last token exists and is often sep.
    # We'll replicate that exactly: set last token to sep_id.

    Sfull = 2 * seq_len - 1
    all_samples = np.empty((n_total, B, Sfull), dtype=dtype)

    # Major tasks: first 3
    P_major = sampler_clone0.major_trans_mat.cpu().numpy()
    for i in range(3):
        X = sample_markov_trajectories(P_major[i], T=T, num_samples=B, rng=rng)  # (B, seq_len)
        seq = trajectories_to_interleaved_tokens(X, sep_id=sep_id, dtype=dtype)  # (B, Sfull)

        # Match your previous convention: you effectively didn't care about the final state.
        # If you want the last token to be sep (like many of your constructed sequences), do:
        seq[:, -1] = sep_id

        all_samples[i] = seq

    # Minor tasks: next n_tasks
    P_minors = sampler_clone0.minor_trans_mat[:n_tasks].cpu().numpy()
    for j in range(n_tasks):
        i = 3 + j
        X = sample_markov_trajectories(P_minors[j], T=T, num_samples=B, rng=rng)
        seq = trajectories_to_interleaved_tokens(X, sep_id=sep_id, dtype=dtype)
        seq[:, -1] = sep_id
        all_samples[i] = seq

    # Return torch.LongTensor (your model expects long token ids)
    return torch.from_numpy(all_samples).long()
