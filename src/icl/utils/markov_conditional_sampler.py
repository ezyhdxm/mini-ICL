import numpy as np
from typing import Iterator, Tuple, Optional

def stationary_distribution(P, atol: float = 1e-12) -> np.ndarray:
    n = P.shape[0]
    A = np.vstack([P - np.eye(n), np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0

    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.real(pi)
    pi[np.abs(pi) < atol] = 0.0
    pi = np.clip(pi, 0.0, None)
    pi = pi / pi.sum()
    return pi

def reverse_transition(P, pi, atol: float = 1e-15) -> np.ndarray:
    n = P.shape[0]
    numer = (pi[:, None] * P)
    R = np.zeros((n, n), dtype=float)
    mask = pi > atol
    R[mask, :] = numer.T[mask, :] / pi[mask, None]

    row_sums = R.sum(axis=1, keepdims=True)
    good = row_sums[:, 0] > 0
    R[good, :] /= row_sums[good, :]

    return R

def build_cdf(M):
    cdf = np.cumsum(M, axis=1)
    cdf[:, -1] = 1.0
    return cdf

def sample_from_cdf_rows(
    states: np.ndarray,
    cdf: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Vectorized grouped categorical sampling.

    states: shape (m,), integer in [0..n-1]
    cdf: shape (n,n), row r is CDF for categorical distribution over [0..n-1]
    Returns samples: shape (m,), integer in [0..n-1]
    """
    m = states.shape[0]
    n = cdf.shape[0]
    out = np.empty(m, dtype=np.int32)

    # Group by state value (n is small, so this is fast)
    for s in range(n):
        idx = np.where(states == s)[0]
        if idx.size:
            u = rng.random(idx.size)
            out[idx] = np.searchsorted(cdf[s], u, side="right")

    return out

def sample_centered_given_value(
    cdfP: np.ndarray,
    cdfR: np.ndarray,
    T: int,
    v: int,
    num_samples: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample num_samples independent trajectories on times [-T..T] (length 2T+1),
    conditioned on X_0 = v, in a stationary Markov chain.

    Returns: X_centered, shape (num_samples, 2T+1)
             where X_centered[:, T] is the center time 0.
    """
    n = cdfP.shape[0]
    L = 2 * T + 1
    center = T

    X = np.empty((num_samples, L), dtype=np.int32)
    X[:, center] = v

    # Sample past: times -1, -2, ..., -T using reverse kernel R
    # In array indices: center-1 down to 0
    for k in range(center - 1, -1, -1):
        next_states = X[:, k + 1]  # corresponds to time (k+1-center)
        X[:, k] = sample_from_cdf_rows(next_states, cdfR, rng)

    # Sample future: times 1, 2, ..., T using forward kernel P
    # In array indices: center up to L-2
    for k in range(center, L - 1):
        cur_states = X[:, k]
        X[:, k + 1] = sample_from_cdf_rows(cur_states, cdfP, rng)

    return X

def iter_samples_all_positions_all_vocab(
    P: np.ndarray,
    T: int,
    num_samples: int,
    rng: Optional[np.random.Generator] = None,
    copy_windows: bool = False
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """
    Generator that yields (t, v, samples_tv), where samples_tv has shape
    (num_samples, T+1) and is distributed as (X_0..X_T) | (X_t = v),
    assuming the chain is stationary (started in pi).

    Implementation uses the centered trick for efficiency.

    Parameters:
      copy_windows:
        - False: yield views into an internal centered array (fast, low-memory).
        - True:  yield a copy per (t,v) (safer if you store results).
    """
    if rng is None:
        rng = np.random.default_rng()

    P = np.asarray(P, dtype=float)
    n = P.shape[0]

    pi = stationary_distribution(P)
    # If some states have pi[v]=0, conditioning on them is impossible under stationarity
    R = reverse_transition(P, pi)

    cdfP = build_cdf(P)
    cdfR = build_cdf(R)

    for v in range(n):
        if pi[v] <= 0:
            continue  # impossible event X_t=v under stationary start

        centered = sample_centered_given_value(cdfP, cdfR, T, v, num_samples, rng)
        # Extract windows for each t: window corresponds to times [-t .. T-t]
        # centered index for time u is (u + T)
        for t in range(T + 1):
            start = T - t
            window = centered[:, start:start + (T + 1)]
            yield (t, v, window.copy() if copy_windows else window)

def sample_all_positions_all_vocab_array(
    P: np.ndarray,
    T: int,
    num_samples: int,
    rng: Optional[np.random.Generator] = None,
    dtype=np.int16
) -> np.ndarray:
    """
    Returns a dense array A with shape (T+1, n, num_samples, T+1) where
    A[t, v] are samples from (X_0..X_T) | (X_t=v).

    Note: requires memory O((T+1)^2 * n * num_samples).
    """
    if rng is None:
        rng = np.random.default_rng()

    P = np.asarray(P, dtype=float)
    n = P.shape[0]

    out = np.empty((T + 1, n, num_samples, T + 1), dtype=dtype)

    for t, v, samples_tv in iter_samples_all_positions_all_vocab(
        P, T, num_samples, rng=rng, copy_windows=True
    ):
        out[t, v, :, :] = samples_tv.astype(dtype, copy=False)

    return out
