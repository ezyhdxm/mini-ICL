import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List


# -----------------------------
# Core math utilities
# -----------------------------
def sample_dirichlet1(n: int, V: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n categorical distributions p ~ Dirichlet(1) on V categories."""
    return rng.dirichlet(np.ones(V, dtype=np.float64), size=n).astype(np.float64)


def smooth_simplex(P: np.ndarray, gamma: float) -> np.ndarray:
    """
    Smooth distributions to ensure a positive floor:
        P <- (1-gamma) P + gamma * Uniform
    This guarantees P[i] >= gamma/V (up to floating error).
    """
    if gamma <= 0:
        return P
    V = P.shape[-1]
    U = np.full((V,), 1.0 / V, dtype=np.float64)
    return (1.0 - gamma) * P + gamma * U


def kl_matrix_forward(P: np.ndarray, Q: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Compute the matrix of forward KL divergences:
        KL(P_n || Q_k) for all n,k.

    P: (N, V) rows are distributions.
    Q: (K, V) rows are distributions.
    returns: (N, K)
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    P_c = np.clip(P, eps, 1.0)
    Q_c = np.clip(Q, eps, 1.0)

    logP = np.log(P_c)              # (N,V)
    logQ = np.log(Q_c)              # (K,V)

    # KL(P||Q) = sum_i P_i(log P_i - log Q_i)
    #          = sum_i P_i logP_i - sum_i P_i logQ_i
    Hp = np.sum(P_c * logP, axis=1, keepdims=True)   # (N,1)
    cross = P_c @ logQ.T                             # (N,K) where cross[n,k] = sum_i P_n,i log Q_k,i
    return Hp - cross


def delta_min(P: np.ndarray, centers: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    For each distribution p in P, compute:
        delta_min(p) = (2nd smallest KL) - (smallest KL)
    among the candidate centers.
    """
    D = kl_matrix_forward(P, centers, eps=eps)  # (N, K)
    # First 2 order statistics via partition:
    two = np.partition(D, kth=1, axis=1)[:, :2]  # (N,2), unordered among first 2
    best = np.minimum(two[:, 0], two[:, 1])
    second = np.maximum(two[:, 0], two[:, 1])
    return second - best


# -----------------------------
# KL–Bregman k-means
# -----------------------------
def init_farthest(P: np.ndarray, K: int, rng: np.random.Generator, eps: float = 1e-15) -> np.ndarray:
    """
    Deterministic farthest-point initialization under KL(P||center):
    - pick first center uniformly at random from samples
    - then repeatedly pick the point farthest from its nearest current center.
    """
    N = P.shape[0]
    idx0 = int(rng.integers(0, N))
    centers = [P[idx0].copy()]

    for _ in range(1, K):
        Q = np.stack(centers, axis=0)       # (k,V)
        D = kl_matrix_forward(P, Q, eps=eps)  # (N,k)
        dmin = D.min(axis=1)
        idx = int(np.argmax(dmin))
        centers.append(P[idx].copy())

    return np.stack(centers, axis=0)


@dataclass
class KLBregmanKMeansResult:
    centers: np.ndarray          # (K,V)
    assignments: np.ndarray      # (N,)
    obj_history: List[float]     # mean min KL each iter


def fit_kl_bregman_kmeans(
    P: np.ndarray,
    K: int = 3,
    gamma: float = 0.02,
    max_iter: int = 100,
    tol: float = 1e-8,
    n_init: int = 5,
    rng: Optional[np.random.Generator] = None,
    eps: float = 1e-15,
    verbose: bool = False,
) -> KLBregmanKMeansResult:
    """
    Monte Carlo KL–Bregman k-means minimizing E[min_m KL(p || center_m)]
    on samples P ~ Dirichlet(1).

    Update rule: Bregman centroid for forward KL is the arithmetic mean of points in the cluster.
    Smoothing: after each update, apply P <- (1-gamma)P + gamma*Uniform.

    n_init: multiple restarts; pick the solution with smallest final objective.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    P = np.asarray(P, dtype=np.float64)
    N, V = P.shape

    best_centers = None
    best_assign = None
    best_hist = None
    best_obj = np.inf

    for rep in range(n_init):
        # Init
        centers = init_farthest(P, K, rng=rng, eps=eps)
        centers = smooth_simplex(centers, gamma)

        assignments = np.zeros(N, dtype=np.int64)
        obj_history = []

        for it in range(max_iter):
            D = kl_matrix_forward(P, centers, eps=eps)     # (N,K)
            new_assign = D.argmin(axis=1)
            minD = D[np.arange(N), new_assign]
            obj = float(np.mean(minD))
            obj_history.append(obj)

            # Stop if assignments unchanged
            if it > 0 and np.array_equal(new_assign, assignments):
                if verbose:
                    print(f"[rep={rep}] converged by assignments at iter {it}, obj={obj:.6f}")
                break

            assignments = new_assign

            # Update centers = mean of cluster points, with empty-cluster handling
            new_centers = np.zeros((K, V), dtype=np.float64)
            for k in range(K):
                mask = (assignments == k)
                if not np.any(mask):
                    # Re-seed empty cluster to farthest point
                    dmin = D.min(axis=1)
                    idx = int(np.argmax(dmin))
                    new_centers[k] = P[idx]
                else:
                    new_centers[k] = P[mask].mean(axis=0)

            new_centers = smooth_simplex(new_centers, gamma)

            # Stop if centers barely move
            shift = float(np.max(np.abs(new_centers - centers)))
            centers = new_centers
            if shift < tol:
                if verbose:
                    print(f"[rep={rep}] converged by shift at iter {it}, shift={shift:.3e}, obj={obj:.6f}")
                break

        # Track best restart by final objective (coverage objective)
        final_obj = obj_history[-1]
        if final_obj < best_obj:
            best_obj = final_obj
            best_centers = centers.copy()
            best_assign = assignments.copy()
            best_hist = obj_history

    return KLBregmanKMeansResult(
        centers=best_centers,
        assignments=best_assign,
        obj_history=best_hist,
    )


# -----------------------------
# Random Dirichlet baseline prototypes
# -----------------------------
def random_dirichlet_prototypes(
    V: int,
    K: int = 3,
    gamma: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Baseline: sample K prototypes independently from Dirichlet(1) and smooth."""
    if rng is None:
        rng = np.random.default_rng(0)
    centers = rng.dirichlet(np.ones(V, dtype=np.float64), size=K).astype(np.float64)
    return smooth_simplex(centers, gamma)


# -----------------------------
# Evaluation helpers
# -----------------------------
def summarize_deltas(d: np.ndarray, eps_grid: Optional[List[float]] = None) -> Dict[str, float]:
    """Return summary stats for a delta array."""
    d = np.asarray(d, dtype=np.float64)
    out = {
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "p10": float(np.quantile(d, 0.10)),
        "p25": float(np.quantile(d, 0.25)),
        "p75": float(np.quantile(d, 0.75)),
        "p90": float(np.quantile(d, 0.90)),
    }
    if eps_grid is not None:
        for eps in eps_grid:
            out[f"P(delta>= {eps})"] = float(np.mean(d >= eps))
    return out


def demo_compare(
    V: int = 10,
    N_train: int = 20000,
    N_test: int = 50000,
    gamma: float = 0.02,
    n_init: int = 5,
    seed: int = 123,
) -> None:
    """
    Train KL–Bregman k-means on Dir(1) samples, compare to random Dir(1) prototypes
    in terms of delta_min on fresh Dir(1) test samples.
    """
    rng = np.random.default_rng(seed)

    # Training data: distributions p ~ Dir(1)
    P_train = sample_dirichlet1(N_train, V, rng)

    # Fit k-means prototypes
    km = fit_kl_bregman_kmeans(P_train, K=3, gamma=gamma, n_init=n_init, rng=rng, max_iter=100)

    # Random baseline prototypes
    rnd_centers = random_dirichlet_prototypes(V, K=3, gamma=gamma, rng=rng)

    # Test distributions
    P_test = sample_dirichlet1(N_test, V, rng)

    d_km = delta_min(P_test, km.centers)
    d_rnd = delta_min(P_test, rnd_centers)

    eps_grid = [0.01, 0.05, 0.10, 0.20, 0.30]

    print("=== KL–Bregman k-means prototypes (trained on Dir(1)) ===")
    print("centers shape:", km.centers.shape)
    print("objective last:", km.obj_history[-1])
    print(summarize_deltas(d_km, eps_grid=eps_grid))

    print("\n=== Random Dirichlet(1) prototypes baseline ===")
    print("centers shape:", rnd_centers.shape)
    print(summarize_deltas(d_rnd, eps_grid=eps_grid))





import numpy as np
from typing import Tuple, Dict, List, Optional


# ---------- reuse from your earlier code ----------
def sample_dirichlet1(n: int, V: int, rng: np.random.Generator) -> np.ndarray:
    return rng.dirichlet(np.ones(V, dtype=np.float64), size=n).astype(np.float64)

def smooth_simplex(P: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        return P
    V = P.shape[-1]
    U = np.full((V,), 1.0 / V, dtype=np.float64)
    return (1.0 - gamma) * P + gamma * U

def kl_matrix_forward(P: np.ndarray, Q: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    P_c = np.clip(P, eps, 1.0)
    Q_c = np.clip(Q, eps, 1.0)
    logP = np.log(P_c)
    logQ = np.log(Q_c)
    Hp = np.sum(P_c * logP, axis=1, keepdims=True)    # (N,1)
    cross = P_c @ logQ.T                               # (N,K)
    return Hp - cross

def delta_min(P: np.ndarray, centers: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    D = kl_matrix_forward(P, centers, eps=eps)         # (N,K)
    two = np.partition(D, kth=1, axis=1)[:, :2]        # first two order stats
    best = np.minimum(two[:, 0], two[:, 1])
    second = np.maximum(two[:, 0], two[:, 1])
    return second - best

def summarize(d: np.ndarray, eps_grid: Optional[List[float]] = None) -> Dict[str, float]:
    d = np.asarray(d, dtype=np.float64)
    out = {
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "p10": float(np.quantile(d, 0.10)),
        "p25": float(np.quantile(d, 0.25)),
        "p75": float(np.quantile(d, 0.75)),
        "p90": float(np.quantile(d, 0.90)),
    }
    if eps_grid is not None:
        for eps in eps_grid:
            out[f"P(delta>= {eps})"] = float(np.mean(d >= eps))
    return out

def winner_frequencies(P: np.ndarray, centers: np.ndarray) -> np.ndarray:
    D = kl_matrix_forward(P, centers)
    w = D.argmin(axis=1)
    K = centers.shape[0]
    return np.bincount(w, minlength=K) / len(w)


# ---------- proposed better method ----------
def block_prototypes(V: int, sizes: Tuple[int, int, int], gamma: float = 0.02) -> np.ndarray:
    """
    Construct 3 block prototypes with full support:
      - prototype m is uniform on its block, tiny outside, plus Dirichlet-style smoothing gamma.
    For Dirichlet(1), only the block sizes matter (not which labels), but we still build an explicit partition.
    """
    assert sum(sizes) == V and all(s > 0 for s in sizes), "sizes must be positive and sum to V"
    centers = np.zeros((3, V), dtype=np.float64)
    start = 0
    U = np.full(V, 1.0 / V, dtype=np.float64)
    for m, s in enumerate(sizes):
        ids = np.arange(start, start + s)
        start += s
        base = np.zeros(V, dtype=np.float64)
        base[ids] = 1.0 / s
        centers[m] = (1.0 - gamma) * base + gamma * U
    return centers

def random_dirichlet_prototypes(V: int, K: int = 3, gamma: float = 0.02,
                               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(0)
    centers = rng.dirichlet(np.ones(V, dtype=np.float64), size=K).astype(np.float64)
    return smooth_simplex(centers, gamma)


def search_best_block_sizes(
    V: int,
    gamma: float,
    P_eval: np.ndarray,
    objective: str = "p10",
    min_winner_prob: float = 0.0,
) -> Tuple[Tuple[int, int, int], Dict[str, float]]:
    """
    Search over all ordered integer triples (a,b,c) with a+b+c=V, a,b,c>=1.
    Choose the one maximizing a chosen objective of Δmin on P_eval.

    objective in {"mean","median","p10","p25","P>=0.1","P>=0.2",...}
    min_winner_prob: optional fairness constraint so no prototype is almost never selected.
    """
    best_sizes = None
    best_value = -np.inf
    best_summary = None

    eps_grid = [0.01, 0.05, 0.1, 0.2, 0.3]

    for a in range(1, V - 1):
        for b in range(1, V - a):
            c = V - a - b
            sizes = (a, b, c)
            centers = block_prototypes(V, sizes, gamma=gamma)
            wf = winner_frequencies(P_eval, centers)
            if wf.min() < min_winner_prob:
                continue

            d = delta_min(P_eval, centers)
            s = summarize(d, eps_grid=eps_grid)

            # parse objective
            if objective in s:
                val = s[objective]
            elif objective.startswith("P>="):
                thr = float(objective.split(">=")[1])
                val = float(np.mean(d >= thr))
            else:
                raise ValueError(f"Unknown objective: {objective}")

            if val > best_value:
                best_value = val
                best_sizes = sizes
                best_summary = s

    if best_sizes is None:
        raise RuntimeError("No feasible block sizes found (try lowering min_winner_prob).")

    return best_sizes, best_summary


def demo_block_vs_random(
    V: int = 10,
    gamma: float = 0.02,
    N_eval: int = 50000,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    P_eval = sample_dirichlet1(N_eval, V, rng)

    # Choose block sizes automatically (optimize p10 margin with a mild fairness constraint)
    sizes, best_sum = search_best_block_sizes(
        V=V,
        gamma=gamma,
        P_eval=P_eval,
        objective="p10",           # optimize lower-tail margin
        min_winner_prob=0.10,      # ensure each prototype wins at least 10% of the time
    )
    block_centers = block_prototypes(V, sizes, gamma=gamma)

    # Random baseline
    rand_centers = random_dirichlet_prototypes(V, K=3, gamma=gamma, rng=rng)

    # Evaluate Δmin on fresh samples
    P_test = sample_dirichlet1(N_eval, V, rng)
    d_block = delta_min(P_test, block_centers)
    d_rand = delta_min(P_test, rand_centers)

    eps_grid = [0.01, 0.05, 0.10, 0.20, 0.30]

    print("=== Max-margin block prototypes ===")
    print("sizes:", sizes, "gamma:", gamma)
    print("winner frequencies:", winner_frequencies(P_test, block_centers))
    print(summarize(d_block, eps_grid=eps_grid))

    print("\n=== Random Dirichlet(1) baseline prototypes ===")
    print("winner frequencies:", winner_frequencies(P_test, rand_centers))
    print(summarize(d_rand, eps_grid=eps_grid))

