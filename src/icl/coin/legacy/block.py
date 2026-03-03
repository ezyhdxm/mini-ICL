import numpy as np
from typing import Tuple, Dict, List, Optional


# -----------------------------
# Sampling: p ~ Dirichlet(1)
# -----------------------------
def sample_dirichlet1(n: int, V: int, rng: np.random.Generator) -> np.ndarray:
    return rng.dirichlet(np.ones(V, dtype=np.float64), size=n).astype(np.float64)


# -----------------------------
# KL utilities
# -----------------------------
def kl_matrix_forward(P: np.ndarray, Q: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Compute KL(P_n || Q_k) for all n,k.
    P: (N,V), Q: (K,V) returns: (N,K)
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    P_c = np.clip(P, eps, 1.0)
    Q_c = np.clip(Q, eps, 1.0)

    logP = np.log(P_c)  # (N,V)
    logQ = np.log(Q_c)  # (K,V)

    Hp = np.sum(P_c * logP, axis=1, keepdims=True)  # (N,1) = sum_i P log P
    cross = P_c @ logQ.T                             # (N,K) = sum_i P log Q
    return Hp - cross


def delta_min(P: np.ndarray, centers: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Δmin(p) = (2nd smallest KL) - (smallest KL), among centers.
    """
    D = kl_matrix_forward(P, centers, eps=eps)  # (N,K)
    two = np.partition(D, kth=1, axis=1)[:, :2]
    best = np.minimum(two[:, 0], two[:, 1])
    second = np.maximum(two[:, 0], two[:, 1])
    return second - best


def min_kl(P: np.ndarray, centers: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """min_m KL(p||center_m) for each p."""
    D = kl_matrix_forward(P, centers, eps=eps)
    return D.min(axis=1)


def winner_frequencies(P: np.ndarray, centers: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Fraction of p's for which each center is the argmin KL."""
    D = kl_matrix_forward(P, centers, eps=eps)
    w = D.argmin(axis=1)
    K = centers.shape[0]
    return np.bincount(w, minlength=K) / len(w)


# -----------------------------
# Block prototypes with fixed sizes
# -----------------------------
def make_block_partition_indices(
    V: int,
    sizes: Tuple[int, int, int],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return 3 arrays of indices forming a partition of {0,...,V-1} with given sizes.
    If rng is provided, randomly permute categories first. Under Dirichlet(1), labels are exchangeable,
    so permuting doesn't change the distribution of results, but it can be nice for sanity.
    """
    assert sum(sizes) == V and all(s > 0 for s in sizes)
    idx = np.arange(V)
    if rng is not None:
        rng.shuffle(idx)

    a, b, c = sizes
    S1 = idx[:a]
    S2 = idx[a:a + b]
    S3 = idx[a + b:]
    return S1, S2, S3


def block_prototypes_fixed_sizes(
    V: int,
    sizes: Tuple[int, int, int] = (3, 3, 4),
    gamma: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Build 3 block-emphasis prototypes with full support:
      q_m(x) = (1-gamma)*Uniform(on block m) + gamma*Uniform(on all categories)
    """
    S1, S2, S3 = make_block_partition_indices(V, sizes, rng=rng)
    blocks = [S1, S2, S3]

    U = np.full(V, 1.0 / V, dtype=np.float64)
    centers = np.zeros((3, V), dtype=np.float64)

    for m, S in enumerate(blocks):
        base = np.zeros(V, dtype=np.float64)
        base[S] = 1.0 / len(S)
        centers[m] = (1.0 - gamma) * base + gamma * U

    return centers


# -----------------------------
# Baselines: random prototypes
# -----------------------------
def random_dirichlet_prototypes(
    V: int,
    K: int = 3,
    gamma: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Baseline: sample K prototypes from Dirichlet(1), then (optionally) smooth them
    by mixing with uniform using the same gamma for fairness/comparability.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    centers = rng.dirichlet(np.ones(V, dtype=np.float64), size=K).astype(np.float64)
    U = np.full(V, 1.0 / V, dtype=np.float64)
    centers = (1.0 - gamma) * centers + gamma * U
    return centers


# -----------------------------
# Reporting / objective
# -----------------------------
def summarize_delta(d: np.ndarray, eps_grid: Optional[List[float]] = None) -> Dict[str, float]:
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


def objective_from_summary(summary: Dict[str, float], objective: str) -> float:
    """
    objective examples:
      - "p10", "median", "mean"
      - "P(delta>= 0.2)" (exact key)
    """
    if objective in summary:
        return float(summary[objective])
    raise ValueError(f"Unknown objective key: {objective}. Available keys: {list(summary.keys())}")


# -----------------------------
# Optimize gamma (fixed blocks)
# -----------------------------
def optimize_gamma_fixed_blocks(
    V: int = 10,
    sizes: Tuple[int, int, int] = (3, 3, 4),
    gamma_grid: Optional[np.ndarray] = None,
    N_val: int = 50000,
    seed: int = 0,
    objective: str = "p10",
    eps_grid: Optional[List[float]] = None,
) -> Dict[str, object]:
    """
    Grid-search gamma for block prototypes with fixed sizes.

    objective:
      - "mean", "median", "p10", "p25", "p75", "p90"
      - or threshold form like "P(delta>=0.2)" (no spaces).
    """
    rng = np.random.default_rng(seed)

    if gamma_grid is None:
        gamma_grid = np.logspace(-3, -0.7, 18)  # ~[0.001, 0.2]

    if eps_grid is None:
        eps_grid = [0.01, 0.05, 0.10, 0.20, 0.30]

    # Validation distributions
    P_val = sample_dirichlet1(N_val, V, rng)

    def compute_objective(d: np.ndarray, summary: Dict[str, float]) -> float:
        if objective in summary:
            return float(summary[objective])

        # Allow "P(delta>=0.2)" format
        if objective.startswith("P(delta>=") and objective.endswith(")"):
            thr_str = objective[len("P(delta>="):-1]
            thr = float(thr_str)
            return float(np.mean(d >= thr))

        raise ValueError(
            f"Unknown objective '{objective}'. "
            f"Use one of {list(summary.keys())} or 'P(delta>=0.2)'."
        )

    rows: List[Dict[str, float]] = []

    best_val = -np.inf
    best_gamma = None
    best_centers = None
    best_row = None

    for gamma in gamma_grid:
        gamma = float(gamma)
        centers = block_prototypes_fixed_sizes(V, sizes=sizes, gamma=gamma, rng=rng)
        d = delta_min(P_val, centers)

        s = summarize_delta(d, eps_grid=eps_grid)

        # extra diagnostics
        s["gamma"] = gamma
        s["mean_minKL"] = float(np.mean(min_kl(P_val, centers)))
        wf = winner_frequencies(P_val, centers)
        s["winner_freq_0"] = float(wf[0])
        s["winner_freq_1"] = float(wf[1])
        s["winner_freq_2"] = float(wf[2])

        val = compute_objective(d, s)
        s["objective_value"] = float(val)

        rows.append(s)

        if val > best_val:
            best_val = val
            best_gamma = gamma
            best_centers = centers.copy()
            best_row = s

    return {
        "best_gamma": best_gamma,
        "best_centers": best_centers,
        "best_row": best_row,
        "table": rows,
    }



# -----------------------------
# End-to-end demo: optimize gamma, then compare on test set
# -----------------------------
def demo_optimize_gamma_and_compare(
    V: int = 10,
    sizes: Tuple[int, int, int] = (3, 3, 4),
    N_val: int = 50000,
    N_test: int = 50000,
    seed: int = 0,
    objective: str = "p10",
):
    rng = np.random.default_rng(seed)
    eps_grid = [0.01, 0.05, 0.10, 0.20, 0.30]

    # 1) Choose gamma on validation set
    result = optimize_gamma_fixed_blocks(
        V=V, sizes=sizes, N_val=N_val, seed=seed, objective=objective, eps_grid=eps_grid
    )
    best_gamma = result["best_gamma"]
    centers_block = result["best_centers"]

    print("=== Fixed block sizes, optimized gamma ===")
    print("sizes:", sizes)
    print("objective:", objective)
    print("best_gamma:", best_gamma)
    print("val summary row:", result["best_row"])

    # 2) Evaluate on fresh test set
    P_test = sample_dirichlet1(N_test, V, rng)

    d_block = delta_min(P_test, centers_block)
    print("\n=== Block prototypes on TEST ===")
    print("winner frequencies:", winner_frequencies(P_test, centers_block))
    print(summarize_delta(d_block, eps_grid=eps_grid))

    # 3) Baseline: one random Dirichlet triple (smoothed with same gamma)
    centers_rnd = random_dirichlet_prototypes(V, K=3, gamma=best_gamma, rng=rng)
    d_rnd = delta_min(P_test, centers_rnd)
    print("\n=== Random Dirichlet baseline (same gamma) on TEST ===")
    print("winner frequencies:", winner_frequencies(P_test, centers_rnd))
    print(summarize_delta(d_rnd, eps_grid=eps_grid))

    # 4) Optional stronger baseline: best-of-R random triples by the same objective on validation set
    R = 50
    P_val2 = sample_dirichlet1(N_val, V, rng)
    best_val = -np.inf
    best_centers = None
    for _ in range(R):
        c = random_dirichlet_prototypes(V, K=3, gamma=best_gamma, rng=rng)
        d = delta_min(P_val2, c)
        s = summarize_delta(d, eps_grid=eps_grid)
        val = objective_from_summary(s, objective if objective.startswith("P(") else objective)
        if val > best_val:
            best_val = val
            best_centers = c

    d_best_rnd = delta_min(P_test, best_centers)
    print(f"\n=== Best-of-{R} random baseline (picked by {objective}) on TEST ===")
    print("winner frequencies:", winner_frequencies(P_test, best_centers))
    print(summarize_delta(d_best_rnd, eps_grid=eps_grid))

