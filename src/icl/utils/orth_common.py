"""Shared helpers for optimal orthogonal direction interventions.

Used by coin, latent_markov, and linear ``optimal_orth`` modules.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# OLS R² (80/20 held-out)
# ---------------------------------------------------------------------------

def ols_r2(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Held-out R² (80/20 split) for OLS with intercept: Y = [X, 1] W."""
    N = X.shape[0]
    n_tr = int(0.8 * N)
    perm = torch.randperm(N)
    Xtr, Ytr = X[perm[:n_tr]], Y[perm[:n_tr]]
    Xte, Yte = X[perm[n_tr:]], Y[perm[n_tr:]]
    Xa = torch.cat([Xtr, torch.ones(n_tr, 1)], dim=1)
    W = torch.linalg.pinv(Xa) @ Ytr
    pred = torch.cat([Xte, torch.ones(N - n_tr, 1)], dim=1) @ W
    ss_r = ((Yte - pred) ** 2).sum().item()
    ss_t = ((Yte - Yte.mean(0)) ** 2).sum().item()
    return 1.0 - ss_r / ss_t if ss_t > 0 else float("nan")


# ---------------------------------------------------------------------------
# Trial statistics (random-baseline summaries)
# ---------------------------------------------------------------------------

def trial_stats(vals) -> dict:
    """Mean and IQR (25th/75th percentile) of a list of scalar values."""
    if not vals:
        return {"mean": float("nan"), "q25": float("nan"), "q75": float("nan")}
    arr = np.array(vals, dtype=float)
    return {
        "mean": float(arr.mean()),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def iqr_err_norm(all_results, layers, key_batch, g_m):
    """IQR error bars in normalized (%) units across layers."""
    lo_arr, hi_arr = [], []
    for layer in layers:
        vals = all_results[layer].get(key_batch, [])
        if len(vals) < 2:
            lo_arr.append(0.0)
            hi_arr.append(0.0)
            continue
        arr = np.array(vals, dtype=float) / g_m * 100
        mn = arr.mean()
        lo_arr.append(abs(mn - np.percentile(arr, 25)))
        hi_arr.append(abs(np.percentile(arr, 75) - mn))
    return np.array(lo_arr), np.array(hi_arr)


# ---------------------------------------------------------------------------
# SVD basis with optional cumulative-variance threshold
# ---------------------------------------------------------------------------

def svd_basis(M: torch.Tensor, var_thresh=None):
    """Extract an orthonormal basis from the rows of M via SVD.

    Returns ``(basis, rank)`` where ``basis`` is ``(N, rank)`` —
    columns are the right-singular vectors kept.
    """
    _, S, Vt = torch.linalg.svd(M, full_matrices=False)
    r = int((S > 1e-6 * S[0]).sum().item())
    if var_thresh is not None and r > 0:
        cum = torch.cumsum(S[:r] ** 2, 0)
        r = min(r, int((cum < var_thresh * cum[-1]).sum().item()) + 1)
    return Vt[:r].T, r


# ---------------------------------------------------------------------------
# Forward-hook factory for subspace projection ablation
# ---------------------------------------------------------------------------

def make_projection_hook(P: torch.Tensor, scale: float = 1.0):
    """Return a forward hook that removes the component along projector P.

    The hook computes ``h' = h - scale * (h @ P)`` and preserves the
    tuple structure of the module output when present.
    """
    def _hook(mod, inp, out, _P=P):
        h = out if torch.is_tensor(out) else out[0]
        hm = h - scale * (h @ _P)
        return hm if torch.is_tensor(out) else (hm,) + out[1:]
    return _hook


# ---------------------------------------------------------------------------
# Generate sequences and cache unhooked logits
# ---------------------------------------------------------------------------

def gen_and_cache(model, sampler, mode, n, B, device):
    """Generate sequences, run unhooked forward pass, cache logits."""
    seqs, logits = [], []
    for _ in range(max(1, (n + B - 1) // B)):
        g = sampler.generate(mode=mode, task=None, num_samples=B, epochs=1)
        s = (g[0] if isinstance(g, (tuple, list)) else g)
        if s.dim() == 3:
            s = s.squeeze(0)
        s = s.to(device)
        with torch.no_grad():
            logits.append(model(s).cpu())
        seqs.append(s.cpu())
    return seqs, logits


# ---------------------------------------------------------------------------
# Plot cosmetics shared across all three optimal-orth modules
# ---------------------------------------------------------------------------

ORTH_COLORS = {"maj": "#2166ac", "ood": "#d6604d", "minor": "#1a9850"}
ORTH_BAR_WIDTH = 0.22
ORTH_GROUP_STEP = 0.24
ORTH_RANDOM_BAND_COLOR = "#b0b8c8"
ORTH_RANDOM_BAND_ALPHA = 0.35
ORTH_RANDOM_BAND_HATCH = "///"
ORTH_REFERENCE_LINE_COLOR = "#556070"


def orth_bar_offsets(has_minor: bool = True) -> dict:
    """Bar-center offsets for 2- or 3-bar layout."""
    g = ORTH_GROUP_STEP
    if has_minor:
        return {"maj": -g, "ood": 0.0, "minor": +g}
    return {"maj": -g / 2, "ood": +g / 2, "minor": 0.0}
