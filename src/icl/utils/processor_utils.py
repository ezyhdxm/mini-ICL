"""
Utility functions for linear processor operations.

This module contains common utility functions used across linear processing tasks,
including device setup, path management, caching, and data transformations.
"""

import os
import pickle
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, Union
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from scipy.stats import chi2
import torch.nn.functional as F

from icl.linear.lr_task import get_task
from icl.linear.linear_utils import compute_circumcenter


# =============================================================================
# Device and Environment Setup
# =============================================================================

def setup_device(device: Optional[str] = None) -> str:
    """Setup and return the appropriate device."""
    return device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")


def setup_experiment_paths(config: Union[Dict, Any], exp_name: str) -> str:
    """Setup experiment directory paths."""
    exp_dir = os.path.join(config.work_dir, exp_name)   
    cur_dir = os.getcwd()
    if cur_dir.endswith("notebooks"):
        exp_dir = os.path.join("..", exp_dir)
    return exp_dir


# =============================================================================
# File I/O and Caching
# =============================================================================

def create_result_path(exp_dir: str, orthogonal_offset: float, radius: float, 
                      is_on_sphere: bool, prefix: str = "probe_results") -> str:
    """Create result file path."""
    file_path = f'{prefix}_h_{orthogonal_offset}_r_{radius}_on_{is_on_sphere}.pkl'
    return os.path.join(exp_dir, file_path)


def load_cached_results(result_path: str, forced: bool = False) -> Optional[Dict]:
    """Load cached results if they exist and not forced to recompute."""
    if os.path.exists(result_path) and not forced:
        with open(result_path, 'rb') as f:
            return pickle.load(f)
    return None


def save_results(result_path: str, results_dict: Dict) -> None:
    """Save results to pickle file."""
    with open(result_path, 'wb') as f:
        pickle.dump(results_dict, f)


# =============================================================================
# Task Setup and Configuration  
# =============================================================================

def setup_eval_task(config: Union[Dict, Any], K: int, device: str, batch_size: int = 256):
    """Create and configure evaluation task."""
    eval_config = config.copy() if isinstance(config, dict) else config
    eval_config.task.n_tasks = K
    eval_config.device = device 
    
    eval_task = get_task(**eval_config["task"], device=device)
    eval_task.batch_size = batch_size
    return eval_task


# =============================================================================
# PCA and Projection Computations
# =============================================================================

def compute_pca_projections(anchor_pool: torch.Tensor, eval_task_pool: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PCA projections and related metrics."""
    anchor_np = anchor_pool.cpu().numpy()
    eval_np = eval_task_pool.cpu().numpy()
    
    pca = PCA(n_components=2)
    eval_2d = pca.fit_transform(eval_np)
    anchor_2d = eval_2d[:3]
    center_2d = compute_circumcenter(anchor_2d[0], anchor_2d[1], anchor_2d[2])
    
    directions = anchor_2d - center_2d[None, :]  # (3, 2)
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    eval_2d_directions = eval_2d - center_2d[None, :]  # (K+3, 2)
    eval_2d_directions = eval_2d_directions / np.linalg.norm(eval_2d_directions, axis=-1, keepdims=True)
    projections = eval_2d_directions @ directions.T  # (K+3, 3)
    assigned = projections.argmax(axis=-1)  # (K+3,)
    scores = np.eye(3)[assigned]  # (K+3, 3)
    
    return scores, projections


# =============================================================================
# Compositional Data Analysis Utilities
# =============================================================================

def replace_zeros_and_renorm(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    X: (..., 3) composition, each last-dim sums to 1 (approximately).
    Replace zeros with small eps, then renormalize to sum=1.
    """
    X = np.clip(X, eps, None)
    X = X / X.sum(axis=-1, keepdims=True)
    return X


# ILR Transform for 3-part compositions
_SQ12 = np.sqrt(1/2)
_SQ23 = np.sqrt(2/3)


def ilr3(x: np.ndarray) -> np.ndarray:
    """
    x: (..., 3) strictly positive, sum to 1 (compositions)
    return: (..., 2)
    """
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    z1 = _SQ12 * (np.log(x1) - np.log(x2))
    z2 = _SQ23 * (0.5 * (np.log(x1) + np.log(x2)) - np.log(x3))
    return np.stack([z1, z2], axis=-1)


def ilr3_inv(z: np.ndarray) -> np.ndarray:
    """
    z: (..., 2)
    return: (..., 3) composition on simplex (sum=1)
    Uses the canonical inverse via clr.
    """
    z1, z2 = z[..., 0], z[..., 1]
    # clr coords for 3 parts (orthonormal basis implied by ilr above)
    c1 = (1/np.sqrt(2)) * z1 + (1/np.sqrt(6)) * z2
    c2 = (-1/np.sqrt(2)) * z1 + (1/np.sqrt(6)) * z2
    c3 = (-2/np.sqrt(6)) * z2
    ex = np.stack([np.exp(c1), np.exp(c2), np.exp(c3)], axis=-1)
    return ex / ex.sum(axis=-1, keepdims=True)


def robust_loc_scatter_single_time(
    X_t: np.ndarray,
    support_fraction: float = 0.75,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    X_t: (K, 3) composition samples at one time t
    Returns:
      loc_ilr: (2,) robust location in ilr space
      cov_ilr: (2,2) robust covariance (scatter) in ilr space
      loc_comp: (3,) robust center mapped back to simplex (sum=1)
      d2: (K,) robust squared Mahalanobis distances in ilr space
    """
    X_t = replace_zeros_and_renorm(X_t)  # handle zeros -> eps, renorm
    Z = ilr3(X_t)                        # (K, 2)

    # Try MCD; if numerical warnings occur, you can increase support_fraction
    mcd = MinCovDet(support_fraction=support_fraction, random_state=random_state)
    mcd.fit(Z)

    loc_ilr = mcd.location_          # (2,)
    cov_ilr = mcd.covariance_        # (2,2)
    d2 = mcd.mahalanobis(Z)          # squared distances
    loc_comp = ilr3_inv(loc_ilr)     # (3,)

    return loc_ilr, cov_ilr, loc_comp, d2


def robust_compositional_timeseries(
    X: np.ndarray,
    support_fraction: float = 0.75,
    random_state: int = 0,
    compute_outlier_flags: bool = True,
    alpha: float = 0.975,
) -> Dict[str, Any]:
    """
    X: (K, T, 3) compositions (each vector sums to 1, may contain zeros)
    Returns a dict with:
      'loc_ilr':    (T, 2) robust location per time (ilr)
      'cov_ilr':    (T, 2, 2) robust covariance per time (ilr)
      'loc_comp':   (T, 3) robust center per time mapped back to simplex
      'd2':         (T, K) robust squared Mahalanobis distances per time
      'spread_trace': (T,) trace of cov_ilr (scalar spread)
      'spread_det':   (T,) determinant of cov_ilr
      'eigvals':      (T, 2) eigenvalues of cov_ilr (descending)
      'outlier_flags': (T, K) bool array if compute_outlier_flags=True
      'chi2_cut':      float, chi-square cutoff used (df=2, alpha)
    """
    assert X.ndim == 3 and X.shape[2] == 3, "X must be (K, T, 3)"
    K, T, _ = X.shape

    loc_ilr = np.zeros((T, 2))
    cov_ilr = np.zeros((T, 2, 2))
    loc_comp = np.zeros((T, 3))
    d2_all = np.zeros((T, K))

    for t in range(T):
        li, Si, lc, d2 = robust_loc_scatter_single_time(
            X[:, t, :],
            support_fraction=support_fraction,
            random_state=random_state,
        )
        loc_ilr[t] = li
        cov_ilr[t] = Si
        loc_comp[t] = lc
        d2_all[t] = d2

    # scalar spreads
    spread_trace = np.trace(cov_ilr, axis1=1, axis2=2)
    spread_det = np.linalg.det(cov_ilr)

    # eigenvalues (descending)
    eigvals = np.linalg.eigvalsh(cov_ilr)  # returns ascending for Hermitian
    eigvals = eigvals[:, ::-1]

    # outlier flags via chi-square cutoff in 2D ilr space
    if compute_outlier_flags:
        chi2_cut = float(chi2.ppf(alpha, df=2))
        outlier_flags = d2_all > chi2_cut
    else:
        chi2_cut = None
        outlier_flags = None

    return {
        "loc_ilr": loc_ilr,
        "cov_ilr": cov_ilr,
        "loc_comp": loc_comp,          # robust centers on simplex
        "d2": d2_all,
        "spread_trace": spread_trace, 
        "spread_det": spread_det,      
        "eigvals": eigvals,           
        "outlier_flags": outlier_flags,
        "chi2_cut": chi2_cut,
    }


# =============================================================================
# Similarity and Distance Metrics
# =============================================================================

def pairwise_cosine_similarity(X: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix."""
    X_norm = F.normalize(X, p=2, dim=1)  # Normalize each row to unit norm
    sim_matrix = X_norm @ X_norm.T       # Dot product between rows
    return sim_matrix


def robust_location_and_scatter(X: np.ndarray, support_fraction: float = 0.85) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: array of shape (K, 3), the samples at a single time point
    Returns: robust_location (3,), robust_cov (3x3)
    """
    mcd = MinCovDet(support_fraction=support_fraction, random_state=0).fit(X)
    return mcd.location_, mcd.covariance_