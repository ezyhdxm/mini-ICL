"""
OOD evolution analysis utilities for latent models.

Combines low-level OOD metric functions (originally from latent_ood_analysis)
and higher-level OOD metric processing and plotting (originally from
latent_nonpadded).
"""

import copy
import gc
import os
import pickle

import numpy as np
import torch
from typing import Any, Dict, Optional, Sequence, Union
from tqdm.notebook import tqdm

import icl.utils.notebook_utils as nu
from icl.linear.linear_utils import estimate_lambda_with_r2
from icl.utils.device_utils import get_default_device
from icl.utils.linear_algebra_utils import stable_rank
from icl.utils.logger import setup_logger
get_all_samples_base_only = None

logger = setup_logger(__name__)


# ===========================================================================
# Low-level OOD metric functions (from latent_ood_analysis)
# ===========================================================================


def _get_exp_dir(config, exp_name: str) -> str:
    """Get the experiment directory path, accounting for notebook context."""
    exp_dir = os.path.join(config.work_dir, exp_name)
    cur_dir = os.getcwd()
    if cur_dir.endswith("notebooks"):
        exp_dir = os.path.join("..", exp_dir)
    return exp_dir


def _get_cache_path(config, exp_name: str, k_minor: int, layer_indices: Sequence[int]) -> str:
    """Generate cache file path based on experiment parameters."""
    exp_dir = _get_exp_dir(config, exp_name)
    layers_str = "_".join(map(str, layer_indices))
    cache_file = f"latent_ood_evolve_ckpt_kminor_{k_minor}_layers_{layers_str}.pkl"
    return os.path.join(exp_dir, cache_file)


def _load_cached_results(cache_path: str, forced: bool = False) -> Optional[Dict[str, Any]]:
    """Load cached results if they exist and forced is False."""
    if forced:
        return None
    
    if os.path.exists(cache_path):
        print(f"Loading cached results from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    
    return None


def _save_results(cache_path: str, results_dict: Dict[str, Any]):
    """Save results to cache file."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"Results saved to {cache_path}")


def compute_latent_ood_metrics(
    hiddens_voc,
    k_minor: int,
    device: Optional[str] = None,
):
    """
    Compute OOD R² and lambda dispersion metrics for latent model.
    Similar to plot_hidden_proj but returns metrics instead of plotting.
    
    Args:
        hiddens_voc: Hidden representations (K, V, T, B, D)
        k_minor: Number of minor (in-distribution) tasks
        device: Device for computation (default: from get_default_device())
        
    Returns:
        summary_r2: Mean OOD R² at final time
        lambda_dispersion: Lambda dispersion at final time (mean distance to centroid)
    """
    if device is None:
        device = get_default_device()
    hiddens_voc = hiddens_voc.to(torch.float32)
    K, _, T, B, _ = hiddens_voc.shape
    
    hiddens = hiddens_voc.permute(0, 2, 3, 4, 1).reshape(K, T, B, -1)  # (K, T, B, D*V)
    
    global_mean = hiddens[:3].mean(dim=(0, 2))  # (T, D*V)
    
    task_vecs_over_all_time = hiddens.mean(dim=2) - global_mean.unsqueeze(dim=0)  # (K, T, D*V)
    
    final_task_vecs = task_vecs_over_all_time[:3, -1]
    
    lambdas, r2_scores, task_norms, ortho_norms = estimate_lambda_with_r2(
        final_task_vecs, task_vecs_over_all_time
    )
    
    lambdas = torch.as_tensor(lambdas, device=device, dtype=torch.float32)
    r2_scores = torch.as_tensor(r2_scores, device=device, dtype=torch.float32)
    
    ood_start_idx = 3 + k_minor
    
    r2_ood_final = r2_scores[ood_start_idx:, -1]  # (K_ood,)
    
    summary_r2 = float(r2_ood_final.mean())
    
    lambdas_ood_final = lambdas[ood_start_idx:, -1]  # (K_ood, n_basis)
    
    center = lambdas_ood_final.mean(dim=0, keepdim=True)  # (1, n_basis)
    distances = (lambdas_ood_final - center).norm(dim=-1)  # (K_ood,)
    lambda_dispersion = float(distances.mean())
    
    return summary_r2, lambda_dispersion


def get_latent_sampler(exp_name, n_minor=256, n_ood=40, sampler=None):
    """Build an OOD+minor sampler clone for the latent task.

    Parameters
    ----------
    sampler : optional
        If provided, use this already-loaded sampler instead of calling
        ``nu.load_everything`` (which would waste a GPU allocation).
    """
    if sampler is None:
        _, sampler, _ = nu.load_everything("latent", exp_name)
    sampler_clone0 = copy.deepcopy(sampler)

    if n_minor == -1:
        n_minor = 0
    
    k_minor = min(n_minor, sampler_clone0.n_minor_tasks) if n_minor >= 0 else 0
    n_tasks = k_minor + n_ood
    sampler_clone0.n_minor_tasks = n_tasks

    orig = sampler_clone0.minor_trans_mat

    if n_tasks == 0:
        new_shape = (0, *orig.shape[1:])
        new_minor = orig.new_empty(new_shape)
    else:
        if n_ood > 0:
            # Dirichlet(1,...,1) for each row of each transition matrix via the Exp(1) trick
            ood = orig.new_empty((n_ood, *orig.shape[1:])).exponential_()
            ood = ood / ood.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            ood = orig.new_empty((0, *orig.shape[1:]))

        new_shape = (n_tasks, *orig.shape[1:])
        new_minor = orig.new_empty(new_shape)

        if n_ood > 0:
            new_minor[:n_ood].copy_(ood)
        if k_minor > 0:
            new_minor[n_ood:].copy_(orig[:k_minor])

    sampler_clone0.minor_trans_mat = new_minor
    
    return sampler_clone0, k_minor, n_tasks


def get_all_samples(exp_name, n_minor=256, n_ood=40, B=96, sampler=None):
    """Generate all samples for latent task evaluation.

    Parameters
    ----------
    sampler : optional
        If provided, pass directly to :func:`get_latent_sampler` to avoid a
        redundant ``nu.load_everything`` call.
    """
    sampler_clone0, k_minor, n_tasks = get_latent_sampler(exp_name, n_minor, n_ood, sampler=sampler)
    all_samples = get_all_samples_base_only(n_tasks, sampler_clone0, B)

    return all_samples, k_minor
