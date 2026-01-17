"""
OOD evolution analysis utilities for latent models.
Similar to linear_ood_analysis but adapted for latent model architectures.
"""

import os
import pickle
import torch
from typing import Dict, Sequence, Any, Optional
from tqdm.notebook import tqdm
import copy

import icl.utils.notebook_utils as nu
from icl.linear.linear_utils import estimate_lambda_with_r2
from icl.utils.simple_markov_sampler import get_all_samples_base_only


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
    device: str = "cuda"
):
    """
    Compute OOD R² and lambda dispersion metrics for latent model.
    Similar to plot_hidden_proj but returns metrics instead of plotting.
    
    Args:
        hiddens_voc: Hidden representations (K, V, T, B, D)
        k_minor: Number of minor (in-distribution) tasks
        device: Device for computation
        
    Returns:
        summary_r2: Mean OOD R² at final time
        lambda_dispersion: Lambda dispersion at final time (mean distance to centroid)
    """
    
    hiddens_voc = hiddens_voc.to(torch.float32)
    K, _, T, B, _ = hiddens_voc.shape
    
    # Concatenate all hidden representations for different vocabularies
    hiddens = hiddens_voc.permute(0, 2, 3, 4, 1).reshape(K, T, B, -1)  # (K, T, B, D*V)
    
    # Compute global mean over first 3 (anchor) tasks
    global_mean = hiddens[:3].mean(dim=(0, 2))  # (T, D*V)
    
    # Task vectors over all time
    task_vecs_over_all_time = hiddens.mean(dim=2) - global_mean.unsqueeze(dim=0)  # (K, T, D*V)
    
    # Final anchor task vectors
    final_task_vecs = task_vecs_over_all_time[:3, -1]
    
    # Estimate lambdas and R²
    lambdas, r2_scores, task_norms, ortho_norms = estimate_lambda_with_r2(
        final_task_vecs, task_vecs_over_all_time
    )
    
    # Convert to tensors
    lambdas = torch.as_tensor(lambdas, device=device, dtype=torch.float32)
    r2_scores = torch.as_tensor(r2_scores, device=device, dtype=torch.float32)
    
    # OOD slice: tasks after the first 3 anchors AND after k_minor in-distribution tasks
    # Total OOD tasks start from index 3 + k_minor
    ood_start_idx = 3 + k_minor
    
    # OOD R² at final time
    r2_ood_final = r2_scores[ood_start_idx:, -1]  # (K_ood,)
    
    # Metric 1: mean OOD R² at final time
    summary_r2 = float(r2_ood_final.mean())
    
    # OOD lambdas at final time
    lambdas_ood_final = lambdas[ood_start_idx:, -1]  # (K_ood, n_basis)
    
    # Metric 2: lambda dispersion (mean distance to centroid)
    center = lambdas_ood_final.mean(dim=0, keepdim=True)  # (1, n_basis)
    distances = (lambdas_ood_final - center).norm(dim=-1)  # (K_ood,)
    lambda_dispersion = float(distances.mean())
    
    return summary_r2, lambda_dispersion


def process_latent_ood_evolve_checkpoints(
    exp_group: str,
    exp_name: str,
    steps: Sequence[int],
    all_samples,
    k_minor: int,
    layer_indices: Sequence[int] = [2, 3, 4, 5],
    device: str = "cuda",
    k_step: int = 32,
    b_step: int = 32,
    t_step: int = 4,
    forced: bool = False,
) -> Dict[str, Any]:
    """
    Process all checkpoints for a latent experiment and compute OOD metrics across multiple layers.
    Similar to process_ood_evolve_checkpoints from linear_ood_analysis but for latent models.
    
    Args:
        exp_group: e.g. "latent"
        exp_name: e.g. "train_..."
        steps: list of checkpoint steps to process
        all_samples: pre-generated samples (K, T, B)
        k_minor: number of minor (in-distribution) tasks
        layer_indices: which layers to analyze
        device: device string
        k_step, b_step, t_step: parameters for compute_hiddens_onepos_all_layers_kvcache_beta
        forced: if True, recompute even if cached results exist
        
    Returns:
        {
            'steps': list[int],
            'layers': list[int],
            'summary_r2_ood': dict[layer][step] -> float,
            'lambda_dispersion_ood': dict[layer][step] -> float,
        }
    """
    from icl.utils.kv_latent_task_vec_beta import compute_hiddens_onepos_all_layers_kvcache_beta
    import icl.utils.notebook_utils as nu
    
    # Load sampler + config once
    _, sampler, config = nu.load_everything(exp_group, exp_name)
    
    # Check for cached results
    cache_path = _get_cache_path(config, exp_name, k_minor, layer_indices)
    cached_results = _load_cached_results(cache_path, forced)
    if cached_results is not None:
        return cached_results
    
    print("Computing results from scratch...")
    
    # Storage for results
    summary_r2_ood: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    lambda_dispersion_ood: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    processed_steps = []
        
    for step in tqdm(steps, desc="Processing checkpoints"):
        try:
            # Load model at this checkpoint
            model = nu.load_checkpoint(config, step=step)
            model = model.to(device)
            model.eval()
            
            with torch.no_grad():
                # Compute hiddens for all layers at once
                hiddens = compute_hiddens_onepos_all_layers_kvcache_beta(
                    config, model, all_samples,
                    k_step=k_step,
                    b_step=b_step,
                    t_step=t_step
                )
                
                # Permute to get (L, K, V, T, B, D)
                hiddens_voc = hiddens.permute(0, 1, 3, 2, 4, 5)
                
                # Process each layer
                for L in layer_indices:
                    hiddens_layer = hiddens_voc[L]  # (K, V, T, B, D)
                    
                    # Compute metrics
                    summary_r2, lambda_dispersion = compute_latent_ood_metrics(
                        hiddens_layer, k_minor=k_minor, device=device
                    )
                    
                    summary_r2_ood[L][step] = summary_r2
                    lambda_dispersion_ood[L][step] = lambda_dispersion
            
            processed_steps.append(step)
            
        except Exception as e:
            print(f"Error processing step {step}: {e}")
            continue
    
    if not processed_steps:
        print("No checkpoints processed successfully.")
        return {}
    
    results_dict = {
        "steps": processed_steps,
        "layers": layer_indices,
        "summary_r2_ood": summary_r2_ood,
        "lambda_dispersion_ood": lambda_dispersion_ood,
        "k_minor": k_minor,
    }
    
    # Save results to cache
    _save_results(cache_path, results_dict)
    
    return results_dict


def _check_cache(result_path: str, forced: bool = False):
    """Check if cached results exist and load if available."""
    if (not forced) and os.path.exists(result_path):
        print(f"Already computed. Loading existing results from {result_path}.")
        with open(result_path, "rb") as f:
            return pickle.load(f)
    return None

def get_all_samples(exp_name, n_minor=256, n_ood=40, B=96):
    model, sampler, config = nu.load_everything("latent", exp_name)
    sampler_clone0 = copy.deepcopy(sampler)

    # Original minor count in the clone (capacity before expansion)
    k_minor = min(n_minor, sampler_clone0.n_minor_tasks)
    n_tasks = k_minor + n_ood
    sampler_clone0.n_minor_tasks = n_tasks

    orig = sampler_clone0.minor_trans_mat

    # Sample new OOD matrices (match device/dtype/shape)
    ood = sampler_clone0._sample_banded_trans_mats(n_ood)
    # Make sure ood is on same device/dtype as orig if needed
    ood = ood.to(device=orig.device, dtype=orig.dtype)

    # Create expanded matrix: (n_tasks, ...)
    new_shape = (n_tasks, *orig.shape[1:])
    new_minor = orig.new_empty(new_shape)  # same dtype/device as orig

    # Fill: first k_ood are new; remaining are old shifted later
    new_minor[:n_ood].copy_(ood)
    new_minor[n_ood:].copy_(orig[:k_minor])

    # Swap in
    sampler_clone0.minor_trans_mat = new_minor
    all_samples = get_all_samples_base_only(n_tasks, sampler_clone0, B)

    return all_samples, k_minor

def process_latent_ood_evolve_task_diversity(
    exp_names: Sequence[str],
    steps: Sequence[int],
    n_minor: int = 256,
    layer_indices: Sequence[int] = list(range(6)),
    device: str = "cuda",
    k_step: int = 32,
    b_step: int = 32,
    t_step: int = 4,
    forced: bool = False,
):
    from icl.utils.kv_latent_task_vec_beta import compute_hiddens_onepos_all_layers_kvcache_beta
    import icl.utils.notebook_utils as nu

    print(f"Processing {len(exp_names)} experiments for task diversity analysis...")
    _, _, base_config = nu.load_everything("latent", exp_names[0])
    layer_indices = list(range(6)) if layer_indices is None else layer_indices
    
    # Check for cached results
    exp_dir = os.path.join(base_config.work_dir, "task_diversity_analysis")
    cur_dir = os.getcwd()
    if cur_dir.endswith("notebooks"):
        exp_dir = os.path.join("..", exp_dir)
    exp_names_hash = hash(tuple(sorted(exp_names))) & 0xFFFFFFFF
    
    base_file_name = (
        f'task_diversity_{len(exp_names)}_hash_{exp_names_hash:08x}.pkl'
    )
    result_path = os.path.join(exp_dir, base_file_name)
    cached = _check_cache(result_path, forced)
    if cached is not None:
        return cached
    
    print("Computing results from scratch...")
    
    # Storage for results
    summary_r2_ood: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    lambda_dispersion_ood: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    processed_experiments = []
    processed_steps = []

    try:
        for i, exp_name in enumerate(tqdm(exp_names, desc="Processing experiments")):
            # Load sampler + config once
            print(f"\nProcessing experiment: {exp_name}")
            exp_step = 60000
            model, sampler, config = nu.load_everything("latent", exp_name)
            all_samples, k_minor = get_all_samples(
                exp_name=exp_name,
                n_minor=n_minor,
                n_ood=40,
                B=96
            )
        
            try:
                # Load model at this checkpoint
                model = nu.load_checkpoint(config, step=exp_step)
                model = model.to(device)
                model.eval()
                
                with torch.no_grad():
                    # Compute hiddens for all layers at once
                    hiddens = compute_hiddens_onepos_all_layers_kvcache_beta(
                        config, model, all_samples,
                        k_step=k_step,
                        b_step=b_step,
                        t_step=t_step
                    )
                    
                    # Permute to get (L, K, V, T, B, D)
                    hiddens_voc = hiddens.permute(0, 1, 3, 2, 4, 5)
                    
                    # Process each layer
                    for L in layer_indices:
                        hiddens_layer = hiddens_voc[L]  # (K, V, T, B, D)
                        
                        # Compute metrics
                        summary_r2, lambda_dispersion = compute_latent_ood_metrics(
                            hiddens_layer, k_minor=k_minor, device=device
                        )
                        
                        summary_r2_ood[L][steps[i]] = summary_r2
                        lambda_dispersion_ood[L][steps[i]] = lambda_dispersion
                
                processed_steps.append(steps[i])
                
            except Exception as e:
                print(f"Error processing step {steps[i]}: {e}")
                continue
        
    except KeyboardInterrupt:
        print(f"\nInterrupted. Processed {len(processed_experiments)} experiments so far.")
    
    if not processed_steps:
        print("No checkpoints processed successfully.")
        return {}
    
    results_dict = {
        "steps": processed_steps,
        "layers": layer_indices,
        "summary_r2_ood": summary_r2_ood,
        "lambda_dispersion_ood": lambda_dispersion_ood,
    }
    
    # Save results to cache
    _save_results(result_path, results_dict)
    
    return results_dict
    