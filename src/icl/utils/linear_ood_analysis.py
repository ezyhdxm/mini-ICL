"""
Out-of-distribution enhancement and analysis functions.

This module contains functions for OOD task analysis, lambda projection,
and task vector injection experiments.
"""

import torch
import numpy as np
import os
import pickle
from typing import List, Tuple, Optional, Dict, Any, Sequence
import re
from tqdm.notebook import tqdm

from icl.linear.linear_utils import load_model_task_config, compute_hiddens, estimate_lambda_with_r2
from icl.linear.sampling import sample_points_from_balls
from icl.linear.lr_task import get_task
# from icl.utils.linear_algebra_utils import project_points_to_plane
# from icl.linear.linear_utils import compute_circumcenter
from icl.utils.processor_utils import setup_device
from icl.linear.linear_path_utils import get_checkpoint_files, load_checkpoint


def _to_tensor(x, device):
    """Convert input to torch tensor on specified device."""
    if torch.is_tensor(x):
        return x.to(device)
    # assume numpy or array-like
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def _setup_exp_dir(config, exp_name: Optional[str] = None):
    """Setup experiment directory path, accounting for notebook context."""
    exp_dir = os.path.join(config.work_dir, exp_name) if exp_name else config.work_dir
    cur_dir = os.getcwd()
    if cur_dir.endswith("notebooks"):
        exp_dir = os.path.join("..", exp_dir)
    
    return exp_dir


def _check_cache(result_path: str, forced: bool = False):
    """Check if cached results exist and load if available."""
    if (not forced) and os.path.exists(result_path):
        print(f"Already computed. Loading existing results from {result_path}.")
        with open(result_path, "rb") as f:
            return pickle.load(f)
    return None


def _save_results(result_path: str, results_dict: Dict[str, Any]):
    """Save results dictionary to pickle file."""
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"Saved results to {result_path}")


def _create_eval_task_pool(
    train_task,
    K: int,
    radius: float,
    include_minor: bool,
    device: str
) -> Tuple[torch.Tensor, int]:
    """
    Create evaluation task pool with OOD tasks and optional minority tasks.
    
    Returns:
        eval_task_pool: Combined task pool tensor
        n_minor_sampled: Number of minority tasks included
    """
    anchor_pool = train_task.task_pool.squeeze(-1).to(device)
    
    # Generate OOD evaluation task pool
    n_per_ball = K // 3
    eval_task_pool, _ = sample_points_from_balls(anchor_pool, r=radius, n_per_ball=n_per_ball)
    
    n_minor_sampled = 0
    if include_minor:
        minor_pool = train_task.minor_pool.squeeze(-1).to(device)
        
        # Sample minority tasks if too many
        if train_task.n_minor_tasks > 64:
            print(f"Too many minority tasks ({train_task.n_minor_tasks}). Randomly sampling 64.")
            torch.manual_seed(42)
            indices = torch.randperm(train_task.n_minor_tasks)[:64]
            minor_pool = minor_pool[indices]
            n_minor_sampled = 64
        else:
            n_minor_sampled = train_task.n_minor_tasks
        
        eval_task_pool = torch.cat([eval_task_pool, minor_pool], dim=0)
    
    return eval_task_pool, n_minor_sampled


def _setup_eval_task(config, eval_task_pool: torch.Tensor, batch_size: int, device: str):
    """Setup evaluation task with the given task pool."""
    K = eval_task_pool.shape[0]
    
    eval_config = config.copy() if isinstance(config, dict) else config
    eval_config["task"].n_tasks = K + 3
    eval_config["device"] = device
    
    eval_task = get_task(**eval_config["task"], device=device)
    eval_task.batch_size = batch_size
    eval_task.task_pool = eval_task_pool.unsqueeze(-1)  # (K, d, 1)
    
    return eval_task


def _compute_task_vectors(hiddens_all_time: torch.Tensor):
    """
    Compute task vectors from hidden states.
    
    Args:
        hiddens_all_time: Hidden states (K+3, T, B, d_emb)
        
    Returns:
        task_mean: Mean over anchor tasks (1, T, d_emb)
        task_vecs_over_all_time: Task vectors over time (K+3, T, d_emb)
        final_task_vecs: Final anchor task vectors (3, d_emb)
    """
    # Mean over first 3 anchor tasks
    task_mean = hiddens_all_time[:3].mean(dim=(0, 2)).unsqueeze(0)
    
    # Task vectors over time
    task_vecs_over_all_time = hiddens_all_time.mean(dim=-2) - task_mean
    
    # Final anchor vectors at last time step
    final_task_vecs = (hiddens_all_time[:3].mean(dim=-2) - task_mean)[:, -1]
    
    return task_mean, task_vecs_over_all_time, final_task_vecs


def _compute_ood_metrics(
    final_task_vecs: torch.Tensor,
    task_vecs_over_all_time: torch.Tensor,
    device: str
) -> Tuple[float, float]:
    """
    Compute OOD R² and lambda dispersion metrics.
    
    Args:
        final_task_vecs: Final anchor task vectors (3, d_emb)
        task_vecs_over_all_time: Task vectors over time (K+3, T, d_emb)
        device: Device for computation
        
    Returns:
        summary_r2: Mean OOD R² at final time
        dispersion: Lambda dispersion at final time
    """
    # Lambda + R^2
    lambdas, r2_scores, task_norms, ortho_norms = estimate_lambda_with_r2(
        final_task_vecs, task_vecs_over_all_time
    )
    
    lambdas = _to_tensor(lambdas, device)
    r2_scores = _to_tensor(r2_scores, device)
    
    # OOD part at final time
    r2_ood_final = r2_scores[3:, -1]      # (K_ood,)
    lambdas_ood_final = lambdas[3:, -1]   # (K_ood, n_basis)
    
    # Metric 1: mean OOD R^2 (final time)
    summary_r2 = float(r2_ood_final.mean())
    
    # Metric 2: lambda dispersion (final time)
    center = lambdas_ood_final.mean(dim=0, keepdim=True)
    dispersion = (lambdas_ood_final - center).norm(dim=-1).mean().item()
    
    return summary_r2, dispersion


def process_ood_evolve(exp_name: str, K: int = 300, layer_index: int = 3, 
                      orthogonal_offset: float = 0, is_on_sphere: bool = False, include_minor: bool = False,
                      radius: float = 2, device: Optional[str] = None,
                      batch_size: int = 256):
    """
    Process OOD evolution analysis with PCA projections and lambda estimation.
    
    Args:
        exp_name: Experiment name
        K: Number of evaluation tasks
        layer_index: Layer index for analysis
        orthogonal_offset: Orthogonal offset for sampling
        is_on_sphere: Whether to sample on sphere
        radius: Sampling radius
        device: Device to use
        batch_size: Batch size
        
    Returns:
        Results dictionary if cached, None otherwise
    """
    print("Preprocessing...")
    
    # Setup
    device = setup_device(device)
    model, train_task, config = load_model_task_config(exp_name)
    exp_dir = _setup_exp_dir(config, exp_name)

    # Check cache
    file_path = f'ood_results_h_{orthogonal_offset}_r_{radius}_on_{is_on_sphere}.pkl'
    result_path = os.path.join(exp_dir, file_path)
    cached = _check_cache(result_path)
    if cached is not None:
        return cached
    
    # Create evaluation task pool
    eval_task_pool, n_minor_sampled = _create_eval_task_pool(
        train_task, K, radius, include_minor, device
    )
    
    # Setup evaluation task
    eval_task = _setup_eval_task(config, eval_task_pool, batch_size, device)
    
    # Compute hiddens and task vectors
    hiddens_all_time, _ = compute_hiddens(config, model, eval_task, layer_index)
    _, task_vecs_over_all_time, final_task_vecs = _compute_task_vectors(hiddens_all_time)
    
    # Estimate lambdas and R²
    lambdas, r2_scores, task_norms, ortho_norms = estimate_lambda_with_r2(
        final_task_vecs, task_vecs_over_all_time
    )
    
    # Visualize if available
    try:
        from icl.utils.latent_task_vec import project_with_r2_size
        n_minors = n_minor_sampled if include_minor else 0
        fig = project_with_r2_size(task_vecs_over_all_time, final_task_vecs, r2_scores, lambdas, n_minors=n_minors)
        fig.show()
    except ImportError:
        print("Visualization function not available")
        
    return None  # No caching in this simplified version





def process_ood_evolve_checkpoints(
    exp_name: str,
    K: int = 300,
    layer_index: int = 12,
    layer_indices: Optional[Sequence[int]] = list(range(0,16)),
    orthogonal_offset: float = 0.0,
    is_on_sphere: bool = False,
    include_minor: bool = False,
    radius: float = 2.0,
    device: Optional[str] = None,
    batch_size: int = 256,
    skip_factor: int = 10,
    max_checkpoints: Optional[int] = None,
    forced: bool = False,
) -> Dict[str, Any]:
    """
    OOD evolution analysis over checkpoints for multiple layers.

    For each checkpoint and each layer in `layer_indices`, we compute:
      - summary OOD R^2 at the final time step
      - lambda dispersion at the final time step (mean distance to centroid)

    Returns:
        results_dict with keys:
            - 'steps': list[int]
            - 'layers': list[int]
            - 'summary_r2_ood': dict[layer][step] -> float
            - 'lambda_dispersion_ood': dict[layer][step] -> float
    """
    print("Preprocessing...")

    # Setup
    device = setup_device(device)
    base_model, train_task, config = load_model_task_config(exp_name)
    layer_indices = [layer_index] if layer_indices is None else list(layer_indices)

    # Setup paths and cache
    exp_dir = _setup_exp_dir(config, exp_name)
    base_file_name = f'ood_evolve_ckpt_all_layers_h_{orthogonal_offset}_r_{radius}_on_{is_on_sphere}.pkl'
    result_path = os.path.join(exp_dir, base_file_name)
    
    cached = _check_cache(result_path, forced)
    if cached is not None:
        return cached

    # Create evaluation task pool
    eval_task_pool, n_minor_sampled = _create_eval_task_pool(
        train_task, K, radius, include_minor, device
    )
    eval_task = _setup_eval_task(config, eval_task_pool, batch_size, device)

    # Get checkpoints to process
    checkpoint_files = get_checkpoint_files(exp_name)
    checkpoint_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    if max_checkpoints:
        checkpoint_files = checkpoint_files[:max_checkpoints]
        print(f"Limited to {max_checkpoints} checkpoints for testing.")

    print(f"Found {len(checkpoint_files)} checkpoint files.")

    steps_to_process = []
    checkpoint_indices = []
    for k in range(0, len(checkpoint_files), 8):
        curr_step = int(re.search(r'\d+', checkpoint_files[k]).group())
        steps_to_process.append(curr_step)
        checkpoint_indices.append(k)

    print(f"Will process {len(steps_to_process)} checkpoints (skipping {len(checkpoint_files) - len(steps_to_process)}).")

    # Storage
    summary_r2_ood: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    lambda_dispersion_ood: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    processed_steps = []

    try:
        for step, k in tqdm(zip(steps_to_process, checkpoint_indices), total=len(steps_to_process)):
            model, ckpt_config, _ = load_checkpoint(exp_name, checkpoint_files[k])
            model = model.to(device)
            model.eval()

            try:
                with torch.no_grad():
                    for L in layer_indices:
                        hiddens_all_time, _ = compute_hiddens(ckpt_config, model, eval_task, layer_index=L)
                        _, task_vecs_over_all_time, final_task_vecs = _compute_task_vectors(hiddens_all_time)
                        summary_r2, dispersion = _compute_ood_metrics(
                            final_task_vecs, task_vecs_over_all_time, device
                        )
                        
                        summary_r2_ood[L][step] = summary_r2
                        lambda_dispersion_ood[L][step] = dispersion

                    if step not in processed_steps:
                        processed_steps.append(step)

            except RuntimeError as e:
                print(f"Error processing checkpoint {checkpoint_files[k]} at step {step}: {e}")
                continue

    except KeyboardInterrupt:
        print(f"\nInterrupted. Processed {len(processed_steps)} checkpoints so far.")

    if not processed_steps:
        print("No checkpoints processed successfully.")
        return {}

    results_dict = {
        "steps": processed_steps,
        "layers": layer_indices,
        "summary_r2_ood": summary_r2_ood,
        "lambda_dispersion_ood": lambda_dispersion_ood,
        "include_minor": include_minor,
        "n_minor_sampled": n_minor_sampled,
        "radius": radius,
    }

    _save_results(result_path, results_dict)
    return results_dict


def process_ood_evolve_task_diversity(
    exp_names: List[str],
    steps: List[int],
    K: int = 100,
    layer_indices: Optional[Sequence[int]] = None,
    orthogonal_offset: float = 0.0,
    is_on_sphere: bool = False,
    include_minor: bool = False,
    radius: float = 2.0,
    device: Optional[str] = None,
    batch_size: int = 256,
    forced: bool = False,
) -> Dict[str, Any]:
    """
    OOD evolution analysis across multiple experiments with different task diversities.
    
    For each experiment and each layer in `layer_indices`, we compute:
      - summary OOD R^2 at the final time step
      - lambda dispersion at the final time step (mean distance to centroid)
    
    Args:
        exp_names: List of experiment names to process
        K: Number of OOD evaluation tasks
        layer_indices: List of layer indices to analyze (default: all layers 0-15)
        orthogonal_offset: Orthogonal offset for sampling
        is_on_sphere: Whether to sample on sphere
        include_minor: Whether to include minority tasks
        radius: Sampling radius for OOD tasks
        device: Device to use
        batch_size: Batch size for evaluation
        forced: Force recomputation even if cached results exist
    
    Returns:
        results_dict with keys:
            - 'exp_names': list[str]
            - 'layers': list[int]
            - 'summary_r2_ood': dict[layer][exp_name] -> float
            - 'lambda_dispersion_ood': dict[layer][exp_name] -> float
    """
    print(f"Processing {len(exp_names)} experiments for task diversity analysis...")

    # Setup
    device = setup_device(device)
    _, _, base_config = load_model_task_config(exp_names[0])
    layer_indices = list(range(0, 16)) if layer_indices is None else list(layer_indices)

    # Setup paths and cache
    exp_dir = os.path.join(base_config.work_dir, "task_diversity_analysis")
    cur_dir = os.getcwd()
    if cur_dir.endswith("notebooks"):
        exp_dir = os.path.join("..", exp_dir)
    exp_names_hash = hash(tuple(sorted(exp_names))) & 0xFFFFFFFF
    base_file_name = (
        f'task_diversity_h_{orthogonal_offset}_r_{radius}_on_{is_on_sphere}_n_{len(exp_names)}_hash_{exp_names_hash:08x}.pkl'
    )
    result_path = os.path.join(exp_dir, base_file_name)
    cached = _check_cache(result_path, forced)
    if cached is not None:
        return cached

    # Storage
    summary_r2_ood: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    lambda_dispersion_ood: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    processed_experiments = []
    processed_steps = []

    try:
        for i, exp_name in enumerate(
            tqdm(exp_names, desc="Processing experiments")
        ):
            print(f"\nProcessing experiment: {exp_name}")
            exp_step = steps[i]
            
            try:
                # Load model and configuration
                model, train_task, config = load_model_task_config(exp_name)
                model = model.to(device)
                model.eval()

                # Create evaluation task pool
                eval_task_pool, n_minor_sampled = _create_eval_task_pool(
                    train_task, K, radius, include_minor, device
                )
                eval_task = _setup_eval_task(config, eval_task_pool, batch_size, device)

                with torch.no_grad():
                    for L in layer_indices:
                        hiddens_all_time, _ = compute_hiddens(config, model, eval_task, layer_index=L)
                        _, task_vecs_over_all_time, final_task_vecs = _compute_task_vectors(hiddens_all_time)
                        summary_r2, dispersion = _compute_ood_metrics(
                            final_task_vecs, task_vecs_over_all_time, device
                        )

                        summary_r2_ood[L][exp_step] = summary_r2
                        lambda_dispersion_ood[L][exp_step] = dispersion

                processed_experiments.append(exp_name)
                processed_steps.append(exp_step)
                print(f"✓ Completed {exp_name}")

            except Exception as e:
                print(f"Error processing experiment {exp_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    except KeyboardInterrupt:
        print(f"\nInterrupted. Processed {len(processed_experiments)} experiments so far.")

    if not processed_experiments:
        print("No experiments processed successfully.")
        return {}

    results_dict = {
        "exp_names": processed_experiments,
        "steps": processed_steps,
        "layers": layer_indices,
        "summary_r2_ood": summary_r2_ood,
        "lambda_dispersion_ood": lambda_dispersion_ood,
        "include_minor": include_minor,
        "radius": radius,
        "K": K,
    }

    _save_results(result_path, results_dict)
    return results_dict