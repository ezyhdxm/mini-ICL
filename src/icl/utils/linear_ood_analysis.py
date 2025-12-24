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
from icl.utils.linear_algebra_utils import project_points_to_plane
from icl.linear.linear_utils import compute_circumcenter, get_checkpoint_files, load_checkpoint
from icl.utils.processor_utils import setup_device





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
    
    # Device setup
    device = setup_device(device)
    
    # Load configuration and setup
    model, train_task, config = load_model_task_config(exp_name)
    
    # Setup paths
    exp_dir = os.path.join(config.work_dir, exp_name)   
    cur_dir = os.getcwd()
    if cur_dir.endswith("notebooks"):
        exp_dir = os.path.join("..", exp_dir)

    # Check for cached results
    file_path = f'ood_results_h_{orthogonal_offset}_r_{radius}_on_{is_on_sphere}.pkl'
    result_path = os.path.join(exp_dir, file_path)
    if os.path.exists(result_path):
        print("Already computed. Loading existing results.")
        with open(result_path, 'rb') as f:
            results_dict = pickle.load(f)
        return results_dict
    
    anchor_pool = train_task.task_pool.squeeze(-1).to(device)
    
    # Generate evaluation task pool
    n_per_ball = K // 3
    K = n_per_ball * 3
    eval_task_pool, weights = sample_points_from_balls(
        anchor_pool, r=radius, n_per_ball=n_per_ball
    )

    # Initialize n_minor_sampled
    n_minor_sampled = 0
    
    if include_minor:
        minor_pool = train_task.minor_pool.squeeze(-1).to(device)
        
        # If there are too many minority tasks, randomly sample 64
        if train_task.n_minor_tasks > 64:
            print(f"Too many minority tasks ({train_task.n_minor_tasks}). Randomly sampling 64.")
            # Set random seed for reproducibility
            torch.manual_seed(42)
            indices = torch.randperm(train_task.n_minor_tasks)[:64]
            minor_pool = minor_pool[indices]
            n_minor_sampled = 64
        else:
            n_minor_sampled = train_task.n_minor_tasks
            
        eval_task_pool = torch.cat([eval_task_pool, minor_pool], dim=0)
        K += n_minor_sampled
    
    # Setup evaluation task
    eval_config = config.copy() if isinstance(config, dict) else config
    eval_config.task.n_tasks = K + 3
    eval_config.device = device 
    
    eval_task = get_task(**eval_config["task"], device=device)
    eval_task.batch_size = batch_size
    eval_task.task_pool = eval_task_pool.unsqueeze(-1)  # shape (K+3, d, 1)
    
    hiddens_all_time, _ = compute_hiddens(config, model, eval_task, layer_index) # (K+3, T, B, d_emb)
    
    task_mean = hiddens_all_time[:3].mean(dim=(0,2)).unsqueeze(0) # (1, T, d_emb)
    task_vecs_over_all_time = hiddens_all_time.mean(dim=-2) - task_mean # (K+3, T, d_emb)
    final_task_vecs = hiddens_all_time[:3].mean(dim=-2) - task_mean 
    final_task_vecs = final_task_vecs[:, -1] # (3, d_emb)
    lambdas, r2_scores, task_norms, ortho_norms = estimate_lambda_with_r2(
        final_task_vecs, task_vecs_over_all_time
    )
    
    # Import and use visualization function
    try:
        from icl.utils.latent_task_vec import project_with_r2_size
        if include_minor:
            n_minors = n_minor_sampled
        else:
            n_minors = 0
        fig = project_with_r2_size(task_vecs_over_all_time, final_task_vecs, r2_scores, lambdas, n_minors=n_minors)
        fig.show()
    except ImportError:
        print("Visualization function not available")
        
    return None  # No caching in this simplified version



def _to_tensor(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    # assume numpy or array-like
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def process_ood_evolve_checkpoints(
    exp_name: str,
    K: int = 300,
    layer_index: int = 12,                # kept for backward compatibility
    layer_indices: Optional[Sequence[int]] = list(range(0,16)),  # NEW: list of layers to collect
    orthogonal_offset: float = 0.0,   # kept for symmetry with process_ood_evolve
    is_on_sphere: bool = False,       # kept for symmetry
    include_minor: bool = False,
    radius: float = 2.0,
    device: Optional[str] = None,
    batch_size: int = 256,
    skip_factor: int = 10,            # currently unused in your for-loop, kept for API
    max_checkpoints: Optional[int] = None,
    forced: bool = False,
) -> Dict[str, Any]:
    """
    OOD evolution analysis *over checkpoints*, now collecting metrics for
    multiple layers.

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

    # ---------------- Device & base config ----------------
    device = setup_device(device)
    base_model, train_task, config = load_model_task_config(exp_name)

    # Decide which layers to process
    if layer_indices is None:
        layer_indices = [layer_index]
    layer_indices = list(layer_indices)

    # ---------------- Paths & cache ----------------
    exp_dir = os.path.join(config.work_dir, exp_name)
    cur_dir = os.getcwd()
    if cur_dir.endswith("notebooks"):
        exp_dir = os.path.join("..", exp_dir)

    base_file_name = (
        f'ood_evolve_ckpt_all_layers_h_{orthogonal_offset}_r_{radius}_on_{is_on_sphere}.pkl'
    )
    result_path = os.path.join(exp_dir, base_file_name)

    if (not forced) and os.path.exists(result_path):
        print(f"Already computed. Loading existing results from {result_path}.")
        with open(result_path, "rb") as f:
            return pickle.load(f)

    # ---------------- Evaluation task pool (same as process_ood_evolve) ----------------
    anchor_pool = train_task.task_pool.squeeze(-1).to(device)  # (n_anchor, d)

    n_per_ball = K // 3
    K = n_per_ball * 3
    eval_task_pool, weights = sample_points_from_balls(
        anchor_pool, r=radius, n_per_ball=n_per_ball
    )

    n_minor_sampled = 0
    if include_minor:
        minor_pool = train_task.minor_pool.squeeze(-1).to(device)
        if train_task.n_minor_tasks > 64:
            print(f"Too many minority tasks ({train_task.n_minor_tasks}). Randomly sampling 64.")
            torch.manual_seed(42)
            indices = torch.randperm(train_task.n_minor_tasks)[:64]
            minor_pool = minor_pool[indices]
            n_minor_sampled = 64
        else:
            n_minor_sampled = train_task.n_minor_tasks

        eval_task_pool = torch.cat([eval_task_pool, minor_pool], dim=0)
        K += n_minor_sampled

    eval_config = config.copy() if isinstance(config, dict) else config
    eval_config["task"].n_tasks = K + 3
    eval_config["device"] = device

    eval_task = get_task(**eval_config["task"], device=device)
    eval_task.batch_size = batch_size
    eval_task.task_pool = eval_task_pool.unsqueeze(-1)  # (K, d, 1)

    # ---------------- Checkpoints to process ----------------
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

    num_checkpoints = len(steps_to_process)
    print(
        f"Will process {num_checkpoints} checkpoints "
        f"(skipping {len(checkpoint_files) - num_checkpoints})."
    )

    # ---------------- Storage: nested dicts [layer][step] -> metric ----------------
    summary_r2_ood: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    lambda_dispersion_ood: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}

    processed_steps = []

    try:
        for step, k in tqdm(zip(steps_to_process, checkpoint_indices), total=num_checkpoints):
            model, ckpt_config, _ = load_checkpoint(exp_name, checkpoint_files[k])
            model = model.to(device)
            model.eval()

            try:
                with torch.no_grad():
                    # For each layer, compute metrics
                    for L in layer_indices:
                        # ----- Compute hiddens for this layer -----
                        # hiddens_all_time: (K+3, T, B, d_emb)
                        hiddens_all_time, _ = compute_hiddens(
                            ckpt_config, model, eval_task, layer_index=L
                        )

                        # Mean over first 3 anchor tasks
                        task_mean = hiddens_all_time[:3].mean(dim=(0, 2)).unsqueeze(0)

                        # Task vectors over time
                        task_vecs_over_all_time = (
                            hiddens_all_time.mean(dim=-2) - task_mean
                        )  # (K+3, T, d_emb)

                        # Final anchor vectors at last time step
                        final_task_vecs = (
                            hiddens_all_time[:3].mean(dim=-2) - task_mean
                        )[:, -1]  # (3, d_emb)

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
                        dispersion = (
                            (lambdas_ood_final - center).norm(dim=-1).mean().item()
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
        "summary_r2_ood": summary_r2_ood,             # dict[layer][step] -> float
        "lambda_dispersion_ood": lambda_dispersion_ood,  # dict[layer][step] -> float
        "include_minor": include_minor,
        "n_minor_sampled": n_minor_sampled,
        "radius": radius,
    }

    os.makedirs(exp_dir, exist_ok=True)
    with open(result_path, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"Saved results to {result_path}")

    return results_dict
