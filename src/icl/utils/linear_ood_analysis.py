"""
Out-of-distribution enhancement and analysis functions.

This module contains functions for OOD task analysis, lambda projection,
and task vector injection experiments.
"""

import torch
import numpy as np
import os
import pickle
from typing import List, Tuple, Optional

from icl.linear.linear_utils import load_model_task_config, compute_hiddens, estimate_lambda_with_r2
from icl.linear.sampling import sample_points_from_balls
from icl.linear.lr_task import get_task
from icl.utils.linear_algebra_utils import project_points_to_plane
from icl.linear.linear_utils import compute_circumcenter
from icl.utils.processor_utils import setup_device


def predict_with_ood_vector(
    model, query_data: torch.Tensor, query_target: torch.Tensor, 
    task_vector: torch.Tensor, l: int = 0, pos: int = 1
) -> torch.Tensor:
    """
    Predict with OOD vector injection into model.
    
    Args:
        model: Trained model
        query_data: Query input data
        query_target: Query target data
        task_vector: Task vector to inject
        l: Layer index for injection
        pos: Position for injection
        
    Returns:
        Predictions with injected task vector
    """
    task_pos = pos
    if task_vector.device != model.device:
        task_vector = task_vector.to(model.device)
    alpha = 1.0

    def inject_hook(module, input, output):
        output[:, task_pos, :] += alpha * task_vector
        return output
    
    hook_handle = model.transformer.blocks[l].attn_block.register_forward_hook(inject_hook)
    with torch.no_grad():
        preds = model(query_data, query_target)
    hook_handle.remove()

    return preds


def enhance_ood(exp_name: str, lambda_ood: List[float], K: int = 6, layer_index: int = 10, 
               pos: int = 1, radius: float = 1, device: Optional[str] = None, 
               batch_size: int = 256, step: int = 12345):
    """
    Enhance OOD performance using lambda projection and task vector injection.
    
    Args:
        exp_name: Experiment name
        lambda_ood: Target lambda values for OOD
        K: Number of tasks per ball
        layer_index: Layer index for task vector extraction
        pos: Position for analysis
        radius: Radius for task sampling
        device: Device to use
        batch_size: Batch size
        step: Random step for sampling
    """
    print("Preprocessing...")
    
    # Device setup
    device = setup_device(device)
    model, train_task, config = load_model_task_config(exp_name)
    
    # Setup paths
    exp_dir = os.path.join(config.work_dir, exp_name)   
    cur_dir = os.getcwd()
    if cur_dir.endswith("notebooks"):
        exp_dir = os.path.join("..", exp_dir)
    
    anchor_pool = train_task.task_pool.squeeze(-1).to(device)
    
    # Generate evaluation task pool on GPU
    n_per_ball = K // 3
    K = n_per_ball * 3
    eval_task_pool, weights = sample_points_from_balls(
        anchor_pool, r=radius, n_per_ball=n_per_ball
    )
    
    # Setup evaluation task
    eval_config = config.copy() if isinstance(config, dict) else config
    eval_config.task.n_tasks = K + 3
    eval_config.device = device 
    eval_task = get_task(**eval_config["task"], device=device)
    eval_task.batch_size = batch_size
    eval_task.task_pool = eval_task_pool.unsqueeze(-1)  # shape (K+3, d, 1)
    
    hiddens_all_time, _ = compute_hiddens(config, model, eval_task, layer_index) # (K+3, T, B, d_emb)
    
    task_mean = hiddens_all_time[:3].mean(dim=(0,2)).unsqueeze(0) # (1, T, d_emb)
    task_vecs_selected_time = hiddens_all_time[:, pos] - task_mean[:, pos:pos+1] # (K+3, B, d_emb)
    final_task_vecs = hiddens_all_time[:3].mean(dim=-2) - task_mean 
    final_task_vecs = final_task_vecs[:, -1] # (3, d_emb)
    lambdas, _, _, _ = estimate_lambda_with_r2(final_task_vecs, task_vecs_selected_time) # (K+3, B, 3)
    
    lambda_ood = torch.tensor(lambda_ood, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    lambdas = torch.tensor(lambdas, device=device, dtype=torch.float32)
    distances = torch.norm(lambdas - lambda_ood, dim=-1)   # (K+3, B)

    # Average distance for each k
    avg_dist = distances.mean(dim=1)   # (K,)
    
    # Find tasks with largest distances
    topk = torch.topk(avg_dist, k=6)
    topk_idx = topk.indices.cpu().numpy()
    print(topk_idx)
    inj_vectors = (lambda_ood - lambdas[topk.indices]) @ final_task_vecs # (6, B, d_emb)
    
    for i, tid in enumerate(topk_idx):
        if tid in [0, 1, 2]:
            continue
        query_data, query_target = eval_task.sample_from_task(eval_task.task_pool[tid], step=50)
        preds_inj = predict_with_ood_vector(
            model=model,
            query_data=query_data,
            query_target=query_target,
            task_vector=inj_vectors[i],
            l=layer_index,             
            pos=3 * pos + 1,
        )
        with torch.no_grad():
            preds = model(query_data, query_target)

        loss = ((preds_inj - query_target.to(preds_inj.device))**2).mean(dim=0)
        baseline_loss = ((preds - query_target.to(preds.device))**2).mean(dim=0)
        print("Injection Loss:", loss[pos:pos+10])
        print("Baseline Loss:", baseline_loss[pos:pos+10])


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
    
    #print("Computing PCA projections...")
    #anchor_np = anchor_pool.cpu().numpy()
    #eval_np = eval_task_pool.cpu().numpy()
    #eval_2d = project_points_to_plane(eval_np, anchor_np)
    #anchor_2d = eval_2d[:3]
    #center_2d = compute_circumcenter(anchor_2d[0], anchor_2d[1], anchor_2d[2])
    
    #directions = anchor_2d - center_2d[None, :]  # (3, 2)
    #directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    #eval_2d_directions = eval_2d - center_2d[None, :]  # (K+3, 2)
    #eval_2d_directions = eval_2d_directions / np.linalg.norm(eval_2d_directions, axis=-1, keepdims=True)
    #projections = eval_2d_directions @ directions.T  # (K+3, 3)
    
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