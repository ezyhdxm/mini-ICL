"""
Linear processor for ICL experiments.

This module contains the main processing functions for linear experiments,
including model processing, checkpoint analysis, and out-of-distribution evaluation.
"""

import re
import numpy as np
import torch
import plotly.graph_objects as go
from tqdm import tqdm

from icl.linear.sampling import *
from icl.linear.task_vecs import *
from icl.linear.linear_utils import *
from icl.utils.linear_algebra_utils import project_points_to_plane
from icl.utils.processor_utils import (
    setup_device, setup_experiment_paths, create_result_path, 
    load_cached_results, save_results, setup_eval_task,
    compute_pca_projections, robust_compositional_timeseries,
    pairwise_cosine_similarity
)


def process_model(exp_name, radius=2, orthogonal_offset=1, is_on_sphere=True, is_zero_mean=True, K=90, task_layer_index=3):
    """
    Processes checkpoints for a given experiment, performs PCA projection, 
    and plots the lambda projection for each checkpoint.

    Args:
    - exp_name (str): Experiment name.
    - K (int): Number of tasks for evaluation (default 90).
    - task_layer_index (int): Layer index for task vector extraction (default 3).
    """
    model, train_task, config = load_model_task_config(exp_name)
    
    # Extract task vectors from training tasks
    final_hiddens, final_task_vectors = get_task_vector_from_hidden(
        config, model, train_task, task_layer_index, 
        compute_mean=True, return_final=True
    )
    if not is_zero_mean:
        final_task_vectors = final_hiddens.mean(dim=-2)

    # Setup evaluation task pool
    anchor_pool = train_task.task_pool.squeeze(-1)
    eval_task_pool, weights = sample_union_unit_balls_affine_span_with_weights(
        anchor_pool, K, radius, orthogonal_offset, is_on_sphere
    )

    # Create evaluation task
    eval_task = setup_eval_task(config, K, device=None, batch_size=256)
    eval_task.task_pool = eval_task_pool.unsqueeze(-1)  # shape (K, d, 1)
        
    # Extract task vectors from evaluation tasks
    eval_hiddens, eval_task_vectors = get_task_vector_from_hidden(
        config, model, eval_task, layer_index=task_layer_index, 
        compute_mean=True, return_final=True
    )
    if not is_zero_mean:
        eval_task_vectors = eval_hiddens.mean(dim=-2)
    
    # Estimate lambdas and plot
    lambdas, r2_score = estimate_lambda_super_fast(
        final_task_vectors, eval_task_vectors, compute_r2=True, is_zero_mean=is_zero_mean
    )
    plot_lambda_projection(model, eval_task, train_task, r2_score, lambdas, weights, title=None)


def process_checkpoints_lambda_metrics(exp_name, K=60, layer_index=3, 
                                       orthogonal_offset=1, is_on_sphere=True,
                                       radius=2,
                                       device=None, skip_factor=10, 
                                       batch_size=256, max_checkpoints=None, forced=False):
    
    print("Preprocessing...")
    
    # Setup device and load configuration
    device = setup_device(device)
    _, train_task, config = load_model_task_config(exp_name)
    
    # Setup paths and check for cached results
    exp_dir = setup_experiment_paths(config, exp_name)
    result_path = create_result_path(exp_dir, orthogonal_offset, radius, is_on_sphere)
    
    cached_results = load_cached_results(result_path, forced)
    if cached_results is not None:
        return cached_results
    
    # Setup checkpoint files
    checkpoint_files = get_checkpoint_files(exp_name)
    checkpoint_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    
    if max_checkpoints:
        checkpoint_files = checkpoint_files[:max_checkpoints]
        print(f"Limited to {max_checkpoints} checkpoints for testing")
    
    # Setup evaluation tasks and compute PCA projections
    anchor_pool = train_task.task_pool.squeeze(-1).to(device)
    eval_task_pool, weights = sample_union_unit_balls_affine_span_with_weights(
        anchor_pool, K, radius, orthogonal_offset, is_on_sphere
    )
    
    eval_task = setup_eval_task(config, K+3, device, batch_size)
    eval_task.task_pool = eval_task_pool.unsqueeze(-1)  # shape (K+3, d, 1)
    
    print("Computing PCA projections...")
    scores, projections = compute_pca_projections(anchor_pool, eval_task_pool)
    scores_tensor = torch.tensor(scores, device=device, dtype=torch.float32)
    uniform_target = torch.ones(3, device=device, dtype=torch.float32) / 3  # (3,)
    
    print(f"Processing {len(checkpoint_files)} checkpoints...")
    prev_step = -skip_factor
    
    # Pre-determine which checkpoints we'll actually process
    steps_to_process = []
    checkpoint_indices = []
    
    for k in range(0, len(checkpoint_files), 2):
        curr_step = int(re.search(r'\d+', checkpoint_files[k]).group())
        if curr_step > 8000 and curr_step % 500 != 0:
            continue
        skip_threshold = skip_factor
        
        if curr_step - prev_step >= skip_threshold:
            steps_to_process.append(curr_step)
            checkpoint_indices.append(k)
            prev_step = curr_step
    
    num_checkpoints = len(steps_to_process)
    print(f"Will process {num_checkpoints} checkpoints (skipping {len(checkpoint_files) - num_checkpoints})")
    
    all_lambdas = torch.empty(num_checkpoints, K+3, 3, device=device, dtype=torch.float32)
    orthogonal_score = torch.empty(num_checkpoints, K+3, device=device, dtype=torch.float32)
    task_vec_norm = torch.empty(num_checkpoints, K+3, device=device, dtype=torch.float32)
    processed_steps = []
    processed_count = 0
    
    try:
        for idx, (step, k) in enumerate(tqdm(zip(steps_to_process, checkpoint_indices), total=num_checkpoints)):
            model, config, _ = load_checkpoint(exp_name, checkpoint_files[k])
            model = model.to(device)  # Ensure model is on correct device
            model.eval()  # Set to eval mode

            _, train_task_vectors = get_task_vector_from_hidden(
                                        config, model, train_task, 
                                        layer_index=layer_index, 
                                        compute_mean=True, 
                                        return_final=True
                                    )
            
            try:
                with torch.no_grad():  
                    _, eval_task_vectors = get_task_vector_from_hidden(
                        config, model, eval_task, 
                        layer_index=layer_index, 
                        compute_mean=True, 
                        return_final=True
                    )
                    
                    lambdas, *_ = estimate_lambda_super_fast(
                        train_task_vectors, eval_task_vectors, compute_r2=False
                    )
                    
                    all_lambdas[idx] = lambdas
                    processed_steps.append(step)
                    processed_count += 1

                    A = eval_task_vectors # (K+3, emb_dim)
                    # Compute projection matrix P = Bᵗ (B Bᵗ)⁻¹ B
                    B = train_task_vectors[:2]  # (2, emb_dim)
                    BBT_pinv = torch.pinverse(B @ B.T)
                    proj = A @ B.T @ BBT_pinv @ B                   
                    residual = A - proj
                    orthogonal_score[idx] = residual.norm(dim=-1) / A.norm(dim=-1) # K+3
                    task_vec_norm[idx] = A.norm(dim=-1)
                                    
            except RuntimeError as e:
                print(f"Error processing checkpoint {checkpoint_files[k]}: {e}")
                # Mark this slot as invalid (fill with NaN)
                all_lambdas[idx] = float('nan')
                orthogonal_score[idx] = float('nan')
                task_vec_norm[idx] = float('nan')
                processed_steps.append(step)
                continue
    
    except KeyboardInterrupt:
        print(f"\nInterrupted. Processed {processed_count} checkpoints.")
        # Truncate to what we actually processed
        all_lambdas = all_lambdas[:len(processed_steps)]
        orthogonal_score = orthogonal_score[:len(processed_steps)]
        task_vec_norm = task_vec_norm[:len(processed_steps)]
    
    # Compute metrics efficiently on GPU
    print("Computing metrics...")
    
    # Filter out failed checkpoints (NaN values)
    valid_mask = ~torch.isnan(all_lambdas).any(dim=-1).any(dim=-1)  # (num_checkpoints,)
    valid_lambdas = all_lambdas[valid_mask]  # (valid_count, K+3, 3)
    valid_orthogonal_score = orthogonal_score[valid_mask]
    valid_task_vec_norm = task_vec_norm[valid_mask]
    valid_steps = [step for i, step in enumerate(processed_steps) if valid_mask[i]]
    
    if len(valid_lambdas) == 0:
        print("No valid checkpoints processed!")
        return {}
    
    # Batch compute all metrics on GPU
    uniform_diffs = torch.abs(uniform_target.unsqueeze(0).unsqueeze(0) - valid_lambdas).sum(dim=-1).mean(dim=-1)
    part_diffs = torch.abs(scores_tensor.unsqueeze(0) - valid_lambdas).sum(dim=-1).mean(dim=-1)
    weight_diffs = torch.abs(weights.unsqueeze(0) - valid_lambdas).sum(dim=-1).mean(dim=-1)
    
    # Transfer to CPU once and convert to dictionaries
    results_cpu = {
        'uniform': uniform_diffs.cpu().numpy(),
        'part': part_diffs.cpu().numpy(), 
        'weight': weight_diffs.cpu().numpy(),
        'orthogonal': valid_orthogonal_score.cpu().numpy(),
        'task_vec_norm': valid_task_vec_norm.cpu().numpy()
    }
    
    # Convert arrays to step-indexed dictionaries  
    unif_metrics = {step: float(result) for step, result in zip(valid_steps, results_cpu['uniform'])}
    part_metrics = {step: float(result) for step, result in zip(valid_steps, results_cpu['part'])}
    weight_metrics = {step: float(result) for step, result in zip(valid_steps, results_cpu['weight'])}
    
    results_dict = {
        'unif_metrics': unif_metrics,
        'part_metrics': part_metrics, 
        'weight_metrics': weight_metrics,
        'orthogonal_results': results_cpu['orthogonal'],
        'task_vec_norm_results': results_cpu['task_vec_norm'], 
        'weights': weights,
        'valid_lambdas': valid_lambdas
    }
    
    print(f"Completed: {len(valid_steps)} processed successfully")
    save_results(result_path, results_dict)
    return results_dict


def process_ood_evolve_lambda_metrics(exp_name, K=300, layer_index=3, 
                                      orthogonal_offset=0, is_on_sphere=False, is_zero_mean=True,
                                      radius=2, device=None, batch_size=256):
    
    print("Preprocessing...")
    
    # Setup device and load configuration
    device = setup_device(device)
    model, train_task, config = load_model_task_config(exp_name)
    
    # Setup paths and check for cached results
    exp_dir = setup_experiment_paths(config, exp_name)
    result_path = create_result_path(exp_dir, orthogonal_offset, radius, is_on_sphere, prefix="ood_results")
    
    cached_results = load_cached_results(result_path, forced=False)
    if cached_results is not None:
        print("Already computed. Loading existing results.")
        return cached_results
    
    # Setup evaluation task pool
    anchor_pool = train_task.task_pool.squeeze(-1).to(device)
    n_per_ball = K // 3
    K = n_per_ball * 3
    eval_task_pool, weights = sample_points_from_balls(anchor_pool, r=radius, n_per_ball=n_per_ball)
    
    eval_task = setup_eval_task(config, K+3, device, batch_size)
    eval_task.task_pool = eval_task_pool.unsqueeze(-1)  # shape (K+3, d, 1)
    
    print("Computing PCA projections...")
    # Use specialized projection for this case
    anchor_np = anchor_pool.cpu().numpy()
    eval_np = eval_task_pool.cpu().numpy()
    eval_2d = project_points_to_plane(eval_np, anchor_np)
    anchor_2d = eval_2d[:3]
    center_2d = compute_circumcenter(anchor_2d[0], anchor_2d[1], anchor_2d[2])
    
    directions = anchor_2d - center_2d[None, :]  # (3, 2)
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    eval_2d_directions = eval_2d - center_2d[None, :]  # (K+3, 2)
    eval_2d_directions = eval_2d_directions / np.linalg.norm(eval_2d_directions, axis=-1, keepdims=True)
    projections = eval_2d_directions @ directions.T  # (K+3, 3)
    
    hiddens_all_time, _ = compute_hiddens(config, 
                                          model, 
                                          eval_task, 
                                          layer_index) # (K+3, T, B, d_emb)
    if is_zero_mean:
        task_mean = hiddens_all_time[:3].mean(dim=(0,2)).unsqueeze(0) # (1, T, d_emb)
        task_vecs_over_all_time = hiddens_all_time.mean(dim=-2) - task_mean #task_mean  # (K+3, T, d_emb)
        final_task_vecs = hiddens_all_time[:3].mean(dim=-2) - task_mean 
        final_task_vecs = final_task_vecs[:, -1] # (3, d_emb)
        lambdas, r2_scores, task_norms, ortho_norms = estimate_lambda_with_r2(final_task_vecs, 
                                                                 task_vecs_over_all_time)
    else:
        final_hidden = hiddens_all_time[:3, -1].mean(dim=-2)
        lambdas, r2_scores, task_norms, ortho_norms = estimate_lambda_with_r2(final_hidden, 
                                                                              hiddens_all_time.mean(dim=-2), 
                                                                              is_zero_mean)
    plot_lambda_projection_with_slider(
        model, eval_task, train_task, r2_scores, lambdas, weights, n_points=config.task.n_points, title=None
    )

    plot_lambda_projection_norm(
        model, eval_task, train_task,
        task_norms,                # (K,) or (K,T)
        ortho_norms,               # (K,) or (K,T)
        weights,                   # (K,3) or (K,T,3) or None
        lambdas,                   # (K,3) or (K,T,3)  (used only for hover, NOT for color)
        n_points=config.task.n_points,
        title=None,
        slider_mode="log",         # "log" or "linear"
        log_gamma=5.0,             # curvature; bigger => denser at small t
        max_slider_steps=256       # cap number of slider steps (deduped later)
    )
    return lambdas  # (K,T,3)

