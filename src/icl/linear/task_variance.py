"""
Task Variance Computation

Computes variance of batch-averaged hidden states across tasks.
For each position, measures how much task-specific means vary around the global mean.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Sequence
from dataclasses import dataclass, field
import json


@dataclass
class TaskVarianceResults:
    """Results from task variance computation."""
    var_pos: Dict[int, float]  # position -> variance across tasks
    var_pos_norm: Dict[int, float]  # position -> normalized variance
    k_mean: Dict[int, torch.Tensor]  # position -> global mean vector (averaged across tasks)
    b_mean: Optional[torch.Tensor] = None  # (n_tasks, n_points, n_embd) - task-specific means (optional, for debugging)
    layer_idx: int = 0  # index in the layers list (not the actual layer number)
    layer_num: Optional[int] = field(default=None)  # actual layer number (if provided)


def compute_task_variance(
    all_hiddens: torch.Tensor,
    positions_of_interest: Sequence[int] = None,
    layer_idx: int = 0,
    layer_num: Optional[int] = None,
    eps: float = 1e-8,
) -> TaskVarianceResults:
    """
    Compute variance of diff (hiddens - task_mean) over batch_size, then average across tasks.
    
    For each position p:
    1. Compute task_mean[p] = mean_k H[k, p, :, :] (average over tasks)
       Shape: (batch_size, n_embd)
    2. Compute diff[k, p, b, :] = H[k, p, b, :] - task_mean[p, b, :]
       Shape: (n_tasks, batch_size, n_embd)
    3. For each task k: compute diff_mean[k, p, :] = mean_b diff[k, p, b, :]
       Shape: (n_tasks, n_embd)
    4. For each task k: compute variance over batch_size
       var_task[k, p] = mean_b ||diff[k, p, b, :] - diff_mean[k, p, :]||_2^2
    5. Average across tasks: var_pos[p] = mean_k var_task[k, p]
    6. Normalize: var_pos_norm[p] = var_pos[p] / (||task_mean[p]||_2^2 + eps)
       where ||task_mean[p]||_2^2 is averaged over batch_size
    
    Parameters:
    -----------
    all_hiddens : torch.Tensor
        Shape: (L, n_tasks, n_points, batch_size, n_embd)
        Output from compute_hiddens_multi with return_final=False
    positions_of_interest : Sequence[int], optional
        Which positions (point indices) to analyze. If None, uses all positions.
    layer_idx : int, default=0
        Which layer to analyze (index in the layers list)
    layer_num : int, optional
        Actual layer number (for reference/display)
    eps : float, default=1e-8
        Small epsilon for numerical stability in normalization
    
    Returns:
    --------
    TaskVarianceResults
        Structured results containing variance metrics:
        - var_pos: variance of diff over batch_size, averaged across tasks
        - var_pos_norm: normalized variance
        - k_mean: task_mean vectors at each position (averaged over batch_size)
    """
    if layer_idx >= all_hiddens.shape[0]:
        raise ValueError(f"layer_idx {layer_idx} out of range [0, {all_hiddens.shape[0]})")
    
    # Extract data for the specified layer
    # Shape: (n_tasks, n_points, batch_size, n_embd)
    hiddens = all_hiddens[layer_idx]
    
    L, n_tasks, n_points, batch_size, n_embd = all_hiddens.shape
    
    # Determine positions of interest
    if positions_of_interest is None:
        positions_of_interest = list(range(n_points))
    else:
        positions_of_interest = list(positions_of_interest)
        if not all(0 <= p < n_points for p in positions_of_interest):
            raise ValueError(f"All positions must be in [0, {n_points-1}]")
    
    # Step 1: Compute task_mean = mean over n_tasks
    # Shape: (n_points, batch_size, n_embd)
    task_mean = hiddens.mean(dim=0)  # Average over task dimension
    
    # Step 2: Compute diff = hiddens - task_mean
    # Shape: (n_tasks, n_points, batch_size, n_embd)
    diff = hiddens - task_mean.unsqueeze(0)  # Broadcast task_mean to match hiddens shape
    
    # Step 3-5: For each position, compute variance of diff over batch_size, then average across tasks
    var_pos = {}
    var_pos_norm = {}
    k_mean_dict = {}
    
    for pos_idx in positions_of_interest:
        # Extract diff for this position
        # Shape: (n_tasks, batch_size, n_embd)
        diff_pos = diff[:, pos_idx, :, :]
        
        # Compute diff_mean = mean of diff over batch_size for each task
        # Shape: (n_tasks, n_embd)
        diff_mean = diff_pos.mean(dim=1)
        
        # Compute variance of diff about diff_mean over batch_size for each task
        # For each task k: var_task[k] = mean_b ||diff[k, b, :] - diff_mean[k, :]||_2^2
        # Shape: (n_tasks, batch_size, n_embd)
        residuals = diff_pos - diff_mean.unsqueeze(1)  # Broadcast diff_mean
        
        # Compute squared L2 norm for each (task, batch) pair
        # Shape: (n_tasks, batch_size)
        residuals_norm_sq = (residuals ** 2).sum(dim=2)
        
        # Average over batch_size for each task
        # Shape: (n_tasks,)
        var_task = residuals_norm_sq.mean(dim=1)
        
        # Average across tasks
        var_pos[pos_idx] = var_task.mean().item()
        
        # Normalize by squared L2 norm of task_mean (averaged over batch_size)
        # task_mean for this position: Shape: (batch_size, n_embd)
        task_mean_pos = task_mean[pos_idx, :, :]
        task_mean_pos_avg = task_mean_pos.mean(dim=0)  # Average over batch_size
        task_mean_norm_sq = (task_mean_pos_avg ** 2).sum().item()
        var_pos_norm[pos_idx] = var_pos[pos_idx] / (task_mean_norm_sq + eps)
        
        # Store task_mean (averaged over batch_size) for this position
        k_mean_dict[pos_idx] = task_mean_pos_avg.detach().clone()
    
    return TaskVarianceResults(
        var_pos=var_pos,
        var_pos_norm=var_pos_norm,
        k_mean=k_mean_dict,
        b_mean=None,  # Not storing b_mean in this version
        layer_idx=layer_idx,
        layer_num=layer_num,
    )


def compute_task_variance_multi_layer(
    all_hiddens: torch.Tensor,
    positions_of_interest: Sequence[int] = None,
    layers: Sequence[int] = None,
    eps: float = 1e-8,
) -> Dict[int, TaskVarianceResults]:
    """
    Compute task variance for multiple layers.
    
    Parameters:
    -----------
    all_hiddens : torch.Tensor
        Shape: (L, n_tasks, n_points, batch_size, n_embd)
    positions_of_interest : Sequence[int], optional
        Which positions to analyze. If None, uses all positions.
    layers : Sequence[int], optional
        Which layers to analyze. If None, analyzes all layers.
    eps : float, default=1e-8
        Small epsilon for numerical stability
    
    Returns:
    --------
    Dict[int, TaskVarianceResults]
        Mapping from layer number to results
    """
    L = all_hiddens.shape[0]
    
    if layers is None:
        layers = list(range(L))
    
    results = {}
    for i, layer_num in enumerate(layers):
        if i >= L:
            continue
        results[layer_num] = compute_task_variance(
            all_hiddens=all_hiddens,
            positions_of_interest=positions_of_interest,
            layer_idx=i,  # Index in layers list
            layer_num=layer_num,  # Actual layer number
            eps=eps,
        )
    
    return results


def results_to_dict(results: TaskVarianceResults) -> dict:
    """Convert TaskVarianceResults to a JSON-serializable dictionary."""
    output = {
        'var_pos': {str(pos): float(v) for pos, v in results.var_pos.items()},
        'var_pos_norm': {str(pos): float(v) for pos, v in results.var_pos_norm.items()},
        'layer_idx': results.layer_idx,
    }
    if results.layer_num is not None:
        output['layer_num'] = results.layer_num
    # Note: k_mean and b_mean are tensors, so we don't include them in JSON
    return output


def results_to_table(results: TaskVarianceResults, positions: Sequence[int] = None) -> str:
    """
    Convert results to a formatted table string.
    
    Parameters:
    -----------
    results : TaskVarianceResults
        Results to format
    positions : Sequence[int], optional
        Which positions to include. If None, uses all positions in results.
    
    Returns:
    --------
    str
        Formatted table as string
    """
    if positions is None:
        positions = sorted(results.var_pos.keys())
    
    lines = []
    layer_str = f"Layer {results.layer_num}" if results.layer_num is not None else f"Layer index {results.layer_idx}"
    lines.append(f"Task Variance Results ({layer_str})")
    lines.append("=" * 80)
    lines.append(f"{'Position':<10} {'Var_Pos':<15} {'Var_Pos_Norm':<15}")
    lines.append("-" * 80)
    
    for pos in positions:
        var_pos_val = results.var_pos.get(pos, np.nan)
        var_pos_norm_val = results.var_pos_norm.get(pos, np.nan)
        
        lines.append(
            f"{pos:<10} "
            f"{var_pos_val:<15.6f} "
            f"{var_pos_norm_val:<15.6f}"
        )
    
    return "\n".join(lines)


def save_results_json(results: TaskVarianceResults, filepath: str):
    """Save results to a JSON file."""
    results_dict = results_to_dict(results)
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=2)


def save_results_multi_layer_json(results_dict: Dict[int, TaskVarianceResults], filepath: str):
    """Save multi-layer results to a JSON file."""
    output = {
        'layers': {str(layer_idx): results_to_dict(results) 
                   for layer_idx, results in results_dict.items()}
    }
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)


def extract_plotting_data(results: TaskVarianceResults) -> dict:
    """
    Extract data in a format convenient for plotting.
    
    Returns:
    --------
    dict with keys:
        - 'positions': list of position indices
        - 'var_pos': list of var_pos values (same order as positions)
        - 'var_pos_norm': list of var_pos_norm values
        - 'layer_idx': layer index in layers list
        - 'layer_num': actual layer number (if available)
    """
    positions = sorted(results.var_pos.keys())
    output = {
        'positions': positions,
        'var_pos': [results.var_pos[p] for p in positions],
        'var_pos_norm': [results.var_pos_norm[p] for p in positions],
        'layer_idx': results.layer_idx,
    }
    if results.layer_num is not None:
        output['layer_num'] = results.layer_num
    return output


def extract_plotting_data_multi_layer(results_dict: Dict[int, TaskVarianceResults]) -> dict:
    """
    Extract plotting data for multiple layers.
    
    Returns:
    --------
    dict with keys:
        - 'layers': list of layer indices
        - 'positions': list of position indices (same for all layers)
        - 'var_pos': dict mapping layer_idx -> list of var_pos values
        - 'var_pos_norm': dict mapping layer_idx -> list of var_pos_norm values
    """
    # Get positions from first layer (should be same for all)
    first_layer = next(iter(results_dict.values()))
    positions = sorted(first_layer.var_pos.keys())
    
    return {
        'layers': sorted(results_dict.keys()),
        'positions': positions,
        'var_pos': {
            layer_idx: [results_dict[layer_idx].var_pos[p] for p in positions]
            for layer_idx in results_dict.keys()
        },
        'var_pos_norm': {
            layer_idx: [results_dict[layer_idx].var_pos_norm[p] for p in positions]
            for layer_idx in results_dict.keys()
        },
    }

