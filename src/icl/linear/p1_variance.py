"""
P1 Variance Computation

Computes conditional residual variance of hidden states given fixed current tokens.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Sequence
from dataclasses import dataclass, field
import json


@dataclass
class P1VarianceResults:
    """Results from P1 variance computation."""
    var_token: Dict[Tuple[int, int], float]  # (position, token_idx) -> variance
    var_pos: Dict[int, float]  # position -> mean variance
    var_pos_std: Dict[int, float]  # position -> std of variances across tokens
    var_pos_norm: Dict[int, float]  # position -> normalized variance
    means: Dict[Tuple[int, int], torch.Tensor]  # (position, token_idx) -> mean vector
    counts: Dict[Tuple[int, int], int]  # (position, token_idx) -> sample count
    layer_idx: int  # index in the layers list (not the actual layer number)
    layer_num: Optional[int] = field(default=None)  # actual layer number (if provided)


def compute_p1_variance(
    all_hiddens: torch.Tensor,
    token_info: dict,
    layer_idx: int = 0,
    layer_num: Optional[int] = None,
    min_count: int = 1,
    eps: float = 1e-8,
) -> P1VarianceResults:
    """
    Compute conditional residual variance of hidden states given fixed tokens AND tasks.
    
    For each position t, token x, and task k:
    1. Computes conditional variance: var_token_task[t][x][k] = Var(H | token=x, task=k)
       This measures how much hidden states vary when BOTH token x and task k are fixed.
    
    2. Averages conditional variances across tasks: var_token[t][x] = mean_k var_token_task[t][x][k]
       This gives the average conditional variance for token x (averaged across tasks).
    
    3. Averages conditional variances across tokens: var_pos[t] = mean_x var_token[t][x]
       This gives the average conditional variance at position t.
    
    Parameters:
    -----------
    all_hiddens : torch.Tensor
        Shape: (L, n_positions, n_unique_tokens, n_tasks, batch_size, n_embd)
        Output from compute_hiddens_token_conditioned
        all_hiddens[l, t, x, k, b, :] contains hidden state when token x is fixed at position t,
        for task k, batch b
    token_info : dict
        Token information from compute_hiddens_token_conditioned
    layer_idx : int, default=0
        Which layer to analyze (index in the layers list)
    layer_num : int, optional
        Actual layer number (for reference/display)
    min_count : int, default=1
        Minimum number of samples required per token group
    eps : float, default=1e-8
        Small epsilon for numerical stability in normalization
    
    Returns:
    --------
    P1VarianceResults
        Structured results containing variance metrics:
        - var_token: conditional variance for each (position, token) pair (averaged across tasks)
        - var_pos: average conditional variance at each position (averaged across tokens and tasks)
        - var_pos_std: std of conditional variances across tokens at each position
        - var_pos_norm: normalized average conditional variance
    """
    device = all_hiddens.device
    dtype = all_hiddens.dtype
    
    # Extract data for the specified layer
    # Shape: (n_positions, n_unique_tokens, n_tasks, batch_size, n_embd)
    if layer_idx >= all_hiddens.shape[0]:
        raise ValueError(f"layer_idx {layer_idx} out of range [0, {all_hiddens.shape[0]})")
    hiddens = all_hiddens[layer_idx]
    
    L, n_positions, n_unique_tokens, n_tasks, batch_size, n_embd = all_hiddens.shape
    positions = token_info['positions']
    
    # Initialize result dictionaries
    var_token = {}
    var_pos = {}
    var_pos_std = {}
    var_pos_norm = {}
    means = {}
    counts = {}
    
    # For each position t
    for pos_idx_idx, pos_idx in enumerate(positions):
        if pos_idx_idx >= n_positions:
            continue
            
        # Store conditional variances for each token at this position
        conditional_variances = []  # var_token[t][x] for each token x
        token_means_norm_sq = []
        
        # For each unique token x at this position
        n_unique_at_pos = len(token_info['unique_tokens'][pos_idx])
        
        for token_idx in range(min(n_unique_at_pos, n_unique_tokens)):
            # Extract hidden states CONDITIONED on token x being fixed at position t
            # Shape: (n_tasks, batch_size, n_embd)
            # All samples in H_tx have the same token x fixed at position t
            H_tx = hiddens[pos_idx_idx, token_idx]
            
            # STEP 1: Compute conditional variance separately for each task
            # We condition on BOTH task and token: Var(H | token=x, task=k)
            task_variances = []  # Store variance for each task
            task_means_norm_sq = []  # Store ||mu||² for each task
            
            task_means_list = []  # Store means for each task (for debugging/storage)
            
            for task_idx in range(n_tasks):
                # Extract hidden states for this specific (task, token) combination
                # Shape: (batch_size, n_embd)
                H_txk = H_tx[task_idx]  # Hidden states for task k, token x, position t
                
                # Check if we have enough samples
                if batch_size < min_count:
                    continue
                
                # Compute CONDITIONAL mean given token x AND task k are fixed
                # mu[t][x][k] = E[H | token at position t = x, task = k]
                mu_txk = H_txk.mean(dim=0)  # Shape: [d] - mean across batches for this task
                task_means_list.append(mu_txk.detach().clone())
                
                # Compute residuals (deviations from conditional mean)
                R_txk = H_txk - mu_txk.unsqueeze(0)  # Shape: [batch_size, d]
                
                # Compute CONDITIONAL VARIANCE for this (token, task) combination
                # var_token_task[t][x][k] = Var(H | token=x, task=k)
                # This is the variance of hidden states GIVEN token x AND task k are fixed
                residuals_norm_sq = (R_txk ** 2).sum(dim=1)  # Shape: [batch_size]
                var_token_task_txk = residuals_norm_sq.mean().item()  # Average across batches
                task_variances.append(var_token_task_txk)
                
                # For normalization: ||mu[t][x][k]||_2^2
                mu_norm_sq = (mu_txk ** 2).sum().item()
                task_means_norm_sq.append(mu_norm_sq)
            
            # STEP 2: Average conditional variances across tasks
            # var_token[t][x] = mean_k Var(H | token=x, task=k)
            if len(task_variances) > 0:
                var_token_tx = np.mean(task_variances)  # Average across tasks
                var_token[(pos_idx, token_idx)] = var_token_tx
                conditional_variances.append(var_token_tx)  # Store conditional variance for this token
                
                # Store average mean across tasks (for debugging)
                if len(task_means_list) > 0:
                    avg_mean_tx = torch.stack(task_means_list).mean(dim=0)  # Average mean across tasks
                    means[(pos_idx, token_idx)] = avg_mean_tx
                    counts[(pos_idx, token_idx)] = len(task_variances) * batch_size  # Total samples
                
                # For normalization: average ||mu||² across tasks
                mean_mu_norm_sq_tx = np.mean(task_means_norm_sq)
                token_means_norm_sq.append(mean_mu_norm_sq_tx)
            else:
                # No valid tasks for this token
                continue
        
        # STEP 4: Average conditional variances across all tokens at this position
        # var_pos[t] = (1/|X|) * Σ_x var_token[t][x]
        # This gives the average conditional variance at position t
        if len(conditional_variances) > 0:
            var_pos[pos_idx] = np.mean(conditional_variances)  # Average of conditional variances
            var_pos_std[pos_idx] = np.std(conditional_variances)  # Std of conditional variances across tokens
            
            # Normalized version: var_pos_norm[t] = var_pos[t] / mean_x ||mu[t][x]||_2^2
            mean_mu_norm_sq = np.mean(token_means_norm_sq)
            var_pos_norm[pos_idx] = var_pos[pos_idx] / (mean_mu_norm_sq + eps)
        else:
            # No valid tokens for this position
            var_pos[pos_idx] = np.nan
            var_pos_std[pos_idx] = np.nan
            var_pos_norm[pos_idx] = np.nan
    
    return P1VarianceResults(
        var_token=var_token,
        var_pos=var_pos,
        var_pos_std=var_pos_std,
        var_pos_norm=var_pos_norm,
        means=means,
        counts=counts,
        layer_idx=layer_idx,
        layer_num=layer_num,
    )


def compute_p1_variance_multi_layer(
    all_hiddens: torch.Tensor,
    token_info: dict,
    layers: Sequence[int] = None,
    min_count: int = 1,
    eps: float = 1e-8,
) -> Dict[int, P1VarianceResults]:
    """
    Compute P1 variance for multiple layers.
    
    Parameters:
    -----------
    all_hiddens : torch.Tensor
        Shape: (L, n_positions, n_unique_tokens, n_tasks, batch_size, n_embd)
    token_info : dict
        Token information
    layers : Sequence[int], optional
        Which layers to analyze. If None, analyzes all layers.
    min_count : int, default=1
        Minimum number of samples required per token group
    eps : float, default=1e-8
        Small epsilon for numerical stability
    
    Returns:
    --------
    Dict[int, P1VarianceResults]
        Mapping from layer index to results
    """
    L = all_hiddens.shape[0]
    
    if layers is None:
        layers = list(range(L))
    
    results = {}
    for i, layer_num in enumerate(layers):
        results[layer_num] = compute_p1_variance(
            all_hiddens=all_hiddens,
            token_info=token_info,
            layer_idx=i,  # Index in layers list
            layer_num=layer_num,  # Actual layer number
            min_count=min_count,
            eps=eps,
        )
    
    return results


def results_to_dict(results: P1VarianceResults) -> dict:
    """Convert P1VarianceResults to a JSON-serializable dictionary."""
    output = {
        'var_token': {f"{pos}_{tok}": float(v) for (pos, tok), v in results.var_token.items()},
        'var_pos': {str(pos): float(v) for pos, v in results.var_pos.items()},
        'var_pos_std': {str(pos): float(v) for pos, v in results.var_pos_std.items()},
        'var_pos_norm': {str(pos): float(v) for pos, v in results.var_pos_norm.items()},
        'counts': {f"{pos}_{tok}": int(c) for (pos, tok), c in results.counts.items()},
        'layer_idx': results.layer_idx,
    }
    if results.layer_num is not None:
        output['layer_num'] = results.layer_num
    return output


def results_to_table(results: P1VarianceResults, positions: Sequence[int] = None) -> str:
    """
    Convert results to a formatted table string.
    
    Parameters:
    -----------
    results : P1VarianceResults
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
    lines.append(f"P1 Variance Results ({layer_str})")
    lines.append("=" * 80)
    lines.append(f"{'Position':<10} {'Var_Pos':<15} {'Var_Pos_Std':<15} {'Var_Pos_Norm':<15}")
    lines.append("-" * 80)
    
    for pos in positions:
        var_pos_val = results.var_pos.get(pos, np.nan)
        var_pos_std_val = results.var_pos_std.get(pos, np.nan)
        var_pos_norm_val = results.var_pos_norm.get(pos, np.nan)
        
        lines.append(
            f"{pos:<10} "
            f"{var_pos_val:<15.6f} "
            f"{var_pos_std_val:<15.6f} "
            f"{var_pos_norm_val:<15.6f}"
        )
    
    return "\n".join(lines)


def save_results_json(results: P1VarianceResults, filepath: str):
    """Save results to a JSON file."""
    results_dict = results_to_dict(results)
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=2)


def save_results_multi_layer_json(results_dict: Dict[int, P1VarianceResults], filepath: str):
    """Save multi-layer results to a JSON file."""
    output = {
        'layers': {str(layer_idx): results_to_dict(results) 
                   for layer_idx, results in results_dict.items()}
    }
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)


def extract_plotting_data(results: P1VarianceResults) -> dict:
    """
    Extract data in a format convenient for plotting.
    
    Returns:
    --------
    dict with keys:
        - 'positions': list of position indices
        - 'var_pos': list of var_pos values (same order as positions)
        - 'var_pos_std': list of var_pos_std values
        - 'var_pos_norm': list of var_pos_norm values
        - 'layer_idx': layer index in layers list
        - 'layer_num': actual layer number (if available)
    """
    positions = sorted(results.var_pos.keys())
    output = {
        'positions': positions,
        'var_pos': [results.var_pos[p] for p in positions],
        'var_pos_std': [results.var_pos_std[p] for p in positions],
        'var_pos_norm': [results.var_pos_norm[p] for p in positions],
        'layer_idx': results.layer_idx,
    }
    if results.layer_num is not None:
        output['layer_num'] = results.layer_num
    return output


def extract_plotting_data_multi_layer(results_dict: Dict[int, P1VarianceResults]) -> dict:
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
        'var_pos_std': {
            layer_idx: [results_dict[layer_idx].var_pos_std[p] for p in positions]
            for layer_idx in results_dict.keys()
        },
        'var_pos_norm': {
            layer_idx: [results_dict[layer_idx].var_pos_norm[p] for p in positions]
            for layer_idx in results_dict.keys()
        },
    }

