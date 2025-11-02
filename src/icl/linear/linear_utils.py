import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from ipywidgets import interact, Dropdown
import ipywidgets as widgets
from typing import Dict, Any, Union, Optional, Tuple
import itertools
from tqdm.notebook import trange, tqdm
from sklearn import linear_model
from sklearn.decomposition import PCA
import cvxpy as cp
from ml_collections import config_flags, ConfigDict
import os
import json
import re
import glob
import imageio
import matplotlib.pyplot as plt

from icl.linear.lr_models import DiscreteMMSE, Ridge
from icl.linear.lr_task import *
from icl.linear.task_vecs import *
from icl.models import apply_rotary_emb
from icl.utils import visualize_attention
from icl.linear.sampling import sample_union_unit_balls_affine_span_with_weights

#TODO: Needs cleaning up


def get_dmmse_posterior(train_task, k):
    """
    Compute Bayesian posterior over task weights using DiscreteMMSE.
    
    Given observed data from a training task, computes the posterior probability
    distribution over which task from the task pool generated the data.
    Uses Gaussian likelihood with known noise scale.
    
    Args:
        train_task: Training task object with task_pool and noise_scale attributes
        k: Task index to sample data from
    
    Returns:
        posterior: Posterior distribution over tasks, shape (B, T, K) where B is batch size,
                  T is sequence length, K is number of tasks
        xs: Input features, shape (B, T, D)
    """
    task_pool = train_task.task_pool.squeeze(-1)  # (n_tasks, D)
    noise_scale = train_task.noise_scale
    xs, ys = train_task.sample_from_task(train_task.task_pool[k], step=0)
    B, T, D = xs.shape
    K = task_pool.shape[0]

    lognum = torch.zeros((B, K), device=xs.device, dtype=xs.dtype)
    posterior = torch.zeros((B, T, K), device=xs.device, dtype=xs.dtype)
    posterior[:, 0, :] = 1.0 / K  # uniform prior at t=0

    for t in range(T-1):
        curr_log = (ys[:,t].unsqueeze(1) - (xs[:,t] @ task_pool.transpose(0,1))).pow(2) / (2 * noise_scale ** 2)
        lognum -= curr_log.squeeze(0).squeeze(0)
        logdenom = torch.logsumexp(lognum, dim=1)
        posterior[:, t+1] = torch.exp(lognum - logdenom[:, None])
    
    return posterior, xs


def get_dmmse_posterior_eval(eval_task, train_task, k):
    """
    Compute Bayesian posterior over task weights for evaluation data.
    
    Similar to get_dmmse_posterior but uses an evaluation task's data
    while still computing posterior over training task pool.
    
    Args:
        eval_task: Evaluation task object to sample data from
        train_task: Training task object with task_pool and noise_scale
        k: Task index to sample data from
    
    Returns:
        posterior: Posterior distribution over tasks, shape (B, T, K)
        xs: Input features, shape (B, T, D)
    """
    task_pool = train_task.task_pool.squeeze(-1)  # (n_tasks, D)
    noise_scale = train_task.noise_scale
    xs, ys = eval_task.sample_from_task(eval_task.task_pool[k], step=0)
    B, T, D = xs.shape
    K = task_pool.shape[0]

    lognum = torch.zeros((B, K), device=xs.device, dtype=xs.dtype)
    posterior = torch.zeros((B, T, K), device=xs.device, dtype=xs.dtype)
    posterior[:, 0, :] = 1.0 / K  # uniform prior at t=0

    for t in range(T-1):
        curr_log = (ys[:,t].unsqueeze(1) - (xs[:,t] @ task_pool.transpose(0,1))).pow(2) / (2 * noise_scale ** 2)
        lognum -= curr_log.squeeze(0).squeeze(0)
        logdenom = torch.logsumexp(lognum, dim=1)
        posterior[:, t+1] = torch.exp(lognum - logdenom[:, None])
    
    return posterior, xs


def estimate_lambda_with_r2(
    task_vecs: torch.Tensor,
    task_vecs_over_all_time: torch.Tensor,
    is_zero_mean: bool = True,
    chunk_size: int = 32,
):
    """
    Memory-friendly version using column-chunking over k.

    Args:
        task_vecs: (num_tasks, d)
        task_vecs_over_all_time: (k, seq_len, d)
        is_zero_mean: exclude last task vec from X and enforce sum-to-1 (same as your code)
        chunk_size: number of columns (k) to solve per batch

    Returns:
        lambdas: (k, seq_len, num_tasks)  [numpy]
        r2_scores: (k, seq_len)           [numpy]
    """
    eps = 1e-12
    device = task_vecs.device
    dtype = task_vecs.dtype

    task_vecs_over_all_time = task_vecs_over_all_time.to(device=device, dtype=dtype)

    k, seq_len, d = task_vecs_over_all_time.shape
    num_tasks = task_vecs.shape[0]

    # outputs on device; convert to numpy at the end
    lambdas = torch.zeros((k, seq_len, num_tasks), dtype=dtype)
    r2_scores = torch.zeros((k, seq_len), dtype=dtype)

    # Design matrix X: (d, num_tasks or num_tasks-1)
    X = task_vecs.T.contiguous()  # (d, num_tasks)
    if is_zero_mean:
        X = X[:, :-1]  # (d, num_tasks-1)

    # Precompute stats used in R^2 denominator per chunk: we still need Y means per chunk
    # We'll compute them on-the-fly to avoid storing big intermediates.

    task_norms = torch.zeros((k, seq_len), dtype=dtype)  
    ortho_norms = torch.zeros((k, seq_len), dtype=dtype)


    with torch.no_grad():
        for t in range(seq_len):
            # Y_full: (d, k)  -- do not clone to avoid extra memory; slice columns per chunk
            Y_full = task_vecs_over_all_time[:, t, :].T  # (d, k)

            # Process k-dimension in chunks
            for start in range(0, k, chunk_size):
                end = min(start + chunk_size, k)

                Y = Y_full[:, start:end]                    # (d, k_chunk)
                # Solve X W = Y for W, least squares (stable)
                # W: (num_tasks-1 or num_tasks, k_chunk)
                W = torch.linalg.lstsq(X, Y).solution

                if is_zero_mean:
                    # Your original "distribute residual equally across all tasks" behavior:
                    # last_lambda = (1 - sum(W)) / num_tasks, then add to all rows (including last)
                    # to make the total sum exactly 1.
                    # residual = 1.0 - W.sum(dim=0, keepdim=True)           # (1, k_chunk)
                    # last_lambda = residual / num_tasks                    # (1, k_chunk)
                    # zero_pad = torch.zeros((1, W.shape[1]), device=device, dtype=dtype)
                    # lambda_full = torch.cat([W, zero_pad], dim=0) + last_lambda  # (num_tasks, k_chunk)
                    # k_chunk = lambda_full.shape[1]
                    # assert torch.allclose(lambda_full.sum(dim=0), torch.ones(k_chunk, device=device, dtype=dtype), atol=1e-6)

                    lambda_full = torch.zeros((num_tasks, W.shape[1]), device=device, dtype=dtype)
                    lambda_full[:-1, :] = W
                    # Make sums = 1 by adding the same offset to all entries (matches your original logic)
                    shift = (1.0 - W.sum(dim=0, keepdim=True)) / num_tasks  # (1, k)
                    lambda_full += shift
                    
                else:
                    lambda_full = W  # (num_tasks, k_chunk)

                # Store lambdas
                lambdas[start:end, t, :] = lambda_full.T.cpu()  # (k_chunk, num_tasks)

                # R^2 per column
                y_pred = X @ W                      # (d, k_chunk)
                resid = Y - y_pred                  # (d, k_chunk)
                ss_res = (resid * resid).sum(dim=0) # (k_chunk,)

                task_norms[start:end, t] = torch.norm(y_pred, dim=0).cpu()  # (k_chunk,)
                ortho_norms[start:end, t] = torch.norm(resid, dim=0).cpu()  # (k_chunk,)

                y_mean = Y.mean(dim=0, keepdim=True)     # (1, k_chunk)
                ss_tot = ((Y - y_mean) ** 2).sum(dim=0)  # (k_chunk,)

                r2 = 1.0 - ss_res / (ss_tot + eps)
                r2.clamp_(min=0.0, max=1.0)
                r2_scores[start:end, t] = r2.cpu()

    return lambdas.numpy(), r2_scores.numpy(), task_norms.numpy(), ortho_norms.numpy()

def decompose_task_vector(
    task_vecs: torch.Tensor,
    task_vecs_over_all_time: torch.Tensor,
    alpha: float = 1.,
    chunk_size: int = 32,
):
    """
    Decompose task vectors into task-space and orthogonal directions.
    
    Projects each task vector onto the span of reference task vectors and the orthogonal
    complement, then combines them with an alpha-weighted enhancement for OOD detection.
    
    Args:
        task_vecs: Reference task vectors, shape (num_tasks, d)
        task_vecs_over_all_time: Task vectors to decompose, shape (k, seq_len, d)
        alpha: Weight for orthogonal component, controls OOD enhancement
        chunk_size: Batch size for memory-efficient processing
    
    Returns:
        ood_enhanced_task_vectors: Enhanced vectors, shape (k, seq_len, d)
        task_directions: Projections onto task span, shape (k, seq_len, d)
        ortho_directions: Projections onto orthogonal complement, shape (k, seq_len, d)
    """
    eps = 1e-12
    device = task_vecs.device
    dtype = task_vecs.dtype

    task_vecs_over_all_time = task_vecs_over_all_time.to(device=device, dtype=dtype)

    k, seq_len, d = task_vecs_over_all_time.shape
    num_tasks = task_vecs.shape[0]

    # Design matrix X: (d, num_tasks or num_tasks-1)
    X = task_vecs.T.contiguous()  # (d, num_tasks)
    X = X[:, :-1]  # (d, num_tasks-1)

    # Precompute stats used in R^2 denominator per chunk: we still need Y means per chunk
    # We'll compute them on-the-fly to avoid storing big intermediates.

    task_directions = torch.zeros((k, seq_len, d), dtype=dtype)
    ortho_directions = torch.zeros((k, seq_len, d), dtype=dtype)

    with torch.no_grad():
        for t in range(seq_len):
            # Y_full: (d, k)  -- do not clone to avoid extra memory; slice columns per chunk
            Y_full = task_vecs_over_all_time[:, t, :].T  # (d, k)

            # Process k-dimension in chunks
            for start in range(0, k, chunk_size):
                end = min(start + chunk_size, k)

                Y = Y_full[:, start:end]   # (d, k_chunk)
                # Solve X W = Y for W, least squares (stable)
                # W: (num_tasks-1 or num_tasks, k_chunk)
                W = torch.linalg.lstsq(X, Y).solution

                task_dir = X @ W                      # (d, k_chunk)
                ortho_dir = Y - task_dir              # (d, k_chunk)
                task_directions[start:end, t, :] = task_dir.T.cpu() # (k_chunk, d)
                ortho_directions[start:end, t, :] = ortho_dir.T.cpu()

    # origin_norm = torch.norm(task_vecs_over_all_time, dim=-1).cpu() # (k, seq_len)
    assert torch.allclose(task_vecs_over_all_time.cpu(), task_directions + ortho_directions, atol=1e-5)
    ood_enhanced_task_vectors = 2 * (task_directions + alpha * ortho_directions) / (1+alpha)
    # ood_enhanced_task_vectors /= (ood_enhanced_task_vectors.norm(dim=-1, keepdim=True) + eps)
    #ood_enhanced_task_vectors *= origin_norm.unsqueeze(-1) # (k, seq_len, d), keep original norm while enhancing OOD direction

    return ood_enhanced_task_vectors, task_directions, ortho_directions


def estimate_lambda_super_fast(task_vecs, last_eval_task_vecs, 
                               is_zero_mean=True, ridge=0.0, compute_r2=False):
    """
    Fast estimation of lambda weights by solving least squares regression.
    
    Fits each target task vector as a linear combination of reference task vectors.
    Supports optional Ridge regularization and R² computation.
    
    Args:
        task_vecs: Reference task vectors, shape (num_tasks, d)
        last_eval_task_vecs: Target task vectors to fit, shape (k, d)
        is_zero_mean: If True, uses zero-mean constraint trick
        ridge: Ridge regularization strength (0 = no regularization)
        compute_r2: Whether to compute R² goodness-of-fit scores
    
    Returns:
        lambdas: Coefficient weights, shape (k, num_tasks)
        r2_scores: Optional R² scores, shape (k,) if compute_r2=True, else None
    """
    assert task_vecs.ndim == 2, "task_vecs should be of shape (num_tasks, d)"
    assert last_eval_task_vecs.ndim == 2, "last_eval_task_vecs should be of shape (k, d)"

    device = task_vecs.device
    dtype  = task_vecs.dtype

    k, d = last_eval_task_vecs.shape
    num_tasks = task_vecs.shape[0]

    # Design matrix X: (d, num_tasks)  -> fit on first (num_tasks-1) cols if zero-mean trick
    X = task_vecs.T  # (d, num_tasks)
    if is_zero_mean:
        X_fit = X[:, :-1]       # (d, num_tasks-1)
        m = num_tasks - 1
    else:
        X_fit = X
        m = num_tasks

    # Targets stacked as columns: Y in R^{d x k} (use the last time step)
    Y = last_eval_task_vecs.transpose(0, 1).to(device=device, dtype=dtype)  # (d, k)

    # Solve for W in R^{m x k}: minimize ||X_fit W - Y||_F
    if ridge == 0.0:
        # Robust, batched over RHS
        W = torch.linalg.lstsq(X_fit, Y).solution  # (m, k)
    else:
        XtX = X_fit.T @ X_fit                      # (m, m)
        XtY = X_fit.T @ Y                          # (m, k)
        XtX = XtX + ridge * torch.eye(m, device=device, dtype=dtype)
        W = torch.linalg.solve(XtX, XtY)          # (m, k)

    # Build lambdas: (k, num_tasks)
    if is_zero_mean:
        lambdas = torch.zeros((num_tasks, k), device=device, dtype=dtype)
        lambdas[:-1, :] = W
        # Make sums = 1 by adding the same offset to all entries (matches your original logic)
        shift = (1.0 - W.sum(dim=0, keepdim=True)) / num_tasks  # (1, k)
        lambdas += shift
        lambdas = lambdas.T  # (k, num_tasks)
    else:
        lambdas = W.T  # (k, num_tasks)

    if not compute_r2:
        return lambdas, None

    # R^2 (vectorized)
    Y_pred = X_fit @ W                          # (d, k)
    resid  = Y - Y_pred                         # (d, k)
    ss_res = (resid**2).sum(dim=0)              # (k,)
    Y_mean = Y.mean(dim=0, keepdim=True)
    ss_tot = ((Y - Y_mean)**2).sum(dim=0)       # (k,)
    r2_scores = torch.where(ss_tot > 0, 1.0 - ss_res / ss_tot, torch.tensor(float("nan"), device=device, dtype=dtype))

    return lambdas, r2_scores





def evaluate_and_estimate_lambdas(
    model,
    train_task,
    task_vectors,
    config,
    K=3000,
    layer_index=3,
    weight_seed=None,
    weight_scale=0.2,
    mean_zero=True,
    global_mean=None,
    convex=False,
    orthogonal_offset=0.0,
    compute_r2=True,
):
    """
    Generate evaluation tasks as linear combinations of anchor tasks, compute task vectors, 
    and estimate lambda weights and R^2 scores.

    Args:
        model: The neural model used for computing hidden representations.
        train_task: Task object containing `task_pool` (shape: [3, d, 1] or [3, d]).
        task_vectors: Tensor of shape (num_tasks, num_layers, d) containing anchor task vectors.
        global_mean: Tensor of shape (d,) to center hidden representations.
        config: Hydra config object or dict containing task configuration.
        K (int): Number of evaluation tasks.
        layer_index (int): Layer index to extract representations from.
        weight_seed (int or None): Optional random seed for reproducibility.

    Returns:
        lambdas (Tensor): shape (K, 3), estimated λ per evaluation task.
        r2_scores (Tensor): shape (K,), R² score of each fit.
        eval_task_vectors (Tensor): shape (K, d), representation of each evaluation task.
        eval_task (Task): task object containing the evaluation task pool.
    """
    if weight_seed is not None: torch.manual_seed(weight_seed)

    d = config.task.n_dims

    anchor_pool = train_task.task_pool.squeeze(-1)  # shape (3, d)

    # Sample near convex weights (not normalized) for eval task pool
    if convex:
        weights = torch.rand(K, 3)  # shape (K, 3)  
        weights = weights / weights.sum(dim=1, keepdim=True)
        weights += weight_scale * torch.randn(K, 3) # add noise
        weights = torch.cat([torch.eye(3), weights], dim=0)  # shape (3+K, 3)
        # Create eval task pool: linear combinations of anchor tasks
        eval_task_pool = weights @ anchor_pool  # shape (3+K, d)
    else:
        eval_task_pool, weights = sample_union_unit_balls_affine_span_with_weights(anchor_pool, K, weight_scale, orthogonal_offset)

    # Clone config and prepare eval task
    eval_config = config.copy() if isinstance(config, dict) else config
    eval_config.task.n_tasks = K
    eval_task = get_task(**eval_config["task"])
    eval_task.batch_size = 256

    
    eval_task.task_pool = eval_task_pool.unsqueeze(-1)  # shape (K, d, 1)

    # Compute eval task vectors
    eval_hiddens, _ = compute_hiddens(eval_config, model, eval_task, layer_index=layer_index)
    eval_global_mean = eval_hiddens.mean(dim=(0,2))
    # print((eval_global_mean - global_mean).norm(dim=-1) / global_mean.norm(dim=-1))
    if mean_zero:
        eval_task_vectors = eval_hiddens - eval_global_mean.unsqueeze(0).unsqueeze(2)  # center
        eval_task_vectors = eval_task_vectors.mean(dim=-2)  # shape (K, d)
    else:
        eval_task_vectors = eval_hiddens.mean(dim=-2)  # shape (K, d)

    # Estimate lambdas and R²
    if mean_zero:
        lambdas, r2_scores = estimate_lambda_super_fast(task_vectors[:, -1], eval_task_vectors, compute_r2=compute_r2)
    else:
        assert global_mean is not None, "global_mean must be provided for non-zero-mean fitting"
        lambdas, r2_scores = estimate_lambda_super_fast(global_mean[-1] + task_vectors[:, -1], eval_task_vectors, is_zero_mean=False, compute_r2=compute_r2)

    return lambdas, r2_scores, eval_task_vectors, eval_task, weights


#######################
# Attention Extraction Functions #
#######################



def get_attn(model, data, target):
    """
    Extract attention weights from all layers of the model.
    
    Registers forward hooks to capture attention matrices from each layer
    during forward pass. Returns attention for first sample in batch.
    
    Args:
        model: Transformer model with layers
        data: Input data, shape (B, T)
        target: Target data for computing context
    
    Returns:
        attns: Dictionary mapping layer indices to attention weights, shape varies by layer
    """
    attns = {}
    
    def create_hook_fn(layer_idx):
        def hook_fn(module, input, output):
            # Get the input to the attention module
            x = input[0]
            batch_size, seq_len, _ = x.size()
            
            # Compute Q, K, V
            Q = module.query(x).view(batch_size, seq_len, module.n_head, module.head_dim).transpose(1,2)
            K = module.key(x).view(batch_size, seq_len, module.n_head, module.head_dim).transpose(1,2)
            
            # Apply rotary embeddings
            Q, K = apply_rotary_emb(Q.transpose(1, 2), K.transpose(1, 2), freqs_cis=module.freqs_cis[:seq_len])
            Q, K = Q.transpose(1, 2), K.transpose(1, 2)
            
            # Compute attention weights
            scale = 1.0 / (module.head_dim ** 0.5)
            attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
            
            # Apply causal mask
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            attn_weights.masked_fill_(mask, float('-inf'))
            
            # Softmax to get attention probabilities
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            attns[layer_idx] = attn_weights.detach().squeeze(0)
        
        return hook_fn
    
    hook_handles = []
    num_layers = len(model.transformer.blocks)
    
    for l in range(num_layers):
        handle = model.transformer.blocks[l].attn_block.attn.register_forward_hook(create_hook_fn(l))
        hook_handles.append(handle)
    
    with torch.no_grad():
        _ = model(data, target)
    
    # Clean up hooks
    for handle in hook_handles:
        handle.remove()
    
    return attns

def get_filtered_attn_output_at_layer(model, data, target, l, task_pos=-1):
    """
    Extract filtered attention output from a specific layer.
    
    Computes attention output with zeroed attention from task position to itself
    and previous position. Useful for ablation studies.
    
    Args:
        model: Transformer model
        data: Input data, shape (B, T)
        target: Target data
        l: Layer index
        task_pos: Task position index to zero out
    
    Returns:
        filtered_output: Dictionary with filtered attention output for the layer
    """
    filtered_output = {}
    
    def hook_fn(module, input, output):
        # Get the input to the attention module
        x = input[0]
        batch_size, seq_len, _ = x.size()
        
        # Compute Q, K, V
        Q = module.query(x).view(batch_size, seq_len, module.n_head, module.head_dim).transpose(1,2)
        K = module.key(x).view(batch_size, seq_len, module.n_head, module.head_dim).transpose(1,2)
        V = module.value(x).view(batch_size, seq_len, module.n_head, module.head_dim).transpose(1,2)
        
        # Apply rotary embeddings
        Q, K = apply_rotary_emb(Q.transpose(1, 2), K.transpose(1, 2), freqs_cis=module.freqs_cis[:seq_len])
        Q, K = Q.transpose(1, 2), K.transpose(1, 2)
        
        # Compute attention weights
        scale = 1.0 / (module.head_dim ** 0.5)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # Apply causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_weights.masked_fill_(mask, float('-inf'))
        
        # Softmax to get attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_weights[:, :, task_pos, task_pos] = 0  # Zero out attention to the task position
        attn_weights[:, :, task_pos, task_pos-1] = 0  # Zero out attention to the task position
        out = attn_weights @ V  # (B, H, T, D)
        out = out.transpose(1,2).contiguous().view(batch_size,seq_len,-1) # (B,T,C)
        out = module.out(out)
        filtered_output[l] = out.detach().squeeze(0)
    
    handle = model.transformer.blocks[l].attn_block.attn.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(data, target)
    
    # Clean up hooks
    handle.remove()
    
    return filtered_output

# View attention map
def view_attn(train_task, model):
    """
    Create an interactive widget to visualize attention maps.
    
    Samples data from first task and displays attention patterns
    in a Jupyter widget for interactive exploration.
    
    Args:
        train_task: Training task object
        model: Transformer model
    
    Returns:
        Interactive widget showing attention heatmaps
    """
    train_task.batch_size = 1
    demo_data0, demo_target = train_task.sample_from_task(train_task.task_pool[0], step=2)
    attns = get_attn(model, demo_data0, demo_target)
    cap = 100
    attns_capped = {layer_key: tensor[:, :cap, :cap] for layer_key, tensor in attns.items()}
    
    widget = visualize_attention(attns_capped, mode='widget')
    return widget

def get_attn_mean_var(train_task, model):
    """
    Compute mean and variance statistics of attention weights.
    
    Analyzes attention patterns by computing norms and variances
    to understand attention concentration across layers.
    
    Args:
        train_task: Training task object
        model: Transformer model
    
    Returns:
        attn_vars: Dictionary mapping layer indices to variance statistics
        attn_means: Dictionary mapping layer indices to mean statistics
    """
    train_task.batch_size = 256
    demo_data, demo_target = train_task.sample_from_task(train_task.task_pool[1], step=2)
    attns = get_attn(model, demo_data, demo_target)
    attn_means = {layer_key: tensor[:, :, 1::3].mean(dim=0).norm(dim=(-1,-2)).square().cpu().item() for layer_key, tensor in attns.items()}
    attn_vars = {layer_key: tensor[:, :, 1::3].var(dim=0).sum(dim=(-1,-2)).cpu().item() for layer_key, tensor in attns.items()}
    return attn_vars, attn_means



def get_path_to_exp_dir(exp_name):
    """
    Get the path to an experiment's directory.
    
    Args:
        exp_name: Name of the experiment
    
    Returns:
        Path to the experiment directory
    """
    work_dir = os.path.join("..", "results", "linear")
    exp_dir = os.path.join(work_dir, exp_name)
    return exp_dir

def load_model_task_config(exp_name):
    """
    Load a trained model, task, and configuration from an experiment.
    
    Args:
        exp_name: Name of the experiment
    
    Returns:
        model: Trained model loaded from checkpoint
        train_task: Task object used for training
        config: Configuration dictionary
    """
    exp_dir = get_path_to_exp_dir(exp_name)
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "r") as f: config_dict = json.load(f)
    
    config = ConfigDict(config_dict)
    checkpoint_path = os.path.join(exp_dir, "checkpoint.pt")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    data_type = torch.float
    model = get_model(**config["model"], dtype=data_type)
    model.load_state_dict(checkpoint["model"])
    train_task = get_task(**config["task"], device=config.device)
    return model, train_task, config



def compare_check_point_task_vector_to_final(
    config,
    train_task,
    final_task_vectors,
    checkpoint_path: str,
    layer_index: int = 3,
):
    """
    Compare task vectors from a checkpoint to final task vectors.
    
    Loads a model checkpoint and computes differences between checkpoint
    task vectors and final trained task vectors (measured as normalized distance).
    
    Args:
        config: Configuration object
        train_task: Training task object
        final_task_vectors: Final task vectors, shape (n_tasks, n_embd)
        checkpoint_path: Path to checkpoint file
        layer_index: Layer to extract task vectors from
    
    Returns:
        task_vec_diff_norms: Normalized differences, shape (n_tasks,)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model = get_model(**config["model"], dtype=torch.float32)
    model.load_state_dict(checkpoint["model"])

    # Extract task vectors
    _, task_vectors = get_task_vector_from_hidden(config, model, train_task, layer_index=layer_index, 
                                                  compute_mean=True, return_final=True) # (n_tasks, n_embd)

    # Compare with final task vectors
    task_vec_diff_norms = (final_task_vectors / final_task_vectors.norm(dim=-1, keepdim=True) - task_vectors / task_vectors.norm(dim=-1, keepdim=True)).norm(dim=-1) 

    return task_vec_diff_norms


def get_checkpoint_files(exp_name):
    """
    Get list of checkpoint files in an experiment directory.
    
    Args:
        exp_name: Name of the experiment
    
    Returns:
        List of checkpoint filenames
    """
    exp_dir = get_path_to_exp_dir(exp_name)
    checkpoint_files = [f for f in os.listdir(exp_dir) if f.startswith("model_") and f.endswith(".pt")]
    return checkpoint_files

def load_checkpoint(exp_name, checkpoint_file):
    """
    Load a checkpoint from the experiment directory.
    
    Args:
        exp_name: Name of the experiment
        checkpoint_file: Name of the checkpoint file to load
    
    Returns:
        model: The model loaded from the checkpoint
        config: The configuration used for the model
        train_task: The training task object
    """
    exp_dir = get_path_to_exp_dir(exp_name)
    checkpoint_path = os.path.join(exp_dir, checkpoint_file)
    _, train_task, config = load_model_task_config(exp_name)

    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model = get_model(**config["model"], dtype=torch.float32)
    model.load_state_dict(checkpoint["model"])
    model = model.to("cuda")
    return model, config, train_task


def compare_all_check_point_task_vector_to_final(exp_name, layer_index=3):
    """
    Compare task vectors across all checkpoints to final task vectors.
    
    Loads all checkpoint files and computes differences for each checkpoint,
    allowing analysis of how task vectors evolve during training.
    
    Args:
        exp_name: Name of the experiment
        layer_index: Layer to extract task vectors from
    
    Returns:
        diff_means: Dictionary mapping step numbers to mean differences per task
    """
    exp_dir = get_path_to_exp_dir(exp_name)
    final_model, train_task, config = load_model_task_config(exp_name)
    _, final_task_vectors = get_task_vector_from_hidden(config, final_model, train_task, layer_index, compute_mean=True, return_final=True) # (n_tasks, n_embd)
    checkpoint_files = [f for f in os.listdir(exp_dir) if f.startswith("model_") and f.endswith(".pt")]
    diff_means = {}
    # diff_stds = {}
    for checkpoint_file in tqdm(checkpoint_files, desc="Processing checkpoints"):
        checkpoint_path = os.path.join(exp_dir, checkpoint_file)
        match = re.search(r"model_(\d+)\.pt", checkpoint_file)
        if match:
            step = int(match.group(1))
            diff_means[step] = compare_check_point_task_vector_to_final(config, train_task, final_task_vectors, checkpoint_path, layer_index)
        else:
            continue
    return diff_means #, diff_stds


def _to_np(x):
    """
    Convert tensor to numpy array, handling both torch tensors and existing arrays.
    
    Args:
        x: Input tensor or array
    
    Returns:
        Numpy array
    """
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def plot_checkpoint_diffs(exp_name, layer_index, tasks=None, title=None, yaxis_title="Mean difference"):
    """
    Plot task vector differences between checkpoints and final model.
    
    Visualizes how task vectors evolve during training by comparing
    checkpoint representations to the final trained state.
    
    Args:
        exp_name: Name of the experiment
        layer_index: Layer to extract task vectors from
        tasks: List of task indices to plot; default = all tasks
        title: Custom plot title
        yaxis_title: Y-axis label
    """
    diff_means = compare_all_check_point_task_vector_to_final(exp_name, layer_index)

    # Steps in ascending order
    steps = sorted(diff_means.keys())
    # Stack into (n_steps, n_tasks)
    means = np.stack([_to_np(diff_means[s]) for s in steps], axis=0)
    #stds  = np.stack([_to_np(diff_stds[s])  for s in steps], axis=0)

    n_tasks = means.shape[1]
    if tasks is None: tasks = list(range(n_tasks))

    fig = go.Figure()
    x = np.array(steps, dtype=int)

    for t in tasks:
        m = means[:, t]
        lg = f"Task {t}"

        fig.add_trace(go.Scatter(
            x=x, y=m, mode="lines", name=lg, legendgroup=lg,
            hovertemplate="step=%{x}<br>mean=%{y:.4g}<extra>" + lg + "</extra>"
        ))

    fig.update_layout(
        title=title or "Checkpoint vs Final: mean per task",
        xaxis_title="step",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        template="plotly_white"
    )
    fig.show()

def get_mse(model, eval_task, train_task, n_points, step: int = 1):
    """
    Compute MSE loss for model predictions compared to oracle baselines.
    
    Evaluates model performance on evaluation tasks and compares to
    oracle predictions from individual training tasks.
    
    Args:
        model: Trained model
        eval_task: Evaluation task object
        train_task: Training task object with task pool
        n_points: Number of points in evaluation sequence
        step: Step size for evaluation
    
    Returns:
        mean_losses: Mean losses per task, shape (n_tasks,)
        oracle_losses: Oracle baseline losses per task, shape (n_tasks, n_points, n_tasks)
    """
    def compute_metrics(k):
        _data, _target = eval_task.sample_from_task(eval_task.task_pool[k], step=step) # (batch, n_points, n_dims)
        tasks = train_task.task_pool.squeeze(-1) # (n_tasks, n_dims)
        oracle_targets = (_data @ tasks.transpose(0,1)).squeeze(-1)  # (batch, n_points, n_task)

        with torch.no_grad():
            preds = model(_data, _target) # (batch, n_points)
        loss = ((preds - _target.to(preds.device))**2).mean(dim=0) # n_points
        oracle_loss = ((preds.unsqueeze(-1) - oracle_targets.to(preds.device))**2).mean(dim=0) # (n_points, n_task)

        return loss, oracle_loss

    n_tasks = len(train_task.task_pool)
    n_eval_tasks = len(eval_task.task_pool) 
    oracle_results = torch.empty((n_eval_tasks, n_points, n_tasks), device=eval_task.device)
    results = torch.empty((n_eval_tasks, n_points), device=eval_task.device)
    
    for k in range(n_eval_tasks):
        results[k], oracle_results[k] = compute_metrics(k) 

    return results.cpu().numpy(), oracle_results.cpu().numpy()


def get_mse_last(model, eval_task, train_task, step: int = 1):
    """
    Compute MSE loss on the last position only.
    
    Similar to get_mse but only evaluates performance on the final token,
    useful for analyzing convergence behavior.
    
    Args:
        model: Trained model
        eval_task: Evaluation task object
        train_task: Training task object with task pool
        step: Step size for evaluation
    
    Returns:
        results: Last position losses per task, shape (n_eval_tasks,)
        oracle_results: Oracle losses per task, shape (n_eval_tasks, n_tasks)
    """
    def compute_metrics(k):
        _data, _target = eval_task.sample_from_task(eval_task.task_pool[k], step=step)
        last_data = _data[:,-1] # (batch, n_dims)
        tasks = train_task.task_pool.squeeze(-1) # (n_tasks, n_dims)
        oracle_targets = (last_data @ tasks.transpose(0,1)).squeeze(-1)  # (batch, n_task)

        with torch.no_grad():
            preds = model(_data, _target) # (batch, n_points)
        loss = ((preds[:,-1:] - _target[:,-1:].to(preds.device))**2).mean(dim=0) # 1
        oracle_loss = ((preds[:,-1:] - oracle_targets.to(preds.device))**2).mean(dim=0) # n_task

        return loss.cpu().item(), oracle_loss.cpu().numpy()

    n_tasks = len(train_task.task_pool)
    n_eval_tasks = len(eval_task.task_pool)
    oracle_results = np.zeros((n_eval_tasks, n_tasks))
    results = np.zeros(n_eval_tasks)
    
    for k in range(n_eval_tasks):
        results[k], oracle_results[k] = compute_metrics(k) 

    return results, oracle_results









def plot_lambda_projection_with_slider(
    model, eval_task, train_task, r2_score, lambdas, weights, n_points, title=None,
    slider_mode="log",            # "log" or "linear"
    log_gamma=5.0,                # curvature; bigger => denser at small t
    max_slider_steps=256          # cap number of slider steps (dedupbed later)
):
    """
    λ maps to marker fill color (R,G,B).
    R² maps to marker size & opacity (bigger/darker => higher R²).
    MSE maps to outline (stroke) width (thicker => higher MSE).

    Shapes:
      r2_score: (K,) or (K, T)
      lambdas:  (K, T, 3) or (K, 3) [treated as T=1]
      weights:  (K, 3) or (K, T, 3)
    """
    # ---- Metrics ----
    results, oracle_results = get_mse(model, eval_task, train_task, n_points)  # results ≈ MSE

    # ---- Convert to NumPy ----
    to_np = lambda x: x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    anchor_np = to_np(train_task.task_pool).squeeze(-1)  # (3, d)
    eval_np   = to_np(eval_task.task_pool).squeeze(-1)   # (K, d)
    lambda_np = to_np(lambdas)
    weights_np = None if weights is None else to_np(weights)
    r2_np = None if r2_score is None else to_np(r2_score)
    mse_np = None if results is None else to_np(results)

    # Ensure lambdas has shape (K, T, 3), even if old (K, 3)
    if lambda_np.ndim == 2 and lambda_np.shape[1] == 3:
        lambda_np = lambda_np[:, None, :]  # (K, 1, 3)

    K, T = lambda_np.shape[0], lambda_np.shape[1]

    # ---- PCA on eval points ----
    pca = PCA(n_components=2)
    eval_2d = pca.fit_transform(eval_np)        # (K, 2)
    anchor_2d = eval_2d[:3]                     # first 3 as anchors

    # ---- Utilities ----
    def compute_circumcenter(p1, p2, p3):
        A = np.stack([p2 - p1, p3 - p1])  # (2, 2)
        b = np.array([
            np.dot(p2, p2) - np.dot(p1, p1),
            np.dot(p3, p3) - np.dot(p1, p1)
        ]) / 2.0
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return x  # (2,)

    def add_half_bisector(fig, a, b, center, length=5.0):
        direction = b - a
        normal = np.array([-direction[1], direction[0]])
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        midpoint = 0.5 * (a + b)
        if np.dot(normal, midpoint - center) < 0:
            normal = -normal
        line_pts = center[None, :] + np.linspace(0, 1, 2)[:, None] * normal[None, :] * length
        fig.add_trace(go.Scatter(
            x=line_pts[:, 0], y=line_pts[:, 1],
            mode='lines', line=dict(dash='dot', color='gray'),
            showlegend=False
        ))

    def select_for_t(arr, t):
        """Slice helper for arrays that might be (K,), (K,3), (K,T), or (K,T,3)."""
        if arr is None:
            return None
        if arr.ndim == 3:      # (K, T, 3)
            return arr[:, t, :]
        if arr.ndim == 2:
            if arr.shape[1] == T:   # (K, T)
                return arr[:, t]
            return arr               # assume (K, 3)
        return arr 
    
    def _build_t_schedule(T, mode="log", max_steps=160, gamma=5.0):
        """
        Returns a sorted, unique array of t indices in [0, T-1].
        mode="log": denser near small t using an exponential mapping.
        mode="linear": all t (or capped to max_steps if T is huge).
        """
        if T <= 1:
            return np.array([0], dtype=int)

        if mode == "linear":
            if max_steps and T > max_steps:
                # uniform thin-out if there are too many steps
                idx = np.linspace(0, T - 1, max_steps)
                t_vals = np.unique(np.round(idx).astype(int))
            else:
                t_vals = np.arange(T, dtype=int)
        else:
            # Exponential mapping in [0,1] -> [0,T-1]
            S = min(max_steps, T) if max_steps else T
            u = np.linspace(0.0, 1.0, S)
            x = (np.exp(gamma * u) - 1.0) / (np.exp(gamma) - 1.0)  # in [0,1]
            t_vals = np.clip(np.round(x * (T - 1)).astype(int), 0, T - 1)
            t_vals = np.unique(t_vals)  # remove duplicates from rounding

        # Ensure endpoints present
        if t_vals[0] != 0:
            t_vals = np.insert(t_vals, 0, 0)
        if t_vals[-1] != T - 1:
            t_vals = np.append(t_vals, T - 1)
        return t_vals                  # (K,)

    center_2d = compute_circumcenter(anchor_2d[0], anchor_2d[1], anchor_2d[2])

    # ---- Global normalization (across all t) ----
    # R² range
    if r2_np is not None:
        r2_all = r2_np.reshape(-1)
        r2_valid = np.isfinite(r2_all)
        r2_min = np.nanmin(r2_all[r2_valid]) if r2_valid.any() else 0.0
        r2_max = np.nanmax(r2_all[r2_valid]) if r2_valid.any() else 1.0
    else:
        r2_min, r2_max = 0.0, 1.0

    def _normalize01(v, vmin, vmax):
        v = np.asarray(v, dtype=float)
        out = (v - vmin) / (vmax - vmin + 1e-12)
        out = np.clip(out, 0.0, 1.0)
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        return out

    def style_from_r2(r2_t):
        """Map R² -> marker size & opacity."""
        if r2_t is None:
            return np.full((K,), 6.0), np.full((K,), 0.8)
        r = _normalize01(r2_t, r2_min, r2_max)    # [0,1]
        size = 2.0 + r * (15.0 - 2.0)             # 2..15
        opacity = 0.3 + r * (1.0 - 0.3)           # 0.3..1.0
        return size, opacity


    # ---- Per-t styling ----
    def colors_hover_style_for_t(t):
        lam_t = lambda_np[:, t, :]                 # (K, 3)
        w_t   = select_for_t(weights_np, t)        # (K, 3) or (K,)
        r2_t  = select_for_t(r2_np, t)             # (K,) or None
        mse_t = select_for_t(mse_np, t)            # (K,) or None
        try:
            oracle_t = select_for_t(to_np(oracle_results), t)
        except Exception:
            oracle_t = to_np(oracle_results)

        # λ -> fill color (RGB)
        colors = lam_t @ np.array([[255, 0, 0],
                                   [0, 255, 0],
                                   [0, 0, 255]])
        colors = np.clip(colors, 0, 255).astype(int)
        colors_hex = [f"rgb({r},{g},{b})" for r, g, b in colors]

        # Hover text
        hover = []
        for k in range(K):
            parts = [f"λ={np.round(lam_t[k], 3)}"]
            if isinstance(w_t, np.ndarray) and w_t.ndim == 2 and w_t.shape[1] == 3:
                parts.append(f"w={np.round(w_t[k], 3)}")
            elif isinstance(w_t, np.ndarray) and w_t.ndim == 1:
                parts.append(f"w={np.round(w_t[k], 3)}")

            # MSEs
            try:
                mse_val = float(mse_t[k]) if mse_t is not None else np.nan
            except Exception:
                mse_val = float(mse_np[k]) if mse_np is not None else np.nan
            parts.append(f"MSE: {np.round(mse_val, 4)}")

            # Oracle MSE
            try:
                parts.append(f"Oracle MSE: {np.round(oracle_t[k], 4)}")
            except Exception:
                parts.append(f"Oracle MSE: {np.round(oracle_t[k], 4)}")

            # R²
            if r2_t is not None:
                parts.append(f"R²: {r2_t[k]:.3f}")

            hover.append("<br>".join(parts))

        # Styles
        size, opacity = style_from_r2(r2_t)
        return colors_hex, hover, size, opacity

    # ---- Initial figure (t=0) ----
    t0 = 0
    colors_hex_0, hover_0, size_0, opacity_0 = colors_hover_style_for_t(t0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eval_2d[:, 0], y=eval_2d[:, 1],
        mode='markers',
        marker=dict(
            size=size_0,
            color=colors_hex_0,          # λ-based fill
            opacity=opacity_0,
            line=dict(color="black", width=0.2),  # MSE -> outline width
        ),
        name="Eval Points",
        hoverinfo='text',
        text=hover_0
    ))

    # Static traces: anchors, center, bisectors
    fig.add_trace(go.Scatter(
        x=anchor_2d[:, 0], y=anchor_2d[:, 1],
        mode='markers+text',
        marker=dict(size=10, color='black', symbol='x', opacity=0.6),
        text=[f"w{i}" for i in range(anchor_2d.shape[0])],
        textposition='top center',
        name="Anchor Points"
    ))

    center_2d = center_2d
    fig.add_trace(go.Scatter(
        x=[center_2d[0]], y=[center_2d[1]],
        mode='markers+text',
        marker=dict(size=12, color='black', symbol='x', opacity=0.6),
        text=["Equidistant center"],
        textposition='bottom right',
        name="Equidistant Center"
    ))

    add_half_bisector(fig, anchor_2d[0], anchor_2d[1], center_2d)
    add_half_bisector(fig, anchor_2d[1], anchor_2d[2], center_2d)
    add_half_bisector(fig, anchor_2d[0], anchor_2d[2], center_2d)

    # ---- Frames for each t ----
    # Use log/linear schedule for t
    t_schedule = _build_t_schedule(T, mode=slider_mode, max_steps=max_slider_steps, gamma=log_gamma)

    frames = []
    for t in t_schedule:
        t_int = int(t)
        colors_hex_t, hover_t, size_t, opacity_t = colors_hover_style_for_t(t_int)
        frames.append(go.Frame(
            name=f"t={t_int}",
            data=[
                go.Scatter(
                    x=eval_2d[:, 0], y=eval_2d[:, 1],
                    mode='markers',
                    marker=dict(
                        size=size_t,
                        color=colors_hex_t,
                        opacity=opacity_t,
                        line=dict(color="black", width=0.2),
                    ),
                    hoverinfo='text',
                    text=hover_t,
                    name="Eval Points",
                )
            ]
        ))
    fig.frames = frames

    # Set initial frame to first in schedule
    t0 = int(t_schedule[0])
    colors_hex_0, hover_0, size_0, opacity_0 = colors_hover_style_for_t(t0)
    fig.data[0].update(marker=dict(size=size_0, color=colors_hex_0, opacity=opacity_0,
                                line=dict(color="black", width=0.2)),
                    text=hover_0)

    # ---- Slider ----
    slider_steps = []
    for t in t_schedule:
        t_int = int(t)
        slider_steps.append({
            "method": "animate",
            "label": str(t_int),
            "args": [[f"t={t_int}"],
                    {"mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0}}],
        })

    sliders = [{
        "active": 0,
        "currentvalue": {"prefix": "t = "},
        "pad": {"t": 8, "b": 0},
        "steps": slider_steps,
        "len": 0.95,
        "x": 0.025, "xanchor": "left",
        "y": -0.10, "yanchor": "top",
    }]
    fig.update_layout(sliders=sliders)

    # Annotation describing mappings
    r2_min_text = f"{r2_min:.3f}" if np.isfinite(r2_min) else "NA"
    r2_max_text = f"{r2_max:.3f}" if np.isfinite(r2_max) else "NA"

    title_main = title or "λ→RGB fill, R²→size & opacity, MSE→outline width"
    subtitle = (f"R² (global): [{r2_min_text}, {r2_max_text}]")

    fig.update_layout(
        title=dict(
            text=title_main + "<br><sup>" + subtitle + "</sup>",
            x=0.48, xanchor="center",
            y=0.91, yanchor="top",        # keep the title just under the top margin
            pad=dict(b=10)                # space between title and plot
        ),
        xaxis_title="PC 1",
        yaxis_title="PC 2",
        width=760,
        height=760,
        showlegend=True,
        sliders=sliders,
        margin=dict(l=60, r=20, t=120, b=110)  # enough top margin so nothing collides
    )

    # keep this after layout
    fig.update_yaxes(scaleanchor="x", scaleratio=1)


    fig.show()



def plot_lambda_projection_norm(
    model, eval_task, train_task,
    task_norm,                 # (K,) or (K,T)
    ortho_norm,                # (K,) or (K,T)
    weights,                   # (K,3) or (K,T,3) or None
    lambdas,                   # (K,3) or (K,T,3)  (hover only)
    n_points,
    title=None,
    slider_mode="log",
    log_gamma=5.0,
    max_slider_steps=256
):
    """
    Plot task vector projections showing norm decompositions over time.
    
    Visualizes task vectors in 2D PCA space with multiple views showing:
    - Total/orthogonal/task space norms as marker sizes
    - Time derivatives as colors (using asinh compression)
    - MSE analysis view
    
    Args:
        model: Trained model for MSE computation
        eval_task: Evaluation task object
        train_task: Training task with anchor points
        task_norm: Norms in task space, shape (K,) or (K,T)
        ortho_norm: Norms in orthogonal space, shape (K,) or (K,T)
        weights: Task weights for reference
        lambdas: Lambda coefficients for hover text
        n_points: Number of points for MSE evaluation
        title: Optional plot title
        slider_mode: "log" or "linear" for time slider
        log_gamma: Curvature parameter for log schedule
        max_slider_steps: Maximum slider steps
    """
    # ---- Metrics (for hover + MSE view) ----
    results, oracle_results = get_mse(model, eval_task, train_task, n_points)

    # ---- Convert to NumPy ----
    def to_np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    anchor_np = to_np(train_task.task_pool).squeeze(-1)  # (3, d)
    eval_np   = to_np(eval_task.task_pool).squeeze(-1)   # (K, d)
    weights_np = None if weights is None else to_np(weights)
    task_norm_np = None if task_norm is None else to_np(task_norm)
    ortho_norm_np = None if ortho_norm is None else to_np(ortho_norm)
    mse_np = None if results is None else to_np(results)
    oracle_np = None if oracle_results is None else to_np(oracle_results)

    lambda_np = None if lambdas is None else to_np(lambdas)
    if lambda_np is not None and lambda_np.ndim == 2 and lambda_np.shape[1] == 3:
        lambda_np = lambda_np[:, None, :]  # (K,1,3)

    # ---- Determine K, T ----
    if lambda_np is not None:
        K, T = lambda_np.shape[0], lambda_np.shape[1]
    elif task_norm_np is not None and task_norm_np.ndim == 2:
        K, T = task_norm_np.shape
    elif ortho_norm_np is not None and ortho_norm_np.ndim == 2:
        K, T = ortho_norm_np.shape
    else:
        K, T = eval_np.shape[0], 1

    # ---- PCA ----
    pca = PCA(n_components=2)
    eval_2d = pca.fit_transform(eval_np)
    anchor_2d = eval_2d[:3]

    # ---- Geometry helpers ----
    def compute_circumcenter(p1, p2, p3):
        A = np.stack([p2 - p1, p3 - p1])
        b = np.array([np.dot(p2, p2) - np.dot(p1, p1),
                      np.dot(p3, p3) - np.dot(p1, p1)]) / 2.0
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return x

    def add_half_bisector(fig, a, b, center, length=5.0):
        d = b - a
        n = np.array([-d[1], d[0]])
        n = n / (np.linalg.norm(n) + 1e-12)
        m = 0.5 * (a + b)
        if np.dot(n, m - center) < 0:
            n = -n
        pts = center[None, :] + np.linspace(0, 1, 2)[:, None] * n[None, :] * length
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1],
                                 mode='lines', line=dict(dash='dot', color='gray'),
                                 showlegend=False))

    def select_for_t(arr, t):
        if arr is None:
            return None
        if arr.ndim == 3:      # (K,T,3)
            return arr[:, t, :]
        if arr.ndim == 2:
            return arr[:, t] if arr.shape[1] == T else arr
        return arr             # (K,)

    def _build_t_schedule(T, mode="log", max_steps=160, gamma=5.0):
        if T <= 1:
            return np.array([0], dtype=int)
        if mode == "linear":
            if max_steps and T > max_steps:
                idx = np.linspace(0, T - 1, max_steps)
                t_vals = np.unique(np.round(idx).astype(int))
            else:
                t_vals = np.arange(T, dtype=int)
        else:
            S = min(max_steps, T) if max_steps else T
            u = np.linspace(0.0, 1.0, S)
            x = (np.exp(gamma * u) - 1.0) / (np.exp(gamma) - 1.0)
            t_vals = np.clip(np.round(x * (T - 1)).astype(int), 0, T - 1)
            t_vals = np.unique(t_vals)
        if t_vals[0] != 0:
            t_vals = np.insert(t_vals, 0, 0)
        if t_vals[-1] != T - 1:
            t_vals = np.append(t_vals, T - 1)
        return t_vals

    center_2d = compute_circumcenter(anchor_2d[0], anchor_2d[1], anchor_2d[2])

    # ---- Helpers for ranges/deltas ----
    def _minmax_from(arr, default=(0.0, 1.0)):
        if arr is None:
            return default
        flat = arr.reshape(-1)
        mask = np.isfinite(flat)
        if not mask.any():
            return default
        return float(np.nanmin(flat[mask])), float(np.nanmax(flat[mask]))

    def _normalize01(v, vmin, vmax):
        v = np.asarray(v, dtype=float)
        out = (v - vmin) / (vmax - vmin + 1e-12)
        out = np.clip(out, 0.0, 1.0)
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        return out

    def _total_norm(task_arr, ortho_arr):
        return np.sqrt(np.asarray(task_arr, float)**2 + np.asarray(ortho_arr, float)**2)

    def _to_KT(arr):
        """Return an array shaped (K,T) for norms; broadcast if needed."""
        if arr is None:
            return None
        if arr.ndim == 1:
            return np.repeat(arr[:, None], T, axis=1)
        if arr.ndim == 2 and arr.shape[1] == T:
            return arr
        raise ValueError("Array has incompatible shape for (K,T) conversion.")

    # ---- Build (K,T) matrices for norms ----
    task_KT  = _to_KT(task_norm_np)   # (K,T) or None
    ortho_KT = _to_KT(ortho_norm_np)  # (K,T) or None
    total_KT = _total_norm(task_KT, ortho_KT) if (task_KT is not None and ortho_KT is not None) else None

    # Global ranges for magnitudes (size)
    task_min, task_max   = _minmax_from(task_KT,  (0.0, 1.0))
    ortho_min, ortho_max = _minmax_from(ortho_KT, (0.0, 1.0))
    if total_KT is not None:
        total_min, total_max = float(np.nanmin(total_KT)), float(np.nanmax(total_KT))
    else:
        total_min, total_max = 0.0, 1.0

    # ---- Deltas (K,T) for color: current - previous, t=0 delta = 0 ----
    def _delta_prev(X):
        if X is None:
            return None
        d = np.zeros_like(X, dtype=float)
        if X.shape[1] > 1:
            d[:, 1:] = X[:, 1:] - X[:, :-1]
        return d

    task_dKT  = _delta_prev(task_KT)     # (K,T) or None
    ortho_dKT = _delta_prev(ortho_KT)    # (K,T) or None
    total_dKT = _delta_prev(total_KT)    # (K,T) or None

    # ---- asinh compression for color sensitivity (robust scaling) ----
    def _asinh_compress(D):
        """Return (D_comp, comp_max_abs, scale) with robust scale by median(|D|)."""
        if D is None or not np.isfinite(D).any():
            return None, 1.0, 1.0
        absD = np.abs(D[np.isfinite(D)])
        s = np.nanmedian(absD)
        if not np.isfinite(s) or s <= 1e-12:
            # fallback to 90th percentile or 1.0
            s = np.nanpercentile(absD, 90) if absD.size else 1.0
            if not np.isfinite(s) or s <= 1e-12:
                s = 1.0
        Dc = np.arcsinh(D / s)
        m = float(np.nanmax(np.abs(Dc))) if np.isfinite(Dc).any() else 1.0
        if m <= 0:
            m = 1.0
        return Dc, m, s

    task_dC,  task_cabsC,  task_scale = _asinh_compress(task_dKT)
    ortho_dC, ortho_cabsC, ortho_scale = _asinh_compress(ortho_dKT)
    total_dC, total_cabsC, total_scale = _asinh_compress(total_dKT)

    # ---- Scalarize for hover ----
    def _scalarize(x):
        a = np.asarray(x)
        if a.size == 0:
            return float('nan')
        return float(np.nanmean(a))

    # ---- MSE (K,T) + deltas to previous for MSE view ----
    if mse_np is not None:
        mse_scalar = np.empty((K, T))
        for t in range(T):
            mt = select_for_t(mse_np, t)
            mse_scalar[:, t] = np.array([_scalarize(mt[k]) for k in range(K)]) if mt is not None else np.nan

        mse_abs = np.abs(mse_scalar)
        mse_abs_min = float(np.nanmin(mse_abs)) if np.isfinite(mse_abs).any() else 0.0
        mse_abs_max = float(np.nanmax(mse_abs)) if np.isfinite(mse_abs).any() else 1.0

        mse_delta_prev = np.zeros_like(mse_scalar)
        if T > 1:
            mse_delta_prev[:, 1:] = mse_scalar[:, 1:] - mse_scalar[:, :-1]

        # asinh compress MSE deltas too
        mse_dC, mse_cabsC, mse_scale = _asinh_compress(mse_delta_prev)
    else:
        mse_scalar = None
        mse_abs_min, mse_abs_max = 0.0, 1.0
        mse_delta_prev = None
        mse_dC, mse_cabsC, mse_scale = None, 1.0, 1.0

    # ---- Size mappings (global scaling) ----
    def size_from_total(tot_t):
        if tot_t is None:
            return np.full((K,), 8.0)
        r = _normalize01(tot_t, total_min, total_max)
        return 4.0 + r * (18.0 - 4.0)

    def size_from_task(task_t):
        if task_t is None:
            return np.full((K,), 8.0)
        r = _normalize01(task_t, task_min, task_max)
        return 4.0 + r * (18.0 - 4.0)

    def size_from_ortho(ortho_t):
        if ortho_t is None:
            return np.full((K,), 8.0)
        r = _normalize01(ortho_t, ortho_min, ortho_max)
        return 4.0 + r * (18.0 - 4.0)

    def size_from_mse_abs(mse_t):
        if mse_t is None:
            return np.full((K,), 8.0)
        r = (np.abs(mse_t) - mse_abs_min) / (mse_abs_max - mse_abs_min + 1e-12)
        r = np.clip(r, 0.0, 1.0)
        return 4.0 + r * (20.0 - 4.0)

    # ---- Colorbar helpers (dark friendly) ----
    def _cb(title_text):
        return dict(
            title=dict(text=title_text, font=dict(color="white")),
            tickcolor="white",
            tickfont=dict(color="white"),
            bgcolor="rgba(0,0,0,0)"
        )
    def _mse_colorbar():
        return _cb("asinh ΔMSE")

    # ---- Hover ----
    def hover_for_t(t):
        lam_t = select_for_t(lambda_np, t)
        w_t   = select_for_t(weights_np, t)
        tn_t  = select_for_t(task_KT,  t)
        on_t  = select_for_t(ortho_KT, t)
        tot_t = select_for_t(total_KT, t) if total_KT is not None else None
        mse_t = select_for_t(mse_np, t)
        orc_t = select_for_t(oracle_np, t)

        hover = []
        for k in range(K):
            parts = []
            if tot_t is not None:
                parts.append(f"total_norm: {np.round(_scalarize(tot_t[k]), 4)}")
            if tn_t is not None:
                parts.append(f"task_norm: {np.round(_scalarize(tn_t[k]), 4)}")
            if on_t is not None:
                parts.append(f"ortho_norm: {np.round(_scalarize(on_t[k]), 4)}")
            if mse_t is not None:
                parts.append(f"MSE: {np.round(_scalarize(mse_t[k]), 4)}")
            if isinstance(w_t, np.ndarray):
                parts.append(f"w: {np.round(w_t[k], 3)}")
            if lam_t is not None:
                parts.append(f"λ: {np.round(lam_t[k], 3)}")
            if orc_t is not None:
                parts.append(f"Oracle MSE: {np.round(_scalarize(orc_t[k]), 4)}")
            hover.append("<br>".join(parts))
        return hover

    # ---- Time schedule ----
    t_schedule = _build_t_schedule(T, mode=slider_mode, max_steps=max_slider_steps, gamma=log_gamma)
    t0 = int(t_schedule[0])

    # Initial values at t0 for all views
    tn0   = select_for_t(task_KT,  t0)
    on0   = select_for_t(ortho_KT, t0)
    tot0  = select_for_t(total_KT, t0) if total_KT is not None else None

    # Compressed deltas at t0
    dtn0C  = select_for_t(task_dC,  t0) if task_dC  is not None else None
    don0C  = select_for_t(ortho_dC, t0) if ortho_dC is not None else None
    dtot0C = select_for_t(total_dC, t0) if total_dC is not None else None

    sizes_total_0 = size_from_total(tot0)
    sizes_ortho_0 = size_from_ortho(on0)
    sizes_task_0  = size_from_task(tn0)

    hover0 = hover_for_t(t0)

    # MSE view
    if mse_scalar is not None:
        mse0       = mse_scalar[:, t0]
        dprev0C    = mse_dC[:, t0]
        sizes_mse_0 = size_from_mse_abs(mse0)
        colors_mse_0 = dprev0C
    else:
        sizes_mse_0 = np.full((K,), 8.0)
        colors_mse_0 = np.zeros((K,))

    # Common marker outline on dark bg
    outline = dict(color="rgba(255,255,255,0.25)", width=0.6)

    fig = go.Figure()

    # 0) Total norm: size = total_t, color = asinh Δtotal
    fig.add_trace(go.Scatter(
        x=eval_2d[:, 0], y=eval_2d[:, 1], mode='markers',
        marker=dict(
            size=sizes_total_0,
            color=(dtot0C if dtot0C is not None else np.zeros(K)),
            colorscale="RdBu",
            cmin=-total_cabsC, cmax=total_cabsC,
            colorbar=_cb("asinh ΔTotal"),
            line=outline, opacity=0.9
        ),
        name="Total (size=|total|, color=asinh Δtotal)", hoverinfo='text', text=hover0, visible=True
    ))

    # 1) Ortho norm
    fig.add_trace(go.Scatter(
        x=eval_2d[:, 0], y=eval_2d[:, 1], mode='markers',
        marker=dict(
            size=sizes_ortho_0,
            color=(don0C if don0C is not None else np.zeros(K)),
            colorscale="RdBu",
            cmin=-ortho_cabsC, cmax=ortho_cabsC,
            colorbar=_cb("asinh ΔOrtho"),
            line=outline, opacity=0.9
        ),
        name="Ortho (size=|ortho|, color=asinh Δortho)", hoverinfo='text', text=hover0, visible=False
    ))

    # 2) Task norm
    fig.add_trace(go.Scatter(
        x=eval_2d[:, 0], y=eval_2d[:, 1], mode='markers',
        marker=dict(
            size=sizes_task_0,
            color=(dtn0C if dtn0C is not None else np.zeros(K)),
            colorscale="RdBu",
            cmin=-task_cabsC, cmax=task_cabsC,
            colorbar=_cb("asinh ΔTask"),
            line=outline, opacity=0.9
        ),
        name="Task (size=|task|, color=asinh Δtask)", hoverinfo='text', text=hover0, visible=False
    ))

    # 3) MSE view: size=|MSE(t)|, color=asinh ΔMSE(t)-MSE(t-1)
    fig.add_trace(go.Scatter(
        x=eval_2d[:, 0], y=eval_2d[:, 1], mode='markers',
        marker=dict(
            size=sizes_mse_0,
            color=colors_mse_0,
            colorscale="RdBu",
            cmin=-mse_cabsC, cmax=mse_cabsC,
            colorbar=_mse_colorbar(),
            line=outline, opacity=0.9
        ),
        name="MSE (size=|MSE|, color=asinh Δ to t-1)", hoverinfo='text', text=hover0, visible=False
    ))

    # Static traces (always visible)
    fig.add_trace(go.Scatter(
        x=anchor_2d[:, 0], y=anchor_2d[:, 1], mode='markers+text',
        marker=dict(size=10, color='white', symbol='x', opacity=0.85),
        text=[f"w{i}" for i in range(anchor_2d.shape[0])],
        textposition='top center', name="Anchor Points", hoverinfo='skip', visible=True
    ))
    fig.add_trace(go.Scatter(
        x=[center_2d[0]], y=[center_2d[1]], mode='markers+text',
        marker=dict(size=12, color='white', symbol='x', opacity=0.85),
        text=["Equidistant center"], textposition='bottom right',
        name="Equidistant Center", hoverinfo='skip', visible=True
    ))
    add_half_bisector(fig, anchor_2d[0], anchor_2d[1], center_2d)
    add_half_bisector(fig, anchor_2d[1], anchor_2d[2], center_2d)
    add_half_bisector(fig, anchor_2d[0], anchor_2d[2], center_2d)

    N_TRACES = len(fig.data)  # 4 views + 1 anchor + 1 center + 3 bisectors = 9 total
    vis_total = [True,  False, False, False] + [True] * (N_TRACES - 4)
    vis_ortho = [False, True,  False, False] + [True] * (N_TRACES - 4)
    vis_task  = [False, False, True,  False] + [True] * (N_TRACES - 4)
    vis_mse   = [False, False, False, True ] + [True] * (N_TRACES - 4)

    # ---- Frames (update all 4 eval traces in fixed order) ----
    frames = []
    for t in _build_t_schedule(T, mode=slider_mode, max_steps=max_slider_steps, gamma=log_gamma):
        t = int(t)
        tn_t   = select_for_t(task_KT,  t)
        on_t   = select_for_t(ortho_KT, t)
        tot_t  = select_for_t(total_KT, t) if total_KT is not None else None

        dtn_tC  = select_for_t(task_dC,  t) if task_dC  is not None else None
        don_tC  = select_for_t(ortho_dC, t) if ortho_dC is not None else None
        dtot_tC = select_for_t(total_dC, t) if total_dC is not None else None

        # MSE at t
        if mse_scalar is not None:
            mse_t   = mse_scalar[:, t]
            dprev_tC = mse_dC[:, t]
            siz_mse_t = size_from_mse_abs(mse_t)
            col_mse_t = dprev_tC
        else:
            siz_mse_t = np.full((K,), 8.0)
            col_mse_t = np.zeros((K,))

        frames.append(go.Frame(
            name=f"t={t}",
            data=[
                # 0 total
                go.Scatter(
                    x=eval_2d[:, 0], y=eval_2d[:, 1], mode='markers',
                    marker=dict(
                        size=size_from_total(tot_t),
                        color=(dtot_tC if dtot_tC is not None else np.zeros(K)),
                        colorscale="RdBu", cmin=-total_cabsC, cmax=total_cabsC,
                        colorbar=_cb("asinh ΔTotal"), line=outline, opacity=0.9
                    ),
                    hoverinfo='text', text=hover_for_t(t), name="Total (size=|total|, color=asinh Δtotal)"
                ),
                # 1 ortho
                go.Scatter(
                    x=eval_2d[:, 0], y=eval_2d[:, 1], mode='markers',
                    marker=dict(
                        size=size_from_ortho(on_t),
                        color=(don_tC if don_tC is not None else np.zeros(K)),
                        colorscale="RdBu", cmin=-ortho_cabsC, cmax=ortho_cabsC,
                        colorbar=_cb("asinh ΔOrtho"), line=outline, opacity=0.9
                    ),
                    hoverinfo='text', text=hover_for_t(t), name="Ortho (size=|ortho|, color=asinh Δortho)"
                ),
                # 2 task
                go.Scatter(
                    x=eval_2d[:, 0], y=eval_2d[:, 1], mode='markers',
                    marker=dict(
                        size=size_from_task(tn_t),
                        color=(dtn_tC if dtn_tC is not None else np.zeros(K)),
                        colorscale="RdBu", cmin=-task_cabsC, cmax=task_cabsC,
                        colorbar=_cb("asinh ΔTask"), line=outline, opacity=0.9
                    ),
                    hoverinfo='text', text=hover_for_t(t), name="Task (size=|task|, color=asinh Δtask)"
                ),
                # 3 MSE
                go.Scatter(
                    x=eval_2d[:, 0], y=eval_2d[:, 1], mode='markers',
                    marker=dict(
                        size=siz_mse_t,
                        color=col_mse_t,
                        colorscale="RdBu",
                        cmin=-mse_cabsC, cmax=mse_cabsC,
                        colorbar=_mse_colorbar(),
                        line=outline, opacity=0.9
                    ),
                    hoverinfo='text', text=hover_for_t(t), name="MSE (size=|MSE|, color=asinh Δ to t-1)"
                ),
            ]
        ))
    fig.frames = frames

    # ---- Slider ----
    slider_steps = [{
        "method": "animate",
        "label": str(int(t)),
        "args": [[f"t={int(t)}"],
                 {"mode": "immediate",
                  "frame": {"duration": 0, "redraw": True},
                  "transition": {"duration": 0}}]
    } for t in _build_t_schedule(T, mode=slider_mode, max_steps=max_slider_steps, gamma=log_gamma)]

    sliders = [{
        "active": 0,
        "currentvalue": {"prefix": "t = ", "font": {"color": "white"}},
        "pad": {"t": 8, "b": 0},
        "steps": slider_steps,
        "len": 0.95,
        "x": 0.025, "xanchor": "left",
        "y": -0.10, "yanchor": "top",
        "font": {"color": "white"}
    }]

    # ---- Dropdown (opaque dark, readable on hover) ----
    def _title_text():
        base = title or "Switch views: Total / Ortho / Task / MSE"
        return base + "<br><sup>" \
               "Total/Ortho/Task: size = magnitude at t, color = asinh Δ(t−t−1) • " \
               "MSE: size = |MSE(t)|, color = asinh ΔMSE(t−t−1)" \
               "</sup>"

    updatemenus = [{
        "type": "dropdown",
        "direction": "down",
        "x": 0.8, "xanchor": "center",
        "y": 0.99, "yanchor": "top",      # inside the plot, under the title
        "bgcolor": "rgba(32,32,32,1.0)",  # OPAQUE dark to avoid white flash
        "bordercolor": "rgba(255,255,255,0.35)",
        "font": {"color": "#f0f0f0"},
        "pad": {"r": 8, "t": 4, "b": 4, "l": 8},
        "buttons": [
            {"label": "Total (size=|total|, color=asinh Δtotal)",
             "method": "update",
             "args": [{"visible": vis_total},
                      {"title": {"text": _title_text()}}]},
            {"label": "Ortho (size=|ortho|, color=asinh Δortho)",
             "method": "update",
             "args": [{"visible": vis_ortho},
                      {"title": {"text": _title_text()}}]},
            {"label": "Task (size=|task|, color=asinh Δtask)",
             "method": "update",
             "args": [{"visible": vis_task},
                      {"title": {"text": _title_text()}}]},
            {"label": "MSE (size=|MSE|, color=asinh Δ to t-1)",
             "method": "update",
             "args": [{"visible": vis_mse},
                      {"title": {"text": _title_text()}}]},
        ]
    }]

    # ---- Layout (dark) ----
    subtitle = (f"task_norm: [{task_min:.4g}, {task_max:.4g}] • "
                f"ortho_norm: [{ortho_min:.4g}, {ortho_max:.4g}] • "
                f"total_norm: [{total_min:.4g}, {total_max:.4g}]")

    fig.update_layout(
        title=dict(
            text=(title or "Switch views: Total / Ortho / Task / MSE") + "<br><sup>" + subtitle + "</sup>",
            x=0.5, xanchor="center", y=0.98, yanchor="top",
            font=dict(color="white"), pad=dict(b=8)
        ),
        xaxis_title="PC 1",
        yaxis_title="PC 2",
        width=860,
        height=840,
        showlegend=False,
        margin=dict(l=60, r=20, t=120, b=110),
        sliders=sliders,
        updatemenus=updatemenus,
        plot_bgcolor="rgb(18,18,18)",
        paper_bgcolor="rgb(18,18,18)",
        font=dict(color="white")
    )

    # axis styling for dark background
    fig.update_xaxes(
        showgrid=True, gridcolor="rgba(255,255,255,0.12)",
        zeroline=False, linecolor="rgba(255,255,255,0.3)"
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(255,255,255,0.12)",
        zeroline=False, linecolor="rgba(255,255,255,0.3)",
        scaleanchor="x", scaleratio=1
    )

    fig.show()






def plot_lambda_projection(
    model, eval_task, train_task, r2_score, lambdas, weights, title=None):
    """
    Plot convex combination projections of eval tasks onto anchor tasks using PCA and color coding for λ.

    Args:
        train_task_pool (Tensor): shape (3, d, 1) or (3, d) — anchor vectors
        eval_task_pool (Tensor): shape (K, d, 1) or (K, d) — evaluation vectors (not used for PCA)
        lambdas (Tensor or ndarray): shape (K, 3) — convex weights
        weights (Tensor or ndarray): shape (K, 3) — convex weights used for projection (can be same as lambdas)
        results (list or ndarray, optional): MSEs or other metrics to display per point
        title (str, optional): plot title
    """
    results, oracle_results = get_mse_last(model, eval_task, train_task)
    # Step 1: Convert to NumPy
    anchor_np = train_task.task_pool.squeeze(-1).cpu().numpy()
    eval_np = eval_task.task_pool.squeeze(-1).cpu().numpy()
    lambda_np = lambdas.cpu().numpy()
    weights_np = weights.cpu().numpy()
    r2_score_np = r2_score.cpu().numpy()

    # Step 2: PCA on anchor points
    pca = PCA(n_components=2)
    # anchor_2d = pca.fit_transform(anchor_np)

    # Step 3: Project eval points using λ ⋅ anchor_2d
    eval_2d = pca.fit_transform(eval_np) # weights_np @ anchor_2d
    anchor_2d = eval_2d[:3]

    def compute_circumcenter(p1, p2, p3):
        A = np.stack([p2 - p1, p3 - p1])  # shape (2, 2)
        b = np.array([
            np.dot(p2, p2) - np.dot(p1, p1),
            np.dot(p3, p3) - np.dot(p1, p1)
        ]) / 2
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        return x  # shape (2,)

    center_2d = compute_circumcenter(anchor_2d[0], anchor_2d[1], anchor_2d[2])

    # Step 4: Color map from λ to RGB
    colors = lambda_np @ np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    colors = np.clip(colors, 0, 255).astype(int)
    colors_hex = [f"rgb({r},{g},{b})" for r, g, b in colors]

    def add_half_bisector(fig, a, b, center, length=5.0):
        """
        Draw a bisector ray starting at 'center', orthogonal to the segment a-b,
        pointing away from the segment (direction based on cross-product sign).
        """
        # Get normal vector (orthogonal to a-b)
        direction = b - a
        normal = np.array([-direction[1], direction[0]])
        normal = normal / np.linalg.norm(normal)
    
        # Ensure consistent direction: point roughly away from midpoint
        midpoint = 0.5 * (a + b)
        dot = np.dot(normal, midpoint - center)
        if dot < 0:
            normal = -normal  # flip
    
        # Ray from center
        line_pts = center[None, :] + np.linspace(0, 1, 2)[:, None] * normal[None, :] * length
        fig.add_trace(go.Scatter(
            x=line_pts[:, 0],
            y=line_pts[:, 1],
            mode='lines',
            line=dict(dash='dot', color='gray'),
            showlegend=False
        ))



    # Step 5: Plot
    fig = go.Figure()

    # Eval points
    if results is not None:
        hover_text = [f"λ={np.round(lambda_np[k], 2)}<br>w={np.round(weights_np[k], 2)}<br>Oracle MSE:{np.round(oracle_results[k], 2)}<br>MSE:{results[k]:.2f}<br>R2:{r2_score_np[k]:.2f}" for k in range(len(lambda_np))]
    else:
        hover_text = [f"λ = {np.round(lambda_np[k], 2)}" for k in range(len(lambda_np))]

    fig.add_trace(go.Scatter(
        x=eval_2d[:, 0],
        y=eval_2d[:, 1],
        mode='markers',
        marker=dict(size=6, color=colors_hex, opacity=0.8),
        name="Eval Points",
        hoverinfo='text',
        text=hover_text
    ))

    # Anchor points
    fig.add_trace(go.Scatter(
        x=anchor_2d[:, 0],
        y=anchor_2d[:, 1],
        mode='markers+text',
        marker=dict(size=10, color='black', symbol='x', opacity=0.6),
        text=[f"w{i}" for i in range(anchor_2d.shape[0])],
        textposition='top center',
        name="Anchor Points"
    ))

    fig.add_trace(go.Scatter(
        x=[center_2d[0]],
        y=[center_2d[1]],
        mode='markers+text',
        marker=dict(size=12, color='black', symbol='x', opacity=0.6),
        textposition='bottom right',
        name="Equidistant Center"
    ))

    add_half_bisector(fig, anchor_2d[0], anchor_2d[1], center_2d)
    add_half_bisector(fig, anchor_2d[1], anchor_2d[2], center_2d)
    add_half_bisector(fig, anchor_2d[0], anchor_2d[2], center_2d)

    fig.update_layout(
        title=title or "Attraction to Anchor Points (λ0→R, λ1→G, λ2→B)",
        xaxis_title="PC 1",
        yaxis_title="PC 2",
        width=700,
        height=700,
        showlegend=True
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.show()


def process_checkpoints(exp_name, K=120, pca_components=2, task_layer_index=3):
    """
    Processes checkpoints for a given experiment, performs PCA projection, 
    and plots the lambda projection for each checkpoint.

    Args:
    - exp_name (str): Experiment name.
    - config (dict or object): Configuration dictionary or object.
    - train_task (object): Task object for training.
    - K (int): Number of tasks for evaluation (default 100).
    - pca_components (int): Number of PCA components (default 2).
    - task_layer_index (int): Layer index for task vector extraction (default 3).
    """
    model, train_task, config = load_model_task_config(exp_name)
    _, final_task_vectors = get_task_vector_from_hidden(config, 
                                                        model, 
                                                        train_task, 
                                                        task_layer_index, 
                                                        compute_mean=True, return_final=True) # (n_tasks, n_embd)
    del model
    torch.cuda.empty_cache()

    checkpoint_files = get_checkpoint_files(exp_name)
    checkpoint_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    exp_dir = os.path.join(config.work_dir, exp_name)   
    cur_dir = os.getcwd()
    if cur_dir.endswith("notebooks"):
        exp_dir = os.path.join("..", exp_dir)

    # Prepare task pools and evaluation configurations
    anchor_pool = train_task.task_pool.squeeze(-1)
    eval_task_pool, weights = sample_union_unit_balls_affine_span_with_weights(anchor_pool, K, 2.5)

    eval_config = config.copy() if isinstance(config, dict) else config
    eval_config.task.n_tasks = K
    eval_task = get_task(**eval_config["task"])
    eval_task.batch_size = 256
    eval_task.task_pool = eval_task_pool.unsqueeze(-1)  # shape (K, d, 1)

    # PCA projection of anchor and eval task pools
    anchor_np = anchor_pool.cpu().numpy()
    eval_np = eval_task_pool.squeeze(-1).cpu().numpy()
    pca = PCA(n_components=pca_components)
    eval_2d = pca.fit_transform(eval_np)

    # Compute projections and bisectors
    anchor_2d = eval_2d[:3]
    center_2d = compute_circumcenter(anchor_2d[0], anchor_2d[1], anchor_2d[2])
    line_pts0 = add_half_bisector(anchor_2d[0], anchor_2d[1], center_2d)
    line_pts1 = add_half_bisector(anchor_2d[1], anchor_2d[2], center_2d)
    line_pts2 = add_half_bisector(anchor_2d[0], anchor_2d[2], center_2d)

    # Iterate through checkpoint files and generate plots
    for k in trange(0, len(checkpoint_files), 3):
        model, config, train_task = load_checkpoint(exp_name, checkpoint_files[k])
        
        # Extract task vectors from hidden layers
        _, eval_task_vectors = get_task_vector_from_hidden(config, model, eval_task, 
                                                           layer_index=task_layer_index, 
                                                           compute_mean=True, return_final=False)

        del model
        # Estimate lambdas
        lambdas, _ = estimate_lambda_super_fast(final_task_vectors, eval_task_vectors, compute_r2=False)

        # Get current step from filename
        curr_step = int(re.search(r'\d+', checkpoint_files[k]).group())

        # Plot lambda projection
        plot_lambda_projection_matplotlib(
            anchor_2d, eval_2d, center_2d, 
            line_pts0, line_pts1, line_pts2,
            lambdas, weights, exp_dir,
            curr_step, save_path_prefix=f"task_vec", dpi=100)

def compute_circumcenter(p1, p2, p3):
    """
    Compute the circumcenter (equidistant point) of three points.
    
    Finds the center of the circle passing through the three points,
    which is equidistant from all three points.
    
    Args:
        p1, p2, p3: Three 2D points, each of shape (2,)
    
    Returns:
        Circumcenter point, shape (2,)
    """
    A = np.stack([p2 - p1, p3 - p1])
    b = np.array([
        np.dot(p2, p2) - np.dot(p1, p1),
        np.dot(p3, p3) - np.dot(p1, p1)
    ]) / 2
    x = np.linalg.lstsq(A.T, b, rcond=None)[0]
    return x

def add_half_bisector(a, b, center, length=5.0):
    """
    Compute points for a half-angle bisector line.
    
    Draws a line from the circumcenter in the direction of the angle bisector
    between two anchor points. Used for visualization.
    
    Args:
        a, b: Two anchor points, each shape (2,)
        center: Circumcenter point, shape (2,)
        length: Length of the bisector line
    
    Returns:
        line_pts: Points defining the bisector line, shape (2, 2)
    """
    direction = b - a
    normal = np.array([-direction[1], direction[0]])
    normal = normal / np.linalg.norm(normal)

    midpoint = 0.5 * (a + b)
    if np.dot(normal, midpoint - center) < 0:
        normal = -normal

    line_pts = center[None, :] + np.linspace(0, 1, 2)[:, None] * normal[None, :] * length
    return line_pts



def plot_lambda_projection_matplotlib(
    anchor_2d, eval_2d, center_2d, line_pts0, line_pts1, line_pts2,
    lambdas, weights, folder,
    step, save_path_prefix="frame", dpi=100
    ):
    """
    Create a matplotlib scatter plot showing task vector projections with lambda coloring.
    
    Visualizes task vectors projected to 2D PCA space, with colors determined by
    lambda weights (RGB mapping) and includes anchor points and bisectors.
    
    Args:
        anchor_2d: Anchor points in 2D, shape (3, 2)
        eval_2d: Evaluation task points in 2D, shape (K, 2)
        center_2d: Circumcenter in 2D, shape (2,)
        line_pts0, line_pts1, line_pts2: Bisector line points
        lambdas: Lambda weights, shape (K, 3)
        weights: Task weights (for reference)
        folder: Directory to save the plot
        step: Training step number
        save_path_prefix: Prefix for saved filename
        dpi: Resolution for saved image
    """
    # results, oracle_results = get_mse_last(model, eval_task, train_task)

    # Convert to NumPy
    # train_task_pool = train_task.task_pool
    # eval_task_pool = eval_task.task_pool
    lambda_np = lambdas.cpu().numpy() if torch.is_tensor(lambdas) else np.asarray(lambdas)
    weights_np = weights.cpu().numpy() if torch.is_tensor(weights) else np.asarray(weights)

    # Color map
    colors = lambda_np @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    colors = np.clip(colors, 0, 1)
    # Create plot
    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)

    # Eval points
    ax.scatter(eval_2d[3:, 0], eval_2d[3:, 1], c=colors[3:], s=30, alpha=0.8)

    # Anchor points
    ax.scatter(anchor_2d[:, 0], anchor_2d[:, 1], color='black', s=100, marker='x', label="Anchor Points")
    for i, (x, y) in enumerate(anchor_2d):
        ax.text(x, y + 0.05, f"w{i}", ha='center')

    # Equidistant center
    ax.scatter(center_2d[0], center_2d[1], color='black', s=80, marker='o', label="Equidistant Center")

    # Bisectors
    ax.plot(line_pts0[:, 0], line_pts0[:, 1], linestyle='dotted', color='gray')
    ax.plot(line_pts1[:, 0], line_pts1[:, 1], linestyle='dotted', color='gray')
    ax.plot(line_pts2[:, 0], line_pts2[:, 1], linestyle='dotted', color='gray')

    # Add the current step to the title
    ax.set_title(f"Attraction to Anchor Points (Step {step}) - λ0→R, λ1→G, λ2→B")

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True)

    # Add the current step as a text annotation (optional)
    ax.text(0.95, 0.05, f"Step {step}", transform=ax.transAxes,
            fontsize=12, ha='right', va='bottom', color='black', weight='bold')

    # Save figure
    filename = f"{save_path_prefix}_step_{step}.png"
    save_path = os.path.join(folder, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def create_projection_gif(config, exp_name, gif_filename="projection_animation.gif"):
    """
    Creates an animated GIF from task vector PNG images in the specified directory.

    Args:
    - exp_dir (str): The directory where the PNG images are stored.
    - gif_filename (str): The name of the output GIF file (default is "projection_animation.gif").
    """
    # Initialize list for images and durations
    exp_dir = os.path.join(config.work_dir, exp_name)   
    cur_dir = os.getcwd()
    if cur_dir.endswith("notebooks"): 
        exp_dir = os.path.join("..", exp_dir)
    images = []
    durations = []

    # Parse filenames and compute step gaps
    previous_step = 0
    file_paths = glob.glob(os.path.join(exp_dir, "task_vec_step_*.png"))
    file_paths = sorted(file_paths, key=lambda x: int(re.search(r"step_(\d+)", x).group(1)))

    for filename in file_paths:
        # Extract step number from filename using regex
        match = re.search(r"step_(\d+).png", filename)
        if match:
            step = int(match.group(1))

            # Compute the step gap and set the duration
            step_gap = step - previous_step
            duration = step_gap / 30  # Adjust this logic based on your needs

            # Append the image and its duration
            images.append(imageio.imread(filename))
            durations.append(duration)

            # Update the previous step for the next iteration
            previous_step = step

    # Save the GIF with varying frame durations
    gif_path = os.path.join(exp_dir, gif_filename)
    imageio.mimsave(gif_path, images, duration=durations)



def plot_mse_vs_position(model, samplers_eval, bayes_ood, bayes_id, step: int = 1):
    """
    Plot MSE and prediction differences across positions for ID and OOD tasks.
    
    Compares model performance to Ridge and DiscreteMMSE baselines,
    showing both MSE loss and absolute prediction differences.
    
    Args:
        model: Trained model to evaluate
        samplers_eval: Dictionary of evaluation samplers for different modes
        bayes_ood: Bayesian out-of-distribution estimator
        bayes_id: Bayesian in-distribution (DiscreteMMSE) estimator
        step: Step size for evaluation
    """
    import numpy as np
    import torch
    import plotly.graph_objects as go

    torch.cuda.empty_cache()

    def compute_metrics(mode: str):
        _data, _, _target = samplers_eval[mode](step=step)
        _data = _data.squeeze(0)
        _target = _target.squeeze(0)

        with torch.no_grad():
            preds = model(_data, _target)
        loss = ((preds - _target.to(preds.device))**2).mean(dim=0)

        re_preds = bayes_ood(_data, _target)
        re_loss = ((re_preds - _target.to(re_preds.device))**2).mean(dim=0)

        dmmse_preds = bayes_id(_data, _target)
        dmmse_loss = ((dmmse_preds - _target.to(dmmse_preds.device))**2).mean(dim=0)

        diff_dmmse = np.abs(preds.mean(dim=0).cpu().numpy() - dmmse_preds.mean(dim=0).cpu().numpy())
        diff_re = np.abs(preds.mean(dim=0).cpu().numpy() - re_preds.mean(dim=0).cpu().numpy())

        return (
            loss.cpu().numpy(),
            re_loss.cpu().numpy(),
            dmmse_loss.cpu().numpy(),
            diff_dmmse,
            diff_re,
        )

    results = {
        "ID": compute_metrics("Pretrain"),
        "OOD": compute_metrics("Latent"),
    }

    t = np.arange(1, results["ID"][0].shape[0] + 1)

    fig = go.Figure()
    modes = ["ID", "OOD"]
    views = ["MSE", "Δ"]
    trace_labels = {
        "MSE": ["Model MSE", "Ridge MSE", "dMMSE MSE"],
        "Δ": ["dMMSE Δ", "Ridge Δ"]
    }
    line_styles = [
        dict(width=2),
        dict(width=2, dash="dash"),
        dict(width=2, dash="dot")
    ]

    # Add all traces
    for mode in modes:
        vals = results[mode]
        for i in range(3):
            fig.add_trace(go.Scatter(
                x=t, y=vals[i], mode="lines+markers",
                name=f"{trace_labels['MSE'][i]} ({mode})",
                line=line_styles[i],
                marker=dict(size=4),
                visible=(mode == "ID" and i < 3)  # default
            ))
        for i in range(3, 5):
            fig.add_trace(go.Scatter(
                x=t, y=vals[i], mode="lines+markers",
                name=f"{trace_labels['Δ'][i - 3]} ({mode})",
                line=line_styles[i - 3],
                marker=dict(size=4),
                visible=False
            ))

    # Utility: visibility mask for 4 modes
    def vis_mask(mode, view):
        out = []
        for m in modes:
            for i in range(5):
                out.append((m == mode and ((view == "MSE" and i < 3) or (view == "Δ" and i >= 3))))
        return out

    # One dropdown, 4 buttons
    dropdown = dict(
        buttons=[
            dict(label="MSE vs Position (ID)",
                 method="update",
                 args=[
                     {"visible": vis_mask("ID", "MSE")},
                     {"title": {"text": "MSE vs Position (ID)"},
                      "yaxis": {"title": "MSE"}}
                 ]),
            dict(label="Prediction Difference (ID)",
                 method="update",
                 args=[
                     {"visible": vis_mask("ID", "Δ")},
                     {"title": {"text": "Prediction Difference (ID)"},
                      "yaxis": {"title": "Absolute Difference"}}
                 ]),
            dict(label="MSE vs Position (OOD)",
                 method="update",
                 args=[
                     {"visible": vis_mask("OOD", "MSE")},
                     {"title": {"text": "MSE vs Position (OOD)"},
                      "yaxis": {"title": "MSE"}}
                 ]),
            dict(label="Prediction Difference (OOD)",
                 method="update",
                 args=[
                     {"visible": vis_mask("OOD", "Δ")},
                     {"title": {"text": "Prediction Difference (OOD)"},
                      "yaxis": {"title": "Absolute Difference"}}
                 ]),
        ],
        direction="down",
        x=0.01,
        y=1.15,
        showactive=True,
        xanchor="left"
    )

    fig.update_layout(
        updatemenus=[dropdown],
        title="MSE vs Position (ID)",
        xaxis_title="Position",
        yaxis_title="MSE",
        template="plotly_white",
        height=500,
        width=800,
        legend=dict(title="Legend", itemsizing="constant")
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGray')

    fig.show()







def plot_all_relative_errors(train_task, task_vectors, get_dmmse_posterior):
    """
    Plots relative error curves for all k:
        rel_error_k(t) = ||approx_vec - true_vec|| / ||true_vec||
    where approx_vec = E_posterior[lambda] @ task_vectors[:-1, -1]

    Args:
        train_task: input to get_dmmse_posterior (custom format)
        task_vectors: Tensor of shape (num_tasks, seq_len, d)
        get_dmmse_posterior: function(train_task, k) -> (posterior, xs)
    """
    num_tasks, seq_len, _ = task_vectors.shape

    fig = go.Figure()
    x_vals = list(range(seq_len))

    for k in range(num_tasks):
        posterior, xs = get_dmmse_posterior(train_task, k)  # shape: (num_samples, seq_len, num_tasks)
        bayes_lambdas = posterior.mean(dim=0)               # shape: (seq_len, num_tasks)
        approx_vecs = bayes_lambdas @ task_vectors[:, -1]  # shape: (d,)
        true_vecs = task_vectors[k]                         # shape: (seq_len, d)

        # Relative error at each t
        rel_error = (approx_vecs - true_vecs).norm(dim=-1) / true_vecs.norm(dim=-1)

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=rel_error.cpu().numpy(),
            mode='lines+markers',
            name=f"k = {k}",
            hovertemplate="t: %{x}<br>Rel. error: %{y:.4f}<extra>k = " + str(k) + "</extra>"
        ))

    fig.update_layout(
        title="Relative Error for Posterior-Weighted Task Vector",
        xaxis_title="t (position)",
        yaxis_title="Relative Error",
        height=500,
        legend_title="Task Index k"
    )

    fig.show()

def plot_all_relative_errors_eval(eval_task, train_task, final_task_vectors, eval_vectors, get_dmmse_posterior_eval):
    """
    Plots relative error curves for all k:
        rel_error_k(t) = ||approx_vec - true_vec|| / ||true_vec||
    where approx_vec = E_posterior[lambda] @ final_task_vectors[:-1]

    Args:
        train_task: input to get_dmmse_posterior (custom format)
        task_vectors: Tensor of shape (num_tasks, seq_len, d)
        get_dmmse_posterior: function(train_task, k) -> (posterior, xs)
    """
    num_tasks, seq_len, _ = eval_vectors.shape

    fig = go.Figure()
    x_vals = list(range(seq_len))

    for k in range(num_tasks):
        posterior, xs = get_dmmse_posterior_eval(eval_task, train_task, k)  # shape: (num_samples, num_tasks)
        bayes_lambdas = posterior.mean(dim=0)               # shape: (num_tasks,)
        approx_vecs = bayes_lambdas[:-1] @ final_task_vectors  # shape: (d,)
        true_vecs = eval_vectors[k]                         # shape: (seq_len, d)

        # Relative error at each t
        rel_error = (approx_vecs - true_vecs).norm(dim=-1) / true_vecs.norm(dim=-1)

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=rel_error.cpu().numpy(),
            mode='lines+markers',
            name=f"k = {k}",
            hovertemplate="t: %{x}<br>Rel. error: %{y:.4f}<extra>k = " + str(k) + "</extra>"
        ))

    fig.update_layout(
        title="Relative Error for Posterior-Weighted Task Vector",
        xaxis_title="t (position)",
        yaxis_title="Relative Error",
        height=500,
        legend_title="Task Index k"
    )

    fig.show()