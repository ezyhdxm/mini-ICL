import os
import torch
import numpy as np
from typing import Optional, Sequence, Tuple, Union
from torch import nn

import icl.utils.notebook_utils as nu
from icl.utils.basic import get_hash
from icl.linear.lr_config import get_config
from icl.latent_markov.latent_config import get_config_base
from icl.utils.latent_ood_analysis import get_all_samples, get_latent_sampler
from icl.utils.linear_ood_analysis import (
    _create_eval_task_pool,
    _setup_eval_task,
    setup_device,
    )
from icl.utils.kv_latent_task_vec_beta import compute_hiddens_onepos_all_layers_kvcache_beta
from icl.utils.ultra_latent_task_vec import compute_hiddens_onepos_all_layers_ultra
from icl.linear.linear_path_utils import load_model_task_config
from icl.linear.task_vecs import compute_hiddens_multi, compute_hiddens_token_conditioned
from icl.linear.task_variance import compute_task_variance_multi_layer, extract_plotting_data_multi_layer
from icl.utils.coin_latent_task_vecs import (
    compute_hiddens_multi_coin_latent,
    compute_hiddens_token_conditioned_coin_latent,
    extract_hidden_multi_coin_latent,
)
from icl.utils.logger import setup_logger
from icl.utils.unified_path_finder import (
    _get_hidden_cache_path,
    unified_get_config,
)
from icl.utils.latent_task_vec import compute_hiddens
from icl.utils.coin_ood_analysis import get_new_sampler
from icl.dyck.dyck_task_vec import get_dyck_sampler, compute_hiddens_dyck
from icl.dyck.dyck_utils import sample_binary_mask
from icl.utils.linear_algebra_utils import stable_rank

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = setup_logger(__name__)


def _get_hiddens(
        task_name, 
        exp_name, 
        n_minor=64, 
        n_ood=30, 
        B=64,
        step: Optional[int] = None,
        force_recompute=False,
        verbose=False,
        **kwargs,
        ):

    if task_name == "latent":
        _, sampler, config = nu.load_everything("latent", exp_name)
        k_minor = min(n_minor, sampler.n_minor_tasks)
        if step is None:
            step = config.training.num_epochs

        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
        if verbose:
            logger.info("Getting samples...")
        all_samples, k_minor = get_all_samples(exp_name, n_minor=n_minor, n_ood=n_ood, B=B)
        if verbose:
            logger.info("Computing hiddens...")
        hiddens = compute_hiddens_onepos_all_layers_kvcache_beta(
                config, 
                model, 
                all_samples, 
                k_step = 32,
                b_step = 32,
                t_step = 4
            ).permute(0, 1, 3, 2, 4, 5) # (n_layers, n_tasks, num_states, T, B, D)
    
    elif task_name == "coin":
        _, sampler, config = nu.load_everything("coin", exp_name)
        k_minor = min(n_minor, sampler.n_minor_tasks)
        if step is None:
            step = config.training.num_epochs

        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
        if verbose:
            logger.info("Getting samples...")
        sampler_clone, k_minor = get_new_sampler(exp_name, n_minor, n_ood)
        if kwargs.get("return_data", False):
            hiddens, demo_data = compute_hiddens(config, model, sampler_clone, B, return_data=True)
        else:
            hiddens = compute_hiddens(config, model, sampler_clone, B)
        
        if kwargs.get("return_p", False):
            return hiddens, k_minor, torch.concat([sampler_clone.major_p, sampler_clone.minor_p])
        if kwargs.get("return_data", False):
            return hiddens, k_minor, demo_data, sampler_clone

    
    elif task_name == "dyck":
        _, sampler, config = nu.load_everything("dyck", exp_name)
        k_minor = min(n_minor, sampler.n_minor_tasks)
        if step is None:
            step = config.training.num_epochs
        
        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
        if verbose:
            logger.info("Getting samples...")
        sampler_clone, k_minor = get_dyck_sampler(exp_name, n_minor, n_ood)
        mask = sample_binary_mask(config)
        hiddens = compute_hiddens_dyck(config,
                    model,
                    sampler = sampler_clone,
                    dyck_mask = mask,
                    batch_size = B,
                    )
        return hiddens, k_minor, mask

    elif task_name == "linear":
        _, train_task, config = load_model_task_config(exp_name)
        k_minor = min(n_minor, train_task.n_minor_tasks)
        if step is None:
            step = config.training.total_steps
        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
        
        device = setup_device(None)
        if verbose:
            logger.info("Creating eval task pool...")
        
        eval_task_pool, k_minor = _create_eval_task_pool(
            train_task, 
            K=n_ood, 
            include_minor=True, 
            radius=2.0, 
            device=device,
            n_minor=n_minor,
        )
        eval_task = _setup_eval_task(config, eval_task_pool, B, device)
        if verbose:
            logger.info("Computing hiddens...")
        
        # Use smaller chunk_size for compute_hiddens_multi to reduce memory usage
        # Calculate chunk_size based on number of tasks and batch size to avoid OOM
        # For linear tasks, eval_task_pool is a tensor with shape (n_tasks, n_dims)
        n_tasks_estimate = eval_task_pool.shape[0] if isinstance(eval_task_pool, torch.Tensor) else (train_task.n_tasks + min(n_minor, train_task.n_minor_tasks) + n_ood)
        # Use smaller chunks for large numbers of tasks or large batch sizes
        # For B=96 and many tasks, use very small chunks
        if B >= 64 and n_tasks_estimate >= 64:
            chunk_size_hiddens = max(2, min(8, 32 // max(1, B // 32)))
        elif n_tasks_estimate >= 256:
            chunk_size_hiddens = max(2, min(8, 64 // max(1, n_tasks_estimate // 64)))
        else:
            chunk_size_hiddens = min(16, max(4, 64 // max(1, n_tasks_estimate // 64)))
        if verbose:
            logger.info(f"Using chunk_size={chunk_size_hiddens} for compute_hiddens_multi (n_tasks={n_tasks_estimate}, B={B})")
        
        hiddens, _ = compute_hiddens_multi(
            config, model, eval_task, 
            chunk_size=chunk_size_hiddens
        ) # (n_layers, n_tasks, T, B, D)
        
        # hiddens is already on CPU from compute_hiddens_multi
        
        # Clean up model and other GPU objects
        del model, eval_task, eval_task_pool, train_task, config
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return hiddens, k_minor


def get_token_conditioned_hiddens(
    exp_name: str,
    layers: Optional[list] = None,
    chunk_size: int = 16,
    step: Optional[int] = None,
    positions_of_interest: Optional[list] = None,
    max_unique_tokens: Optional[int] = None,
    batch_size: int = 16,
    n_minor: int = 64,
    n_ood: int = 30,
    verbose: bool = False,
) -> tuple:
    """
    Get token-conditioned hidden representations for a linear regression experiment.
    
    This function always fixes DATA tokens (not target tokens) at specified positions
    and extracts hidden representations at the PAD tokens immediately following them.
    
    Parameters:
    -----------
    exp_name : str
        Experiment name (e.g., "train_...")
    layers : list, optional
        Which layers to analyze. If None, uses all layers.
    chunk_size : int, default=16
        Chunk size for processing tasks
    step : int, optional
        Step for data generation. If None, uses config.training.total_steps
    positions_of_interest : list, optional
        Position indices to analyze. If None, uses all positions.
    max_unique_tokens : int, optional
        Maximum number of unique tokens per position
    batch_size : int, default=16
        Batch size to use for data sampling. This overrides the batch_size from the config.
        This determines how many unique data tokens are collected per position.
    n_minor : int, default=64
        Number of minor tasks to include in the task pool
    n_ood : int, default=30
        Number of out-of-distribution (OOD) tasks to include in the task pool
    verbose : bool, default=False
        Whether to print progress messages
    
    Returns:
    --------
    all_hiddens : torch.Tensor
        Shape: (L, n_positions, n_unique_tokens, n_tasks, batch_size, n_embd)
    demo_data : torch.Tensor
        Original demo_data batch: (batch_size, n_points, n_dims)
    token_info : dict
        Information about tokens used
    """
    # Load task and config (we'll load model separately to get the right step)
    _, train_task, config = load_model_task_config(exp_name)
    
    # Set step if not provided
    if step is None:
        step = config.training.total_steps
    
    # Load model at specified step (consistent with _get_hiddens)
    # nu.load_checkpoint returns the model itself, not a state_dict
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    
    # Create eval task pool with OOD and minor tasks (similar to _get_hiddens)
    device = setup_device(None)
    if verbose:
        logger.info("Creating eval task pool...")
    
    eval_task_pool, k_minor = _create_eval_task_pool(
        train_task,
        K=n_ood,
        include_minor=True,
        radius=2.0,
        device=device,
        n_minor=n_minor,
    )
    
    # Create eval task with the specified task pool
    eval_task = _setup_eval_task(config, eval_task_pool, batch_size, device)
    
    # Override batch_size (already set by _setup_eval_task, but ensure it's correct)
    eval_task.batch_size = batch_size
    
    # Set default layers if not provided
    if layers is None:
        layers = list(range(config.model.n_layer))
    
    # Check model padding format
    if not hasattr(model, "pad") or model.pad != "mapsto":
        raise ValueError(
            f"Model must use 'mapsto' padding format for token-conditioned analysis. "
            f"Got: {getattr(model, 'pad', None)}"
        )
    
    if verbose:
        logger.info(f"Computing token-conditioned hiddens for exp: {exp_name}")
        logger.info(f"  Layers: {layers}")
        logger.info(f"  Positions: {positions_of_interest if positions_of_interest else 'all'}")
        logger.info(f"  Token type: data (always)")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  n_minor: {k_minor}, n_ood: {n_ood}")
        logger.info(f"  Total tasks: {eval_task.task_pool.shape[0]}")
        logger.info(f"  Step: {step}")
    
    # Compute token-conditioned hiddens using eval_task (which has the specified task pool)
    all_hiddens, demo_data, token_info = compute_hiddens_token_conditioned(
        config=config,
        model=model,
        train_task=eval_task,  # Use eval_task instead of train_task
        layers=layers,
        chunk_size=chunk_size,
        step=step,
        positions_of_interest=positions_of_interest,
        max_unique_tokens=max_unique_tokens,
    )
    
    if verbose:
        logger.info(f"Computed hiddens shape: {all_hiddens.shape}")
        logger.info(f"Demo data shape: {demo_data.shape}")
        logger.info(f"Token info - positions: {token_info['positions']}")
        logger.info(f"Token info - n_unique_tokens: {token_info['n_unique_tokens']}")
    
    return all_hiddens, demo_data, token_info


def get_task_variance(
    exp_name: str,
    layers: Optional[list] = None,
    chunk_size: int = 16,
    step: Optional[int] = None,
    positions_of_interest: Optional[list] = None,
    batch_size: int = 16,
    n_minor: int = 64,
    n_ood: int = 30,
    verbose: bool = False,
    eps: float = 1e-8,
) -> tuple:
    """
    Get hidden representations using compute_hiddens_multi and compute task variance.
    
    This function:
    1. Loads model and creates eval task pool (with n_minor and n_ood tasks)
    2. Uses compute_hiddens_multi to extract hiddens of shape (L, n_tasks, n_points, batch_size, n_embd)
       ensuring the same demo_data is used for all tasks
    3. Computes task variance: variance of batch-averaged hiddens across tasks
    
    Parameters:
    -----------
    exp_name : str
        Experiment name (e.g., "train_...")
    layers : list, optional
        Which layers to analyze. If None, uses all layers.
    chunk_size : int, default=16
        Chunk size for processing tasks
    step : int, optional
        Step for data generation. If None, uses config.training.total_steps
    positions_of_interest : list, optional
        Position indices to analyze. If None, uses all positions.
    batch_size : int, default=16
        Batch size to use for data sampling. This overrides the batch_size from the config.
    n_minor : int, default=64
        Number of minor tasks to include in the task pool
    n_ood : int, default=30
        Number of out-of-distribution (OOD) tasks to include in the task pool
    verbose : bool, default=False
        Whether to print progress messages
    eps : float, default=1e-8
        Small epsilon for numerical stability in normalization
    
    Returns:
    --------
    all_hiddens : torch.Tensor
        Shape: (L, n_tasks, n_points, batch_size, n_embd)
        Hidden representations for each layer, task, position, batch, and embedding dim
    demo_data : torch.Tensor
        Original demo_data batch: (batch_size, n_points, n_dims)
    results_dict : dict
        Mapping from layer number to TaskVarianceResults
    plotting_data : dict
        Data formatted for plotting (from extract_plotting_data_multi_layer)
    """
    # Load task and config (we'll load model separately to get the right step)
    _, train_task, config = load_model_task_config(exp_name)
    
    # Set step if not provided
    if step is None:
        step = config.training.total_steps
    
    # Load model at specified step (consistent with _get_hiddens)
    # nu.load_checkpoint returns the model itself, not a state_dict
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    
    # Create eval task pool with OOD and minor tasks (similar to _get_hiddens)
    device = setup_device(None)
    if verbose:
        logger.info("Creating eval task pool...")
    
    eval_task_pool, k_minor = _create_eval_task_pool(
        train_task,
        K=n_ood,
        include_minor=True,
        radius=2.0,
        device=device,
        n_minor=n_minor,
    )
    
    # Create eval task with the specified task pool
    eval_task = _setup_eval_task(config, eval_task_pool, batch_size, device)
    
    # Override batch_size (already set by _setup_eval_task, but ensure it's correct)
    eval_task.batch_size = batch_size
    
    # Set default layers if not provided
    if layers is None:
        layers = list(range(config.model.n_layer))
    
    if verbose:
        logger.info(f"Computing hiddens for exp: {exp_name}")
        logger.info(f"Layers: {layers}, Batch size: {batch_size}, n_tasks: {eval_task.task_pool.shape[0]}")
    
    # Use compute_hiddens_multi to get hiddens
    # This ensures the same demo_data is used for all tasks
    all_hiddens, demo_data = compute_hiddens_multi(
        config=config,
        model=model,
        train_task=eval_task,
        layers=layers,
        chunk_size=chunk_size,
        return_final=False,  # We need (L, n_tasks, n_points, batch_size, n_embd)
        step=step,
    )
    
    if verbose:
        logger.info(f"Computed hiddens shape: {all_hiddens.shape}")
        logger.info(f"Demo data shape: {demo_data.shape}")
    
    # Compute task variance
    if verbose:
        logger.info("Computing task variance...")
    
    results_dict = compute_task_variance_multi_layer(
        all_hiddens=all_hiddens,
        positions_of_interest=positions_of_interest,
        layers=layers,
        eps=eps,
    )
    
    # Extract plotting data
    plotting_data = extract_plotting_data_multi_layer(results_dict)
    
    if verbose:
        logger.info(f"Computed variance for {len(results_dict)} layers")
    
    return all_hiddens, demo_data, results_dict, plotting_data


def get_task_variance_coin(
    exp_name: str,
    layers: Optional[list] = None,
    batch_size: int = 64,
    positions_of_interest: Optional[list] = None,
    n_minor: int = 64,
    n_ood: int = 30,
    step: Optional[int] = None,
    verbose: bool = False,
    eps: float = 1e-8,
) -> tuple:
    """
    Get hidden representations and compute task variance for Coin task.
    
    This function:
    1. Loads model and creates sampler with n_minor and n_ood tasks
    2. For each task, samples a new batch of data
    3. Extracts hiddens at padded positions (odd indices: 1, 3, 5, ...)
    4. Computes task variance: variance of batch-averaged hiddens across tasks
    
    Parameters:
    -----------
    exp_name : str
        Experiment name (e.g., "train_...")
    layers : list, optional
        Which layers to analyze. If None, uses all layers.
    batch_size : int, default=64
        Batch size for sampling
    positions_of_interest : list, optional
        Position indices (0 to seq_len-2) to analyze. If None, uses all positions.
        These map to padded sequence positions [1, 3, 5, ...]
    n_minor : int, default=64
        Number of minor tasks to include
    n_ood : int, default=30
        Number of out-of-distribution (OOD) tasks to include
    step : int, optional
        Step for checkpoint loading. If None, uses final checkpoint.
    verbose : bool, default=False
        Whether to print progress messages
    eps : float, default=1e-8
        Small epsilon for numerical stability in normalization
    
    Returns:
    --------
    all_hiddens : torch.Tensor
        Shape: (L, n_tasks, n_positions, batch_size, n_embd)
    position_info : dict
        Information about positions analyzed
    results_dict : dict
        Mapping from layer number to TaskVarianceResults
    plotting_data : dict
        Data formatted for plotting
    """
    # Load config/model
    _, _sampler_orig, config = nu.load_everything("coin", exp_name)
    
    if step is None:
        step = config.training.num_epochs
    
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval()
    model.to(config.device)
    
    # Get sampler with specified n_minor and n_ood
    sampler, k_minor = get_new_sampler(exp_name, n_minor, n_ood)
    
    if not getattr(sampler, "pad", False):
        raise ValueError("This function requires padded coin sequences (sampler.pad must be True)")
    
    # Set default layers if not provided
    if layers is None:
        layers = list(range(len(model.layers)))
    
    if verbose:
        logger.info(f"Computing hiddens for coin exp: {exp_name}")
        logger.info(f"Layers: {layers}, Batch size: {batch_size}, n_tasks: {sampler.n_major_tasks + sampler.n_minor_tasks}")
    
    # Compute hiddens (each task gets its own batch)
    all_hiddens, position_info = compute_hiddens_multi_coin_latent(
        config=config,
        model=model,
        sampler=sampler,
        layers=layers,
        batch_size=batch_size,
        positions_of_interest=positions_of_interest,
    )
    
    if verbose:
        logger.info(f"Computed hiddens shape: {all_hiddens.shape}")
        logger.info("Computing task variance...")
    
    # Compute task variance (same logic as linear)
    results_dict = compute_task_variance_multi_layer(
        all_hiddens=all_hiddens,
        positions_of_interest=positions_of_interest,
        layers=layers,
        eps=eps,
    )
    
    # Extract plotting data
    plotting_data = extract_plotting_data_multi_layer(results_dict)
    
    if verbose:
        logger.info(f"Computed variance for {len(results_dict)} layers")
    
    return all_hiddens, position_info, results_dict, plotting_data


def get_task_variance_latent(
    exp_name: str,
    layers: Optional[list] = None,
    batch_size: int = 96,
    positions_of_interest: Optional[list] = None,
    n_minor: int = 256,
    n_ood: int = 40,
    step: Optional[int] = None,
    verbose: bool = False,
    eps: float = 1e-8,
) -> tuple:
    """
    Get hidden representations and compute task variance for Latent task.
    
    This function:
    1. Loads model and creates sampler with n_minor and n_ood tasks
    2. For each task, samples a new batch of data
    3. Extracts hiddens at padded positions (odd indices: 1, 3, 5, ...)
    4. Computes task variance: variance of batch-averaged hiddens across tasks
    
    Parameters:
    -----------
    exp_name : str
        Experiment name (e.g., "train_...")
    layers : list, optional
        Which layers to analyze. If None, uses all layers.
    batch_size : int, default=96
        Batch size for sampling
    positions_of_interest : list, optional
        Position indices (0 to seq_len-2) to analyze. If None, uses all positions.
        These map to padded sequence positions [1, 3, 5, ...]
    n_minor : int, default=256
        Number of minor tasks to include
    n_ood : int, default=40
        Number of out-of-distribution (OOD) tasks to include
    step : int, optional
        Step for checkpoint loading. If None, uses final checkpoint.
    verbose : bool, default=False
        Whether to print progress messages
    eps : float, default=1e-8
        Small epsilon for numerical stability in normalization
    
    Returns:
    --------
    all_hiddens : torch.Tensor
        Shape: (L, n_tasks, n_positions, batch_size, n_embd)
    position_info : dict
        Information about positions analyzed
    results_dict : dict
        Mapping from layer number to TaskVarianceResults
    plotting_data : dict
        Data formatted for plotting
    """
    # Load config/model
    _, _sampler_orig, config = nu.load_everything("latent", exp_name)
    
    if step is None:
        step = config.training.num_epochs
    
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval()
    model.to(config.device)
    
    # Get sampler with specified n_minor and n_ood
    sampler, k_minor, n_tasks = get_latent_sampler(exp_name, n_minor, n_ood)
    
    if not getattr(sampler, "pad", False):
        raise ValueError("This function requires padded latent sequences (sampler.pad must be True)")
    
    # Set default layers if not provided
    if layers is None:
        layers = list(range(len(model.layers)))
    
    if verbose:
        logger.info(f"Computing hiddens for latent exp: {exp_name}")
        logger.info(f"Layers: {layers}, Batch size: {batch_size}, n_tasks: {sampler.n_major_tasks + sampler.n_minor_tasks}")
    
    # Compute hiddens (each task gets its own batch)
    all_hiddens, position_info = compute_hiddens_multi_coin_latent(
        config=config,
        model=model,
        sampler=sampler,
        layers=layers,
        batch_size=batch_size,
        positions_of_interest=positions_of_interest,
    )
    
    if verbose:
        logger.info(f"Computed hiddens shape: {all_hiddens.shape}")
        logger.info("Computing task variance...")
    
    # Compute task variance (same logic as linear)
    results_dict = compute_task_variance_multi_layer(
        all_hiddens=all_hiddens,
        positions_of_interest=positions_of_interest,
        layers=layers,
        eps=eps,
    )
    
    # Extract plotting data
    plotting_data = extract_plotting_data_multi_layer(results_dict)
    
    if verbose:
        logger.info(f"Computed variance for {len(results_dict)} layers")
    
    return all_hiddens, position_info, results_dict, plotting_data


def get_token_conditioned_hiddens_coin(
    exp_name: str,
    layers: Optional[list] = None,
    batch_size: int = 64,
    positions_of_interest: Optional[list] = None,
    max_unique_tokens: Optional[int] = None,
    n_minor: int = 64,
    n_ood: int = 30,
    step: Optional[int] = None,
    verbose: bool = False,
) -> tuple:
    """
    Get token-conditioned hidden representations for Coin task.
    
    This function fixes tokens at specific positions and extracts hiddens at the following PAD tokens.
    
    Parameters:
    -----------
    exp_name : str
        Experiment name (e.g., "train_...")
    layers : list, optional
        Which layers to analyze. If None, uses all layers.
    batch_size : int, default=64
        Batch size for sampling
    positions_of_interest : list, optional
        Position indices (0 to seq_len-2) to analyze. If None, uses all positions.
        These are indices into the real token positions [0, 2, 4, ...]
    max_unique_tokens : int, optional
        Maximum number of unique tokens per position
    n_minor : int, default=64
        Number of minor tasks to include
    n_ood : int, default=30
        Number of out-of-distribution (OOD) tasks to include
    step : int, optional
        Step for checkpoint loading. If None, uses final checkpoint.
    verbose : bool, default=False
        Whether to print progress messages
    
    Returns:
    --------
    all_hiddens : torch.Tensor
        Shape: (L, n_positions, n_unique_tokens, n_tasks, batch_size, n_embd)
    token_info : dict
        Information about tokens used
    """
    # Load config/model
    _, _sampler_orig, config = nu.load_everything("coin", exp_name)
    
    if step is None:
        step = config.training.num_epochs
    
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval()
    model.to(config.device)
    
    # Get sampler with specified n_minor and n_ood
    sampler, k_minor = get_new_sampler(exp_name, n_minor, n_ood)
    
    if not getattr(sampler, "pad", False):
        raise ValueError("This function requires padded coin sequences (sampler.pad must be True)")
    
    # Set default layers if not provided
    if layers is None:
        layers = list(range(len(model.layers)))
    
    if verbose:
        logger.info(f"Computing token-conditioned hiddens for coin exp: {exp_name}")
        logger.info(f"Layers: {layers}, Batch size: {batch_size}, n_tasks: {sampler.n_major_tasks + sampler.n_minor_tasks}")
    
    # Compute token-conditioned hiddens
    all_hiddens, token_info = compute_hiddens_token_conditioned_coin_latent(
        config=config,
        model=model,
        sampler=sampler,
        layers=layers,
        batch_size=batch_size,
        positions_of_interest=positions_of_interest,
        max_unique_tokens=max_unique_tokens,
    )
    
    if verbose:
        logger.info(f"Computed hiddens shape: {all_hiddens.shape}")
        logger.info(f"Token info - positions: {token_info['positions']}")
        logger.info(f"Token info - n_unique_tokens: {token_info['n_unique_tokens']}")
    
    return all_hiddens, token_info


def get_token_conditioned_hiddens_latent(
    exp_name: str,
    layers: Optional[list] = None,
    batch_size: int = 96,
    positions_of_interest: Optional[list] = None,
    max_unique_tokens: Optional[int] = None,
    n_minor: int = 256,
    n_ood: int = 40,
    step: Optional[int] = None,
    verbose: bool = False,
) -> tuple:
    """
    Get token-conditioned hidden representations for Latent task.
    
    This function fixes tokens at specific positions and extracts hiddens at the following PAD tokens.
    
    Parameters:
    -----------
    exp_name : str
        Experiment name (e.g., "train_...")
    layers : list, optional
        Which layers to analyze. If None, uses all layers.
    batch_size : int, default=96
        Batch size for sampling
    positions_of_interest : list, optional
        Position indices (0 to seq_len-2) to analyze. If None, uses all positions.
        These are indices into the real token positions [0, 2, 4, ...]
    max_unique_tokens : int, optional
        Maximum number of unique tokens per position
    n_minor : int, default=256
        Number of minor tasks to include
    n_ood : int, default=40
        Number of out-of-distribution (OOD) tasks to include
    step : int, optional
        Step for checkpoint loading. If None, uses final checkpoint.
    verbose : bool, default=False
        Whether to print progress messages
    
    Returns:
    --------
    all_hiddens : torch.Tensor
        Shape: (L, n_positions, n_unique_tokens, n_tasks, batch_size, n_embd)
    token_info : dict
        Information about tokens used
    """
    # Load config/model
    _, _sampler_orig, config = nu.load_everything("latent", exp_name)
    
    if step is None:
        step = config.training.num_epochs
    
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval()
    model.to(config.device)
    
    # Get sampler with specified n_minor and n_ood
    sampler, k_minor, n_tasks = get_latent_sampler(exp_name, n_minor, n_ood)
    
    if not getattr(sampler, "pad", False):
        raise ValueError("This function requires padded latent sequences (sampler.pad must be True)")
    
    # Set default layers if not provided
    if layers is None:
        layers = list(range(len(model.layers)))
    
    if verbose:
        logger.info(f"Computing token-conditioned hiddens for latent exp: {exp_name}")
        logger.info(f"Layers: {layers}, Batch size: {batch_size}, n_tasks: {sampler.n_major_tasks + sampler.n_minor_tasks}")
    
    # Compute token-conditioned hiddens
    all_hiddens, token_info = compute_hiddens_token_conditioned_coin_latent(
        config=config,
        model=model,
        sampler=sampler,
        layers=layers,
        batch_size=batch_size,
        positions_of_interest=positions_of_interest,
        max_unique_tokens=max_unique_tokens,
    )
    
    if verbose:
        logger.info(f"Computed hiddens shape: {all_hiddens.shape}")
        logger.info(f"Token info - positions: {token_info['positions']}")
        logger.info(f"Token info - n_unique_tokens: {token_info['n_unique_tokens']}")
    
    return all_hiddens, token_info


def compute_stable_rank_at_padded_positions(
    exp_name: str,
    task_name: Optional[str] = None,
    B: int = 64,
    step: Optional[int] = None,
    n_minor: Optional[int] = None,
    n_ood: int = 0,
    verbose: bool = False,
    task_chunk_size: Optional[int] = None,
) -> dict:
    """
    Collect hidden representations for a given exp_name and compute the stable rank 
    of the hidden representations at every padded position.
    
    Parameters:
    -----------
    exp_name : str
        Experiment name (e.g., "train_...")
    task_name : str, optional
        Task name ("linear", "coin", "latent", "dyck"). If None, will try to infer it.
    B : int, default=64
        Batch size for sampling
    step : int, optional
        Step for checkpoint loading. If None, uses final checkpoint.
    n_minor : int, optional
        Number of minor tasks to use. If None, uses all available minor tasks.
        If -1, uses no minor tasks (only OOD if n_ood > 0).
    n_ood : int, default=0
        Number of OOD tasks to include
    verbose : bool, default=False
        Whether to print progress messages
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'stable_ranks': torch.Tensor of shape (n_layers, n_positions)
          Stable rank at each layer and each padded position
        - 'task_name': str, the task name used
        - 'n_layers': int, number of layers
        - 'n_positions': int, number of padded positions
        - 'k_minor': int, number of minor tasks used
    """
    # Try to infer task_name if not provided
    if task_name is None:
        task_names = ["linear", "coin", "latent", "dyck"]
        task_name = None
        for tn in task_names:
            try:
                if tn == "linear":
                    _, _, config = load_model_task_config(exp_name)
                else:
                    _, _, config = nu.load_everything(tn, exp_name)
                task_name = tn
                if verbose:
                    logger.info(f"Inferred task_name: {task_name}")
                break
            except Exception:
                continue
        
        if task_name is None:
            raise ValueError(f"Could not infer task_name from exp_name: {exp_name}. Please specify task_name.")
    
    # Handle n_minor parameter
    if n_minor is None:
        # Use all available minor tasks
        n_minor = 1000000  # Large number to get all minor tasks
    elif n_minor == -1:
        # Use no minor tasks (placeholder)
        n_minor = 0
    
    if verbose:
        logger.info(f"Collecting hiddens for exp_name: {exp_name}, task_name: {task_name}")
        logger.info(f"Using n_minor={n_minor} and n_ood={n_ood}")
    
    # For latent task, use coin-style approach
    if task_name == "latent":
        # Load config/model
        _, _sampler_orig, config = nu.load_everything("latent", exp_name)
        
        if step is None:
            step = config.training.num_epochs
        
        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
        model.eval()
        model.to(config.device)
        
        # Get latent sampler with specified n_minor and n_ood
        # get_latent_sampler handles -1 internally
        sampler, k_minor, _ = get_latent_sampler(exp_name, n_minor=n_minor, n_ood=n_ood)
        
        if not getattr(sampler, "pad", False):
            raise ValueError("This function requires padded latent sequences (sampler.pad must be True)")
        
        if verbose:
            logger.info("Computing hiddens using coin-style approach...")
        
        # Determine task chunk size (default: process all at once, or use provided value)
        n_tasks_total = sampler.n_major_tasks + sampler.n_minor_tasks
        if task_chunk_size is None:
            task_chunk_size = n_tasks_total  # Process all at once
        else:
            task_chunk_size = min(task_chunk_size, n_tasks_total)
        
        if verbose and task_chunk_size < n_tasks_total:
            logger.info(f"Processing {n_tasks_total} tasks in chunks of {task_chunk_size}")
        
        # Compute hiddens using compute_hiddens_multi_coin_latent (works for both coin and latent)
        # If chunking is needed, we'll process tasks in chunks
        if task_chunk_size >= n_tasks_total:
            # Process all tasks at once (original behavior)
            hiddens, position_info = compute_hiddens_multi_coin_latent(
                config=config,
                model=model,
                sampler=sampler,
                layers=None,  # All layers
                batch_size=B,
                positions_of_interest=None,  # All positions
            )
        else:
            # Process tasks in chunks to reduce memory
            device = config.device
            seq_len = sampler.seq_len
            n_embd = config.model.emb_dim
            layers = list(range(len(model.layers)))
            L = len(layers)
            n_positions = seq_len - 1
            padded_positions = [2 * p + 1 for p in range(n_positions)]
            task_pos = torch.tensor(padded_positions, device=device, dtype=torch.long)
            
            # Initialize output tensor on CPU to save GPU memory
            output_shape = (L, n_tasks_total, n_positions, B, n_embd)
            hiddens = torch.empty(output_shape, dtype=torch.float32, device='cpu')
            
            # Process tasks in chunks
            for chunk_start in range(0, n_tasks_total, task_chunk_size):
                chunk_end = min(chunk_start + task_chunk_size, n_tasks_total)
                task_chunk = list(range(chunk_start, chunk_end))
                
                if verbose:
                    logger.info(f"  Processing tasks {chunk_start} to {chunk_end-1} ({len(task_chunk)} tasks)")
                
                # Process this chunk of tasks
                chunk_hiddens_list = []
                for task_idx in task_chunk:
                    demo_data, _ = sampler.generate(
                        mode="testing", task=task_idx, num_samples=B
                    )
                    demo_data = demo_data.to(device)
                    
                    # Extract hiddens
                    chunk_hiddens = extract_hidden_multi_coin_latent(
                        model=model,
                        batch_data=demo_data,
                        layers=layers,
                        task_pos=task_pos,
                    )
                    # Reshape: (L, B, P, D) -> (L, P, B, D) -> move to CPU
                    chunk_hiddens = chunk_hiddens.permute(0, 2, 1, 3).cpu()
                    chunk_hiddens_list.append(chunk_hiddens)
                    
                    # Clear GPU memory after each task
                    del demo_data, chunk_hiddens
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Stack chunk results and store
                chunk_hiddens_stacked = torch.stack(chunk_hiddens_list, dim=1)  # (L, chunk_size, P, B, D)
                hiddens[:, chunk_start:chunk_end] = chunk_hiddens_stacked
                
                # Clean up chunk
                del chunk_hiddens_list, chunk_hiddens_stacked
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            position_info = {
                'positions': list(range(n_positions)),
                'padded_positions': padded_positions,
                'seq_len': seq_len,
            }
        
        # hiddens shape: (L, n_tasks, n_positions, B, D)
        L, K, T, B_actual, D = hiddens.shape
        n_layers = L
        n_positions = T
        
        if verbose:
            logger.info(f"Retrieved hiddens with shape: {hiddens.shape}")
            logger.info(f"Number of minor tasks used: {k_minor}")
        
        # Compute stable rank for each layer and position
        # For each position, reshape as (n_tasks*B, D) = (-1, D)
        stable_ranks = torch.zeros(n_layers, n_positions)
        
        # hiddens is already on CPU if chunked, otherwise move to CPU
        if hiddens.device.type != 'cpu':
            hiddens = hiddens.cpu()
        
        for l in range(n_layers):
            for t in range(n_positions):
                # Get hiddens at layer l, position t: (K, B, D)
                h = hiddens[l, :, t, :, :]  # (K, B, D)
                # Reshape to (-1, D) where -1 flattens all tasks and batch dimensions
                h_flat = h.reshape(-1, D).float().numpy()
                # Compute stable rank
                stable_ranks[l, t] = torch.tensor(stable_rank(h_flat), dtype=torch.float32)
        
        # Clean up GPU memory
        # Move model to CPU before deletion to ensure it's off GPU
        if hasattr(model, 'cpu'):
            model = model.cpu()
        del hiddens, model, sampler
        if 'config' in locals():
            del config
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    else:
        # For other tasks, use _get_hiddens
        hiddens, k_minor = _get_hiddens(
            task_name=task_name,
            exp_name=exp_name,
            n_minor=n_minor,
            n_ood=n_ood,
            B=B,
            step=step,
            verbose=verbose,
        )
        
        if verbose:
            logger.info(f"Retrieved hiddens with shape: {hiddens.shape}")
            logger.info(f"Number of minor tasks used: {k_minor}")
        
        # Handle different task types and their hiddens shapes
        if task_name == "latent":
            # This branch should not be reached, but kept for safety
            raise NotImplementedError("Latent task should use coin-style approach above")
    
        elif task_name == "linear":
            # hiddens shape: (n_layers, n_tasks, n_points, B, D)
            # n_points are the padded positions
            L, K, T, B_actual, D = hiddens.shape
            n_layers = L
            n_positions = T
            
            stable_ranks = torch.zeros(n_layers, n_positions)
            
            # Move hiddens to CPU to free GPU memory
            hiddens = hiddens.cpu()
            
            for l in range(n_layers):
                for t in range(n_positions):
                    # Get hiddens at layer l, position t: (K, B, D)
                    h = hiddens[l, :, t, :, :]  # (K, B, D)
                    # Reshape to (K*B, D) and convert to float32 for numpy linalg
                    h_flat = h.reshape(-1, D).float().numpy()
                    # Compute stable rank
                    stable_ranks[l, t] = torch.tensor(stable_rank(h_flat), dtype=torch.float32)
        
        elif task_name in ["coin", "dyck"]:
            # hiddens shape: (n_layers, n_tasks, n_positions, B, D)
            # n_positions are the padded positions
            L, K, T, B_actual, D = hiddens.shape
            n_layers = L
            n_positions = T
            
            stable_ranks = torch.zeros(n_layers, n_positions)
            
            # Move hiddens to CPU to free GPU memory
            hiddens = hiddens.cpu()
            
            for l in range(n_layers):
                for t in range(n_positions):
                    # Get hiddens at layer l, position t: (K, B, D)
                    h = hiddens[l, :, t, :, :]  # (K, B, D)
                    # Reshape to (K*B, D) and convert to float32 for numpy linalg
                    h_flat = h.reshape(-1, D).float().numpy()
                    # Compute stable rank
                    stable_ranks[l, t] = torch.tensor(stable_rank(h_flat), dtype=torch.float32)
        
        else:
            raise ValueError(f"Unsupported task_name: {task_name}")
        
        # Clean up GPU memory (hiddens already moved to CPU above)
        del hiddens
        # Also clean up model and sampler if they exist
        if 'model' in locals():
            del model
        if 'sampler' in locals():
            del sampler
        if 'config' in locals():
            del config
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    if verbose:
        logger.info(f"Computed stable ranks with shape: {stable_ranks.shape}")
    
    return {
        'stable_ranks': stable_ranks,
        'task_name': task_name,
        'n_layers': n_layers,
        'n_positions': n_positions,
        'k_minor': k_minor,
    }


def compute_logits_multi_coin_latent(
    config,
    model: nn.Module,
    sampler,
    batch_size: int = 64,
    positions_of_interest: Sequence[int] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute logits for Coin or Latent tasks at specified padded positions.
    
    For each task, samples a new batch of data and extracts logits at specified positions.
    
    Parameters:
    -----------
    config : ConfigDict
        Configuration object
    model : nn.Module
        The model (should have forward method that returns logits)
    sampler : Coins or LatentMarkov
        The sampler (should have generate() method and pad attribute)
    batch_size : int, default=64
        Batch size for sampling
    positions_of_interest : Sequence[int], optional
        Position indices (0 to seq_len-2) to analyze. If None, uses all positions.
        These are indices into the padded sequence positions [1, 3, 5, ...]
        So position 0 means the first padded token (at index 1), position 1 means second (at index 3), etc.
    
    Returns:
    --------
    all_logits : torch.Tensor
        Shape: (n_tasks, n_positions, batch_size, vocab_size)
    position_info : dict
        Information about positions:
        - 'positions': list of position indices analyzed
        - 'padded_positions': list of actual sequence positions (odd indices)
        - 'seq_len': sequence length
    """
    device = config.device
    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    seq_len = sampler.seq_len
    vocab_size = config.vocab_size
    
    # Check padding
    if not getattr(sampler, "pad", False):
        raise ValueError("This function requires padded sequences (sampler.pad must be True)")
    
    # Determine positions of interest
    if positions_of_interest is None:
        positions_of_interest = list(range(seq_len - 1))
    else:
        positions_of_interest = list(positions_of_interest)
        if not all(0 <= p < seq_len - 1 for p in positions_of_interest):
            raise ValueError(f"All positions must be in [0, {seq_len-2}]")
    
    n_positions = len(positions_of_interest)
    
    # Map position indices to actual sequence positions (odd indices: 1, 3, 5, ...)
    # position i -> sequence position 2*i + 1
    padded_positions = [2 * p + 1 for p in positions_of_interest]
    
    # Initialize output tensor
    output_shape = (n_tasks, n_positions, batch_size, vocab_size)
    all_logits = torch.empty(output_shape, dtype=torch.float32, device=device)
    
    # For each task, sample new data and extract logits
    for task_idx in range(n_tasks):
        # Sample new batch for this task
        demo_data, _ = sampler.generate(
            mode="testing", task=task_idx, num_samples=batch_size
        )
        demo_data = demo_data.to(device)
        
        # Forward pass to get logits
        with torch.no_grad():
            logits = model(demo_data)  # (B, seq_len, vocab_size)
        
        # Extract logits at padded positions
        for pos_idx, seq_pos in enumerate(padded_positions):
            all_logits[task_idx, pos_idx] = logits[:, seq_pos, :].detach()  # (B, vocab_size)
    
    position_info = {
        'positions': positions_of_interest,
        'padded_positions': padded_positions,
        'seq_len': seq_len,
    }
    
    return all_logits.detach().cpu(), position_info


def compute_stable_rank_logits_at_padded_positions(
    exp_name: str,
    task_name: Optional[str] = None,
    B: int = 64,
    step: Optional[int] = None,
    n_minor: Optional[int] = None,
    n_ood: int = 0,
    verbose: bool = False,
) -> dict:
    """
    Collect logits for a given exp_name and compute the stable rank 
    of the logits at every padded position.
    
    Parameters:
    -----------
    exp_name : str
        Experiment name (e.g., "train_...")
    task_name : str, optional
        Task name ("linear", "coin", "latent", "dyck"). If None, will try to infer it.
    B : int, default=64
        Batch size for sampling
    step : int, optional
        Step for checkpoint loading. If None, uses final checkpoint.
    n_minor : int, optional
        Number of minor tasks to use. If None, uses all available minor tasks.
        If -1, uses no minor tasks (only OOD if n_ood > 0).
    n_ood : int, default=0
        Number of OOD tasks to include
    verbose : bool, default=False
        Whether to print progress messages
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'stable_ranks': torch.Tensor of shape (n_positions,)
          Stable rank at each padded position
        - 'task_name': str, the task name used
        - 'n_positions': int, number of padded positions
        - 'k_minor': int, number of minor tasks used
    """
    # Try to infer task_name if not provided
    if task_name is None:
        task_names = ["linear", "coin", "latent", "dyck"]
        task_name = None
        for tn in task_names:
            try:
                if tn == "linear":
                    _, _, config = load_model_task_config(exp_name)
                else:
                    _, _, config = nu.load_everything(tn, exp_name)
                task_name = tn
                if verbose:
                    logger.info(f"Inferred task_name: {task_name}")
                break
            except Exception:
                continue
        
        if task_name is None:
            raise ValueError(f"Could not infer task_name from exp_name: {exp_name}. Please specify task_name.")
    
    # Handle n_minor parameter
    if n_minor is None:
        # Use all available minor tasks
        n_minor = 1000000  # Large number to get all minor tasks
    elif n_minor == -1:
        # Use no minor tasks (placeholder)
        n_minor = 0
    
    if verbose:
        logger.info(f"Collecting logits for exp_name: {exp_name}, task_name: {task_name}")
        logger.info(f"Using n_minor={n_minor} and n_ood={n_ood}")
    
    # For latent and coin tasks, use coin-style approach
    if task_name in ["latent", "coin"]:
        # Load config/model
        _, _sampler_orig, config = nu.load_everything(task_name, exp_name)
        
        if step is None:
            step = config.training.num_epochs
        
        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
        model.eval()
        model.to(config.device)
        
        # Get sampler with specified n_minor and n_ood
        if task_name == "latent":
            sampler, k_minor, _ = get_latent_sampler(exp_name, n_minor=n_minor, n_ood=n_ood)
        else:  # coin
            sampler, k_minor = get_new_sampler(exp_name, n_minor, n_ood)
        
        if not getattr(sampler, "pad", False):
            raise ValueError(f"This function requires padded {task_name} sequences (sampler.pad must be True)")
        
        if verbose:
            logger.info("Computing logits using coin-style approach...")
        
        # Compute logits using compute_logits_multi_coin_latent
        logits, position_info = compute_logits_multi_coin_latent(
            config=config,
            model=model,
            sampler=sampler,
            batch_size=B,
            positions_of_interest=None,  # All positions
        )
        
        # logits shape: (n_tasks, n_positions, B, vocab_size)
        K, T, B_actual, vocab_size = logits.shape
        n_positions = T
        
        if verbose:
            logger.info(f"Retrieved logits with shape: {logits.shape}")
            logger.info(f"Number of minor tasks used: {k_minor}")
        
        # Compute stable rank for each position
        # For each position, reshape as (n_tasks*B, vocab_size) = (-1, vocab_size)
        stable_ranks = torch.zeros(n_positions)
        
        for t in range(n_positions):
            # Get logits at position t: (K, B, vocab_size)
            logits_t = logits[:, t, :, :]  # (K, B, vocab_size)
            # Reshape to (-1, vocab_size) where -1 flattens all tasks and batch dimensions
            logits_flat = logits_t.reshape(-1, vocab_size).cpu().float().numpy()
            # Compute stable rank
            stable_ranks[t] = torch.tensor(stable_rank(logits_flat), dtype=torch.float32)
    
    elif task_name == "dyck":
        # Similar to coin/latent
        _, _sampler_orig, config = nu.load_everything("dyck", exp_name)
        
        if step is None:
            step = config.training.num_epochs
        
        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
        model.eval()
        model.to(config.device)
        
        sampler, k_minor = get_dyck_sampler(exp_name, n_minor, n_ood)
        
        if not getattr(sampler, "pad", False):
            raise ValueError("This function requires padded dyck sequences (sampler.pad must be True)")
        
        if verbose:
            logger.info("Computing logits for dyck task...")
        
        # For dyck, we can use similar approach
        # This is a simplified version - may need adjustment based on dyck model structure
        seq_len = sampler.seq_len
        vocab_size = config.vocab_size
        n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
        
        # Determine positions
        positions_of_interest = list(range(seq_len - 1))
        padded_positions = [2 * p + 1 for p in positions_of_interest]
        n_positions = len(positions_of_interest)
        
        all_logits = []
        for task_idx in range(n_tasks):
            demo_data, _ = sampler.generate(
                mode="testing", task=task_idx, num_samples=B
            )
            demo_data = demo_data.to(config.device)
            
            with torch.no_grad():
                logits = model(demo_data)  # (B, seq_len, vocab_size)
            
            # Extract logits at padded positions
            task_logits = logits[:, padded_positions, :].detach().cpu()  # (B, n_positions, vocab_size)
            all_logits.append(task_logits)
        
        # Stack: (n_tasks, B, n_positions, vocab_size) -> (n_tasks, n_positions, B, vocab_size)
        logits = torch.stack(all_logits, dim=0).permute(0, 2, 1, 3)
        
        # Compute stable rank for each position
        stable_ranks = torch.zeros(n_positions)
        for t in range(n_positions):
            logits_t = logits[:, t, :, :]  # (K, B, vocab_size)
            logits_flat = logits_t.reshape(-1, vocab_size).cpu().float().numpy()
            stable_ranks[t] = torch.tensor(stable_rank(logits_flat), dtype=torch.float32)
    
    elif task_name == "linear":
        # Linear task doesn't have logits in the same way (it's regression)
        # We could compute stable rank of the output predictions, but that's different
        raise NotImplementedError(
            "Linear task uses regression outputs, not logits. "
            "Use compute_stable_rank_at_padded_positions for hidden representations instead."
        )
    
    else:
        raise ValueError(f"Unsupported task_name: {task_name}")
    
    if verbose:
        logger.info(f"Computed stable ranks for logits with shape: {stable_ranks.shape}")
    
    return {
        'stable_ranks': stable_ranks,
        'task_name': task_name,
        'n_positions': n_positions,
        'k_minor': k_minor,
    }


def plot_stable_rank_vs_positions(
    results: dict,
    layers: Optional[list] = None,
    average_layers: bool = False,
    backend: str = "matplotlib",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot stable rank against padded positions.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from compute_stable_rank_at_padded_positions
    layers : list, optional
        Which layers to plot. If None, plots all layers. If average_layers=True, this is ignored.
    average_layers : bool, default=False
        If True, plot the average stable rank across all layers. If False, plot individual layers.
    backend : str, default="matplotlib"
        Plotting backend: "matplotlib" or "plotly"
    figsize : tuple, default=(10, 6)
        Figure size for matplotlib (ignored for plotly)
    save_path : str, optional
        Path to save the figure
    show : bool, default=True
        Whether to display the plot
    
    Returns:
    --------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        The figure object (for further customization)
    """
    stable_ranks = results['stable_ranks'].numpy()
    task_name = results['task_name']
    n_layers, n_positions = stable_ranks.shape
    k_minor = results['k_minor']
    
    positions = np.arange(n_positions)
    
    if backend == "plotly" and PLOTLY_AVAILABLE:
        fig = go.Figure()
        
        if average_layers:
            # Plot average across all layers
            avg_stable_rank = stable_ranks.mean(axis=0)
            fig.add_trace(go.Scatter(
                x=positions,
                y=avg_stable_rank,
                mode='lines+markers',
                name='Average across layers',
                line=dict(width=2, color='blue'),
                marker=dict(size=6)
            ))
        else:
            # Plot individual layers
            if layers is None:
                layers = list(range(n_layers))
            
            # Use different colors for different layers
            colors = plt.cm.tab20(np.linspace(0, 1, len(layers))) if MATPLOTLIB_AVAILABLE else None
            
            for i, layer_idx in enumerate(layers):
                if layer_idx < n_layers:
                    color = f"rgb({int(colors[i][0]*255)}, {int(colors[i][1]*255)}, {int(colors[i][2]*255)})" if colors is not None else None
                    fig.add_trace(go.Scatter(
                        x=positions,
                        y=stable_ranks[layer_idx, :],
                        mode='lines+markers',
                        name=f'Layer {layer_idx}',
                        line=dict(width=2, color=color),
                        marker=dict(size=4)
                    ))
        
        fig.update_layout(
            title=f'Stable Rank vs Padded Positions<br>Task: {task_name}, Minor Tasks: {k_minor}',
            xaxis_title='Padded Position Index',
            yaxis_title='Stable Rank',
            hovermode='x unified',
            template='plotly_white',
            width=1000,
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved plot to {save_path}")
        
        if show:
            fig.show()
        
        return fig
    
    elif backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=figsize)
        
        if average_layers:
            # Plot average across all layers
            avg_stable_rank = stable_ranks.mean(axis=0)
            ax.plot(positions, avg_stable_rank, 'o-', linewidth=2, markersize=6, 
                   label='Average across layers', color='blue')
        else:
            # Plot individual layers
            if layers is None:
                layers = list(range(n_layers))
            
            # Use colormap for different layers
            cmap = plt.cm.tab20
            colors = [cmap(i / max(len(layers), 1)) for i in range(len(layers))]
            
            for i, layer_idx in enumerate(layers):
                if layer_idx < n_layers:
                    ax.plot(positions, stable_ranks[layer_idx, :], 'o-', 
                           linewidth=1.5, markersize=4, 
                           label=f'Layer {layer_idx}', 
                           color=colors[i], alpha=0.7)
        
        ax.set_xlabel('Padded Position Index', fontsize=12)
        ax.set_ylabel('Stable Rank', fontsize=12)
        ax.set_title(f'Stable Rank vs Padded Positions\nTask: {task_name}, Minor Tasks: {k_minor}', 
                    fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    else:
        if backend == "plotly" and not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available, falling back to matplotlib")
            backend = "matplotlib"
        if backend == "matplotlib" and not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is not installed. Install it with: pip install matplotlib")
        
        # Recursive call with corrected backend
        return plot_stable_rank_vs_positions(
            results, layers, average_layers, backend, figsize, save_path, show
        )


def plot_max_stable_rank_vs_k(
    task_name: str,
    k_values: list,
    vocab_size: Optional[int] = None,
    B: int = 64,
    step: Optional[int] = None,
    n_ood: int = 0,
    backend: str = "matplotlib",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
    verbose: bool = False,
    chunk_size: Optional[int] = None,
    task_chunk_size: Optional[int] = 32,
    layer: Optional[int] = None,
) -> dict:
    """
    Compute stable ranks for different numbers of tasks (parameterized by k = log2(number of tasks))
    and plot the maximum stable rank at the specified layer against k.
    
    Parameters:
    -----------
    task_name : str
        Task name ("linear", "coin", "latent", "dyck")
    k_values : list
        List of k values where number of tasks = 2^k
    vocab_size : int, optional
        Vocabulary size (for non-linear tasks)
    B : int, default=64
        Batch size for sampling
    step : int, optional
        Step for checkpoint loading. If None, uses final checkpoint.
    n_ood : int, default=0
        Number of OOD tasks to include
    backend : str, default="matplotlib"
        Plotting backend: "matplotlib" or "plotly"
    figsize : tuple, default=(10, 6)
        Figure size for matplotlib (ignored for plotly)
    save_path : str, optional
        Path to save the figure
    show : bool, default=True
        Whether to display the plot
    verbose : bool, default=False
        Whether to print progress messages
    chunk_size : int, optional, default=None
        Number of k values to process in each chunk before clearing memory.
        If None, automatically set to max(1, len(k_values) // 4) to reduce memory usage.
        Smaller values use less memory but may be slower.
    task_chunk_size : int, optional, default=32
        Number of tasks to process at once within each k value computation.
        Smaller values use less memory. If None, processes all tasks at once.
    layer : int, optional
        Layer index to use. If None, uses the final layer (default behavior).
        Layer indices are 0-based.
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'k_values': list of k values
        - 'max_stable_ranks': list of maximum stable ranks at specified layer for each k
        - 'n_tasks': list of number of tasks (2^k) for each k
        - 'layer': int, the layer index used
        - 'fig': matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    max_stable_ranks = []
    n_tasks_list = []
    
    # Get config to determine layer count and validate layer index
    if task_name == "linear":
        _, _, config = load_model_task_config(get_exp_name(task_name, k_values[0], vocab_size=vocab_size))
        n_layers = config.model.n_layer
        del config  # Free memory
    else:
        _, _, config = nu.load_everything(task_name, get_exp_name(task_name, k_values[0], vocab_size=vocab_size))
        n_layers = config.model.num_layers
        del config  # Free memory
    
    # Determine which layer to use
    if layer is None:
        layer_idx = n_layers - 1  # Final layer
    else:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"Layer index {layer} is out of range. Valid layer indices are 0 to {n_layers-1}.")
        layer_idx = layer
    
    # Set default chunk_size if not provided - make it adaptive based on max k
    # For large k values, process one at a time to avoid OOM
    max_k = max([k for k in k_values if k >= 0], default=0)
    max_n_tasks = 2 ** max_k if max_k >= 0 else 0
    
    if chunk_size is None:
        # Adaptive chunking: smaller chunks for larger k values
        if max_n_tasks >= 512:  # k >= 9
            chunk_size = 1  # Process one k at a time for very large k
        elif max_n_tasks >= 256:  # k >= 8
            chunk_size = max(1, len(k_values) // 8)  # Smaller chunks
        elif max_n_tasks >= 128:  # k >= 7
            chunk_size = max(1, len(k_values) // 4)  # Medium chunks
        else:
            chunk_size = max(1, len(k_values) // 4)  # Default: 4 chunks
        
        if verbose:
            logger.info(f"Auto-set chunk_size to {chunk_size} for {len(k_values)} k values (max_k={max_k}, max_n_tasks={max_n_tasks})")
    
    if verbose:
        logger.info(f"Computing stable ranks for {len(k_values)} different k values")
        logger.info(f"Using layer index: {layer_idx} (out of {n_layers} layers)")
        logger.info(f"Processing in chunks of {chunk_size} k values")
    
    # Process k values in chunks to reduce memory usage
    import gc
    for chunk_start in range(0, len(k_values), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(k_values))
        k_chunk = k_values[chunk_start:chunk_end]
        
        if verbose:
            logger.info(f"Processing chunk {chunk_start//chunk_size + 1}/{(len(k_values)-1)//chunk_size + 1}: k values {k_chunk}")
        
        for k in k_chunk:
            # Handle k=-1 case (no minor tasks)
            if k == -1:
                n_tasks = 0
                n_minor = -1  # No minor tasks (placeholder)
            else:
                n_tasks = 2 ** k
                n_minor = n_tasks
            
            n_tasks_list.append(n_tasks)
            
            if verbose:
                logger.info(f"Processing k={k} (n_tasks={n_tasks})...")
            
            # Adaptive task_chunk_size and B based on n_tasks to reduce memory
            # For very large k, use smaller task chunks and potentially smaller batch size
            adaptive_task_chunk_size = task_chunk_size
            adaptive_B = B
            
            if n_tasks >= 512:  # k >= 9
                # Very aggressive chunking for very large k
                adaptive_task_chunk_size = min(task_chunk_size or 32, 16)  # Max 16 tasks at a time
                adaptive_B = min(B, 32)  # Reduce batch size if needed
            elif n_tasks >= 256:  # k >= 8
                adaptive_task_chunk_size = min(task_chunk_size or 32, 24)  # Max 24 tasks at a time
                adaptive_B = min(B, 48)  # Slightly reduce batch size
            elif n_tasks >= 128:  # k >= 7
                adaptive_task_chunk_size = min(task_chunk_size or 32, 32)  # Use provided or 32
                adaptive_B = min(B, 64)  # Keep B or reduce slightly
            
            if verbose and (adaptive_task_chunk_size != task_chunk_size or adaptive_B != B):
                logger.info(f"  Adaptive settings: task_chunk_size={adaptive_task_chunk_size}, B={adaptive_B}")
            
            # Get experiment name for this k
            exp_name = get_exp_name(task_name, k, vocab_size=vocab_size)
            
            results = None
            stable_ranks_layer = None
            try:
                results = compute_stable_rank_at_padded_positions(
                    exp_name=exp_name,
                    task_name=task_name,
                    B=adaptive_B,
                    step=step,
                    n_minor=n_minor,
                    n_ood=n_ood,
                    verbose=verbose,
                    task_chunk_size=adaptive_task_chunk_size,
                )
                
                # Get stable ranks at specified layer: shape (n_positions,)
                # Move to CPU immediately to free GPU memory
                stable_ranks_layer = results['stable_ranks'][layer_idx, :].cpu()
                
                # Take maximum stable rank across all positions
                max_stable_rank = stable_ranks_layer.max().item()
                max_stable_ranks.append(max_stable_rank)
                
                if verbose:
                    logger.info(f"  k={k}, n_tasks={n_tasks}, max_stable_rank={max_stable_rank:.4f}")
            
            finally:
                # Clean up GPU memory after each iteration
                # Delete results to free memory
                if results is not None:
                    del results
                if 'stable_ranks_layer' in locals():
                    del stable_ranks_layer
                
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
        
        # Extra cleanup between chunks
        if verbose:
            logger.info(f"Completed chunk, clearing memory...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Plot results
    if backend == "plotly" and PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_values,
            y=max_stable_ranks,
            mode='lines+markers',
            name='Max Stable Rank',
            line=dict(width=2, color='blue'),
            marker=dict(size=8)
        ))
        layer_label = f"Layer {layer_idx}" if layer is not None else "Final Layer"
        fig.update_layout(
            title=f'Maximum Stable Rank vs k (log2 of number of tasks)<br>Task: {task_name}, {layer_label}',
            xaxis_title='k (log2 of number of tasks)',
            yaxis_title=f'Maximum Stable Rank at {layer_label}',
            hovermode='x unified',
            template='plotly_white',
            width=1000,
            height=600,
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved plot to {save_path}")
        
        if show:
            fig.show()
    
    elif backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(k_values, max_stable_ranks, 'o-', linewidth=2, markersize=8, 
               label='Max Stable Rank', color='blue')
        layer_label = f"Layer {layer_idx}" if layer is not None else "Final Layer"
        ax.set_xlabel('k (log2 of number of tasks)', fontsize=12)
        ax.set_ylabel(f'Maximum Stable Rank at {layer_label}', fontsize=12)
        ax.set_title(f'Maximum Stable Rank vs k\nTask: {task_name}, {layer_label}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        if show:
            plt.show()
    
    else:
        if backend == "plotly" and not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available, falling back to matplotlib")
            backend = "matplotlib"
        if backend == "matplotlib" and not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is not installed. Install it with: pip install matplotlib")
        
        # Recursive call with corrected backend
        return plot_max_stable_rank_vs_k(
            task_name, k_values, vocab_size, B, step, n_ood, backend, figsize, save_path, show, verbose, chunk_size, task_chunk_size, layer
        )
    
    return {
        'k_values': k_values,
        'max_stable_ranks': max_stable_ranks,
        'n_tasks': n_tasks_list,
        'layer': layer_idx,
        'fig': fig,
    }


def train_linear_softmax_posterior_predictor(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples: int = 1000,
    step: Optional[int] = None,
    n_minor: Optional[int] = None,
    n_ood: int = 0,
    learning_rate: float = 0.01,
    num_epochs: int = 100,
    verbose: bool = False,
    positions: Optional[list] = None,
    validation_split: float = 0.2,
    uniform_sampling: bool = True,
    skip_baselines: bool = False,
) -> dict:
    """
    Train a linear softmax model to predict task posteriors from hidden representations.
    
    For latent task:
    1. Gets samples from the sampler
    2. Computes task posteriors using task_posterior_over_time
    3. Extracts hidden representations at the specified layer
    4. Trains a linear softmax model to map hidden representations to posteriors
    5. Reports training loss
    
    Parameters:
    -----------
    exp_name : str
        Experiment name (e.g., "train_...")
    layer : int
        Layer index to extract hidden representations from
    B : int, default=64
        Batch size for sampling
    n_samples : int, default=1000
        Total number of samples to use for training
    step : int, optional
        Step for checkpoint loading. If None, uses final checkpoint.
    n_minor : int, optional
        Number of minor tasks to use. If None, uses all available minor tasks.
        If -1, uses no minor tasks (only OOD if n_ood > 0).
    n_ood : int, default=0
        Number of OOD tasks to include
    learning_rate : float, default=0.01
        Learning rate for training the linear model
    num_epochs : int, default=100
        Number of training epochs
    verbose : bool, default=False
        Whether to print progress messages
    positions : list, optional
        List of real token position indices to use (0-indexed). 
        If None, uses first 10 positions [0, 1, 2, ..., 9].
        These correspond to padded positions at indices [1, 3, 5, ...] in the padded sequence.
    validation_split : float, default=0.2
        Fraction of data to use for validation (between 0 and 1).
        The remaining fraction is used for training.
    uniform_sampling : bool, default=True
        If True, modifies p_minor to achieve uniform sampling across all tasks
        (each task has equal probability). If False, uses the original sampler's p_minor.
    skip_baselines : bool, default=False
        If True, skips training permutation and logits baselines to save time.
        Only trains the main model (hiddens -> posteriors).
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'final_loss': float, final training loss
        - 'final_val_loss': float, final validation loss
        - 'loss_history': list of training losses during training
        - 'val_loss_history': list of validation losses during training
        - 'model': the trained linear model
        - 'baseline_final_loss': float, final training loss for permutation baseline
        - 'baseline_final_val_loss': float, final validation loss for permutation baseline
        - 'baseline_loss_history': list of training losses for baseline (shuffled data)
        - 'baseline_val_loss_history': list of validation losses for baseline (shuffled data)
        - 'baseline_model': the trained baseline model (trained on shuffled data)
        - 'logits_baseline_final_loss': float, final training loss for logits baseline
        - 'logits_baseline_final_val_loss': float, final validation loss for logits baseline
        - 'logits_baseline_loss_history': list of training losses for logits baseline
        - 'logits_baseline_val_loss_history': list of validation losses for logits baseline
        - 'logits_baseline_model': the trained logits baseline model (trained on logits)
        - 'layer': int, the layer index used
        - 'n_tasks': int, number of tasks
    """
    from icl.latent_markov.markov_latent import task_posterior_over_time
    from torch import nn
    import torch.optim as optim
    
    # Load config/model/sampler
    _, sampler_orig, config = nu.load_everything("latent", exp_name)
    
    if step is None:
        step = config.training.num_epochs
    
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval()
    model.to(config.device)
    
    # Get sampler with specified n_minor and n_ood=0 (no OOD tasks)
    if n_minor is None:
        n_minor = 1000000  # Use all available
    elif n_minor == -1:
        n_minor = 0
    
    sampler, k_minor, _ = get_latent_sampler(exp_name, n_minor=n_minor, n_ood=0)  # Always use n_ood=0
    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    
    if not getattr(sampler, "pad", False):
        raise ValueError("This function requires padded latent sequences (sampler.pad must be True)")
    
    # Modify p_minor to achieve uniform sampling across all tasks
    # For uniform sampling: P(any task) = 1/(n_major + n_minor)
    # This requires: p_minor = n_minor / (n_major + n_minor)
    original_p_minor = sampler.p_minor
    if uniform_sampling and sampler.n_minor_tasks > 0:
        sampler.p_minor = sampler.n_minor_tasks / (sampler.n_major_tasks + sampler.n_minor_tasks)
        if verbose:
            logger.info(f"Modified p_minor from {original_p_minor:.6f} to {sampler.p_minor:.6f} for uniform task sampling")
    elif verbose and not uniform_sampling:
        logger.info(f"Using original p_minor: {original_p_minor:.6f} (not modified for uniform sampling)")
    
    if verbose:
        logger.info(f"Training linear softmax model to predict posteriors from layer {layer} hidden representations")
        logger.info(f"Number of tasks: {n_tasks} (major: {sampler.n_major_tasks}, minor: {sampler.n_minor_tasks}), Batch size: {B}, Total samples: {n_samples}")
    
    # Determine which positions to use (before the loop)
    seq_len = sampler.seq_len
    if positions is None:
        # Default: use first 10 positions
        positions = list(range(min(10, seq_len)))
    else:
        # Validate positions
        positions = list(positions)
        if not all(0 <= p < seq_len for p in positions):
            raise ValueError(f"All positions must be in [0, {seq_len-1}], got {positions}")
    
    if verbose:
        logger.info(f"Using positions: {positions} (real token indices)")
    
    # Map real token positions to padded positions: position t -> 2*t + 1
    device = config.device
    padded_positions = torch.tensor([2 * t + 1 for t in positions], device=device, dtype=torch.long)
    
    # Collect samples, hiddens, logits, and posteriors
    all_hiddens = []
    all_logits = []
    all_posteriors = []
    
    n_batches = (n_samples + B - 1) // B  # Ceiling division
    
    if verbose:
        logger.info(f"Collecting {n_batches} batches of data...")
    
    for batch_idx in range(n_batches):
        # Always use train mode: sample from mixture of major/minor tasks (no OOD)
        # Note: train mode returns (epochs, num_samples//epochs, -1), so we need to reshape
        samples, _ = sampler.generate(
            mode="train", task=None, num_samples=B, epochs=1
        )
        # Reshape from (1, B, L) to (B, L)
        if samples.dim() == 3:
            samples = samples.squeeze(0)  # (1, B, L) -> (B, L)
        samples = samples.to(device)
        
        # Compute task posteriors: shape (B, L_real, T)
        # where L_real is sequence length of real tokens
        posteriors = task_posterior_over_time(sampler, samples)  # (B, L_real, T)
        
        # Select only the positions we're interested in
        posteriors = posteriors[:, positions, :]  # (B, len(positions), T)
        
        # Extract hidden representations at specified layer
        # We need hiddens at padded positions that correspond to real token positions
        # For padded sequences: real tokens at even positions (0, 2, 4, ...)
        # Padded tokens at odd positions (1, 3, 5, ...)
        # We want hiddens at padded positions that follow real tokens
        
        # Extract hiddens at these padded positions for the whole batch at once
        cache = {}
        layer_module = model.layers[layer].attn_block
        
        def hook_fn(module, inp, out):
            if torch.is_tensor(out):
                # out: (B, L_obs, D)
                cached = out.index_select(dim=1, index=padded_positions).detach()  # (B, seq_len, D)
                cache["hidden"] = cached
            elif isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                cached = out[0].index_select(dim=1, index=padded_positions).detach()  # (B, seq_len, D)
                cache["hidden"] = cached
            else:
                raise RuntimeError(f"Unsupported hook output type: {type(out)}")
        
        handle = layer_module.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                # Get logits from model output
                logits_full = model(samples)  # (B, L_obs, vocab_size)
                # Extract logits at padded positions
                logits_batch = logits_full.index_select(dim=1, index=padded_positions)  # (B, len(positions), vocab_size)
            hiddens_batch = cache["hidden"]  # (B, len(positions), D)
        finally:
            handle.remove()
        
        # Move to CPU to save GPU memory
        all_hiddens.append(hiddens_batch.cpu())
        all_logits.append(logits_batch.cpu())
        all_posteriors.append(posteriors.cpu())
        
        # Clear GPU memory
        del samples, posteriors, hiddens_batch, logits_batch, logits_full
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Concatenate all batches
    hiddens_all = torch.cat(all_hiddens, dim=0)  # (n_samples, len(positions), D)
    logits_all = torch.cat(all_logits, dim=0)  # (n_samples, len(positions), vocab_size)
    posteriors_all = torch.cat(all_posteriors, dim=0)  # (n_samples, len(positions), T)
    
    # Reshape: flatten sequence dimension
    # hiddens: (n_samples * len(positions), D)
    # logits: (n_samples * len(positions), vocab_size)
    # posteriors: (n_samples * len(positions), T)
    n_total = hiddens_all.shape[0] * hiddens_all.shape[1]
    hiddens_flat = hiddens_all.reshape(n_total, -1)  # (n_total, D)
    logits_flat = logits_all.reshape(n_total, -1)  # (n_total, vocab_size)
    posteriors_flat = posteriors_all.reshape(n_total, -1)  # (n_total, T)
    
    D = hiddens_flat.shape[1]
    vocab_size = logits_flat.shape[1]
    T = posteriors_flat.shape[1]
    
    # Split into training and validation sets
    n_train = int(n_total * (1 - validation_split))
    indices = torch.randperm(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    hiddens_train = hiddens_flat[train_indices]
    logits_train = logits_flat[train_indices]
    posteriors_train = posteriors_flat[train_indices]
    hiddens_val = hiddens_flat[val_indices]
    logits_val = logits_flat[val_indices]
    posteriors_val = posteriors_flat[val_indices]
    
    if verbose:
        logger.info(f"Total data: {n_total} samples")
        logger.info(f"Training data: {n_train} samples ({100*(1-validation_split):.1f}%)")
        logger.info(f"Validation data: {len(val_indices)} samples ({100*validation_split:.1f}%)")
        logger.info(f"Training data shape: hiddens {hiddens_train.shape}, logits {logits_train.shape}, posteriors {posteriors_train.shape}")
    
    # Create linear softmax model: hidden (D) -> logits (T) -> softmax -> posterior (T)
    linear_model = nn.Sequential(
        nn.Linear(D, T, bias=True),
        nn.Softmax(dim=-1)
    ).to(device)
    
    # Move data to device
    hiddens_train = hiddens_train.to(device)
    logits_train = logits_train.to(device)
    posteriors_train = posteriors_train.to(device)
    hiddens_val = hiddens_val.to(device)
    logits_val = logits_val.to(device)
    posteriors_val = posteriors_val.to(device)
    
    # Training setup
    optimizer = optim.Adam(linear_model.parameters(), lr=learning_rate)
    criterion = nn.KLDivLoss(reduction='batchmean')  # KL divergence for probability distributions
    
    loss_history = []
    val_loss_history = []
    
    if verbose:
        logger.info(f"Training linear model for {num_epochs} epochs...")
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        linear_model.train()
        optimizer.zero_grad()
        
        # Forward pass on training data
        pred_posteriors_train = linear_model(hiddens_train)  # (n_train, T)
        
        # Compute training loss: KL divergence between predicted and true posteriors
        # KLDivLoss expects log probabilities, so we need log of predictions
        log_pred_train = torch.log(pred_posteriors_train + 1e-10)  # Add small epsilon for numerical stability
        train_loss = criterion(log_pred_train, posteriors_train)
        
        # Backward pass
        train_loss.backward()
        optimizer.step()
        
        # Validation phase
        linear_model.eval()
        with torch.no_grad():
            pred_posteriors_val = linear_model(hiddens_val)  # (n_val, T)
            log_pred_val = torch.log(pred_posteriors_val + 1e-10)
            val_loss = criterion(log_pred_val, posteriors_val)
        
        loss_history.append(train_loss.item())
        val_loss_history.append(val_loss.item())
        
        if verbose and (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
    
    final_loss = loss_history[-1]
    final_val_loss = val_loss_history[-1]
    
    if verbose:
        logger.info(f"Training completed. Final train loss: {final_loss:.6f}, Final val loss: {final_val_loss:.6f}")
    
    # Initialize baseline results with None/NaN if skipping
    final_baseline_loss = float('nan')
    final_baseline_val_loss = float('nan')
    baseline_loss_history = []
    baseline_val_loss_history = []
    baseline_model = None
    
    final_logits_baseline_loss = float('nan')
    final_logits_baseline_val_loss = float('nan')
    logits_baseline_loss_history = []
    logits_baseline_val_loss_history = []
    logits_baseline_model = None
    
    if not skip_baselines:
        # Permutation baseline: shuffle posteriors to break the pairing with hiddens
        if verbose:
            logger.info("Training permutation baseline (shuffled posteriors)...")
        
        # Create shuffled posteriors (shuffle independently for train and val)
        posteriors_train_shuffled = posteriors_train[torch.randperm(n_train)]
        posteriors_val_shuffled = posteriors_val[torch.randperm(len(val_indices))]
        
        # Create a new model for the baseline
        baseline_model = nn.Sequential(
            nn.Linear(D, T, bias=True),
            nn.Softmax(dim=-1)
        ).to(device)
        
        # Training setup for baseline
        baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=learning_rate)
        
        baseline_loss_history = []
        baseline_val_loss_history = []
        
        # Training loop for baseline
        for epoch in range(num_epochs):
            # Training phase
            baseline_model.train()
            baseline_optimizer.zero_grad()
            
            # Forward pass on training data with shuffled posteriors
            pred_posteriors_train_baseline = baseline_model(hiddens_train)  # (n_train, T)
            log_pred_train_baseline = torch.log(pred_posteriors_train_baseline + 1e-10)
            train_loss_baseline = criterion(log_pred_train_baseline, posteriors_train_shuffled)
            
            # Backward pass
            train_loss_baseline.backward()
            baseline_optimizer.step()
            
            # Validation phase
            baseline_model.eval()
            with torch.no_grad():
                pred_posteriors_val_baseline = baseline_model(hiddens_val)  # (n_val, T)
                log_pred_val_baseline = torch.log(pred_posteriors_val_baseline + 1e-10)
                val_loss_baseline = criterion(log_pred_val_baseline, posteriors_val_shuffled)
            
            baseline_loss_history.append(train_loss_baseline.item())
            baseline_val_loss_history.append(val_loss_baseline.item())
            
            if verbose and (epoch + 1) % 20 == 0:
                logger.info(f"  Baseline Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_baseline.item():.6f}, Val Loss: {val_loss_baseline.item():.6f}")
        
        final_baseline_loss = baseline_loss_history[-1]
        final_baseline_val_loss = baseline_val_loss_history[-1]
        
        if verbose:
            logger.info(f"Baseline completed. Final train loss: {final_baseline_loss:.6f}, Final val loss: {final_baseline_val_loss:.6f}")
            logger.info(f"Improvement over baseline - Train: {final_loss - final_baseline_loss:.6f}, Val: {final_val_loss - final_baseline_val_loss:.6f}")
        
        # Logits baseline: train a model to predict posteriors from logits
        if verbose:
            logger.info("Training logits baseline (predicting posteriors from logits)...")
        
        # Create a new model for logits baseline
        logits_baseline_model = nn.Sequential(
            nn.Linear(vocab_size, T, bias=True),
            nn.Softmax(dim=-1)
        ).to(device)
        
        # Training setup for logits baseline
        logits_baseline_optimizer = optim.Adam(logits_baseline_model.parameters(), lr=learning_rate)
        
        logits_baseline_loss_history = []
        logits_baseline_val_loss_history = []
        
        # Training loop for logits baseline
        for epoch in range(num_epochs):
            # Training phase
            logits_baseline_model.train()
            logits_baseline_optimizer.zero_grad()
            
            # Forward pass on training data using logits
            pred_posteriors_train_logits = logits_baseline_model(logits_train)  # (n_train, T)
            log_pred_train_logits = torch.log(pred_posteriors_train_logits + 1e-10)
            train_loss_logits = criterion(log_pred_train_logits, posteriors_train)
            
            # Backward pass
            train_loss_logits.backward()
            logits_baseline_optimizer.step()
            
            # Validation phase
            logits_baseline_model.eval()
            with torch.no_grad():
                pred_posteriors_val_logits = logits_baseline_model(logits_val)  # (n_val, T)
                log_pred_val_logits = torch.log(pred_posteriors_val_logits + 1e-10)
                val_loss_logits = criterion(log_pred_val_logits, posteriors_val)
            
            logits_baseline_loss_history.append(train_loss_logits.item())
            logits_baseline_val_loss_history.append(val_loss_logits.item())
            
            if verbose and (epoch + 1) % 20 == 0:
                logger.info(f"  Logits Baseline Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_logits.item():.6f}, Val Loss: {val_loss_logits.item():.6f}")
        
        final_logits_baseline_loss = logits_baseline_loss_history[-1]
        final_logits_baseline_val_loss = logits_baseline_val_loss_history[-1]
        
        if verbose:
            logger.info(f"Logits baseline completed. Final train loss: {final_logits_baseline_loss:.6f}, Final val loss: {final_logits_baseline_val_loss:.6f}")
            logger.info(f"Comparison - Hiddens vs Logits - Train: {final_loss - final_logits_baseline_loss:.6f}, Val: {final_val_loss - final_logits_baseline_val_loss:.6f}")
    
    # Restore original p_minor
    if hasattr(sampler, 'p_minor') and 'original_p_minor' in locals():
        sampler.p_minor = original_p_minor
    
    # Move models back to CPU
    linear_model = linear_model.cpu()
    if baseline_model is not None:
        baseline_model = baseline_model.cpu()
    if logits_baseline_model is not None:
        logits_baseline_model = logits_baseline_model.cpu()
    
    return {
        'final_loss': final_loss,
        'final_val_loss': final_val_loss,
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'model': linear_model,
        'baseline_final_loss': final_baseline_loss,
        'baseline_final_val_loss': final_baseline_val_loss,
        'baseline_loss_history': baseline_loss_history,
        'baseline_val_loss_history': baseline_val_loss_history,
        'baseline_model': baseline_model,
        'logits_baseline_final_loss': final_logits_baseline_loss,
        'logits_baseline_final_val_loss': final_logits_baseline_val_loss,
        'logits_baseline_loss_history': logits_baseline_loss_history,
        'logits_baseline_val_loss_history': logits_baseline_val_loss_history,
        'logits_baseline_model': logits_baseline_model,
        'layer': layer,
        'n_tasks': n_tasks,
        'hidden_dim': D,
        'vocab_size': vocab_size,
        'n_samples': n_total,
        'n_train': n_train,
        'n_val': len(val_indices),
    }


def train_linear_hidden_predictor(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples: int = 1000,
    step: Optional[int] = None,
    n_minor: Optional[int] = None,
    n_ood: int = 0,
    verbose: bool = False,
    positions: Optional[list] = None,
    validation_split: float = 0.2,
    uniform_sampling: bool = True,
    skip_baselines: bool = False,
) -> dict:
    """
    Train a linear regression model to predict hidden representations from task posteriors.
    
    This is the reverse of train_linear_softmax_posterior_predictor:
    instead of predicting posteriors from hiddens, we predict hiddens from posteriors.
    
    The task posteriors are first transformed to logit vectors (log of posteriors),
    then linear regression is used to fit: posterior_logits -> hiddens.
    
    For latent task:
    1. Gets samples from the sampler
    2. Computes task posteriors using task_posterior_over_time
    3. Extracts hidden representations at the specified layer
    4. Converts posteriors to logit vectors: log(posterior + eps)
    5. Fits a linear regression model: posterior_logits -> hiddens
    6. Reports MSE and R^2 for training and validation data
    
    Parameters:
    -----------
    exp_name : str
        Experiment name (e.g., "train_...")
    layer : int
        Layer index to extract hidden representations from
    B : int, default=64
        Batch size for sampling
    n_samples : int, default=1000
        Total number of samples to use for training
    step : int, optional
        Step for checkpoint loading. If None, uses final checkpoint.
    n_minor : int, optional
        Number of minor tasks to use. If None, uses all available minor tasks.
        If -1, uses no minor tasks (only OOD if n_ood > 0).
    n_ood : int, default=0
        Number of OOD tasks to include
    verbose : bool, default=False
        Whether to print progress messages
    positions : list, optional
        List of real token position indices to use (0-indexed). 
        If None, uses first 10 positions [0, 1, 2, ..., 9].
        These correspond to padded positions at indices [1, 3, 5, ...] in the padded sequence.
    validation_split : float, default=0.2
        Fraction of data to use for validation (between 0 and 1).
        The remaining fraction is used for training.
    uniform_sampling : bool, default=True
        If True, modifies p_minor to achieve uniform sampling across all tasks.
        If False, uses the original sampler's p_minor.
    skip_baselines : bool, default=False
        If True, skips training permutation and logits baselines to save time.
        Only trains the main model (posterior_logits -> hiddens).
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'train_mse': float, training MSE
        - 'val_mse': float, validation MSE
        - 'train_r2': float, training R^2
        - 'val_r2': float, validation R^2
        - 'model_weight': Tensor, weight matrix W of shape (T, D)
        - 'model_bias': Tensor, bias vector b of shape (D,)
        - 'baseline_train_mse': float, permutation baseline training MSE
        - 'baseline_val_mse': float, permutation baseline validation MSE
        - 'baseline_train_r2': float, permutation baseline training R^2
        - 'baseline_val_r2': float, permutation baseline validation R^2
        - 'logits_baseline_train_mse': float, logits baseline training MSE
        - 'logits_baseline_val_mse': float, logits baseline validation MSE
        - 'logits_baseline_train_r2': float, logits baseline training R^2
        - 'logits_baseline_val_r2': float, logits baseline validation R^2
        - 'onehot_baseline_train_mse': float, one-hot baseline training MSE
        - 'onehot_baseline_val_mse': float, one-hot baseline validation MSE
        - 'onehot_baseline_train_r2': float, one-hot baseline training R^2
        - 'onehot_baseline_val_r2': float, one-hot baseline validation R^2
        - 'combined_train_mse': float, combined model training MSE
        - 'combined_val_mse': float, combined model validation MSE
        - 'combined_train_r2': float, combined model training R^2
        - 'combined_val_r2': float, combined model validation R^2
        - 'orthogonality': dict or None, orthogonality analysis between predictions
        - 'layer': int, the layer index used
        - 'n_tasks': int, number of tasks
        - 'hidden_dim': int, hidden dimension D
        - 'vocab_size': int, vocabulary size
        - 'n_samples': int, total number of data points
        - 'n_train': int, number of training data points
        - 'n_val': int, number of validation data points
    """
    from icl.latent_markov.markov_latent import task_posterior_over_time
    
    # Load config/model/sampler
    _, sampler_orig, config = nu.load_everything("latent", exp_name)
    
    if step is None:
        step = config.training.num_epochs
    
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval()
    model.to(config.device)
    
    # Get sampler with specified n_minor and n_ood=0 (no OOD tasks)
    if n_minor is None:
        n_minor = 1000000  # Use all available
    elif n_minor == -1:
        n_minor = 0
    
    sampler, k_minor, _ = get_latent_sampler(exp_name, n_minor=n_minor, n_ood=0)
    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    
    if not getattr(sampler, "pad", False):
        raise ValueError("This function requires padded latent sequences (sampler.pad must be True)")
    
    # Modify p_minor to achieve uniform sampling across all tasks
    original_p_minor = sampler.p_minor
    if uniform_sampling and sampler.n_minor_tasks > 0:
        sampler.p_minor = sampler.n_minor_tasks / (sampler.n_major_tasks + sampler.n_minor_tasks)
        if verbose:
            logger.info(f"Modified p_minor from {original_p_minor:.6f} to {sampler.p_minor:.6f} for uniform task sampling")
    elif verbose and not uniform_sampling:
        logger.info(f"Using original p_minor: {original_p_minor:.6f} (not modified for uniform sampling)")
    
    if verbose:
        logger.info(f"Training linear regression: posterior_logits -> hidden representations (layer {layer})")
        logger.info(f"Number of tasks: {n_tasks} (major: {sampler.n_major_tasks}, minor: {sampler.n_minor_tasks}), Batch size: {B}, Total samples: {n_samples}")
    
    # Determine which positions to use (before the loop)
    seq_len = sampler.seq_len
    if positions is None:
        positions = list(range(min(10, seq_len)))
    else:
        positions = list(positions)
        if not all(0 <= p < seq_len for p in positions):
            raise ValueError(f"All positions must be in [0, {seq_len-1}], got {positions}")
    
    if verbose:
        logger.info(f"Using positions: {positions} (real token indices)")
    
    # Map real token positions to padded positions: position t -> 2*t + 1
    device = config.device
    padded_positions = torch.tensor([2 * t + 1 for t in positions], device=device, dtype=torch.long)
    
    # Real token positions in padded sequence: position t -> 2*t (the token just before the pad)
    real_token_padded_positions = torch.tensor([2 * t for t in positions], device=device, dtype=torch.long)
    
    # Collect samples, hiddens, logits, posteriors, and real tokens
    all_hiddens = []
    all_logits = []
    all_posteriors = []
    all_real_tokens = []
    
    n_batches = (n_samples + B - 1) // B
    
    if verbose:
        logger.info(f"Collecting {n_batches} batches of data...")
    
    for batch_idx in range(n_batches):
        samples, _ = sampler.generate(
            mode="train", task=None, num_samples=B, epochs=1
        )
        if samples.dim() == 3:
            samples = samples.squeeze(0)
        samples = samples.to(device)
        
        posteriors = task_posterior_over_time(sampler, samples)  # (B, L_real, T)
        posteriors = posteriors[:, positions, :]  # (B, len(positions), T)
        
        # Extract the real tokens at the padded positions just before the pad tokens
        real_tokens_batch = samples.index_select(dim=1, index=real_token_padded_positions)  # (B, len(positions))
        
        cache = {}
        layer_module = model.layers[layer].attn_block
        
        def hook_fn(module, inp, out):
            if torch.is_tensor(out):
                cached = out.index_select(dim=1, index=padded_positions).detach()
                cache["hidden"] = cached
            elif isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                cached = out[0].index_select(dim=1, index=padded_positions).detach()
                cache["hidden"] = cached
            else:
                raise RuntimeError(f"Unsupported hook output type: {type(out)}")
        
        handle = layer_module.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                logits_full = model(samples)
                logits_batch = logits_full.index_select(dim=1, index=padded_positions)
            hiddens_batch = cache["hidden"]
        finally:
            handle.remove()
        
        all_hiddens.append(hiddens_batch.cpu())
        all_logits.append(logits_batch.cpu())
        all_posteriors.append(posteriors.cpu())
        all_real_tokens.append(real_tokens_batch.cpu())
        
        del samples, posteriors, hiddens_batch, logits_batch, logits_full, real_tokens_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Concatenate all batches
    hiddens_all = torch.cat(all_hiddens, dim=0)   # (n_samples, len(positions), D)
    logits_all = torch.cat(all_logits, dim=0)      # (n_samples, len(positions), vocab_size)
    posteriors_all = torch.cat(all_posteriors, dim=0)  # (n_samples, len(positions), T)
    real_tokens_all = torch.cat(all_real_tokens, dim=0)  # (n_samples, len(positions))
    
    # Reshape: flatten sequence dimension
    n_total = hiddens_all.shape[0] * hiddens_all.shape[1]
    hiddens_flat = hiddens_all.reshape(n_total, -1)      # (n_total, D)
    logits_flat = logits_all.reshape(n_total, -1)         # (n_total, vocab_size)
    posteriors_flat = posteriors_all.reshape(n_total, -1)  # (n_total, T)
    real_tokens_flat = real_tokens_all.reshape(n_total)    # (n_total,)
    
    D = hiddens_flat.shape[1]
    vocab_size = logits_flat.shape[1]
    T = posteriors_flat.shape[1]
    
    # Convert real tokens to one-hot vectors
    # Determine the number of classes from the vocab size of the sampler
    n_vocab = int(real_tokens_flat.max().item()) + 1
    if n_vocab < vocab_size:
        n_vocab = vocab_size  # Use model vocab_size to be safe
    onehot_flat = torch.zeros(n_total, n_vocab, dtype=hiddens_flat.dtype)
    onehot_flat.scatter_(1, real_tokens_flat.unsqueeze(1).long(), 1.0)  # (n_total, n_vocab)
    
    # Convert posteriors to logit vectors
    eps = 1e-10
    posterior_logits = torch.log(posteriors_flat + eps)  # (n_total, T)
    
    # Split into training and validation sets
    n_train = int(n_total * (1 - validation_split))
    indices = torch.randperm(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    posterior_logits_train = posterior_logits[train_indices]
    posterior_logits_val = posterior_logits[val_indices]
    hiddens_train = hiddens_flat[train_indices]
    hiddens_val = hiddens_flat[val_indices]
    logits_train = logits_flat[train_indices]
    logits_val = logits_flat[val_indices]
    onehot_train = onehot_flat[train_indices]
    onehot_val = onehot_flat[val_indices]
    
    if verbose:
        logger.info(f"Total data: {n_total} samples")
        logger.info(f"Training data: {n_train} samples ({100*(1-validation_split):.1f}%)")
        logger.info(f"Validation data: {len(val_indices)} samples ({100*validation_split:.1f}%)")
        logger.info(f"Input shape (posterior logits): {posterior_logits_train.shape}, Target shape (hiddens): {hiddens_train.shape}")
        logger.info(f"One-hot baseline shape: {onehot_train.shape} (vocab: {n_vocab})")
    
    # --- Helper: closed-form linear regression with bias ---
    def _fit_linear_regression(X_train, Y_train, X_val, Y_val):
        """
        Fit linear regression Y = X @ W + b using pseudoinverse (SVD-based).
        More numerically stable than lstsq, especially for combined feature matrices
        that may be poorly conditioned.
        Returns (W, b, train_mse, val_mse, train_r2, val_r2).
        """
        # Cast to float32 for numerical stability
        X_tr = X_train.float()
        Y_tr = Y_train.float()
        X_v = X_val.float()
        Y_v = Y_val.float()
        
        # Append column of ones for bias: X_aug = [X, 1]
        ones_train = torch.ones(X_tr.shape[0], 1, dtype=torch.float32, device=X_tr.device)
        X_aug_train = torch.cat([X_tr, ones_train], dim=1)  # (n_train, input_dim+1)
        
        # Solve using pseudoinverse (SVD-based, robust to rank deficiency)
        # W_aug = pinv(X_aug) @ Y
        X_pinv = torch.linalg.pinv(X_aug_train)  # (input_dim+1, n_train)
        W_aug = X_pinv @ Y_tr  # (input_dim+1, D)
        
        W = W_aug[:-1, :]  # (input_dim, D)
        b = W_aug[-1, :]   # (D,)
        
        # Training predictions and metrics
        Y_pred_train = X_tr @ W + b
        train_residuals = Y_tr - Y_pred_train
        train_mse = (train_residuals ** 2).mean().item()
        train_ss_tot = ((Y_tr - Y_tr.mean(dim=0)) ** 2).sum().item()
        train_ss_res = (train_residuals ** 2).sum().item()
        train_r2 = 1.0 - train_ss_res / train_ss_tot if train_ss_tot > 0 else float('nan')
        
        # Validation predictions and metrics
        Y_pred_val = X_v @ W + b
        val_residuals = Y_v - Y_pred_val
        val_mse = (val_residuals ** 2).mean().item()
        val_ss_tot = ((Y_v - Y_v.mean(dim=0)) ** 2).sum().item()
        val_ss_res = (val_residuals ** 2).sum().item()
        val_r2 = 1.0 - val_ss_res / val_ss_tot if val_ss_tot > 0 else float('nan')
        
        return W, b, train_mse, val_mse, train_r2, val_r2
    
    # --- Main model: posterior_logits -> hiddens ---
    if verbose:
        logger.info("Fitting linear regression: posterior_logits -> hiddens...")
    
    W, b, train_mse, val_mse, train_r2, val_r2 = _fit_linear_regression(
        posterior_logits_train, hiddens_train, posterior_logits_val, hiddens_val
    )
    
    if verbose:
        logger.info(f"Main model - Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}")
        logger.info(f"Main model - Train R²: {train_r2:.6f}, Val R²: {val_r2:.6f}")
    
    # Initialize baseline results
    baseline_train_mse = float('nan')
    baseline_val_mse = float('nan')
    baseline_train_r2 = float('nan')
    baseline_val_r2 = float('nan')
    baseline_W = None
    baseline_b = None
    
    logits_baseline_train_mse = float('nan')
    logits_baseline_val_mse = float('nan')
    logits_baseline_train_r2 = float('nan')
    logits_baseline_val_r2 = float('nan')
    logits_baseline_W = None
    logits_baseline_b = None
    
    onehot_baseline_train_mse = float('nan')
    onehot_baseline_val_mse = float('nan')
    onehot_baseline_train_r2 = float('nan')
    onehot_baseline_val_r2 = float('nan')
    onehot_baseline_W = None
    onehot_baseline_b = None
    
    combined_train_mse = float('nan')
    combined_val_mse = float('nan')
    combined_train_r2 = float('nan')
    combined_val_r2 = float('nan')
    combined_W = None
    combined_b = None
    
    ortho = None
    
    if not skip_baselines:
        # --- Permutation baseline: shuffle hiddens to break pairing ---
        if verbose:
            logger.info("Fitting permutation baseline (shuffled hiddens)...")
        
        hiddens_train_shuffled = hiddens_train[torch.randperm(n_train)]
        hiddens_val_shuffled = hiddens_val[torch.randperm(len(val_indices))]
        
        baseline_W, baseline_b, baseline_train_mse, baseline_val_mse, baseline_train_r2, baseline_val_r2 = (
            _fit_linear_regression(
                posterior_logits_train, hiddens_train_shuffled,
                posterior_logits_val, hiddens_val_shuffled
            )
        )
        
        if verbose:
            logger.info(f"Permutation baseline - Train MSE: {baseline_train_mse:.6f}, Val MSE: {baseline_val_mse:.6f}")
            logger.info(f"Permutation baseline - Train R²: {baseline_train_r2:.6f}, Val R²: {baseline_val_r2:.6f}")
        
        # --- Logits baseline: model_logits -> hiddens ---
        if verbose:
            logger.info("Fitting logits baseline (model logits -> hiddens)...")
        
        logits_baseline_W, logits_baseline_b, logits_baseline_train_mse, logits_baseline_val_mse, logits_baseline_train_r2, logits_baseline_val_r2 = (
            _fit_linear_regression(
                logits_train, hiddens_train,
                logits_val, hiddens_val
            )
        )
        
        if verbose:
            logger.info(f"Logits baseline - Train MSE: {logits_baseline_train_mse:.6f}, Val MSE: {logits_baseline_val_mse:.6f}")
            logger.info(f"Logits baseline - Train R²: {logits_baseline_train_r2:.6f}, Val R²: {logits_baseline_val_r2:.6f}")
        
        # --- One-hot baseline: one-hot of prior real token -> hiddens ---
        if verbose:
            logger.info("Fitting one-hot baseline (one-hot of prior real token -> hiddens)...")
        
        onehot_baseline_W, onehot_baseline_b, onehot_baseline_train_mse, onehot_baseline_val_mse, onehot_baseline_train_r2, onehot_baseline_val_r2 = (
            _fit_linear_regression(
                onehot_train, hiddens_train,
                onehot_val, hiddens_val
            )
        )
        
        if verbose:
            logger.info(f"One-hot baseline - Train MSE: {onehot_baseline_train_mse:.6f}, Val MSE: {onehot_baseline_val_mse:.6f}")
            logger.info(f"One-hot baseline - Train R²: {onehot_baseline_train_r2:.6f}, Val R²: {onehot_baseline_val_r2:.6f}")
        
        # --- Combined model: [posterior_logits, one-hot, model_logits] -> hiddens ---
        if verbose:
            logger.info("Fitting combined model (posterior_logits + one-hot + model logits -> hiddens)...")
        
        combined_train = torch.cat([posterior_logits_train, onehot_train, logits_train], dim=1)  # (n_train, T + n_vocab + vocab_size)
        combined_val = torch.cat([posterior_logits_val, onehot_val, logits_val], dim=1)
        
        combined_W, combined_b, combined_train_mse, combined_val_mse, combined_train_r2, combined_val_r2 = (
            _fit_linear_regression(
                combined_train, hiddens_train,
                combined_val, hiddens_val
            )
        )
        
        if verbose:
            logger.info(f"Combined model - Train MSE: {combined_train_mse:.6f}, Val MSE: {combined_val_mse:.6f}")
            logger.info(f"Combined model - Train R²: {combined_train_r2:.6f}, Val R²: {combined_val_r2:.6f}")
            logger.info(f"Comparison (Val R²) - Main: {val_r2:.6f}, Permutation: {baseline_val_r2:.6f}, Logits: {logits_baseline_val_r2:.6f}, One-hot: {onehot_baseline_val_r2:.6f}, Combined: {combined_val_r2:.6f}")
        
        # --- Orthogonality analysis between fitted predictions ---
        # Compute centered predictions on the validation set for main, logits, and one-hot models
        pred_main_val = posterior_logits_val @ W + b  # (n_val, D)
        pred_logits_val = logits_val @ logits_baseline_W + logits_baseline_b  # (n_val, D)
        pred_onehot_val = onehot_val @ onehot_baseline_W + onehot_baseline_b  # (n_val, D)
        
        # Center predictions (subtract mean across samples)
        pred_main_centered = pred_main_val - pred_main_val.mean(dim=0, keepdim=True)
        pred_logits_centered = pred_logits_val - pred_logits_val.mean(dim=0, keepdim=True)
        pred_onehot_centered = pred_onehot_val - pred_onehot_val.mean(dim=0, keepdim=True)
        
        # Flatten each prediction matrix to a single vector for overall cosine similarity
        # This measures global alignment of the predicted variation patterns
        def _cosine_sim_flat(A, B):
            """Cosine similarity between two flattened matrices."""
            a = A.reshape(-1).float()
            b = B.reshape(-1).float()
            return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-10)).item()
        
        # Per-dimension cosine similarity (average over D dimensions)
        # For each hidden dimension d, compute cosine similarity between the n_val prediction vectors
        def _cosine_sim_per_dim(A, B):
            """Mean cosine similarity across dimensions (columns)."""
            # A, B: (n_val, D)
            A_f = A.float()
            B_f = B.float()
            dot = (A_f * B_f).sum(dim=0)  # (D,)
            norm_a = A_f.norm(dim=0)  # (D,)
            norm_b = B_f.norm(dim=0)  # (D,)
            cos_per_d = dot / (norm_a * norm_b + 1e-10)  # (D,)
            return cos_per_d.mean().item(), cos_per_d
        
        # Variance explained overlap: fraction of val variance explained by both models
        # R² of regressing one model's predictions on another
        def _cross_r2(pred_A_centered, pred_B_centered):
            """R² from regressing pred_A on pred_B (how much of A's prediction is captured by B)."""
            # Solve pred_A = pred_B @ W_cross + bias using lstsq
            ones = torch.ones(pred_B_centered.shape[0], 1, dtype=pred_B_centered.dtype, device=pred_B_centered.device)
            B_aug = torch.cat([pred_B_centered, ones], dim=1)
            result = torch.linalg.lstsq(B_aug, pred_A_centered)
            W_cross = result.solution
            pred_A_from_B = B_aug @ W_cross
            ss_res = ((pred_A_centered - pred_A_from_B) ** 2).sum().item()
            ss_tot = (pred_A_centered ** 2).sum().item()  # Already centered
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        
        ortho = {}
        
        # Pairwise cosine similarities (flattened)
        ortho['cos_main_logits'] = _cosine_sim_flat(pred_main_centered, pred_logits_centered)
        ortho['cos_main_onehot'] = _cosine_sim_flat(pred_main_centered, pred_onehot_centered)
        ortho['cos_logits_onehot'] = _cosine_sim_flat(pred_logits_centered, pred_onehot_centered)
        
        # Pairwise mean per-dimension cosine similarities
        ortho['cos_per_dim_main_logits'], _ = _cosine_sim_per_dim(pred_main_centered, pred_logits_centered)
        ortho['cos_per_dim_main_onehot'], _ = _cosine_sim_per_dim(pred_main_centered, pred_onehot_centered)
        ortho['cos_per_dim_logits_onehot'], _ = _cosine_sim_per_dim(pred_logits_centered, pred_onehot_centered)
        
        # Cross R² (how much of one model's predictions can be captured by the other)
        ortho['cross_r2_main_from_logits'] = _cross_r2(pred_main_centered, pred_logits_centered)
        ortho['cross_r2_main_from_onehot'] = _cross_r2(pred_main_centered, pred_onehot_centered)
        ortho['cross_r2_logits_from_main'] = _cross_r2(pred_logits_centered, pred_main_centered)
        ortho['cross_r2_logits_from_onehot'] = _cross_r2(pred_logits_centered, pred_onehot_centered)
        ortho['cross_r2_onehot_from_main'] = _cross_r2(pred_onehot_centered, pred_main_centered)
        ortho['cross_r2_onehot_from_logits'] = _cross_r2(pred_onehot_centered, pred_logits_centered)
        
        if verbose:
            logger.info("--- Orthogonality analysis (val set, centered predictions) ---")
            logger.info(f"  Cosine sim (flat) - Main vs Logits: {ortho['cos_main_logits']:.6f}")
            logger.info(f"  Cosine sim (flat) - Main vs One-hot: {ortho['cos_main_onehot']:.6f}")
            logger.info(f"  Cosine sim (flat) - Logits vs One-hot: {ortho['cos_logits_onehot']:.6f}")
            logger.info(f"  Mean per-dim cosine - Main vs Logits: {ortho['cos_per_dim_main_logits']:.6f}")
            logger.info(f"  Mean per-dim cosine - Main vs One-hot: {ortho['cos_per_dim_main_onehot']:.6f}")
            logger.info(f"  Mean per-dim cosine - Logits vs One-hot: {ortho['cos_per_dim_logits_onehot']:.6f}")
            logger.info(f"  Cross R² - Main from Logits: {ortho['cross_r2_main_from_logits']:.6f}, Main from One-hot: {ortho['cross_r2_main_from_onehot']:.6f}")
            logger.info(f"  Cross R² - Logits from Main: {ortho['cross_r2_logits_from_main']:.6f}, Logits from One-hot: {ortho['cross_r2_logits_from_onehot']:.6f}")
            logger.info(f"  Cross R² - One-hot from Main: {ortho['cross_r2_onehot_from_main']:.6f}, One-hot from Logits: {ortho['cross_r2_onehot_from_logits']:.6f}")
    
    # Restore original p_minor
    if hasattr(sampler, 'p_minor') and 'original_p_minor' in locals():
        sampler.p_minor = original_p_minor
    
    return {
        'train_mse': train_mse,
        'val_mse': val_mse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'model_weight': W.cpu(),
        'model_bias': b.cpu(),
        'baseline_train_mse': baseline_train_mse,
        'baseline_val_mse': baseline_val_mse,
        'baseline_train_r2': baseline_train_r2,
        'baseline_val_r2': baseline_val_r2,
        'baseline_weight': baseline_W.cpu() if baseline_W is not None else None,
        'baseline_bias': baseline_b.cpu() if baseline_b is not None else None,
        'logits_baseline_train_mse': logits_baseline_train_mse,
        'logits_baseline_val_mse': logits_baseline_val_mse,
        'logits_baseline_train_r2': logits_baseline_train_r2,
        'logits_baseline_val_r2': logits_baseline_val_r2,
        'logits_baseline_weight': logits_baseline_W.cpu() if logits_baseline_W is not None else None,
        'logits_baseline_bias': logits_baseline_b.cpu() if logits_baseline_b is not None else None,
        'onehot_baseline_train_mse': onehot_baseline_train_mse,
        'onehot_baseline_val_mse': onehot_baseline_val_mse,
        'onehot_baseline_train_r2': onehot_baseline_train_r2,
        'onehot_baseline_val_r2': onehot_baseline_val_r2,
        'onehot_baseline_weight': onehot_baseline_W.cpu() if onehot_baseline_W is not None else None,
        'onehot_baseline_bias': onehot_baseline_b.cpu() if onehot_baseline_b is not None else None,
        'combined_train_mse': combined_train_mse,
        'combined_val_mse': combined_val_mse,
        'combined_train_r2': combined_train_r2,
        'combined_val_r2': combined_val_r2,
        'combined_weight': combined_W.cpu() if combined_W is not None else None,
        'combined_bias': combined_b.cpu() if combined_b is not None else None,
        'orthogonality': ortho,
        'layer': layer,
        'n_tasks': n_tasks,
        'hidden_dim': D,
        'vocab_size': vocab_size,
        'n_samples': n_total,
        'n_train': n_train,
        'n_val': len(val_indices),
    }


def train_mlp_hidden_predictor(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples: int = 1000,
    step: Optional[int] = None,
    n_minor: Optional[int] = None,
    n_ood: int = 0,
    hidden_size: int = 64,
    learning_rate: float = 0.001,
    num_epochs: int = 200,
    verbose: bool = False,
    positions: Optional[list] = None,
    validation_split: float = 0.2,
    uniform_sampling: bool = True,
    skip_baselines: bool = False,
) -> dict:
    """
    Train a one-hidden-layer MLP to predict hidden representations from task posteriors.
    
    Similar to train_linear_hidden_predictor but uses an MLP instead of linear regression.
    Architecture: posterior_logits -> Linear(T, hidden_size) -> ReLU -> Linear(hidden_size, D) -> hiddens
    
    This is trained with gradient descent (Adam + MSE loss), so it produces epoch-wise
    metric histories that can be plotted.
    
    For latent task:
    1. Gets samples from the sampler
    2. Computes task posteriors using task_posterior_over_time
    3. Extracts hidden representations at the specified layer
    4. Converts posteriors to logit vectors: log(posterior + eps)
    5. Trains an MLP: posterior_logits -> hiddens
    6. Reports MSE and R^2 for training and validation data at each epoch
    
    Parameters:
    -----------
    exp_name : str
        Experiment name (e.g., "train_...")
    layer : int
        Layer index to extract hidden representations from
    B : int, default=64
        Batch size for sampling
    n_samples : int, default=1000
        Total number of samples to use for training
    step : int, optional
        Step for checkpoint loading. If None, uses final checkpoint.
    n_minor : int, optional
        Number of minor tasks to use. If None, uses all available minor tasks.
        If -1, uses no minor tasks (only OOD if n_ood > 0).
    n_ood : int, default=0
        Number of OOD tasks to include
    hidden_size : int, default=64
        Number of neurons in the hidden layer of the MLP
    learning_rate : float, default=0.001
        Learning rate for Adam optimizer
    num_epochs : int, default=200
        Number of training epochs
    verbose : bool, default=False
        Whether to print progress messages
    positions : list, optional
        List of real token position indices to use (0-indexed). 
        If None, uses first 10 positions [0, 1, 2, ..., 9].
    validation_split : float, default=0.2
        Fraction of data to use for validation (between 0 and 1).
    uniform_sampling : bool, default=True
        If True, modifies p_minor to achieve uniform sampling across all tasks.
    skip_baselines : bool, default=False
        If True, skips training permutation and logits baselines to save time.
    
    Returns:
    --------
    results : dict
        Dictionary containing (for main, permutation baseline, and logits baseline):
        - 'train_mse' / 'val_mse': final MSE
        - 'train_r2' / 'val_r2': final R^2
        - 'mse_history': dict with 'train' and 'val' lists (per-epoch MSE)
        - 'r2_history': dict with 'train' and 'val' lists (per-epoch R^2)
        - 'model': the trained MLP model
        - 'baseline_*': permutation baseline metrics and histories
        - 'logits_baseline_*': logits baseline metrics and histories
        - Metadata: 'layer', 'n_tasks', 'hidden_dim', 'vocab_size', 'n_samples', etc.
    """
    from icl.latent_markov.markov_latent import task_posterior_over_time
    import torch.optim as optim
    
    # Load config/model/sampler
    _, sampler_orig, config = nu.load_everything("latent", exp_name)
    
    if step is None:
        step = config.training.num_epochs
    
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval()
    model.to(config.device)
    
    # Get sampler with specified n_minor and n_ood=0
    if n_minor is None:
        n_minor = 1000000
    elif n_minor == -1:
        n_minor = 0
    
    sampler, k_minor, _ = get_latent_sampler(exp_name, n_minor=n_minor, n_ood=0)
    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    
    if not getattr(sampler, "pad", False):
        raise ValueError("This function requires padded latent sequences (sampler.pad must be True)")
    
    # Modify p_minor to achieve uniform sampling across all tasks
    original_p_minor = sampler.p_minor
    if uniform_sampling and sampler.n_minor_tasks > 0:
        sampler.p_minor = sampler.n_minor_tasks / (sampler.n_major_tasks + sampler.n_minor_tasks)
        if verbose:
            logger.info(f"Modified p_minor from {original_p_minor:.6f} to {sampler.p_minor:.6f} for uniform task sampling")
    elif verbose and not uniform_sampling:
        logger.info(f"Using original p_minor: {original_p_minor:.6f} (not modified for uniform sampling)")
    
    if verbose:
        logger.info(f"Training MLP: posterior_logits -> hidden representations (layer {layer})")
        logger.info(f"MLP hidden_size: {hidden_size}, lr: {learning_rate}, epochs: {num_epochs}")
        logger.info(f"Number of tasks: {n_tasks} (major: {sampler.n_major_tasks}, minor: {sampler.n_minor_tasks}), Batch size: {B}, Total samples: {n_samples}")
    
    # Determine which positions to use
    seq_len = sampler.seq_len
    if positions is None:
        positions = list(range(min(10, seq_len)))
    else:
        positions = list(positions)
        if not all(0 <= p < seq_len for p in positions):
            raise ValueError(f"All positions must be in [0, {seq_len-1}], got {positions}")
    
    if verbose:
        logger.info(f"Using positions: {positions} (real token indices)")
    
    # Map real token positions to padded positions: position t -> 2*t + 1
    device = config.device
    padded_positions = torch.tensor([2 * t + 1 for t in positions], device=device, dtype=torch.long)
    
    # Collect samples, hiddens, logits, and posteriors
    all_hiddens = []
    all_logits = []
    all_posteriors = []
    
    n_batches = (n_samples + B - 1) // B
    
    if verbose:
        logger.info(f"Collecting {n_batches} batches of data...")
    
    for batch_idx in range(n_batches):
        samples, _ = sampler.generate(
            mode="train", task=None, num_samples=B, epochs=1
        )
        if samples.dim() == 3:
            samples = samples.squeeze(0)
        samples = samples.to(device)
        
        posteriors = task_posterior_over_time(sampler, samples)  # (B, L_real, T)
        posteriors = posteriors[:, positions, :]  # (B, len(positions), T)
        
        cache = {}
        layer_module = model.layers[layer].attn_block
        
        def hook_fn(module, inp, out):
            if torch.is_tensor(out):
                cached = out.index_select(dim=1, index=padded_positions).detach()
                cache["hidden"] = cached
            elif isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                cached = out[0].index_select(dim=1, index=padded_positions).detach()
                cache["hidden"] = cached
            else:
                raise RuntimeError(f"Unsupported hook output type: {type(out)}")
        
        handle = layer_module.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                logits_full = model(samples)
                logits_batch = logits_full.index_select(dim=1, index=padded_positions)
            hiddens_batch = cache["hidden"]
        finally:
            handle.remove()
        
        all_hiddens.append(hiddens_batch.cpu())
        all_logits.append(logits_batch.cpu())
        all_posteriors.append(posteriors.cpu())
        
        del samples, posteriors, hiddens_batch, logits_batch, logits_full
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Concatenate all batches
    hiddens_all = torch.cat(all_hiddens, dim=0)
    logits_all = torch.cat(all_logits, dim=0)
    posteriors_all = torch.cat(all_posteriors, dim=0)
    
    # Reshape: flatten sequence dimension
    n_total = hiddens_all.shape[0] * hiddens_all.shape[1]
    hiddens_flat = hiddens_all.reshape(n_total, -1)
    logits_flat = logits_all.reshape(n_total, -1)
    posteriors_flat = posteriors_all.reshape(n_total, -1)
    
    D = hiddens_flat.shape[1]
    vocab_size = logits_flat.shape[1]
    T = posteriors_flat.shape[1]
    
    # Convert posteriors to logit vectors
    eps = 1e-10
    posterior_logits = torch.log(posteriors_flat + eps)
    
    # Split into training and validation sets
    n_train = int(n_total * (1 - validation_split))
    indices = torch.randperm(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    posterior_logits_train = posterior_logits[train_indices].to(device)
    posterior_logits_val = posterior_logits[val_indices].to(device)
    hiddens_train = hiddens_flat[train_indices].to(device)
    hiddens_val = hiddens_flat[val_indices].to(device)
    logits_train = logits_flat[train_indices].to(device)
    logits_val = logits_flat[val_indices].to(device)
    
    if verbose:
        logger.info(f"Total data: {n_total} samples")
        logger.info(f"Training data: {n_train} samples ({100*(1-validation_split):.1f}%)")
        logger.info(f"Validation data: {len(val_indices)} samples ({100*validation_split:.1f}%)")
        logger.info(f"Input shape (posterior logits): {posterior_logits_train.shape}, Target shape (hiddens): {hiddens_train.shape}")
    
    # --- Helper: compute R^2 ---
    def _compute_r2(y_true, y_pred):
        ss_res = ((y_true - y_pred) ** 2).sum().item()
        ss_tot = ((y_true - y_true.mean(dim=0)) ** 2).sum().item()
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    
    # --- Helper: train an MLP and return metrics histories ---
    def _train_mlp(input_train, target_train, input_val, target_val, input_dim, output_dim, label="Model"):
        """
        Train a one-hidden-layer MLP: Linear(input_dim, hidden_size) -> ReLU -> Linear(hidden_size, output_dim).
        Returns (mlp_model, mse_history, r2_history, final_train_mse, final_val_mse, final_train_r2, final_val_r2).
        """
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim, bias=True),
        ).to(device)
        
        optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        mse_hist = {'train': [], 'val': []}
        r2_hist = {'train': [], 'val': []}
        
        for epoch in range(num_epochs):
            # Training phase
            mlp.train()
            optimizer.zero_grad()
            pred_train = mlp(input_train)
            train_loss = criterion(pred_train, target_train)
            train_loss.backward()
            optimizer.step()
            
            # Validation phase
            mlp.eval()
            with torch.no_grad():
                pred_val = mlp(input_val)
                val_loss = criterion(pred_val, target_val)
                
                train_r2 = _compute_r2(target_train, pred_train.detach())
                val_r2 = _compute_r2(target_val, pred_val)
            
            mse_hist['train'].append(train_loss.item())
            mse_hist['val'].append(val_loss.item())
            r2_hist['train'].append(train_r2)
            r2_hist['val'].append(val_r2)
            
            if verbose and (epoch + 1) % 20 == 0:
                logger.info(f"  {label} Epoch {epoch+1}/{num_epochs}, Train MSE: {train_loss.item():.6f}, Val MSE: {val_loss.item():.6f}, Train R²: {train_r2:.6f}, Val R²: {val_r2:.6f}")
        
        final_train_mse = mse_hist['train'][-1]
        final_val_mse = mse_hist['val'][-1]
        final_train_r2 = r2_hist['train'][-1]
        final_val_r2 = r2_hist['val'][-1]
        
        return mlp, mse_hist, r2_hist, final_train_mse, final_val_mse, final_train_r2, final_val_r2
    
    # --- Main model: posterior_logits -> hiddens ---
    if verbose:
        logger.info("Training MLP: posterior_logits -> hiddens...")
    
    main_model, mse_history, r2_history, train_mse, val_mse, train_r2, val_r2 = _train_mlp(
        posterior_logits_train, hiddens_train,
        posterior_logits_val, hiddens_val,
        T, D, label="Main"
    )
    
    if verbose:
        logger.info(f"Main model - Final Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}")
        logger.info(f"Main model - Final Train R²: {train_r2:.6f}, Val R²: {val_r2:.6f}")
    
    # Initialize baseline results
    baseline_train_mse = float('nan')
    baseline_val_mse = float('nan')
    baseline_train_r2 = float('nan')
    baseline_val_r2 = float('nan')
    baseline_model = None
    baseline_mse_history = {'train': [], 'val': []}
    baseline_r2_history = {'train': [], 'val': []}
    
    logits_baseline_train_mse = float('nan')
    logits_baseline_val_mse = float('nan')
    logits_baseline_train_r2 = float('nan')
    logits_baseline_val_r2 = float('nan')
    logits_baseline_model = None
    logits_baseline_mse_history = {'train': [], 'val': []}
    logits_baseline_r2_history = {'train': [], 'val': []}
    
    if not skip_baselines:
        # --- Permutation baseline: shuffle hiddens to break pairing ---
        if verbose:
            logger.info("Training permutation baseline (shuffled hiddens)...")
        
        hiddens_train_shuffled = hiddens_train[torch.randperm(n_train)]
        hiddens_val_shuffled = hiddens_val[torch.randperm(len(val_indices))]
        
        baseline_model, baseline_mse_history, baseline_r2_history, baseline_train_mse, baseline_val_mse, baseline_train_r2, baseline_val_r2 = _train_mlp(
            posterior_logits_train, hiddens_train_shuffled,
            posterior_logits_val, hiddens_val_shuffled,
            T, D, label="Permutation"
        )
        
        if verbose:
            logger.info(f"Permutation baseline - Final Train MSE: {baseline_train_mse:.6f}, Val MSE: {baseline_val_mse:.6f}")
            logger.info(f"Permutation baseline - Final Train R²: {baseline_train_r2:.6f}, Val R²: {baseline_val_r2:.6f}")
        
        # --- Logits baseline: model_logits -> hiddens ---
        if verbose:
            logger.info("Training logits baseline (model logits -> hiddens)...")
        
        logits_baseline_model, logits_baseline_mse_history, logits_baseline_r2_history, logits_baseline_train_mse, logits_baseline_val_mse, logits_baseline_train_r2, logits_baseline_val_r2 = _train_mlp(
            logits_train, hiddens_train,
            logits_val, hiddens_val,
            vocab_size, D, label="Logits"
        )
        
        if verbose:
            logger.info(f"Logits baseline - Final Train MSE: {logits_baseline_train_mse:.6f}, Val MSE: {logits_baseline_val_mse:.6f}")
            logger.info(f"Logits baseline - Final Train R²: {logits_baseline_train_r2:.6f}, Val R²: {logits_baseline_val_r2:.6f}")
            logger.info(f"Comparison (Val R²) - Main: {val_r2:.6f}, Permutation: {baseline_val_r2:.6f}, Logits: {logits_baseline_val_r2:.6f}")
    
    # Restore original p_minor
    if hasattr(sampler, 'p_minor') and 'original_p_minor' in locals():
        sampler.p_minor = original_p_minor
    
    # Move models back to CPU
    main_model = main_model.cpu()
    if baseline_model is not None:
        baseline_model = baseline_model.cpu()
    if logits_baseline_model is not None:
        logits_baseline_model = logits_baseline_model.cpu()
    
    return {
        'train_mse': train_mse,
        'val_mse': val_mse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'mse_history': mse_history,
        'r2_history': r2_history,
        'model': main_model,
        'baseline_train_mse': baseline_train_mse,
        'baseline_val_mse': baseline_val_mse,
        'baseline_train_r2': baseline_train_r2,
        'baseline_val_r2': baseline_val_r2,
        'baseline_mse_history': baseline_mse_history,
        'baseline_r2_history': baseline_r2_history,
        'baseline_model': baseline_model,
        'logits_baseline_train_mse': logits_baseline_train_mse,
        'logits_baseline_val_mse': logits_baseline_val_mse,
        'logits_baseline_train_r2': logits_baseline_train_r2,
        'logits_baseline_val_r2': logits_baseline_val_r2,
        'logits_baseline_mse_history': logits_baseline_mse_history,
        'logits_baseline_r2_history': logits_baseline_r2_history,
        'logits_baseline_model': logits_baseline_model,
        'layer': layer,
        'n_tasks': n_tasks,
        'hidden_dim': D,
        'vocab_size': vocab_size,
        'n_samples': n_total,
        'n_train': n_train,
        'n_val': len(val_indices),
    }


def train_linear_softmax_posterior_predictor_coin(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples: int = 1000,
    step: Optional[int] = None,
    n_minor: Optional[int] = None,
    n_ood: int = 0,
    learning_rate: float = 0.01,
    num_epochs: int = 100,
    verbose: bool = False,
    positions: Optional[list] = None,
    validation_split: float = 0.2,
    sample_mode: str = "train",
    uniform_sampling: bool = True,
    skip_baselines: bool = False,
) -> dict:
    """
    Train a linear softmax model to predict task posteriors from hidden representations.
    
    For coin task:
    1. Gets samples from the sampler
    2. Computes task posteriors using task_posterior_coins
    3. Extracts hidden representations at the specified layer
    4. Trains a linear softmax model to map hidden representations to posteriors
    5. Reports training loss
    
    Parameters:
    -----------
    exp_name : str
        Experiment name (e.g., "train_...")
    layer : int
        Layer index to extract hidden representations from
    B : int, default=64
        Batch size for sampling
    n_samples : int, default=1000
        Total number of samples to use for training
    step : int, optional
        Step for checkpoint loading. If None, uses final checkpoint.
    n_minor : int, optional
        Number of minor tasks to use. If None, uses all available minor tasks.
        If -1, uses no minor tasks (only OOD if n_ood > 0).
    n_ood : int, default=0
        Number of OOD tasks to include (always set to 0, no OOD tasks used)
    learning_rate : float, default=0.01
        Learning rate for training the linear model
    num_epochs : int, default=100
        Number of training epochs
    verbose : bool, default=False
        Whether to print progress messages
    positions : list, optional
        List of real token position indices to use (0-indexed). 
        If None, uses first 10 positions [0, 1, 2, ..., 9].
        These correspond to padded positions at indices [1, 3, 5, ...] in the padded sequence.
    validation_split : float, default=0.2
        Fraction of data to use for validation (between 0 and 1).
        The remaining fraction is used for training.
    uniform_sampling : bool, default=True
        If True, modifies p_minor to achieve uniform sampling across all tasks
        (each task has equal probability). If False, uses the original sampler's p_minor.
    skip_baselines : bool, default=False
        If True, skips training permutation and logits baselines to save time.
        Only trains the main model (hiddens -> posteriors).
    
    Note:
    - Always uses "train" mode to sample from mixture of major/minor tasks (no OOD)
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'final_loss': float, final training loss
        - 'final_val_loss': float, final validation loss
        - 'loss_history': list of training losses during training
        - 'val_loss_history': list of validation losses during training
        - 'model': the trained linear model
        - 'baseline_final_loss': float, final training loss for permutation baseline
        - 'baseline_final_val_loss': float, final validation loss for permutation baseline
        - 'baseline_loss_history': list of training losses for baseline (shuffled data)
        - 'baseline_val_loss_history': list of validation losses for baseline (shuffled data)
        - 'baseline_model': the trained baseline model (trained on shuffled data)
        - 'logits_baseline_final_loss': float, final training loss for logits baseline
        - 'logits_baseline_final_val_loss': float, final validation loss for logits baseline
        - 'logits_baseline_loss_history': list of training losses for logits baseline
        - 'logits_baseline_val_loss_history': list of validation losses for logits baseline
        - 'logits_baseline_model': the trained logits baseline model (trained on logits)
        - 'layer': int, the layer index used
        - 'n_tasks': int, number of tasks
    """
    from icl.coin.coin import task_posterior_coins
    from icl.utils.coin_ood_analysis import get_new_sampler
    from torch import nn
    import torch.optim as optim
    
    # Load config/model/sampler
    _, sampler_orig, config = nu.load_everything("coin", exp_name)
    
    if step is None:
        step = config.training.num_epochs
    
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval()
    model.to(config.device)
    
    # Get sampler with specified n_minor and n_ood=0 (no OOD tasks)
    if n_minor is None:
        n_minor = 1000000  # Use all available
    elif n_minor == -1:
        n_minor = 0
    
    sampler, k_minor = get_new_sampler(exp_name, n_minor, n_ood=0)  # Always use n_ood=0
    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    
    if not getattr(sampler, "pad", False):
        raise ValueError("This function requires padded coin sequences (sampler.pad must be True)")
    
    # Optionally modify p_minor to achieve uniform sampling across all tasks
    # For uniform sampling: P(any task) = 1/(n_major + n_minor)
    # This requires: p_minor = n_minor / (n_major + n_minor)
    original_p_minor = sampler.p_minor
    if uniform_sampling and sampler.n_minor_tasks > 0:
        sampler.p_minor = sampler.n_minor_tasks / (sampler.n_major_tasks + sampler.n_minor_tasks)
        if verbose:
            logger.info(f"Modified p_minor from {original_p_minor:.6f} to {sampler.p_minor:.6f} for uniform task sampling")
    elif verbose and not uniform_sampling:
        logger.info(f"Using original p_minor: {original_p_minor:.6f} (not modified for uniform sampling)")
    
    if verbose:
        logger.info(f"Training linear softmax model to predict posteriors from layer {layer} hidden representations")
        logger.info(f"Number of tasks: {n_tasks} (major: {sampler.n_major_tasks}, minor: {sampler.n_minor_tasks}), Batch size: {B}, Total samples: {n_samples}")
    
    # Determine which positions to use (before the loop)
    seq_len = sampler.seq_len
    if positions is None:
        # Default: use first 10 positions
        positions = list(range(min(10, seq_len)))
    else:
        # Validate positions
        positions = list(positions)
        if not all(0 <= p < seq_len for p in positions):
            raise ValueError(f"All positions must be in [0, {seq_len-1}], got {positions}")
    
    if verbose:
        logger.info(f"Using positions: {positions} (real token indices)")
    
    # Map real token positions to padded positions: position t -> 2*t + 1
    device = config.device
    padded_positions = torch.tensor([2 * t + 1 for t in positions], device=device, dtype=torch.long)
    
    # Collect samples, hiddens, logits, and posteriors
    all_hiddens = []
    all_logits = []
    all_posteriors = []
    
    n_batches = (n_samples + B - 1) // B  # Ceiling division
    
    if verbose:
        logger.info(f"Collecting {n_batches} batches of data...")
    
    for batch_idx in range(n_batches):
        # Always use train mode: sample from mixture of major/minor tasks (no OOD)
        # Note: train mode returns (epochs, num_samples//epochs, -1), so we need to reshape
        samples, _ = sampler.generate(
            mode="train", task=None, num_samples=B, epochs=1
        )
        # Reshape from (1, B, L) to (B, L) if needed
        if samples.dim() == 3:
            samples = samples.squeeze(0)  # (1, B, L) -> (B, L)
        samples = samples.to(device)
        
        # Compute task posteriors: shape (B, T) where T is number of tasks
        # task_posterior_coins returns (N, Ktot) for 2D input or (E, N, Ktot) for 3D
        posteriors = task_posterior_coins(sampler, samples, include_minor=True)  # (B, T)
        
        # Extract hidden representations at specified layer
        # We need hiddens at padded positions that correspond to real token positions
        
        # Extract hiddens at these padded positions for the whole batch at once
        cache = {}
        layer_module = model.layers[layer].attn_block
        
        def hook_fn(module, inp, out):
            if torch.is_tensor(out):
                # out: (B, L_obs, D)
                cached = out.index_select(dim=1, index=padded_positions).detach()  # (B, len(positions), D)
                cache["hidden"] = cached
            elif isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                cached = out[0].index_select(dim=1, index=padded_positions).detach()  # (B, len(positions), D)
                cache["hidden"] = cached
            else:
                raise RuntimeError(f"Unsupported hook output type: {type(out)}")
        
        handle = layer_module.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                # Get logits from model output
                logits_full = model(samples)  # (B, L_obs, vocab_size)
                # Extract logits at padded positions
                logits_batch = logits_full.index_select(dim=1, index=padded_positions)  # (B, len(positions), vocab_size)
            hiddens_batch = cache["hidden"]  # (B, len(positions), D)
        finally:
            handle.remove()
        
        # Expand posteriors to match positions: (B, T) -> (B, len(positions), T)
        # For coin task, posteriors are the same across positions (i.i.d. assumption)
        posteriors_expanded = posteriors.unsqueeze(1).expand(-1, len(positions), -1)  # (B, len(positions), T)
        
        # Move to CPU to save GPU memory
        all_hiddens.append(hiddens_batch.cpu())
        all_logits.append(logits_batch.cpu())
        all_posteriors.append(posteriors_expanded.cpu())
        
        # Clear GPU memory
        del samples, posteriors, hiddens_batch, logits_batch, logits_full
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Concatenate all batches
    hiddens_all = torch.cat(all_hiddens, dim=0)  # (n_samples, len(positions), D)
    logits_all = torch.cat(all_logits, dim=0)  # (n_samples, len(positions), vocab_size)
    posteriors_all = torch.cat(all_posteriors, dim=0)  # (n_samples, len(positions), T)
    
    # Reshape: flatten sequence dimension
    # hiddens: (n_samples * len(positions), D)
    # logits: (n_samples * len(positions), vocab_size)
    # posteriors: (n_samples * len(positions), T)
    n_total = hiddens_all.shape[0] * hiddens_all.shape[1]
    hiddens_flat = hiddens_all.reshape(n_total, -1)  # (n_total, D)
    logits_flat = logits_all.reshape(n_total, -1)  # (n_total, vocab_size)
    posteriors_flat = posteriors_all.reshape(n_total, -1)  # (n_total, T)
    
    D = hiddens_flat.shape[1]
    vocab_size = logits_flat.shape[1]
    T = posteriors_flat.shape[1]
    
    # Split into training and validation sets
    n_train = int(n_total * (1 - validation_split))
    indices = torch.randperm(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    hiddens_train = hiddens_flat[train_indices]
    logits_train = logits_flat[train_indices]
    posteriors_train = posteriors_flat[train_indices]
    hiddens_val = hiddens_flat[val_indices]
    logits_val = logits_flat[val_indices]
    posteriors_val = posteriors_flat[val_indices]
    
    if verbose:
        logger.info(f"Total data: {n_total} samples")
        logger.info(f"Training data: {n_train} samples ({100*(1-validation_split):.1f}%)")
        logger.info(f"Validation data: {len(val_indices)} samples ({100*validation_split:.1f}%)")
        logger.info(f"Training data shape: hiddens {hiddens_train.shape}, logits {logits_train.shape}, posteriors {posteriors_train.shape}")
    
    # Create linear softmax model: hidden (D) -> logits (T) -> softmax -> posterior (T)
    linear_model = nn.Sequential(
        nn.Linear(D, T, bias=True),
        nn.Softmax(dim=-1)
    ).to(device)
    
    # Move data to device
    hiddens_train = hiddens_train.to(device)
    logits_train = logits_train.to(device)
    posteriors_train = posteriors_train.to(device)
    hiddens_val = hiddens_val.to(device)
    logits_val = logits_val.to(device)
    posteriors_val = posteriors_val.to(device)
    
    # Training setup
    optimizer = optim.Adam(linear_model.parameters(), lr=learning_rate)
    criterion = nn.KLDivLoss(reduction='batchmean')  # KL divergence for probability distributions
    
    loss_history = []
    val_loss_history = []
    
    if verbose:
        logger.info(f"Training linear model for {num_epochs} epochs...")
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        linear_model.train()
        optimizer.zero_grad()
        
        # Forward pass on training data
        pred_posteriors_train = linear_model(hiddens_train)  # (n_train, T)
        
        # Compute training loss: KL divergence between predicted and true posteriors
        # KLDivLoss expects log probabilities, so we need log of predictions
        log_pred_train = torch.log(pred_posteriors_train + 1e-10)  # Add small epsilon for numerical stability
        train_loss = criterion(log_pred_train, posteriors_train)
        
        # Backward pass
        train_loss.backward()
        optimizer.step()
        
        # Validation phase
        linear_model.eval()
        with torch.no_grad():
            pred_posteriors_val = linear_model(hiddens_val)  # (n_val, T)
            log_pred_val = torch.log(pred_posteriors_val + 1e-10)
            val_loss = criterion(log_pred_val, posteriors_val)
        
        loss_history.append(train_loss.item())
        val_loss_history.append(val_loss.item())
        
        if verbose and (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
    
    final_loss = loss_history[-1]
    final_val_loss = val_loss_history[-1]
    
    if verbose:
        logger.info(f"Training completed. Final train loss: {final_loss:.6f}, Final val loss: {final_val_loss:.6f}")
    
    # Initialize baseline results with None/NaN if skipping
    final_baseline_loss = float('nan')
    final_baseline_val_loss = float('nan')
    baseline_loss_history = []
    baseline_val_loss_history = []
    baseline_model = None
    
    final_logits_baseline_loss = float('nan')
    final_logits_baseline_val_loss = float('nan')
    logits_baseline_loss_history = []
    logits_baseline_val_loss_history = []
    logits_baseline_model = None
    
    if not skip_baselines:
        # Permutation baseline: shuffle posteriors to break the pairing with hiddens
        if verbose:
            logger.info("Training permutation baseline (shuffled posteriors)...")
        
        # Create shuffled posteriors (shuffle independently for train and val)
        posteriors_train_shuffled = posteriors_train[torch.randperm(n_train)]
        posteriors_val_shuffled = posteriors_val[torch.randperm(len(val_indices))]
        
        # Create a new model for the baseline
        baseline_model = nn.Sequential(
            nn.Linear(D, T, bias=True),
            nn.Softmax(dim=-1)
        ).to(device)
        
        # Training setup for baseline
        baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=learning_rate)
        
        baseline_loss_history = []
        baseline_val_loss_history = []
        
        # Training loop for baseline
        for epoch in range(num_epochs):
            # Training phase
            baseline_model.train()
            baseline_optimizer.zero_grad()
            
            # Forward pass on training data with shuffled posteriors
            pred_posteriors_train_baseline = baseline_model(hiddens_train)  # (n_train, T)
            log_pred_train_baseline = torch.log(pred_posteriors_train_baseline + 1e-10)
            train_loss_baseline = criterion(log_pred_train_baseline, posteriors_train_shuffled)
            
            # Backward pass
            train_loss_baseline.backward()
            baseline_optimizer.step()
            
            # Validation phase
            baseline_model.eval()
            with torch.no_grad():
                pred_posteriors_val_baseline = baseline_model(hiddens_val)  # (n_val, T)
                log_pred_val_baseline = torch.log(pred_posteriors_val_baseline + 1e-10)
                val_loss_baseline = criterion(log_pred_val_baseline, posteriors_val_shuffled)
            
            baseline_loss_history.append(train_loss_baseline.item())
            baseline_val_loss_history.append(val_loss_baseline.item())
            
            if verbose and (epoch + 1) % 20 == 0:
                logger.info(f"  Baseline Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_baseline.item():.6f}, Val Loss: {val_loss_baseline.item():.6f}")
        
        final_baseline_loss = baseline_loss_history[-1]
        final_baseline_val_loss = baseline_val_loss_history[-1]
        
        if verbose:
            logger.info(f"Baseline completed. Final train loss: {final_baseline_loss:.6f}, Final val loss: {final_baseline_val_loss:.6f}")
            logger.info(f"Improvement over baseline - Train: {final_loss - final_baseline_loss:.6f}, Val: {final_val_loss - final_baseline_val_loss:.6f}")
        
        # Logits baseline: train a model to predict posteriors from logits
        if verbose:
            logger.info("Training logits baseline (predicting posteriors from logits)...")
        
        # Create a new model for logits baseline
        logits_baseline_model = nn.Sequential(
            nn.Linear(vocab_size, T, bias=True),
            nn.Softmax(dim=-1)
        ).to(device)
        
        # Training setup for logits baseline
        logits_baseline_optimizer = optim.Adam(logits_baseline_model.parameters(), lr=learning_rate)
        
        logits_baseline_loss_history = []
        logits_baseline_val_loss_history = []
        
        # Training loop for logits baseline
        for epoch in range(num_epochs):
            # Training phase
            logits_baseline_model.train()
            logits_baseline_optimizer.zero_grad()
            
            # Forward pass on training data using logits
            pred_posteriors_train_logits = logits_baseline_model(logits_train)  # (n_train, T)
            log_pred_train_logits = torch.log(pred_posteriors_train_logits + 1e-10)
            train_loss_logits = criterion(log_pred_train_logits, posteriors_train)
            
            # Backward pass
            train_loss_logits.backward()
            logits_baseline_optimizer.step()
            
            # Validation phase
            logits_baseline_model.eval()
            with torch.no_grad():
                pred_posteriors_val_logits = logits_baseline_model(logits_val)  # (n_val, T)
                log_pred_val_logits = torch.log(pred_posteriors_val_logits + 1e-10)
                val_loss_logits = criterion(log_pred_val_logits, posteriors_val)
            
            logits_baseline_loss_history.append(train_loss_logits.item())
            logits_baseline_val_loss_history.append(val_loss_logits.item())
            
            if verbose and (epoch + 1) % 20 == 0:
                logger.info(f"  Logits Baseline Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_logits.item():.6f}, Val Loss: {val_loss_logits.item():.6f}")
        
        final_logits_baseline_loss = logits_baseline_loss_history[-1]
        final_logits_baseline_val_loss = logits_baseline_val_loss_history[-1]
        
        if verbose:
            logger.info(f"Logits baseline completed. Final train loss: {final_logits_baseline_loss:.6f}, Final val loss: {final_logits_baseline_val_loss:.6f}")
            logger.info(f"Comparison - Hiddens vs Logits - Train: {final_loss - final_logits_baseline_loss:.6f}, Val: {final_val_loss - final_logits_baseline_val_loss:.6f}")
    
    # Restore original p_minor
    if hasattr(sampler, 'p_minor') and 'original_p_minor' in locals():
        sampler.p_minor = original_p_minor
    
    # Move models back to CPU
    linear_model = linear_model.cpu()
    if baseline_model is not None:
        baseline_model = baseline_model.cpu()
    if logits_baseline_model is not None:
        logits_baseline_model = logits_baseline_model.cpu()
    
    return {
        'final_loss': final_loss,
        'final_val_loss': final_val_loss,
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'model': linear_model,
        'baseline_final_loss': final_baseline_loss,
        'baseline_final_val_loss': final_baseline_val_loss,
        'baseline_loss_history': baseline_loss_history,
        'baseline_val_loss_history': baseline_val_loss_history,
        'baseline_model': baseline_model,
        'logits_baseline_final_loss': final_logits_baseline_loss,
        'logits_baseline_final_val_loss': final_logits_baseline_val_loss,
        'logits_baseline_loss_history': logits_baseline_loss_history,
        'logits_baseline_val_loss_history': logits_baseline_val_loss_history,
        'logits_baseline_model': logits_baseline_model,
        'layer': layer,
        'n_tasks': n_tasks,
        'hidden_dim': D,
        'vocab_size': vocab_size,
        'n_samples': n_total,
        'n_train': n_train,
        'n_val': len(val_indices),
    }


def train_linear_softmax_posterior_predictor_linear(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples: int = 1000,
    step: Optional[int] = None,
    learning_rate: float = 0.01,
    num_epochs: int = 100,
    verbose: bool = False,
    position: Union[int, list] = -1,
    validation_split: float = 0.2,
    uniform_sampling: bool = True,
    skip_baselines: bool = False,
) -> dict:
    """
    Train a linear softmax model to predict task posteriors from hidden representations.
    
    For linear regression task:
    1. Gets samples from the task (data and targets)
    2. Computes task posteriors using task_posterior_linear_regression
    3. Extracts hidden representations at the specified layer and position(s)
    4. Trains a linear softmax model to map hidden representations to posteriors
    5. Reports training loss
    
    Parameters:
    -----------
    exp_name : str
        Experiment name (e.g., "train_...")
    layer : int
        Layer index to extract hidden representations from
    B : int, default=64
        Batch size for sampling
    n_samples : int, default=1000
        Total number of samples to use for training
    step : int, optional
        Step for checkpoint loading. If None, uses final checkpoint.
    learning_rate : float, default=0.01
        Learning rate for training the linear model
    num_epochs : int, default=100
        Number of training epochs
    verbose : bool, default=False
        Whether to print progress messages
    position : int or list, default=-1
        Position index(ices) to extract hidden representations from.
        - If int: single position index. -1 means the final position (after all data points).
        - If list: multiple position indices. Each position's hidden will be paired with the same posterior.
        For linear regression, positions are typically at 3*i+1 for padded sequences.
    validation_split : float, default=0.2
        Fraction of data to use for validation (between 0 and 1).
        The remaining fraction is used for training.
    uniform_sampling : bool, default=True
        If True, modifies p_minor to achieve uniform sampling across all tasks
        (each task has equal probability). If False, uses the original sampler's p_minor.
    skip_baselines : bool, default=False
        If True, skips training permutation and logits baselines to save time.
        Only trains the main model (hiddens -> posteriors).
    
    Note:
    - Always uses train mode to sample from mixture of major/minor tasks (no OOD)
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'final_loss': float, final training loss
        - 'final_val_loss': float, final validation loss
        - 'loss_history': list of training losses during training
        - 'val_loss_history': list of validation losses during training
        - 'model': the trained linear model
        - 'baseline_final_loss': float, final training loss for permutation baseline
        - 'baseline_final_val_loss': float, final validation loss for permutation baseline
        - 'baseline_loss_history': list of training losses for baseline (shuffled data)
        - 'baseline_val_loss_history': list of validation losses for baseline (shuffled data)
        - 'baseline_model': the trained baseline model (trained on shuffled data)
        - 'logits_baseline_final_loss': float, final training loss for logits baseline
        - 'logits_baseline_final_val_loss': float, final validation loss for logits baseline
        - 'logits_baseline_loss_history': list of training losses for logits baseline
        - 'logits_baseline_val_loss_history': list of validation losses for logits baseline
        - 'logits_baseline_model': the trained logits baseline model (trained on logits)
        - 'layer': int, the layer index used
        - 'n_tasks': int, number of tasks
    """
    from icl.linear.lr_task import task_posterior_linear_regression
    from torch import nn
    import torch.optim as optim
    
    # Load config/model/task
    model_loaded, train_task, config = load_model_task_config(exp_name)
    
    if step is None:
        step = config.training.total_steps
    
    # Load model from checkpoint (may be different step than default)
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval()
    model.to(config.device)
    
    n_tasks = train_task.n_tasks
    if n_tasks <= 0:
        raise ValueError("This function requires a finite task pool (n_tasks > 0)")
    
    # Determine total number of tasks (major + minor if included)
    include_minor = train_task.n_minor_tasks > 0 and train_task.minor_pool is not None
    if include_minor:
        n_total_tasks = n_tasks + train_task.n_minor_tasks
    else:
        n_total_tasks = n_tasks
    
    # Optionally modify p_minor to achieve uniform sampling across all tasks
    # For uniform sampling: P(any task) = 1/(n_tasks + n_minor_tasks)
    # This requires: p_minor = n_minor_tasks / (n_tasks + n_minor_tasks)
    original_p_minor = train_task.p_minor
    if uniform_sampling and include_minor:
        train_task.p_minor = train_task.n_minor_tasks / (train_task.n_tasks + train_task.n_minor_tasks)
        if verbose:
            logger.info(f"Modified p_minor from {original_p_minor:.6f} to {train_task.p_minor:.6f} for uniform task sampling")
    elif verbose and not uniform_sampling:
        logger.info(f"Using original p_minor: {original_p_minor:.6f} (not modified for uniform sampling)")
    
    # Normalize position to a list
    if isinstance(position, int):
        positions = [position]
    else:
        positions = list(position)
    
    if verbose:
        logger.info(f"Training linear softmax model to predict posteriors from layer {layer} hidden representations")
        logger.info(f"Number of tasks: {n_tasks} (major), {train_task.n_minor_tasks} (minor), Total: {n_total_tasks}")
        logger.info(f"Batch size: {B}, Total samples: {n_samples}, Positions: {positions}")
        if uniform_sampling:
            logger.info("Using uniform task sampling (modified p_minor)")
        else:
            logger.info("Using original sampler's p_minor (not modified)")
    
    # Collect samples, hiddens, logits, and posteriors
    device = config.device
    all_hiddens = []
    all_logits = []
    all_posteriors = []
    
    n_batches = (n_samples + B - 1) // B  # Ceiling division
    
    if verbose:
        logger.info(f"Collecting {n_batches} batches of data...")
    
    # Store original batch size and restore later
    original_batch_size = train_task.batch_size
    train_task.batch_size = B
    
    try:
        for batch_idx in range(n_batches):
            # Sample data and targets from the task (mixture of tasks)
            # sample_batch returns (data, tasks, targets)
            demo_data, _, demo_target = train_task.sample_batch(step=batch_idx, is_eval=False)
            demo_data = demo_data.to(device)
            demo_target = demo_target.to(device)
            
            # Compute task posteriors: shape (B, K) where K = n_total_tasks
            posteriors = task_posterior_linear_regression(
                train_task, 
                demo_data, 
                demo_target,
                include_minor=include_minor
            )  # (B, K)
            
            # Extract hidden representations at specified layer and positions
            cache = {}
            layer_module = model.transformer.blocks[layer].attn_block
            
            # Helper function to convert positions to indices
            def get_pos_indices(seq_length):
                pos_indices = []
                for pos in positions:
                    if pos == -1:
                        pos_indices.append(seq_length - 1)
                    else:
                        if pos >= seq_length:
                            raise ValueError(f"Position {pos} >= sequence length {seq_length}")
                        pos_indices.append(pos)
                return torch.tensor(pos_indices, device=device, dtype=torch.long)
            
            # Forward pass to get sequence length first
            with torch.no_grad():
                logits_full = model(demo_data, demo_target)  # (B, L, vocab_size)
                seq_len = logits_full.size(1)
            
            # Now we know seq_len, compute position indices
            pos_indices = get_pos_indices(seq_len)
            
            def hook_fn(module, inp, out):
                if torch.is_tensor(out):
                    # out: (B, L, D)
                    cache["hidden"] = out.index_select(dim=1, index=pos_indices).detach()  # (B, len(positions), D)
                elif isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                    cache["hidden"] = out[0].index_select(dim=1, index=pos_indices).detach()  # (B, len(positions), D)
                else:
                    raise RuntimeError(f"Unsupported hook output type: {type(out)}")
            
            handle = layer_module.register_forward_hook(hook_fn)
            try:
                with torch.no_grad():
                    # Forward pass again to extract hiddens (we already have logits)
                    _ = model(demo_data, demo_target)
                    # Extract logits at the same positions
                    logits_batch = logits_full.index_select(dim=1, index=pos_indices)  # (B, len(positions), vocab_size)
                hiddens_batch = cache["hidden"]  # (B, len(positions), D)
            finally:
                handle.remove()
            
            # Expand posteriors to match positions: (B, K) -> (B, len(positions), K)
            # Each position gets the same posterior
            posteriors_expanded = posteriors.unsqueeze(1).expand(-1, len(positions), -1)  # (B, len(positions), K)
            
            # Move to CPU to save GPU memory
            all_hiddens.append(hiddens_batch.cpu())
            all_logits.append(logits_batch.cpu())
            all_posteriors.append(posteriors_expanded.cpu())
            
            # Clear GPU memory
            del demo_data, demo_target, posteriors, hiddens_batch, logits_batch, logits_full
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        # Restore original batch size and p_minor
        train_task.batch_size = original_batch_size
        if 'original_p_minor' in locals():
            train_task.p_minor = original_p_minor
    
    # Concatenate all batches
    hiddens_all = torch.cat(all_hiddens, dim=0)  # (n_samples, len(positions), D)
    logits_all = torch.cat(all_logits, dim=0)  # (n_samples, len(positions), vocab_size)
    posteriors_all = torch.cat(all_posteriors, dim=0)  # (n_samples, len(positions), K)
    
    # Reshape: flatten position dimension
    # hiddens: (n_samples * len(positions), D)
    # logits: (n_samples * len(positions), vocab_size)
    # posteriors: (n_samples * len(positions), K)
    n_samples_actual = hiddens_all.shape[0]
    n_positions = hiddens_all.shape[1]
    n_total = n_samples_actual * n_positions
    
    hiddens_flat = hiddens_all.reshape(n_total, -1)  # (n_total, D)
    logits_flat = logits_all.reshape(n_total, -1)  # (n_total, vocab_size)
    posteriors_flat = posteriors_all.reshape(n_total, -1)  # (n_total, K)
    
    D = hiddens_flat.shape[1]
    vocab_size = logits_flat.shape[1]
    T = posteriors_flat.shape[1]  # Number of tasks
    
    # Split into training and validation sets
    n_train = int(n_total * (1 - validation_split))
    indices = torch.randperm(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Use flattened tensors for indexing - keep on CPU to save GPU memory
    hiddens_train = hiddens_flat[train_indices].cpu()
    logits_train = logits_flat[train_indices].cpu()
    posteriors_train = posteriors_flat[train_indices].cpu()
    hiddens_val = hiddens_flat[val_indices].cpu()
    logits_val = logits_flat[val_indices].cpu()
    posteriors_val = posteriors_flat[val_indices].cpu()
    
    # Clean up intermediate tensors
    del hiddens_all, logits_all, posteriors_all, hiddens_flat, logits_flat, posteriors_flat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    if verbose:
        logger.info(f"Total data: {n_total} samples")
        logger.info(f"Training data: {n_train} samples ({100*(1-validation_split):.1f}%)")
        logger.info(f"Validation data: {len(val_indices)} samples ({100*validation_split:.1f}%)")
        logger.info(f"Training data shape: hiddens {hiddens_train.shape}, logits {logits_train.shape}, posteriors {posteriors_train.shape}")
    
    # Use mini-batch training to reduce GPU memory usage
    train_batch_size = min(4096, n_train)  # Use smaller batches for training
    val_batch_size = min(4096, len(val_indices))  # Use smaller batches for validation
    
    if verbose:
        logger.info(f"Using mini-batch training: train_batch_size={train_batch_size}, val_batch_size={val_batch_size}")
    
    # Create linear softmax model: hidden (D) -> logits (T) -> softmax -> posterior (T)
    linear_model = nn.Sequential(
        nn.Linear(D, T, bias=True),
        nn.Softmax(dim=-1)
    ).to(device)
    
    # Training setup
    optimizer = optim.Adam(linear_model.parameters(), lr=learning_rate)
    criterion = nn.KLDivLoss(reduction='batchmean')  # KL divergence for probability distributions
    
    loss_history = []
    val_loss_history = []
    
    if verbose:
        logger.info(f"Training linear model for {num_epochs} epochs...")
    
    # Training loop with mini-batches
    for epoch in range(num_epochs):
        # Training phase - use mini-batches
        linear_model.train()
        optimizer.zero_grad()
        
        train_losses = []
        # Process training data in mini-batches
        for i in range(0, n_train, train_batch_size):
            end_idx = min(i + train_batch_size, n_train)
            h_batch = hiddens_train[i:end_idx].to(device)
            p_batch = posteriors_train[i:end_idx].to(device)
            
            pred_posteriors_batch = linear_model(h_batch)
            log_pred_batch = torch.log(pred_posteriors_batch + 1e-10)
            batch_loss = criterion(log_pred_batch, p_batch) * (end_idx - i) / n_train  # Weight by batch size
            
            batch_loss.backward()
            train_losses.append(batch_loss.item() * n_train / (end_idx - i))  # Unweight for logging
            
            del h_batch, p_batch, pred_posteriors_batch, log_pred_batch, batch_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        optimizer.step()
        train_loss = sum(train_losses) / len(train_losses)
        
        # Validation phase - use mini-batches
        linear_model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(0, len(val_indices), val_batch_size):
                end_idx = min(i + val_batch_size, len(val_indices))
                h_batch = hiddens_val[i:end_idx].to(device)
                p_batch = posteriors_val[i:end_idx].to(device)
                
                pred_posteriors_batch = linear_model(h_batch)
                log_pred_batch = torch.log(pred_posteriors_batch + 1e-10)
                batch_loss = criterion(log_pred_batch, p_batch)
                val_losses.append(batch_loss.item())
                
                del h_batch, p_batch, pred_posteriors_batch, log_pred_batch, batch_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        val_loss = sum(val_losses) / len(val_losses)
        
        loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        if verbose and (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    final_loss = loss_history[-1]
    final_val_loss = val_loss_history[-1]
    
    if verbose:
        logger.info(f"Training completed. Final train loss: {final_loss:.6f}, Final val loss: {final_val_loss:.6f}")
    
    # Initialize baseline results with None/NaN if skipping
    final_baseline_loss = float('nan')
    final_baseline_val_loss = float('nan')
    baseline_loss_history = []
    baseline_val_loss_history = []
    baseline_model = None
    
    final_logits_baseline_loss = float('nan')
    final_logits_baseline_val_loss = float('nan')
    logits_baseline_loss_history = []
    logits_baseline_val_loss_history = []
    logits_baseline_model = None
    
    if not skip_baselines:
        # Permutation baseline: shuffle posteriors to break the pairing with hiddens
        if verbose:
            logger.info("Training permutation baseline (shuffled posteriors)...")
        
        # Create shuffled posteriors (shuffle independently for train and val) - keep on CPU
        posteriors_train_shuffled = posteriors_train[torch.randperm(n_train)].cpu()
        posteriors_val_shuffled = posteriors_val[torch.randperm(len(val_indices))].cpu()
        
        # Clear GPU cache before baseline training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Create a new model for the baseline
        baseline_model = nn.Sequential(
            nn.Linear(D, T, bias=True),
            nn.Softmax(dim=-1)
        ).to(device)
        
        # Training setup for baseline
        baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=learning_rate)
        
        baseline_loss_history = []
        baseline_val_loss_history = []
        
        # Training loop for baseline - use mini-batches
        for epoch in range(num_epochs):
            # Training phase
            baseline_model.train()
            baseline_optimizer.zero_grad()
            
            baseline_train_losses = []
            for i in range(0, n_train, train_batch_size):
                end_idx = min(i + train_batch_size, n_train)
                h_batch = hiddens_train[i:end_idx].to(device)
                p_batch = posteriors_train_shuffled[i:end_idx].to(device)
                
                pred_posteriors_batch = baseline_model(h_batch)
                log_pred_batch = torch.log(pred_posteriors_batch + 1e-10)
                batch_loss = criterion(log_pred_batch, p_batch) * (end_idx - i) / n_train
                
                batch_loss.backward()
                baseline_train_losses.append(batch_loss.item() * n_train / (end_idx - i))
                
                del h_batch, p_batch, pred_posteriors_batch, log_pred_batch, batch_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            baseline_optimizer.step()
            train_loss_baseline = sum(baseline_train_losses) / len(baseline_train_losses)
            
            # Validation phase
            baseline_model.eval()
            baseline_val_losses = []
            with torch.no_grad():
                for i in range(0, len(val_indices), val_batch_size):
                    end_idx = min(i + val_batch_size, len(val_indices))
                    h_batch = hiddens_val[i:end_idx].to(device)
                    p_batch = posteriors_val_shuffled[i:end_idx].to(device)
                    
                    pred_posteriors_batch = baseline_model(h_batch)
                    log_pred_batch = torch.log(pred_posteriors_batch + 1e-10)
                    batch_loss = criterion(log_pred_batch, p_batch)
                    baseline_val_losses.append(batch_loss.item())
                    
                    del h_batch, p_batch, pred_posteriors_batch, log_pred_batch, batch_loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            val_loss_baseline = sum(baseline_val_losses) / len(baseline_val_losses)
            
            baseline_loss_history.append(train_loss_baseline)
            baseline_val_loss_history.append(val_loss_baseline)
            
            if verbose and (epoch + 1) % 20 == 0:
                logger.info(f"  Baseline Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_baseline:.6f}, Val Loss: {val_loss_baseline:.6f}")
        
        final_baseline_loss = baseline_loss_history[-1]
        final_baseline_val_loss = baseline_val_loss_history[-1]
        
        if verbose:
            logger.info(f"Baseline completed. Final train loss: {final_baseline_loss:.6f}, Final val loss: {final_baseline_val_loss:.6f}")
            logger.info(f"Improvement over baseline - Train: {final_loss - final_baseline_loss:.6f}, Val: {final_val_loss - final_baseline_val_loss:.6f}")
        
        # Logits baseline: train a model to predict posteriors from logits
        if verbose:
            logger.info("Training logits baseline (predicting posteriors from logits)...")
        
        # Clear GPU cache before logits baseline training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Create a new model for logits baseline
        logits_baseline_model = nn.Sequential(
            nn.Linear(vocab_size, T, bias=True),
            nn.Softmax(dim=-1)
        ).to(device)
        
        # Training setup for logits baseline
        logits_baseline_optimizer = optim.Adam(logits_baseline_model.parameters(), lr=learning_rate)
        
        logits_baseline_loss_history = []
        logits_baseline_val_loss_history = []
        
        # Training loop for logits baseline - use mini-batches
        for epoch in range(num_epochs):
            # Training phase
            logits_baseline_model.train()
            logits_baseline_optimizer.zero_grad()
            
            logits_train_losses = []
            for i in range(0, n_train, train_batch_size):
                end_idx = min(i + train_batch_size, n_train)
                l_batch = logits_train[i:end_idx].to(device)
                p_batch = posteriors_train[i:end_idx].to(device)
                
                pred_posteriors_batch = logits_baseline_model(l_batch)
                log_pred_batch = torch.log(pred_posteriors_batch + 1e-10)
                batch_loss = criterion(log_pred_batch, p_batch) * (end_idx - i) / n_train
                
                batch_loss.backward()
                logits_train_losses.append(batch_loss.item() * n_train / (end_idx - i))
                
                del l_batch, p_batch, pred_posteriors_batch, log_pred_batch, batch_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            logits_baseline_optimizer.step()
            train_loss_logits = sum(logits_train_losses) / len(logits_train_losses)
            
            # Validation phase
            logits_baseline_model.eval()
            logits_val_losses = []
            with torch.no_grad():
                for i in range(0, len(val_indices), val_batch_size):
                    end_idx = min(i + val_batch_size, len(val_indices))
                    l_batch = logits_val[i:end_idx].to(device)
                    p_batch = posteriors_val[i:end_idx].to(device)
                    
                    pred_posteriors_batch = logits_baseline_model(l_batch)
                    log_pred_batch = torch.log(pred_posteriors_batch + 1e-10)
                    batch_loss = criterion(log_pred_batch, p_batch)
                    logits_val_losses.append(batch_loss.item())
                    
                    del l_batch, p_batch, pred_posteriors_batch, log_pred_batch, batch_loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            val_loss_logits = sum(logits_val_losses) / len(logits_val_losses)
            
            logits_baseline_loss_history.append(train_loss_logits)
            logits_baseline_val_loss_history.append(val_loss_logits)
            
            if verbose and (epoch + 1) % 20 == 0:
                logger.info(f"  Logits Baseline Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_logits:.6f}, Val Loss: {val_loss_logits:.6f}")
        
        final_logits_baseline_loss = logits_baseline_loss_history[-1]
        final_logits_baseline_val_loss = logits_baseline_val_loss_history[-1]
        
        if verbose:
            logger.info(f"Logits baseline completed. Final train loss: {final_logits_baseline_loss:.6f}, Final val loss: {final_logits_baseline_val_loss:.6f}")
            logger.info(f"Comparison - Hiddens vs Logits - Train: {final_loss - final_logits_baseline_loss:.6f}, Val: {final_val_loss - final_logits_baseline_val_loss:.6f}")
    
    # Move models back to CPU
    linear_model = linear_model.cpu()
    if baseline_model is not None:
        baseline_model = baseline_model.cpu()
    if logits_baseline_model is not None:
        logits_baseline_model = logits_baseline_model.cpu()
    
    return {
        'final_loss': final_loss,
        'final_val_loss': final_val_loss,
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'model': linear_model,
        'baseline_final_loss': final_baseline_loss,
        'baseline_final_val_loss': final_baseline_val_loss,
        'baseline_loss_history': baseline_loss_history,
        'baseline_val_loss_history': baseline_val_loss_history,
        'baseline_model': baseline_model,
        'logits_baseline_final_loss': final_logits_baseline_loss,
        'logits_baseline_final_val_loss': final_logits_baseline_val_loss,
        'logits_baseline_loss_history': logits_baseline_loss_history,
        'logits_baseline_val_loss_history': logits_baseline_val_loss_history,
        'logits_baseline_model': logits_baseline_model,
        'layer': layer,
        'n_tasks': n_total_tasks,
        'hidden_dim': D,
        'vocab_size': vocab_size,
        'n_samples': n_total,
        'n_train': n_train,
        'n_val': len(val_indices),
    }


def train_linear_softmax_posterior_predictor_dyck(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples: int = 1000,
    step: Optional[int] = None,
    n_minor: Optional[int] = None,
    n_ood: int = 0,
    learning_rate: float = 0.01,
    num_epochs: int = 100,
    verbose: bool = False,
    dyck_mask: Optional[torch.Tensor] = None,
    n_masks: int = 1,
    min_dyck_positions: int = 0,
    max_dyck_positions: Optional[int] = None,
    validation_split: float = 0.2,
    uniform_sampling: bool = True,
    skip_baselines: bool = False,
) -> dict:
    """
    Train a linear softmax model to predict task posteriors from hidden representations.

    For Dyck task:
    1. Generates padded Dyck samples using one or more dyck_masks
    2. Computes task posteriors using dyck_task_posterior_over_time_nonpadded
    3. For each Dyck token position t_dyck (in the padded sequence):
       - Extracts the hidden state at t_pad = t_dyck + 1 (the pad slot right after)
       - Uses the posterior at real-token index t_dyck // 2 as the training target
    4. Trains a linear + softmax model to map hidden states to posteriors

    Parameters
    ----------
    exp_name : str
        Experiment name (e.g., "train_...")
    layer : int
        Layer index to extract hidden representations from
    B : int, default=64
        Batch size for sampling
    n_samples : int, default=1000
        Total number of sequences to generate for training (split across masks)
    step : int, optional
        Checkpoint step to load. If None, uses the final checkpoint.
    n_minor : int, optional
        Number of minor tasks to use. If None, uses all available minor tasks.
        If -1, uses no minor tasks (only OOD if n_ood > 0).
    n_ood : int, default=0
        Number of OOD tasks to include
    learning_rate : float, default=0.01
        Learning rate for training the linear model
    num_epochs : int, default=100
        Number of training epochs
    verbose : bool, default=False
        Whether to print progress messages
    dyck_mask : torch.Tensor, optional
        1D binary mask of length (seq_len+1)//2 indicating which real-token positions
        receive Dyck tokens. If provided, only this single mask is used and n_masks
        is ignored. If None, n_masks independent masks are sampled via
        sample_binary_mask.
    n_masks : int, default=1
        Number of independently-sampled dyck_masks to use for data collection.
        Samples are distributed evenly across masks. Ignored when dyck_mask is
        provided explicitly.
    min_dyck_positions : int, default=0
        Number of leading Dyck-token positions to skip (per mask). Useful for
        discarding early positions where the posterior is still near-uniform and
        uninformative. 0-indexed: ``min_dyck_positions=5`` skips positions 0-4.
    max_dyck_positions : int, optional
        If set, only Dyck-token positions up to index ``max_dyck_positions``
        (exclusive, per mask) are used. Combined with ``min_dyck_positions``,
        the positions used are ``[min_dyck_positions : max_dyck_positions]``.
        If None, all positions from ``min_dyck_positions`` onward are used.
    validation_split : float, default=0.2
        Fraction of data to use for validation.
    uniform_sampling : bool, default=True
        If True, modifies p_minor to achieve uniform sampling across all tasks.
    skip_baselines : bool, default=False
        If True, skips training permutation and logits baselines to save time.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'final_loss': float, final training loss
        - 'final_val_loss': float, final validation loss
        - 'loss_history': list of training losses
        - 'val_loss_history': list of validation losses
        - 'model': the trained linear model
        - 'baseline_final_loss': float, final training loss for permutation baseline
        - 'baseline_final_val_loss': float, final validation loss for permutation baseline
        - 'baseline_loss_history': list of training losses for baseline
        - 'baseline_val_loss_history': list of validation losses for baseline
        - 'baseline_model': the trained baseline model
        - 'logits_baseline_final_loss': float, final training loss for logits baseline
        - 'logits_baseline_final_val_loss': float, final validation loss for logits baseline
        - 'logits_baseline_loss_history': list of training losses for logits baseline
        - 'logits_baseline_val_loss_history': list of validation losses for logits baseline
        - 'logits_baseline_model': the trained logits baseline model
        - 'layer': int, the layer index used
        - 'n_tasks': int, number of tasks
        - 'dyck_masks': list of dyck_masks used
    """
    from icl.dyck.dyck import dyck_task_posterior_over_time_nonpadded
    import torch.optim as optim

    # Load config / model / sampler
    _, _sampler_orig, config = nu.load_everything("dyck", exp_name)

    if step is None:
        step = config.training.num_epochs

    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval()
    model.to(config.device)

    # Get sampler with specified n_minor and n_ood
    if n_minor is None:
        n_minor = 1000000  # Use all available
    elif n_minor == -1:
        n_minor = 0

    sampler, k_minor = get_dyck_sampler(exp_name, n_minor, n_ood)
    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks

    if not getattr(sampler, "pad", False):
        raise ValueError("This function requires padded Dyck sequences (sampler.pad must be True)")

    device = config.device

    # Build list of masks
    if dyck_mask is not None:
        masks_list = [dyck_mask.to(device)]
    else:
        assert n_masks >= 1, f"n_masks must be >= 1, got {n_masks}"
        masks_list = [sample_binary_mask(config).to(device) for _ in range(n_masks)]

    # Optionally modify p_minor to achieve uniform sampling across all tasks
    original_p_minor = sampler.p_minor
    if uniform_sampling and sampler.n_minor_tasks > 0:
        sampler.p_minor = sampler.n_minor_tasks / (sampler.n_major_tasks + sampler.n_minor_tasks)
        if verbose:
            logger.info(f"Modified p_minor from {original_p_minor:.6f} to {sampler.p_minor:.6f} for uniform task sampling")
    elif verbose and not uniform_sampling:
        logger.info(f"Using original p_minor: {original_p_minor:.6f} (not modified for uniform sampling)")

    if verbose:
        logger.info(f"Training linear softmax model to predict Dyck posteriors from layer {layer} hidden representations")
        logger.info(f"Number of tasks: {n_tasks} (major: {sampler.n_major_tasks}, minor: {sampler.n_minor_tasks}), Batch size: {B}, Total samples: {n_samples}")
        logger.info(f"Using {len(masks_list)} mask(s) for data collection")

    # -----------------------------------------------------------
    # Collect hiddens, logits, and posteriors across masks & batches
    # -----------------------------------------------------------
    # Distribute samples evenly across masks
    num_masks = len(masks_list)
    n_samples_per_mask = (n_samples + num_masks - 1) // num_masks

    all_hiddens = []
    all_logits = []
    all_posteriors = []

    seq_len_padded = sampler.seq_len  # length of the padded sequence

    for mask_idx, current_mask in enumerate(masks_list):
        # Determine positions for this mask
        dyck_real_positions = torch.nonzero(current_mask == 1, as_tuple=True)[0]
        padded_positions_raw = (2 * dyck_real_positions + 1)

        # Filter out positions where the pad slot falls outside the sequence.
        # When seq_len is odd, the last real token sits at index seq_len-1 and
        # there is no pad slot after it (2*j+1 == seq_len, which is OOB).
        valid = padded_positions_raw < seq_len_padded
        padded_positions = padded_positions_raw[valid].to(device=device, dtype=torch.long)
        posterior_indices = dyck_real_positions[valid].to(device=device, dtype=torch.long)

        # Slice to the requested range [min_dyck_positions : max_dyck_positions]
        start = min_dyck_positions
        end = max_dyck_positions  # None means no upper bound
        padded_positions = padded_positions[start:end]
        posterior_indices = posterior_indices[start:end]

        n_positions = len(padded_positions)

        if n_positions == 0:
            if verbose:
                logger.info(f"  Mask {mask_idx+1}/{num_masks}: 0 valid Dyck positions — skipping")
            continue

        if verbose:
            logger.info(f"  Mask {mask_idx+1}/{num_masks}: {n_positions} Dyck positions, "
                        f"real-token indices {posterior_indices.tolist()}")

        n_batches = (n_samples_per_mask + B - 1) // B

        for batch_idx in range(n_batches):
            # Generate padded samples with the current mask
            # train mode returns (epochs, B, L)
            samples_raw, masks_raw = sampler.generate(
                mode="train", task=None, num_samples=B, epochs=1,
                dyck_mask=current_mask.clone(),
            )
            # Reshape from (1, B, L) to (B, L)
            if samples_raw.dim() == 3:
                samples_raw = samples_raw.squeeze(0)
                masks_raw = masks_raw.squeeze(0)

            # Compute task posteriors on CPU (no model needed, saves GPU memory)
            posteriors = dyck_task_posterior_over_time_nonpadded(
                sampler, samples_raw, masks_raw
            )  # (B, L_real, T) on CPU
            posteriors_batch = posteriors[:, posterior_indices.cpu(), :]
            del posteriors

            # Model forward pass on GPU
            samples = samples_raw.to(device)
            del samples_raw, masks_raw

            cache = {}
            layer_module = model.layers[layer].attn_block

            def hook_fn(module, inp, out, _pp=padded_positions):
                if torch.is_tensor(out):
                    cache["hidden"] = out.index_select(dim=1, index=_pp).detach()
                elif isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                    cache["hidden"] = out[0].index_select(dim=1, index=_pp).detach()
                else:
                    raise RuntimeError(f"Unsupported hook output type: {type(out)}")

            handle = layer_module.register_forward_hook(hook_fn)
            try:
                with torch.no_grad():
                    logits_full = model(samples)  # (B, L, vocab_size)
                    logits_batch = logits_full.index_select(dim=1, index=padded_positions).cpu()
                    del logits_full
                hiddens_batch = cache["hidden"].cpu()
            finally:
                handle.remove()
                cache.clear()

            del samples
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Flatten (B, n_positions, *) -> (B * n_positions, *) before appending
            bsz = hiddens_batch.shape[0]
            all_hiddens.append(hiddens_batch.reshape(bsz * n_positions, -1))
            all_logits.append(logits_batch.reshape(bsz * n_positions, -1))
            all_posteriors.append(posteriors_batch.reshape(bsz * n_positions, -1))
            del hiddens_batch, logits_batch, posteriors_batch

    # -----------------------------------------------------------
    # Free the model from GPU — no longer needed for data collection
    # -----------------------------------------------------------
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------
    # Concatenate all (hidden, posterior) pairs across masks & batches
    # -----------------------------------------------------------
    hiddens_flat = torch.cat(all_hiddens, dim=0)       # (N_total, D)
    logits_flat = torch.cat(all_logits, dim=0)          # (N_total, vocab_size)
    posteriors_flat = torch.cat(all_posteriors, dim=0)   # (N_total, T)
    del all_hiddens, all_logits, all_posteriors

    n_total = hiddens_flat.shape[0]
    D = hiddens_flat.shape[1]
    vocab_size = logits_flat.shape[1]
    T = posteriors_flat.shape[1]

    # -----------------------------------------------------------
    # Train / validation split (data stays on CPU)
    # -----------------------------------------------------------
    n_train = int(n_total * (1 - validation_split))
    indices = torch.randperm(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    hiddens_train = hiddens_flat[train_indices]
    logits_train = logits_flat[train_indices]
    posteriors_train = posteriors_flat[train_indices]
    hiddens_val = hiddens_flat[val_indices]
    logits_val = logits_flat[val_indices]
    posteriors_val = posteriors_flat[val_indices]
    del hiddens_flat, logits_flat, posteriors_flat, indices

    if verbose:
        logger.info(f"Total data: {n_total} samples")
        logger.info(f"Training data: {n_train} samples ({100*(1-validation_split):.1f}%)")
        logger.info(f"Validation data: {len(val_indices)} samples ({100*validation_split:.1f}%)")
        logger.info(f"Training data shape: hiddens {hiddens_train.shape}, logits {logits_train.shape}, posteriors {posteriors_train.shape}")

    # -----------------------------------------------------------
    # Create and train the linear softmax model (mini-batch SGD)
    # -----------------------------------------------------------
    linear_model = nn.Sequential(
        nn.Linear(D, T, bias=True),
        nn.Softmax(dim=-1)
    ).to(device)

    # Training setup
    optimizer = optim.Adam(linear_model.parameters(), lr=learning_rate)
    criterion = nn.KLDivLoss(reduction='batchmean')

    # Mini-batch size for training (data stays on CPU, chunks moved to GPU)
    train_batch_size = min(2048, n_train)

    loss_history = []
    val_loss_history = []

    if verbose:
        logger.info(f"Training linear model for {num_epochs} epochs (mini-batch size {train_batch_size})...")

    for epoch in range(num_epochs):
        # Training phase: mini-batch SGD
        linear_model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_chunks = 0

        for start in range(0, n_train, train_batch_size):
            end = min(start + train_batch_size, n_train)
            idx = perm[start:end]
            h_batch = hiddens_train[idx].to(device)
            p_batch = posteriors_train[idx].to(device)

            optimizer.zero_grad()
            pred = linear_model(h_batch)
            log_pred = torch.log(pred + 1e-10)
            loss = criterion(log_pred, p_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * (end - start)
            n_chunks += 1
            del h_batch, p_batch, pred, log_pred, loss

        train_loss = epoch_loss / n_train

        # Validation phase: chunked to avoid OOM
        linear_model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for start in range(0, len(val_indices), train_batch_size):
                end = min(start + train_batch_size, len(val_indices))
                h_val = hiddens_val[start:end].to(device)
                p_val = posteriors_val[start:end].to(device)
                pred_val = linear_model(h_val)
                log_pred_val = torch.log(pred_val + 1e-10)
                val_loss_sum += criterion(log_pred_val, p_val).item() * (end - start)
                del h_val, p_val, pred_val, log_pred_val

        val_loss = val_loss_sum / len(val_indices)

        loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if verbose and (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    final_loss = loss_history[-1]
    final_val_loss = val_loss_history[-1]

    if verbose:
        logger.info(f"Training completed. Final train loss: {final_loss:.6f}, Final val loss: {final_val_loss:.6f}")

    # -----------------------------------------------------------
    # Baselines
    # -----------------------------------------------------------
    final_baseline_loss = float('nan')
    final_baseline_val_loss = float('nan')
    baseline_loss_history = []
    baseline_val_loss_history = []
    baseline_model = None

    final_logits_baseline_loss = float('nan')
    final_logits_baseline_val_loss = float('nan')
    logits_baseline_loss_history = []
    logits_baseline_val_loss_history = []
    logits_baseline_model = None

    if not skip_baselines:
        # ---- Permutation baseline: shuffle posteriors ----
        if verbose:
            logger.info("Training permutation baseline (shuffled posteriors)...")

        posteriors_train_shuffled = posteriors_train[torch.randperm(n_train)]
        posteriors_val_shuffled = posteriors_val[torch.randperm(len(val_indices))]

        baseline_model = nn.Sequential(
            nn.Linear(D, T, bias=True),
            nn.Softmax(dim=-1)
        ).to(device)

        baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=learning_rate)
        baseline_loss_history = []
        baseline_val_loss_history = []

        for epoch in range(num_epochs):
            baseline_model.train()
            perm = torch.randperm(n_train)
            epoch_loss_bl = 0.0
            for start in range(0, n_train, train_batch_size):
                end = min(start + train_batch_size, n_train)
                idx = perm[start:end]
                h_b = hiddens_train[idx].to(device)
                p_b = posteriors_train_shuffled[idx].to(device)
                baseline_optimizer.zero_grad()
                pred_bl = baseline_model(h_b)
                log_pred_bl = torch.log(pred_bl + 1e-10)
                loss_bl = criterion(log_pred_bl, p_b)
                loss_bl.backward()
                baseline_optimizer.step()
                epoch_loss_bl += loss_bl.item() * (end - start)
                del h_b, p_b, pred_bl, log_pred_bl, loss_bl

            baseline_model.eval()
            val_loss_bl_sum = 0.0
            with torch.no_grad():
                for start in range(0, len(val_indices), train_batch_size):
                    end = min(start + train_batch_size, len(val_indices))
                    h_v = hiddens_val[start:end].to(device)
                    p_v = posteriors_val_shuffled[start:end].to(device)
                    pred_v = baseline_model(h_v)
                    log_pred_v = torch.log(pred_v + 1e-10)
                    val_loss_bl_sum += criterion(log_pred_v, p_v).item() * (end - start)
                    del h_v, p_v, pred_v, log_pred_v

            baseline_loss_history.append(epoch_loss_bl / n_train)
            baseline_val_loss_history.append(val_loss_bl_sum / len(val_indices))

            if verbose and (epoch + 1) % 20 == 0:
                logger.info(f"  Baseline Epoch {epoch+1}/{num_epochs}, Train Loss: {baseline_loss_history[-1]:.6f}, Val Loss: {baseline_val_loss_history[-1]:.6f}")

        final_baseline_loss = baseline_loss_history[-1]
        final_baseline_val_loss = baseline_val_loss_history[-1]

        if verbose:
            logger.info(f"Baseline completed. Final train loss: {final_baseline_loss:.6f}, Final val loss: {final_baseline_val_loss:.6f}")
            logger.info(f"Improvement over baseline - Train: {final_loss - final_baseline_loss:.6f}, Val: {final_val_loss - final_baseline_val_loss:.6f}")

        del posteriors_train_shuffled, posteriors_val_shuffled

        # ---- Logits baseline: predict posteriors from logits ----
        if verbose:
            logger.info("Training logits baseline (predicting posteriors from logits)...")

        logits_baseline_model = nn.Sequential(
            nn.Linear(vocab_size, T, bias=True),
            nn.Softmax(dim=-1)
        ).to(device)

        logits_baseline_optimizer = optim.Adam(logits_baseline_model.parameters(), lr=learning_rate)
        logits_baseline_loss_history = []
        logits_baseline_val_loss_history = []

        for epoch in range(num_epochs):
            logits_baseline_model.train()
            perm = torch.randperm(n_train)
            epoch_loss_lg = 0.0
            for start in range(0, n_train, train_batch_size):
                end = min(start + train_batch_size, n_train)
                idx = perm[start:end]
                l_b = logits_train[idx].to(device)
                p_b = posteriors_train[idx].to(device)
                logits_baseline_optimizer.zero_grad()
                pred_lg = logits_baseline_model(l_b)
                log_pred_lg = torch.log(pred_lg + 1e-10)
                loss_lg = criterion(log_pred_lg, p_b)
                loss_lg.backward()
                logits_baseline_optimizer.step()
                epoch_loss_lg += loss_lg.item() * (end - start)
                del l_b, p_b, pred_lg, log_pred_lg, loss_lg

            logits_baseline_model.eval()
            val_loss_lg_sum = 0.0
            with torch.no_grad():
                for start in range(0, len(val_indices), train_batch_size):
                    end = min(start + train_batch_size, len(val_indices))
                    l_v = logits_val[start:end].to(device)
                    p_v = posteriors_val[start:end].to(device)
                    pred_v = logits_baseline_model(l_v)
                    log_pred_v = torch.log(pred_v + 1e-10)
                    val_loss_lg_sum += criterion(log_pred_v, p_v).item() * (end - start)
                    del l_v, p_v, pred_v, log_pred_v

            logits_baseline_loss_history.append(epoch_loss_lg / n_train)
            logits_baseline_val_loss_history.append(val_loss_lg_sum / len(val_indices))

            if verbose and (epoch + 1) % 20 == 0:
                logger.info(f"  Logits Baseline Epoch {epoch+1}/{num_epochs}, Train Loss: {logits_baseline_loss_history[-1]:.6f}, Val Loss: {logits_baseline_val_loss_history[-1]:.6f}")

        final_logits_baseline_loss = logits_baseline_loss_history[-1]
        final_logits_baseline_val_loss = logits_baseline_val_loss_history[-1]

        if verbose:
            logger.info(f"Logits baseline completed. Final train loss: {final_logits_baseline_loss:.6f}, Final val loss: {final_logits_baseline_val_loss:.6f}")
            logger.info(f"Comparison - Hiddens vs Logits - Train: {final_loss - final_logits_baseline_loss:.6f}, Val: {final_val_loss - final_logits_baseline_val_loss:.6f}")

    # Restore original p_minor
    if hasattr(sampler, 'p_minor') and 'original_p_minor' in locals():
        sampler.p_minor = original_p_minor

    # Move models back to CPU
    linear_model = linear_model.cpu()
    if baseline_model is not None:
        baseline_model = baseline_model.cpu()
    if logits_baseline_model is not None:
        logits_baseline_model = logits_baseline_model.cpu()

    return {
        'final_loss': final_loss,
        'final_val_loss': final_val_loss,
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'model': linear_model,
        'baseline_final_loss': final_baseline_loss,
        'baseline_final_val_loss': final_baseline_val_loss,
        'baseline_loss_history': baseline_loss_history,
        'baseline_val_loss_history': baseline_val_loss_history,
        'baseline_model': baseline_model,
        'logits_baseline_final_loss': final_logits_baseline_loss,
        'logits_baseline_final_val_loss': final_logits_baseline_val_loss,
        'logits_baseline_loss_history': logits_baseline_loss_history,
        'logits_baseline_val_loss_history': logits_baseline_val_loss_history,
        'logits_baseline_model': logits_baseline_model,
        'layer': layer,
        'n_tasks': n_tasks,
        'hidden_dim': D,
        'vocab_size': vocab_size,
        'n_samples': n_total,
        'n_train': n_train,
        'n_val': len(val_indices),
        'dyck_masks': [m.cpu() for m in masks_list],
    }


@torch.no_grad()
def compute_p1_variance_dyck(
    exp_name: str,
    layers: Optional[Sequence[int]] = None,
    B: int = 64,
    n_masks: int = 30,
    step: Optional[int] = None,
    n_minor: int = 32,
    n_ood: int = 0,
    verbose: bool = False,
) -> dict:
    """
    Compute Var(H | dyck_prefix, task) at each Dyck position index for one or
    more layers of the Dyck task.

    For each task k and Dyck position index j, the Dyck prefix steps[0:j+1] is
    deterministic. The variance captures residual variability from random Markov
    noise tokens and mask placement across multiple masks.

    Parameters
    ----------
    exp_name : str
        Experiment name (folder under results/dyck/).
    layers : list of int, optional
        Layer indices to extract hidden representations from. If None, uses all
        layers. A single int is also accepted.
    B : int, default=64
        Batch size per (task, mask) forward pass.
    n_masks : int, default=30
        Number of independently-sampled masks for position diversity.
    step : int, optional
        Checkpoint step. If None, uses the final checkpoint.
    n_minor : int, default=32
        Number of minor tasks to subsample. Use -1 for none, or a large number
        (e.g. 1000000) for all available.
    n_ood : int, default=0
        Number of OOD tasks.
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    dict with keys:
        - 'var_pos': dict mapping layer_idx -> list of floats (variance per Dyck position)
        - 'var_pos_per_task': dict mapping layer_idx -> (n_tasks, n_dyck) tensor
        - 'var_pos_norm': dict mapping layer_idx -> list of floats (normalized variance)
        - 'n_tasks': int
        - 'n_dyck_positions': int
        - 'layers': list of int — layer indices used
        - 'n_masks': int
        - 'B': int
        - 'samples_per_task': int
    """
    import gc

    _, _sampler_orig, config = nu.load_everything("dyck", exp_name)

    if step is None:
        step = config.training.num_epochs

    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval()
    device = config.device
    model.to(device)

    n_layers = len(model.layers)

    # Normalize layers argument
    if layers is None:
        layers = list(range(n_layers))
    elif isinstance(layers, int):
        layers = [layers]
    else:
        layers = list(layers)

    n_minor_val = n_minor if n_minor is not None else 1000000
    if n_minor_val == -1:
        n_minor_val = 0
    sampler, _ = get_dyck_sampler(exp_name, n_minor_val, n_ood)

    if not getattr(sampler, "pad", False):
        raise ValueError("Requires padded Dyck sequences")

    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    seq_len_padded = sampler.seq_len
    d_model = config.model.emb_dim

    # Sample masks and compute valid Dyck positions for each
    masks_list = [sample_binary_mask(config).to(device) for _ in range(n_masks)]

    masks_info = []
    for mask in masks_list:
        dyck_real = torch.nonzero(mask == 1, as_tuple=True)[0]
        pp_raw = 2 * dyck_real + 1
        valid = pp_raw < seq_len_padded
        pp = pp_raw[valid].to(device=device, dtype=torch.long)
        if len(pp) > 0:
            masks_info.append({'mask': mask, 'padded_positions': pp, 'n_pos': len(pp)})

    if not masks_info:
        raise RuntimeError("No masks produced valid Dyck positions.")

    min_n_pos = min(info['n_pos'] for info in masks_info)

    if verbose:
        logger.info(f"Dyck P1 variance: {n_tasks} tasks, {len(masks_info)} masks, "
                     f"{min_n_pos} Dyck positions, layers {layers}, B={B}")

    # Register hooks on all requested layers
    caches = {}

    def make_hook(li):
        def hook_fn(module, inp, out):
            if torch.is_tensor(out):
                caches[li] = out.detach()
            elif isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                caches[li] = out[0].detach()
        return hook_fn

    handles = []
    for li in layers:
        handles.append(model.layers[li].attn_block.register_forward_hook(make_hook(li)))

    # Accumulate running statistics per (layer, task, Dyck position)
    sum_h = {li: torch.zeros((n_tasks, min_n_pos, d_model), dtype=torch.float64) for li in layers}
    sum_h2 = {li: torch.zeros((n_tasks, min_n_pos), dtype=torch.float64) for li in layers}
    count = torch.zeros((n_tasks,), dtype=torch.int64)

    try:
        for task_idx in range(n_tasks):
            for m_info in masks_info:
                mask = m_info['mask']
                pp = m_info['padded_positions'][:min_n_pos]

                demo_data, _ = sampler.generate(
                    mode="testing", task=task_idx, num_samples=B,
                    dyck_mask=mask.clone(),
                )
                demo_data = demo_data.to(device)

                caches.clear()
                _ = model(demo_data)

                for li in layers:
                    h = caches[li].index_select(dim=1, index=pp).cpu().to(torch.float64)
                    sum_h[li][task_idx] += h.sum(dim=0)
                    sum_h2[li][task_idx] += (h ** 2).sum(dim=-1).sum(dim=0)

                count[task_idx] += B

                del demo_data
                caches.clear()

            if verbose and (task_idx == 0 or (task_idx + 1) % 50 == 0 or task_idx == n_tasks - 1):
                logger.info(f"  Task {task_idx + 1}/{n_tasks} done")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        for h in handles:
            h.remove()

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Compute variance per (layer, task, Dyck position)
    eps = 1e-8
    n = count.unsqueeze(1).to(torch.float64)  # (n_tasks, 1)

    var_pos_dict = {}
    var_per_task_dict = {}
    var_pos_norm_dict = {}
    mean_norm_sq_dict = {}

    for li in layers:
        mean_h = sum_h[li] / n.unsqueeze(2)
        mean_h2 = sum_h2[li] / n
        mean_norm_sq = (mean_h ** 2).sum(dim=-1)

        var_per_task = (mean_h2 - mean_norm_sq).clamp(min=0.0)
        var_pos = var_per_task.mean(dim=0)

        norm_per_task = var_per_task / (mean_norm_sq + eps)
        var_pos_norm = norm_per_task.mean(dim=0)

        var_pos_dict[li] = var_pos.tolist()
        var_per_task_dict[li] = var_per_task
        var_pos_norm_dict[li] = var_pos_norm.tolist()
        mean_norm_sq_dict[li] = mean_norm_sq

    return {
        'layers': layers,
        'positions': list(range(min_n_pos)),
        'var_pos': var_pos_dict,
        'var_pos_per_task': var_per_task_dict,
        'var_pos_norm': var_pos_norm_dict,
        'mean_norm_sq_per_task': mean_norm_sq_dict,
        'n_tasks': n_tasks,
        'n_dyck_positions': min_n_pos,
        'n_masks': len(masks_info),
        'B': B,
        'samples_per_task': int(count[0].item()),
    }


def plot_dyck_task_posterior(
    exp_name: str,
    n_plots: int = 3,
    dyck_mask: Optional[torch.Tensor] = None,
    uniform_prior: bool = True,
    max_positions: Optional[int] = None,
    figsize: tuple = (12, 4),
    title: Optional[str] = None,
) -> dict:
    """
    Generate random Dyck samples and plot the task posterior over Dyck-token positions.

    Parameters
    ----------
    exp_name : str
        Experiment name (folder under results/dyck/).
    n_plots : int, default=3
        Number of independent samples to plot (one subplot each).
    dyck_mask : torch.Tensor, optional
        1D binary mask. If None, one is sampled via sample_binary_mask.
    uniform_prior : bool, default=True
        If True, sets p_minor for uniform task prior.
    max_positions : int, optional
        Maximum number of Dyck positions to show per subplot.
    figsize : tuple, default=(12, 4)
        Figure size per subplot row (total height = figsize[1] * n_plots).
    title : str, optional
        Custom suptitle. If None, a default is generated.

    Returns
    -------
    info : dict
        - 'posteriors': list of (n_dyck, T) tensors, one per sample
        - 'dyck_mask': the mask used
        - 'fig': matplotlib Figure
        - 'axes': list of matplotlib Axes
    """
    import matplotlib.pyplot as plt
    from icl.dyck.dyck import dyck_task_posterior_over_time_nonpadded

    _, sampler, config = nu.load_everything("dyck", exp_name)
    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    device = config.device

    if not getattr(sampler, "pad", False):
        raise ValueError("This function requires padded Dyck sequences (sampler.pad must be True)")

    original_p_minor = getattr(sampler, 'p_minor', 0.0)
    if uniform_prior and sampler.n_minor_tasks > 0:
        sampler.p_minor = sampler.n_minor_tasks / (sampler.n_major_tasks + sampler.n_minor_tasks)

    if dyck_mask is None:
        dyck_mask = sample_binary_mask(config).to(device)
    else:
        dyck_mask = dyck_mask.to(device)

    # Generate n_plots samples
    samples_raw, masks_raw = sampler.generate(
        mode="train", task=None, num_samples=n_plots, epochs=1,
        dyck_mask=dyck_mask.clone(),
    )
    if samples_raw.dim() == 3:
        samples_raw = samples_raw.squeeze(0)
        masks_raw = masks_raw.squeeze(0)
    samples = samples_raw.to(device)
    masks = masks_raw.to(device)

    posterior_all = dyck_task_posterior_over_time_nonpadded(sampler, samples, masks)

    dyck_positions = torch.nonzero(dyck_mask == 1, as_tuple=True)[0].cpu()

    n_major = sampler.n_major_tasks
    n_minor_tasks = sampler.n_minor_tasks
    major_cmap = plt.cm.Blues
    minor_cmap = plt.cm.Reds
    major_colors = [major_cmap(0.3 + 0.6 * i / max(n_major - 1, 1)) for i in range(n_major)]
    minor_colors = [minor_cmap(0.3 + 0.6 * i / max(n_minor_tasks - 1, 1)) for i in range(n_minor_tasks)]
    T = n_tasks

    fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], figsize[1] * n_plots), squeeze=False)
    axes = [axes[i, 0] for i in range(n_plots)]
    posteriors_out = []

    for idx, ax in enumerate(axes):
        posterior = posterior_all[idx].cpu()
        posterior_dyck = posterior[dyck_positions, :]
        if max_positions is not None:
            posterior_dyck = posterior_dyck[:max_positions]
        n_dyck = len(posterior_dyck)
        posteriors_out.append(posterior_dyck)

        x_axis = torch.arange(n_dyck)
        major_labeled = False
        minor_labeled = False
        for k in range(T):
            if k < n_major:
                color = major_colors[k]
                label = ("Major" if not major_labeled else None) if idx == 0 else None
                major_labeled = True
            else:
                color = minor_colors[k - n_major]
                label = ("Minor" if not minor_labeled else None) if idx == 0 else None
                minor_labeled = True
            ax.plot(x_axis.numpy(), posterior_dyck[:, k].numpy(), label=label,
                    color=color, alpha=0.8, linewidth=1.5)

        ax.set_ylabel("P(Z=k | obs)")
        ax.set_xlim(0, max(n_dyck - 1, 1))
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Sample {idx + 1}", fontsize=10)
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=max(1, T // 20))

    axes[-1].set_xlabel("Dyck token index")
    if title is None:
        title = f"Dyck task posterior over time — {exp_name}"
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    plt.show()

    sampler.p_minor = original_p_minor

    return {
        'posteriors': posteriors_out,
        'dyck_mask': dyck_mask.cpu(),
        'fig': fig,
        'axes': axes,
    }


def plot_coin_task_posterior(
    exp_name: str,
    n_plots: int = 3,
    uniform_prior: bool = True,
    max_positions: Optional[int] = None,
    figsize: tuple = (12, 4),
    title: Optional[str] = None,
) -> dict:
    """
    Generate random coin samples and plot the task posterior over real-token positions.

    Parameters
    ----------
    exp_name : str
        Experiment name (folder under results/coin/).
    n_plots : int, default=3
        Number of independent samples to plot (one subplot each).
    uniform_prior : bool, default=True
        If True, sets p_minor for uniform task prior.
    max_positions : int, optional
        Maximum number of real-token positions to show per subplot.
    figsize : tuple, default=(12, 4)
        Figure size per subplot row (total height = figsize[1] * n_plots).
    title : str, optional
        Custom suptitle.

    Returns
    -------
    info : dict
        - 'posteriors': list of (L_real, T) tensors, one per sample
        - 'fig': matplotlib Figure
        - 'axes': list of matplotlib Axes
    """
    import matplotlib.pyplot as plt

    _, sampler, config = nu.load_everything("coin", exp_name)
    device = config.device

    if not getattr(sampler, "pad", False):
        raise ValueError("This function requires padded coin sequences (sampler.pad must be True)")

    original_p_minor = getattr(sampler, 'p_minor', 0.0)
    if uniform_prior and sampler.n_minor_tasks > 0:
        sampler.p_minor = sampler.n_minor_tasks / (sampler.n_major_tasks + sampler.n_minor_tasks)

    # Generate n_plots samples
    samples_raw = sampler.generate(mode="train", num_samples=n_plots, epochs=1)
    if isinstance(samples_raw, tuple):
        samples_raw = samples_raw[0]
    if samples_raw.dim() == 3:
        samples_raw = samples_raw.squeeze(0)
    samples = samples_raw.to(device)

    # Build pools and prior
    K = sampler.num_states
    maj = sampler.major_p.to(device=device, dtype=torch.float32)
    Kmaj = maj.shape[0]
    use_minor = sampler.n_minor_tasks > 0 and sampler.minor_p is not None
    if use_minor:
        minp = sampler.minor_p.to(device=device, dtype=torch.float32)
        Kmin = minp.shape[0]
        P = torch.cat([maj, minp], dim=0)
    else:
        Kmin = 0
        P = maj
    Ktot = P.shape[0]

    eps = 1e-30
    if Kmin == 0:
        prior = torch.full((Ktot,), 1.0 / max(1, Kmaj), device=device, dtype=torch.float32)
    else:
        p0 = float(sampler.p_minor)
        prior_major = (1.0 - p0) / max(1, Kmaj)
        prior_minor = p0 / max(1, Kmin)
        prior = torch.cat([
            torch.full((Kmaj,), prior_major, device=device, dtype=torch.float32),
            torch.full((Kmin,), prior_minor, device=device, dtype=torch.float32),
        ])
    prior = torch.clamp(prior, min=eps)
    log_prior = prior.log()
    logP = torch.log(torch.clamp(P, min=eps))  # (Ktot, K)

    # Get real tokens and compute time-varying posterior via cumulative counts
    x = samples[..., ::2].long()  # (B, L_real)
    B_actual, L_real_full = x.shape
    posterior_time = torch.empty((B_actual, L_real_full, Ktot), device=device, dtype=torch.float32)
    counts = torch.zeros((B_actual, K), device=device, dtype=torch.float32)

    for t in range(L_real_full):
        counts.scatter_add_(
            dim=-1, index=x[:, t:t+1],
            src=torch.ones((B_actual, 1), device=device, dtype=torch.float32),
        )
        loglik = counts @ logP.T
        unnorm = loglik + log_prior.unsqueeze(0)
        log_post = unnorm - torch.logsumexp(unnorm, dim=-1, keepdim=True)
        posterior_time[:, t, :] = torch.exp(log_post)

    n_major = sampler.n_major_tasks
    n_minor_tasks = sampler.n_minor_tasks
    major_cmap = plt.cm.Blues
    minor_cmap = plt.cm.Reds
    major_colors = [major_cmap(0.3 + 0.6 * i / max(n_major - 1, 1)) for i in range(n_major)]
    minor_colors = [minor_cmap(0.3 + 0.6 * i / max(n_minor_tasks - 1, 1)) for i in range(n_minor_tasks)]
    T_plot = Ktot

    fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], figsize[1] * n_plots), squeeze=False)
    axes = [axes[i, 0] for i in range(n_plots)]
    posteriors_out = []

    for idx, ax in enumerate(axes):
        posterior = posterior_time[idx].cpu()
        if max_positions is not None:
            posterior = posterior[:max_positions]
        L_real = posterior.shape[0]
        posteriors_out.append(posterior)

        x_axis = torch.arange(L_real)
        major_labeled = False
        minor_labeled = False
        for k in range(T_plot):
            if k < n_major:
                color = major_colors[k]
                label = ("Major" if not major_labeled else None) if idx == 0 else None
                major_labeled = True
            else:
                color = minor_colors[k - n_major]
                label = ("Minor" if not minor_labeled else None) if idx == 0 else None
                minor_labeled = True
            ax.plot(x_axis.numpy(), posterior[:, k].numpy(), label=label,
                    color=color, alpha=0.8, linewidth=1.5)

        ax.set_ylabel("P(Z=k | obs)")
        ax.set_xlim(0, max(L_real - 1, 1))
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Sample {idx + 1}", fontsize=10)
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=max(1, T_plot // 20))

    axes[-1].set_xlabel("Real-token position")
    if title is None:
        title = f"Coin task posterior over time — {exp_name}"
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    plt.show()

    sampler.p_minor = original_p_minor

    return {
        'posteriors': posteriors_out,
        'fig': fig,
        'axes': axes,
    }


def plot_latent_task_posterior(
    exp_name: str,
    n_plots: int = 3,
    uniform_prior: bool = True,
    max_positions: Optional[int] = None,
    figsize: tuple = (12, 4),
    title: Optional[str] = None,
) -> dict:
    """
    Generate random latent Markov samples and plot the task posterior over real-token positions.

    Parameters
    ----------
    exp_name : str
        Experiment name (folder under results/latent/).
    n_plots : int, default=3
        Number of independent samples to plot (one subplot each).
    uniform_prior : bool, default=True
        If True, sets p_minor for uniform task prior.
    max_positions : int, optional
        Maximum number of real-token positions to show per subplot.
    figsize : tuple, default=(12, 4)
        Figure size per subplot row (total height = figsize[1] * n_plots).
    title : str, optional
        Custom suptitle.

    Returns
    -------
    info : dict
        - 'posteriors': list of (L_real, T) tensors, one per sample
        - 'fig': matplotlib Figure
        - 'axes': list of matplotlib Axes
    """
    import matplotlib.pyplot as plt
    from icl.latent_markov.markov_latent import task_posterior_over_time

    _, sampler, config = nu.load_everything("latent", exp_name)
    T = sampler.total_trans
    device = config.device

    if not getattr(sampler, "pad", False):
        raise ValueError("This function requires padded latent sequences (sampler.pad must be True)")

    original_p_minor = getattr(sampler, 'p_minor', 0.0)
    if uniform_prior and sampler.n_minor_tasks > 0:
        sampler.p_minor = sampler.n_minor_tasks / (sampler.n_major_tasks + sampler.n_minor_tasks)

    # Generate n_plots samples
    samples_raw = sampler.generate(mode="train", num_samples=n_plots, epochs=1)
    if isinstance(samples_raw, tuple):
        samples_raw = samples_raw[0]
    if samples_raw.dim() == 3:
        samples_raw = samples_raw.squeeze(0)
    samples = samples_raw.to(device)

    posterior_all = task_posterior_over_time(sampler, samples)  # (n_plots, L_real, T)

    n_major = sampler.n_major_tasks
    n_minor_tasks = sampler.n_minor_tasks
    major_cmap = plt.cm.Blues
    minor_cmap = plt.cm.Reds
    major_colors = [major_cmap(0.3 + 0.6 * i / max(n_major - 1, 1)) for i in range(n_major)]
    minor_colors = [minor_cmap(0.3 + 0.6 * i / max(n_minor_tasks - 1, 1)) for i in range(n_minor_tasks)]
    T_plot = T

    fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], figsize[1] * n_plots), squeeze=False)
    axes = [axes[i, 0] for i in range(n_plots)]
    posteriors_out = []

    for idx, ax in enumerate(axes):
        posterior = posterior_all[idx].cpu()
        if max_positions is not None:
            posterior = posterior[:max_positions]
        L_real = posterior.shape[0]
        posteriors_out.append(posterior)

        x_axis = torch.arange(L_real)
        major_labeled = False
        minor_labeled = False
        for k in range(T_plot):
            if k < n_major:
                color = major_colors[k]
                label = ("Major" if not major_labeled else None) if idx == 0 else None
                major_labeled = True
            else:
                color = minor_colors[k - n_major]
                label = ("Minor" if not minor_labeled else None) if idx == 0 else None
                minor_labeled = True
            ax.plot(x_axis.numpy(), posterior[:, k].numpy(), label=label,
                    color=color, alpha=0.8, linewidth=1.5)

        ax.set_ylabel("P(Z=k | obs)")
        ax.set_xlim(0, max(L_real - 1, 1))
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Sample {idx + 1}", fontsize=10)
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=max(1, T_plot // 20))

    axes[-1].set_xlabel("Real-token position")
    if title is None:
        title = f"Latent task posterior over time — {exp_name}"
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    plt.show()

    sampler.p_minor = original_p_minor

    return {
        'posteriors': posteriors_out,
        'fig': fig,
        'axes': axes,
    }


def plot_max_stable_rank_all_layers_vs_k(
    task_name: str,
    k_values: list,
    vocab_size: Optional[int] = None,
    B: int = 64,
    step: Optional[int] = None,
    n_ood: int = 0,
    backend: str = "matplotlib",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
    verbose: bool = False,
    chunk_size: int = 2,
    task_chunk_size: Optional[int] = 32,
) -> dict:
    """
    Compute stable ranks for all layers and positions for different numbers of tasks 
    (parameterized by k = log2(number of tasks)), take the maximum over layers, then 
    maximum over positions, and plot against k.
    
    Parameters:
    -----------
    task_name : str
        Task name ("linear", "coin", "latent", "dyck")
    k_values : list
        List of k values where number of tasks = 2^k
    vocab_size : int, optional
        Vocabulary size (for non-linear tasks)
    B : int, default=64
        Batch size for sampling
    step : int, optional
        Step for checkpoint loading. If None, uses final checkpoint.
    n_ood : int, default=0
        Number of OOD tasks to include
    backend : str, default="matplotlib"
        Plotting backend: "matplotlib" or "plotly"
    figsize : tuple, default=(10, 6)
        Figure size for matplotlib (ignored for plotly)
    save_path : str, optional
        Path to save the figure
    show : bool, default=True
        Whether to display the plot
    verbose : bool, default=False
        Whether to print progress messages
    chunk_size : int, default=2
        Number of k values to process in each chunk before clearing memory.
        Smaller values use less memory but may be slower.
    task_chunk_size : int, optional, default=32
        Number of tasks to process at once within each k value computation.
        Smaller values use less memory. If None, processes all tasks at once.
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'k_values': list of k values
        - 'max_stable_ranks': list of maximum stable ranks (max over layers, then max over positions) for each k
        - 'n_tasks': list of number of tasks (2^k) for each k
        - 'fig': matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    max_stable_ranks = []
    n_tasks_list = []
    
    if verbose:
        logger.info(f"Computing stable ranks for all layers for {len(k_values)} different k values")
        logger.info(f"Processing in chunks of {chunk_size} k values")
    
    # Process k values in chunks to reduce memory usage
    import gc
    for chunk_start in range(0, len(k_values), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(k_values))
        k_chunk = k_values[chunk_start:chunk_end]
        
        if verbose:
            logger.info(f"Processing chunk {chunk_start//chunk_size + 1}/{(len(k_values)-1)//chunk_size + 1}: k values {k_chunk}")
        
        for k in k_chunk:
            # Handle k=-1 case (no minor tasks)
            if k == -1:
                n_tasks = 0
                n_minor = -1  # No minor tasks (placeholder)
            else:
                n_tasks = 2 ** k
                n_minor = n_tasks
            
            n_tasks_list.append(n_tasks)
            
            if verbose:
                logger.info(f"Processing k={k} (n_tasks={n_tasks})...")
            
            # Get experiment name for this k
            exp_name = get_exp_name(task_name, k, vocab_size=vocab_size)
            
            results = None
            try:
                results = compute_stable_rank_at_padded_positions(
                    exp_name=exp_name,
                    task_name=task_name,
                    B=B,
                    step=step,
                    n_minor=n_minor,
                    n_ood=n_ood,
                    verbose=verbose,
                    task_chunk_size=task_chunk_size,
                )
                
                # Get stable ranks for all layers: shape (n_layers, n_positions)
                # Move to CPU immediately to free GPU memory
                stable_ranks_all = results['stable_ranks'].cpu()  # (n_layers, n_positions)
                
                # Take maximum over layers: (n_layers, n_positions) -> (n_positions,)
                max_over_layers = stable_ranks_all.max(dim=0)[0]
                
                # Take maximum over positions: (n_positions,) -> scalar
                max_over_positions = max_over_layers.max().item()
                max_stable_ranks.append(max_over_positions)
                
                if verbose:
                    logger.info(f"  k={k}, n_tasks={n_tasks}, max_stable_rank={max_over_positions:.4f}")
            
            finally:
                # Clean up GPU memory after each iteration
                if results is not None:
                    del results
                if 'stable_ranks_all' in locals():
                    del stable_ranks_all
                if 'max_over_layers' in locals():
                    del max_over_layers
                
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
        
        # Extra cleanup between chunks
        if verbose:
            logger.info(f"Completed chunk, clearing memory...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Plot results
    if backend == "plotly" and PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_values,
            y=max_stable_ranks,
            mode='lines+markers',
            name='Max Stable Rank (All Layers)',
            line=dict(width=2, color='purple'),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title=f'Maximum Stable Rank (Max over All Layers & Positions) vs k (log2 of number of tasks)<br>Task: {task_name}',
            xaxis_title='k (log2 of number of tasks)',
            yaxis_title='Maximum Stable Rank (Max over Layers & Positions)',
            hovermode='x unified',
            template='plotly_white',
            width=1000,
            height=600,
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved plot to {save_path}")
        
        if show:
            fig.show()
    
    elif backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(k_values, max_stable_ranks, 'o-', linewidth=2, markersize=8, 
               label='Max Stable Rank (All Layers)', color='purple')
        ax.set_xlabel('k (log2 of number of tasks)', fontsize=12)
        ax.set_ylabel('Maximum Stable Rank (Max over Layers & Positions)', fontsize=12)
        ax.set_title(f'Maximum Stable Rank (Max over All Layers & Positions) vs k\nTask: {task_name}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        if show:
            plt.show()
    
    else:
        if backend == "plotly" and not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available, falling back to matplotlib")
            backend = "matplotlib"
        if backend == "matplotlib" and not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is not installed. Install it with: pip install matplotlib")
        
        # Recursive call with corrected backend
        return plot_max_stable_rank_all_layers_vs_k(
            task_name, k_values, vocab_size, B, step, n_ood, backend, figsize, save_path, show, verbose, chunk_size, task_chunk_size
        )
    
    return {
        'k_values': k_values,
        'max_stable_ranks': max_stable_ranks,
        'n_tasks': n_tasks_list,
        'fig': fig,
    }


def plot_max_stable_rank_logits_vs_k(
    task_name: str,
    k_values: list,
    vocab_size: Optional[int] = None,
    B: int = 64,
    step: Optional[int] = None,
    n_ood: int = 0,
    backend: str = "matplotlib",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Compute stable ranks for logits at different numbers of tasks (parameterized by k = log2(number of tasks))
    and plot the maximum stable rank across all positions against k.
    
    Parameters:
    -----------
    task_name : str
        Task name ("coin", "latent", "dyck"). Note: linear task is not supported (uses regression outputs).
    k_values : list
        List of k values where number of tasks = 2^k
    vocab_size : int, optional
        Vocabulary size (for non-linear tasks)
    B : int, default=64
        Batch size for sampling
    step : int, optional
        Step for checkpoint loading. If None, uses final checkpoint.
    n_ood : int, default=0
        Number of OOD tasks to include
    backend : str, default="matplotlib"
        Plotting backend: "matplotlib" or "plotly"
    figsize : tuple, default=(10, 6)
        Figure size for matplotlib (ignored for plotly)
    save_path : str, optional
        Path to save the figure
    show : bool, default=True
        Whether to display the plot
    verbose : bool, default=False
        Whether to print progress messages
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'k_values': list of k values
        - 'max_stable_ranks': list of maximum stable ranks across all positions for each k
        - 'n_tasks': list of number of tasks (2^k) for each k
        - 'fig': matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    if task_name == "linear":
        raise ValueError(
            "Linear task uses regression outputs, not logits. "
            "Use plot_max_stable_rank_vs_k for hidden representations instead."
        )
    
    max_stable_ranks = []
    n_tasks_list = []
    
    if verbose:
        logger.info(f"Computing stable ranks for logits at {len(k_values)} different k values")
    
    for k in k_values:
        # Handle k=-1 case (no minor tasks)
        if k == -1:
            n_tasks = 0
            n_minor = -1  # No minor tasks (placeholder)
        else:
            n_tasks = 2 ** k
            n_minor = n_tasks
        
        n_tasks_list.append(n_tasks)
        
        if verbose:
            logger.info(f"Processing k={k} (n_tasks={n_tasks})...")
        
        # Get experiment name for this k
        exp_name = get_exp_name(task_name, k, vocab_size=vocab_size)
        
        # Compute stable ranks for logits
        results = compute_stable_rank_logits_at_padded_positions(
            exp_name=exp_name,
            task_name=task_name,
            B=B,
            step=step,
            n_minor=n_minor,
            n_ood=n_ood,
            verbose=verbose,
        )
        
        # Get stable ranks: shape (n_positions,)
        stable_ranks = results['stable_ranks']
        
        # Take maximum stable rank across all positions
        max_stable_rank = stable_ranks.max().item()
        max_stable_ranks.append(max_stable_rank)
        
        if verbose:
            logger.info(f"  k={k}, n_tasks={n_tasks}, max_stable_rank={max_stable_rank:.4f}")
    
    # Plot results
    if backend == "plotly" and PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_values,
            y=max_stable_ranks,
            mode='lines+markers',
            name='Max Stable Rank (Logits)',
            line=dict(width=2, color='red'),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title=f'Maximum Stable Rank (Logits) vs k (log2 of number of tasks)<br>Task: {task_name}',
            xaxis_title='k (log2 of number of tasks)',
            yaxis_title='Maximum Stable Rank (Logits)',
            hovermode='x unified',
            template='plotly_white',
            width=1000,
            height=600,
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved plot to {save_path}")
        
        if show:
            fig.show()
    
    elif backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(k_values, max_stable_ranks, 'o-', linewidth=2, markersize=8, 
               label='Max Stable Rank (Logits)', color='red')
        ax.set_xlabel('k (log2 of number of tasks)', fontsize=12)
        ax.set_ylabel('Maximum Stable Rank (Logits)', fontsize=12)
        ax.set_title(f'Maximum Stable Rank (Logits) vs k\nTask: {task_name}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        if show:
            plt.show()
    
    else:
        if backend == "plotly" and not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available, falling back to matplotlib")
            backend = "matplotlib"
        if backend == "matplotlib" and not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is not installed. Install it with: pip install matplotlib")
        
        # Recursive call with corrected backend
        return plot_max_stable_rank_logits_vs_k(
            task_name, k_values, vocab_size, B, step, n_ood, backend, figsize, save_path, show, verbose
        )
    
    return {
        'k_values': k_values,
        'max_stable_ranks': max_stable_ranks,
        'n_tasks': n_tasks_list,
        'fig': fig,
    }


def plot_stable_rank_final_layer_weight_vs_k(
    task_name: str,
    k_values: list,
    vocab_size: Optional[int] = None,
    step: Optional[int] = None,
    backend: str = "matplotlib",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
    verbose: bool = False,
    layers: Optional[list] = None,
) -> dict:
    """
    Compute stable rank of MLP layer weights for all layers for different numbers of tasks 
    (parameterized by k = log2(number of tasks)) and plot against k.
    
    Parameters:
    -----------
    task_name : str
        Task name ("linear", "coin", "latent", "dyck")
    k_values : list
        List of k values where number of tasks = 2^k
    vocab_size : int, optional
        Vocabulary size (for non-linear tasks)
    step : int, optional
        Step for checkpoint loading. If None, uses final checkpoint.
    backend : str, default="matplotlib"
        Plotting backend: "matplotlib" or "plotly"
    figsize : tuple, default=(10, 6)
        Figure size for matplotlib (ignored for plotly)
    save_path : str, optional
        Path to save the figure
    show : bool, default=True
        Whether to display the plot
    verbose : bool, default=False
        Whether to print progress messages
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'k_values': list of k values
        - 'stable_ranks': dict mapping layer index to list of stable ranks for each k
        - 'n_tasks': list of number of tasks (2^k) for each k
        - 'n_layers': number of layers
        - 'fig': matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    # First, get the number of layers from the first k value
    exp_name_first = get_exp_name(task_name, k_values[0], vocab_size=vocab_size)
    if task_name == "linear":
        _, _, config_first = load_model_task_config(exp_name_first)
        n_layers = config_first.model.n_layer
    else:
        _, _, config_first = nu.load_everything(task_name, exp_name_first)
        n_layers = config_first.model.num_layers
    
    # Determine which layers to process
    if layers is None:
        layers_to_process = list(range(n_layers))
    else:
        layers_to_process = list(layers)
        # Validate layer indices
        if not all(0 <= l < n_layers for l in layers_to_process):
            raise ValueError(f"All layer indices must be in [0, {n_layers-1}], got {layers_to_process}")
    
    # Initialize storage: stable_ranks[layer_idx] = list of stable ranks for each k
    stable_ranks = {layer_idx: [] for layer_idx in layers_to_process}
    n_tasks_list = []
    
    if verbose:
        logger.info(f"Computing stable ranks of layer weights for {len(k_values)} different k values")
        logger.info(f"Number of layers: {n_layers}, Plotting layers: {layers_to_process}")
    
    for k in k_values:
        # Handle k=-1 case (no minor tasks)
        if k == -1:
            n_tasks = 0
        else:
            n_tasks = 2 ** k
        
        n_tasks_list.append(n_tasks)
        
        if verbose:
            logger.info(f"Processing k={k} (n_tasks={n_tasks})...")
        
        # Get experiment name for this k
        exp_name = get_exp_name(task_name, k, vocab_size=vocab_size)
        
        # Load model
        if task_name == "linear":
            _, _, config = load_model_task_config(exp_name)
            if step is None:
                step = config.training.total_steps
            model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
            
            # Extract weights from specified layers
            for layer_idx in layers_to_process:
                # Get MLP output projection weights from each layer
                # Path: transformer.blocks[l].mlp_block.mlp.c_proj.weight (or c_ff.weight if linear activation)
                # Shape: (emb_dim, 4*emb_dim) for c_proj or (emb_dim, emb_dim) for c_ff
                mlp = model.transformer.blocks[layer_idx].mlp_block.mlp
                if hasattr(mlp, 'c_proj'):
                    # Non-linear activation: use output projection
                    layer_weight = mlp.c_proj.weight.detach().cpu()  # (emb_dim, 4*emb_dim)
                elif hasattr(mlp, 'c_ff'):
                    # Linear activation: use single linear layer
                    layer_weight = mlp.c_ff.weight.detach().cpu()  # (emb_dim, emb_dim)
                else:
                    raise ValueError(f"Unexpected MLP structure in layer {layer_idx}")
                weight_matrix = layer_weight
                
                # Compute stable rank
                weight_np = weight_matrix.float().numpy()
                stable_rank_val = stable_rank(weight_np)
                stable_ranks[layer_idx].append(stable_rank_val)
        else:
            _, _, config = nu.load_everything(task_name, exp_name)
            if step is None:
                step = config.training.num_epochs
            model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
            
            # Extract weights from specified layers
            for layer_idx in layers_to_process:
                # Get MLP output projection weights from each layer
                # Path: layers[l].mlp.mlp[-1].weight (last linear layer in Sequential)
                # or layers[l].mlp.mlp.weight (single linear layer)
                # Shape: (emb_dim, ff_dim) -> (emb_dim, emb_dim) or (emb_dim, emb_dim)
                mlp_block = model.layers[layer_idx].mlp
                mlp_module = mlp_block.mlp
                if isinstance(mlp_module, nn.Sequential):
                    # Sequential MLP: use the last (output) linear layer
                    layer_weight = mlp_module[-1].weight.detach().cpu()  # (emb_dim, ff_dim) or (emb_dim, emb_dim)
                elif isinstance(mlp_module, nn.Linear):
                    # Single linear layer MLP
                    layer_weight = mlp_module.weight.detach().cpu()  # (emb_dim, emb_dim)
                else:
                    raise ValueError(f"Unexpected MLP structure in layer {layer_idx}: {type(mlp_module)}")
                weight_matrix = layer_weight
                
                # Compute stable rank
                weight_np = weight_matrix.float().numpy()
                stable_rank_val = stable_rank(weight_np)
                stable_ranks[layer_idx].append(stable_rank_val)
        
        if verbose:
            logger.info(f"  k={k}, n_tasks={n_tasks}, computed stable ranks for {len(layers_to_process)} layers")
    
    # Plot results
    if backend == "plotly" and PLOTLY_AVAILABLE:
        fig = go.Figure()
        
        # Use colormap for different layers
        n_layers_to_plot = len(layers_to_process)
        colors = plt.cm.tab20(np.linspace(0, 1, n_layers_to_plot)) if MATPLOTLIB_AVAILABLE else None
        
        for plot_idx, layer_idx in enumerate(layers_to_process):
            color = f"rgb({int(colors[plot_idx][0]*255)}, {int(colors[plot_idx][1]*255)}, {int(colors[plot_idx][2]*255)})" if colors is not None else None
            fig.add_trace(go.Scatter(
                x=k_values,
                y=stable_ranks[layer_idx],
                mode='lines+markers',
                name=f'Layer {layer_idx}',
                line=dict(width=2, color=color),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=f'Stable Rank of MLP Layer Weights vs k (log2 of number of tasks)<br>Task: {task_name}',
            xaxis_title='k (log2 of number of tasks)',
            yaxis_title='Stable Rank (MLP Weight)',
            hovermode='x unified',
            template='plotly_white',
            width=1000,
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved plot to {save_path}")
        
        if show:
            fig.show()
    
    elif backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use colormap for different layers
        cmap = plt.cm.tab20
        n_layers_to_plot = len(layers_to_process)
        colors = [cmap(i / max(n_layers_to_plot, 1)) for i in range(n_layers_to_plot)]
        
        for plot_idx, layer_idx in enumerate(layers_to_process):
            ax.plot(k_values, stable_ranks[layer_idx], 'o-', 
                   linewidth=1.5, markersize=4, 
                   label=f'Layer {layer_idx}', 
                   color=colors[plot_idx], alpha=0.7)
        
        ax.set_xlabel('k (log2 of number of tasks)', fontsize=12)
        ax.set_ylabel('Stable Rank (MLP Weight)', fontsize=12)
        ax.set_title(f'Stable Rank of MLP Layer Weights vs k\nTask: {task_name}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        if show:
            plt.show()
    
    else:
        if backend == "plotly" and not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available, falling back to matplotlib")
            backend = "matplotlib"
        if backend == "matplotlib" and not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is not installed. Install it with: pip install matplotlib")
        
        # Recursive call with corrected backend
        return plot_stable_rank_final_layer_weight_vs_k(
            task_name, k_values, vocab_size, step, backend, figsize, save_path, show, verbose
        )
    
    return {
        'k_values': k_values,
        'stable_ranks': stable_ranks,  # dict mapping layer_idx to list of stable ranks
        'n_tasks': n_tasks_list,
        'n_layers': n_layers,
        'layers_plotted': layers_to_process,
        'fig': fig,
    }


def plot_posterior_predictor_loss_vs_k_latent(
    k_values: list,
    layer: int,
    vocab_size: Optional[int] = None,
    B: int = 64,
    n_samples: int = 1000,
    step: Optional[int] = None,
    n_minor: Optional[int] = None,
    learning_rate: float = 0.01,
    num_epochs: int = 100,
    verbose: bool = False,
    positions: Optional[list] = None,
    validation_split: float = 0.2,
    uniform_sampling: bool = True,
    backend: str = "matplotlib",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> dict:
    """
    Train posterior predictors for different k values (log2 of number of minor tasks)
    and plot training and validation losses against k.
    
    Parameters:
    -----------
    k_values : list
        List of k values where number of minor tasks = 2^k
    layer : int
        Layer index to extract hidden representations from
    vocab_size : int, optional
        Vocabulary size. If None, uses default from config.
    B : int, default=64
        Batch size for sampling
    n_samples : int, default=1000
        Total number of samples to use for training
    step : int, optional
        Step for checkpoint loading. If None, uses final checkpoint.
    n_minor : int, optional
        Number of minor tasks to use. If None, uses all available minor tasks.
        If -1, uses no minor tasks.
    learning_rate : float, default=0.01
        Learning rate for training the linear model
    num_epochs : int, default=100
        Number of training epochs
    verbose : bool, default=False
        Whether to print progress messages
    positions : list, optional
        List of real token position indices to use. If None, uses first 10 positions.
    validation_split : float, default=0.2
        Fraction of data to use for validation
    uniform_sampling : bool, default=True
        If True, modifies p_minor for uniform sampling. If False, uses original p_minor.
    backend : str, default="matplotlib"
        Plotting backend: "matplotlib" or "plotly"
    figsize : tuple, default=(10, 6)
        Figure size for matplotlib
    save_path : str, optional
        Path to save the figure
    show : bool, default=True
        Whether to display the plot
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'k_values': list of k values
        - 'train_losses': list of final training losses for each k
        - 'val_losses': list of final validation losses for each k
        - 'fig': matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    train_losses = []
    val_losses = []
    
    for k in k_values:
        if verbose:
            logger.info(f"Processing k={k} (n_minor_tasks = {2**k})...")
        
        exp_name = get_exp_name("latent", k, vocab_size=vocab_size)
        
        try:
            results = train_linear_softmax_posterior_predictor(
                exp_name=exp_name,
                layer=layer,
                B=B,
                n_samples=n_samples,
                step=step,
                n_minor=n_minor,
                n_ood=0,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                verbose=verbose,
                positions=positions,
                validation_split=validation_split,
                uniform_sampling=uniform_sampling,
                skip_baselines=True,  # Skip baselines for plotting
            )
            
            train_losses.append(results['final_loss'])
            val_losses.append(results['final_val_loss'])
            
            if verbose:
                logger.info(f"  k={k}: Train Loss: {results['final_loss']:.6f}, Val Loss: {results['final_val_loss']:.6f}")
            
            # Clean up
            del results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error processing k={k}: {e}")
            train_losses.append(float('nan'))
            val_losses.append(float('nan'))
    
    # Plot results
    if backend == "matplotlib":
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(k_values, train_losses, 'o-', label='Training Loss', linewidth=2, markersize=8)
        ax.plot(k_values, val_losses, 's-', label='Validation Loss', linewidth=2, markersize=8)
        ax.set_xlabel('k (log2 of number of minor tasks)', fontsize=12)
        ax.set_ylabel('KL Divergence Loss', fontsize=12)
        ax.set_title(f'Posterior Predictor Loss vs k (Latent Task, Layer {layer})', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
            
    elif backend == "plotly":
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_values, y=train_losses,
            mode='lines+markers',
            name='Training Loss',
            line=dict(width=2),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=k_values, y=val_losses,
            mode='lines+markers',
            name='Validation Loss',
            line=dict(width=2),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title=f'Posterior Predictor Loss vs k (Latent Task, Layer {layer})',
            xaxis_title='k (log2 of number of minor tasks)',
            yaxis_title='KL Divergence Loss',
            width=figsize[0]*100,
            height=figsize[1]*100,
        )
        
        if save_path:
            fig.write_image(save_path)
        if show:
            fig.show()
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    return {
        'k_values': k_values,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'fig': fig,
    }


def plot_posterior_predictor_loss_vs_k_coin(
    k_values: list,
    layer: int,
    vocab_size: Optional[int] = None,
    B: int = 64,
    n_samples: int = 1000,
    step: Optional[int] = None,
    n_minor: Optional[int] = None,
    learning_rate: float = 0.01,
    num_epochs: int = 100,
    verbose: bool = False,
    positions: Optional[list] = None,
    validation_split: float = 0.2,
    uniform_sampling: bool = True,
    backend: str = "matplotlib",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> dict:
    """
    Train posterior predictors for different k values (log2 of number of minor tasks)
    and plot training and validation losses against k.
    
    Parameters are the same as plot_posterior_predictor_loss_vs_k_latent, but for coin task.
    """
    train_losses = []
    val_losses = []
    
    for k in k_values:
        if verbose:
            logger.info(f"Processing k={k} (n_minor_tasks = {2**k})...")
        
        exp_name = get_exp_name("coin", k, vocab_size=vocab_size)
        
        try:
            results = train_linear_softmax_posterior_predictor_coin(
                exp_name=exp_name,
                layer=layer,
                B=B,
                n_samples=n_samples,
                step=step,
                n_minor=n_minor,
                n_ood=0,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                verbose=verbose,
                positions=positions,
                validation_split=validation_split,
                uniform_sampling=uniform_sampling,
                skip_baselines=True,  # Skip baselines for plotting
            )
            
            train_losses.append(results['final_loss'])
            val_losses.append(results['final_val_loss'])
            
            if verbose:
                logger.info(f"  k={k}: Train Loss: {results['final_loss']:.6f}, Val Loss: {results['final_val_loss']:.6f}")
            
            # Clean up
            del results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error processing k={k}: {e}")
            train_losses.append(float('nan'))
            val_losses.append(float('nan'))
    
    # Plot results
    if backend == "matplotlib":
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(k_values, train_losses, 'o-', label='Training Loss', linewidth=2, markersize=8)
        ax.plot(k_values, val_losses, 's-', label='Validation Loss', linewidth=2, markersize=8)
        ax.set_xlabel('k (log2 of number of minor tasks)', fontsize=12)
        ax.set_ylabel('KL Divergence Loss', fontsize=12)
        ax.set_title(f'Posterior Predictor Loss vs k (Coin Task, Layer {layer})', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
            
    elif backend == "plotly":
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_values, y=train_losses,
            mode='lines+markers',
            name='Training Loss',
            line=dict(width=2),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=k_values, y=val_losses,
            mode='lines+markers',
            name='Validation Loss',
            line=dict(width=2),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title=f'Posterior Predictor Loss vs k (Coin Task, Layer {layer})',
            xaxis_title='k (log2 of number of minor tasks)',
            yaxis_title='KL Divergence Loss',
            width=figsize[0]*100,
            height=figsize[1]*100,
        )
        
        if save_path:
            fig.write_image(save_path)
        if show:
            fig.show()
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    return {
        'k_values': k_values,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'fig': fig,
    }


def plot_posterior_predictor_loss_vs_k_dyck(
    k_values: list,
    layer: int,
    B: int = 64,
    n_samples: int = 3000,
    step: Optional[int] = None,
    n_minor: Optional[int] = None,
    n_ood: int = 0,
    learning_rate: float = 0.01,
    num_epochs: int = 200,
    verbose: bool = False,
    n_masks: int = 60,
    min_dyck_positions: int = 5,
    max_dyck_positions: Optional[int] = 15,
    validation_split: float = 0.2,
    uniform_sampling: bool = True,
    skip_baselines: bool = True,
    backend: str = "matplotlib",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> dict:
    """
    Train posterior predictors for different k values (log2 of number of minor tasks)
    and plot training and validation losses against k, for the Dyck task.

    Parameters
    ----------
    k_values : list
        List of k values where number of minor tasks = 2^k.
    layer : int
        Layer index to extract hidden representations from.
    B : int, default=64
        Batch size for sampling.
    n_samples : int, default=3000
        Total number of sequences to generate for training.
    step : int, optional
        Checkpoint step. If None, uses final checkpoint.
    n_minor : int, optional
        Number of minor tasks to use. None = all available, -1 = none.
    n_ood : int, default=0
        Number of OOD tasks to include.
    learning_rate : float, default=0.01
        Learning rate for training.
    num_epochs : int, default=200
        Number of training epochs.
    verbose : bool, default=False
        Whether to print progress messages.
    n_masks : int, default=60
        Number of independently-sampled dyck masks for data collection.
    min_dyck_positions : int, default=5
        Skip the first N Dyck positions (near-uniform, uninformative).
    max_dyck_positions : int, optional, default=15
        Use Dyck positions up to this index (exclusive).
    validation_split : float, default=0.2
        Fraction of data for validation.
    uniform_sampling : bool, default=True
        If True, uses uniform prior across all tasks.
    skip_baselines : bool, default=True
        If True, skips permutation and logits baselines.
    backend : str, default="matplotlib"
        Plotting backend: "matplotlib" or "plotly".
    figsize : tuple, default=(10, 6)
        Figure size for matplotlib.
    save_path : str, optional
        Path to save the figure.
    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    results : dict
        - 'k_values': list of k values
        - 'train_losses': list of final training losses
        - 'val_losses': list of final validation losses
        - 'fig': matplotlib or plotly Figure
    """
    import gc

    train_losses = []
    val_losses = []

    for k in k_values:
        if verbose:
            logger.info(f"Processing k={k} (n_minor_tasks = {2**k})...")

        exp_name = get_exp_name("dyck", k)

        try:
            results = train_linear_softmax_posterior_predictor_dyck(
                exp_name=exp_name,
                layer=layer,
                B=B,
                n_samples=n_samples,
                step=step,
                n_minor=n_minor,
                n_ood=n_ood,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                verbose=verbose,
                n_masks=n_masks,
                min_dyck_positions=min_dyck_positions,
                max_dyck_positions=max_dyck_positions,
                validation_split=validation_split,
                uniform_sampling=uniform_sampling,
                skip_baselines=skip_baselines,
            )

            train_losses.append(results['final_loss'])
            val_losses.append(results['final_val_loss'])

            if verbose:
                logger.info(f"  k={k}: Train Loss: {results['final_loss']:.6f}, Val Loss: {results['final_val_loss']:.6f}")

            del results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing k={k}: {e}")
            train_losses.append(float('nan'))
            val_losses.append(float('nan'))

    # Plot results
    if backend == "matplotlib":
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(k_values, train_losses, 'o-', label='Training Loss', linewidth=2, markersize=8)
        ax.plot(k_values, val_losses, 's-', label='Validation Loss', linewidth=2, markersize=8)
        ax.set_xlabel('k (log2 of number of minor tasks)', fontsize=12)
        ax.set_ylabel('KL Divergence Loss', fontsize=12)
        ax.set_title(f'Posterior Predictor Loss vs k (Dyck Task, Layer {layer})', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    elif backend == "plotly":
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_values, y=train_losses,
            mode='lines+markers',
            name='Training Loss',
            line=dict(width=2),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=k_values, y=val_losses,
            mode='lines+markers',
            name='Validation Loss',
            line=dict(width=2),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title=f'Posterior Predictor Loss vs k (Dyck Task, Layer {layer})',
            xaxis_title='k (log2 of number of minor tasks)',
            yaxis_title='KL Divergence Loss',
            width=figsize[0]*100,
            height=figsize[1]*100,
        )

        if save_path:
            fig.write_image(save_path)
        if show:
            fig.show()
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return {
        'k_values': k_values,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'fig': fig,
    }


def plot_max_stable_rank_vs_k_dyck(
    k_values: list,
    layer: Optional[int] = None,
    B: int = 64,
    n_samples_per_mask: int = 100,
    step: Optional[int] = None,
    n_minor: Optional[int] = None,
    n_ood: int = 0,
    n_masks: int = 30,
    verbose: bool = False,
    backend: str = "matplotlib",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> dict:
    """
    For each k value, collect hidden representations at all valid Dyck-token
    positions using multiple masks, compute the stable rank per layer, and
    plot the maximum stable rank at the specified layer against k.

    Parameters
    ----------
    k_values : list
        List of k values where number of minor tasks = 2^k.
    layer : int, optional
        Layer index to report. If None, uses the final layer.
    B : int, default=64
        Batch size per forward pass.
    n_samples_per_mask : int, default=100
        Number of sequences to generate per mask.
    step : int, optional
        Checkpoint step. If None, uses the final checkpoint.
    n_minor : int, optional
        Number of minor tasks. None = all, -1 = none.
    n_ood : int, default=0
        Number of OOD tasks.
    n_masks : int, default=30
        Number of independently-sampled masks for position diversity.
    verbose : bool, default=False
        Print progress.
    backend : str, default="matplotlib"
        "matplotlib" or "plotly".
    figsize : tuple, default=(10, 6)
        Figure size for matplotlib.
    save_path : str, optional
        Path to save the figure.
    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    results : dict
        - 'k_values': list of k values
        - 'max_stable_ranks': list of max stable ranks for each k
        - 'stable_ranks_per_layer': list of (n_layers,) arrays for each k
        - 'layer': int, the layer index used
        - 'fig': Figure object
    """
    import gc

    max_stable_ranks = []
    stable_ranks_per_layer_all = []

    # Determine layer count from first experiment
    _, _, config0 = nu.load_everything("dyck", get_exp_name("dyck", k_values[0]))
    n_layers = config0.model.num_layers
    del config0

    if layer is None:
        layer_idx = n_layers - 1
    else:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"Layer {layer} out of range [0, {n_layers - 1}]")
        layer_idx = layer

    if verbose:
        logger.info(f"Computing Dyck stable ranks for k_values={k_values}")
        logger.info(f"Using layer {layer_idx} (of {n_layers}), {n_masks} masks, "
                     f"{n_samples_per_mask} samples/mask, B={B}")

    for k in k_values:
        exp_name = get_exp_name("dyck", k)

        if verbose:
            logger.info(f"Processing k={k} (2^k={2**k} minor tasks), exp={exp_name}")

        try:
            # Load model and sampler
            _, _sampler_orig, config = nu.load_everything("dyck", exp_name)
            _step = step if step is not None else config.training.num_epochs
            model, _ = nu.load_checkpoint(config, step=_step, exp_name=exp_name, return_actual_step=True)
            model.eval()
            device = config.device
            model.to(device)

            n_minor_val = n_minor if n_minor is not None else 1000000
            if n_minor_val == -1:
                n_minor_val = 0
            sampler, _ = get_dyck_sampler(exp_name, n_minor_val, n_ood)

            if not getattr(sampler, "pad", False):
                raise ValueError("Requires padded Dyck sequences")

            seq_len_padded = sampler.seq_len

            # Register hooks on all layers
            caches = {}

            def make_hook(li):
                def hook_fn(module, inp, out):
                    if torch.is_tensor(out):
                        caches[li] = out.detach()
                    elif isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                        caches[li] = out[0].detach()
                return hook_fn

            handles = []
            for li in range(n_layers):
                handles.append(model.layers[li].attn_block.register_forward_hook(make_hook(li)))

            # Collect hiddens: per-layer list of CPU tensors
            layer_hiddens = [[] for _ in range(n_layers)]

            masks_list = [sample_binary_mask(config).to(device) for _ in range(n_masks)]

            n_batches_per_mask = (n_samples_per_mask + B - 1) // B

            for mask_idx, current_mask in enumerate(masks_list):
                # Valid Dyck positions for this mask
                dyck_real = torch.nonzero(current_mask == 1, as_tuple=True)[0]
                padded_pos_raw = 2 * dyck_real + 1
                valid = padded_pos_raw < seq_len_padded
                padded_pos = padded_pos_raw[valid].to(device=device, dtype=torch.long)
                n_pos = len(padded_pos)

                if n_pos == 0:
                    continue

                for _ in range(n_batches_per_mask):
                    samples_raw, _ = sampler.generate(
                        mode="train", task=None, num_samples=B, epochs=1,
                        dyck_mask=current_mask.clone(),
                    )
                    if samples_raw.dim() == 3:
                        samples_raw = samples_raw.squeeze(0)
                    samples = samples_raw.to(device)
                    del samples_raw

                    caches.clear()
                    with torch.no_grad():
                        _ = model(samples)

                    for li in range(n_layers):
                        h = caches[li].index_select(dim=1, index=padded_pos)  # (B, n_pos, D)
                        # Flatten to (B * n_pos, D)
                        layer_hiddens[li].append(h.reshape(-1, h.shape[-1]).cpu())

                    del samples
                    caches.clear()

                if verbose and (mask_idx + 1) % 10 == 0:
                    logger.info(f"  Mask {mask_idx + 1}/{n_masks} done")

            # Remove hooks and free model
            for h in handles:
                h.remove()
            model.cpu()
            del model, sampler, _sampler_orig, config, masks_list
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Compute stable rank per layer
            sr_per_layer = torch.zeros(n_layers)
            for li in range(n_layers):
                h_all = torch.cat(layer_hiddens[li], dim=0).float().numpy()  # (N_total, D)
                sr_per_layer[li] = float(stable_rank(h_all))
                del h_all
            del layer_hiddens
            gc.collect()

            max_sr = sr_per_layer[layer_idx].item()
            max_stable_ranks.append(max_sr)
            stable_ranks_per_layer_all.append(sr_per_layer)

            if verbose:
                logger.info(f"  k={k}: stable_rank at layer {layer_idx} = {max_sr:.4f}")

        except Exception as e:
            logger.error(f"Error processing k={k}: {e}")
            max_stable_ranks.append(float('nan'))
            stable_ranks_per_layer_all.append(torch.full((n_layers,), float('nan')))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Plot
    layer_label = f"Layer {layer_idx}" if layer is not None else "Final Layer"

    if backend == "matplotlib":
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(k_values, max_stable_ranks, 'o-', linewidth=2, markersize=8,
                label='Stable Rank', color='blue')
        ax.set_xlabel('k (log2 of number of minor tasks)', fontsize=12)
        ax.set_ylabel(f'Stable Rank at {layer_label}', fontsize=12)
        ax.set_title(f'Stable Rank vs k (Dyck Task, {layer_label})', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    elif backend == "plotly":
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_values, y=max_stable_ranks,
            mode='lines+markers',
            name='Stable Rank',
            line=dict(width=2, color='blue'),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title=f'Stable Rank vs k (Dyck Task, {layer_label})',
            xaxis_title='k (log2 of number of minor tasks)',
            yaxis_title=f'Stable Rank at {layer_label}',
            hovermode='x unified',
            template='plotly_white',
            width=figsize[0]*100,
            height=figsize[1]*100,
        )
        if save_path:
            fig.write_html(save_path)
        if show:
            fig.show()
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return {
        'k_values': k_values,
        'max_stable_ranks': max_stable_ranks,
        'stable_ranks_per_layer': stable_ranks_per_layer_all,
        'layer': layer_idx,
        'fig': fig,
    }


def plot_posterior_predictor_loss_vs_k_linear(
    k_values: list,
    layer: int,
    B: int = 64,
    n_samples: int = 1000,
    step: Optional[int] = None,
    learning_rate: float = 0.01,
    num_epochs: int = 100,
    verbose: bool = False,
    position: Union[int, list] = -1,
    positions: Optional[Union[int, list]] = None,
    validation_split: float = 0.2,
    uniform_sampling: bool = True,
    backend: str = "matplotlib",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> dict:
    """
    Train posterior predictors for different k values (log2 of number of minor tasks)
    and plot training and validation losses against k.
    
    Parameters are similar to plot_posterior_predictor_loss_vs_k_latent, but for linear task.
    Note: linear task doesn't use vocab_size.
    
    Parameters:
    -----------
    positions : int, list, or None, optional
        Alias for 'position' parameter. If provided, overrides 'position'.
        If None, uses 'position' parameter value.
    """
    # Handle positions alias for consistency with other plotting functions
    if positions is not None:
        position = positions
    
    train_losses = []
    val_losses = []
    
    for k in k_values:
        if verbose:
            logger.info(f"Processing k={k} (n_minor_tasks = {2**k})...")
        
        exp_name = get_exp_name("linear", k)
        
        try:
            results = train_linear_softmax_posterior_predictor_linear(
                exp_name=exp_name,
                layer=layer,
                B=B,
                n_samples=n_samples,
                step=step,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                verbose=verbose,
                position=position,
                validation_split=validation_split,
                uniform_sampling=uniform_sampling,
                skip_baselines=True,  # Skip baselines for plotting
            )
            
            train_losses.append(results['final_loss'])
            val_losses.append(results['final_val_loss'])
            
            if verbose:
                logger.info(f"  k={k}: Train Loss: {results['final_loss']:.6f}, Val Loss: {results['final_val_loss']:.6f}")
            
            # Clean up
            del results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error processing k={k}: {e}")
            train_losses.append(float('nan'))
            val_losses.append(float('nan'))
    
    # Plot results
    if backend == "matplotlib":
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(k_values, train_losses, 'o-', label='Training Loss', linewidth=2, markersize=8)
        ax.plot(k_values, val_losses, 's-', label='Validation Loss', linewidth=2, markersize=8)
        ax.set_xlabel('k (log2 of number of minor tasks)', fontsize=12)
        ax.set_ylabel('KL Divergence Loss', fontsize=12)
        ax.set_title(f'Posterior Predictor Loss vs k (Linear Task, Layer {layer})', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
            
    elif backend == "plotly":
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_values, y=train_losses,
            mode='lines+markers',
            name='Training Loss',
            line=dict(width=2),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=k_values, y=val_losses,
            mode='lines+markers',
            name='Validation Loss',
            line=dict(width=2),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title=f'Posterior Predictor Loss vs k (Linear Task, Layer {layer})',
            xaxis_title='k (log2 of number of minor tasks)',
            yaxis_title='KL Divergence Loss',
            width=figsize[0]*100,
            height=figsize[1]*100,
        )
        
        if save_path:
            fig.write_image(save_path)
        if show:
            fig.show()
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    return {
        'k_values': k_values,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'fig': fig,
    }


def get_exp_name(
    task_name, 
    k: int,
    vocab_size = None,
    log2: bool = True
    ) -> str:
    """Generate standardized experiment name based on task and parameters."""
    config = unified_get_config(task_name)
    if task_name != "linear" and vocab_size is not None:
        config.vocab_size = vocab_size
    if k >= 0:
        if log2:
            config.task.n_minor_tasks = 2 ** k
        else:
            config.task.n_minor_tasks = k
    else:
        config.task.n_minor_tasks = 1
        config.task.p_minor = 1e-12  # Practically no minor tasks
    
    exp_name = f"train_{get_hash(config)}"
    return exp_name
    

from icl.models.base_models import Transformer
from icl.utils.train import train_model_with_plot
from icl.linear.train_linear import train

def unified_train(
    task_name,
    k: int,
    vocab_size: int = 11,
    log2: bool = True,
):
    config = unified_get_config(task_name)
    if k >= 0:
        if log2:
            config.task.n_minor_tasks = 2 ** k
        else:
            config.task.n_minor_tasks = k
    else:
        config.task.n_minor_tasks = 1
        config.task.p_minor = 1e-12  # Practically no minor tasks
    if task_name == "linear":
        return train(config)
    else:
        config.vocab_size = vocab_size
        model = Transformer(config)
        model = model.to(config.device)
        return train_model_with_plot(model, config, show=False, verbose=False)
    
