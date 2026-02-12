"""
KL divergence analysis between model predictions and GroupUniformKnownBayes.

This module provides functions to compute and visualize KL divergence between
a trained model's predictions and the Bayesian optimal predictor (GroupUniformKnownBayes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

import icl.utils.notebook_utils as nu
from icl.latent_markov.markov_latent import GroupUniformKnownBayes, ThreeKnownPlusNewDirichletBayes, ThreeKnownUniformBayes
from icl.utils.latent_ood_analysis import get_latent_sampler
from icl.utils.simple_markov_sampler import get_all_samples_base_only
import copy


def compute_kl_divergence_vs_bayes(
    exp_name: str,
    n_minor: int = 64,
    n_ood: int = 30,
    batch_size: int = 64,
    step: Optional[int] = None,
    p_common: float = 0.9,
    device: str = "cuda",
) -> dict:
    """
    Compare model output with GroupUniformKnownBayes predictor using KL divergence.
    
    This function:
    1. Loads the model and sampler based on exp_name
    2. Samples sequences from major and minor tasks
    3. Gets model predictions at padded token positions
    4. Compares with GroupUniformKnownBayes predictions on non-padded sequences
    5. Computes average KL divergence separately for major and minor tasks
    
    Args:
        exp_name: Experiment name to load model and sampler
        n_minor: Number of minority tasks to sample
        n_ood: Number of OOD tasks to sample (not used for this analysis)
        batch_size: Number of sequences to sample per task
        step: Training step to load (if None, uses final checkpoint)
        p_common: Prior probability for major tasks in GroupUniformKnownBayes
        device: Device to run computation on
        
    Returns:
        Dictionary containing:
            - 'kl_major': KL divergence for major tasks, shape (n_major_tasks, n_positions)
            - 'kl_minor': KL divergence for minor tasks, shape (n_minor_tasks, n_positions)
            - 'kl_major_mean': Average KL for major tasks across tasks, shape (n_positions,)
            - 'kl_minor_mean': Average KL for minor tasks across tasks, shape (n_positions,)
            - 'positions': Position indices (accounting for padding)
            - 'n_major_tasks': Number of major tasks
            - 'n_minor_tasks': Number of minor tasks sampled
    """
    # Load model, sampler, and config
    result = nu.load_everything("latent", exp_name)
    if len(result) == 4:
        model, sampler, config, _ = result
    else:
        model, sampler, config = result
    
    if step is None:
        step = config.training.num_epochs  # type: ignore
    
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model = model.to(device)
    model.eval()
    
    # Get sampler parameters
    n_major_tasks = sampler.n_major_tasks
    k_minor = min(n_minor, sampler.n_minor_tasks)
    num_states = sampler.num_states
    seq_len = sampler.seq_len
    use_padding = hasattr(sampler, 'pad') and sampler.pad
    
    # Collect all transition matrices for GroupUniformKnownBayes
    # First 3 are major (common), rest are minor tasks
    trans_mats_list = []
    for i in range(n_major_tasks):
        trans_mats_list.append(sampler.major_trans_mat[i])
    for i in range(k_minor):
        trans_mats_list.append(sampler.minor_trans_mat[i])
    
    all_trans_mat = torch.stack(trans_mats_list, dim=0)  # (M, num_states_order, num_states)
    
    # Initialize GroupUniformKnownBayes
    bayes_predictor = GroupUniformKnownBayes(
        trans_mat=all_trans_mat,
        p_common=p_common,
        device=device
    )
    
    # Storage for KL divergences
    if use_padding:
        # For padded sequences, model output at position 2*i+1 predicts token i+1
        # We can predict from position 0 to seq_len-2 (predicting tokens 1 to seq_len-1)
        # The last token (at position seq_len-1) cannot have a next token predicted
        n_positions = seq_len - 1  # Number of positions where we can make predictions
    else:
        n_positions = seq_len - 1  # Same logic for non-padded
    
    kl_major = torch.zeros(n_major_tasks, n_positions, device=device)
    kl_minor = torch.zeros(k_minor, n_positions, device=device)
    
    # Process major tasks
    print(f"Processing {n_major_tasks} major tasks...")
    for task_idx in range(n_major_tasks):
        # Generate samples for this task
        samples_padded, _ = sampler.generate(
            num_samples=batch_size,
            mode="major",
            task=task_idx
        )
        samples_padded = samples_padded.to(device)
        
        # Get non-padded samples for Bayes predictor
        if use_padding:
            # Extract non-padded tokens (even indices in padded sequence)
            samples_nonpadded = samples_padded[:, ::2]  # (B, seq_len)
        else:
            samples_nonpadded = samples_padded
        
        # Get model predictions on padded input
        with torch.no_grad():
            model_output = model(samples_padded)
            # Handle different model output formats
            if isinstance(model_output, tuple):
                logits = model_output[0]
            else:
                logits = model_output
            
            # Debug: check shapes on first task
            if task_idx == 0:
                print(f"Debug - Major tasks:")
                print(f"  samples_padded shape: {samples_padded.shape}")
                print(f"  logits shape: {logits.shape}")
                print(f"  num_states: {num_states}")
            
            model_probs = F.softmax(logits, dim=-1)  # (B, T, vocab_size) or (B, vocab_size)
        
        # Get Bayes predictions on non-padded sequence
        bayes_probs = bayes_predictor.predict(samples_nonpadded)  # (B, seq_len, num_states)
        
        # Compute KL divergence at each position
        # Note: If model returns predictions for padded seq, use padded_pos
        # If model returns predictions for non-padded seq, use pos directly
        for pos in range(n_positions):
            # Check if we should use padded position or direct position based on model output shape
            if use_padding and model_probs.shape[1] > n_positions:
                # Model returns predictions for full padded sequence
                padded_pos = 2 * pos + 1
                model_p = model_probs[:, padded_pos, :num_states]  # (B, num_states)
            else:
                # Model returns predictions for non-padded positions only
                # (or no padding is used)
                model_p = model_probs[:, pos, :num_states]  # (B, num_states)
            
            # Bayes predictor always works with non-padded sequence
            bayes_p = bayes_probs[:, pos, :]  # (B, num_states)
            
            # KL divergence: KL(model || bayes)
            # F.kl_div expects log(model_p) and bayes_p
            kl = F.kl_div(
                model_p.log().clamp(min=-10),  # Clamp to avoid log(0)
                bayes_p.clamp(min=1e-10),
                reduction='none'
            ).sum(dim=-1)  # (B,)
            
            kl_major[task_idx, pos] = kl.mean()
    
    # Process minor tasks
    print(f"Processing {k_minor} minor tasks...")
    for task_idx in range(k_minor):
        # Generate samples for this task
        samples_padded, _ = sampler.generate(
            num_samples=batch_size,
            mode="minor",
            task=task_idx
        )
        samples_padded = samples_padded.to(device)
        
        # Get non-padded samples for Bayes predictor
        if use_padding:
            samples_nonpadded = samples_padded[:, ::2]  # (B, seq_len)
        else:
            samples_nonpadded = samples_padded
        
        # Get model predictions on padded input
        with torch.no_grad():
            model_output = model(samples_padded)
            # Handle different model output formats
            if isinstance(model_output, tuple):
                logits = model_output[0]
            else:
                logits = model_output
            
            # Debug: check shapes on first minor task
            if task_idx == 0:
                print(f"Debug - Minor tasks:")
                print(f"  samples_padded shape: {samples_padded.shape}")
                print(f"  logits shape: {logits.shape}")
            
            model_probs = F.softmax(logits, dim=-1)  # (B, T, vocab_size) or (B, vocab_size)
        
        # Get Bayes predictions on non-padded sequence
        bayes_probs = bayes_predictor.predict(samples_nonpadded)  # (B, seq_len, num_states)
        
        # Compute KL divergence at each position
        for pos in range(n_positions):
            # Check if we should use padded position or direct position based on model output shape
            if use_padding and model_probs.shape[1] > n_positions:
                # Model returns predictions for full padded sequence
                padded_pos = 2 * pos + 1
                model_p = model_probs[:, padded_pos, :num_states]  # (B, num_states)
            else:
                # Model returns predictions for non-padded positions only
                model_p = model_probs[:, pos, :num_states]  # (B, num_states)
            
            # Bayes predictor always works with non-padded sequence
            bayes_p = bayes_probs[:, pos, :]  # (B, num_states)
            
            # KL divergence: KL(model || bayes)
            kl = F.kl_div(
                model_p.log().clamp(min=-10),
                bayes_p.clamp(min=1e-10),
                reduction='none'
            ).sum(dim=-1)  # (B,)
            
            kl_minor[task_idx, pos] = kl.mean()
    
    # Compute averages and standard deviations
    kl_major_mean = kl_major.mean(dim=0).cpu().numpy()
    kl_minor_mean = kl_minor.mean(dim=0).cpu().numpy()
    kl_major_std = kl_major.std(dim=0).cpu().numpy()
    kl_minor_std = kl_minor.std(dim=0).cpu().numpy()
    
    positions = np.arange(n_positions)
    
    return {
        'kl_major': kl_major.cpu().numpy(),
        'kl_minor': kl_minor.cpu().numpy(),
        'kl_major_mean': kl_major_mean,
        'kl_minor_mean': kl_minor_mean,
        'kl_major_std': kl_major_std,
        'kl_minor_std': kl_minor_std,
        'positions': positions,
        'n_major_tasks': n_major_tasks,
        'n_minor_tasks': k_minor,
    }


def plot_kl_divergence(
    results: dict,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot KL divergence between model and GroupUniformKnownBayes vs position.
    
    Args:
        results: Dictionary returned by compute_kl_divergence_vs_bayes
        title: Optional plot title
        save_path: Optional path to save the plot
        show: Whether to display the plot
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    positions = results['positions']
    kl_major_mean = results['kl_major_mean']
    kl_minor_mean = results['kl_minor_mean']
    kl_major_std = results['kl_major_std']
    kl_minor_std = results['kl_minor_std']
    
    # Plot major tasks
    ax.plot(positions, kl_major_mean, 
            label=f"Major tasks (n={results['n_major_tasks']})",
            linewidth=2, marker='o', markersize=3, alpha=0.7, color='C0')
    ax.fill_between(positions, 
                    kl_major_mean - kl_major_std, 
                    kl_major_mean + kl_major_std,
                    alpha=0.2, color='C0')
    
    # Plot minor tasks
    ax.plot(positions, kl_minor_mean,
            label=f"Minor tasks (n={results['n_minor_tasks']})",
            linewidth=2, marker='s', markersize=3, alpha=0.7, color='C1')
    ax.fill_between(positions, 
                    kl_minor_mean - kl_minor_std, 
                    kl_minor_mean + kl_minor_std,
                    alpha=0.2, color='C1')
    
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('KL Divergence', fontsize=12)
    
    if title is None:
        title = 'KL Divergence: Model vs GroupUniformKnownBayes'
    ax.set_title(title, fontsize=14)
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, ax


def analyze_kl_divergence(
    exp_name: str,
    n_minor: int = 64,
    batch_size: int = 64,
    step: Optional[int] = None,
    p_common: float = 0.9,
    device: str = "cuda",
    save_path: Optional[str] = None,
    show: bool = True,
) -> dict:
    """
    Complete analysis: compute and plot KL divergence.
    
    This is a convenience function that combines compute_kl_divergence_vs_bayes
    and plot_kl_divergence.
    
    Args:
        exp_name: Experiment name to load model and sampler
        n_minor: Number of minority tasks to sample
        batch_size: Number of sequences to sample per task
        step: Training step to load (if None, uses final checkpoint)
        p_common: Prior probability for major tasks in GroupUniformKnownBayes
        device: Device to run computation on
        save_path: Optional path to save the plot
        show: Whether to display the plot
        
    Returns:
        Dictionary containing KL divergence results
    """
    print(f"Analyzing experiment: {exp_name}")
    
    # Compute KL divergence
    results = compute_kl_divergence_vs_bayes(
        exp_name=exp_name,
        n_minor=n_minor,
        batch_size=batch_size,
        step=step,
        p_common=p_common,
        device=device,
    )
    
    # Plot results
    plot_kl_divergence(
        results=results,
        title=f'KL Divergence for {exp_name}',
        save_path=save_path,
        show=show,
    )
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Major tasks: n={results['n_major_tasks']}")
    print(f"  Mean KL (first 10 pos): {results['kl_major_mean'][:10].mean():.4f}")
    print(f"  Mean KL (last 10 pos): {results['kl_major_mean'][-10:].mean():.4f}")
    print(f"\nMinor tasks: n={results['n_minor_tasks']}")
    print(f"  Mean KL (first 10 pos): {results['kl_minor_mean'][:10].mean():.4f}")
    print(f"  Mean KL (last 10 pos): {results['kl_minor_mean'][-10:].mean():.4f}")
    
    return results


def compute_kl_divergence_vs_dirichlet_bayes(
    exp_name: str,
    n_minor: int = 64,
    n_ood: int = 30,
    batch_size: int = 64,
    step: Optional[int] = None,
    p_common: float = 0.9,
    alpha: float = 1.0,
    device: str = "cuda",
) -> dict:
    """
    Compare model output with ThreeKnownPlusNewDirichletBayes predictor using KL divergence.
    
    This function uses ThreeKnownPlusNewDirichletBayes which:
    - Uses only the first 3 known transition matrices (major tasks)
    - Treats all other tasks as "new" unknown tasks with Dirichlet prior
    
    Args:
        exp_name: Experiment name to load model and sampler
        n_minor: Number of minority tasks to sample
        n_ood: Number of OOD tasks to sample (not used for this analysis)
        batch_size: Number of sequences to sample per task
        step: Training step to load (if None, uses final checkpoint)
        p_common: Prior probability for the 3 known major tasks
        alpha: Dirichlet concentration parameter (default: 1.0)
        device: Device to run computation on
        
    Returns:
        Dictionary containing:
            - 'kl_major': KL divergence for major tasks, shape (n_major_tasks, n_positions)
            - 'kl_minor': KL divergence for minor tasks, shape (n_minor_tasks, n_positions)
            - 'kl_major_mean': Average KL for major tasks across tasks, shape (n_positions,)
            - 'kl_minor_mean': Average KL for minor tasks across tasks, shape (n_positions,)
            - 'kl_major_std': Std KL for major tasks across tasks, shape (n_positions,)
            - 'kl_minor_std': Std KL for minor tasks across tasks, shape (n_positions,)
            - 'positions': Position indices (accounting for padding)
            - 'n_major_tasks': Number of major tasks
            - 'n_minor_tasks': Number of minor tasks sampled
    """
    # Load model, sampler, and config
    result = nu.load_everything("latent", exp_name)
    if len(result) == 4:
        model, sampler, config, _ = result
    else:
        model, sampler, config = result
    
    if step is None:
        step = config.training.num_epochs  # type: ignore
    
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model = model.to(device)
    model.eval()
    
    # Get sampler parameters
    n_major_tasks = sampler.n_major_tasks
    k_minor = min(n_minor, sampler.n_minor_tasks)
    num_states = sampler.num_states
    seq_len = sampler.seq_len
    use_padding = hasattr(sampler, 'pad') and sampler.pad
    
    # For ThreeKnownPlusNewDirichletBayes, we only use the first 3 major task matrices
    trans_mat_3 = sampler.major_trans_mat[:3]  # (3, num_states_order, num_states)
    
    # Initialize ThreeKnownPlusNewDirichletBayes
    bayes_predictor = ThreeKnownPlusNewDirichletBayes(
        trans_mat_3=trans_mat_3,
        p_common_total=p_common,
        alpha=alpha,
        device=device
    )
    
    # Storage for KL divergences
    if use_padding:
        n_positions = seq_len - 1
    else:
        n_positions = seq_len - 1
    
    kl_major = torch.zeros(n_major_tasks, n_positions, device=device)
    kl_minor = torch.zeros(k_minor, n_positions, device=device)
    
    # Process major tasks
    print(f"Processing {n_major_tasks} major tasks...")
    for task_idx in range(n_major_tasks):
        # Generate samples for this task
        samples_padded, _ = sampler.generate(
            num_samples=batch_size,
            mode="major",
            task=task_idx
        )
        samples_padded = samples_padded.to(device)
        
        # Get non-padded samples for Bayes predictor
        if use_padding:
            samples_nonpadded = samples_padded[:, ::2]  # (B, seq_len)
        else:
            samples_nonpadded = samples_padded
        
        # Get model predictions on padded input
        with torch.no_grad():
            model_output = model(samples_padded)
            if isinstance(model_output, tuple):
                logits = model_output[0]
            else:
                logits = model_output
            
            if task_idx == 0:
                print(f"Debug - Major tasks:")
                print(f"  samples_padded shape: {samples_padded.shape}")
                print(f"  logits shape: {logits.shape}")
                print(f"  num_states: {num_states}")
            
            model_probs = F.softmax(logits, dim=-1)
        
        # Get Bayes predictions on non-padded sequence
        bayes_probs = bayes_predictor.predict(samples_nonpadded)  # (B, seq_len, num_states)
        
        # Compute KL divergence at each position
        for pos in range(n_positions):
            if use_padding and model_probs.shape[1] > n_positions:
                padded_pos = 2 * pos + 1
                model_p = model_probs[:, padded_pos, :num_states]
            else:
                model_p = model_probs[:, pos, :num_states]
            
            bayes_p = bayes_probs[:, pos, :]
            
            # KL divergence: KL(model || bayes)
            kl = F.kl_div(
                model_p.log().clamp(min=-10),
                bayes_p.clamp(min=1e-10),
                reduction='none'
            ).sum(dim=-1)  # (B,)
            
            kl_major[task_idx, pos] = kl.mean()
    
    # Process minor tasks
    print(f"Processing {k_minor} minor tasks...")
    for task_idx in range(k_minor):
        # Generate samples for this task
        samples_padded, _ = sampler.generate(
            num_samples=batch_size,
            mode="minor",
            task=task_idx
        )
        samples_padded = samples_padded.to(device)
        
        # Get non-padded samples for Bayes predictor
        if use_padding:
            samples_nonpadded = samples_padded[:, ::2]
        else:
            samples_nonpadded = samples_padded
        
        # Get model predictions on padded input
        with torch.no_grad():
            model_output = model(samples_padded)
            if isinstance(model_output, tuple):
                logits = model_output[0]
            else:
                logits = model_output
            
            if task_idx == 0:
                print(f"Debug - Minor tasks:")
                print(f"  samples_padded shape: {samples_padded.shape}")
                print(f"  logits shape: {logits.shape}")
            
            model_probs = F.softmax(logits, dim=-1)
        
        # Get Bayes predictions on non-padded sequence
        bayes_probs = bayes_predictor.predict(samples_nonpadded)
        
        # Compute KL divergence at each position
        for pos in range(n_positions):
            if use_padding and model_probs.shape[1] > n_positions:
                padded_pos = 2 * pos + 1
                model_p = model_probs[:, padded_pos, :num_states]
            else:
                model_p = model_probs[:, pos, :num_states]
            
            bayes_p = bayes_probs[:, pos, :]
            
            kl = F.kl_div(
                model_p.log().clamp(min=-10),
                bayes_p.clamp(min=1e-10),
                reduction='none'
            ).sum(dim=-1)
            
            kl_minor[task_idx, pos] = kl.mean()
    
    # Compute averages and standard deviations
    kl_major_mean = kl_major.mean(dim=0).cpu().numpy()
    kl_minor_mean = kl_minor.mean(dim=0).cpu().numpy()
    kl_major_std = kl_major.std(dim=0).cpu().numpy()
    kl_minor_std = kl_minor.std(dim=0).cpu().numpy()
    
    positions = np.arange(n_positions)
    
    return {
        'kl_major': kl_major.cpu().numpy(),
        'kl_minor': kl_minor.cpu().numpy(),
        'kl_major_mean': kl_major_mean,
        'kl_minor_mean': kl_minor_mean,
        'kl_major_std': kl_major_std,
        'kl_minor_std': kl_minor_std,
        'positions': positions,
        'n_major_tasks': n_major_tasks,
        'n_minor_tasks': k_minor,
    }


def analyze_kl_divergence_dirichlet(
    exp_name: str,
    n_minor: int = 64,
    batch_size: int = 64,
    step: Optional[int] = None,
    p_common: float = 0.9,
    alpha: float = 1.0,
    device: str = "cuda",
    save_path: Optional[str] = None,
    show: bool = True,
) -> dict:
    """
    Complete analysis: compute and plot KL divergence vs ThreeKnownPlusNewDirichletBayes.
    
    This is a convenience function that combines compute_kl_divergence_vs_dirichlet_bayes
    and plot_kl_divergence.
    
    Args:
        exp_name: Experiment name to load model and sampler
        n_minor: Number of minority tasks to sample
        batch_size: Number of sequences to sample per task
        step: Training step to load (if None, uses final checkpoint)
        p_common: Prior probability for the 3 known major tasks
        alpha: Dirichlet concentration parameter (default: 1.0)
        device: Device to run computation on
        save_path: Optional path to save the plot
        show: Whether to display the plot
        
    Returns:
        Dictionary containing KL divergence results
    """
    print(f"Analyzing experiment: {exp_name}")
    print(f"Using ThreeKnownPlusNewDirichletBayes with p_common={p_common}, alpha={alpha}")
    
    # Compute KL divergence
    results = compute_kl_divergence_vs_dirichlet_bayes(
        exp_name=exp_name,
        n_minor=n_minor,
        batch_size=batch_size,
        step=step,
        p_common=p_common,
        alpha=alpha,
        device=device,
    )
    
    # Plot results
    plot_kl_divergence(
        results=results,
        title=f'KL Divergence vs Dirichlet Bayes: {exp_name}',
        save_path=save_path,
        show=show,
    )
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Major tasks: n={results['n_major_tasks']}")
    print(f"  Mean KL (first 10 pos): {results['kl_major_mean'][:10].mean():.4f}")
    print(f"  Mean KL (last 10 pos): {results['kl_major_mean'][-10:].mean():.4f}")
    print(f"\nMinor tasks: n={results['n_minor_tasks']}")
    print(f"  Mean KL (first 10 pos): {results['kl_minor_mean'][:10].mean():.4f}")
    print(f"  Mean KL (last 10 pos): {results['kl_minor_mean'][-10:].mean():.4f}")
    
    return results


def compare_bayes_predictors(
    exp_name: str,
    n_minor: int = 64,
    n_ood: int = 30,
    batch_size: int = 64,
    step: Optional[int] = None,
    p_common: float = 0.9,
    alpha: float = 1.0,
    device: str = "cuda",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    plot_list: Optional[list] = None,
) -> dict:
    """
    Compare model KL divergence against multiple predictors.
    
    This function:
    1. Samples minor tasks and OOD tasks (following get_all_samples pattern)
    2. Computes KL divergence vs GroupUniformKnownBayes (knows all tasks)
    3. Computes KL divergence vs ThreeKnownPlusNewDirichletBayes (knows 3, Dirichlet for rest)
    4. Computes KL divergence vs ThreeKnownUniformBayes (knows 3, uniform for rest)
    5. Computes KL divergence vs Uniform distribution (over real tokens)
    6. Plots selected comparisons side-by-side, distinguishing minor vs OOD tasks
    
    Args:
        exp_name: Experiment name to load model and sampler
        n_minor: Number of minority tasks to sample
        n_ood: Number of OOD tasks to sample
        batch_size: Number of sequences to sample per task
        step: Training step to load (if None, uses final checkpoint)
        p_common: Prior probability for major/known tasks
        alpha: Dirichlet concentration parameter for ThreeKnownPlusNewDirichletBayes
        device: Device to run computation on
        save_path: Optional path to save the plot
        show: Whether to display the plot
        figsize: Figure size (width, height), auto-calculated if None
        plot_list: List of plots to show. Options: 'group', 'dirichlet', 'uniform_bayes', 'uniform_dist'.
                   If None, shows all plots
        
    Returns:
        Dictionary containing results for both predictors and both task types
    """
    # Default plot list: show all plots
    if plot_list is None:
        plot_list = ['group', 'dirichlet', 'uniform_bayes', 'uniform_dist']
    
    # Auto-calculate figsize based on number of plots
    n_plots = len(plot_list)
    if figsize is None:
        figsize = (6 * n_plots, 6)
    
    print(f"Analyzing experiment: {exp_name}")
    print(f"Sampling {n_minor} minor tasks and {n_ood} OOD tasks")
    print(f"Plotting: {', '.join(plot_list)}")
    
    # Load model, sampler, and config
    result = nu.load_everything("latent", exp_name)
    if len(result) == 4:
        model, sampler, config, _ = result
    else:
        model, sampler, config = result
    
    if step is None:
        step = config.training.num_epochs  # type: ignore
    
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model = model.to(device)
    model.eval()
    
    # Get sampler parameters
    n_major_tasks = sampler.n_major_tasks
    num_states = sampler.num_states
    seq_len = sampler.seq_len
    use_padding = hasattr(sampler, 'pad') and sampler.pad
    original_n_minor = sampler.n_minor_tasks  # Get original number of minor tasks
    
    # Create expanded sampler with OOD tasks (following get_latent_sampler pattern)
    sampler_clone = copy.deepcopy(sampler)
    k_minor = min(n_minor, sampler_clone.n_minor_tasks)
    n_tasks = k_minor + n_ood
    sampler_clone.n_minor_tasks = n_tasks
    
    orig = sampler_clone.minor_trans_mat
    ood = sampler_clone._sample_banded_trans_mats(n_ood)
    ood = ood.to(device=orig.device, dtype=orig.dtype)
    
    new_shape = (n_tasks, *orig.shape[1:])
    new_minor = orig.new_empty(new_shape)
    new_minor[:n_ood].copy_(ood)
    if k_minor > 0:
        new_minor[n_ood:].copy_(orig[:k_minor])
    sampler_clone.minor_trans_mat = new_minor
    
    # Collect all transition matrices for GroupUniformKnownBayes
    trans_mats_list = []
    for i in range(n_major_tasks):
        trans_mats_list.append(sampler.major_trans_mat[i])
    for i in range(k_minor):
        trans_mats_list.append(sampler.minor_trans_mat[i])
    for i in range(n_ood):
        trans_mats_list.append(ood[i])
    
    all_trans_mat = torch.stack(trans_mats_list, dim=0)  # (M, num_states_order, num_states)
    
    # Initialize both predictors
    group_bayes = GroupUniformKnownBayes(
        trans_mat=all_trans_mat,
        p_common=p_common,
        device=device
    )
    
    dirichlet_bayes = ThreeKnownPlusNewDirichletBayes(
        trans_mat_3=sampler.major_trans_mat[:3],
        p_common_total=p_common,
        alpha=alpha,
        device=device
    )
    
    uniform_bayes = ThreeKnownUniformBayes(
        trans_mat_3=sampler.major_trans_mat[:3],
        device=device
    )
    
    # Storage for KL divergences
    if use_padding:
        n_positions = seq_len - 1
    else:
        n_positions = seq_len - 1
    
    # Results for minor and OOD tasks, for all predictors
    kl_minor_group = torch.zeros(k_minor, n_positions, device=device)
    kl_minor_dirichlet = torch.zeros(k_minor, n_positions, device=device)
    kl_minor_uniform = torch.zeros(k_minor, n_positions, device=device)
    kl_minor_uniform_dist = torch.zeros(k_minor, n_positions, device=device)
    kl_ood_group = torch.zeros(n_ood, n_positions, device=device)
    kl_ood_dirichlet = torch.zeros(n_ood, n_positions, device=device)
    kl_ood_uniform = torch.zeros(n_ood, n_positions, device=device)
    kl_ood_uniform_dist = torch.zeros(n_ood, n_positions, device=device)
    
    # Create uniform distribution over real tokens (excluding padding)
    uniform_dist = torch.ones(num_states, device=device) / num_states
    
    # Process minor tasks
    print(f"Processing {k_minor} minor tasks...")
    for task_idx in range(k_minor):
        samples_padded, _ = sampler_clone.generate(
            num_samples=batch_size,
            mode="minor",
            task=n_ood + task_idx  # Offset by n_ood since OOD tasks are first
        )
        samples_padded = samples_padded.to(device)
        
        if use_padding:
            samples_nonpadded = samples_padded[:, ::2]
        else:
            samples_nonpadded = samples_padded
        
        with torch.no_grad():
            model_output = model(samples_padded)
            if isinstance(model_output, tuple):
                logits = model_output[0]
            else:
                logits = model_output
            model_probs = F.softmax(logits, dim=-1)
        
        # Get predictions from all three Bayes predictors
        bayes_probs_group = group_bayes.predict(samples_nonpadded)
        bayes_probs_dirichlet = dirichlet_bayes.predict(samples_nonpadded)
        bayes_probs_uniform = uniform_bayes.predict(samples_nonpadded)
        
        # Compute KL divergence at each position for all three predictors
        for pos in range(n_positions):
            if use_padding and model_probs.shape[1] > n_positions:
                padded_pos = 2 * pos + 1
                model_p = model_probs[:, padded_pos, :num_states]
            else:
                model_p = model_probs[:, pos, :num_states]
            
            # KL vs GroupUniformKnownBayes
            bayes_p_group = bayes_probs_group[:, pos, :]
            kl_group = F.kl_div(
                model_p.log().clamp(min=-10),
                bayes_p_group.clamp(min=1e-10),
                reduction='none'
            ).sum(dim=-1)
            kl_minor_group[task_idx, pos] = kl_group.mean()
            
            # KL vs ThreeKnownPlusNewDirichletBayes
            bayes_p_dirichlet = bayes_probs_dirichlet[:, pos, :]
            kl_dirichlet = F.kl_div(
                model_p.log().clamp(min=-10),
                bayes_p_dirichlet.clamp(min=1e-10),
                reduction='none'
            ).sum(dim=-1)
            kl_minor_dirichlet[task_idx, pos] = kl_dirichlet.mean()
            
            # KL vs ThreeKnownUniformBayes
            bayes_p_uniform = bayes_probs_uniform[:, pos, :]
            kl_uniform = F.kl_div(
                model_p.log().clamp(min=-10),
                bayes_p_uniform.clamp(min=1e-10),
                reduction='none'
            ).sum(dim=-1)
            kl_minor_uniform[task_idx, pos] = kl_uniform.mean()
            
            # KL vs Uniform Distribution
            kl_uniform_dist = F.kl_div(
                model_p.log().clamp(min=-10),
                uniform_dist.unsqueeze(0).expand_as(model_p).clamp(min=1e-10),
                reduction='none'
            ).sum(dim=-1)
            kl_minor_uniform_dist[task_idx, pos] = kl_uniform_dist.mean()
    
    # Process OOD tasks
    print(f"Processing {n_ood} OOD tasks...")
    for task_idx in range(n_ood):
        samples_padded, _ = sampler_clone.generate(
            num_samples=batch_size,
            mode="minor",
            task=task_idx  # OOD tasks are at the beginning
        )
        samples_padded = samples_padded.to(device)
        
        if use_padding:
            samples_nonpadded = samples_padded[:, ::2]
        else:
            samples_nonpadded = samples_padded
        
        with torch.no_grad():
            model_output = model(samples_padded)
            if isinstance(model_output, tuple):
                logits = model_output[0]
            else:
                logits = model_output
            model_probs = F.softmax(logits, dim=-1)
        
        # Get predictions from all three Bayes predictors
        bayes_probs_group = group_bayes.predict(samples_nonpadded)
        bayes_probs_dirichlet = dirichlet_bayes.predict(samples_nonpadded)
        bayes_probs_uniform = uniform_bayes.predict(samples_nonpadded)
        
        # Compute KL divergence at each position for all three predictors
        for pos in range(n_positions):
            if use_padding and model_probs.shape[1] > n_positions:
                padded_pos = 2 * pos + 1
                model_p = model_probs[:, padded_pos, :num_states]
            else:
                model_p = model_probs[:, pos, :num_states]
            
            # KL vs GroupUniformKnownBayes
            bayes_p_group = bayes_probs_group[:, pos, :]
            kl_group = F.kl_div(
                model_p.log().clamp(min=-10),
                bayes_p_group.clamp(min=1e-10),
                reduction='none'
            ).sum(dim=-1)
            kl_ood_group[task_idx, pos] = kl_group.mean()
            
            # KL vs ThreeKnownPlusNewDirichletBayes
            bayes_p_dirichlet = bayes_probs_dirichlet[:, pos, :]
            kl_dirichlet = F.kl_div(
                model_p.log().clamp(min=-10),
                bayes_p_dirichlet.clamp(min=1e-10),
                reduction='none'
            ).sum(dim=-1)
            kl_ood_dirichlet[task_idx, pos] = kl_dirichlet.mean()
            
            # KL vs ThreeKnownUniformBayes
            bayes_p_uniform = bayes_probs_uniform[:, pos, :]
            kl_uniform = F.kl_div(
                model_p.log().clamp(min=-10),
                bayes_p_uniform.clamp(min=1e-10),
                reduction='none'
            ).sum(dim=-1)
            kl_ood_uniform[task_idx, pos] = kl_uniform.mean()
            
            # KL vs Uniform Distribution
            kl_uniform_dist = F.kl_div(
                model_p.log().clamp(min=-10),
                uniform_dist.unsqueeze(0).expand_as(model_p).clamp(min=1e-10),
                reduction='none'
            ).sum(dim=-1)
            kl_ood_uniform_dist[task_idx, pos] = kl_uniform_dist.mean()
    
    # Compute statistics
    positions = np.arange(n_positions)
    
    results = {
        'positions': positions,
        'minor': {
            'group_mean': kl_minor_group.mean(dim=0).cpu().numpy(),
            'group_std': kl_minor_group.std(dim=0).cpu().numpy(),
            'dirichlet_mean': kl_minor_dirichlet.mean(dim=0).cpu().numpy(),
            'dirichlet_std': kl_minor_dirichlet.std(dim=0).cpu().numpy(),
            'uniform_mean': kl_minor_uniform.mean(dim=0).cpu().numpy(),
            'uniform_std': kl_minor_uniform.std(dim=0).cpu().numpy(),
            'uniform_dist_mean': kl_minor_uniform_dist.mean(dim=0).cpu().numpy(),
            'uniform_dist_std': kl_minor_uniform_dist.std(dim=0).cpu().numpy(),
        },
        'ood': {
            'group_mean': kl_ood_group.mean(dim=0).cpu().numpy(),
            'group_std': kl_ood_group.std(dim=0).cpu().numpy(),
            'dirichlet_mean': kl_ood_dirichlet.mean(dim=0).cpu().numpy(),
            'dirichlet_std': kl_ood_dirichlet.std(dim=0).cpu().numpy(),
            'uniform_mean': kl_ood_uniform.mean(dim=0).cpu().numpy(),
            'uniform_std': kl_ood_uniform.std(dim=0).cpu().numpy(),
            'uniform_dist_mean': kl_ood_uniform_dist.mean(dim=0).cpu().numpy(),
            'uniform_dist_std': kl_ood_uniform_dist.std(dim=0).cpu().numpy(),
        },
        'n_minor': k_minor,
        'n_ood': n_ood,
    }
    
    # Create side-by-side plots
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]  # Make it iterable
    
    # Compute global y-axis limits for consistent scale across all selected plots
    all_means_list = []
    all_stds_list = []
    for plot_name in plot_list:
        if plot_name == 'group':
            all_means_list.extend([results['minor']['group_mean'], results['ood']['group_mean']])
            all_stds_list.extend([results['minor']['group_std'], results['ood']['group_std']])
        elif plot_name == 'dirichlet':
            all_means_list.extend([results['minor']['dirichlet_mean'], results['ood']['dirichlet_mean']])
            all_stds_list.extend([results['minor']['dirichlet_std'], results['ood']['dirichlet_std']])
        elif plot_name == 'uniform_bayes':
            all_means_list.extend([results['minor']['uniform_mean'], results['ood']['uniform_mean']])
            all_stds_list.extend([results['minor']['uniform_std'], results['ood']['uniform_std']])
        elif plot_name == 'uniform_dist':
            all_means_list.extend([results['minor']['uniform_dist_mean'], results['ood']['uniform_dist_mean']])
            all_stds_list.extend([results['minor']['uniform_dist_std'], results['ood']['uniform_dist_std']])
    
    all_means = np.concatenate(all_means_list)
    all_stds = np.concatenate(all_stds_list)
    y_min = (all_means - all_stds).min()
    y_max = (all_means + all_stds).max()
    y_margin = (y_max - y_min) * 0.05  # 5% margin
    y_lim = (y_min - y_margin, y_max + y_margin)
    
    # Create plots dynamically based on plot_list
    for idx, plot_name in enumerate(plot_list):
        ax = axes[idx]
        
        if plot_name == 'group':
            # GroupUniformKnownBayes (knows all tasks)
            ax.plot(positions, results['minor']['group_mean'],
                    label=f'Minor tasks (n={k_minor})', linewidth=2, marker='o', markersize=3, color='C0')
            ax.fill_between(positions,
                            results['minor']['group_mean'] - results['minor']['group_std'],
                            results['minor']['group_mean'] + results['minor']['group_std'],
                            alpha=0.2, color='C0')
            
            ax.plot(positions, results['ood']['group_mean'],
                    label=f'OOD tasks (n={n_ood})', linewidth=2, marker='s', markersize=3, color='C1')
            ax.fill_between(positions,
                            results['ood']['group_mean'] - results['ood']['group_std'],
                            results['ood']['group_mean'] + results['ood']['group_std'],
                            alpha=0.2, color='C1')
            
            ax.set_title('vs GroupUniformKnownBayes\\n(knows all tasks)', fontsize=13)
            
        elif plot_name == 'dirichlet':
            # ThreeKnownPlusNewDirichletBayes (knows 3, Dirichlet for rest)
            ax.plot(positions, results['minor']['dirichlet_mean'],
                    label=f'Minor tasks (n={k_minor})', linewidth=2, marker='o', markersize=3, color='C0')
            ax.fill_between(positions,
                            results['minor']['dirichlet_mean'] - results['minor']['dirichlet_std'],
                            results['minor']['dirichlet_mean'] + results['minor']['dirichlet_std'],
                            alpha=0.2, color='C0')
            
            ax.plot(positions, results['ood']['dirichlet_mean'],
                    label=f'OOD tasks (n={n_ood})', linewidth=2, marker='s', markersize=3, color='C1')
            ax.fill_between(positions,
                            results['ood']['dirichlet_mean'] - results['ood']['dirichlet_std'],
                            results['ood']['dirichlet_mean'] + results['ood']['dirichlet_std'],
                            alpha=0.2, color='C1')
            
            ax.set_title(f'vs ThreeKnownPlusNewDirichletBayes\\n(knows 3, Dirichlet α={alpha})', fontsize=13)
            
        elif plot_name == 'uniform_bayes':
            # ThreeKnownUniformBayes (knows 3, uniform for rest)
            ax.plot(positions, results['minor']['uniform_mean'],
                    label=f'Minor tasks (n={k_minor})', linewidth=2, marker='o', markersize=3, color='C0')
            ax.fill_between(positions,
                            results['minor']['uniform_mean'] - results['minor']['uniform_std'],
                            results['minor']['uniform_mean'] + results['minor']['uniform_std'],
                            alpha=0.2, color='C0')
            
            ax.plot(positions, results['ood']['uniform_mean'],
                    label=f'OOD tasks (n={n_ood})', linewidth=2, marker='s', markersize=3, color='C1')
            ax.fill_between(positions,
                            results['ood']['uniform_mean'] - results['ood']['uniform_std'],
                            results['ood']['uniform_mean'] + results['ood']['uniform_std'],
                            alpha=0.2, color='C1')
            
            ax.set_title('vs ThreeKnownUniformBayes\\n(knows 3)', fontsize=13)
            
        elif plot_name == 'uniform_dist':
            # Uniform Distribution
            ax.plot(positions, results['minor']['uniform_dist_mean'],
                    label=f'Minor tasks (n={k_minor})', linewidth=2, marker='o', markersize=3, color='C0')
            ax.fill_between(positions,
                            results['minor']['uniform_dist_mean'] - results['minor']['uniform_dist_std'],
                            results['minor']['uniform_dist_mean'] + results['minor']['uniform_dist_std'],
                            alpha=0.2, color='C0')
            
            ax.plot(positions, results['ood']['uniform_dist_mean'],
                    label=f'OOD tasks (n={n_ood})', linewidth=2, marker='s', markersize=3, color='C1')
            ax.fill_between(positions,
                            results['ood']['uniform_dist_mean'] - results['ood']['uniform_dist_std'],
                            results['ood']['uniform_dist_mean'] + results['ood']['uniform_dist_std'],
                            alpha=0.2, color='C1')
            
            ax.set_title('vs Uniform Distribution\\n(no task info)', fontsize=13)
        
        # Common formatting for all plots
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('KL Divergence', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_lim)
    
    plt.suptitle(f'KL Divergence (Original n_minor={original_n_minor}, Sampled: minor={k_minor}, OOD={n_ood})', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    if 'group' in plot_list:
        print("\nGroupUniformKnownBayes (knows all tasks):")
        print(f"  Minor: first 10 pos = {results['minor']['group_mean'][:10].mean():.4f}, "
              f"last 10 pos = {results['minor']['group_mean'][-10:].mean():.4f}")
        print(f"  OOD:   first 10 pos = {results['ood']['group_mean'][:10].mean():.4f}, "
              f"last 10 pos = {results['ood']['group_mean'][-10:].mean():.4f}")
    
    if 'dirichlet' in plot_list:
        print(f"\nThreeKnownPlusNewDirichletBayes (knows 3, α={alpha}):")
        print(f"  Minor: first 10 pos = {results['minor']['dirichlet_mean'][:10].mean():.4f}, "
              f"last 10 pos = {results['minor']['dirichlet_mean'][-10:].mean():.4f}")
        print(f"  OOD:   first 10 pos = {results['ood']['dirichlet_mean'][:10].mean():.4f}, "
              f"last 10 pos = {results['ood']['dirichlet_mean'][-10:].mean():.4f}")
    
    if 'uniform_bayes' in plot_list:
        print("\nThreeKnownUniformBayes (knows 3):")
        print(f"  Minor: first 10 pos = {results['minor']['uniform_mean'][:10].mean():.4f}, "
              f"last 10 pos = {results['minor']['uniform_mean'][-10:].mean():.4f}")
        print(f"  OOD:   first 10 pos = {results['ood']['uniform_mean'][:10].mean():.4f}, "
              f"last 10 pos = {results['ood']['uniform_mean'][-10:].mean():.4f}")
    
    if 'uniform_dist' in plot_list:
        print("\nUniform Distribution (no task info):")
        print(f"  Minor: first 10 pos = {results['minor']['uniform_dist_mean'][:10].mean():.4f}, "
              f"last 10 pos = {results['minor']['uniform_dist_mean'][-10:].mean():.4f}")
        print(f"  OOD:   first 10 pos = {results['ood']['uniform_dist_mean'][:10].mean():.4f}, "
              f"last 10 pos = {results['ood']['uniform_dist_mean'][-10:].mean():.4f}")
    
    return results

