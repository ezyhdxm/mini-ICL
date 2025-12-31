"""
Notebook-specific utility functions for linear experiments.

This module contains helper functions extracted from notebooks to improve
code organization and reusability.
"""

import torch
import os
import numpy as np
import plotly.graph_objects as go
from typing import Optional, Dict, List, Tuple

from icl.linear.linear_utils import load_model_task_config
from icl.linear.lr_task import get_task
from icl.linear.sufficient_stats import get_sufficient_statistics_proj_fit, get_betahat_fit
from icl.utils.processor_utils import setup_device


def process_sufficient_statistics(exp_name: str, device: Optional[str] = None):
    """
    Process sufficient statistics for a given experiment.
    
    Args:
        exp_name: Name of the experiment
        device: Device to use ('cuda' or 'cpu'). If None, auto-detected.
        
    Returns:
        Results from sufficient statistics computation
    """
    print("Loading...")
    
    # Device setup
    device = setup_device(device)
    
    # Load configuration and setup
    model, train_task, config = load_model_task_config(exp_name)

    print("Computing Sufficient Statistics and Fitting...")

    results = get_sufficient_statistics_proj_fit(
        config,
        model,
        train_task,
    )
    return results


def get_eval_task(config, device: str, train_task, K: int):
    """
    Create evaluation task with random task pool.
    
    Args:
        config: Experiment configuration
        device: Device to use
        train_task: Training task
        K: Number of evaluation tasks
        
    Returns:
        Evaluation task object
    """
    anchor = train_task.task_pool.squeeze(-1).to(device)
    eval_task_pool = torch.randn(K, config.task.n_dims, device=device)
    eval_task_pool = torch.cat([anchor, eval_task_pool], dim=0)
    
    # Setup evaluation task
    eval_config = config.copy() if isinstance(config, dict) else config
    eval_config.task.n_tasks = K + 3
    eval_config.device = device 
    
    eval_task = get_task(**eval_config["task"], device=device)
    eval_task.task_pool = eval_task_pool.unsqueeze(-1)  # shape (K+3, d, 1)
    return eval_task


def process_beta_fit(exp_name: str, task_layer_index: int = 3, device: Optional[str] = None, 
                    is_eval: bool = False, K: int = 10):
    """
    Process beta fitting for OLS estimation.
    
    Args:
        exp_name: Name of the experiment
        task_layer_index: Layer index for task vector extraction
        device: Device to use
        is_eval: Whether to use evaluation task
        K: Number of tasks for evaluation
        
    Returns:
        Results from beta fitting computation
    """
    # Device setup
    device = setup_device(device)
    
    # Load configuration and setup
    model, train_task, config = load_model_task_config(exp_name)

    if is_eval:
        task = get_eval_task(config, device, train_task, K)
    else:
        task = train_task

    # Compute OLS and fitting
    results = get_betahat_fit(
        config,
        model,
        task,
        task_layer_index
    )
    return results


def plot_metrics(results_dict: Dict):
    """
    Plot various metrics from results dictionary.
    
    Args:
        results_dict: Dictionary containing metrics data
    """
    # Extract keys and values for plotting
    x_values = list(results_dict["part_metrics"].keys())
    y_part_values = list(results_dict["part_metrics"].values())
    y_unif_values = list(results_dict["unif_metrics"].values())
    y_weight_values = list(results_dict["weight_metrics"].values())
    y_orthogonal_values = np.mean(results_dict["orthogonal_results"], axis=-1)
    y_norm_values = np.mean(results_dict["task_vec_norm_results"], axis=-1)
    
    # Create the plot
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=x_values, y=y_part_values, 
        mode='lines+markers', name="partition", 
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=x_values, y=y_weight_values, 
        mode='lines+markers', name="weights", 
        line=dict(color='green')
    ))
    
    fig.add_trace(go.Scatter(
        x=x_values, y=y_orthogonal_values, 
        mode='lines+markers', name="orthogonal", 
        line=dict(color='orange')
    ))
    
    fig.add_trace(go.Scatter(
        x=x_values, y=y_norm_values, 
        mode='lines+markers', name="norm", 
        line=dict(color='pink')
    ))
    
    # Update layout
    fig.update_layout(
        title='Plot of Values Across Steps',
        xaxis_title='Step',
        yaxis_title='Value',
        legend_title='Legend',
        template='plotly_white'
    )
    
    fig.show()


def plot_r2_curves_plotly(
    process_beta_fit_func,
    exp_name: str,
    layer_indices: List[int],
    is_eval: bool = True,
    K: int = 1024,
    title: Optional[str] = None,
    save_html: Optional[str] = None,
) -> Tuple[go.Figure, Dict]:
    """
    Plot R² curves for different layers.
    
    Args:
        process_beta_fit_func: Function to process beta fitting
        exp_name: Experiment name
        layer_indices: List of layer indices to plot
        is_eval: Whether to use evaluation mode
        K: Number of tasks
        title: Plot title
        save_html: Path to save HTML file
        
    Returns:
        Tuple of (figure, r2_dict)
    """
    r2_dict = {}
    for li in layer_indices:
        out = process_beta_fit_func(exp_name, task_layer_index=int(li), is_eval=is_eval, K=K)
        r2 = out["r2"]
        r2 = r2.detach().cpu().numpy() if isinstance(r2, torch.Tensor) else np.asarray(r2)
        r2_dict[int(li)] = r2

    fig = go.Figure()
    for li in sorted(r2_dict.keys()):
        r2 = r2_dict[li]
        x = np.arange(1, len(r2) + 1)
        fig.add_trace(go.Scatter(
            x=x, y=r2,
            mode="lines+markers",
            name=f"Layer {li}",
            hovertemplate="t=%{x}<br>R²=%{y:.3f}<extra></extra>",
        ))

    fig.update_layout(
        title=title or f"R² curves per layer (exp: {exp_name})",
        xaxis_title="t (index)",
        yaxis_title="R²",
        hovermode="x unified",
        legend_title="layer_index",
        template="plotly_white",
        xaxis=dict(zeroline=False),
        yaxis=dict(zeroline=False, rangemode="tozero"),
    )

    if save_html is not None:
        fig.write_html(save_html, include_plotlyjs="cdn")

    return fig, r2_dict