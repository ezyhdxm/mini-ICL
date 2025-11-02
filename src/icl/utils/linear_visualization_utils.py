"""
Visualization and analysis functions for linear experiments.

This module contains plotting functions and analysis utilities extracted 
from notebooks for better organization.
"""

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from typing import Dict, List, Tuple, Optional, Any

from icl.linear.linear_utils import load_model_task_config, get_task_vector_from_hidden
from icl.linear.lr_task import get_task
from icl.linear.train_linear import get_sharded_batch_sampler
from icl.linear.lr_models import UnbalancedMMSE, MixedRidge


def plot_mse_vs_position(model, samplers_eval: Dict, bayes_ood, bayes_id, step: int = 1):
    """
    Plot MSE vs position with interactive dropdown for different modes.
    
    Args:
        model: Trained model
        samplers_eval: Dictionary of evaluation samplers
        bayes_ood: Out-of-distribution Bayes estimator
        bayes_id: In-distribution Bayes estimator
        step: Step number for sampling
    """
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

    # Dropdown menu
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


def plot_mse(model, train_task, config):
    """
    Setup and plot MSE analysis for a model.
    
    Args:
        model: Trained model
        train_task: Training task
        config: Configuration object
    """
    samplers_eval = {}
    is_eval = [True, False]
    k = 0
    for task in train_task.get_default_eval_tasks(**config["eval"]):
        from icl.linear.linear_utils import get_task_name
        samplers_eval[get_task_name(task)] = get_sharded_batch_sampler(task, is_eval[k])
        k += 1
    
    MRE = MixedRidge(
        config.task.noise_scale**2 / config.task.task_scale**2, 
        train_task.task_pool, 
        config.task.p_minor,
        config.task.noise_scale
    )
    UdMMSE = UnbalancedMMSE(
        config.task.noise_scale, 
        train_task.task_pool, 
        config.task.p_minor, 
        train_task.minor_pool
    )
    
    plot_mse_vs_position(model, samplers_eval, MRE, UdMMSE)


def sample_unit_vectors(n: int, d: int) -> torch.Tensor:
    """Sample n unit vectors in d dimensions."""
    x = torch.randn(n, d)           
    return x / x.norm(dim=1, keepdim=True)


def create_task_pool_and_groups(train_task, K: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create evaluation task pool with different perturbation groups.
    
    Args:
        train_task: Training task object
        K: Number of tasks per group
        d: Dimensionality
        
    Returns:
        Tuple of (eval_task_pool, groups)
    """
    base_task = train_task.task_pool[:1].squeeze(-1)  # Base task vector
    
    # Create task pools for each group with different perturbation magnitudes
    group_configs = [
        (0.5, 0),   # Group 0: small perturbation
        (2.0, 1),   # Group 1: large perturbation  
        (1.2, 2)    # Group 2: medium perturbation
    ]
    
    task_pools = []
    group_labels = []
    
    # Add perturbed tasks
    for magnitude, group_id in group_configs:
        perturbed_tasks = base_task + magnitude * sample_unit_vectors(K, d).to(base_task.device)
        task_pools.append(perturbed_tasks)
        group_labels.extend([group_id] * K)
    
    # Add original training tasks
    task_pools.append(train_task.task_pool[:2].squeeze(-1))
    group_labels.extend([3, 4])
    
    eval_task_pool = torch.cat(task_pools, dim=0)
    groups = torch.tensor(group_labels, dtype=torch.long)
    
    return eval_task_pool, groups


def plot_task_vector_norms(eval_task_vectors: torch.Tensor, groups: torch.Tensor) -> go.Figure:
    """
    Plot normalized task vector norms over positions.
    
    Args:
        eval_task_vectors: Task vectors tensor
        groups: Group labels tensor
        
    Returns:
        Plotly figure
    """
    # Normalize relative to the second-to-last task's final position
    reference = eval_task_vectors[-2:-1, -1:]
    ys = (eval_task_vectors - reference).norm(dim=-1)
    x = np.arange(1, ys.shape[1] + 1)
    
    # Create color mapping for groups
    unique_groups = sorted(set(groups.tolist()))
    group_to_color = {g: pc.qualitative.Set2[i % len(pc.qualitative.Set2)] 
                      for i, g in enumerate(unique_groups)}
    
    # Create plot
    fig = go.Figure()
    for k in range(ys.shape[0]):
        group_label = groups[k].item()
        fig.add_trace(go.Scatter(
            x=x,
            y=ys[k],
            mode='lines',
            name=f'Group {group_label}',
            line=dict(width=1, color=group_to_color[group_label])
        ))
    
    fig.update_layout(
        xaxis_title="Position",
        yaxis_title="Task Vector Norm",
        width=800,
        height=500,
        template="plotly_white"
    )
    
    return fig


def plot_task_norm(exp_name: str, config, train_task):
    """
    Plot task norm analysis for an experiment.
    
    Args:
        exp_name: Experiment name
        config: Configuration object
        train_task: Training task
    """
    model, _, _ = load_model_task_config(exp_name)
    # Main execution
    K = 5
    d = config.task.n_dims
    
    # Setup evaluation configuration
    eval_config = config.copy() if isinstance(config, dict) else config
    eval_config.task.n_tasks = K
    eval_task = get_task(**eval_config["task"])
    
    # Create task pool and group assignments
    eval_task_pool, groups = create_task_pool_and_groups(train_task, K, d)
    eval_task.task_pool = eval_task_pool.unsqueeze(-1) 
    eval_config.task.n_tasks = eval_task.task_pool.shape[0]
    
    # Compute and process task vectors
    _, eval_task_vectors = get_task_vector_from_hidden(
        eval_config, model, eval_task, layer_index=3, 
        compute_mean=True, return_final=False
    )  # shape (K, T, d)
    
    # Create and display plot
    eval_task_vectors = eval_task_vectors.cpu()
    fig = plot_task_vector_norms(eval_task_vectors, groups)
    fig.show()

    # Plot orthogonal components
    groups = groups.long()
    unique_groups = sorted(set(groups.tolist()))
    group_to_color = {g: pc.qualitative.Set2[i % len(pc.qualitative.Set2)] 
                      for i, g in enumerate(unique_groups)}

    A = eval_task_vectors.view(-1, 128)  
    # Compute projection matrix P = Bᵗ (B Bᵗ)⁻¹ B
    B = eval_task_vectors[-2:, -1:].squeeze(1).clone()  # (128, 2)
    proj = A @ (B.T @ torch.inverse(B @ B.T) @ B)                   
    residual = A - proj
    
    # Reshape back
    residual = residual.view(eval_task_vectors.shape)
    
    ys = residual.norm(dim=-1)
    x = np.arange(1, ys.shape[1] + 1)
    fig = go.Figure()
    for k in range(ys.shape[0]):
        group_label = groups[k].item()
        fig.add_trace(go.Scatter(
            x=x,
            y=ys[k],
            mode='lines',
            name=str(group_label),
            opacity=1,
            line=dict(width=1, color=group_to_color[group_label])
        ))
    fig.update_layout(
        xaxis_title="Position",
        width=800,
        height=500,
        template="plotly_white"
    )
    fig.show()