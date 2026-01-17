import torch
from typing import Optional, Dict, Sequence
import pickle
import os
import json
import plotly.graph_objects as go
import numpy as np

from icl.linear.linear_utils import estimate_lambda_with_r2
from icl.utils.unified_path_finder import unified_get_config, _get_metrics_cache_path
from icl.utils.unified_interface import _get_hiddens, get_exp_name
import icl.utils.notebook_utils as nu
from icl.linear.linear_path_utils import load_model_task_config
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)

def _compute_ood_and_minor_metrics(
    final_task_vecs,
    task_vecs_over_all_time,
    k_minor: int,
    is_zero_mean: bool = True,
):
    lambdas, r2_scores, _, _ = estimate_lambda_with_r2(
        final_task_vecs, 
        task_vecs_over_all_time,
        is_zero_mean=is_zero_mean
        )
    lambdas = torch.as_tensor(lambdas, dtype=torch.float32)
    r2_scores = torch.as_tensor(r2_scores, dtype=torch.float32)
    r2_ood_final = r2_scores[3:-k_minor, -1]      # (K_ood,)
    r2_minor_final = r2_scores[-k_minor:, -1]   # (K_minor,)
    lambdas_ood_final = lambdas[3:-k_minor, -1]
    lambdas_minor_final = lambdas[-k_minor:, -1]
    r2_ood = float(r2_ood_final.mean())
    r2_min = float(r2_minor_final.mean())
    ood_var = (lambdas_ood_final - lambdas_ood_final.mean(dim=0, keepdim=True)).norm(dim=-1).mean().item()
    minor_var = (lambdas_minor_final - lambdas_minor_final.mean(dim=0, keepdim=True)).norm(dim=-1).mean().item()
    
    return r2_ood, r2_min, ood_var, minor_var

def _get_minor_final_task_vecs(
    task_vecs_over_all_time,
    k_minor: int,
):
    minor_final_task_vecs_raw = task_vecs_over_all_time[-k_minor:, -1]
    # compute the svd of minor_final_task_vecs and take the top 3 singular vectors
    _, _, Vh = torch.linalg.svd(minor_final_task_vecs_raw, full_matrices=False) # Vh: shape (k_minor, D)
    minor_final_task_vecs = Vh[:3, :]  # (3, D)
    return minor_final_task_vecs
    

def _compute_metrics(
    hiddens: torch.Tensor,
    k_minor: int,
):
    task_mean = hiddens[:3].mean(dim=(0, 2)).unsqueeze(0)
    task_vecs_over_all_time = hiddens.mean(dim=-2) - task_mean
    maj_final_task_vecs = (hiddens[:3].mean(dim=-2) - task_mean)[:, -1]
    min_final_task_vecs = _get_minor_final_task_vecs(
        task_vecs_over_all_time,
        k_minor,
    )

    maj_r2_ood, maj_r2_min, maj_ood_var, maj_minor_var = _compute_ood_and_minor_metrics(
        maj_final_task_vecs,
        task_vecs_over_all_time,
        k_minor,
    )

    min_r2_ood, min_r2_min, min_ood_var, min_minor_var = _compute_ood_and_minor_metrics(
        min_final_task_vecs,
        task_vecs_over_all_time,
        k_minor,
        is_zero_mean=False,
    )

    metrics_dict = {
        "maj_r2_ood": maj_r2_ood,
        "maj_r2_min": maj_r2_min,
        "maj_ood_var": maj_ood_var,
        "maj_minor_var": maj_minor_var,
        "min_r2_ood": min_r2_ood,
        "min_r2_min": min_r2_min,
        "min_ood_var": min_ood_var,
        "min_minor_var": min_minor_var,
    }
    return metrics_dict


def _set_layer_indices(
    config,
):
    if config.task.name == "noisy_linear_regression":
        layer_indices = list(range(config.model.n_layer))
    else:
        layer_indices = list(range(config.model.num_layers))
    return layer_indices

def process_ood_minor_metric(
    task_name: str,
    exp_name: str,
    steps: Sequence[int],
    n_minor: int = 64,
    n_ood: int = 30,
    B: int = 64,
    force_recompute=False,
):
    # Setup
    if task_name == "linear":
        _, train_task, config = load_model_task_config(exp_name)
        k_minor = min(n_minor, train_task.n_minor_tasks)
    else:
        _, sampler, config = nu.load_everything(task_name, exp_name)
        k_minor = min(n_minor, sampler.n_minor_tasks)
    layer_indices = _set_layer_indices(config)
    cache_paths = _get_metrics_cache_path(
        config,
        task_name,
        exp_name,
        k_minor,
        n_ood,
        B,
        steps,
        layer_indices,
        )

    # Storage
    maj_r2_ood: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    maj_r2_min: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    min_r2_ood: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    min_r2_min: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    maj_ood_var: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    maj_min_var: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    min_ood_var: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    min_min_var: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}   
    processed_steps = []

    try:
        for step in steps:
            flag = True
            for L in layer_indices:
                if os.path.exists(cache_paths[step][L]) and not force_recompute:
                    with open(cache_paths[step][L], "rb") as f:
                        metrics_dict = pickle.load(f)
                    maj_r2_ood[L][step] = metrics_dict["maj_r2_ood"]
                    maj_r2_min[L][step] = metrics_dict["maj_r2_min"]
                    min_r2_ood[L][step] = metrics_dict["min_r2_ood"]
                    min_r2_min[L][step] = metrics_dict["min_r2_min"]
                    maj_ood_var[L][step] = metrics_dict["maj_ood_var"]
                    maj_min_var[L][step] = metrics_dict["maj_minor_var"]
                    min_ood_var[L][step] = metrics_dict["min_ood_var"]
                    min_min_var[L][step] = metrics_dict["min_minor_var"]
                else:
                    flag = False
                    break
            if not flag:
                hiddens, _ = _get_hiddens(
                    task_name,
                    exp_name,
                    n_minor,
                    n_ood,
                    B,
                    step=step,
                )
                if task_name == "latent":
                    L, K, V, T, B, D = hiddens.shape
                    hiddens = hiddens.permute(0, 1, 3, 4, 5, 2).reshape(L, K, T, B, D*V)

                for L in layer_indices:
                    metrics_dict = _compute_metrics(
                        hiddens[L].to(torch.float32),
                        k_minor,
                    )
                    pickle.dump(metrics_dict, open(cache_paths[step][L], "wb"))
                    maj_r2_ood[L][step] = metrics_dict["maj_r2_ood"]
                    maj_r2_min[L][step] = metrics_dict["maj_r2_min"]
                    min_r2_ood[L][step] = metrics_dict["min_r2_ood"]
                    min_r2_min[L][step] = metrics_dict["min_r2_min"]
                    maj_ood_var[L][step] = metrics_dict["maj_ood_var"]
                    maj_min_var[L][step] = metrics_dict["maj_minor_var"]
                    min_ood_var[L][step] = metrics_dict["min_ood_var"]
                    min_min_var[L][step] = metrics_dict["min_minor_var"]
            processed_steps.append(step)

    except KeyboardInterrupt:
        print(f"\nInterrupted. Processed {len(processed_steps)} checkpoints so far.")

    if not processed_steps:
        print("No checkpoints processed successfully.")
        return {}
    
    results_dict = {
        "steps": processed_steps,
        "layers": layer_indices,
        "maj_r2_ood": maj_r2_ood,
        "maj_r2_min": maj_r2_min,
        "min_r2_ood": min_r2_ood,
        "min_r2_min": min_r2_min,
        "maj_ood_var": maj_ood_var,
        "maj_min_var": maj_min_var,
        "min_ood_var": min_ood_var,
        "min_min_var": min_min_var,
    }

    return results_dict



def plot_training_curves_all_experiments(
        task_name,
        steps,
        k_list, 
        n_minor: int = 64,
        n_ood: int = 30, 
        B: int = 64,
        metric="maj_r2_ood",
        layer_index=None,
        show_first_plot=True,
        show_second_plot=True,
        verbose=False,
        vocab_size=8,
        ):
    """
    Unified function to plot OOD metrics and/or Transformer|True MSE across all experiments.
    Each experiment (different k/n_minor_tasks) gets its own line.
    
    Args:
        exp_names_with_k: List of tuples (k, exp_name) where n_minor_tasks = 2^k
        include_ood_metric: If True, plot OOD metric (R² or λ dispersion)
        metric: "maj_r2_ood", "maj_r2_min", "min_r2_ood", "min_r2_min", "maj_ood_var", "maj_min_var", "min_ood_var", "min_min_var" 
        layer_index: Which layer to extract from results_dict
        K: Number of batches for process_ood_evolve_checkpoints
        include_transformer_mse: If True, plot Transformer|True MSE
        eval_key: One of "Latent_false", "Latent_true", "Pretrain_false", "Pretrain_true"
        combine_plots: If True and both metrics included, plot on same figure with dual y-axes
    """
    results = {}
    if layer_index is None:
        layer_index = 15 if task_name == "linear" else 5
    
    # Collect data for all experiments
    for k in k_list:
        if task_name == "coin":
            exp_name = get_exp_name(task_name, k, vocab_size=vocab_size)
        else:
            exp_name = get_exp_name(task_name, k)
        n_minor_tasks = 2**k
        
        try:
            results_dict = process_ood_minor_metric(
                task_name,
                exp_name,
                steps,
                n_minor,
                n_ood,
                B,
                force_recompute=False,
            )
            layer_metric = results_dict[metric][layer_index]
            ood_steps = sorted(layer_metric.keys())
            ood_values = [layer_metric[step] for step in ood_steps]
            
            # Get Transformer MSE if requested
            if task_name == "linear":
                log_path = f"../results/linear/{exp_name}/log.json"
                with open(log_path) as f:
                    data = json.load(f)
                
                train_steps = data['train/step']
                eval_data = data['eval/Latent_false']
                transformer_true = eval_data['Transformer | True']
                ood_loss = [np.mean(values) for values in transformer_true]
            else:
                log_path = f"../results/{task_name}/{exp_name}/log.json"
                with open(log_path) as f:
                    data = json.load(f)
                
                train_steps = data['eval/step']
                ood_loss = data['eval/OODLoss']
            
            results[k] = {
                'n_minor': n_minor_tasks,
                'ood_steps': ood_steps,
                'ood_values': ood_values,
                'train_steps': train_steps,
                'ood_loss': ood_loss
            }
            if verbose:
                logger.info(f"Prepared data for plotting for experiment: {exp_name}")
            
        except Exception as e:
            print(f"Warning: Could not process k={k}, exp={exp_name}: {e}")
    
    
    figs = []
    fig1 = go.Figure()
    
    for k, data in results.items():
        fig1.add_trace(go.Scatter(
            x=data['ood_steps'],
            y=data['ood_values'],
            mode='lines+markers',
            name=f'k={k} (number of minority={data["n_minor"]})',
            marker=dict(size=6),
            line=dict(width=2)
        ))
    
    metric_title = "Metric"
    
    fig1.update_layout(
        title=f'{metric_title} Across Training Steps<br>Layer {layer_index}, All Experiments',
        xaxis_title='Training Step',
        yaxis_title=metric_title,
        hovermode='x unified',
        template='plotly_white',
        width=1200,
        height=600,
        legend_title='Experiment'
    )
    figs.append(fig1)
    
    fig2 = go.Figure()
    
    for k, data in results.items():
        train_steps = np.asarray(data['train_steps'])
        ood_loss    = np.asarray(data['ood_loss'])
        ood_steps   = np.asarray(data['ood_steps'])

        lo, hi = ood_steps.min(), ood_steps.max()

        l = np.searchsorted(train_steps, lo, side="left")
        r = np.searchsorted(train_steps, hi, side="right")

        data['train_steps'] = train_steps[l:r]
        data['ood_loss']    = ood_loss[l:r]

        fig2.add_trace(go.Scatter(
            x=data['train_steps'],
            y=data['ood_loss'],
            mode='lines',
            name=f'k={k} (number of minority={data["n_minor"]})',
            line=dict(width=2)
        ))
    
    fig2.update_layout(
        title=f'OOD Loss Across Training Steps<br> All Experiments',
        xaxis_title='Training Step',
        yaxis_title='OOD Loss',
        hovermode='x unified',
        template='plotly_white',
        width=1200,
        height=600,
        legend_title='Experiment'
    )
    figs.append(fig2)

    if show_second_plot:
        figs[1].show()  # OOD Loss vs Training Steps
    if show_first_plot:
        figs[0].show()  # OOD Metric vs Training Steps
    
    return figs

