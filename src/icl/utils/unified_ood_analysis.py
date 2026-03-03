import torch
from typing import Dict, Sequence
import pickle
import os

from icl.linear.linear_utils import estimate_lambda_with_r2
from icl.utils.unified_path_finder import _get_metrics_cache_path
from icl.utils.unified_interface import _get_hiddens
import icl.utils.notebook_utils as nu
from icl.linear.linear_path_utils import load_model_task_config
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)

def _compute_ood_and_minor_metrics(
    final_task_vecs,
    task_vecs_over_all_time,
    k_minor: int,
    is_zero_mean: bool = True,
    position: int = -1,
):
    """
    position : int or tuple
        Position index for R² / lambda. -1 = final position.
        Tuple (start, end) = average over positions [start, end) (end exclusive).
    """
    lambdas, r2_scores, _, _ = estimate_lambda_with_r2(
        final_task_vecs,
        task_vecs_over_all_time,
        is_zero_mean=is_zero_mean,
    )
    lambdas = torch.as_tensor(lambdas, dtype=torch.float32)
    r2_scores = torch.as_tensor(r2_scores, dtype=torch.float32)
    if isinstance(position, tuple):
        start, end = position
        r2_ood_final = r2_scores[3:-k_minor, start:end].mean(dim=1)
        r2_minor_final = r2_scores[-k_minor:, start:end].mean(dim=1)
        lambdas_ood_final = lambdas[3:-k_minor, start:end, :].mean(dim=1)
        lambdas_minor_final = lambdas[-k_minor:, start:end, :].mean(dim=1)
    else:
        r2_ood_final = r2_scores[3:-k_minor, position]
        r2_minor_final = r2_scores[-k_minor:, position]
        lambdas_ood_final = lambdas[3:-k_minor, position]
        lambdas_minor_final = lambdas[-k_minor:, position]
    r2_ood = float(r2_ood_final.mean())
    r2_min = float(r2_minor_final.mean())
    ood_var = (lambdas_ood_final - lambdas_ood_final.mean(dim=0, keepdim=True)).norm(dim=-1).mean().item()
    minor_var = (lambdas_minor_final - lambdas_minor_final.mean(dim=0, keepdim=True)).norm(dim=-1).mean().item()
    return r2_ood, r2_min, ood_var, minor_var


def _get_minor_task_vecs_at_positions(
    task_vecs_over_all_time,
    k_minor: int,
    position: int = -1,
):
    """
    Get minor task vectors at given position(s).
    position : int or tuple
        -1 = final position. Tuple (start, end) = average over [start, end).
    """
    if isinstance(position, tuple):
        start, end = position
        minor_raw = task_vecs_over_all_time[-k_minor:, start:end, :].mean(dim=1)
    else:
        minor_raw = task_vecs_over_all_time[-k_minor:, position]
    _, _, Vh = torch.linalg.svd(minor_raw, full_matrices=False)
    minor_task_vecs = Vh[:3, :]
    return minor_task_vecs


def _get_minor_final_task_vecs(task_vecs_over_all_time, k_minor: int):
    """Backward-compatible alias: minor task vectors at final position."""
    return _get_minor_task_vecs_at_positions(task_vecs_over_all_time, k_minor, position=-1)
    

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
