import torch
from torch import nn
from typing import Callable, Optional, Dict, List

from icl.linear.lr_task import Task
from icl.linear.lr_models import get_model_name

Preds = dict[str, dict[str, torch.Tensor]]


########################################################################################################################
# Utilities                                                                                                            #
########################################################################################################################

def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a - b) ** 2).mean(dim=0)




########################################################################################################################
# Evaluation                                                                                                           #
########################################################################################################################

# evaluate xs @ ws
def get_oracle_step(task) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def step(xs: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
        return task.evaluate_oracle(xs, ws)
    return step

def get_baseline_step(model: nn.Module) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def step(data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return model(data, targets)
    return step

def get_bsln_preds(
    train_task,  # Task
    batch_samplers: Dict[str, Callable[[int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    n_samples: int,
    batch_size: int
) -> Dict[str, Dict[str, torch.Tensor]]:
    preds = {}


    # Oracle prediction function (fixed for task)
    oracle_fn = get_oracle_step(train_task)

    # Baseline models
    baseline_models = {
        get_model_name(model): get_baseline_step(model)
        for model in train_task.get_default_eval_models()
    }

    for task_name, sample_batch_fn in batch_samplers.items():
        preds[task_name] = {"True": []}
        for model_name in baseline_models:
            preds[task_name][model_name] = []

        for i in range(1, n_samples // batch_size + 1):
            xs, ws, ys = sample_batch_fn(i)
            _, _, n_points = ys.shape

            # Oracle predictions
            true_preds = oracle_fn(xs, ws).reshape(batch_size, n_points)
            preds[task_name]["True"].append(true_preds)

            # Baseline model predictions
            for model_name, model_fn in baseline_models.items():
                pred = model_fn(xs, ys).reshape(batch_size, n_points)
                preds[task_name][model_name].append(pred)

        # Concatenate all collected predictions along batch axis
        preds[task_name]["True"] = torch.cat(preds[task_name]["True"], dim=0)
        for model_name in baseline_models:
            preds[task_name][model_name] = torch.cat(preds[task_name][model_name], dim=0)

    return preds

def get_model_preds(
    model,
    eval_step: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    batch_samplers: Dict[str, Callable[[int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    n_samples: int,
    batch_size: int
) -> Dict[str, Dict[str, torch.Tensor]]:
    preds = {}

    for task_name, sample_batch_fn in batch_samplers.items():
        preds[task_name] = {"Transformer": []}
        for i in range(1, n_samples // batch_size + 1):
            xs, _, ys = sample_batch_fn(i)
            _, _, n_points = ys.shape

            with torch.no_grad():
                pred = eval_step(model, xs, ys)  # shape: (batch_size, n_points)
                pred = pred.reshape(batch_size, n_points)

            preds[task_name]["Transformer"].append(pred)

        # Concatenate along the batch dimension
        preds[task_name]["Transformer"] = torch.cat(preds[task_name]["Transformer"], dim=0)

    return preds