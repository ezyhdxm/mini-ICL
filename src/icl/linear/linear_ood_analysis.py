"""
Out-of-distribution task pool construction for linear regression experiments.
"""

import torch
from typing import Optional, Tuple

from icl.linear.lr_task import get_task
from icl.linear.processor_utils import setup_device


def _create_eval_task_pool(
    train_task,
    K: int,
    include_minor: bool = False,
    device: str = "cpu",
    n_minor: int = 256,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, int]:
    """Create evaluation task pool with OOD tasks and optional minority tasks.

    OOD tasks are sampled from N(0, task_scale^2 I).

    Returns
    -------
    eval_task_pool : Tensor  (M + K [+ n_minor], d)
    n_minor_sampled : int
    """
    anchor_pool = train_task.task_pool.squeeze(-1).to(device)
    M, n_dims = anchor_pool.shape

    ood_tasks = torch.randn(
        K,
        n_dims,
        device=device,
        dtype=anchor_pool.dtype,
        generator=generator,
    ) * train_task.task_scale
    eval_task_pool = torch.cat([anchor_pool, ood_tasks], dim=0)

    n_minor_sampled = 0
    if include_minor:
        minor_pool = train_task.minor_pool.squeeze(-1).to(device)

        if train_task.n_minor_tasks > n_minor:
            indices = torch.randperm(
                train_task.n_minor_tasks,
                device=device,
                generator=generator,
            )[:n_minor]
            minor_pool = minor_pool[indices]
            n_minor_sampled = n_minor
        else:
            n_minor_sampled = train_task.n_minor_tasks

        eval_task_pool = torch.cat([eval_task_pool, minor_pool], dim=0)

    return eval_task_pool, n_minor_sampled


def _setup_eval_task(config, eval_task_pool: torch.Tensor, batch_size: int, device: str):
    """Setup evaluation task with the given task pool."""
    K = eval_task_pool.shape[0]

    eval_config = config.copy() if isinstance(config, dict) else config
    eval_config["task"].n_tasks = K
    eval_config["task"].pool_type = None
    eval_config["device"] = device

    eval_task = get_task(**eval_config["task"], device=device)
    eval_task.batch_size = batch_size
    eval_task.task_pool = eval_task_pool.unsqueeze(-1)  # (K, d, 1)

    return eval_task
