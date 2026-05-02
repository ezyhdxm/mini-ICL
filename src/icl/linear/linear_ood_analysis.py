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
    ood_far_oversample: int = 1,
) -> Tuple[torch.Tensor, int]:
    """Create evaluation task pool with OOD tasks and optional minority tasks.

    OOD tasks are sampled from N(0, task_scale^2 I).

    When ``ood_far_oversample > 1``, ``ood_far_oversample * K`` candidate OOD
    weights are drawn and the ``K`` with the **smallest** projected variance
    fraction onto the majority-task span (anchor_pool) are kept. This biases
    the OOD sample toward weights that are far from the majority subspace,
    which is useful for figures that want to emphasise the orthogonality of
    OOD vs majority hidden states (the projected R^2 stays small).

    Returns
    -------
    eval_task_pool : Tensor  (M + K [+ n_minor], d)
    n_minor_sampled : int
    """
    anchor_pool = train_task.task_pool.squeeze(-1).to(device)
    M, n_dims = anchor_pool.shape

    n_cand = max(1, int(ood_far_oversample)) * int(K)
    ood_cand = torch.randn(
        n_cand,
        n_dims,
        device=device,
        dtype=anchor_pool.dtype,
        generator=generator,
    ) * train_task.task_scale

    if n_cand == K:
        ood_tasks = ood_cand
    else:
        # Orthogonal projector onto span(anchor_pool) using the pseudo-inverse,
        # which handles the (typical) M < n_dims rank-deficient case cleanly.
        # P_S = A^T (A A^T)^+ A    where A = anchor_pool of shape (M, d).
        A = anchor_pool
        AAt = A @ A.T  # (M, M)
        AAt_pinv = torch.linalg.pinv(AAt)
        P = A.T @ AAt_pinv @ A  # (d, d), idempotent

        proj = ood_cand @ P.T  # (n_cand, d)
        proj_sq = proj.pow(2).sum(dim=-1)
        full_sq = ood_cand.pow(2).sum(dim=-1).clamp_min(1e-12)
        proj_frac = proj_sq / full_sq  # in [0, 1]; lower = farther from span(A)

        keep_idx = torch.argsort(proj_frac, descending=False)[:K]
        ood_tasks = ood_cand[keep_idx]

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
