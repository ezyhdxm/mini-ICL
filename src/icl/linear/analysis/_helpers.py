"""Shared helpers for the linear analysis package."""

from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch


def _show_or_close(fig, show: bool, tight: bool = True):
    """Optionally call ``tight_layout``, then show or close a figure."""
    import matplotlib.pyplot as plt
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    if show:
        plt.show()
    else:
        plt.close(fig)


def _task_positions(pad_mode: str, n_points: int, device="cpu"):
    """Return a tensor of sequence positions for each data point given *pad_mode*."""
    t = torch.arange(n_points, device=device)
    if pad_mode == "mapsto":
        return 3 * t
    elif pad_mode == "none":
        return 2 * t
    elif pad_mode == "bos":
        return 2 * t + 1
    raise ValueError(f"Unknown pad mode: {pad_mode}")


def _project_onto_simplex(v):
    """Project rows of *v* onto the probability simplex."""
    v = np.asarray(v, dtype=float)
    squeeze = v.ndim == 1
    if squeeze:
        v = v[np.newaxis, :]
    n = v.shape[1]
    u = np.sort(v, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - 1.0
    rho = np.array([
        np.where(u[i] - cssv[i] / np.arange(1, n + 1) > 0)[0][-1]
        for i in range(v.shape[0])
    ])
    theta = cssv[np.arange(v.shape[0]), rho] / (rho + 1.0)
    w_out = np.maximum(v - theta[:, np.newaxis], 0.0)
    return w_out[0] if squeeze else w_out


@contextmanager
def _temporary_task_attributes(task, **overrides):
    """Temporarily set task attributes and restore them afterward."""
    saved = {name: getattr(task, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(task, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(task, name, value)


@contextmanager
def _temporary_linear_minor_task_setup(
    train_task,
    *,
    n_minor: Optional[int],
    uniform_sampling: bool,
    sample_mode: str,
):
    """
    Temporarily override minor-task configuration for linear analysis routines.

    Yields
    ------
    tuple[bool, int]
        ``(include_minor, n_tasks_total)`` under the temporary configuration.
    """
    original_minor_pool = train_task.minor_pool
    original_n_minor_tasks = int(train_task.n_minor_tasks)
    original_p_minor = float(train_task.p_minor)

    if n_minor is None:
        n_minor_effective = original_n_minor_tasks
    elif n_minor == -1:
        n_minor_effective = 0
    else:
        n_minor_effective = max(0, int(n_minor))

    try:
        if original_n_minor_tasks > 0:
            if n_minor_effective == 0:
                train_task.minor_pool = None
                train_task.n_minor_tasks = 0
                train_task.p_minor = 0.0
            elif n_minor_effective < original_n_minor_tasks:
                idx = torch.randperm(original_n_minor_tasks)[:n_minor_effective]
                train_task.minor_pool = original_minor_pool[idx]
                train_task.n_minor_tasks = n_minor_effective

        if uniform_sampling and sample_mode == "train" and train_task.n_minor_tasks > 0:
            train_task.p_minor = train_task.n_minor_tasks / (train_task.n_tasks + train_task.n_minor_tasks)

        include_minor = train_task.n_minor_tasks > 0 and train_task.minor_pool is not None
        n_tasks_total = train_task.n_tasks + (train_task.n_minor_tasks if include_minor else 0)
        yield include_minor, n_tasks_total, original_p_minor
    finally:
        train_task.minor_pool = original_minor_pool
        train_task.n_minor_tasks = original_n_minor_tasks
        train_task.p_minor = original_p_minor
