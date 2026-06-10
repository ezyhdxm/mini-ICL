"""Unified training interface for all task types (linear, coin, latent, dyck)."""

import torch
from typing import Optional

from icl.models.base_models import Transformer
from icl.models.factory import build_model
from icl.utils.train import train_model_with_plot
from icl.linear.train_linear import train
from icl.utils.unified_path_finder import unified_get_config
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)


def unified_train(
    task_name,
    k: int,
    vocab_size: int = 8,
    log2: bool = True,
    pad = None,
    major_pool_type: str = None,
    major_means = None,
    major_seed: int = None,
    total_steps: int = None,
    num_epochs: int = None,
    warmup_steps: int = None,
    lr: float = None,
    schedule: str = None,
    max_grad_norm: float = None,
    batch_size: int = None,
    noise_scale: float = None,
    p_minor: float = None,
    n_layer: int = None,
    n_points: int = None,
    min_lr: float = None,
    decay_power: float = None,
    batch_size_schedule: list = None,
    p_minor_schedule: list = None,
    final_layernorm: bool = None,
    quiet: bool = True,
    device: Optional[str] = None,
    n_tasks: Optional[int] = None,
    n_minor_tasks: Optional[int] = None,
    arch: Optional[str] = None,
):
    import os
    _prev_wandb_silent = os.environ.get("WANDB_SILENT")
    if quiet:
        os.environ["WANDB_SILENT"] = "true"

    try:
        config = unified_get_config(task_name)
        if device is not None:
            config.device = device
        if k >= 0:
            if log2:
                config.task.n_minor_tasks = 2 ** k
            else:
                config.task.n_minor_tasks = k
        else:
            config.task.n_minor_tasks = 1
            config.task.p_minor = 1e-12
        if n_minor_tasks is not None:
            config.task.n_minor_tasks = int(n_minor_tasks)
        if n_tasks is not None:
            config.task.n_tasks = int(n_tasks)
        if arch is not None:
            config.model.arch = arch
        if pad is not None:
            if task_name == "linear":
                config.model.pad = pad
        if major_pool_type is not None:
            config.task.major_pool_type = major_pool_type
        if major_means is not None:
            config.task.major_means = list(major_means)
        if major_seed is not None:
            config.task.major_seed = major_seed
        if num_epochs is not None:
            if task_name == "linear":
                config.training.total_steps = num_epochs
            else:
                config.training.num_epochs = num_epochs
        elif total_steps is not None:
            if task_name == "linear":
                config.training.total_steps = total_steps
            else:
                config.training.num_epochs = total_steps
        if warmup_steps is not None:
            config.training.warmup_steps = warmup_steps
        if lr is not None:
            config.training.lr = lr
        if schedule is not None:
            config.training.schedule = schedule
        if max_grad_norm is not None:
            config.training.max_grad_norm = max_grad_norm
        if batch_size is not None:
            config.task.batch_size = batch_size
        if noise_scale is not None:
            config.task.noise_scale = noise_scale
        if p_minor is not None:
            config.task.p_minor = p_minor
        if n_layer is not None:
            config.model.n_layer = n_layer
        if n_points is not None:
            if task_name == "linear":
                config.task.n_points = n_points
                config.model.n_points = n_points
        if min_lr is not None:
            config.training.min_lr = min_lr
        if decay_power is not None:
            config.training.decay_power = decay_power
        if batch_size_schedule is not None:
            config.training.batch_size_schedule = list(batch_size_schedule)
        if p_minor_schedule is not None:
            config.training.p_minor_schedule = list(p_minor_schedule)
        if final_layernorm is not None:
            if task_name == "linear":
                config.model.final_layernorm = final_layernorm
        if task_name == "linear":
            return train(config)
        else:
            config.vocab_size = vocab_size
            model = build_model(config)
            model = model.to(config.device)
            return train_model_with_plot(model, config, show=False, verbose=False)
    finally:
        if _prev_wandb_silent is None:
            os.environ.pop("WANDB_SILENT", None)
        else:
            os.environ["WANDB_SILENT"] = _prev_wandb_silent


def _to_cpu(obj):
    """Recursively move all tensors in a nested structure to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_to_cpu(v) for v in obj]
        return type(obj)(converted)
    if isinstance(obj, torch.nn.Module):
        return obj.cpu()
    return obj


_worker_device: Optional[str] = None


def _init_worker(gpu_queue):
    """Initializer for each worker process: claim a fixed GPU from the queue."""
    global _worker_device
    _worker_device = gpu_queue.get()
    if _worker_device.startswith("cuda"):
        torch.cuda.set_device(_worker_device)


def _unified_train_worker(args):
    """Picklable worker for ProcessPoolExecutor: run one unified_train on a given device."""
    task_name, k, kwargs = args
    verbose = kwargs.pop("_parallel_verbose", False)
    device = _worker_device
    if verbose:
        print(f"[unified_train_parallel] Training k={k} ({task_name}) on {device} ...", flush=True)
    result = unified_train(task_name, k, device=device, **kwargs)
    return _to_cpu(result)


def unified_train_parallel(
    task_name: str,
    k_list: list,
    n_gpus: Optional[int] = None,
    verbose: bool = False,
    **kwargs,
) -> list:
    """Run multiple k experiments in parallel across GPUs.

    Each worker process is bound to a fixed GPU via an initializer, so
    tasks never contend for the same device regardless of completion order.

    Parameters
    ----------
    task_name : str
        Same as unified_train (e.g. "linear", "coin", "latent", "dyck").
    k_list : list of int
        List of k values to train (e.g. [0, 1, 2, 3]).
    n_gpus : int, optional
        Number of GPUs to use. Default: min(2, torch.cuda.device_count()) or 1.
    verbose : bool, default False
        If True, log which k is being trained and when each run completes.
    **kwargs
        Passed through to unified_train for each run (e.g. total_steps, quiet, pad).

    Returns
    -------
    list
        Results from unified_train for each k, in the same order as k_list.

    Raises
    ------
    RuntimeError
        If any experiment fails (collects all errors and raises once).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_workers = n_gpus if n_gpus is not None else min(2, n_available or 1)
    n_workers = min(n_workers, len(k_list))
    if n_workers <= 0:
        n_workers = 1
    use_gpu = n_available > 0

    kwargs = {k: v for k, v in kwargs.items() if k != "device"}
    worker_kwargs = {**kwargs, "_parallel_verbose": verbose}

    def _device(i: int) -> str:
        return f"cuda:{i % n_available}" if use_gpu else "cpu"

    args_list = [
        (task_name, k, {**worker_kwargs})
        for k in k_list
    ]

    if n_workers == 1:
        out = []
        for k in k_list:
            if verbose:
                print(f"[unified_train_parallel] Training k={k} ({task_name}) on {_device(0)} ...", flush=True)
            out.append(unified_train(task_name, k, device=_device(0), **kwargs))
        return out

    if verbose:
        print(
            f"[unified_train_parallel] Starting parallel training for {task_name} "
            f"k_list={k_list} (n_workers={n_workers})",
            flush=True,
        )

    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    gpu_queue = ctx.Queue()
    for i in range(n_workers):
        gpu_queue.put(_device(i))

    results = [None] * len(k_list)
    errors = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(gpu_queue,),
    ) as ex:
        future_to_idx = {ex.submit(_unified_train_worker, args): i for i, args in enumerate(args_list)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            k = k_list[idx]
            try:
                results[idx] = future.result()
                if verbose:
                    print(f"[unified_train_parallel] Completed k={k}", flush=True)
            except Exception as e:
                logger.exception("unified_train_parallel failed for k=%s: %s", k, e)
                errors.append((k, e))

    if errors:
        msg = "; ".join(f"k={k}: {e}" for k, e in errors)
        raise RuntimeError(f"unified_train_parallel: {len(errors)} experiment(s) failed — {msg}")

    return results


def _major_only_train_worker(args):
    """Run one major-only experiment: same minor setup as ``k=-1`` (dummy minor + tiny p_minor)."""
    task_name, n_major, kwargs = args
    verbose = kwargs.pop("_parallel_verbose", False)
    device = _worker_device
    if verbose:
        print(
            f"[major_only] Training {task_name} n_major={n_major} on {device} ...",
            flush=True,
        )
    result = unified_train(
        task_name,
        0,
        device=device,
        n_tasks=n_major,
        n_minor_tasks=1,
        p_minor=1e-12,
        **kwargs,
    )
    return _to_cpu(result)


def unified_train_major_only_parallel(
    task_names: list,
    n_major_list: list,
    n_gpus: Optional[int] = None,
    verbose: bool = False,
    **train_kwargs,
) -> list:
    """Train (task, n_major) pairs with effectively major-only sampling.

    For each ``task_name`` in ``task_names`` and each ``n_major`` in
    ``n_major_list``, runs ``unified_train`` with ``n_tasks=n_major``,
    ``n_minor_tasks=1``, and ``p_minor=1e-12`` — the same minor configuration
    as ``k=-1`` in :func:`unified_train` (one dummy minor task, negligible
    sampling weight). Order of results matches
    nested iteration: all ``n_major`` values for task_names[0], then
    task_names[1], etc.

    Parameters
    ----------
    task_names : list of str
        e.g. ``["latent", "coin", "linear"]``.
    n_major_list : list of int
        Major-task counts (e.g. ``[4, 8, ..., 1024]``).
    n_gpus, verbose, **train_kwargs
        Same role as in :func:`unified_train_parallel` / :func:`unified_train`.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = [(t, n) for t in task_names for n in n_major_list]
    if not jobs:
        return []

    n_available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_workers = n_gpus if n_gpus is not None else min(2, n_available or 1)
    n_workers = min(n_workers, len(jobs))
    if n_workers <= 0:
        n_workers = 1
    use_gpu = n_available > 0

    train_kwargs = {k: v for k, v in train_kwargs.items() if k != "device"}
    worker_kwargs = {**train_kwargs, "_parallel_verbose": verbose}

    def _device(i: int) -> str:
        return f"cuda:{i % n_available}" if use_gpu else "cpu"

    args_list = [(t, n, {**worker_kwargs}) for t, n in jobs]

    if n_workers == 1:
        out = []
        for t, n in jobs:
            if verbose:
                print(
                    f"[major_only] Training {t} n_major={n} on {_device(0)} ...",
                    flush=True,
                )
            out.append(
                unified_train(
                    t,
                    0,
                    device=_device(0),
                    n_tasks=n,
                    n_minor_tasks=1,
                    p_minor=1e-12,
                    **train_kwargs,
                )
            )
        return out

    if verbose:
        print(
            f"[major_only] Starting parallel sweep: {len(jobs)} jobs, n_workers={n_workers}",
            flush=True,
        )

    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    gpu_queue = ctx.Queue()
    for i in range(n_workers):
        gpu_queue.put(_device(i))

    results = [None] * len(jobs)
    errors = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(gpu_queue,),
    ) as ex:
        future_to_idx = {ex.submit(_major_only_train_worker, a): i for i, a in enumerate(args_list)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            t, n = jobs[idx]
            try:
                results[idx] = future.result()
                if verbose:
                    print(f"[major_only] Completed {t} n_major={n}", flush=True)
            except Exception as e:
                logger.exception("major_only sweep failed for %s n_major=%s: %s", t, n, e)
                errors.append(((t, n), e))

    if errors:
        msg = "; ".join(f"{t} n_major={n}: {e}" for (t, n), e in errors)
        raise RuntimeError(f"unified_train_major_only_parallel: {len(errors)} job(s) failed — {msg}")

    return results
