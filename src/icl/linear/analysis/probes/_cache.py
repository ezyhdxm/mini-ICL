"""Cached hidden state collection for linear separability analyses."""

import gc
from typing import Optional

import torch
from tqdm import tqdm

import icl.utils.notebook_utils as nu
from icl.linear.linear_path_utils import load_model_task_config
from icl.linear.task_vecs import extract_hidden_multi
from icl.linear.linear_ood_analysis import _create_eval_task_pool, _setup_eval_task, setup_device
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)

_linear_hiddens_cache: dict = {}


def _get_linear_hiddens_cached(
    exp_name: str,
    layers: Optional[list],
    batch_size: int,
    chunk_size: int,
    step: Optional[int],
    n_minor: int,
    n_ood: int,
    verbose: bool,
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
) -> tuple:
    """Return (all_hiddens, demo_data, layers), reusing cache when params match."""
    from icl.linear.analysis._helpers import _task_positions

    if layers is not None:
        layers = list(layers)

    key = (
        exp_name,
        tuple(layers) if layers is not None else None,
        batch_size,
        n_minor,
        n_ood,
        step,
        post_layernorm,
        extraction_point,
    )
    if key in _linear_hiddens_cache:
        logger.info("[linear cache] reusing cached hiddens")
        return _linear_hiddens_cache[key]

    logger.info(f"[linear] loading config + model for {exp_name} …")
    _, train_task, config = load_model_task_config(exp_name)

    if step is None:
        step = config.training.total_steps

    model, actual_step = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    logger.info(f"[linear] model loaded (step={actual_step}, n_layer={config.model.n_layer})")

    device = setup_device(None)
    eval_task_pool, k_minor = _create_eval_task_pool(
        train_task, K=n_ood, include_minor=True,
        device=device, n_minor=n_minor,
    )
    eval_task = _setup_eval_task(config, eval_task_pool, batch_size, device)
    eval_task.batch_size = batch_size

    if layers is None:
        layers = list(range(config.model.n_layer))

    n_points = config.task.n_points
    n_embd = config.model.n_embd
    n_tasks = eval_task.task_pool.shape[0]
    L = len(layers)
    pad_mode = getattr(model, "pad", "mapsto")
    task_pos = _task_positions(pad_mode, n_points, device=device)

    logger.info(
        f"[linear] extracting hiddens: layers={layers}, B={batch_size}, "
        f"n_tasks={n_tasks}, n_points={n_points}, pad={pad_mode}, device={device}"
    )

    all_hiddens = torch.empty(
        (L, n_tasks, n_points, batch_size, n_embd),
        dtype=torch.float32, device="cpu",
    )
    demo_data = eval_task.sample_data(step=step)

    n_chunks = max(1, (n_tasks + chunk_size - 1) // chunk_size)
    for i in tqdm(range(0, n_tasks, chunk_size), total=n_chunks, desc="linear hiddens (tasks)", unit="chunk"):
        chunk_end = min(i + chunk_size, n_tasks)
        chunk_n = chunk_end - i

        demo_data_rep = (
            demo_data.unsqueeze(0)
            .expand(chunk_n, batch_size, n_points, -1)
            .reshape(-1, n_points, demo_data.size(-1))
        )

        demo_target = eval_task.evaluate(
            demo_data,
            eval_task.task_pool[i:chunk_end].squeeze(-1).T,
            step=step,
        )
        if demo_target.ndim == 3:
            demo_target = demo_target.permute(2, 0, 1).reshape(-1, n_points)

        h = extract_hidden_multi(
            model=model, demo_data=demo_data_rep,
            demo_target=demo_target, layers=layers, task_pos=task_pos,
            post_layernorm=post_layernorm,
            extraction_point=extraction_point,
        )
        h = h.reshape(L, chunk_n, batch_size, n_points, n_embd)
        h = h.permute(0, 1, 3, 2, 4)
        all_hiddens[:, i:chunk_end] = h.cpu()

        del h
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_hiddens = all_hiddens.detach()

    n_major = train_task.task_pool.shape[0]

    model.cpu(); del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    result = (all_hiddens, demo_data.cpu(), layers, n_major, n_ood, k_minor)
    _linear_hiddens_cache.clear()
    _linear_hiddens_cache[key] = result
    return result
