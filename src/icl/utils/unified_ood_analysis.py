import copy
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from icl.utils.linear_algebra_utils import (
    estimate_lambda_with_r2,
    estimate_lambda_with_r2_batched_B,
)
from icl.utils.unified_path_finder import _get_metrics_cache_path
from icl.utils.unified_interface import (
    _get_hiddens,
    _compute_hiddens_at_real_tokens,
    pregenerate_task_sequences,
)
import icl.utils.notebook_utils as nu
from icl.linear.linear_path_utils import load_model_task_config
from icl.utils.logger import setup_logger

from icl.coin.coin_ood_analysis import get_new_sampler
from icl.latent_markov.analysis.ood import get_latent_sampler

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
    # When k_minor==0, -0 == 0 in Python, so [3:-0] becomes [3:0] (empty).
    # Use None to mean "no upper bound" when there are no minor tasks.
    ood_end = -k_minor if k_minor > 0 else None
    if isinstance(position, tuple):
        start, end = position
        r2_ood_final = r2_scores[3:ood_end, start:end].mean(dim=1)
        r2_minor_final = r2_scores[-k_minor:, start:end].mean(dim=1) if k_minor > 0 else r2_scores[:0, start:end].mean(dim=1)
        lambdas_ood_final = lambdas[3:ood_end, start:end, :].mean(dim=1)
        lambdas_minor_final = lambdas[-k_minor:, start:end, :].mean(dim=1) if k_minor > 0 else lambdas[:0, start:end, :]
    else:
        r2_ood_final = r2_scores[3:ood_end, position]
        r2_minor_final = r2_scores[-k_minor:, position] if k_minor > 0 else r2_scores[:0, position]
        lambdas_ood_final = lambdas[3:ood_end, position]
        lambdas_minor_final = lambdas[-k_minor:, position] if k_minor > 0 else lambdas[:0, position]
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
    position_blocks: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
    compute_minor_metrics: bool = False,
):
    task_mean = hiddens[:3].mean(dim=(0, 2)).unsqueeze(0)
    task_vecs_over_all_time = hiddens.mean(dim=-2) - task_mean
    maj_final_task_vecs = (hiddens[:3].mean(dim=-2) - task_mean)[:, -1]

    maj_r2_ood, maj_r2_min, maj_ood_var, maj_minor_var = _compute_ood_and_minor_metrics(
        maj_final_task_vecs,
        task_vecs_over_all_time,
        k_minor,
    )

    metrics_dict: Dict[str, Any] = {
        "maj_r2_ood": maj_r2_ood,
        "maj_r2_min": maj_r2_min,
        "maj_ood_var": maj_ood_var,
        "maj_minor_var": maj_minor_var,
    }

    if compute_minor_metrics:
        min_final_task_vecs = _get_minor_final_task_vecs(
            task_vecs_over_all_time,
            k_minor,
        )
        min_r2_ood, min_r2_min, min_ood_var, min_minor_var = _compute_ood_and_minor_metrics(
            min_final_task_vecs,
            task_vecs_over_all_time,
            k_minor,
            is_zero_mean=False,
        )
        metrics_dict["min_r2_ood"] = min_r2_ood
        metrics_dict["min_r2_min"] = min_r2_min
        metrics_dict["min_ood_var"] = min_ood_var
        metrics_dict["min_minor_var"] = min_minor_var

    if position_blocks is not None:
        maj_r2_ood_by_block: List[float] = []
        for pos in position_blocks:
            r2_ood, _, _, _ = _compute_ood_and_minor_metrics(
                maj_final_task_vecs,
                task_vecs_over_all_time,
                k_minor,
                position=pos,
            )
            maj_r2_ood_by_block.append(r2_ood)
        metrics_dict["maj_r2_ood_by_block"] = maj_r2_ood_by_block

    return metrics_dict


def _maj_r2_ood_values_from_r2_batch(
    r2_scores: torch.Tensor,
    k_minor: int,
    position: Union[int, Tuple[int, int]],
) -> torch.Tensor:
    """Return maj OOD R² values for all ``(b, ood task)`` pairs.

    Parameters
    ----------
    r2_scores : (B, k, seq_len)
    """
    ood_end = -k_minor if k_minor > 0 else None
    if isinstance(position, tuple):
        start, end = position
        r2_ood_final = r2_scores[:, 3:ood_end, start:end].mean(dim=2)
    else:
        r2_ood_final = r2_scores[:, 3:ood_end, position]
    return r2_ood_final.reshape(-1)


def _compute_maj_r2_per_b_stats(
    hiddens: torch.Tensor,
    k_minor: int,
    position_blocks: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
) -> Dict[str, Any]:
    """Mean and IQR (q25, q75) of maj_r2_ood over flattened ``B x OOD`` values.

    Uses one batched ``estimate_lambda_with_r2``-equivalent pass over ``B`` (no
    ``B``-fold ``_compute_metrics`` loop). For block positions, positions inside
    the block are averaged first, then the remaining ``B x OOD`` values are
    treated uniformly for mean / quantiles.
    """
    hiddens = hiddens.to(torch.float32)
    # Match per-slice ``_compute_metrics`` geometry: (n_tasks, T, B, d)
    task_mean_all = hiddens[:3].mean(dim=0)
    centered = hiddens - task_mean_all.unsqueeze(0)
    # centered[:3, -1, :, :] is (3, B, d); batched API expects (B, 3, d)
    maj_final = centered[:3, -1, :, :].permute(1, 0, 2).contiguous()
    tvot = centered.permute(2, 0, 1, 3).contiguous()

    r2 = estimate_lambda_with_r2_batched_B(maj_final, tvot, is_zero_mean=True)

    maj_arr = _maj_r2_ood_values_from_r2_batch(r2, k_minor, -1).detach().cpu().numpy()
    arr = np.asarray(maj_arr, dtype=float)
    out: Dict[str, Any] = {
        "maj_r2_ood": float(arr.mean()),
        "maj_r2_ood_iqr": {
            "q25": float(np.quantile(arr, 0.25)),
            "q75": float(np.quantile(arr, 0.75)),
        },
    }
    n_blocks = len(position_blocks) if position_blocks is not None else 0
    if n_blocks:
        out["maj_r2_ood_by_block"] = []
        out["maj_r2_ood_by_block_iqr"] = []
        for pos in position_blocks:
            blk = _maj_r2_ood_values_from_r2_batch(r2, k_minor, pos).detach().cpu().numpy()
            out["maj_r2_ood_by_block"].append(float(blk.mean()))
            out["maj_r2_ood_by_block_iqr"].append(
                {
                    "q25": float(np.quantile(blk, 0.25)),
                    "q75": float(np.quantile(blk, 0.75)),
                }
            )
    return out


def _merge_maj_r2_iqr_into_metrics(
    metrics_dict: Dict[str, Any],
    hiddens: torch.Tensor,
    k_minor: int,
    position_blocks: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
) -> None:
    """Overwrite maj_r2_ood (mean over B) and add cached IQR keys; other keys unchanged."""
    stats = _compute_maj_r2_per_b_stats(hiddens, k_minor, position_blocks)
    metrics_dict["maj_r2_ood"] = stats["maj_r2_ood"]
    metrics_dict["maj_r2_ood_iqr"] = stats["maj_r2_ood_iqr"]
    if "maj_r2_ood_by_block" in stats:
        metrics_dict["maj_r2_ood_by_block"] = stats["maj_r2_ood_by_block"]
        metrics_dict["maj_r2_ood_by_block_iqr"] = stats["maj_r2_ood_by_block_iqr"]


def _set_layer_indices(
    config,
):
    if config.task.name == "noisy_linear_regression":
        layer_indices = list(range(config.model.n_layer))
    else:
        layer_indices = list(range(config.model.num_layers))
    return layer_indices


def _load_step_from_cache(
    cache_paths: Dict,
    step: int,
    layer_indices: List[int],
    force_recompute: bool,
    maj_r2_ood: Dict,
    maj_r2_min: Dict,
    min_r2_ood: Dict,
    min_r2_min: Dict,
    maj_ood_var: Dict,
    maj_min_var: Dict,
    min_ood_var: Dict,
    min_min_var: Dict,
    maj_r2_ood_by_block: Dict,
    maj_r2_ood_iqr: Dict,
    maj_r2_ood_by_block_iqr: Dict,
) -> bool:
    """Try to load all layers for one step from cache. Returns True if fully cached."""
    for L in layer_indices:
        if not (os.path.exists(cache_paths[step][L]) and not force_recompute):
            return False
    for L in layer_indices:
        with open(cache_paths[step][L], "rb") as f:
            metrics_dict = pickle.load(f)
        _store_metrics(L, step, metrics_dict, maj_r2_ood, maj_r2_min, min_r2_ood, min_r2_min,
                       maj_ood_var, maj_min_var, min_ood_var, min_min_var, maj_r2_ood_by_block,
                       maj_r2_ood_iqr, maj_r2_ood_by_block_iqr)
    return True


def _store_metrics(
    L: int,
    step: int,
    metrics_dict: Dict,
    maj_r2_ood: Dict,
    maj_r2_min: Dict,
    min_r2_ood: Dict,
    min_r2_min: Dict,
    maj_ood_var: Dict,
    maj_min_var: Dict,
    min_ood_var: Dict,
    min_min_var: Dict,
    maj_r2_ood_by_block: Dict,
    maj_r2_ood_iqr: Dict,
    maj_r2_ood_by_block_iqr: Dict,
) -> None:
    maj_r2_ood[L][step] = metrics_dict["maj_r2_ood"]
    maj_r2_min[L][step] = metrics_dict["maj_r2_min"]
    maj_ood_var[L][step] = metrics_dict["maj_ood_var"]
    maj_min_var[L][step] = metrics_dict["maj_minor_var"]
    if "min_r2_ood" in metrics_dict:
        min_r2_ood[L][step] = metrics_dict["min_r2_ood"]
        min_r2_min[L][step] = metrics_dict["min_r2_min"]
        min_ood_var[L][step] = metrics_dict["min_ood_var"]
        min_min_var[L][step] = metrics_dict["min_minor_var"]
    if "maj_r2_ood_by_block" in metrics_dict:
        maj_r2_ood_by_block[L][step] = metrics_dict["maj_r2_ood_by_block"]
    if "maj_r2_ood_iqr" in metrics_dict:
        maj_r2_ood_iqr[L][step] = metrics_dict["maj_r2_ood_iqr"]
    if "maj_r2_ood_by_block_iqr" in metrics_dict:
        maj_r2_ood_by_block_iqr[L][step] = metrics_dict["maj_r2_ood_by_block_iqr"]


def _compute_and_cache_metrics(
    hiddens: torch.Tensor,
    layer_indices: List[int],
    k_minor: int,
    step: int,
    cache_paths: Dict,
    position_blocks,
    maj_r2_ood: Dict,
    maj_r2_min: Dict,
    min_r2_ood: Dict,
    min_r2_min: Dict,
    maj_ood_var: Dict,
    maj_min_var: Dict,
    min_ood_var: Dict,
    min_min_var: Dict,
    maj_r2_ood_by_block: Dict,
    maj_r2_ood_iqr: Dict,
    maj_r2_ood_by_block_iqr: Dict,
    compute_minor_metrics: bool = False,
) -> None:
    """Compute metrics for all layers from CPU hiddens and cache to disk."""
    step_metrics = _compute_step_metrics_and_cache(
        hiddens,
        layer_indices,
        k_minor,
        step,
        cache_paths,
        position_blocks,
        compute_minor_metrics=compute_minor_metrics,
    )
    for L in layer_indices:
        _store_metrics(L, step, step_metrics[L], maj_r2_ood, maj_r2_min, min_r2_ood, min_r2_min,
                       maj_ood_var, maj_min_var, min_ood_var, min_min_var, maj_r2_ood_by_block,
                       maj_r2_ood_iqr, maj_r2_ood_by_block_iqr)


def _compute_step_metrics_and_cache(
    hiddens: torch.Tensor,
    layer_indices: List[int],
    k_minor: int,
    step: int,
    cache_paths: Dict,
    position_blocks,
    compute_minor_metrics: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """Compute metrics for all layers from CPU hiddens and cache to disk."""
    step_metrics: Dict[int, Dict[str, Any]] = {}
    for L in layer_indices:
        h_L = hiddens[L].to(torch.float32)
        metrics_dict = _compute_metrics(
            h_L,
            k_minor,
            position_blocks=position_blocks,
            compute_minor_metrics=compute_minor_metrics,
        )
        _merge_maj_r2_iqr_into_metrics(metrics_dict, h_L, k_minor, position_blocks)
        with open(cache_paths[step][L], "wb") as f:
            pickle.dump(metrics_dict, f)
        step_metrics[L] = metrics_dict
    return step_metrics


def _run_step_coin_or_latent(
    task_name: str,
    exp_name: str,
    step: int,
    config,
    sampler_clone,
    B: int,
    layer_indices: List[int],
    k_minor: int,
    cache_paths: Dict,
    position_blocks,
    compute_minor_metrics: bool = False,
    extraction_point: str = "post_attn",
    precomputed_data=None,
):
    """Run a single step for coin/latent: load model → forward → metrics.

    Returns dict mapping layer → metrics_dict, or None on failure.
    """
    try:
        model, _ = nu.load_checkpoint(
            config, step=step, exp_name=exp_name, return_actual_step=True
        )

        hiddens = _compute_hiddens_at_real_tokens(
            config, model, sampler_clone, B,
            extraction_point=extraction_point,
            precomputed_data=precomputed_data,
        ).cpu()

        del model
        step_metrics: Dict[int, Dict] = {}
        for L in layer_indices:
            h_L = hiddens[L].to(torch.float32)
            metrics_dict = _compute_metrics(
                h_L, k_minor,
                position_blocks=position_blocks,
                compute_minor_metrics=compute_minor_metrics,
            )
            _merge_maj_r2_iqr_into_metrics(metrics_dict, h_L, k_minor, position_blocks)
            with open(cache_paths[step][L], "wb") as f:
                pickle.dump(metrics_dict, f)
            step_metrics[L] = metrics_dict
        del hiddens
        return step_metrics
    except Exception as e:
        logger.warning(f"[_run_step] {task_name}/{exp_name} step {step} failed: {e}")
        return None


def _run_step_linear(
    exp_name: str,
    step: int,
    device: str,
    n_minor: int,
    n_ood: int,
    B: int,
    layer_indices: List[int],
    k_minor: int,
    cache_paths: Dict,
    position_blocks,
    compute_minor_metrics: bool = False,
    extraction_point: str = "post_attn",
):
    """Run a single linear step on one device and return per-layer metrics."""
    try:
        hiddens, _ = _get_hiddens(
            "linear",
            exp_name,
            n_minor,
            n_ood,
            B,
            step=step,
            device=device,
            extraction_point=extraction_point,
        )
        step_metrics = _compute_step_metrics_and_cache(
            hiddens,
            layer_indices,
            k_minor,
            step,
            cache_paths,
            position_blocks,
            compute_minor_metrics=compute_minor_metrics,
        )
        del hiddens
        return step_metrics
    except Exception as e:
        logger.warning(f"[_run_step] linear/{exp_name} step {step} failed: {e}")
        return None


def process_ood_minor_metric(
    task_name: str,
    exp_name: str,
    steps: Sequence[int],
    n_minor: int = 64,
    n_ood: int = 30,
    B: int = 64,
    force_recompute: bool = False,
    device: Optional[str] = None,
    position_blocks: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
    n_gpus: int = 1,
    compute_minor_metrics: bool = False,
    extraction_point: str = "post_attn",
):
    """Compute OOD / minor R² metrics across training steps.

    Per checkpoint/layer pickle files store ``maj_r2_ood`` (mean over ``B``
    forward passes) and ``maj_r2_ood_iqr`` with ``q25`` / ``q75`` only (no
    raw ``B`` values). Cache filenames include ``_iqr_`` so older metric
    pickles are not reused.

    Parameters
    ----------
    n_gpus : int
        Number of GPUs for step-level parallelism.  When > 1, uncached steps
        are distributed across GPUs using ``ThreadPoolExecutor`` (one thread
        per GPU).  ``device`` is ignored when ``n_gpus > 1``.
    compute_minor_metrics : bool
        If True, also compute ``min_r2_ood``, ``min_r2_min``, and related
        variances using the **minor** reference vectors (extra ``estimate_lambda``
        solve + SVD over minor tasks).  Default False skips that work.
    """
    # ── Setup: load config + sampler (no model) ──────────────────────────────
    if task_name == "linear":
        _, train_task, config = load_model_task_config(exp_name)
        k_minor = min(n_minor, train_task.n_minor_tasks)
    else:
        sampler, config = nu.load_config_and_sampler(task_name, exp_name)
        k_minor = min(n_minor, sampler.n_minor_tasks)

    # ── Filter requested steps to only those that exist on disk ───────────────
    # Without this, load_checkpoint silently falls back to the nearest available
    # checkpoint, producing duplicate data points for pruned steps.
    _exp_dir = os.path.join(config.work_dir, exp_name)
    if os.getcwd().endswith("notebooks"):
        _exp_dir = os.path.join("..", _exp_dir)
    _ckpt_dir = _exp_dir if task_name == "linear" else os.path.join(_exp_dir, "checkpoints")
    try:
        _available_steps = set(nu.list_checkpoints(_ckpt_dir)["all_steps"])
        _requested = list(steps)
        steps = [s for s in _requested if s in _available_steps]
        _skipped = [s for s in _requested if s not in _available_steps]
        if _skipped:
            logger.info(
                f"[process_ood_minor_metric] {exp_name}: skipping {len(_skipped)} "
                f"requested steps not found on disk "
                f"(e.g. {_skipped[:3]}{'...' if len(_skipped) > 3 else ''}). "
                f"Pass steps that match available checkpoints to suppress this."
            )
    except Exception as _e:
        logger.warning(
            f"[process_ood_minor_metric] Could not read available checkpoints "
            f"from {_ckpt_dir}: {_e}. Proceeding with all requested steps."
        )

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
        position_blocks=position_blocks,
        extraction_point=extraction_point,
    )

    # ── Storage dicts ─────────────────────────────────────────────────────────
    maj_r2_ood: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    maj_r2_ood_by_block: Dict[int, Dict[int, List[float]]] = {
        L: {} for L in layer_indices
    }
    maj_r2_ood_iqr: Dict[int, Dict[int, Dict[str, float]]] = {L: {} for L in layer_indices}
    maj_r2_ood_by_block_iqr: Dict[int, Dict[int, List[Dict[str, float]]]] = {
        L: {} for L in layer_indices
    }
    maj_r2_min: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    min_r2_ood: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    min_r2_min: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    maj_ood_var: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    maj_min_var: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    min_ood_var: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    min_min_var: Dict[int, Dict[int, float]] = {L: {} for L in layer_indices}
    processed_steps: List[int] = []

    _cache_args = (
        maj_r2_ood, maj_r2_min, min_r2_ood, min_r2_min,
        maj_ood_var, maj_min_var, min_ood_var, min_min_var,
        maj_r2_ood_by_block,
        maj_r2_ood_iqr, maj_r2_ood_by_block_iqr,
    )

    # ── Resolve multi-GPU configuration ───────────────────────────────────────
    n_available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    actual_gpus = min(n_gpus, n_available) if n_gpus > 1 and n_available > 1 else 0
    use_threading = actual_gpus > 1

    # ── Pre-filter cached steps (avoids wasting GPU time) ─────────────────────
    uncached_steps: List[int] = []
    for step in steps:
        if _load_step_from_cache(cache_paths, step, layer_indices,
                                 force_recompute, *_cache_args):
            processed_steps.append(step)
        else:
            uncached_steps.append(step)

    if not uncached_steps:
        return _build_results_dict(processed_steps, layer_indices, position_blocks,
                                   *_cache_args)

    # ── COIN / LATENT: optimized paths ────────────────────────────────────────
    if task_name in ("coin", "latent"):
        if task_name == "coin":
            sampler_clone, k_minor = get_new_sampler(
                exp_name, n_minor, n_ood, sampler=sampler
            )
        else:
            sampler_clone, k_minor, _ = get_latent_sampler(
                exp_name, n_minor, n_ood, sampler=sampler
            )

        if use_threading:
            # ── Multi-GPU: one thread per GPU, steps distributed evenly ───────
            gpu_configs = []
            gpu_samplers = []
            for g in range(actual_gpus):
                cfg = copy.deepcopy(config)
                cfg.device = f"cuda:{g}"
                sc = copy.deepcopy(sampler_clone)
                sc.to(f"cuda:{g}")
                gpu_configs.append(cfg)
                gpu_samplers.append(sc)

            results_lock = threading.Lock()
            _precomputed = pregenerate_task_sequences(sampler_clone, B)

            def _gpu_worker(gpu_id: int, my_steps: List[int]):
                """Process all assigned steps on one GPU sequentially."""
                cfg = gpu_configs[gpu_id]
                sc = gpu_samplers[gpu_id]
                local_ok: List[int] = []
                for step in my_steps:
                    step_metrics = _run_step_coin_or_latent(
                        task_name, exp_name, step, cfg, sc, B,
                        layer_indices, k_minor, cache_paths, position_blocks,
                        compute_minor_metrics,
                        extraction_point=extraction_point,
                        precomputed_data=_precomputed,
                    )
                    if step_metrics is not None:
                        with results_lock:
                            for L in layer_indices:
                                _store_metrics(L, step, step_metrics[L],
                                               *_cache_args)
                        local_ok.append(step)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return local_ok

            per_gpu_steps: List[List[int]] = [[] for _ in range(actual_gpus)]
            for i, step in enumerate(uncached_steps):
                per_gpu_steps[i % actual_gpus].append(step)

            try:
                with ThreadPoolExecutor(max_workers=actual_gpus) as executor:
                    futures = {
                        executor.submit(_gpu_worker, g, per_gpu_steps[g]): g
                        for g in range(actual_gpus) if per_gpu_steps[g]
                    }
                    for future in as_completed(futures):
                        gpu_id = futures[future]
                        try:
                            ok_steps = future.result()
                            processed_steps.extend(ok_steps)
                        except Exception as e:
                            logger.warning(
                                f"[process_ood_minor_metric] GPU {gpu_id} "
                                f"worker failed: {e}"
                            )
            except KeyboardInterrupt:
                logger.info(
                    f"\nInterrupted. Processed {len(processed_steps)} "
                    f"checkpoints so far."
                )

            del gpu_configs, gpu_samplers
        else:
            # ── Single-GPU sequential path ────────────────────────────────────
            if device is not None:
                config.device = device
                sampler_clone.to(device)

            _precomputed = pregenerate_task_sequences(sampler_clone, B)

            try:
                for step in uncached_steps:
                    step_metrics = _run_step_coin_or_latent(
                        task_name, exp_name, step, config, sampler_clone, B,
                        layer_indices, k_minor, cache_paths, position_blocks,
                        compute_minor_metrics,
                        extraction_point=extraction_point,
                        precomputed_data=_precomputed,
                    )
                    if step_metrics is not None:
                        for L in layer_indices:
                            _store_metrics(L, step, step_metrics[L],
                                           *_cache_args)
                        processed_steps.append(step)
            except KeyboardInterrupt:
                logger.info(
                    f"\nInterrupted. Processed {len(processed_steps)} "
                    f"checkpoints so far."
                )
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ── LINEAR: multi-GPU step parallelism over device-specific eval tasks ───
    elif task_name == "linear" and use_threading:
        results_lock = threading.Lock()

        def _gpu_worker(gpu_id: int, my_steps: List[int]):
            """Process linear steps sequentially on one GPU."""
            local_ok: List[int] = []
            dev = f"cuda:{gpu_id}"
            for step in my_steps:
                step_metrics = _run_step_linear(
                    exp_name,
                    step,
                    dev,
                    n_minor,
                    n_ood,
                    B,
                    layer_indices,
                    k_minor,
                    cache_paths,
                    position_blocks,
                    compute_minor_metrics,
                    extraction_point=extraction_point,
                )
                if step_metrics is not None:
                    with results_lock:
                        for L in layer_indices:
                            _store_metrics(L, step, step_metrics[L], *_cache_args)
                    local_ok.append(step)
            return local_ok

        per_gpu_steps: List[List[int]] = [[] for _ in range(actual_gpus)]
        for i, step in enumerate(uncached_steps):
            per_gpu_steps[i % actual_gpus].append(step)

        try:
            with ThreadPoolExecutor(max_workers=actual_gpus) as executor:
                futures = {
                    executor.submit(_gpu_worker, g, per_gpu_steps[g]): g
                    for g in range(actual_gpus) if per_gpu_steps[g]
                }
                for future in as_completed(futures):
                    gpu_id = futures[future]
                    try:
                        ok_steps = future.result()
                        processed_steps.extend(ok_steps)
                    except Exception as e:
                        logger.warning(
                            f"[process_ood_minor_metric] GPU {gpu_id} "
                            f"worker failed: {e}"
                        )
        except KeyboardInterrupt:
            logger.info(
                f"\nInterrupted. Processed {len(processed_steps)} "
                f"checkpoints so far."
            )

    # ── OTHER / fallback: existing _get_hiddens path ──────────────────────────
    else:
        try:
            for step in uncached_steps:
                hiddens, _ = _get_hiddens(
                    task_name, exp_name, n_minor, n_ood, B,
                    step=step, device=device,
                    extraction_point=extraction_point,
                )
                _compute_and_cache_metrics(
                    hiddens, layer_indices, k_minor, step, cache_paths,
                    position_blocks, *_cache_args,
                    compute_minor_metrics=compute_minor_metrics,
                )
                del hiddens
                processed_steps.append(step)
        except KeyboardInterrupt:
            logger.info(
                f"\nInterrupted. Processed {len(processed_steps)} "
                f"checkpoints so far."
            )

    if not processed_steps:
        logger.warning("No checkpoints processed successfully.")
        return {}

    return _build_results_dict(processed_steps, layer_indices, position_blocks,
                               *_cache_args)


def _build_results_dict(
    processed_steps: List[int],
    layer_indices: List[int],
    position_blocks,
    maj_r2_ood, maj_r2_min, min_r2_ood, min_r2_min,
    maj_ood_var, maj_min_var, min_ood_var, min_min_var,
    maj_r2_ood_by_block,
    maj_r2_ood_iqr,
    maj_r2_ood_by_block_iqr,
) -> Dict[str, Any]:
    results_dict: Dict[str, Any] = {
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
        "maj_r2_ood_iqr": maj_r2_ood_iqr,
    }
    if position_blocks is not None:
        results_dict["maj_r2_ood_by_block"] = maj_r2_ood_by_block
        results_dict["maj_r2_ood_by_block_iqr"] = maj_r2_ood_by_block_iqr
        results_dict["position_blocks"] = list(position_blocks)
    return results_dict
