import hashlib
import os
from typing import Optional, Sequence, Tuple, Union

import icl.utils.notebook_utils as nu
from icl.utils.basic import get_hash, canonicalize_config_for_exp
from icl.linear.lr_config import get_config
from icl.latent_markov.latent_config import get_config_base
from icl.linear.linear_path_utils import load_model_task_config
from icl.coin.coin_config import get_config_coin
from icl.dyck.dyck_config import get_config_dyck

def unified_get_config(
    task_name,
):
    if task_name == "linear":
        config = get_config()
    elif task_name == "latent":
        config = get_config_base()
    elif task_name == "coin":
        config = get_config_coin()
    elif task_name == "dyck":
        config = get_config_dyck()
    else:
        raise ValueError(f"Unknown task_name: {task_name}")
    config.training.warmup_steps = 15_000
    if task_name == "latent":
        config.training.warmup_steps = 15_000
        config.training.num_epochs = 30_000
        return config
    if task_name == "dyck":
        config.training.warmup_steps = 15_000
        config.training.num_epochs = 30_000
        return config
    if task_name == "linear":
        config.training.warmup_steps = 15_000
        config.training.total_steps = 30_000
    else:
        config.training.num_epochs = 30_000
    return config

def _get_exp_config_and_k_minor(
        task_name,
        exp_name: str,
        n_minor: int = 64,
):
    if task_name == "linear":
        _, train_task, config = load_model_task_config(exp_name)
        k_minor = min(n_minor, train_task.n_minor_tasks)
    else:
        _, sampler, config = nu.load_everything(task_name, exp_name)
        k_minor = min(n_minor, sampler.n_minor_tasks)
    return config, k_minor

def _get_exp_dir(config, exp_name: str) -> str:
    """Get the experiment directory path, accounting for notebook context."""
    exp_dir = os.path.join(config.work_dir, exp_name)
    cur_dir = os.getcwd()
    if cur_dir.endswith("notebooks"):
        exp_dir = os.path.join("..", exp_dir)
    return exp_dir

def _get_hidden_cache_path(
        config, 
        task_name,
        exp_name: str, 
        k_minor: int, 
        n_ood: int,
        B: int,
        step: int,
        layer_index,
        ) -> str:
    """Generate cache file path based on experiment parameters."""
    exp_dir = _get_exp_dir(config, exp_name)
    cache_file = f"hiddens_{task_name}_kminor_{k_minor}_kood_{n_ood}_b_{B}_l_{layer_index}_{exp_name}_step{step}.pt"
    return os.path.join(exp_dir, cache_file)

def _get_var_plot_cache_path(
        config,
        task_name,
        exp_name: str,
        k_minor: int,
        n_ood: int,
        B: int,
        ignore_vocab: bool = False,
    ) -> str:
    """Generate cache file path for variance plot data."""
    exp_dir = _get_exp_dir(config, exp_name)
    if task_name == "latent" and ignore_vocab:
        exp_name += "_ignorevoc"
    cache_file = f"varplot_{task_name}_kminor_{k_minor}_kood_{n_ood}_b_{B}_{exp_name}.pkl"
    return os.path.join(exp_dir, cache_file)

def _get_projection_plot_cache_path(
        config,
        task_name,
        exp_name: str,
        k_minor: int,
        n_ood: int,
        B: int,
        voc: Optional[int]=None,
        layer_index: Optional[int] = None,
        minor_projection: bool = False,
    ) -> str:
    """Generate cache file path for projection plot data."""
    exp_dir = _get_exp_dir(config, exp_name)
    proj_type = "minorproj" if minor_projection else "stdproj"
    if layer_index is None:
        layer_index = config.model.n_layer-1 if task_name == "linear" else config.model.num_layers-1
    if voc is not None and task_name == "latent":
        exp_name += f"_voc{voc}"
    exp_name += f"_layer{layer_index}"
    cache_file = f"projplot_{proj_type}_{task_name}_kminor_{k_minor}_kood_{n_ood}_b_{B}_{exp_name}.pkl"
    return os.path.join(exp_dir, cache_file)

def _get_traj_plot_cache_path(
        config,
        task_name,
        exp_name: str,
        k_minor: int,
        n_ood: int,
        B: int,
        voc: Optional[int]=None,
        layer_index: Optional[int] = None,
        minor_projection: bool = False,
    ) -> str:
    """Generate cache file path for projection plot data."""
    exp_dir = _get_exp_dir(config, exp_name)
    proj_type = "minorproj" if minor_projection else "stdproj"
    if layer_index is None:
        layer_index = config.model.n_layer-1 if task_name == "linear" else config.model.num_layers-1
    if voc is not None and task_name == "latent":
        exp_name += f"_voc{voc}"
    exp_name += f"_layer{layer_index}"
    cache_file = f"traj_projplot_{proj_type}_{task_name}_kminor_{k_minor}_kood_{n_ood}_b_{B}_{exp_name}.png"
    return os.path.join(exp_dir, cache_file)

def _get_traj_post_plot_cache_path(
        config,
        task_name,
        exp_name: str,
        k_minor: int,
        n_ood: int,
        B: int,
        voc: Optional[int]=None,
        layer_index: Optional[int] = None,
        minor_projection: bool = False,
    ) -> str:
    """Generate cache file path for projection plot data."""
    exp_dir = _get_exp_dir(config, exp_name)
    proj_type = "minorproj" if minor_projection else "stdproj"
    if layer_index is None:
        layer_index = config.model.n_layer-1 if task_name == "linear" else config.model.num_layers-1
    if voc is not None and task_name == "latent":
        exp_name += f"_voc{voc}"
    exp_name += f"_layer{layer_index}"
    cache_file = f"traj_post_projplot_{proj_type}_{task_name}_kminor_{k_minor}_kood_{n_ood}_b_{B}_{exp_name}.png"
    return os.path.join(exp_dir, cache_file)

def _get_metrics_cache_path(
    config,
    task_name: str,
    exp_name: str,
    k_minor: int,
    n_ood: int,
    B: int,
    steps: Sequence[int],
    layer_indices: Sequence[int],
    position_blocks: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
    extraction_point: str = "post_attn",
    avg_over: int = 1,
):
    """Generate cache file path for OOD metrics data."""
    exp_dir = _get_exp_dir(config, exp_name)
    pos_suffix = ""
    if position_blocks is not None:
        key = repr(tuple(position_blocks))
        pos_suffix = "_posblocks_" + hashlib.md5(key.encode()).hexdigest()[:8]
    ep_suffix = f"_ep_{extraction_point}" if extraction_point != "post_attn" else ""
    avg_suffix = f"_avg{avg_over}" if avg_over > 1 else ""
    cache_files = {}
    for step in steps:
        cache_files[step] = {}
        for layer_index in layer_indices:
            cache_file = (
                f"metrics_{task_name}_kminor_{k_minor}_kood_{n_ood}_b_{B}_iqr_"
                f"{layer_index}_{exp_name}_{step}{avg_suffix}{pos_suffix}{ep_suffix}.pkl"
            )
            cache_files[step][layer_index] = os.path.join(exp_dir, cache_file)
    return cache_files


def get_exp_name(
    task_name,
    k: int,
    vocab_size=None,
    log2: bool = True,
    pad=None,
    major_pool_type: str = None,
    major_means=None,
    major_seed: int = None,
    total_steps: Optional[int] = None,
    warmup_steps: Optional[int] = None,
    lr: Optional[float] = None,
    schedule: Optional[str] = None,
    max_grad_norm: Optional[float] = None,
    batch_size: Optional[int] = None,
    noise_scale: Optional[float] = None,
    p_minor: Optional[float] = None,
    n_layer: Optional[int] = None,
    n_points: Optional[int] = None,
    min_lr: Optional[float] = None,
    decay_power: Optional[float] = None,
    batch_size_schedule: list = None,
    p_minor_schedule: list = None,
    final_layernorm: bool = None,
    n_tasks: Optional[int] = None,
    n_minor_tasks: Optional[int] = None,
) -> str:
    """Generate standardized experiment name based on task and parameters."""
    config = unified_get_config(task_name)
    if task_name != "linear" and vocab_size is not None:
        config.vocab_size = vocab_size
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
    if pad is not None:
        if task_name == "linear":
            config.model.pad = pad
    if major_pool_type is not None:
        config.task.major_pool_type = major_pool_type
    if major_means is not None:
        config.task.major_means = list(major_means)
    if major_seed is not None:
        config.task.major_seed = major_seed
    if total_steps is not None:
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

    canonicalize_config_for_exp(config)
    exp_name = f"train_{get_hash(config)}"
    return exp_name