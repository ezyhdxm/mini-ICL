import os
from typing import Sequence, Optional

import icl.utils.notebook_utils as nu
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
    config.training.warmup_steps = 30_000
    if task_name == "linear":
        config.training.total_steps = 60_000
    else:
        config.training.num_epochs = 60_000
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



def _get_metrics_cache_path(
        config,
        task_name,
        exp_name: str,
        k_minor: int,
        n_ood: int,
        B: int,
        steps: Sequence[int],
        layer_indices: Sequence[int],
):
    """Generate cache file path for OOD metrics data."""
    exp_dir = _get_exp_dir(config, exp_name)
    cache_files = {}
    for step in steps:
        cache_files[step] = {}
        for layer_index in layer_indices:
            cache_file = f"metrics_{task_name}_kminor_{k_minor}_kood_{n_ood}_b_{B}_{layer_index}_{exp_name}_{step}.pkl"
            cache_files[step][layer_index] = os.path.join(exp_dir, cache_file)
    return cache_files