import os
import torch
from typing import Optional

import icl.utils.notebook_utils as nu
from icl.utils.basic import get_hash
from icl.linear.lr_config import get_config
from icl.latent_markov.latent_config import get_config_base
from icl.utils.latent_ood_analysis import get_all_samples
from icl.utils.linear_ood_analysis import (
    _create_eval_task_pool,
    _setup_eval_task,
    setup_device,
    )
from icl.utils.kv_latent_task_vec_beta import compute_hiddens_onepos_all_layers_kvcache_beta
from icl.utils.ultra_latent_task_vec import compute_hiddens_onepos_all_layers_ultra
from icl.linear.linear_path_utils import load_model_task_config
from icl.linear.task_vecs import compute_hiddens_multi
from icl.utils.logger import setup_logger
from icl.utils.unified_path_finder import (
    _get_hidden_cache_path,
    unified_get_config,
)
from icl.utils.latent_task_vec import compute_hiddens
from icl.utils.coin_ood_analysis import get_new_sampler
from icl.dyck.dyck_task_vec import get_dyck_sampler, compute_hiddens_dyck
from icl.dyck.dyck_utils import sample_binary_mask

logger = setup_logger(__name__)


def _get_hiddens(
        task_name, 
        exp_name, 
        n_minor=64, 
        n_ood=30, 
        B=64,
        step: Optional[int] = None,
        force_recompute=False,
        verbose=False,
        **kwargs,
        ):

    if task_name == "latent":
        _, sampler, config = nu.load_everything("latent", exp_name)
        k_minor = min(n_minor, sampler.n_minor_tasks)
        if step is None:
            step = config.training.num_epochs

        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
        if verbose:
            logger.info("Getting samples...")
        all_samples, k_minor = get_all_samples(exp_name, n_minor=n_minor, n_ood=n_ood, B=B)
        if verbose:
            logger.info("Computing hiddens...")
        hiddens = compute_hiddens_onepos_all_layers_kvcache_beta(
                config, 
                model, 
                all_samples, 
                k_step = 32,
                b_step = 32,
                t_step = 4
            ).permute(0, 1, 3, 2, 4, 5) # (n_layers, n_tasks, num_states, T, B, D)
    
    elif task_name == "coin":
        _, sampler, config = nu.load_everything("coin", exp_name)
        k_minor = min(n_minor, sampler.n_minor_tasks)
        if step is None:
            step = config.training.num_epochs

        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
        if verbose:
            logger.info("Getting samples...")
        sampler_clone, k_minor = get_new_sampler(exp_name, n_minor, n_ood)
        hiddens = compute_hiddens(config, model, sampler_clone, B)
        if kwargs.get("return_p", False):
            return hiddens, k_minor, torch.concat([sampler_clone.major_p, sampler_clone.minor_p])
    
    elif task_name == "dyck":
        _, sampler, config = nu.load_everything("dyck", exp_name)
        k_minor = min(n_minor, sampler.n_minor_tasks)
        if step is None:
            step = config.training.num_epochs
        
        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
        if verbose:
            logger.info("Getting samples...")
        sampler_clone, k_minor = get_dyck_sampler(exp_name, n_minor, n_ood)
        mask = sample_binary_mask(config)
        hiddens = compute_hiddens_dyck(config,
                    model,
                    sampler = sampler_clone,
                    dyck_mask = mask,
                    batch_size = B,
                    )
        return hiddens, k_minor, mask

    elif task_name == "linear":
        _, train_task, config = load_model_task_config(exp_name)
        k_minor = min(n_minor, train_task.n_minor_tasks)
        if step is None:
            step = config.training.total_steps
        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
        
        device = setup_device(None)
        if verbose:
            logger.info("Creating eval task pool...")
        
        eval_task_pool, k_minor = _create_eval_task_pool(
            train_task, 
            K=n_ood, 
            include_minor=True, 
            radius=2.0, 
            device=device,
            n_minor=n_minor,
        )
        eval_task = _setup_eval_task(config, eval_task_pool, B, device)
        if verbose:
            logger.info("Computing hiddens...")
        
        hiddens, _ = compute_hiddens_multi(config, model, eval_task) # (n_layers, n_tasks, T, B, D)
    
    return hiddens, k_minor




def get_exp_name(
    task_name, 
    k: int,
    vocab_size = None,
    log2: bool = True
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
        config.task.p_minor = 1e-12  # Practically no minor tasks
    
    exp_name = f"train_{get_hash(config)}"
    return exp_name
    

from icl.models.base_models import Transformer
from icl.utils.train import train_model_with_plot
from icl.linear.train_linear import train

def unified_train(
    task_name,
    k: int,
    vocab_size: int = 11,
    log2: bool = True,
):
    config = unified_get_config(task_name)
    if k >= 0:
        if log2:
            config.task.n_minor_tasks = 2 ** k
        else:
            config.task.n_minor_tasks = k
    else:
        config.task.n_minor_tasks = 1
        config.task.p_minor = 1e-12  # Practically no minor tasks
    if task_name == "linear":
        return train(config)
    else:
        config.vocab_size = vocab_size
        model = Transformer(config)
        model = model.to(config.device)
        return train_model_with_plot(model, config, show=False, verbose=False)
    
