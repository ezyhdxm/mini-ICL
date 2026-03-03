from icl.utils.unified_path_finder import (
    _get_var_plot_cache_path,
    _get_projection_plot_cache_path,
    _get_exp_config_and_k_minor,
    _get_traj_plot_cache_path,
    _get_traj_post_plot_cache_path,
)
import icl.utils.notebook_utils as nu
from icl.linear.linear_path_utils import load_model_task_config
from icl.figures.task_vec_viz import plot_task_vector_exp
from icl.utils.unified_interface import (
    _get_hiddens,
)
from icl.utils.unified_ood_analysis import (
    _get_minor_final_task_vecs,
)
from icl.utils.unified_interface import get_exp_name
from icl.linear.linear_utils import estimate_lambda_with_r2
from icl.latent_markov.legacy.latent_task_vec import project_with_r2_size
from icl.utils.traj_plot import project_with_r2_trajectories_group_colors_mpl
from icl.utils.posterior_plot import project_with_r2_ood_posterior_colors_mpl

from typing import Any, Dict
import os
import pickle
import json
import plotly.graph_objects as go
import numpy as np
import torch


def get_var_plot(
        task_name,
        exp_name: str,
        k_minor: int = 64,
        n_ood: int = 30,
        B: int = 64,
        ignore_vocab: bool = False,
        forced_recompute: bool = False,
    ):
    if task_name == "linear":
        _, train_task, config = load_model_task_config(exp_name)
    else:
        _, sampler, config = nu.load_everything(task_name, exp_name)

    cache_path = _get_var_plot_cache_path(
        config,
        task_name,
        exp_name,
        k_minor,
        n_ood,
        B,
        ignore_vocab,
    )
    if os.path.exists(cache_path) and not forced_recompute:
        with open(cache_path, "rb") as f:
            fig = pickle.load(f)
            fig.show()
    else:
        if task_name == "dyck":
            hiddens, k_minor, mask = _get_hiddens(
                task_name,
                exp_name,
                k_minor,
                n_ood,
                B,
            )
            # print(torch.nonzero(mask == 1, as_tuple=True)[0])
        else:
            hiddens, k_minor, *_ = _get_hiddens(
                task_name,
                exp_name,
                k_minor,
                n_ood,
                B,
            )
        fig = plot_task_vector_exp(
            hiddens,
            k_minor,
            ignore_vocab=ignore_vocab,
        )
        pickle.dump(fig, open(cache_path, "wb"))



def projection_plot(
        task_name: str,
        exp_name: str,
        n_minor: int = 64,
        n_ood: int = 30,
        B: int = 64,
        voc = None,
        layer_index = None,
        minor_projection: bool = False,
        forced_recompute: bool = False,
        **kwargs,
    ):
    config, k_minor = _get_exp_config_and_k_minor(
        task_name,
        exp_name,
        n_minor,
    )
    if layer_index is None:
        if task_name == "linear":
            layer_index = config.model.n_layer - 1
        else:
            layer_index = config.model.num_layers - 1
    
    cache_path = _get_projection_plot_cache_path(
        config,
        task_name,
        exp_name,
        k_minor,
        n_ood,
        B,
        voc,
        layer_index,
        minor_projection,
    )
    if os.path.exists(cache_path) and not forced_recompute:
        with open(cache_path, "rb") as f:
            fig = pickle.load(f)
            fig.show()
    else:
        hover_p = None
        hover_name = "hover"
        plot_mask = None
        if task_name == "coin":
            if kwargs.get("return_p", False):
                hiddens, k_minor, hover_p = _get_hiddens(
                    task_name,
                    exp_name,
                    n_minor,
                    n_ood,
                    B,
                    return_p=True,
                )
                hover_name = "p"
            else:
                hiddens, k_minor = _get_hiddens(
                    task_name,
                    exp_name,
                    n_minor,
                    n_ood,
                    B,
                )
        elif task_name == "dyck":
            hiddens, k_minor, mask = _get_hiddens(
                task_name,
                exp_name,
                n_minor,
                n_ood,
                B,
            )
            idx = torch.nonzero(mask == 1, as_tuple=True)[0]
            a = 0
            b = 2*config.task.dyck_length-1
            plot_mask = None
        else:
            hiddens, k_minor = _get_hiddens(
                task_name,
                exp_name,
                n_minor,
                n_ood,
                B,
            )
        hiddens = hiddens[layer_index].to(torch.float32)  # (K, T, B, V, D) or (K, T, B, D)
        if task_name == "dyck" and kwargs.get("use_mask", False):
            plot_mask = mask[idx[a].item():idx[b].item()+1]
            hiddens = hiddens[:, idx[a].item():idx[b].item()+1]
        elif task_name == "latent":
            if voc is not None:
                hiddens = hiddens[:, voc]
            else:
                K, V, T, B, D = hiddens.shape
                hiddens = hiddens.permute(0, 2, 3, 4, 1).reshape(K, T, B, D*V)

        task_mean = hiddens[:3].mean(dim=(0, 2)).unsqueeze(0)
        task_vecs_over_all_time = hiddens.mean(dim=-2) - task_mean
        if minor_projection:
            is_zero_mean = False
            final_task_vecs = _get_minor_final_task_vecs(
                task_vecs_over_all_time,
                k_minor,
            )
        else:
            is_zero_mean = True
            final_task_vecs = task_vecs_over_all_time[:3, -1]
        
        lambdas, r2_scores, _, _ = estimate_lambda_with_r2(
            final_task_vecs, 
            task_vecs_over_all_time, 
            is_zero_mean=is_zero_mean
            )
        if minor_projection:
            show_final_refs = False
        else:
            show_final_refs = True

        fig = project_with_r2_size(
            task_vecs_over_all_time, 
            final_task_vecs, 
            r2_scores, 
            lambdas, 
            n_minors=k_minor,
            show_final_refs=show_final_refs,
            hover_data=hover_p,
            hover_name=hover_name,
            mask=plot_mask,
            )
        fig.show()
        pickle.dump(fig, open(cache_path, "wb"))


    
from PIL import Image
import matplotlib.pyplot as plt


from typing import Optional


import torch
from typing import Optional


@torch.no_grad()
def posterior_over_models_over_time_per_sample(
    x: torch.Tensor,              # (G, B, T)
    model_probs: torch.Tensor,    # (M, S)  S = num_states / vocab size
    prior: Optional[torch.Tensor] = None,   # None, (M,), (G,M), or (G,B,M)
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute posterior P(M=m | x_{1:t}) for each (group g, sample b) sequence.

    Args:
      x: Long tensor (G,B,T) with token ids in {0,...,S-1}
      model_probs: Float tensor (M,S), each row sums to 1
      prior: optional prior over models:
            - None        -> uniform for all (g,b)
            - (M,)        -> same prior for all (g,b)
            - (G,M)       -> per-group prior, shared across b
            - (G,B,M)     -> per-sequence prior
      eps: numerical floor for log

    Returns:
      post: Float tensor (G,B,T,M) where
            post[g,b,t,m] = P(M=m | x[g,b,0:t+1])
    """
    x = x.long()
    model_probs = model_probs.float()

    if x.ndim != 3:
        raise ValueError(f"x must have shape (G,B,T), got {tuple(x.shape)}")
    if model_probs.ndim != 2:
        raise ValueError(f"model_probs must have shape (M,S), got {tuple(model_probs.shape)}")

    G, B, T = x.shape
    M, S = model_probs.shape
    device = model_probs.device
    dtype = model_probs.dtype

    # ---- prior handling -> make it (G,B,M) ----
    if prior is None:
        prior_gbm = torch.full((G, B, M), 1.0 / M, device=device, dtype=dtype)
    else:
        prior = prior.to(device=device, dtype=dtype)
        if prior.ndim == 1:
            if prior.shape != (M,):
                raise ValueError(f"prior must be (M,), got {tuple(prior.shape)}")
            prior_gbm = prior.view(1, 1, M).expand(G, B, M)
        elif prior.ndim == 2:
            if prior.shape != (G, M):
                raise ValueError(f"prior must be (G,M), got {tuple(prior.shape)}")
            prior_gbm = prior.view(G, 1, M).expand(G, B, M)
        elif prior.ndim == 3:
            if prior.shape != (G, B, M):
                raise ValueError(f"prior must be (G,B,M), got {tuple(prior.shape)}")
            prior_gbm = prior
        else:
            raise ValueError(f"prior must be None, (M,), (G,M), or (G,B,M); got {tuple(prior.shape)}")

        prior_gbm = prior_gbm / prior_gbm.sum(dim=-1, keepdim=True).clamp_min(eps)

    # ---- log probs ----
    logP = torch.log(model_probs.clamp_min(eps))  # (M,S)
    logP_T = logP.t().contiguous()                # (S,M)

    # ll: (G,B,T,M), ll[g,b,t,m] = log p_m(x[g,b,t])
    ll = logP_T[x]

    # cumulative log-weights:
    # logw[g,b,t,m] = log prior[g,b,m] + sum_{tau<=t} ll[g,b,tau,m]
    logw0 = torch.log(prior_gbm.clamp_min(eps)).unsqueeze(2)   # (G,B,1,M)
    logw = logw0 + torch.cumsum(ll, dim=2)                     # (G,B,T,M)

    # normalize across models at each (g,b,t)
    log_post = logw - torch.logsumexp(logw, dim=-1, keepdim=True)  # (G,B,T,M)
    post = torch.exp(log_post)
    return post


def traj_projection_plot(task_name: str,
        exp_name: str,
        n_minor: int = 64,
        n_ood: int = 30,
        B: int = 64,
        b_maj: int = 2,
        b_minor: int = 0,
        b_ood: int = 2,
        voc = None,
        layer_index = None,
        minor_projection: bool = False,
        forced_recompute: bool = False,
        use_mean: bool = True,
        step: Optional[int] = None,
        show_legend: bool = True,
        **kwargs,
    ):
    config, k_minor = _get_exp_config_and_k_minor(
        task_name,
        exp_name,
        n_minor,
    )
    k_major = 3
    k_minor = k_minor  # given
    if layer_index is None:
        if task_name == "linear":
            layer_index = config.model.n_layer - 1
        else:
            layer_index = config.model.num_layers - 1
    
    cache_path = _get_traj_plot_cache_path(
        config,
        task_name,
        exp_name,
        k_minor,
        n_ood,
        B,
        voc,
        layer_index,
        minor_projection,
    )
    if os.path.exists(cache_path) and not forced_recompute:
        #with open(cache_path, "rb") as f:
        #    fig = pickle.load(f)
        #    fig.show()
        img = Image.open(cache_path)
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    else:
        hover_p = None
        hover_name = "hover"
        plot_mask = None
        if task_name == "coin":
            if kwargs.get("return_p", False):
                hiddens, k_minor, hover_p = _get_hiddens(
                    task_name,
                    exp_name,
                    n_minor,
                    n_ood,
                    B,
                    step=step,
                    return_p=True,
                )
                hover_name = "p"
            elif kwargs.get("return_data", False):
                hiddens, k_minor, demo_data, sampler_clone = _get_hiddens(
                    task_name,
                    exp_name,
                    n_minor,
                    n_ood,
                    B,
                    step=step,
                    return_data=True,
                )
            else:
                hiddens, k_minor = _get_hiddens(
                    task_name,
                    exp_name,
                    n_minor,
                    n_ood,
                    B,
                    step=step,
                )
        elif task_name == "dyck":
            hiddens, k_minor, mask = _get_hiddens(
                task_name,
                exp_name,
                n_minor,
                n_ood,
                B,
                step=step,
            )
            idx = torch.nonzero(mask == 1, as_tuple=True)[0]
            a = 0
            b = 2*config.task.dyck_length-1
            plot_mask = None
        else:
            hiddens, k_minor = _get_hiddens(
                task_name,
                exp_name,
                n_minor,
                n_ood,
                B,
                step=step,
            )
        hiddens = hiddens[layer_index].to(torch.float32)  # (K, T, B, V, D) or (K, T, B, D)
        if task_name == "dyck" and kwargs.get("use_mask", False):
            plot_mask = mask[idx[a].item():idx[b].item()+1]
            hiddens = hiddens[:, idx[a].item():idx[b].item()+1]
        elif task_name == "latent":
            if voc is not None:
                hiddens = hiddens[:, voc]
            else:
                K, V, T, B, D = hiddens.shape
                hiddens = hiddens.permute(0, 2, 3, 4, 1).reshape(K, T, B, D*V)

        task_mean = hiddens[:3].mean(dim=(0, 2)).unsqueeze(0)
        task_vecs_over_all_time = hiddens.mean(dim=-2) - task_mean
        if minor_projection:
            is_zero_mean = False
            final_task_vecs = _get_minor_final_task_vecs(
                task_vecs_over_all_time,
                k_minor,
            )
        else:
            is_zero_mean = True
            final_task_vecs = task_vecs_over_all_time[:3, -1]
        if not use_mean:
            # group slices in K
            K, T, B, D = hiddens.shape
            s_major = slice(0, k_major)
            s_mid   = slice(k_major, K - k_minor)
            s_minor = slice(K - k_minor, K)

            def take_group_shared_B(h_group: torch.Tensor, 
                                    b: int,
                                    d_group=None,
                                    ):
                # h_group: (Kg, T, B, D) -> out: (Kg*b, T, D) with b grouped within each k
                Kg = h_group.shape[0]
                idx = torch.randint(0, B, (b,), device=h_group.device)      # (b,)
                x = h_group[:, :, idx, :]                                   # (Kg, T, b, D)
                if d_group is not None:
                    d = d_group[:, idx.to(d_group.device), :]
                x = x.permute(0, 2, 1, 3).contiguous().view(Kg * b, T, D)   # (Kg*b, T, D)
                if d_group is not None:
                    return x, d
                return x

            ood_post = None
            if kwargs.get("return_data", False):
                out_major, d_major = take_group_shared_B(hiddens[s_major], b_maj, demo_data[s_major])
                out_mid, d_mid   = take_group_shared_B(hiddens[s_mid],   b_ood, demo_data[s_mid])
                out_minor, d_minor = take_group_shared_B(hiddens[s_minor], b_minor, demo_data[s_minor])
                ood_post = posterior_over_models_over_time_per_sample(d_mid, sampler_clone.major_p)
                ood_post = ood_post[:, :, :-1]
                
                
            else: 
                out_major = take_group_shared_B(hiddens[s_major], b_maj)
                out_mid   = take_group_shared_B(hiddens[s_mid],   b_ood)
                out_minor = take_group_shared_B(hiddens[s_minor], b_minor)
            

            task_vecs_over_all_time = torch.cat([out_major, out_mid, out_minor], dim=0) - task_mean

        
        lambdas, r2_scores, _, _ = estimate_lambda_with_r2(
            final_task_vecs, 
            task_vecs_over_all_time, 
            is_zero_mean=is_zero_mean
            )

        fig, ax, *_ = project_with_r2_trajectories_group_colors_mpl(
            task_vecs_over_all_time,
            final_task_vecs,
            r2_scores,
            n_minor=k_minor,
            n_ood=n_ood,
            use_mean=use_mean,
            b_major=b_maj,
            b_minor=b_minor,
            b_ood=b_ood,
            step=step,
            show_legend=show_legend,
            title="",
        )

        fig.savefig(cache_path, dpi=600, bbox_inches="tight")



def traj_post_projection_plot(
        task_name: str,
        exp_name: str,
        B: int = 64,
        voc = None,
        layer_index = None,
        minor_projection: bool = False,
        forced_recompute: bool = False,
        step: Optional[int] = None,
        **kwargs,
    ):
    n_minor = 0
    n_ood = 1
    b_minor = 0
    b_maj = 0
    b_ood = 1
    config, k_minor = _get_exp_config_and_k_minor(
        task_name,
        exp_name,
        n_minor,
    )
    k_major = 3
    if layer_index is None:
        if task_name == "linear":
            layer_index = config.model.n_layer - 1
        else:
            layer_index = config.model.num_layers - 1
    
    cache_path = _get_traj_post_plot_cache_path(
        config,
        task_name,
        exp_name,
        k_minor,
        n_ood,
        B,
        voc,
        layer_index,
        minor_projection,
    )
    if os.path.exists(cache_path) and not forced_recompute:
        img = Image.open(cache_path)
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    else:
        if task_name == "coin":
            hiddens, k_minor, demo_data, sampler_clone = _get_hiddens(
                task_name,
                exp_name,
                n_minor,
                n_ood,
                B,
                step=step,
                return_data=True,
            )
        elif task_name == "dyck":
            hiddens, k_minor, mask = _get_hiddens(
                task_name,
                exp_name,
                n_minor,
                n_ood,
                B,
                step=step,
            )
            idx = torch.nonzero(mask == 1, as_tuple=True)[0]
            a = 0
            b = 2*config.task.dyck_length-1
            plot_mask = None
        else:
            hiddens, k_minor = _get_hiddens(
                task_name,
                exp_name,
                n_minor,
                n_ood,
                B,
                step=step,
            )
        hiddens = hiddens[layer_index].to(torch.float32)  # (K, T, B, V, D) or (K, T, B, D)
        if task_name == "dyck" and kwargs.get("use_mask", False):
            plot_mask = mask[idx[a].item():idx[b].item()+1]
            hiddens = hiddens[:, idx[a].item():idx[b].item()+1]
        elif task_name == "latent":
            if voc is not None:
                hiddens = hiddens[:, voc]
            else:
                K, V, T, B, D = hiddens.shape
                hiddens = hiddens.permute(0, 2, 3, 4, 1).reshape(K, T, B, D*V)

        task_mean = hiddens[:3].mean(dim=(0, 2)).unsqueeze(0)
        task_vecs_over_all_time = hiddens.mean(dim=-2) - task_mean
        if minor_projection:
            is_zero_mean = False
            final_task_vecs = _get_minor_final_task_vecs(
                task_vecs_over_all_time,
                k_minor,
            )
        else:
            is_zero_mean = True
            final_task_vecs = task_vecs_over_all_time[:3, -1]
        
        K, T, B, D = hiddens.shape
        s_major = slice(0, k_major)
        s_mid   = slice(k_major, K - k_minor)
        s_minor = slice(K - k_minor, K)

        def take_group_shared_B(h_group: torch.Tensor, 
                                b: int,
                                d_group=None,
                                ):
            # h_group: (Kg, T, B, D) -> out: (Kg*b, T, D) with b grouped within each k
            Kg = h_group.shape[0]
            idx = torch.randint(0, B, (b,), device=h_group.device)      # (b,)
            x = h_group[:, :, idx, :]                                   # (Kg, T, b, D)
            if d_group is not None:
                d = d_group[:, idx.to(d_group.device), :]
            x = x.permute(0, 2, 1, 3).contiguous().view(Kg * b, T, D)   # (Kg*b, T, D)
            if d_group is not None:
                return x, d
            return x
        
        ood_post = None
        out_major, d_major = take_group_shared_B(hiddens[s_major], b_maj, demo_data[s_major])
        out_mid, d_mid   = take_group_shared_B(hiddens[s_mid],   b_ood, demo_data[s_mid])
        out_minor, d_minor = take_group_shared_B(hiddens[s_minor], b_minor, demo_data[s_minor])
        ood_post = posterior_over_models_over_time_per_sample(d_mid, sampler_clone.major_p)
        ood_post = ood_post[0, 0, :-1]
        task_vecs_over_all_time = torch.cat([out_major, out_mid, out_minor], dim=0) - task_mean
        
        _, r2_scores, _, _ = estimate_lambda_with_r2(
            final_task_vecs, 
            task_vecs_over_all_time, 
            is_zero_mean=is_zero_mean
            )
        
        fig, ax = project_with_r2_ood_posterior_colors_mpl(
            task_vecs_over_all_time[0],
            final_task_vecs,
            r2_scores,
            ood_posterior=ood_post,
            # size mapping (still uses R² per point)
            size_min=6,
            size_max=15,
            # alpha ramp (early -> late)
            )
        fig.savefig(cache_path, dpi=600, bbox_inches="tight")
