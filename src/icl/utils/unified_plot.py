from icl.utils.unified_path_finder import (
    _get_var_plot_cache_path,
    _get_projection_plot_cache_path,
    _get_exp_config_and_k_minor,
    _get_traj_plot_cache_path,
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
from icl.utils.latent_task_vec import project_with_r2_size, project_with_r2_trajectories_group_colors_mpl

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


def traj_projection_plot(
        task_name: str,
        exp_name: str,
        n_minor: int = 64,
        n_ood: int = 30,
        B: int = 64,
        voc = None,
        layer_index = None,
        minor_projection: bool = False,
        forced_recompute: bool = False,
        alpha_start=0.18,
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
        #if minor_projection:
        #    show_final_refs = False
        #else:
        #    show_final_refs = True

        '''
        fig = project_with_r2_trajectories_group_colors(
            task_vecs_over_all_time,
            final_task_vecs,
            r2_scores,
            lambdas,
            n_minors=k_minor,
            # size mapping (still uses R² per point)
            size_min=6,
            size_max=18,
            # alpha ramp (early -> late)
            alpha_start=0.08,
            alpha_end=0.95,
            # styling
            line_width=1.2,
            marker_line_width=0.0,
            width=900,
            height=700,
            )
        fig.show()
        pickle.dump(fig, open(cache_path, "wb"))
        '''




        fig, ax = project_with_r2_trajectories_group_colors_mpl(
            task_vecs_over_all_time, 
            final_task_vecs, 
            r2_scores, 
            lambdas, 
            n_minors=k_minor,
            alpha_start=alpha_start,
        )
        fig.savefig("trajectories.png", dpi=600, bbox_inches="tight")
