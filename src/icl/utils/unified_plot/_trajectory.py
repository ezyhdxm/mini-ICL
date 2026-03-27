from icl.utils.unified_path_finder import (
    _get_exp_config_and_k_minor,
    _get_traj_plot_cache_path,
    _get_traj_post_plot_cache_path,
)
from icl.utils.unified_interface import _get_hiddens
from icl.utils.unified_ood_analysis import _get_minor_final_task_vecs
from icl.linear.linear_utils import estimate_lambda_with_r2
from icl.utils.traj_plot import project_with_r2_trajectories_group_colors_mpl
from icl.utils.posterior_plot import project_with_r2_ood_posterior_colors_mpl
from icl.utils.unified_plot._variance_projection import (
    posterior_over_models_over_time_per_sample,
)

from typing import Optional
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch


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
