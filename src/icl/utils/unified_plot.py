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
    _get_hiddens_at_real_positions,
)
from icl.utils.unified_ood_analysis import (
    _get_minor_final_task_vecs,
)
from icl.utils.unified_interface import get_exp_name
from icl.linear.linear_utils import estimate_lambda_with_r2
from icl.latent_markov.legacy.latent_task_vec import project_with_r2_size
from icl.utils.traj_plot import project_with_r2_trajectories_group_colors_mpl
from icl.utils.posterior_plot import project_with_r2_ood_posterior_colors_mpl
from icl.utils.ood_major_projection_r2 import (
    plot_maj_r2_ood_across_steps,
    plot_maj_r2_ood_across_steps_coin,
    plot_maj_r2_ood_across_steps_linear,
    plot_maj_r2_ood_across_steps_latent,
)

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


# -----------------------------------------------------------------------
# Task-vector geometry across positions
# -----------------------------------------------------------------------

@torch.no_grad()
def plot_task_vector_geometry(
    task_name: str,
    exp_name: str,
    layer_index: int,
    reference_positions: Optional[list] = None,
    plot_positions: Optional[list] = None,
    B: int = 1024,
    step: Optional[int] = None,
    per_position_mean: bool = True,
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
    balanced: bool = False,
    task_batch_size: int = 8,
    task_colors: tuple = ("#0072B2", "#E69F00", "#009E73"),
    marker_every: int = 5,
    figsize: tuple = (8, 12),
    show: bool = True,
) -> dict:
    """Compare task vectors estimated at each position.

    Produces a 4-panel figure (5 panels for discrete tasks ``"coin"``/``"latent"``):

    1. **Norms** -- ||theta_k(t)|| vs position t for each task k.
    2. **Cosine with reference** -- cos(theta_k(t), theta_k^ref) vs t.
    3. **Inter-task cosine** -- cos(theta_i(t), theta_j(t)) for all pairs.
    4. **Subspace membership** -- cosine of theta_k(t) with reference subspace.
    5. **Task-token overlap** (discrete tasks only) -- cosine of theta_k(t)
       with the token subspace span({nu_a(t)}), verifying orthogonality.
       Token vectors are always estimated from a balanced (interventional)
       design regardless of the ``balanced`` flag.

    Parameters
    ----------
    task_name : str
        One of ``"coin"``, ``"linear"``, ``"latent"``.
    exp_name : str
        Experiment folder name.
    layer_index : int
        Which transformer layer to analyse.
    reference_positions : list[int], optional
        Positions used to estimate reference task vectors.  Defaults to
        the last 10 real-token positions.
    plot_positions : list[int], optional
        Which positions to show on the x-axis.  Defaults to all
        available positions.
    B : int
        Batch size per task (natural) or per cell (balanced).
    step : int, optional
        Checkpoint step (None = final).
    per_position_mean : bool
        If True, centre with a position-specific grand mean.
    balanced : bool
        If True, use an orthogonal (token-conditioned) design so that
        task vectors are uniformly averaged over tokens, eliminating
        token leakage.  Only supported for discrete-token tasks
        (``"coin"``, ``"latent"``).
    task_batch_size : int
        Number of tasks batched per forward pass (balanced mode only).
    task_colors : tuple of str
        Colours for each major task.
    marker_every : int
        Place a marker every this many positions (0 to disable).
    figsize : tuple
        Figure size ``(width, height)``.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    dict
        ``task_vecs_by_pos``, ``reference_vecs``, ``norms``,
        ``cos_with_ref``, ``inter_task_cos``, ``fig``, ``axes``.
    """
    import itertools
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from icl.utils.separability import (
        per_position_task_vectors,
        estimate_task_vectors_by_averaging,
        per_position_task_vectors_balanced,
        estimate_task_vectors_by_averaging_balanced,
        per_position_token_vectors_balanced,
    )

    if balanced:
        # -- Balanced (orthogonal) design: token-conditioned hiddens -----
        if task_name == "coin":
            from icl.coin.analysis._helpers import (
                get_token_conditioned_hiddens_coin,
            )
            all_hiddens_tc, _tok_info = get_token_conditioned_hiddens_coin(
                exp_name,
                layers=[layer_index],
                batch_size=B,
                step=step,
                task_batch_size=task_batch_size,
                post_layernorm=post_layernorm,
                extraction_point=extraction_point,
            )
        elif task_name == "latent":
            from icl.latent_markov.analysis.variance import (
                get_token_conditioned_hiddens,
            )
            all_hiddens_tc, _tok_info = get_token_conditioned_hiddens(
                exp_name,
                layers=[layer_index],
                batch_size=B,
                step=step,
                task_batch_size=task_batch_size,
                post_layernorm=post_layernorm,
                extraction_point=extraction_point,
            )
        else:
            raise ValueError(
                f"balanced=True is not supported for task_name='{task_name}' "
                "(requires discrete tokens)"
            )

        # all_hiddens_tc: (1, T, V, K, B, D) -> (T, V, K, B, D)
        hiddens_tc = all_hiddens_tc[0]
        T, V, K, _B, D = hiddens_tc.shape

        tv_pos, grand_means = per_position_task_vectors_balanced(
            hiddens_tc, per_position_mean=per_position_mean,
        )

        if reference_positions is None:
            reference_positions = list(range(max(0, T - 10), T))
        ref_vecs, _ = estimate_task_vectors_by_averaging_balanced(
            hiddens_tc, reference_positions,
        )

        tok_vecs_pos = per_position_token_vectors_balanced(
            hiddens_tc, per_position_mean=per_position_mean,
        )  # (V, T, D)
    else:
        # -- Natural sampling (original path) ----------------------------
        result = _get_hiddens_at_real_positions(
            task_name, exp_name, n_minor=0, n_ood=0, B=B, step=step,
            post_layernorm=post_layernorm,
            extraction_point=extraction_point,
        )
        if isinstance(result, tuple):
            all_hiddens = result[0]
        else:
            all_hiddens = result

        hiddens_layer = all_hiddens[layer_index]        # (K, T, B, D)
        K, T, _B, D = hiddens_layer.shape

        tv_pos, grand_means = per_position_task_vectors(
            hiddens_layer, per_position_mean=per_position_mean,
        )

        if reference_positions is None:
            reference_positions = list(range(max(0, T - 10), T))
        ref_vecs, _ = estimate_task_vectors_by_averaging(
            hiddens_layer, reference_positions,
        )

        # For discrete tasks, fetch token-conditioned data for token subspace
        tok_vecs_pos = None
        if task_name == "coin":
            from icl.coin.analysis._helpers import (
                get_token_conditioned_hiddens_coin,
            )
            _tc_hiddens, _ = get_token_conditioned_hiddens_coin(
                exp_name,
                layers=[layer_index],
                batch_size=B,
                step=step,
                task_batch_size=task_batch_size,
                post_layernorm=post_layernorm,
                extraction_point=extraction_point,
            )
            tok_vecs_pos = per_position_token_vectors_balanced(
                _tc_hiddens[0], per_position_mean=per_position_mean,
            )
        elif task_name == "latent":
            from icl.latent_markov.analysis.variance import (
                get_token_conditioned_hiddens,
            )
            _tc_hiddens, _ = get_token_conditioned_hiddens(
                exp_name,
                layers=[layer_index],
                batch_size=B,
                step=step,
                task_batch_size=task_batch_size,
                post_layernorm=post_layernorm,
                extraction_point=extraction_point,
            )
            tok_vecs_pos = per_position_token_vectors_balanced(
                _tc_hiddens[0], per_position_mean=per_position_mean,
            )

    # -- 4. metrics (all on CPU) -----------------------------------------
    tv_pos = tv_pos.cpu().float()
    ref_vecs = ref_vecs.cpu().float()
    grand_means = grand_means.cpu().float()

    eps = 1e-12
    norms = tv_pos.norm(dim=-1)                     # (K, T)

    ref_norm = ref_vecs.norm(dim=-1, keepdim=True).clamp_min(eps)  # (K, 1)
    pos_norm = norms.clamp_min(eps)                                # (K, T)
    cos_ref = (tv_pos * ref_vecs.unsqueeze(1)).sum(dim=-1) / (
        pos_norm * ref_norm
    )  # (K, T)

    pairs = list(itertools.combinations(range(K), 2))
    inter_cos = torch.zeros(len(pairs), T)
    for idx, (i, j) in enumerate(pairs):
        dot = (tv_pos[i] * tv_pos[j]).sum(dim=-1)
        inter_cos[idx] = dot / (norms[i].clamp_min(eps) * norms[j].clamp_min(eps))

    # Subspace membership: fraction of ||θ_k(t)||² in span(ref_vecs)
    rv = ref_vecs.float()                                    # (K, D)
    _, _S_rv, Vt_rv = torch.linalg.svd(rv, full_matrices=False)
    sub_rank = int((_S_rv > 1e-6 * _S_rv[0]).sum().item())
    P_ref = Vt_rv[:sub_rank].T @ Vt_rv[:sub_rank]           # (D, D)

    proj_norms = (tv_pos @ P_ref).pow(2).sum(dim=-1).sqrt()    # (K, T)
    subspace_cos = proj_norms / norms.clamp_min(eps)           # (K, T)

    # Task-token subspace overlap (discrete tasks: coin/latent)
    task_token_overlap = None
    if tok_vecs_pos is not None:
        tok_vp = tok_vecs_pos.cpu().float()              # (V, T, D)
        V_tok = tok_vp.shape[0]
        task_token_overlap = torch.zeros(K, T)
        for t_idx in range(T):
            tok_mat = tok_vp[:, t_idx, :]                # (V_tok, D)
            _, _S_tok, Vt_tok = torch.linalg.svd(tok_mat, full_matrices=False)
            tok_rank = int((_S_tok > 1e-6 * _S_tok[0].clamp_min(eps)).sum().item())
            if tok_rank == 0:
                continue
            P_tok = Vt_tok[:tok_rank].T @ Vt_tok[:tok_rank]  # (D, D)
            for k in range(K):
                v = tv_pos[k, t_idx]                     # (D,)
                v_norm = v.norm().clamp_min(eps)
                proj = P_tok @ v
                task_token_overlap[k, t_idx] = proj.norm() / v_norm

    if plot_positions is not None:
        pp = np.array(plot_positions)
        norms = norms[:, pp]
        cos_ref = cos_ref[:, pp]
        inter_cos = inter_cos[:, pp]
        subspace_cos = subspace_cos[:, pp]
        if task_token_overlap is not None:
            task_token_overlap = task_token_overlap[:, pp]
        ts = pp
    else:
        ts = np.arange(T)

    colors = list(task_colors[:K])
    task_markers = ["o", "s", "D", "^", "v", "P"]
    pair_markers = ["o", "s", "D"]
    me = max(marker_every, 1) if marker_every > 0 else len(ts) + 1

    # -- 5. plot ---------------------------------------------------------
    n_panels = 5 if task_token_overlap is not None else 4
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(figsize[0], figsize[1] * n_panels / 3),
    )

    # Panel 1: norms
    ax = axes[0]
    for k in range(K):
        ax.plot(ts, norms[k].numpy(), color=colors[k], lw=2,
                marker=task_markers[k % len(task_markers)],
                markevery=me, markersize=5,
                label=rf"$\theta_{{{k+1}}}$")
    ax.set_xlabel("Position $t$", fontsize=12)
    ax.set_ylabel(r"$\|\theta_k(t)\|$", fontsize=12)
    ax.set_title("Task vector norm", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 2: cosine with reference
    ax = axes[1]
    for k in range(K):
        ax.plot(ts, cos_ref[k].numpy(), color=colors[k], lw=2,
                marker=task_markers[k % len(task_markers)],
                markevery=me, markersize=5,
                label=rf"$\theta_{{{k+1}}}$")
    ax.set_xlabel("Position $t$", fontsize=12)
    ax.set_ylabel(r"$\cos(\theta_k(t),\;\theta_k^{\mathrm{ref}})$", fontsize=12)
    ax.set_title("Directional stability", fontsize=13)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 3: inter-task cosine
    ax = axes[2]
    pair_styles = ["-", "--", ":"]
    for idx, (i, j) in enumerate(pairs):
        blend = tuple(
            0.5 * c1 + 0.5 * c2
            for c1, c2 in zip(
                mcolors.to_rgb(colors[i]),
                mcolors.to_rgb(colors[j]),
            )
        )
        ax.plot(
            ts, inter_cos[idx].numpy(), color=blend,
            lw=2, ls=pair_styles[idx % len(pair_styles)],
            marker=pair_markers[idx % len(pair_markers)],
            markevery=me, markersize=5,
            label=rf"$(\theta_{{{i+1}}},\theta_{{{j+1}}})$",
        )
    ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Position $t$", fontsize=12)
    ax.set_ylabel(r"$\cos(\theta_i(t),\;\theta_j(t))$", fontsize=12)
    ax.set_title("Inter-task cosine", fontsize=13)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 4: subspace membership (cosine with task subspace)
    ax = axes[3]
    for k in range(K):
        ax.plot(ts, subspace_cos[k].numpy(), color=colors[k], lw=2,
                marker=task_markers[k % len(task_markers)],
                markevery=me, markersize=5,
                label=rf"$\theta_{{{k+1}}}$")
    ax.axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Position $t$", fontsize=12)
    ax.set_ylabel(
        r"$\|P_{\mathrm{task}}\,\theta_k(t)\| \,/\, \|\theta_k(t)\|$",
        fontsize=12,
    )
    ax.set_title(
        r"Subspace membership (cosine with reference task subspace)",
        fontsize=13,
    )
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 5 (balanced only): task-token subspace overlap
    if task_token_overlap is not None:
        ax = axes[4]
        for k in range(K):
            ax.plot(ts, task_token_overlap[k].numpy(), color=colors[k], lw=2,
                    marker=task_markers[k % len(task_markers)],
                    markevery=me, markersize=5,
                    label=rf"$\theta_{{{k+1}}}$")
        ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("Position $t$", fontsize=12)
        ax.set_ylabel(
            r"$\|P_{\mathrm{tok}}\,\theta_k(t)\| \,/\, \|\theta_k(t)\|$",
            fontsize=12,
        )
        ax.set_title(
            r"Task–token subspace overlap "
            r"(cosine of $\theta_k(t)$ with token subspace)",
            fontsize=13,
        )
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "task_vecs_by_pos": tv_pos,
        "reference_vecs": ref_vecs,
        "grand_means": grand_means,
        "norms": norms,
        "cos_with_ref": cos_ref,
        "inter_task_cos": inter_cos,
        "subspace_cos": subspace_cos,
        "task_token_overlap": task_token_overlap,
        "pairs": pairs,
        "fig": fig,
        "axes": axes,
    }


# ======================================================================
#  Task-component scaling intervention
# ======================================================================

@torch.no_grad()
def intervene_scale_task_component(
    task_name: str,
    exp_name: str,
    layer: int,
    scale_factors: Optional[list] = None,
    B: int = 64,
    n_samples: int = 1000,
    step: Optional[int] = None,
    estimation_positions: Optional[list] = None,
    estimation_B: int = 128,
    per_position_mean: bool = True,
    late_fraction: float = 0.5,
    figsize: tuple = (14, 5),
    show: bool = True,
    verbose: bool = False,
) -> dict:
    r"""Scale the task-subspace component and measure output sensitivity.

    For each scale factor *c*, modifies the hidden state at *layer*:

    .. math::

        h'_t = h_t + (c - 1)\, P_{\text{task}}(h_t - \mu_t)

    When *c* = 1 this is the identity.  When *c* = 0 the task component
    is removed entirely.  When *c* > 1 the task component is amplified.

    **Metric**: KL(softmax(logits_scaled) || softmax(logits_base)) for
    the coin / latent tasks; MSE(preds_scaled, preds_base) for linear
    regression.

    Parameters
    ----------
    task_name : ``"coin"`` | ``"latent"`` | ``"linear"``
    exp_name : str
        Experiment folder name.
    layer : int
        Transformer layer to intervene on.
    scale_factors : list[float], optional
        Values of *c* to sweep.  Default ``[0, .25, .5, .75, 1, 1.25,
        1.5, 2, 3]``.
    B : int
        Batch size per forward pass.
    n_samples : int
        Total number of samples to evaluate.
    step : int, optional
        Checkpoint step (``None`` = final).
    estimation_positions : list[int], optional
        Positions used to estimate task vectors (default: last 10).
    estimation_B : int
        Batch size for hidden-state extraction (task vector estimation).
    per_position_mean : bool
        If True, centre with a position-specific grand mean.
    late_fraction : float
        Fraction of positions considered "late" for the summary panel.
    figsize : tuple
        ``(width, height)`` for the figure.
    show : bool
        Whether to call ``plt.show()``.
    verbose : bool
        Extra logging.

    Returns
    -------
    dict
        ``scale_factors``, ``metric_by_c`` (c → (n_eval_pos,) array),
        ``eval_positions``, ``fig``, ``axes``.
    """
    import gc
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from icl.utils.separability import estimate_task_vectors_by_averaging

    if scale_factors is None:
        scale_factors = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

    is_linear = task_name == "linear"

    # ── 1. Load model & sampler ───────────────────────────────────────
    if is_linear:
        from icl.linear.analysis.interventions._helpers import (
            _load_and_prepare_model,
        )
        from icl.linear.analysis._helpers import _task_positions

        model, train_task, config, device = _load_and_prepare_model(
            exp_name, step=step,
        )
        if step is None:
            step = config.training.total_steps
        n_points = int(config.task.n_points)
        pad_mode = getattr(model, "pad", "mapsto")
        task_pos = _task_positions(pad_mode, n_points, device)
    else:
        _, sampler, config = nu.load_everything(task_name, exp_name)
        if step is None:
            step = config.training.num_epochs
        model, _ = nu.load_checkpoint(
            config, step=step, exp_name=exp_name, return_actual_step=True,
        )
        model.eval().to(config.device)
        device = config.device

    # ── 2. Estimate task vectors & build projector ─────────────────────
    hiddens_all = _get_hiddens_at_real_positions(
        task_name, exp_name, n_minor=0, n_ood=0,
        B=estimation_B, step=step,
    )
    if isinstance(hiddens_all, tuple):
        hiddens_all = hiddens_all[0]

    hiddens_layer = hiddens_all[layer].float()       # (K, T, B_est, D)
    K, T_est, _, D = hiddens_layer.shape

    if estimation_positions is None:
        estimation_positions = list(range(max(0, T_est - 10), T_est))

    task_vecs, grand_mean = estimate_task_vectors_by_averaging(
        hiddens_layer, estimation_positions,
    )

    tv = task_vecs.float()
    _, S_tv, Vt_tv = torch.linalg.svd(tv, full_matrices=False)
    rank = int((S_tv > 1e-6 * S_tv[0]).sum().item())
    P_task = (Vt_tv[:rank].T @ Vt_tv[:rank]).to(device)

    if per_position_mean:
        mu_per_pos = hiddens_layer.mean(dim=(0, 2)).to(device)  # (T_est, D)
    else:
        mu_global = grand_mean.to(device)                       # (D,)

    if verbose:
        print(f"[scale] rank={rank}, K={K}, T_est={T_est}, D={D}")

    del hiddens_all, hiddens_layer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ── 3. Determine eval positions ───────────────────────────────────
    #  For linear, the model output is (B, n_points) — already indexed
    #  by point, not by sequence position.  So eval_positions_seq should
    #  be point indices.  The hook, however, must modify hidden states at
    #  the *sequence*-level positions given by task_pos.
    if is_linear:
        eval_positions_seq = list(range(n_points))
        eval_positions_label = list(range(n_points))
        task_pos_list = task_pos.cpu().tolist()
    else:
        T_max = T_est
        eval_positions_seq = list(range(T_max))
        eval_positions_label = eval_positions_seq
        task_pos_list = None

    n_eval = len(eval_positions_seq)

    # ── 4. Hook factory ───────────────────────────────────────────────
    def _make_hook(c):
        if c == 1.0:
            return None

        def _hook(mod, inp, out):
            h = out if torch.is_tensor(out) else out[0]
            h_new = h.clone()
            if is_linear:
                _tp = task_pos.to(h.device)
                h_sub = h_new[:, _tp, :]             # (B, n_points, D)
                if per_position_mean:
                    mu = mu_per_pos.unsqueeze(0)      # (1, n_points, D)
                else:
                    mu = mu_global.unsqueeze(0).unsqueeze(0)
                tc = (h_sub - mu) @ P_task
                h_new[:, _tp, :] = h_sub + (c - 1.0) * tc
            else:
                T_h = min(T_est, h_new.shape[1])
                h_sub = h_new[:, :T_h, :]
                if per_position_mean:
                    mu = mu_per_pos[:T_h].unsqueeze(0)
                else:
                    mu = mu_global.unsqueeze(0).unsqueeze(0)
                tc = (h_sub - mu) @ P_task
                h_new[:, :T_h, :] = h_sub + (c - 1.0) * tc
            return h_new if torch.is_tensor(out) else (h_new,) + out[1:]
        return _hook

    def _get_layer_module():
        if is_linear:
            return model.transformer.blocks[layer].attn_block
        return model.layers[layer].attn_block

    # ── 5. KL / MSE helpers ───────────────────────────────────────────
    def _kl_per_pos(logits_a, logits_b, positions):
        """KL(softmax(a) || softmax(b)) at given positions, (B, n_pos)."""
        la = torch.log_softmax(logits_a[:, positions], dim=-1)
        lb = torch.log_softmax(logits_b[:, positions], dim=-1)
        return (la.exp() * (la - lb)).sum(-1)

    def _mse_per_pos(preds_a, preds_b, positions):
        """MSE at given positions, (B, n_pos)."""
        return (preds_a[:, positions] - preds_b[:, positions]).pow(2)

    # ── 6. Evaluation loop ────────────────────────────────────────────
    accum = {c: [] for c in scale_factors}
    fwd_chunk = B
    n_done = 0
    bi = 0

    if is_linear:
        orig_bs = int(train_task.batch_size)
        train_task.batch_size = B

    while n_done < n_samples:
        # Generate data
        if is_linear:
            demo_data, _, demo_target = train_task.sample_batch(
                step=bi + 99999, is_eval=False,
            )
            demo_data = demo_data.to(device)
            demo_target = demo_target.to(device)
            cur_B = demo_data.shape[0]
        else:
            gen = sampler.generate(
                mode="major", task=None, num_samples=B, epochs=1,
            )
            samples = gen[0] if isinstance(gen, (tuple, list)) else gen
            if samples.dim() == 3:
                samples = samples.squeeze(0)
            samples = samples.to(device)
            cur_B = samples.shape[0]

        # Baseline forward
        if is_linear:
            logits_base = model(demo_data, demo_target)
        else:
            logits_base = model(samples)

        # Scaled forwards
        for c in scale_factors:
            hook_fn = _make_hook(c)
            if hook_fn is None:
                metric_c = torch.zeros(cur_B, n_eval, device=device)
            else:
                handle = _get_layer_module().register_forward_hook(hook_fn)
                try:
                    if is_linear:
                        logits_scaled = model(demo_data, demo_target)
                    else:
                        logits_scaled = model(samples)
                finally:
                    handle.remove()

                if is_linear:
                    metric_c = _mse_per_pos(
                        logits_scaled, logits_base, eval_positions_seq,
                    )
                else:
                    metric_c = _kl_per_pos(
                        logits_scaled, logits_base, eval_positions_seq,
                    )
                del logits_scaled

            accum[c].append(metric_c.cpu())

        n_done += cur_B
        bi += 1
        if is_linear:
            del demo_data, demo_target
        else:
            del samples
        del logits_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if is_linear:
        train_task.batch_size = orig_bs

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ── 7. Aggregate ──────────────────────────────────────────────────
    metric_by_c = {}
    for c in scale_factors:
        stacked = torch.cat(accum[c], dim=0)          # (N, n_eval)
        metric_by_c[c] = stacked.mean(dim=0).numpy()  # (n_eval,)

    positions_arr = np.array(eval_positions_label)
    metric_name = "MSE" if is_linear else "KL"

    # ── 8. Plot ───────────────────────────────────────────────────────
    fw, fh = figsize
    norm = mcolors.TwoSlopeNorm(vmin=min(scale_factors), vcenter=1.0,
                                vmax=max(scale_factors))
    cmap = cm.coolwarm

    # Figure 1: per-position output change
    fig1, ax1 = plt.subplots(figsize=(fw, fh))
    for c in scale_factors:
        col = cmap(norm(c))
        lw = 2.5 if c == 1.0 else 1.4
        ls = "--" if c == 1.0 else "-"
        ax1.plot(positions_arr, metric_by_c[c], color=col, lw=lw, ls=ls,
                 label=f"c={c:.2g}")
    ax1.set_xlabel("Position $t$", fontsize=13)
    ax1.set_ylabel(
        rf"${metric_name}(\mathrm{{scaled}} \| \mathrm{{baseline}})$",
        fontsize=13,
    )
    ax1.set_title(
        f"{task_name} — layer {layer} — per-position output change",
        fontsize=14,
    )
    ax1.legend(fontsize=12, ncol=2, loc="best")
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig1)

    # Figure 2: mean metric vs c
    n_late = max(1, int(n_eval * late_fraction))
    late_idx = slice(n_eval - n_late, n_eval)
    mean_late = [metric_by_c[c][late_idx].mean() for c in scale_factors]
    mean_all = [metric_by_c[c].mean() for c in scale_factors]

    fig2, ax2 = plt.subplots(figsize=(fw, fh))
    ax2.plot(scale_factors, mean_all, "o-", color="#555555", lw=2, ms=6,
             label="all positions")
    ax2.plot(scale_factors, mean_late, "s-", color="#D32F2F", lw=2, ms=6,
             label=f"late {n_late} positions")
    ax2.axvline(1.0, color="grey", lw=1, ls="--", alpha=0.5)
    ax2.set_xlabel("Scale factor $c$", fontsize=13)
    ax2.set_ylabel(
        rf"Mean ${metric_name}(\mathrm{{scaled}} \| \mathrm{{baseline}})$",
        fontsize=13,
    )
    ax2.set_title(
        f"{task_name} — layer {layer} — output sensitivity to scaling",
        fontsize=14,
    )
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig2)

    return {
        "scale_factors": scale_factors,
        "metric_by_c": metric_by_c,
        "metric_name": metric_name,
        "eval_positions": positions_arr,
        "late_fraction": late_fraction,
        "layer": layer,
        "rank": rank,
        "figs": (fig1, fig2),
        "axes": (ax1, ax2),
    }


# ──────────────────────────────────────────────────────────────────────
#  Residual-removal intervention (reference task + token subspace)
# ──────────────────────────────────────────────────────────────────────

def intervene_residual_removal(
    task_name: str,
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples: int = 2000,
    eval_positions: Optional[list] = None,
    step: Optional[int] = None,
    estimation_positions: Optional[list] = None,
    estimation_B: int = 128,
    per_position_mean: bool = True,
    per_position_tokens: bool = False,
    per_position_covariate: bool = False,
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
    task_batch_size: int = 8,
    marker_every: int = 5,
    figsize: tuple = (8, 4),
    show: bool = True,
    verbose: bool = False,
) -> dict:
    r"""Remove the residual from the additive decomposition and measure output change.

    Tests whether the additive model
    :math:`h_t \approx \mu_t + \theta_k + \nu_{s_t}` captures everything
    functionally important **using reference (late-position) task vectors**.

    At each evaluation position *p*, the hidden state is replaced with:

    .. math::

        \hat{h}_p = \mu(p) + P(p)\,(h_p - \mu(p))

    where :math:`P(p)` is the orthogonal projector onto
    ``span(θ_1^ref, ..., θ_K^ref, ν_1(p), ..., ν_V(p))`` when
    ``per_position_tokens=True``, or a single fixed projector onto
    ``span(θ_1^ref, ..., θ_K^ref, ν_1^ref, ..., ν_V^ref)`` when
    ``per_position_tokens=False``.  Task vectors are always the
    reference ones estimated from late positions.

    For the **linear** task the projector spans task vectors *and*
    covariate directions from the OLS slope ``B``.  When
    ``per_position_covariate=False`` (default), a single ``B`` is pooled
    across estimation positions; when ``True``, a separate ``B(p)`` is
    fitted at every evaluation position to form per-position projectors.

    **Metric**: KL (coin/latent) or MSE (linear).

    Parameters
    ----------
    task_name : str
        ``"coin"``, ``"latent"``, or ``"linear"``.
    layer : int
        Layer at which to intervene.
    estimation_positions : list, optional
        Positions to pool for reference task/token vectors.  ``None`` ->
        last 30 positions (coin/latent) or last 10 positions (linear).
    estimation_B : int
        Batch size for hidden-state extraction (task vector estimation).
    task_batch_size : int
        Samples per (task, token) cell for token-vector estimation
        (coin/latent only).
    per_position_mean : bool
        If True, use position-specific grand mean when centering.
    per_position_tokens : bool
        If True, token vectors vary by position (combined with fixed
        reference task vectors to form per-position projectors).
        Ignored for ``"linear"`` (no discrete tokens).
    per_position_covariate : bool
        If True, fit a separate covariate slope ``B(p)`` at each
        evaluation position and build per-position projectors.
        Only affects ``"linear"``; ignored for coin/latent.
    marker_every : int
        Place a marker every *n* positions on the plot lines.

    Returns
    -------
    dict with keys: layer, eval_positions, metric_per_pos, metric_means, ...
    """
    import gc
    import matplotlib.pyplot as plt
    from icl.utils.separability import (
        estimate_task_vectors_by_averaging,
        per_position_token_vectors_balanced,
    )

    assert task_name in ("coin", "latent", "linear"), (
        f"residual removal supports coin/latent/linear, got {task_name!r}"
    )
    is_linear = task_name == "linear"

    # ── Load model & config ──────────────────────────────────────────
    if is_linear:
        from icl.linear.analysis.interventions._helpers import (
            _load_and_prepare_model, _extract_hiddens_for_pool,
        )
        from icl.linear.analysis._helpers import _task_positions
        from icl.linear.linear_ood_analysis import (
            _setup_eval_task,
        )

        model, train_task, config, device = _load_and_prepare_model(
            exp_name, step=step,
        )
        if step is None:
            step = config.training.total_steps
        n_points = int(config.task.n_points)
        K_major = int(train_task.n_tasks)
        D_model = int(config.model.n_embd)
        pad_mode = getattr(model, "pad", "mapsto")
        task_pos = _task_positions(pad_mode, n_points, device)
        T_max = n_points

        if estimation_positions is None:
            estimation_positions = list(range(max(0, T_max - 10), T_max))
        if eval_positions is None:
            eval_positions = list(range(1, T_max))
    else:
        _, sampler, config = nu.load_everything(task_name, exp_name)
        if step is None:
            step = config.training.num_epochs
        model, _ = nu.load_checkpoint(
            config, step=step, exp_name=exp_name, return_actual_step=True,
        )
        model.eval().to(config.device)
        device = config.device

        K_major = sampler.n_major_tasks
        seq_len = sampler.seq_len
        T_max = seq_len - 1

        if estimation_positions is None:
            estimation_positions = list(range(max(0, T_max - 30), T_max))
        if eval_positions is None:
            eval_positions = list(range(1, T_max))

    # ── Extract hidden states (natural sampling) ─────────────────────
    if is_linear:
        major_pool = train_task.task_pool.squeeze(-1).to(device)[:K_major]
        eval_task_est = _setup_eval_task(config, major_pool, estimation_B, device)
        eval_task_est.batch_size = estimation_B
        demo_data_est = eval_task_est.sample_data(step=step).to(device)
        hiddens_layer, _ = _extract_hiddens_for_pool(
            model, eval_task_est, demo_data_est,
            step=step, layer=layer, task_pos=task_pos, D=D_model,
            n_tasks=K_major, chunk=8,
            post_layernorm=post_layernorm,
            extraction_point=extraction_point,
        )
        hiddens_layer = hiddens_layer.float()  # (K, T, B_est, D)
        demo_data_est = demo_data_est.cpu().float()  # (B_est, T, D_x)
        del eval_task_est
    else:
        hiddens_all, _ = _get_hiddens_at_real_positions(
            task_name=task_name, exp_name=exp_name,
            n_minor=0, n_ood=0, B=estimation_B, step=step,
            post_layernorm=post_layernorm,
            extraction_point=extraction_point,
        )
        hiddens_layer = hiddens_all[layer].float()  # (K, T, B_est, D)
    K, T, _B_est, D = hiddens_layer.shape

    # ── Reference task vectors (natural, late positions) ──────────────
    ref_task_vecs, _ = estimate_task_vectors_by_averaging(
        hiddens_layer, estimation_positions,
    )  # (K, D), grand_mean (D,)

    # ── Per-position grand means (natural) ───────────────────────────
    if per_position_mean:
        grand_means = hiddens_layer.mean(dim=(0, 2))   # (T, D)
    else:
        grand_means = hiddens_layer.mean(dim=(0, 1, 2)).unsqueeze(0).expand(T, D)
    grand_means = grand_means.cpu().float()

    # ── Token vectors (balanced) — coin/latent only ────────────────
    V = 0
    tok_vecs_pos = None
    ref_token_vecs = None

    if is_linear:
        per_position_tokens = False
        D_x = demo_data_est.shape[-1]
        ref_covariate_vecs = None

        if per_position_covariate:
            pass  # B(p) fitted per position in projector-building section
        else:
            # Estimate a single covariate slope B (no task-covariate interaction):
            #   h_{k,t,b} ≈ μ_t + θ_k + B x_{t,b}
            # Pool residuals across ALL tasks and estimation positions.
            est_pos_t = estimation_positions
            X_one = (demo_data_est[:, est_pos_t, :]           # (B_est, n_est, D_x)
                     .permute(1, 0, 2)                         # (n_est, B_est, D_x)
                     .reshape(-1, D_x))                        # (n_est*B_est, D_x)
            all_residuals = []
            for k in range(K):
                h_k = hiddens_layer[k, est_pos_t, :, :]       # (n_est, B_est, D)
                mu_est = grand_means[est_pos_t]                # (n_est, D)
                r_k = (h_k
                       - mu_est.unsqueeze(1)
                       - ref_task_vecs[k].unsqueeze(0).unsqueeze(0))
                all_residuals.append(r_k.reshape(-1, D))       # (n_est*B_est, D)
            R_pool = torch.cat(all_residuals, dim=0)           # (K*n_est*B_est, D)
            X_pool = X_one.repeat(K, 1)                        # (K*n_est*B_est, D_x)
            B_ref = torch.linalg.lstsq(X_pool, R_pool).solution  # (D_x, D)
            ref_covariate_vecs = B_ref.float()                 # (D_x, D)
            del demo_data_est
    else:
        if task_name == "coin":
            from icl.coin.analysis._helpers import (
                get_token_conditioned_hiddens_coin,
            )
            tc_hiddens_all, _ = get_token_conditioned_hiddens_coin(
                exp_name,
                layers=[layer],
                batch_size=estimation_B,
                step=step,
                task_batch_size=task_batch_size,
                post_layernorm=post_layernorm,
                extraction_point=extraction_point,
            )
        else:
            from icl.latent_markov.analysis.variance import (
                get_token_conditioned_hiddens,
            )
            tc_hiddens_all, _ = get_token_conditioned_hiddens(
                exp_name,
                layers=[layer],
                batch_size=estimation_B,
                step=step,
                task_batch_size=task_batch_size,
                post_layernorm=post_layernorm,
                extraction_point=extraction_point,
            )

        cell_h = tc_hiddens_all[0].float()  # (T, V, K, B, D)
        V = cell_h.shape[1]

        if per_position_tokens:
            tok_vecs_pos = per_position_token_vectors_balanced(
                cell_h, per_position_mean=per_position_mean,
            ).cpu().float()
        else:
            cell_means = cell_h.mean(dim=3)                            # (T, V, K, D)
            est_cell_means = cell_means[estimation_positions]          # (T_est, V, K, D)
            ref_token_means = est_cell_means.mean(dim=(0, 2))          # (V, D)
            ref_token_grand = ref_token_means.mean(dim=0)              # (D,)
            ref_token_vecs = (ref_token_means - ref_token_grand.unsqueeze(0)).cpu().float()

        del cell_h, tc_hiddens_all

    if not is_linear:
        del hiddens_all
    _keep_hiddens = is_linear and per_position_covariate
    if not _keep_hiddens:
        del hiddens_layer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ── Build projector(s) ───────────────────────────────────────────
    ref_task_vecs_f = ref_task_vecs.cpu().float()       # (K, D)
    eps = 1e-6
    use_per_pos_proj = per_position_tokens or (is_linear and per_position_covariate)

    def _build_projector(vecs):
        """SVD-based orthogonal projector from a (N, D) matrix."""
        _, S, Vt = torch.linalg.svd(vecs, full_matrices=False)
        r = int((S > eps * S[0].clamp_min(eps)).sum().item())
        if r == 0:
            return torch.zeros(D, D), 0
        return (Vt[:r].T @ Vt[:r]), r

    valid_positions = [p for p in eval_positions if p < T]

    if is_linear and per_position_covariate:
        P_by_pos = {}
        ranks_by_pos = {}
        hiddens_f = hiddens_layer.cpu().float()          # (K, T, B_est, D)
        demo_data_f = demo_data_est.cpu().float()        # (B_est, T, D_x)
        for p in valid_positions:
            x_p = demo_data_f[:, p, :]                   # (B_est, D_x)
            X_p = x_p.repeat(K, 1)                       # (K*B_est, D_x)
            all_r = []
            for k in range(K):
                r_k = (hiddens_f[k, p, :, :]
                       - grand_means[p].unsqueeze(0)
                       - ref_task_vecs_f[k].unsqueeze(0))
                all_r.append(r_k)                        # (B_est, D)
            R_p = torch.cat(all_r, dim=0)                # (K*B_est, D)
            B_p = torch.linalg.lstsq(X_p, R_p).solution  # (D_x, D)
            all_vecs = torch.cat(
                [ref_task_vecs_f, B_p.float()], dim=0,
            )                                            # (K + D_x, D)
            P_by_pos[p], ranks_by_pos[p] = _build_projector(all_vecs)
        del hiddens_f, demo_data_f, hiddens_layer, demo_data_est
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    elif per_position_tokens:
        P_by_pos = {}
        ranks_by_pos = {}
        for p in valid_positions:
            all_vecs = torch.cat(
                [ref_task_vecs_f, tok_vecs_pos[:, p, :]], dim=0,
            )
            P_by_pos[p], ranks_by_pos[p] = _build_projector(all_vecs)
    elif is_linear:
        all_linear_vecs = torch.cat(
            [ref_task_vecs_f, ref_covariate_vecs], dim=0,
        )  # (K + D_x, D)
        P_ref, rank = _build_projector(all_linear_vecs)
    else:
        all_ref_vecs = torch.cat([ref_task_vecs_f, ref_token_vecs], dim=0)
        P_ref, rank = _build_projector(all_ref_vecs)

    if verbose:
        v_str = f", V={V}" if not is_linear else f", D_x={D_x}"
        print(f"[residual removal] layer={layer}, K={K}{v_str}, D={D}")
        if use_per_pos_proj:
            ranks_list = [ranks_by_pos[p] for p in valid_positions]
            label = ("per-pos covariate (OLS)" if per_position_covariate
                     else "per-position token vectors")
            print(f"  {label}: rank range "
                  f"[{min(ranks_list)}, {max(ranks_list)}]")
        else:
            mode = "task + covariate (OLS)" if is_linear else "task + token"
            print(f"  reference subspace rank = {rank} ({mode})")
        print(f"  estimation_positions = "
              f"{estimation_positions[:3]}..{estimation_positions[-1]}")

    # ── Metric helpers ──────────────────────────────────────────────
    def _kl(logits_a, logits_b):
        """KL(softmax(a) || softmax(b))."""
        lp_a = torch.log_softmax(logits_a, dim=-1)
        lp_b = torch.log_softmax(logits_b, dim=-1)
        return (lp_a.exp() * (lp_a - lp_b)).sum(-1)

    metric_name = "MSE" if is_linear else "KL"

    # ── Layer module helper ──────────────────────────────────────────
    def _get_layer_module():
        if is_linear:
            return model.transformer.blocks[layer].attn_block
        return model.layers[layer].attn_block

    # ── Intervention loop ────────────────────────────────────────────
    fwd_chunk = B
    n_done = 0
    metric_accum = {p: [] for p in valid_positions}
    bi = 0

    if not use_per_pos_proj:
        P_ref_dev = P_ref.to(device)

    if is_linear:
        orig_bs = int(train_task.batch_size)
        train_task.batch_size = B
        task_pos_list = task_pos.cpu().tolist()

    while n_done < n_samples:
        # ── Generate data ────────────────────────────────────────────
        if is_linear:
            demo_data, _, demo_target = train_task.sample_batch(
                step=bi + 99999, is_eval=False,
            )
            demo_data = demo_data.to(device)
            demo_target = demo_target.to(device)
            cur_B = demo_data.shape[0]
        else:
            gen = sampler.generate(
                mode="major", task=None, num_samples=B, epochs=1,
            )
            samples = gen[0] if isinstance(gen, (tuple, list)) else gen
            if samples.dim() == 3:
                samples = samples.squeeze(0)
            samples = samples.to(device)
            cur_B = samples.shape[0]

        with torch.no_grad():
            if is_linear:
                preds_base = model(demo_data, demo_target)
            else:
                logits_base = model(samples)

        for p in valid_positions:
            if is_linear:
                seq_pos = task_pos_list[p]
            else:
                if p + 1 >= samples.shape[1]:
                    continue
                seq_pos = p

            _mu_p = grand_means[p].to(device)
            if use_per_pos_proj:
                _P_p = P_by_pos[p].to(device)
            else:
                _P_p = P_ref_dev

            def _hook_fn(_mu=_mu_p, _P=_P_p, _sp=seq_pos):
                def _hook(mod, inp, out):
                    h = out if torch.is_tensor(out) else out[0]
                    h_new = h.clone()
                    h_centered = h_new[:, _sp, :] - _mu.unsqueeze(0)
                    h_new[:, _sp, :] = _mu.unsqueeze(0) + h_centered @ _P
                    return (h_new if torch.is_tensor(out)
                            else (h_new,) + out[1:])
                return _hook

            while True:
                try:
                    handle = _get_layer_module().register_forward_hook(
                        _hook_fn(),
                    )
                    try:
                        with torch.no_grad():
                            if is_linear:
                                preds_proj = model(demo_data, demo_target)
                            else:
                                logits_proj = model(samples)
                    finally:
                        handle.remove()

                    if is_linear:
                        mse_vals = (
                            (preds_base[:, p] - preds_proj[:, p]) ** 2
                        ).cpu()
                        metric_accum[p].append(mse_vals)
                        del preds_proj
                    else:
                        kl_vals = _kl(
                            logits_base[:, p], logits_proj[:, p],
                        ).cpu()
                        metric_accum[p].append(kl_vals)
                        del logits_proj
                    break
                except torch.cuda.OutOfMemoryError:
                    fwd_chunk = max(1, fwd_chunk // 2)
                    torch.cuda.empty_cache()

        n_done += cur_B
        bi += 1
        if is_linear:
            del demo_data, demo_target, preds_base
        else:
            del samples, logits_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if is_linear:
        train_task.batch_size = orig_bs

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ── Aggregate ────────────────────────────────────────────────────
    metric_per_pos = {}
    for p in valid_positions:
        if metric_accum[p]:
            metric_per_pos[p] = torch.cat(metric_accum[p]).numpy()

    positions_arr = np.array(sorted(metric_per_pos.keys()))
    m_means = np.array([metric_per_pos[p].mean() for p in positions_arr])
    m_medians = np.array([np.median(metric_per_pos[p]) for p in positions_arr])
    m_q75 = np.array([np.percentile(metric_per_pos[p], 75) for p in positions_arr])
    m_q90 = np.array([np.percentile(metric_per_pos[p], 90) for p in positions_arr])
    zeros = np.zeros_like(positions_arr, dtype=float)

    # ── Marker indices ───────────────────────────────────────────────
    me = max(1, marker_every)
    mark_idx = list(range(0, len(positions_arr), me))

    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.fill_between(positions_arr, zeros, m_q90, alpha=0.12,
                     color="#1976D2", label="0–90 pctl")
    ax1.fill_between(positions_arr, zeros, m_q75, alpha=0.22,
                     color="#1976D2", label="0–75 pctl")
    ax1.plot(positions_arr, m_medians, "-o", color="#1976D2", lw=2,
             label="median", markevery=mark_idx, ms=5)
    ax1.plot(positions_arr, m_means, "--s", color="#D32F2F", lw=1.5,
             label="mean", markevery=mark_idx, ms=4)
    ax1.set_xlabel("Position $t$", fontsize=13)
    if is_linear:
        ax1.set_ylabel(
            r"$\mathrm{MSE}(\text{original},\, \text{projected})$",
            fontsize=13,
        )
    else:
        ax1.set_ylabel(
            r"$\mathrm{KL}(\text{original} \| \text{projected})$",
            fontsize=13,
        )

    if is_linear and per_position_covariate:
        tok_mode = "per-pos cov"
    elif is_linear:
        tok_mode = "task + cov"
    elif per_position_tokens:
        tok_mode = "per-pos tokens"
    else:
        tok_mode = "ref tokens"

    if use_per_pos_proj:
        ranks_list = [ranks_by_pos[p] for p in valid_positions]
        rank_str = f"rank {min(ranks_list)}-{max(ranks_list)}"
    else:
        rank_str = f"rank {rank}"
    ax1.set_title(
        f"{task_name} -- layer {layer} -- residual removal "
        f"({tok_mode}, {rank_str})",
        fontsize=13,
    )
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    overall_mean = m_means.mean()
    print(f"\n{'=' * 60}")
    print(f"Residual Removal  ({task_name}, layer {layer})")
    print(f"{'=' * 60}")
    if is_linear:
        cov_label = "per-position" if per_position_covariate else "pooled"
        print(f"  Subspace: task + covariate (OLS {cov_label}, D_x={D_x})")
    else:
        print(f"  Token vectors: {'per-position' if per_position_tokens else 'reference (pooled)'}")
    if use_per_pos_proj:
        print(f"  Subspace rank range: [{min(ranks_list)}, {max(ranks_list)}]")
    else:
        print(f"  Reference subspace rank = {rank}")
    print(f"  Overall mean {metric_name}(original, projected) = {overall_mean:.6f}")
    print(f"  Per-position mean {metric_name} range: "
          f"[{m_means.min():.6f}, {m_means.max():.6f}]")
    print()

    result = {
        "layer": layer,
        "task_name": task_name,
        "eval_positions": positions_arr,
        "metric_per_pos": metric_per_pos,
        "metric_means": m_means,
        "metric_medians": m_medians,
        "metric_overall_mean": overall_mean,
        "metric_name": metric_name,
        "per_position_tokens": per_position_tokens,
        "per_position_covariate": per_position_covariate,
        "fig": fig,
        "ax": ax1,
    }
    if use_per_pos_proj:
        result["subspace_ranks"] = np.array(
            [ranks_by_pos.get(p, 0) for p in positions_arr]
        )
    else:
        result["subspace_rank"] = rank
    return result
