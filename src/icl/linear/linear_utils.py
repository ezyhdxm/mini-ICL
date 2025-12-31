import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from ipywidgets import interact, Dropdown
import ipywidgets as widgets
from typing import Dict, Any, Union, Optional, Tuple
import itertools
from tqdm.notebook import trange, tqdm
from sklearn import linear_model
from sklearn.decomposition import PCA
import os
import re
import glob
import imageio
import matplotlib.pyplot as plt

from icl.linear.lr_models import DiscreteMMSE, Ridge
from icl.linear.lr_task import *
from icl.linear.task_vecs import *
from icl.linear.sampling import sample_union_unit_balls_affine_span_with_weights
from icl.linear.linear_path_utils import get_path_to_exp_dir, load_model_task_config

#TODO: Needs cleaning up


def get_dmmse_posterior(train_task, k):
    """
    Compute Bayesian posterior over task weights using DiscreteMMSE.
    
    Given observed data from a training task, computes the posterior probability
    distribution over which task from the task pool generated the data.
    Uses Gaussian likelihood with known noise scale.
    
    Args:
        train_task: Training task object with task_pool and noise_scale attributes
        k: Task index to sample data from
    
    Returns:
        posterior: Posterior distribution over tasks, shape (B, T, K) where B is batch size,
                  T is sequence length, K is number of tasks
        xs: Input features, shape (B, T, D)
    """
    task_pool = train_task.task_pool.squeeze(-1)  # (n_tasks, D)
    noise_scale = train_task.noise_scale
    xs, ys = train_task.sample_from_task(train_task.task_pool[k], step=0)
    B, T, D = xs.shape
    K = task_pool.shape[0]

    lognum = torch.zeros((B, K), device=xs.device, dtype=xs.dtype)
    posterior = torch.zeros((B, T, K), device=xs.device, dtype=xs.dtype)
    posterior[:, 0, :] = 1.0 / K  # uniform prior at t=0

    for t in range(T-1):
        curr_log = (ys[:,t].unsqueeze(1) - (xs[:,t] @ task_pool.transpose(0,1))).pow(2) / (2 * noise_scale ** 2)
        lognum -= curr_log.squeeze(0).squeeze(0)
        logdenom = torch.logsumexp(lognum, dim=1)
        posterior[:, t+1] = torch.exp(lognum - logdenom[:, None])
    
    return posterior, xs


def get_dmmse_posterior_eval(eval_task, train_task, k):
    """
    Compute Bayesian posterior over task weights for evaluation data.
    
    Similar to get_dmmse_posterior but uses an evaluation task's data
    while still computing posterior over training task pool.
    
    Args:
        eval_task: Evaluation task object to sample data from
        train_task: Training task object with task_pool and noise_scale
        k: Task index to sample data from
    
    Returns:
        posterior: Posterior distribution over tasks, shape (B, T, K)
        xs: Input features, shape (B, T, D)
    """
    task_pool = train_task.task_pool.squeeze(-1)  # (n_tasks, D)
    noise_scale = train_task.noise_scale
    xs, ys = eval_task.sample_from_task(eval_task.task_pool[k], step=0)
    B, T, D = xs.shape
    K = task_pool.shape[0]

    lognum = torch.zeros((B, K), device=xs.device, dtype=xs.dtype)
    posterior = torch.zeros((B, T, K), device=xs.device, dtype=xs.dtype)
    posterior[:, 0, :] = 1.0 / K  # uniform prior at t=0

    for t in range(T-1):
        curr_log = (ys[:,t].unsqueeze(1) - (xs[:,t] @ task_pool.transpose(0,1))).pow(2) / (2 * noise_scale ** 2)
        lognum -= curr_log.squeeze(0).squeeze(0)
        logdenom = torch.logsumexp(lognum, dim=1)
        posterior[:, t+1] = torch.exp(lognum - logdenom[:, None])
    
    return posterior, xs


def estimate_lambda_with_r2(
    task_vecs: torch.Tensor,
    task_vecs_over_all_time: torch.Tensor,
    is_zero_mean: bool = True,
    chunk_size: int = 32,
):
    """
    Memory-friendly version using column-chunking over k.

    Args:
        task_vecs: (num_tasks, d)
        task_vecs_over_all_time: (k, seq_len, d)
        is_zero_mean: exclude last task vec from X and enforce sum-to-1 
        chunk_size: number of columns (k) to solve per batch

    Returns:
        lambdas: (k, seq_len, num_tasks)  [numpy]
        r2_scores: (k, seq_len)           [numpy]
    """
    eps = 1e-12
    device = task_vecs.device
    dtype = task_vecs.dtype

    task_vecs_over_all_time = task_vecs_over_all_time.to(device=device, dtype=dtype)

    k, seq_len, d = task_vecs_over_all_time.shape
    num_tasks = task_vecs.shape[0]

    # outputs on device; convert to numpy at the end
    lambdas = torch.zeros((k, seq_len, num_tasks), dtype=dtype)
    r2_scores = torch.zeros((k, seq_len), dtype=dtype)

    # Design matrix X: (d, num_tasks or num_tasks-1)
    X = task_vecs.T.contiguous()  # (d, num_tasks)
    if is_zero_mean:
        X = X[:, :-1]  # (d, num_tasks-1)

    # Precompute stats used in R^2 denominator per chunk: we still need Y means per chunk
    # We'll compute them on-the-fly to avoid storing big intermediates.

    task_norms = torch.zeros((k, seq_len), dtype=dtype)  
    ortho_norms = torch.zeros((k, seq_len), dtype=dtype)


    with torch.no_grad():
        for t in range(seq_len):
            # Y_full: (d, k)  -- do not clone to avoid extra memory; slice columns per chunk
            Y_full = task_vecs_over_all_time[:, t, :].T  # (d, k)

            # Process k-dimension in chunks
            for start in range(0, k, chunk_size):
                end = min(start + chunk_size, k)

                Y = Y_full[:, start:end]                    # (d, k_chunk)
                # Solve X W = Y for W, least squares (stable)
                # W: (num_tasks-1 or num_tasks, k_chunk)
                W = torch.linalg.lstsq(X, Y).solution

                if is_zero_mean:
                    # Your original "distribute residual equally across all tasks" behavior:
                    # last_lambda = (1 - sum(W)) / num_tasks, then add to all rows (including last)
                    # to make the total sum exactly 1.
                    # residual = 1.0 - W.sum(dim=0, keepdim=True)           # (1, k_chunk)
                    # last_lambda = residual / num_tasks                    # (1, k_chunk)
                    # zero_pad = torch.zeros((1, W.shape[1]), device=device, dtype=dtype)
                    # lambda_full = torch.cat([W, zero_pad], dim=0) + last_lambda  # (num_tasks, k_chunk)
                    # k_chunk = lambda_full.shape[1]
                    # assert torch.allclose(lambda_full.sum(dim=0), torch.ones(k_chunk, device=device, dtype=dtype), atol=1e-6)

                    lambda_full = torch.zeros((num_tasks, W.shape[1]), device=device, dtype=dtype)
                    lambda_full[:-1, :] = W
                    # Make sums = 1 by adding the same offset to all entries (matches your original logic)
                    shift = (1.0 - W.sum(dim=0, keepdim=True)) / num_tasks  # (1, k)
                    lambda_full += shift
                    
                else:
                    lambda_full = W  # (num_tasks, k_chunk)

                # Store lambdas
                lambdas[start:end, t, :] = lambda_full.T.cpu()  # (k_chunk, num_tasks)

                # R^2 per column
                y_pred = X @ W                      # (d, k_chunk)
                resid = Y - y_pred                  # (d, k_chunk)
                ss_res = (resid * resid).sum(dim=0) # (k_chunk,)

                task_norms[start:end, t] = torch.norm(y_pred, dim=0).cpu()  # (k_chunk,)
                ortho_norms[start:end, t] = torch.norm(resid, dim=0).cpu()  # (k_chunk,)

                y_mean = Y.mean(dim=0, keepdim=True)     # (1, k_chunk)
                ss_tot = ((Y - y_mean) ** 2).sum(dim=0)  # (k_chunk,)

                r2 = 1.0 - ss_res / (ss_tot + eps)
                r2.clamp_(min=0.0, max=1.0)
                r2_scores[start:end, t] = r2.cpu()

    return lambdas.numpy(), r2_scores.numpy(), task_norms.numpy(), ortho_norms.numpy()




def estimate_lambda_super_fast(task_vecs, last_eval_task_vecs, 
                               is_zero_mean=True, ridge=0.0, compute_r2=False):
    """
    Fast estimation of lambda weights by solving least squares regression.
    
    Fits each target task vector as a linear combination of reference task vectors.
    Supports optional Ridge regularization and R² computation.
    
    Args:
        task_vecs: Reference task vectors, shape (num_tasks, d)
        last_eval_task_vecs: Target task vectors to fit, shape (k, d)
        is_zero_mean: If True, uses zero-mean constraint trick
        ridge: Ridge regularization strength (0 = no regularization)
        compute_r2: Whether to compute R² goodness-of-fit scores
    
    Returns:
        lambdas: Coefficient weights, shape (k, num_tasks)
        r2_scores: Optional R² scores, shape (k,) if compute_r2=True, else None
    """
    assert task_vecs.ndim == 2, "task_vecs should be of shape (num_tasks, d)"
    assert last_eval_task_vecs.ndim == 2, "last_eval_task_vecs should be of shape (k, d)"

    device = task_vecs.device
    dtype  = task_vecs.dtype

    k, d = last_eval_task_vecs.shape
    num_tasks = task_vecs.shape[0]

    # Design matrix X: (d, num_tasks)  -> fit on first (num_tasks-1) cols if zero-mean trick
    X = task_vecs.T  # (d, num_tasks)
    if is_zero_mean:
        X_fit = X[:, :-1]       # (d, num_tasks-1)
        m = num_tasks - 1
    else:
        X_fit = X
        m = num_tasks

    # Targets stacked as columns: Y in R^{d x k} (use the last time step)
    Y = last_eval_task_vecs.transpose(0, 1).to(device=device, dtype=dtype)  # (d, k)

    # Solve for W in R^{m x k}: minimize ||X_fit W - Y||_F
    if ridge == 0.0:
        # Robust, batched over RHS
        W = torch.linalg.lstsq(X_fit, Y).solution  # (m, k)
    else:
        XtX = X_fit.T @ X_fit                      # (m, m)
        XtY = X_fit.T @ Y                          # (m, k)
        XtX = XtX + ridge * torch.eye(m, device=device, dtype=dtype)
        W = torch.linalg.solve(XtX, XtY)          # (m, k)

    # Build lambdas: (k, num_tasks)
    if is_zero_mean:
        lambdas = torch.zeros((num_tasks, k), device=device, dtype=dtype)
        lambdas[:-1, :] = W
        # Make sums = 1 by adding the same offset to all entries (matches your original logic)
        shift = (1.0 - W.sum(dim=0, keepdim=True)) / num_tasks  # (1, k)
        lambdas += shift
        lambdas = lambdas.T  # (k, num_tasks)
    else:
        lambdas = W.T  # (k, num_tasks)

    if not compute_r2:
        return lambdas, None

    # R^2 (vectorized)
    Y_pred = X_fit @ W                          # (d, k)
    resid  = Y - Y_pred                         # (d, k)
    ss_res = (resid**2).sum(dim=0)              # (k,)
    Y_mean = Y.mean(dim=0, keepdim=True)
    ss_tot = ((Y - Y_mean)**2).sum(dim=0)       # (k,)
    r2_scores = torch.where(ss_tot > 0, 1.0 - ss_res / ss_tot, torch.tensor(float("nan"), device=device, dtype=dtype))

    return lambdas, r2_scores







def get_mse(model, eval_task, train_task, n_points, step: int = 1):
    """
    Compute MSE loss for model predictions compared to oracle baselines.
    
    Evaluates model performance on evaluation tasks and compares to
    oracle predictions from individual training tasks.
    
    Args:
        model: Trained model
        eval_task: Evaluation task object
        train_task: Training task object with task pool
        n_points: Number of points in evaluation sequence
        step: Step size for evaluation
    
    Returns:
        mean_losses: Mean losses per task, shape (n_tasks,)
        oracle_losses: Oracle baseline losses per task, shape (n_tasks, n_points, n_tasks)
    """
    def compute_metrics(k):
        _data, _target = eval_task.sample_from_task(eval_task.task_pool[k], step=step) # (batch, n_points, n_dims)
        tasks = train_task.task_pool.squeeze(-1) # (n_tasks, n_dims)
        oracle_targets = (_data @ tasks.transpose(0,1)).squeeze(-1)  # (batch, n_points, n_task)

        with torch.no_grad():
            preds = model(_data, _target) # (batch, n_points)
        loss = ((preds - _target.to(preds.device))**2).mean(dim=0) # n_points
        oracle_loss = ((preds.unsqueeze(-1) - oracle_targets.to(preds.device))**2).mean(dim=0) # (n_points, n_task)

        return loss, oracle_loss

    n_tasks = len(train_task.task_pool)
    n_eval_tasks = len(eval_task.task_pool) 
    oracle_results = torch.empty((n_eval_tasks, n_points, n_tasks), device=eval_task.device)
    results = torch.empty((n_eval_tasks, n_points), device=eval_task.device)
    
    for k in range(n_eval_tasks):
        results[k], oracle_results[k] = compute_metrics(k) 

    return results.cpu().numpy(), oracle_results.cpu().numpy()


def get_mse_last(model, eval_task, train_task, step: int = 1):
    """
    Compute MSE loss on the last position only.
    
    Similar to get_mse but only evaluates performance on the final token,
    useful for analyzing convergence behavior.
    
    Args:
        model: Trained model
        eval_task: Evaluation task object
        train_task: Training task object with task pool
        step: Step size for evaluation
    
    Returns:
        results: Last position losses per task, shape (n_eval_tasks,)
        oracle_results: Oracle losses per task, shape (n_eval_tasks, n_tasks)
    """
    def compute_metrics(k):
        _data, _target = eval_task.sample_from_task(eval_task.task_pool[k], step=step)
        last_data = _data[:,-1] # (batch, n_dims)
        tasks = train_task.task_pool.squeeze(-1) # (n_tasks, n_dims)
        oracle_targets = (last_data @ tasks.transpose(0,1)).squeeze(-1)  # (batch, n_task)

        with torch.no_grad():
            preds = model(_data, _target) # (batch, n_points)
        loss = ((preds[:,-1:] - _target[:,-1:].to(preds.device))**2).mean(dim=0) # 1
        oracle_loss = ((preds[:,-1:] - oracle_targets.to(preds.device))**2).mean(dim=0) # n_task

        return loss.cpu().item(), oracle_loss.cpu().numpy()

    n_tasks = len(train_task.task_pool)
    n_eval_tasks = len(eval_task.task_pool)
    oracle_results = np.zeros((n_eval_tasks, n_tasks))
    results = np.zeros(n_eval_tasks)
    
    for k in range(n_eval_tasks):
        results[k], oracle_results[k] = compute_metrics(k) 

    return results, oracle_results







def compute_circumcenter(p1, p2, p3):
    """
    Compute the circumcenter (equidistant point) of three points.
    
    Finds the center of the circle passing through the three points,
    which is equidistant from all three points.
    
    Args:
        p1, p2, p3: Three 2D points, each of shape (2,)
    
    Returns:
        Circumcenter point, shape (2,)
    """
    A = np.stack([p2 - p1, p3 - p1])
    b = np.array([
        np.dot(p2, p2) - np.dot(p1, p1),
        np.dot(p3, p3) - np.dot(p1, p1)
    ]) / 2
    x = np.linalg.lstsq(A.T, b, rcond=None)[0]
    return x

def add_half_bisector(a, b, center, length=5.0):
    """
    Compute points for a half-angle bisector line.
    
    Draws a line from the circumcenter in the direction of the angle bisector
    between two anchor points. Used for visualization.
    
    Args:
        a, b: Two anchor points, each shape (2,)
        center: Circumcenter point, shape (2,)
        length: Length of the bisector line
    
    Returns:
        line_pts: Points defining the bisector line, shape (2, 2)
    """
    direction = b - a
    normal = np.array([-direction[1], direction[0]])
    normal = normal / np.linalg.norm(normal)

    midpoint = 0.5 * (a + b)
    if np.dot(normal, midpoint - center) < 0:
        normal = -normal

    line_pts = center[None, :] + np.linspace(0, 1, 2)[:, None] * normal[None, :] * length
    return line_pts








def plot_mse_vs_position(model, samplers_eval, bayes_ood, bayes_id, step: int = 1):
    """
    Plot MSE and prediction differences across positions for ID and OOD tasks.
    
    Compares model performance to Ridge and DiscreteMMSE baselines,
    showing both MSE loss and absolute prediction differences.
    
    Args:
        model: Trained model to evaluate
        samplers_eval: Dictionary of evaluation samplers for different modes
        bayes_ood: Bayesian out-of-distribution estimator
        bayes_id: Bayesian in-distribution (DiscreteMMSE) estimator
        step: Step size for evaluation
    """
    import numpy as np
    import torch
    import plotly.graph_objects as go

    torch.cuda.empty_cache()

    def compute_metrics(mode: str):
        _data, _, _target = samplers_eval[mode](step=step)
        _data = _data.squeeze(0)
        _target = _target.squeeze(0)

        with torch.no_grad():
            preds = model(_data, _target)
        loss = ((preds - _target.to(preds.device))**2).mean(dim=0)

        re_preds = bayes_ood(_data, _target)
        re_loss = ((re_preds - _target.to(re_preds.device))**2).mean(dim=0)

        dmmse_preds = bayes_id(_data, _target)
        dmmse_loss = ((dmmse_preds - _target.to(dmmse_preds.device))**2).mean(dim=0)

        diff_dmmse = np.abs(preds.mean(dim=0).cpu().numpy() - dmmse_preds.mean(dim=0).cpu().numpy())
        diff_re = np.abs(preds.mean(dim=0).cpu().numpy() - re_preds.mean(dim=0).cpu().numpy())

        return (
            loss.cpu().numpy(),
            re_loss.cpu().numpy(),
            dmmse_loss.cpu().numpy(),
            diff_dmmse,
            diff_re,
        )

    results = {
        "ID": compute_metrics("Pretrain"),
        "OOD": compute_metrics("Latent"),
    }

    t = np.arange(1, results["ID"][0].shape[0] + 1)

    fig = go.Figure()
    modes = ["ID", "OOD"]
    views = ["MSE", "Δ"]
    trace_labels = {
        "MSE": ["Model MSE", "Ridge MSE", "dMMSE MSE"],
        "Δ": ["dMMSE Δ", "Ridge Δ"]
    }
    line_styles = [
        dict(width=2),
        dict(width=2, dash="dash"),
        dict(width=2, dash="dot")
    ]

    # Add all traces
    for mode in modes:
        vals = results[mode]
        for i in range(3):
            fig.add_trace(go.Scatter(
                x=t, y=vals[i], mode="lines+markers",
                name=f"{trace_labels['MSE'][i]} ({mode})",
                line=line_styles[i],
                marker=dict(size=4),
                visible=(mode == "ID" and i < 3)  # default
            ))
        for i in range(3, 5):
            fig.add_trace(go.Scatter(
                x=t, y=vals[i], mode="lines+markers",
                name=f"{trace_labels['Δ'][i - 3]} ({mode})",
                line=line_styles[i - 3],
                marker=dict(size=4),
                visible=False
            ))

    # Utility: visibility mask for 4 modes
    def vis_mask(mode, view):
        out = []
        for m in modes:
            for i in range(5):
                out.append((m == mode and ((view == "MSE" and i < 3) or (view == "Δ" and i >= 3))))
        return out

    # One dropdown, 4 buttons
    dropdown = dict(
        buttons=[
            dict(label="MSE vs Position (ID)",
                 method="update",
                 args=[
                     {"visible": vis_mask("ID", "MSE")},
                     {"title": {"text": "MSE vs Position (ID)"},
                      "yaxis": {"title": "MSE"}}
                 ]),
            dict(label="Prediction Difference (ID)",
                 method="update",
                 args=[
                     {"visible": vis_mask("ID", "Δ")},
                     {"title": {"text": "Prediction Difference (ID)"},
                      "yaxis": {"title": "Absolute Difference"}}
                 ]),
            dict(label="MSE vs Position (OOD)",
                 method="update",
                 args=[
                     {"visible": vis_mask("OOD", "MSE")},
                     {"title": {"text": "MSE vs Position (OOD)"},
                      "yaxis": {"title": "MSE"}}
                 ]),
            dict(label="Prediction Difference (OOD)",
                 method="update",
                 args=[
                     {"visible": vis_mask("OOD", "Δ")},
                     {"title": {"text": "Prediction Difference (OOD)"},
                      "yaxis": {"title": "Absolute Difference"}}
                 ]),
        ],
        direction="down",
        x=0.01,
        y=1.15,
        showactive=True,
        xanchor="left"
    )

    fig.update_layout(
        updatemenus=[dropdown],
        title="MSE vs Position (ID)",
        xaxis_title="Position",
        yaxis_title="MSE",
        template="plotly_white",
        height=500,
        width=800,
        legend=dict(title="Legend", itemsizing="constant")
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGray')

    fig.show()







