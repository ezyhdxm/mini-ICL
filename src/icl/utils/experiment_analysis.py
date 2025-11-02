"""
Experiment analysis functions for linear processor.

This module contains experiment-specific analysis functions that were mixed
into the main processor but should be separate.
"""

import numpy as np
import torch
import plotly.graph_objects as go
from sklearn.covariance import MinCovDet

from icl.linear.task_vecs import *
from icl.linear.linear_utils import *
from icl.utils.processor_utils import (
    robust_compositional_timeseries, pairwise_cosine_similarity
)


def process_exp(exp_name, forced=False, layer_index=3, pos=10):
    """
    Process experiment with comprehensive analysis including memory management,
    task vector analysis, and compositional data analysis.
    """
    model, train_task, config = load_model_task_config(exp_name)
    train_task.batch_size = 512
    hiddens, task_vectors = get_task_vector_from_hidden(
        config, model, train_task, 
        layer_index=layer_index, 
        compute_mean=True, 
        return_final=False
    )
    
    # Compute similarity matrices
    task_vec_similarity = pairwise_cosine_similarity(task_vectors[:, -1])
    task_pool_similarity = pairwise_cosine_similarity(train_task.task_pool.squeeze(-1))
    
    # Memory usage reporting
    allocated_memory = torch.cuda.memory_allocated()
    cached_memory = torch.cuda.memory_reserved()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    
    print(f"Allocated Memory: {allocated_memory / 1024**2:.2f} MB")
    print(f"Cached Memory: {cached_memory / 1024**2:.2f} MB")
    print(f"Total Memory: {total_memory / 1024**2:.2f} MB")
    
    # Clean up intermediate results
    print("Cleaning up intermediate results...")
    del model, hiddens, task_vectors
    torch.cuda.empty_cache()

    # Process OOD evolution (would require importing from linear_processor)
    print("Processing OOD...")
    # Note: This function should be imported separately to avoid circular imports
    # lambdas = process_ood_evolve_lambda_metrics(...)
    
    # For demonstration, using dummy data
    # Replace this with actual lambda data
    K, T = 450, 100
    lambdas = np.random.dirichlet([1, 1, 1], (K, T))
    
    # Robust compositional analysis
    X = lambdas[:, 1:] if lambdas.shape[1] > 3 else lambdas
    results = robust_compositional_timeseries(X)
    
    # Extract results for visualization
    spread = results["spread_trace"]
    outlier_flags = results["outlier_flags"]
    time = np.arange(results["loc_comp"].shape[0])
    loc_comp = results["loc_comp"]
    
    # Visualize robust center trajectory
    fig_center = go.Figure()
    for j in range(3):
        fig_center.add_trace(go.Scatter(
            x=time, y=loc_comp[:, j],
            mode='lines',
            name=f"Component {j+1}"
        ))
    
    fig_center.update_layout(
        title="Robust compositional center over time",
        xaxis_title="Time",
        yaxis_title="Robust center (proportion)",
        template="plotly_white"
    )
    fig_center.show()
    
    # Visualize robust spread
    fig_spread = go.Figure()
    fig_spread.add_trace(go.Scatter(
        x=time, y=spread,
        mode='lines',
        line=dict(color="darkred"),
        name="Total variance"
    ))
    fig_spread.update_layout(
        title="Robust spread over time",
        xaxis_title="Time",
        yaxis_title="Robust total variance (ilr space)",
        template="plotly_white"
    )
    fig_spread.show()
    
    return loc_comp[pos]


def robust_location_and_scatter(X, support_fraction=0.85):
    """
    X: array of shape (K, 3), the samples at a single time point
    Returns: robust_location (3,), robust_cov (3x3)
    """
    mcd = MinCovDet(support_fraction=support_fraction, random_state=0).fit(X)
    return mcd.location_, mcd.covariance_