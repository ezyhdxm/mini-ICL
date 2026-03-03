"""
Simple example of using compute_stable_rank_at_padded_positions.

Quick start guide for computing stable ranks.
"""

from icl.utils.unified_interface import compute_stable_rank_at_padded_positions
import torch

# ============================================================================
# Minimal Example
# ============================================================================

# Replace with your experiment name
exp_name = "train_abc123def456"

# Compute stable ranks (auto-detects task type)
results = compute_stable_rank_at_padded_positions(exp_name, verbose=True)

# Access results
stable_ranks = results['stable_ranks']  # Shape: (n_layers, n_positions)
print(f"Stable ranks shape: {stable_ranks.shape}")
print(f"Mean stable rank: {stable_ranks.numpy().mean():.2f}")

# ============================================================================
# With Explicit Task Type
# ============================================================================

results = compute_stable_rank_at_padded_positions(
    exp_name=exp_name,
    task_name="linear",  # or "coin", "latent", "dyck"
    verbose=True
)

# ============================================================================
# Quick Analysis
# ============================================================================

# Get stable rank for a specific layer
layer_idx = 5
stable_rank_at_layer = stable_ranks[layer_idx, :]  # All positions for this layer
print(f"Layer {layer_idx} stable ranks: {stable_rank_at_layer.numpy()}")

# Get stable rank for a specific position
position_idx = 0
stable_rank_at_position = stable_ranks[:, position_idx]  # All layers for this position
print(f"Position {position_idx} stable ranks: {stable_rank_at_position.numpy()}")

