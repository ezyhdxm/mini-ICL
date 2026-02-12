"""
Example: Plot stable rank vs positions

This shows how to compute and visualize stable rank across padded positions.
"""

from icl.utils.unified_interface import (
    compute_stable_rank_at_padded_positions,
    plot_stable_rank_vs_positions
)

# ============================================================================
# Example 1: Basic plot - all layers
# ============================================================================

# Compute stable ranks
exp_name = "train_abc123def456"  # Replace with your exp_name
results = compute_stable_rank_at_padded_positions(exp_name, verbose=True)

# Plot all layers
plot_stable_rank_vs_positions(results, backend="matplotlib")

# ============================================================================
# Example 2: Plot average across all layers
# ============================================================================

# Plot the average stable rank across all layers
plot_stable_rank_vs_positions(
    results, 
    average_layers=True,
    backend="matplotlib"
)

# ============================================================================
# Example 3: Plot specific layers only
# ============================================================================

# Plot only specific layers (e.g., early, middle, late layers)
plot_stable_rank_vs_positions(
    results,
    layers=[0, 5, 10, 15],  # Specify which layers to plot
    backend="matplotlib"
)

# ============================================================================
# Example 4: Interactive plotly plot
# ============================================================================

# Use plotly for interactive plots (better for notebooks)
plot_stable_rank_vs_positions(
    results,
    layers=[0, 5, 10, 15],
    backend="plotly"
)

# ============================================================================
# Example 5: Save the plot
# ============================================================================

# Save matplotlib plot
plot_stable_rank_vs_positions(
    results,
    average_layers=True,
    backend="matplotlib",
    save_path="stable_rank_vs_positions.png"
)

# Save plotly plot (as HTML)
plot_stable_rank_vs_positions(
    results,
    average_layers=True,
    backend="plotly",
    save_path="stable_rank_vs_positions.html"
)

# ============================================================================
# Example 6: Quick one-liner for notebook
# ============================================================================

# In a Jupyter notebook, you can do:
results = compute_stable_rank_at_padded_positions("train_abc123def456")
plot_stable_rank_vs_positions(results, average_layers=True)

