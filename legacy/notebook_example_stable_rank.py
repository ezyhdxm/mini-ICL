"""
Notebook-ready example for plotting stable rank vs positions.

Copy and paste this into your Jupyter notebook.
"""

# ============================================================================
# Step 1: Compute stable ranks
# ============================================================================

from icl.utils.unified_interface import (
    compute_stable_rank_at_padded_positions,
    plot_stable_rank_vs_positions
)

# Replace with your experiment name
exp_name = "train_abc123def456"

# Compute stable ranks (uses all minor tasks, no OOD)
results = compute_stable_rank_at_padded_positions(exp_name, verbose=True)

# ============================================================================
# Step 2: Plot stable rank vs positions
# ============================================================================

# Option A: Plot average across all layers
plot_stable_rank_vs_positions(results, average_layers=True)

# Option B: Plot specific layers
plot_stable_rank_vs_positions(results, layers=[0, 5, 10, 15])

# Option C: Plot all layers (might be cluttered if many layers)
plot_stable_rank_vs_positions(results)

# Option D: Interactive plotly plot (better for notebooks)
plot_stable_rank_vs_positions(results, average_layers=True, backend="plotly")

