"""
Example usage of compute_stable_rank_at_padded_positions function.

This script demonstrates how to:
1. Compute stable rank at all padded positions
2. Access and analyze the results
3. Visualize the stable ranks across layers and positions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from icl.utils.unified_interface import compute_stable_rank_at_padded_positions

# ============================================================================
# Example 1: Basic Usage (auto-detect task_name)
# ============================================================================

# Simply provide the experiment name - the function will try to infer the task type
exp_name = "train_abc123def456"  # Replace with your actual exp_name

# Basic call with default parameters
results = compute_stable_rank_at_padded_positions(
    exp_name=exp_name,
    verbose=True  # Set to True to see progress messages
)

# Access the results
stable_ranks = results['stable_ranks']  # Shape: (n_layers, n_positions)
task_name = results['task_name']
n_layers = results['n_layers']
n_positions = results['n_positions']
k_minor = results['k_minor']

print(f"Task: {task_name}")
print(f"Layers: {n_layers}, Positions: {n_positions}")
print(f"Number of minor tasks used: {k_minor}")
print(f"Stable ranks shape: {stable_ranks.shape}")


# ============================================================================
# Example 2: Explicit task_name specification
# ============================================================================

# If you know the task type, you can specify it explicitly
results = compute_stable_rank_at_padded_positions(
    exp_name=exp_name,
    task_name="linear",  # Options: "linear", "coin", "latent", "dyck"
    verbose=True
)


# ============================================================================
# Example 3: Custom batch size and checkpoint step
# ============================================================================

# Use a larger batch size for more samples
results = compute_stable_rank_at_padded_positions(
    exp_name=exp_name,
    task_name="linear",
    B=128,  # Larger batch size
    step=100000,  # Specific checkpoint step (None uses final checkpoint)
    verbose=True
)


# ============================================================================
# Example 4: Analyze stable ranks
# ============================================================================

results = compute_stable_rank_at_padded_positions(exp_name=exp_name, verbose=True)
stable_ranks = results['stable_ranks']  # (n_layers, n_positions)

# Convert to numpy for easier analysis
stable_ranks_np = stable_ranks.numpy()

# Compute statistics
print("\n=== Stable Rank Statistics ===")
print(f"Mean stable rank: {stable_ranks_np.mean():.2f}")
print(f"Std stable rank: {stable_ranks_np.std():.2f}")
print(f"Min stable rank: {stable_ranks_np.min():.2f}")
print(f"Max stable rank: {stable_ranks_np.max():.2f}")

# Stable rank per layer (averaged over positions)
stable_rank_per_layer = stable_ranks_np.mean(axis=1)
print(f"\nStable rank per layer (averaged over positions):")
for layer_idx, sr in enumerate(stable_rank_per_layer):
    print(f"  Layer {layer_idx}: {sr:.2f}")

# Stable rank per position (averaged over layers)
stable_rank_per_position = stable_ranks_np.mean(axis=0)
print(f"\nStable rank per position (averaged over layers):")
for pos_idx, sr in enumerate(stable_rank_per_position):
    print(f"  Position {pos_idx}: {sr:.2f}")


# ============================================================================
# Example 5: Visualize stable ranks
# ============================================================================

def plot_stable_ranks(results, save_path=None):
    """Plot stable ranks as a heatmap."""
    stable_ranks = results['stable_ranks'].numpy()
    n_layers, n_positions = stable_ranks.shape
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    im = ax.imshow(stable_ranks, aspect='auto', cmap='viridis', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Stable Rank', rotation=270, labelpad=20)
    
    # Set labels
    ax.set_xlabel('Padded Position Index', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12)
    ax.set_title(f'Stable Rank at Padded Positions\nTask: {results["task_name"]}, '
                 f'Minor Tasks: {results["k_minor"]}', fontsize=14)
    
    # Set ticks
    ax.set_xticks(range(0, n_positions, max(1, n_positions // 10)))
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 10)))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    return fig

# Generate the plot
results = compute_stable_rank_at_padded_positions(exp_name=exp_name, verbose=True)
plot_stable_ranks(results, save_path='stable_ranks_heatmap.png')


# ============================================================================
# Example 6: Compare stable ranks across different layers
# ============================================================================

results = compute_stable_rank_at_padded_positions(exp_name=exp_name, verbose=True)
stable_ranks = results['stable_ranks'].numpy()

# Plot stable rank vs position for specific layers
fig, ax = plt.subplots(figsize=(10, 6))

# Select a few layers to plot
layers_to_plot = [0, 5, 10, 15] if results['n_layers'] > 15 else list(range(results['n_layers']))

for layer_idx in layers_to_plot:
    if layer_idx < results['n_layers']:
        ax.plot(stable_ranks[layer_idx, :], label=f'Layer {layer_idx}', marker='o', markersize=4)

ax.set_xlabel('Padded Position Index', fontsize=12)
ax.set_ylabel('Stable Rank', fontsize=12)
ax.set_title('Stable Rank vs Position for Different Layers', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================================
# Example 7: Find layers/positions with highest/lowest stable rank
# ============================================================================

results = compute_stable_rank_at_padded_positions(exp_name=exp_name, verbose=True)
stable_ranks = results['stable_ranks'].numpy()

# Find layer and position with maximum stable rank
max_idx = np.unravel_index(np.argmax(stable_ranks), stable_ranks.shape)
max_layer, max_pos = max_idx
print(f"\nMaximum stable rank: {stable_ranks[max_layer, max_pos]:.2f}")
print(f"  At layer {max_layer}, position {max_pos}")

# Find layer and position with minimum stable rank
min_idx = np.unravel_index(np.argmin(stable_ranks), stable_ranks.shape)
min_layer, min_pos = min_idx
print(f"\nMinimum stable rank: {stable_ranks[min_layer, min_pos]:.2f}")
print(f"  At layer {min_layer}, position {min_pos}")


# ============================================================================
# Example 8: Batch processing multiple experiments
# ============================================================================

exp_names = [
    "train_abc123def456",
    "train_xyz789ghi012",
    # Add more experiment names
]

all_results = {}
for exp_name in exp_names:
    try:
        print(f"\nProcessing {exp_name}...")
        results = compute_stable_rank_at_padded_positions(
            exp_name=exp_name,
            verbose=False
        )
        all_results[exp_name] = results
        print(f"  Completed: {results['task_name']}, "
              f"Mean stable rank: {results['stable_ranks'].numpy().mean():.2f}")
    except Exception as e:
        print(f"  Error processing {exp_name}: {e}")

# Compare mean stable ranks across experiments
print("\n=== Comparison Across Experiments ===")
for exp_name, results in all_results.items():
    mean_sr = results['stable_ranks'].numpy().mean()
    print(f"{exp_name}: {mean_sr:.2f}")

