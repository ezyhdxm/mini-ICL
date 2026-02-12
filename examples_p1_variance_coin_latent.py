"""
Examples of computing P1 variance for Coin and Latent tasks.

P1 variance measures conditional residual variance of hidden states given fixed tokens.
This is similar to the linear regression task, but adapted for coin/latent tasks.
"""

import torch
import matplotlib.pyplot as plt
from icl.utils.unified_interface import (
    get_exp_name,
    get_token_conditioned_hiddens_coin,
    get_token_conditioned_hiddens_latent,
)
from icl.linear.p1_variance import (
    compute_p1_variance_multi_layer,
    extract_plotting_data_multi_layer,
)


# ============================================================================
# Example 1: Coin Task P1 Variance
# ============================================================================

def example_coin_p1_variance():
    """Example of computing P1 variance for coin task."""
    
    # Get experiment name
    exp_name = get_exp_name("coin", k=5, vocab_size=27)
    
    # Get token-conditioned hiddens
    # This fixes tokens at specific positions and extracts hiddens at following PAD tokens
    all_hiddens, token_info = get_token_conditioned_hiddens_coin(
        exp_name,
        layers=[2, 4, 5],
        batch_size=64,
        n_minor=64,
        n_ood=30,
        positions_of_interest=list(range(0, 10)) + list(range(10, 100, 5)),
        max_unique_tokens=20,  # Limit to 20 unique tokens per position
    )
    
    print(f"Hiddens shape: {all_hiddens.shape}")
    print(f"Token info: {token_info}")
    
    # Compute P1 variance (same function as linear task)
    results_dict = compute_p1_variance_multi_layer(
        all_hiddens=all_hiddens,
        token_info=token_info,
        layers=[2, 4, 5],
    )
    
    # Extract plotting data
    plotting_data = extract_plotting_data_multi_layer(results_dict)
    
    # Plot normalized variance
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for layer_idx in plotting_data['layers']:
        positions = plotting_data['positions']
        var_pos_norm = plotting_data['var_pos_norm'][layer_idx]
        
        ax.plot(positions, var_pos_norm, 'o-', label=f'Layer {layer_idx}',
                linewidth=2, markersize=6)
    
    ax.set_xlabel('Position (log scale)', fontsize=16)
    ax.set_ylabel('Normalized P1 Variance', fontsize=16)
    ax.set_xscale('log')
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return all_hiddens, token_info, results_dict, plotting_data


# ============================================================================
# Example 2: Latent Task P1 Variance
# ============================================================================

def example_latent_p1_variance():
    """Example of computing P1 variance for latent task."""
    
    # Get experiment name
    exp_name = get_exp_name("latent", k=8, vocab_size=8)
    
    # Get token-conditioned hiddens
    all_hiddens, token_info = get_token_conditioned_hiddens_latent(
        exp_name,
        layers=[2, 4, 5],
        batch_size=96,
        n_minor=256,
        n_ood=40,
        positions_of_interest=list(range(0, 10)) + list(range(10, 64, 4)),
        max_unique_tokens=10,  # Limit to 10 unique tokens per position
    )
    
    print(f"Hiddens shape: {all_hiddens.shape}")
    print(f"Token info: {token_info}")
    
    # Compute P1 variance
    results_dict = compute_p1_variance_multi_layer(
        all_hiddens=all_hiddens,
        token_info=token_info,
        layers=[2, 4, 5],
    )
    
    # Extract plotting data
    plotting_data = extract_plotting_data_multi_layer(results_dict)
    
    # Plot normalized variance
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for layer_idx in plotting_data['layers']:
        positions = plotting_data['positions']
        var_pos_norm = plotting_data['var_pos_norm'][layer_idx]
        
        ax.plot(positions, var_pos_norm, 'o-', label=f'Layer {layer_idx}',
                linewidth=2, markersize=6)
    
    ax.set_xlabel('Position (log scale)', fontsize=16)
    ax.set_ylabel('Normalized P1 Variance', fontsize=16)
    ax.set_xscale('log')
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return all_hiddens, token_info, results_dict, plotting_data


# ============================================================================
# Example 3: Compare P1 Variance Across Tasks
# ============================================================================

def example_compare_p1_variance():
    """Compare P1 variance across different tasks."""
    
    # Coin task
    exp_name_coin = get_exp_name("coin", k=5, vocab_size=27)
    all_hiddens_coin, token_info_coin = get_token_conditioned_hiddens_coin(
        exp_name_coin,
        layers=[5],
        batch_size=64,
        positions_of_interest=list(range(0, 50, 5)),
        max_unique_tokens=15,
    )
    results_coin = compute_p1_variance_multi_layer(
        all_hiddens_coin, token_info_coin, layers=[5]
    )
    plot_data_coin = extract_plotting_data_multi_layer(results_coin)
    
    # Latent task
    exp_name_latent = get_exp_name("latent", k=8, vocab_size=8)
    all_hiddens_latent, token_info_latent = get_token_conditioned_hiddens_latent(
        exp_name_latent,
        layers=[5],
        batch_size=96,
        positions_of_interest=list(range(0, 50, 5)),
        max_unique_tokens=8,
    )
    results_latent = compute_p1_variance_multi_layer(
        all_hiddens_latent, token_info_latent, layers=[5]
    )
    plot_data_latent = extract_plotting_data_multi_layer(results_latent)
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Coin
    positions_coin = plot_data_coin['positions']
    var_coin = plot_data_coin['var_pos_norm'][5]
    ax.plot(positions_coin, var_coin, 'o-', label='Coin (Layer 5)',
            linewidth=2, markersize=6, color='blue')
    
    # Latent
    positions_latent = plot_data_latent['positions']
    var_latent = plot_data_latent['var_pos_norm'][5]
    ax.plot(positions_latent, var_latent, 's-', label='Latent (Layer 5)',
            linewidth=2, markersize=6, color='red')
    
    ax.set_xlabel('Position (log scale)', fontsize=16)
    ax.set_ylabel('Normalized P1 Variance', fontsize=16)
    ax.set_xscale('log')
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Example 4: Detailed Analysis for Single Layer
# ============================================================================

def example_detailed_p1_analysis():
    """Detailed P1 variance analysis for a single layer."""
    
    from icl.linear.p1_variance import compute_p1_variance, results_to_table
    
    exp_name = get_exp_name("coin", k=5, vocab_size=27)
    
    # Get token-conditioned hiddens
    all_hiddens, token_info = get_token_conditioned_hiddens_coin(
        exp_name,
        layers=[5],
        batch_size=64,
        positions_of_interest=list(range(0, 20)),
        max_unique_tokens=20,
    )
    
    # Compute P1 variance for single layer
    results = compute_p1_variance(
        all_hiddens=all_hiddens,
        token_info=token_info,
        layer_idx=0,  # First (and only) layer in the list
        layer_num=5,  # Actual layer number
    )
    
    # Print detailed table
    print(results_to_table(results))
    
    # Access individual metrics
    print(f"\nVariance by position:")
    for pos in sorted(results.var_pos.keys()):
        print(f"  Position {pos}: var_pos={results.var_pos[pos]:.6f}, "
              f"var_pos_norm={results.var_pos_norm[pos]:.6f}")
    
    return results


if __name__ == "__main__":
    # Uncomment to run examples:
    
    # Example 1: Coin task
    # example_coin_p1_variance()
    
    # Example 2: Latent task
    # example_latent_p1_variance()
    
    # Example 3: Compare tasks
    # example_compare_p1_variance()
    
    # Example 4: Detailed analysis
    # example_detailed_p1_analysis()
    
    pass

