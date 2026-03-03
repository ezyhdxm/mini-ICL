"""
Example script for computing P1 variance.

This is a simpler, more interactive version that can be easily modified.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_collections import ConfigDict
from icl.linear.legacy.task_vecs_padded import compute_hiddens_token_conditioned
from icl.linear.p1_variance import (
    compute_p1_variance,
    compute_p1_variance_multi_layer,
    results_to_table,
    save_results_json,
)


def example_p1_variance():
    """Example usage of P1 variance computation."""
    
    # ============================================================================
    # Configuration - MODIFY THESE
    # ============================================================================
    
    # Model and data config
    model_path = "path/to/your/model.pt"  # UPDATE THIS
    config_path = None  # Or path to config file
    
    # Analysis parameters
    layers_to_analyze = [5]  # Which layers to analyze
    positions_of_interest = None  # None = all positions, or [0, 1, 2, ...]
    max_unique_tokens = 10  # Limit memory usage (None = use all)
    
    # Processing parameters
    chunk_size = 16
    step = 1008600
    min_count = 1  # Minimum samples per token group
    
    # Output
    output_dir = "./results/p1_variance"
    save_json = True
    
    # ============================================================================
    # Load model and config (ADJUST BASED ON YOUR SETUP)
    # ============================================================================
    
    # Example config setup - adjust to match your codebase
    config = ConfigDict()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.task = ConfigDict()
    config.task.n_dims = 20  # UPDATE
    config.task.n_points = 10  # UPDATE
    config.model = ConfigDict()
    config.model.n_layer = 6  # UPDATE
    config.model.n_embd = 128  # UPDATE
    config.model.n_head = 4  # UPDATE
    config.model.seed = 0
    config.model.dtype = "float32"
    config.model.pad = "mapsto"  # Must be "mapsto" for this to work
    
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=config.device)
    
    from icl.linear.lr_models import TransformerLin
    model = TransformerLin(
        n_dims=config.task.n_dims,
        n_points=config.task.n_points,
        n_layer=config.model.n_layer,
        n_embd=config.model.n_embd,
        n_head=config.model.n_head,
        seed=config.model.seed,
        dtype=torch.float32,
        pad=config.model.pad,
    )
    
    # Load state dict (adjust key name if needed)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(config.device)
    
    # Initialize train_task
    from icl.linear.lr_task import LinearRegressionTask
    train_task = LinearRegressionTask(config)
    
    print(f"Model loaded. Device: {config.device}, Padding: {model.pad}")
    
    # ============================================================================
    # Step 1: Collect token-conditioned hiddens
    # ============================================================================
    
    print("\n" + "="*80)
    print("Collecting token-conditioned hidden representations...")
    print("="*80)
    
    all_hiddens, demo_data, token_info = compute_hiddens_token_conditioned(
        config=config,
        model=model,
        train_task=train_task,
        layers=layers_to_analyze,
        chunk_size=chunk_size,
        step=step,
        positions_of_interest=positions_of_interest,
        max_unique_tokens=max_unique_tokens,
    )
    
    print(f"Collected hiddens shape: {all_hiddens.shape}")
    print(f"  - Layers: {len(layers_to_analyze)}")
    print(f"  - Positions: {len(token_info['positions'])}")
    print(f"  - Unique tokens per position: {token_info['n_unique_tokens']}")
    
    # ============================================================================
    # Step 2: Compute P1 variance
    # ============================================================================
    
    print("\n" + "="*80)
    print("Computing P1 variance metrics...")
    print("="*80)
    
    # Compute for each layer
    results_dict = {}
    for i, layer_num in enumerate(layers_to_analyze):
        print(f"\nProcessing layer {layer_num} (index {i} in layers list)...")
        
        results = compute_p1_variance(
            all_hiddens=all_hiddens,
            token_info=token_info,
            layer_idx=i,  # Index in the layers list
            layer_num=layer_num,  # Actual layer number
            min_count=min_count,
        )
        results_dict[layer_num] = results
        
        # Print table
        print(f"\n{results_to_table(results)}")
    
    # ============================================================================
    # Step 3: Save results
    # ============================================================================
    
    if save_json:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for layer_num, results in results_dict.items():
            output_path = f"{output_dir}/p1_variance_layer_{layer_num}.json"
            save_results_json(results, output_path)
            print(f"\nResults for layer {layer_num} saved to: {output_path}")
    
    # ============================================================================
    # Step 4: Extract data for plotting
    # ============================================================================
    
    print("\n" + "="*80)
    print("Summary for plotting:")
    print("="*80)
    
    for layer_num, results in results_dict.items():
        print(f"\nLayer {layer_num}:")
        print("  Positions:", sorted(results.var_pos.keys()))
        print("  Var_Pos values:", [results.var_pos[p] for p in sorted(results.var_pos.keys())])
        print("  Var_Pos_Norm values:", [results.var_pos_norm[p] for p in sorted(results.var_pos_norm.keys())])
    
    return results_dict, token_info


if __name__ == "__main__":
    results_dict, token_info = example_p1_variance()

