"""
Driver script for computing P1 variance.

This script:
1. Loads a model and config
2. Runs compute_hiddens_token_conditioned to collect token-conditioned hiddens
3. Computes P1 variance metrics
4. Outputs results as table and JSON
"""

import sys
import os
import argparse
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_collections import config_flags
from icl.linear.legacy.task_vecs_padded import compute_hiddens_token_conditioned
from icl.linear.p1_variance import (
    compute_p1_variance,
    compute_p1_variance_multi_layer,
    results_to_table,
    save_results_json,
    save_results_multi_layer_json,
)


def main():
    parser = argparse.ArgumentParser(description="Compute P1 variance metrics")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file or config name"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/p1_variance",
        help="Directory to save results"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[5],
        help="Layer indices to analyze (default: [5])"
    )
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        default=None,
        help="Position indices to analyze (default: all)"
    )
    parser.add_argument(
        "--max_unique_tokens",
        type=int,
        default=None,
        help="Maximum number of unique tokens per position (default: None)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=16,
        help="Chunk size for processing tasks (default: 16)"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1008600,
        help="Step for data generation (default: 1008600)"
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=1,
        help="Minimum samples per token group (default: 1)"
    )
    parser.add_argument(
        "--multi_layer",
        action="store_true",
        help="Compute variance for all specified layers"
    )
    
    args = parser.parse_args()
    
    # Load config
    from icl.linear.lr_config import get_config
    if os.path.exists(args.config):
        # Assume it's a config file path
        config = get_config()
        # You might need to load from file - adjust as needed
    else:
        # Assume it's a config name
        config = get_config()
    
    device = torch.device(getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    config.device = device
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Initialize model (adjust based on your model loading code)
    from icl.linear.lr_models import TransformerLin
    model = TransformerLin(
        n_dims=config.task.n_dims,
        n_points=config.task.n_points,
        n_layer=config.model.n_layer,
        n_embd=config.model.n_embd,
        n_head=config.model.n_head,
        seed=config.model.seed,
        dtype=getattr(torch, config.model.dtype, torch.float32),
        pad=config.model.pad,
    )
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()
    model.to(device)
    
    # Initialize train_task
    from icl.linear.lr_task import LinearRegressionTask
    train_task = LinearRegressionTask(config)
    
    print(f"Model loaded. Using device: {device}")
    print(f"Model padding format: {model.pad}")
    
    # Step 1: Collect token-conditioned hiddens
    print("\n" + "="*80)
    print("Step 1: Collecting token-conditioned hidden representations...")
    print("="*80)
    
    all_hiddens, demo_data, token_info = compute_hiddens_token_conditioned(
        config=config,
        model=model,
        train_task=train_task,
        layers=args.layers,
        chunk_size=args.chunk_size,
        step=args.step,
        positions_of_interest=args.positions,
        fix_token_type=args.fix_token_type,
        max_unique_tokens=args.max_unique_tokens,
    )
    
    print(f"Collected hiddens shape: {all_hiddens.shape}")
    print(f"Positions analyzed: {token_info['positions']}")
    print(f"Token type: {token_info['token_type']}")
    print(f"Unique tokens per position: {token_info['n_unique_tokens']}")
    
    # Step 2: Compute P1 variance
    print("\n" + "="*80)
    print("Step 2: Computing P1 variance metrics...")
    print("="*80)
    
    if args.multi_layer:
        results_dict = compute_p1_variance_multi_layer(
            all_hiddens=all_hiddens,
            token_info=token_info,
            layers=args.layers,
            min_count=args.min_count,
        )
        
        # Print tables for each layer
        for layer_idx, results in results_dict.items():
            print(f"\n{results_to_table(results)}")
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "p1_variance_multi_layer.json")
        save_results_multi_layer_json(results_dict, output_path)
        print(f"\nResults saved to: {output_path}")
    else:
        # Single layer (use first specified layer)
        layer_idx = args.layers[0]
        results = compute_p1_variance(
            all_hiddens=all_hiddens,
            token_info=token_info,
            layer_idx=0,  # Index in the layers list, not absolute layer number
            min_count=args.min_count,
        )
        
        # Print table
        print(f"\n{results_to_table(results)}")
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"p1_variance_layer_{layer_idx}.json")
        save_results_json(results, output_path)
        print(f"\nResults saved to: {output_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

