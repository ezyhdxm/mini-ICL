"""
Example script demonstrating KL divergence analysis between model and GroupUniformKnownBayes.

This script shows how to use the kl_divergence_analysis module to:
1. Compute KL divergence between a trained model and the Bayesian optimal predictor
2. Visualize the results
3. Compare major vs minor tasks

Usage:
    python examples_kl_divergence.py --exp_name <your_experiment_name>
"""

import argparse
import torch
from icl.utils.kl_divergence_analysis import (
    compute_kl_divergence_vs_bayes,
    plot_kl_divergence,
    analyze_kl_divergence,
)


def main():
    parser = argparse.ArgumentParser(description='KL Divergence Analysis')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Experiment name to analyze')
    parser.add_argument('--n_minor', type=int, default=64,
                        help='Number of minority tasks to sample (default: 64)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for sampling (default: 64)')
    parser.add_argument('--step', type=int, default=None,
                        help='Training step to load (default: None, uses final checkpoint)')
    parser.add_argument('--p_common', type=float, default=0.9,
                        help='Prior probability for major tasks (default: 0.9)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the plot (default: None)')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display the plot')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Run analysis
    results = analyze_kl_divergence(
        exp_name=args.exp_name,
        n_minor=args.n_minor,
        batch_size=args.batch_size,
        step=args.step,
        p_common=args.p_common,
        device=args.device,
        save_path=args.save_path,
        show=not args.no_show,
    )
    
    print("\n=== Analysis Complete ===")
    print(f"Results shape: major={results['kl_major'].shape}, minor={results['kl_minor'].shape}")


if __name__ == '__main__':
    main()
