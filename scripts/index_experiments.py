#!/usr/bin/env python
"""
Script to index all experiments in the results directory.

Usage:
    python scripts/index_experiments.py [results_dir]
    
Example:
    python scripts/index_experiments.py results
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from icl.utils.experiment_index import index_all_experiments

if __name__ == "__main__":
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    print(f"Indexing experiments in: {root_dir}")
    index_all_experiments(root_dir=root_dir)
    print("\n✅ Indexing complete!")
    print(f"📊 Open results/experiment_browser.html in your browser to view experiments.")

