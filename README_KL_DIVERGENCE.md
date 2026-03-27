# KL Divergence Analysis

This module provides tools to compute and visualize KL divergence between a trained model's predictions and the Bayesian optimal predictor (`GroupUniformKnownBayes`).

## Overview

The analysis compares how well the trained transformer model approximates the Bayesian optimal predictor. This is particularly useful for understanding:
- How quickly the model learns to perform Bayesian inference
- Differences in performance between major (frequent) and minor (rare) tasks
- The effect of training on in-context learning quality

## Key Features

- **Automatic Padding Handling**: Correctly handles padded sequences by extracting model predictions at padded token positions (which predict the next real token)
- **Task Separation**: Separately analyzes major and minor tasks
- **Position-wise Analysis**: Tracks KL divergence across sequence positions
- **Visualization**: Built-in plotting functions for clear presentation

## Module Structure

### Main Functions

1. **`compute_kl_divergence_vs_bayes()`**
   - Computes KL divergence between model and Bayes predictor
   - Returns detailed results for major and minor tasks
   - Parameters:
     - `exp_name`: Experiment name to load model/sampler
     - `n_minor`: Number of minority tasks to sample
     - `batch_size`: Number of sequences per task
     - `step`: Training step (None = final checkpoint)
     - `p_common`: Prior probability for major tasks (default: 0.9)
     - `device`: Computation device

2. **`plot_kl_divergence()`**
   - Visualizes KL divergence vs position
   - Separate curves for major and minor tasks
   - Customizable styling and saving

3. **`analyze_kl_divergence()`**
   - Convenience function combining compute + plot
   - Prints summary statistics
   - One-line analysis

## Usage

### Command Line

```bash
python examples_kl_divergence.py --exp_name <your_experiment> --n_minor 64 --batch_size 64
```

Optional arguments:
- `--step`: Specific training step to analyze
- `--p_common`: Prior probability for major tasks (default: 0.9)
- `--device`: cuda or cpu
- `--save_path`: Path to save plot
- `--no_show`: Don't display plot

### Python Script

```python
from icl.utils.kl_divergence_analysis import analyze_kl_divergence

# Quick analysis
results = analyze_kl_divergence(
    exp_name="train_abc123",
    n_minor=64,
    batch_size=64,
    device="cuda"
)

# Access results
print(f"Major task KL at position 10: {results['kl_major_mean'][10]}")
print(f"Minor task KL at position 10: {results['kl_minor_mean'][10]}")
```

### Jupyter Notebook

Open `notebooks/KL_Divergence_Analysis.ipynb` for an interactive analysis with:
- Step-by-step explanations
- Customizable visualizations
- Per-task analysis
- Multi-experiment comparisons

## Technical Details

### Padding Handling

For padded sequences (common in latent Markov tasks):
- Input sequence: `[s0, PAD, s1, PAD, s2, PAD, ...]`
- Model predicts at PAD positions (odd indices)
- Predictions at PAD position `2*i+1` correspond to predicting `s_{i+1}`
- Bayes predictor receives non-padded sequence: `[s0, s1, s2, ...]`

### GroupUniformKnownBayes

The Bayesian predictor assumes:
- Prior: `p_common` probability mass on first 3 chains (major tasks)
- Remaining `1-p_common` mass on minor tasks (uniform)
- Exact knowledge of all transition matrices
- Perfect Bayesian updating given observations

### KL Divergence Computation

```
KL(model || bayes) = Σ model(s) * log(model(s) / bayes(s))
```

- Computed using `F.kl_div()` with appropriate clamping
- Averaged over batch dimension
- Reported per position

## Results Interpretation

### Typical Patterns

1. **Early Positions**: Higher KL (model hasn't seen much data)
2. **Late Positions**: Lower KL (model performs better inference)
3. **Major vs Minor**: 
   - Major tasks often show lower KL (more training data)
   - Minor tasks may show higher KL (less training exposure)

### Summary Statistics

The analysis reports:
- Mean KL for first 10 positions
- Mean KL for last 10 positions
- Improvement ratio (first/last)

## Example Output

```
Processing 3 major tasks...
Processing 64 minor tasks...

=== Summary Statistics ===
Major tasks: n=3
  Mean KL (first 10 pos): 0.8521
  Mean KL (last 10 pos): 0.1234

Minor tasks: n=64
  Mean KL (first 10 pos): 1.2341
  Mean KL (last 10 pos): 0.2456
```

## Files

- `src/icl/utils/kl_divergence_analysis.py`: Main module
- `examples_kl_divergence.py`: Command-line example
- `notebooks/KL_Divergence_Analysis.ipynb`: Interactive notebook
- `README_KL_DIVERGENCE.md`: This documentation

## Dependencies

- PyTorch
- NumPy
- Matplotlib
- icl.utils.notebook_utils
- icl.latent_markov.markov_latent (for GroupUniformKnownBayes)

## Integration with Existing Code

This module follows the pattern of `unified_interface.py`:
- Uses `nu.load_everything()` to load model/sampler/config
- Uses `nu.load_checkpoint()` for specific training steps
- Compatible with existing latent Markov experiments

## Future Extensions

Potential additions:
- Support for other task types (coin, dyck, linear)
- Different baseline predictors (LatentIDBayes, LatentOODBayes)
- Checkpoint evolution analysis (KL across training)
- Confidence intervals (bootstrap or analytical)
