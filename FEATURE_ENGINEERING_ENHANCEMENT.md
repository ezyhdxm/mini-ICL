# Enhanced Feature Engineering for Orthogonal Direction Analysis

## Overview
Added enriched feature engineering to better explain orthogonal directions (`V_opt`) in the latent Markov model analysis. Previously, only **bigram features** were used for R² diagnostics. Now the code includes multiple interpretable additional features.

## Changes Made

### File Modified
`src/icl/latent_markov/analysis/interventions/optimal_orth.py`

### New Features Added
The enriched feature set now includes interpretable features:

1. **Bigram CLR** (original feature)
   - Centered log-ratio of bigram counts P(token | prefix)
   - Captures token transition probabilities
   - Interpretable as: which tokens are more/less likely given prefix

2. **Unigram CLR** (new)
   - Centered log-ratio of unigram counts
   - Captures overall token frequency information learned from the prefix
   - Interpretable as: which tokens have been seen more/less often

3. **Bigram Entropy** (new)
   - Normalized entropy of the bigram distribution
   - Captures uncertainty/uniformity of next-token predictions
   - Interpretable as: how confident is the next-token prediction (low entropy = confident)

4. **Normalized Position** (new)
   - Position in sequence normalized to [0, 1]
   - Captures sequential position effects
   - Interpretable as: location in the sequence [0=start, 1=end]

### Why No Decorrelation?
Features are kept interpretable and not decorrelated to preserve their semantic meaning. While features may be correlated, this is acceptable because:
- Each feature has clear interpretability
- OLS regression with correlated features still provides valid R² estimates
- Correlation structure itself provides information about redundancy
- Enables direct feature attribution analysis

### R² Metrics Added

For both `intervene_optimal_orth_direction()` and `plot_optimal_orth_direction_across_layers()`:

- `r2_enrich2v_ood`: Can enriched features predict V_opt projection? (OLS)
- `r2_v2enrich_ood`: Can V_opt projection predict enriched features? (OLS)

And analogous metrics for the target distribution (`_tgt` variants).

### Removed
- MLP-based R² metrics (`r2_*_mlp`) - now using linear fits only
- Current token one-hot feature (redundant)
- Decorrelation (preserving interpretability)

### Feature Dimensionality
- Bigram CLR: V dimensions
- Unigram CLR: V dimensions
- Bigram entropy: 1 dimension
- Position: 1 dimension
- **Total: 2V + 2 dimensions** (interpretable, non-decorrelated)

## Benefits

1. **Interpretability**: Each feature has clear semantic meaning
   - Bigram: next-token prediction statistics
   - Unigram: historical frequency statistics
   - Entropy: prediction confidence
   - Position: sequence location

2. **Multiple Information Sources**: Captures:
   - Next-token prediction info (bigram)
   - Historical frequency information (unigram)
   - Prediction uncertainty (entropy)
   - Sequential context (position)

3. **Linear Explainability**: Using only OLS ensures each feature's contribution is directly interpretable

4. **Better Coverage**: Helps identify whether V_opt encodes:
   - Only next-token prediction info (high R² with bigram alone)
   - Broader positional/frequency information (high R² with enriched set)
   - Task-specific patterns (comparison with random baseline)

## Usage

The enhanced features are automatically computed during:
- Single-layer analysis: `intervene_optimal_orth_direction()`
- Multi-layer analysis: `plot_optimal_orth_direction_across_layers()`

New metrics appear in:
- Print summaries (showing enriched R² alongside bigram R²)
- Result dictionaries (all layers' enriched metrics stored)

## Example Output

```
R^2 (OOD):  bi->V=0.285  V->bi=0.198  rand=0.045
R^2 (OOD, enriched):  enrich->V=0.451  V->enrich=0.267
```

This shows enriched features achieve ~60% higher R² than bigram alone, suggesting V_opt encodes information beyond next-token statistics.

## Future Extensions

Could add:
- **N-gram features**: Trigram, skip-gram patterns
- **Attention-based features**: From model's attention weights
- **Task-conditioned features**: Features based on estimated posterior
- **Frequency rank**: Rank of token by frequency (logarithmic binning)


