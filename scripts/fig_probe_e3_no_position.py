"""Generate OLS probe figure for E3 without position nuisance features.

Uses per_position_mean=True to subtract position-specific means from
hidden states, eliminating the need for position nuisance dummies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from icl.latent_markov.analysis import plot_val_r2_across_layers
from icl.utils.unified_interface import get_exp_name

exp_name = get_exp_name("latent", k=-1)

save_path = os.path.join(
    os.path.dirname(__file__), '..', '..', 
    'representation-geometry-two-modes', 'Figs', 'probe_e3.png'
)

fig, all_results = plot_val_r2_across_layers(
    exp_name,
    layers=range(6),
    B=64,
    n_samples=2**13,
    positions=range(170, 190),
    sample_mode="train",
    extraction_point="post_mlp",
    per_position_mean=True,
    include_position_bias=False,
    use_task_identity=True,
    show=False,
    save_path=save_path,
)

print(f"\nFigure saved to: {save_path}")
print("\nResults summary:")
for layer in sorted(all_results.keys()):
    r = all_results[layer]
    d = r.get("diagnostics", {})
    print(f"  Layer {layer}: joint R²={r['val_r2']:.4f}"
          f"  task_marginal={d.get('r2_posterior_only', float('nan')):.4f}"
          f"  token_marginal={d.get('r2_token_only', float('nan')):.4f}"
          f"  logit_marginal={d.get('r2_logit_only', float('nan')):.4f}"
          f"  task_partial={d.get('partial_r2_posterior', float('nan')):.4f}"
          f"  token_partial={d.get('partial_r2_token', float('nan')):.4f}"
          f"  logit_partial={d.get('partial_r2_logit', float('nan')):.4f}")
