"""
Linear probes for Latent Markov task on **non-padded** sequences.

- Joint OLS hidden predictor (posterior + token + logit → hiddens)
- Val R² across layers sweep
"""

from icl.latent_markov.analysis.probes._internals import (
    _collect_multi_layer_data,
    _fit_ols,
    _fit_mlp_r2,
    _fit_probe,
    _print_probe_summary,
)

from icl.latent_markov.analysis.probes._train import (
    train_linear_hidden_predictor,
)

from icl.latent_markov.analysis.probes._plots import (
    plot_val_r2_across_layers,
    plot_averaging_r2_latent,
)

__all__ = [
    "train_linear_hidden_predictor",
    "plot_val_r2_across_layers",
    "plot_averaging_r2_latent",
    "_collect_multi_layer_data",
    "_fit_ols",
    "_fit_mlp_r2",
    "_fit_probe",
    "_print_probe_summary",
]
