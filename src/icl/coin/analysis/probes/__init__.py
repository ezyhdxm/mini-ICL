"""Linear probes and R² sweep analysis for the Coin task."""

from icl.coin.analysis.probes._internals import (
    _collect_coin_probe_data,
    _fit_coin_probe,
)
from icl.coin.analysis.probes._train import train_linear_hidden_predictor_coin
from icl.coin.analysis.probes._plots import (
    plot_val_r2_across_layers_coin,
    plot_averaging_r2_coin,
)

__all__ = [
    "_collect_coin_probe_data",
    "_fit_coin_probe",
    "train_linear_hidden_predictor_coin",
    "plot_val_r2_across_layers_coin",
    "plot_averaging_r2_coin",
]
