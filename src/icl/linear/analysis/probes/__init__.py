"""Probe training for linear regression analysis."""

from icl.linear.analysis.probes._train import train_linear_hidden_predictor
from icl.linear.analysis.probes._cache import _get_linear_hiddens_cached
from icl.linear.analysis.probes._plots import (
    plot_task_vector_r2_linear,
    plot_averaging_r2_linear,
    plot_ancova_separability_linear,
)

__all__ = [
    "train_linear_hidden_predictor",
    "_get_linear_hiddens_cached",
    "plot_task_vector_r2_linear",
    "plot_averaging_r2_linear",
    "plot_ancova_separability_linear",
]
