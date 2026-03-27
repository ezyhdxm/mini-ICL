"""
Dyck prefix history probe.

Tests whether the transformer's hidden representations encode the full Dyck
prefix history by training a shared 2D linear projection with separate MLP
classification heads per prefix length.

Architecture (jointly trained):
    h in R^d  -->  z = Wh + b in R^proj_dim           (shared projection)
              -->  MLP_l(z) in R^n_classes(l)          (per-length MLP head)
"""

from ._model import PrefixProbe
from ._data import (
    collect_hidden_states,
    build_prefix_datasets,
)
from ._train import train_prefix_probe
from ._visualize import (
    plot_accuracy_bar,
    plot_training_loss,
    plot_2d_scatter,
    plot_3d_scatter,
)
from ._pipeline import run_prefix_probe

__all__ = [
    "PrefixProbe",
    "collect_hidden_states",
    "build_prefix_datasets",
    "train_prefix_probe",
    "plot_accuracy_bar",
    "plot_training_loss",
    "plot_2d_scatter",
    "plot_3d_scatter",
    "run_prefix_probe",
]
