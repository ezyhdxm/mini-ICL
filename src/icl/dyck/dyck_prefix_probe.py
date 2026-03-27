"""Backward-compatibility shim — re-exports from ``icl.dyck.prefix_probe``."""

from icl.dyck.prefix_probe import *  # noqa: F401,F403
from icl.dyck.prefix_probe._data import (  # noqa: F401
    _load_model_and_sampler,
    _get_task_paths,
    _subsample_by_len,
    _resample_from_pool,
)
from icl.dyck.prefix_probe._train import _evaluate_probe  # noqa: F401
from icl.dyck.prefix_probe._visualize import (  # noqa: F401
    _prefix_to_hex,
    _prefix_to_paren,
    _hierarchical_colors,
    _darken,
)
