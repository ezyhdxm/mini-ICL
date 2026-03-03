"""
Backward-compatibility shim for ``icl.coin.coin_analysis``.

All analysis functions have been moved to sub-modules under
``icl.coin.analysis.*``.  This file re-exports every public name so
that existing imports such as::

    from icl.coin.coin_analysis import plot_coin_task_posterior

continue to work unchanged.
"""

from icl.coin.analysis import *  # noqa: F401,F403

# Re-exports from legacy module
from icl.coin.legacy.coin_analysis_legacy import (  # noqa: F401,E402
    plot_max_stable_rank_vs_k_coin,
    plot_posterior_predictor_loss_vs_k_coin,
    plot_unigram_count_predictor_loss_vs_k_coin,
    plot_stable_rank_vs_maj_r2_min_coin,
    process_ood_minor_metric_coin,
    plot_training_curves_all_experiments_coin,
)
