"""Backward-compat shim: re-exports from unified_hiddens and unified_training.

All existing ``from icl.utils.unified_interface import X`` statements
continue to work unchanged.  New code should import directly from
:mod:`icl.utils.unified_hiddens` or :mod:`icl.utils.unified_training`.
"""

from icl.utils.unified_hiddens import (  # noqa: F401
    _make_linear_eval_generator,
    _get_hiddens,
    pregenerate_task_sequences,
    _compute_hiddens_at_real_tokens,
    _get_hiddens_at_real_positions,
)
from icl.utils.unified_training import (  # noqa: F401
    unified_train,
    _unified_train_worker,
    unified_train_parallel,
    unified_train_major_only_parallel,
)
from icl.utils.unified_path_finder import unified_get_config, get_exp_name  # noqa: F401

# Re-exports used by notebooks (kept at bottom to avoid circular imports)
from icl.latent_markov.analysis.posterior import plot_latent_task_posterior  # noqa: F401
from icl.coin.analysis._helpers import get_token_conditioned_hiddens_coin  # noqa: F401
from icl.dyck.analysis.probes import (  # noqa: F401
    train_linear_softmax_posterior_predictor_dyck_padded,
    plot_id_ood_loss_dyck,
    compute_p1_variance_dyck_padded,
    compute_p1_variance_dyck,
    plot_p1_variance_dyck,
    plot_dyck_task_posterior_padded,
    plot_dyck_task_posterior,
    plot_posterior_predictor_loss_vs_k_dyck_padded,
)
