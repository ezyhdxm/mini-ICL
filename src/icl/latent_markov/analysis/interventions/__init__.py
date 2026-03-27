"""Causal intervention experiments for latent Markov analysis.

All public symbols are re-exported here so existing imports continue to work::

    from icl.latent_markov.analysis.interventions import intervene_remove_task_subspace
"""

from icl.latent_markov.analysis.interventions.remove_task import (  # noqa: F401
    intervene_remove_task_subspace,
    plot_intervention_remove_task_across_layers,
)

from icl.latent_markov.analysis.interventions.inject_task import (  # noqa: F401
    intervene_inject_task_vector,
    plot_inject_task_vector_across_layers,
)

from icl.latent_markov.analysis.interventions.optimal_orth import (  # noqa: F401
    intervene_optimal_orth_direction,
    plot_optimal_orth_direction_across_layers,
)

from icl.latent_markov.analysis.interventions.ood_deltah import (  # noqa: F401
    intervene_remove_ood_deltah_subspace,
)

from icl.latent_markov.analysis.interventions.bigram import (  # noqa: F401
    intervene_remove_bigram_subspace,
    plot_remove_bigram_subspace_across_layers,
)

from icl.latent_markov.analysis.interventions.inject_posterior import (  # noqa: F401
    intervene_direct_injection,
    intervene_inject_posterior,
    intervene_averaging_injection,
    plot_inject_posterior_across_layers,
)
from icl.utils.inject_common import (  # noqa: F401
    plot_inject_posterior_per_position,
)
