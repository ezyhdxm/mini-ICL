"""Causal intervention experiments for linear regression analysis.

All public symbols are re-exported here so existing imports continue to work::

    from icl.linear.analysis.interventions import intervene_remove_task_subspace
"""

from icl.linear.analysis.interventions.optimal_orth import (  # noqa: F401
    intervene_optimal_orth_direction,
    plot_optimal_orth_direction_across_layers,
)

from icl.linear.analysis.interventions.remove_task import (  # noqa: F401
    intervene_remove_task_subspace,
    plot_intervention_remove_task_across_layers,
)

from icl.linear.analysis.interventions.inject_task import (  # noqa: F401
    intervene_inject_task_vector,
    plot_inject_task_vector_across_layers,
)

from icl.linear.analysis.interventions.suffstats_orth import (  # noqa: F401
    intervene_remove_suffstats_orth,
    plot_remove_suffstats_orth_across_layers,
)

from icl.linear.analysis.interventions.gaussian_direction import (  # noqa: F401
    plot_gaussian_posterior_probe,
    intervene_remove_gaussian_direction,
    plot_remove_gaussian_direction_across_layers,
)

from icl.linear.analysis.interventions.ood_deltah import (  # noqa: F401
    intervene_remove_ood_deltah_subspace,
    analyze_ood_deltah_direction,
)
