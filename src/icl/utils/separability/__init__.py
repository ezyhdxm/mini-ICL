"""Task-vector evaluation metrics: R², additive separability (ANOVA / ANCOVA).

- **Task-vector R²**: fraction of hidden-state variance explained by
  (task, token), testing long-context stability.
- **ANOVA separability**: tests θ_{k,a} = θ_k + ν_a for discrete tokens.
- **ANCOVA separability**: tests slope homogeneity for continuous covariates.
"""

from icl.utils.separability._task_vector_r2 import (  # noqa: F401
    TaskVectorR2Result,
    task_vector_r2,
    task_vector_r2_multi,
    _layer_style,
    plot_task_vector_r2,
    plot_task_vector_r2_on_ax,
    print_task_vector_r2_summary,
)
from icl.utils.separability._anova import (  # noqa: F401
    ANOVAResult,
    anova_separability,
    anova_separability_multi,
    plot_anova_separability,
    plot_anova_interaction_on_ax,
    print_anova_summary,
)
from icl.utils.separability._ancova import (  # noqa: F401
    ANCOVAResult,
    ancova_separability,
    mlp_ancova_separability,
    mlp_ancova_separability_from_hiddens,
    mlp_ancova_separability_joint,
    mlp_ancova_separability_joint_from_hiddens,
    ancova_separability_from_hiddens,
    plot_ancova_separability,
    plot_ancova_interaction_on_ax,
    print_ancova_summary,
)
from icl.utils.separability._task_token_vectors import (  # noqa: F401
    estimate_task_vectors_by_averaging,
    per_position_task_vectors,
    per_position_task_vectors_balanced,
    estimate_task_vectors_by_averaging_balanced,
    per_position_token_vectors_balanced,
    estimate_task_and_token_vectors_jointly,
    estimate_token_vectors_by_averaging,
)
from icl.utils.separability._averaging_r2 import (  # noqa: F401
    AveragingR2Result,
    _simplex_project_coeffs,
    task_subspace_r2_at_position,
    plot_averaging_r2_on_ax,
)
from icl.utils.separability._beta_alpha_traj import (  # noqa: F401
    plot_beta_alpha_on_ax,
    plot_beta_alpha_one_col_on_ax,
)
from icl.utils.separability._kl_transition import (  # noqa: F401
    plot_kl_transition_on_ax,
)
