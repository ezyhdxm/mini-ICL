"""Public training API for latent Markov probes."""

from typing import Optional

from icl.latent_markov.analysis.probes._internals import (
    _collect_multi_layer_data,
    _fit_probe,
    _print_probe_summary,
)


def train_linear_hidden_predictor(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples: int = 1000,
    step: Optional[int] = None,
    n_minor: Optional[int] = None,
    verbose: bool = False,
    positions: Optional[list] = None,
    validation_split: float = 0.2,
    include_position_bias: bool = True,
    include_logit: bool = True,
    uniform_sampling: bool = True,
    sample_mode: str = "train",
    skip_baselines: bool = False,
    print_summary: bool = True,
    device: Optional[str] = None,
    use_log_posterior: bool = False,
    anchor_minor_samples: Optional[int] = None,
    extraction_point: str = "post_attn",
    per_position_mean: bool = False,
    use_task_identity: bool = False,
) -> dict:
    """Joint OLS: h = [phi(pi), onehot_t, (optional) logit_t] @ W + b.

    phi is log or identity (controlled by *use_log_posterior*).
    Joint fitting ensures W_task directions are orthogonal to token/logit
    confounds (Frisch-Waugh-Lovell).

    Returns dict with fitted weights, R^2, partial R^2, F-tests, and
    design-matrix collinearity diagnostics (VIF, condition number).
    When multiple positions are fit jointly, includes one-hot position
    nuisance features (enabled by ``include_position_bias``) so the
    intercept can vary with position.

    If *use_task_identity* is True, the posterior feature is replaced with
    a one-hot task-identity encoding (ground-truth discrete label) to
    eliminate the dummy-variable collinearity.
    """
    data = _collect_multi_layer_data(
        exp_name=exp_name, layers=[layer], B=B, n_samples=n_samples,
        step=step, n_minor=n_minor, positions=positions,
        uniform_sampling=uniform_sampling, sample_mode=sample_mode,
        device=device, verbose=verbose,
        anchor_minor_samples=anchor_minor_samples,
        extraction_point=extraction_point,
        use_task_identity=use_task_identity,
    )
    results = _fit_probe(
        hiddens_all=data["hiddens_by_layer"][layer],
        posteriors_all=data["posteriors"],
        logits_all=data["logits"],
        real_tokens_all=data["real_tokens"],
        n_major=data["n_major"],
        n_tasks=data["n_tasks"],
        layer=layer,
        positions=data["positions"],
        validation_split=validation_split,
        include_position_bias=include_position_bias,
        include_logit=include_logit,
        use_log_posterior=use_log_posterior,
        skip_baselines=skip_baselines,
        sample_mode=sample_mode,
        per_position_mean=per_position_mean,
        task_ids_all=data.get("task_ids"),
    )
    if print_summary:
        _print_probe_summary(results, sample_mode=sample_mode)
    return results
