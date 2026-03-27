"""Public training API for Coin probes."""

from typing import Optional

from icl.coin.analysis.probes._internals import _collect_coin_probe_data, _fit_coin_probe


def train_linear_hidden_predictor_coin(
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
    uniform_sampling: bool = True,
    sample_mode: str = "train",
    skip_baselines: bool = False,
    print_summary: bool = True,
    anchor_minor_samples: Optional[int] = None,
    extraction_point: str = "post_attn",
    per_position_mean: bool = False,
    use_task_identity: bool = False,
) -> dict:
    """Joint OLS: h = [posterior, one_hot(x_t)] @ [W_task; W_tok] + b.

    Joint fitting ensures W_task directions are orthogonal to token
    confounds (Frisch-Waugh-Lovell), eliminating the omitted-variable bias
    that arises when posterior and token features are fit separately.

    Logits are omitted because in the coin task the Bayes-optimal prediction
    is a linear function of the posterior alone (tokens are i.i.d. given
    the task), making logits nearly collinear with the posterior.

    Returns dict with fitted weights, R², partial R², F-tests, and
    design-matrix collinearity diagnostics (VIF, condition number).
    When multiple positions are fit jointly, includes one-hot position
    nuisance features (enabled by ``include_position_bias``) so the
    intercept can vary with position.

    Always uses ``n_ood=0``.

    Parameters
    ----------
    exp_name : str
    layer : int
    B : int
    n_samples : int
    step : int, optional
    n_minor : int, optional
        Capped at ``sampler.n_minor_tasks``.
    verbose : bool
    positions : list, optional
        ``None`` → first 10 positions.
    validation_split : float
    uniform_sampling : bool
    sample_mode : str, default ``"train"``
        Sampling mode passed to ``sampler.generate(mode=...)``.

        - ``"train"`` — mixture of major + minor tasks (default).
        - ``"major"`` — sample **only** from major tasks.
        - ``"minor"`` — sample only from minor tasks.
    skip_baselines : bool
        If True, skips heavier diagnostics (MLP, subspace geometry).
    print_summary : bool, default True
        Print a formatted results table after fitting.

    Returns
    -------
    dict
        Keys aligned with ``train_linear_hidden_predictor`` (linear).
    """
    data = _collect_coin_probe_data(
        exp_name=exp_name, layers=[layer], B=B, n_samples=n_samples,
        step=step, n_minor=n_minor, positions=positions,
        uniform_sampling=uniform_sampling, sample_mode=sample_mode,
        verbose=verbose, anchor_minor_samples=anchor_minor_samples,
        extraction_point=extraction_point,
        use_task_identity=use_task_identity,
    )
    return _fit_coin_probe(
        hiddens_all=data["hiddens_by_layer"][layer],
        posteriors_all=data["posteriors_all"],
        real_tokens_all=data["real_tokens_all"],
        layer=layer, n_tasks=data["n_tasks"], positions=data["positions"],
        include_position_bias=include_position_bias,
        validation_split=validation_split, skip_baselines=skip_baselines,
        print_summary=print_summary, sample_mode=sample_mode,
        n_major=data.get("n_major"),
        per_position_mean=per_position_mean,
        task_ids_all=data.get("task_ids"),
    )
