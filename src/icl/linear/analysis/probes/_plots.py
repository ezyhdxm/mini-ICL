"""Plotting functions for linear regression probes."""

from typing import Optional

from icl.linear.analysis.probes._cache import _get_linear_hiddens_cached
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)


def plot_task_vector_r2_linear(
    exp_name: str,
    layers: Optional[list] = None,
    positions: Optional[list] = None,
    batch_size: int = 64,
    chunk_size: int = 16,
    step: Optional[int] = None,
    n_minor: int = 0,
    n_ood: int = 0,
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
    verbose: bool = False,
    figsize: tuple = (5, 3.2),
    log_x: bool = True,
    show: bool = True,
    show_ylabel: bool = True,
    print_summary: bool = True,
) -> dict:
    """Task-token vector R² for the linear regression task.

    For continuous covariates, the task-token vector R² is the R² of the
    full ANCOVA model ``h = task_intercept + B_k x_t``, which allows
    each task its own slope.  This measures what fraction of h's
    variance is explained by knowing (task, x_t).

    Results are cached: calling ``plot_ancova_separability_linear``
    afterward with the same data parameters reuses the hidden states.

    Parameters
    ----------
    exp_name : str
    layers : list, optional
    positions : list, optional
    batch_size : int
    chunk_size : int
    step : int, optional
    n_minor, n_ood : int
    post_layernorm : bool
    extraction_point : str
        ``"post_attn"`` (default) or ``"post_mlp"`` (residual stream).
    verbose : bool
    figsize, log_x, show, show_ylabel, print_summary : plot options

    Returns
    -------
    dict
        ``{'all_hiddens', 'demo_data', 'ancova_results', 'fig'}``.
    """
    from icl.utils.separability import (
        ancova_separability_from_hiddens,
        print_ancova_summary,
    )
    import matplotlib.pyplot as plt

    all_hiddens, demo_data, layers, *_ = _get_linear_hiddens_cached(
        exp_name, layers, batch_size, chunk_size, step, n_minor, n_ood, verbose,
        post_layernorm=post_layernorm,
        extraction_point=extraction_point,
    )

    ancova_results = ancova_separability_from_hiddens(
        all_hiddens=all_hiddens,
        demo_data=demo_data,
        layers=layers,
        positions=positions,
    )

    if print_summary:
        print_ancova_summary(ancova_results, positions=positions)

    from icl.utils.separability import _layer_style

    fig, ax = plt.subplots(figsize=figsize)
    sorted_layers = sorted(ancova_results.keys())
    for l_num in sorted_layers:
        pos_results = ancova_results[l_num]
        pos_list = sorted(pos_results.keys())
        if not pos_list:
            continue
        vals = [1.0 - pos_results[p].r2_full for p in pos_list]
        ax.plot(
            pos_list, vals, label=str(l_num),
            **_layer_style(l_num, len(pos_list)),
        )

    ax.set_xlabel("Position", fontsize=13)
    if show_ylabel:
        ax.set_ylabel("Residual variance ratio", fontsize=13)
    if log_x and len(pos_list) > 1 and min(pos_list) >= 0:
        ax.set_xscale("symlog", linthresh=1)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=12)
    _ncol = 2 if len(sorted_layers) > 6 else 1
    ax.legend(title="Layer", fontsize=12, title_fontsize=12,
              framealpha=0.9, loc="best", ncol=_ncol,
              borderaxespad=0.3, handlelength=2.2)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "all_hiddens": all_hiddens,
        "demo_data": demo_data,
        "ancova_results": ancova_results,
        "fig": fig,
    }


def plot_averaging_r2_linear(
    exp_name: str,
    layers: Optional[list] = None,
    estimation_positions: Optional[list] = None,
    evaluation_positions: Optional[list] = None,
    batch_size: int = 64,
    chunk_size: int = 16,
    step: Optional[int] = None,
    n_minor: int = 0,
    n_ood: int = 0,
    fit_token: str = "linear",
    per_position_mean: bool = True,
    eval_subset: str = "major",
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
    simplex: bool = True,
    verbose: bool = False,
    figsize: tuple = (5, 3.2),
    log_x: bool = True,
    show: bool = True,
    show_ylabel: bool = True,
    plots: str = "both",
) -> dict:
    """Task-subspace R² using averaging-based task vectors.

    Parameters (plotting)
    ---------------------
    plots : "both" | "task" | "additive"
        Which figure(s) to produce.  ``"task"`` = task-only R²;
        ``"additive"`` = task + token R²; ``"both"`` (default) = both.

    Estimates task vectors from major tasks at late positions by
    conditional averaging, then measures the fraction of hidden-state
    variance explained by the task subspace (and optionally a linear
    token model) at every evaluation position.

    Parameters
    ----------
    exp_name : str
    layers : list, optional — ``None`` -> all layers
    estimation_positions : list, optional
        Positions used to estimate task vectors (default: last 10).
    evaluation_positions : list, optional
        Positions at which R² is computed (default: all).
    batch_size, chunk_size, step, n_minor, n_ood : data parameters
    fit_token : "none" | "linear"
        How to model the token/covariate effect on the residual.
    per_position_mean : bool
        If True (default), subtract the sample mean at each position
        before projection (removes position encoding).  If False,
        subtract the global grand mean from the estimation step at
        all positions.
    eval_subset : "all" | "major" | "minor" | "ood"
        Which task subset to evaluate R² on.  Task vectors are always
        estimated from major tasks.
    verbose : bool
    figsize, log_x, show, show_ylabel : plot options

    Returns
    -------
    dict with 'all_hiddens', 'demo_data', 'task_vecs', 'results',
    'fig_task', 'fig_additive'.
    """
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from icl.utils.separability import (
        estimate_task_vectors_by_averaging,
        task_subspace_r2_at_position,
    )

    all_hiddens, demo_data, layers, n_major, _n_ood, _k_minor = \
        _get_linear_hiddens_cached(
            exp_name, layers, batch_size, chunk_size, step, n_minor, n_ood, verbose,
            post_layernorm=post_layernorm,
            extraction_point=extraction_point,
        )

    L, n_tasks, n_points, B, D = all_hiddens.shape

    _subset_ranges = {
        "all":   (0, n_tasks),
        "major": (0, n_major),
        "ood":   (n_major, n_major + _n_ood),
        "minor": (n_major + _n_ood, n_major + _n_ood + _k_minor),
    }
    if eval_subset not in _subset_ranges:
        raise ValueError(
            f"eval_subset must be one of {list(_subset_ranges)}, got {eval_subset!r}"
        )
    eval_start, eval_end = _subset_ranges[eval_subset]
    if eval_end <= eval_start:
        raise ValueError(
            f"No tasks in subset '{eval_subset}' "
            f"(n_major={n_major}, n_ood={_n_ood}, k_minor={_k_minor})"
        )

    if estimation_positions is None:
        estimation_positions = list(range(max(0, n_points - 10), n_points))
    if evaluation_positions is None:
        evaluation_positions = list(range(n_points))

    if verbose:
        logger.info(
            f"[averaging R²] estimation from major tasks [0:{n_major}], "
            f"eval on '{eval_subset}' [{eval_start}:{eval_end}], "
            f"estimation_pos={estimation_positions}, "
            f"eval_pos={len(evaluation_positions)} positions, "
            f"fit_token={fit_token}, per_position_mean={per_position_mean}"
        )

    results: dict = {}
    task_vecs_by_layer: dict = {}

    eval_positions_list = [p for p in evaluation_positions if p < n_points]
    for l_idx, l_num in tqdm(enumerate(layers), total=L, desc="averaging R² (layers)", unit="layer"):
        if l_idx >= L:
            continue

        tv, gm = estimate_task_vectors_by_averaging(
            all_hiddens[l_idx, :n_major], estimation_positions,
        )
        task_vecs_by_layer[l_num] = tv

        if verbose:
            norms = tv.norm(dim=1)
            logger.info(
                f"  Layer {l_num}: task vec norms = "
                + ", ".join(f"{n:.3f}" for n in norms.tolist())
            )

        fixed_mean = None if per_position_mean else gm

        layer_results: dict = {}
        for pos in eval_positions_list:
            h_pos = all_hiddens[l_idx, eval_start:eval_end, pos, :, :]
            x_pos = demo_data[:, pos, :]               # (B, d)

            r = task_subspace_r2_at_position(
                tv, h_pos, covariates=x_pos, fit_token=fit_token,
                grand_mean=fixed_mean, simplex=simplex,
            )
            r.layer_num = l_num
            r.position = pos
            layer_results[pos] = r

        results[l_num] = layer_results

    from icl.utils.separability import _layer_style

    sorted_layers = sorted(results.keys())
    n_layers = len(sorted_layers)
    _ncol = 2 if n_layers > 6 else 1

    def _make_fig(ylabel, val_key):
        fig, ax = plt.subplots(figsize=figsize)
        pos_list: list = []
        for l_num in sorted_layers:
            pos_results = results[l_num]
            pos_list = sorted(pos_results.keys())
            vals = [getattr(pos_results[p], val_key) for p in pos_list]
            ax.plot(pos_list, vals, label=str(l_num),
                    **_layer_style(l_num, len(pos_list)))
        ax.set_xlabel("Position", fontsize=13)
        if show_ylabel:
            ax.set_ylabel(ylabel, fontsize=13)
        if log_x and len(pos_list) > 1 and min(pos_list) >= 0:
            ax.set_xscale("symlog", linthresh=1)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=12)
        ax.legend(title="Layer", fontsize=12, title_fontsize=12,
                  framealpha=0.9, loc="best", ncol=_ncol,
                  borderaxespad=0.3, handlelength=2.2)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        fig.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    fig_task = (
        _make_fig("Task subspace $R^2$", "r2_task")
        if plots in ("both", "task") else None
    )
    fig_add = (
        _make_fig(r"$R^2$: $\mu_t + \theta_z + \nu_a$", "r2_additive")
        if plots in ("both", "additive") else None
    )

    return {
        "all_hiddens": all_hiddens,
        "demo_data": demo_data,
        "task_vecs": task_vecs_by_layer,
        "results": results,
        "fig_task": fig_task,
        "fig_additive": fig_add,
    }


def plot_ancova_separability_linear(
    exp_name: str,
    layers: Optional[list] = None,
    positions: Optional[list] = None,
    batch_size: int = 64,
    chunk_size: int = 16,
    step: Optional[int] = None,
    n_minor: int = 0,
    n_ood: int = 0,
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
    verbose: bool = False,
    figsize: tuple = (5, 3.2),
    log_x: bool = True,
    show: bool = True,
    show_ylabel: bool = True,
    print_summary: bool = True,
) -> dict:
    """ANCOVA slope-homogeneity test of additive separability for linear regression.

    Compares an additive model ``h = task_intercept + B x_t`` (common slope)
    against a full model ``h = task_intercept + B_k x_t`` (task-specific
    slopes).  A small gap indicates that the effect of x_t on h is
    task-independent, i.e. additive separability holds.

    Results are cached: calling ``plot_task_vector_r2_linear`` beforehand
    with the same data parameters reuses the hidden states.

    Parameters
    ----------
    exp_name : str
    layers : list, optional
        ``None`` → all layers.
    positions : list, optional
        Point indices.  ``None`` → all data positions.
    batch_size : int
    chunk_size : int
        Tasks per forward-pass chunk.
    step : int, optional
    n_minor : int
    n_ood : int
    post_layernorm : bool
    extraction_point : str
        ``"post_attn"`` (default) or ``"post_mlp"`` (residual stream).
    verbose : bool
    figsize : tuple
    log_x : bool
    show : bool
    print_summary : bool

    Returns
    -------
    dict
        ``{'all_hiddens', 'demo_data', 'ancova_results',
        'fig_interaction', 'fig_sep'}``.
    """
    from icl.utils.separability import (
        ancova_separability_from_hiddens,
        plot_ancova_separability,
        print_ancova_summary,
    )

    all_hiddens, demo_data, layers, *_ = _get_linear_hiddens_cached(
        exp_name, layers, batch_size, chunk_size, step, n_minor, n_ood, verbose,
        post_layernorm=post_layernorm,
        extraction_point=extraction_point,
    )

    ancova_results = ancova_separability_from_hiddens(
        all_hiddens=all_hiddens,
        demo_data=demo_data,
        layers=layers,
        positions=positions,
    )

    if print_summary:
        print_ancova_summary(ancova_results, positions=positions)

    fig_interaction, fig_sep = plot_ancova_separability(
        ancova_results,
        figsize=figsize,
        log_x=log_x,
        show=show,
        show_ylabel=show_ylabel,
    )

    return {
        "all_hiddens": all_hiddens,
        "demo_data": demo_data,
        "ancova_results": ancova_results,
        "fig_interaction": fig_interaction,
        "fig_sep": fig_sep,
    }
