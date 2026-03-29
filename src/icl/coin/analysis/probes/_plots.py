"""Plotting functions for Coin probes."""

import gc

import numpy as np
import torch
from typing import Optional

import icl.utils.notebook_utils as nu
from icl.utils.logger import setup_logger

from icl.coin.analysis.probes._internals import _collect_coin_probe_data, _fit_coin_probe

logger = setup_logger(__name__)


def plot_val_r2_across_layers_coin(
    exp_name: str,
    layers: Optional[list] = None,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    extraction_point: str = "post_attn",
    **kwargs,
):
    """Sweep OLS probe h ~ [posterior, one_hot] across layers; plot R² and partial-R² bars.

    Also prints design-matrix collinearity diagnostics (condition number,
    VIF, GVIF, pairwise R² between feature groups).

    Parameters
    ----------
    exp_name : str
    layers : list, optional
        Layer indices to evaluate.  ``None`` → all layers (auto-detected).
    title : str, optional
    show : bool
    save_path : str, optional
    extraction_point : ``"post_attn"`` | ``"post_mlp"`` | ``"both"``
        Where to hook each transformer layer.
        ``"post_attn"`` (default) — after the attention block, before MLP.
        ``"post_mlp"`` — after the full block (attention + MLP).
        ``"both"`` — sweeps both extraction points and displays them
        interleaved on the x-axis (post_attn then post_mlp per layer).
    **kwargs
        Forwarded to ``_collect_coin_probe_data`` and ``_fit_coin_probe``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    all_results : dict
        When ``extraction_point`` is a single point: ``{layer_index: results_dict}``.
        When ``extraction_point="both"``: ``{(layer_index, ep): results_dict}``.
    """
    import matplotlib.pyplot as plt

    if layers is None:
        _, _, config = nu.load_everything("coin", exp_name)
        layers = list(range(config.model.num_layers))

    def _diag(r, key, default=float("nan")):
        d = r.get("diagnostics")
        return d[key] if d is not None else default

    # extraction_point is handled explicitly and excluded from collect_kwargs
    collect_kwargs = {
        k: v for k, v in kwargs.items()
        if k in ("B", "n_samples", "step", "n_minor", "positions",
                 "uniform_sampling", "sample_mode", "verbose",
                 "anchor_minor_samples", "use_task_identity")
    }
    fit_kwargs = {
        k: v for k, v in kwargs.items()
        if k in ("validation_split", "include_position_bias", "skip_baselines", "per_position_mean")
    }
    sample_mode = kwargs.get("sample_mode", "train")
    _use_tid = kwargs.get("use_task_identity", False)

    eps = ["post_attn", "post_mlp"] if extraction_point == "both" else [extraction_point]

    # Collect hidden states (one forward pass per extraction point)
    all_data = {}
    for ep in eps:
        logger.info(f"[sweep] collecting data for {len(layers)} layers (ep={ep!r}) ...")
        all_data[ep] = _collect_coin_probe_data(
            exp_name=exp_name, layers=layers, extraction_point=ep, **collect_kwargs,
        )

    # Fit a probe per (layer, extraction_point)
    all_results = {}
    for ep in eps:
        data = all_data[ep]
        for layer in layers:
            logger.info(f"[sweep] fitting layer {layer} (ep={ep!r}) ...")
            result = _fit_coin_probe(
                hiddens_all=data["hiddens_by_layer"][layer],
                posteriors_all=data["posteriors_all"],
                real_tokens_all=data["real_tokens_all"],
                layer=layer, n_tasks=data["n_tasks"], positions=data["positions"],
                print_summary=False, sample_mode=sample_mode,
                n_major=data.get("n_major"),
                task_ids_all=data.get("task_ids"),
                **fit_kwargs,
            )
            key = (layer, ep) if extraction_point == "both" else layer
            all_results[key] = result

    # ── Print design matrix diagnostics (layer-independent, first layer, first ep) ──
    first_key = (layers[0], eps[0]) if extraction_point == "both" else layers[0]
    first_res = all_results[first_key]
    dd = first_res.get("diagnostics", {}).get("design_diagnostics")
    if dd is not None:
        _r2 = "\u00b2"
        _post_label = "task id" if _use_tid else "posterior"
        _post_abbr  = "task"   if _use_tid else "post"
        print(f"\n{'=' * 60}")
        print(f"  Design Matrix Collinearity Summary (layer-independent)")
        print(f"{'=' * 60}")
        print(f"  Condition number: {dd['condition_number']:.2e}")
        print(f"  Features: {_post_label}={dd['n_features']['posterior']}  "
              f"token={dd['n_features']['token']}  "
              f"(total={dd['n_features']['total']})")
        print()
        print(f"  {'Group':<12} {'dims':>5} {'VIF':>10} "
              f"{'GVIF^(1/2p)':>12} {'R' + _r2 + ' from rest':>14}")
        print(f"  {'-' * 55}")
        _grp_labels = {"posterior": _post_label, "token": "token"}
        for grp in ("posterior", "token"):
            ndim = dd["n_features"][grp]
            vif_val = dd["vif"][grp]
            gvif_val = dd["gvif_adj"][grp]
            r2_rest = dd["r2_from_rest"][grp]
            print(f"  {_grp_labels[grp]:<12} {ndim:>5d} {vif_val:>10.2f} "
                  f"{gvif_val:>12.4f} {r2_rest:>14.4f}")
        print()
        pw = dd["pairwise_r2"]
        _arrow = "\u2194"
        print(f"  Pairwise R{_r2} between feature groups:")
        print(f"    {_post_abbr}{_arrow}tok = {pw['post_tok']:.4f}")
        print(f"{'=' * 60}\n")

    # ── Build ordered key list and x-axis labels ──
    _ep_short = {"post_attn": "attn", "post_mlp": "mlp"}
    if extraction_point == "both":
        ordered_keys = [(l, ep) for l in layers for ep in eps]
        layer_labels = [f"{l}\n{_ep_short.get(ep, ep)}" for l in layers for ep in eps]
        xlabel = "Layer / Extraction point"
    else:
        ordered_keys = [l for l in layers]
        layer_labels = [str(l) for l in layers]
        xlabel = "Layer"

    # ── Two-panel figure: marginal R² and partial R² ──
    x = np.arange(len(ordered_keys))

    _p_label = "Task id" if _use_tid else "Posterior"
    marginal_metrics = {
        "Joint": lambda r: r["val_r2"],
        f"{_p_label} only": lambda r: _diag(r, "r2_posterior_only"),
        "Token only": lambda r: _diag(r, "r2_token_only"),
    }
    _rest_of = "token" if not _use_tid else "token"
    partial_metrics = {
        f"{_p_label} | token": lambda r: _diag(r, "partial_r2_posterior"),
        f"Token | {_p_label.lower()}": lambda r: _diag(r, "partial_r2_token"),
    }

    panels = [marginal_metrics, partial_metrics]
    panel_titles = ["Val R\u00b2 (marginal)", "Partial R\u00b2 (unique contribution)"]
    panel_ylabels = ["R\u00b2", "Partial R\u00b2"]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(5 * len(ordered_keys) / 4, 12), 6),
        dpi=150,
    )

    for ax, metrics, ptitle, ylabel in zip(
        axes, panels, panel_titles, panel_ylabels,
    ):
        n_m = len(metrics)
        bw = 0.8 / n_m
        colors = plt.cm.Set2(np.linspace(0, 0.8, n_m))
        for i, (name, ext) in enumerate(metrics.items()):
            vals = [ext(all_results[k]) for k in ordered_keys]
            offset = (i - (n_m - 1) / 2) * bw
            bars = ax.bar(x + offset, vals, bw, label=name, color=colors[i])
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    txt = f"{v:.2f}".lstrip("0") if 0 < abs(v) < 1 else f"{v:.2f}"
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height(), txt,
                            ha="center", va="bottom", fontsize=9)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=ptitle)
        ax.set_xticks(x, layer_labels)
        ax.tick_params(labelsize=12)
        ax.legend(
            fontsize=10,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=n_m,
            framealpha=0.9,
        )
        ax.grid(axis="y", alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, all_results


# ---------------------------------------------------------------------------
# Averaging-based R² for Coin
# ---------------------------------------------------------------------------

def plot_averaging_r2_coin(
    exp_name: str,
    layers: Optional[list] = None,
    estimation_positions: Optional[list] = None,
    evaluation_positions: Optional[list] = None,
    batch_size: int = 64,
    step: Optional[int] = None,
    n_minor: int = 0,
    fit_token: str = "anova",
    per_position_mean: bool = True,
    per_position_token_vecs: bool = False,
    eval_subset: str = "major",
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
    simplex: bool = True,
    verbose: bool = False,
    figsize: tuple = (5, 3.2),
    log_x: bool = True,
    show: bool = True,
    show_ylabel: bool = True,
    task_batch_size: int = 8,
    plots: str = "both",
) -> dict:
    """Task-subspace R² using interventional (token-conditioned) data (coin).

    Parameters (plotting)
    ---------------------
    plots : "both" | "task" | "additive"
        Which figure(s) to produce.  ``"task"`` = task-only R²;
        ``"additive"`` = task + token R²; ``"both"`` (default) = both.

    Collects hidden states with the token at each position fixed
    (interventional), breaking task-token confounding.  Task and token
    vectors are estimated from cell-mean ANOVA marginals at estimation
    positions, then R² is evaluated at all positions by projecting onto
    the task subspace (task-only) and the combined task + token subspace
    (additive).

    Parameters
    ----------
    exp_name : str
    layers : list, optional — ``None`` -> all layers
    estimation_positions : list, optional
        Positions used to estimate task and token vectors (default: last 10).
    evaluation_positions : list, optional
        Positions at which R² is computed (default: all).
    batch_size : int
        Samples per (task, token) cell.
    step : int, optional
    n_minor : int
    fit_token : "none" | "anova"
    per_position_mean : bool
    eval_subset : "all" | "major" | "minor"
    verbose : bool
    figsize, log_x, show, show_ylabel : plot options
    task_batch_size : int
        Tasks to batch per forward pass during data collection.

    Returns
    -------
    dict with 'task_vecs', 'token_vecs', 'results',
    'fig_task', 'fig_additive'.
    """
    if fit_token not in ("none", "anova"):
        raise ValueError(
            f"fit_token must be 'none' or 'anova' for discrete-token tasks, "
            f"got {fit_token!r}"
        )

    import matplotlib.pyplot as plt
    from icl.coin.analysis._helpers import get_token_conditioned_hiddens_coin
    from icl.utils.separability import AveragingR2Result

    # ---- Determine positions ----
    _, sampler_orig, config = nu.load_everything("coin", exp_name)
    n_major = sampler_orig.n_major_tasks
    T = sampler_orig.seq_len - 1

    if layers is None:
        layers = list(range(config.model.num_layers))
    layers_idx = list(layers)

    if estimation_positions is None:
        estimation_positions = list(range(max(0, T - 10), T))
    if evaluation_positions is None:
        evaluation_positions = list(range(T))

    all_positions = sorted(set(estimation_positions) | set(evaluation_positions))

    if verbose:
        logger.info(
            f"[averaging R² coin] collecting token-conditioned data at "
            f"{len(all_positions)} positions, batch_size={batch_size}, "
            f"n_minor={n_minor}, task_batch_size={task_batch_size}"
        )

    # ---- Collect interventional data ----
    all_hiddens, token_info = get_token_conditioned_hiddens_coin(
        exp_name=exp_name,
        layers=layers_idx,
        batch_size=batch_size,
        positions_of_interest=all_positions,
        n_minor=n_minor,
        step=step,
        verbose=verbose,
        task_batch_size=task_batch_size,
        post_layernorm=post_layernorm,
        extraction_point=extraction_point,
    )
    # all_hiddens: (L, n_positions, n_tokens_max, n_tasks, B, D)
    L, n_pos, n_tok_max, n_tasks_total, B, D = all_hiddens.shape
    pos_list_data = token_info["positions"]
    pos_to_idx = {p: i for i, p in enumerate(pos_list_data)}
    n_unique = token_info["n_unique_tokens"]

    # ---- Eval subset ----
    _subset_ranges = {
        "all":   (0, n_tasks_total),
        "major": (0, n_major),
        "minor": (n_major, n_tasks_total),
    }
    if eval_subset not in _subset_ranges:
        raise ValueError(
            f"eval_subset must be one of {list(_subset_ranges)}, "
            f"got {eval_subset!r}"
        )
    eval_start, eval_end = _subset_ranges[eval_subset]
    if eval_end <= eval_start:
        raise ValueError(f"No tasks in subset '{eval_subset}'")

    if verbose:
        logger.info(
            f"  hiddens shape: {all_hiddens.shape}, "
            f"estimation_pos={estimation_positions}, "
            f"eval on '{eval_subset}' [{eval_start}:{eval_end}]"
        )

    # ---- Per-layer estimation and evaluation ----
    results: dict = {}
    task_vecs_by_layer: dict = {}
    token_vecs_by_layer: dict = {}

    for li, l_num in enumerate(layers_idx):
        # --- Estimate task/token vectors from cell-mean marginals ---
        est_parts = []
        for p in estimation_positions:
            pi = pos_to_idx[p]
            V_p = n_unique.get(p, n_tok_max)
            est_parts.append(
                all_hiddens[li, pi, :V_p, :n_major].float()
            )

        # Cell means per estimation position
        cell_means_list = [part.mean(dim=2) for part in est_parts]

        # Raw grand mean (before demeaning) — needed for per_position_mean=False
        raw_pooled = torch.stack(cell_means_list, dim=0).mean(dim=0)  # (V, K, D)
        grand_mean = raw_pooled.mean(dim=(0, 1))  # (D,)

        # Remove per-position mean μ_t before pooling across positions
        demeaned_list = []
        for cm in cell_means_list:
            mu_t = cm.mean(dim=(0, 1))  # mean over tokens and tasks -> (D,)
            demeaned_list.append(cm - mu_t)
        pooled = torch.stack(demeaned_list, dim=0).mean(dim=0)  # (V, K, D)

        task_vecs = pooled.mean(dim=0) - pooled.mean(dim=(0, 1))
        token_vecs = pooled.mean(dim=1) - pooled.mean(dim=(0, 1))

        task_vecs_by_layer[l_num] = task_vecs
        token_vecs_by_layer[l_num] = token_vecs

        if verbose:
            logger.info(
                f"  Layer {l_num}: task norms = "
                + ", ".join(f"{n:.3f}" for n in task_vecs.norm(dim=1).tolist())
                + " | token norms = "
                + ", ".join(f"{n:.3f}" for n in token_vecs.norm(dim=1).tolist())
            )

        # --- Build task projector ---
        tv = task_vecs.float()
        if not simplex:
            V_basis = tv[:-1]
            P_task = V_basis.T @ torch.linalg.solve(
                V_basis @ V_basis.T, V_basis
            )

        fixed_mean = None if per_position_mean else grand_mean

        # --- Evaluate R² at each position ---
        from icl.utils.separability import _simplex_project_coeffs
        layer_results: dict = {}
        for pos in evaluation_positions:
            if pos not in pos_to_idx:
                continue
            pi = pos_to_idx[pos]
            V_p = n_unique.get(pos, n_tok_max)
            cell_data = all_hiddens[
                li, pi, :V_p, eval_start:eval_end
            ].float()

            h = cell_data.reshape(-1, D)
            N = h.shape[0]
            mu = fixed_mean if fixed_mean is not None else h.mean(dim=0)
            h_c = h - mu

            ss_total = (h_c ** 2).sum().item()
            eps = 1e-10

            if simplex:
                h_task_hat = _simplex_project_coeffs(tv, h_c)
                ss_task_res = ((h_c - h_task_hat) ** 2).sum().item()
                ss_task = ss_total - ss_task_res

                if fit_token == "anova":
                    if per_position_token_vecs:
                        pos_tv = cell_data.mean(dim=(1, 2))
                        pos_tv = pos_tv - pos_tv.mean(dim=0)
                        h_no_tok = cell_data - pos_tv[:, None, None, :]
                    else:
                        h_no_tok = cell_data - token_vecs[:V_p, None, None, :]
                    h_no_tok = h_no_tok.reshape(-1, D) - mu
                    h_task_hat_nt = _simplex_project_coeffs(tv, h_no_tok)
                    residual = h_no_tok - h_task_hat_nt
                    ss_residual = (residual ** 2).sum().item()
                    ss_additive = ss_total - ss_residual
                else:
                    ss_additive = ss_task
            else:
                h_task = h_c @ P_task
                ss_task = (h_task ** 2).sum().item()

                if fit_token == "anova":
                    if per_position_token_vecs:
                        pos_tv = cell_data.mean(dim=(1, 2))
                        pos_tv = pos_tv - pos_tv.mean(dim=0)
                        h_no_tok = cell_data - pos_tv[:, None, None, :]
                    else:
                        h_no_tok = cell_data - token_vecs[:V_p, None, None, :]
                    h_no_tok = h_no_tok.reshape(-1, D) - mu
                    h_no_tok_task = h_no_tok @ P_task
                    residual = h_no_tok - h_no_tok_task
                    ss_residual = (residual ** 2).sum().item()
                    ss_additive = ss_total - ss_residual
                else:
                    ss_additive = ss_task

            r = AveragingR2Result(
                r2_task=ss_task / (ss_total + eps),
                r2_additive=ss_additive / (ss_total + eps),
                ss_total=ss_total,
                ss_task=ss_task,
                ss_token=ss_additive - ss_task,
                n_tasks=eval_end - eval_start,
                n_samples=N,
            )
            r.layer_num = l_num
            r.position = pos
            layer_results[pos] = r

        results[l_num] = layer_results

    del all_hiddens
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ---- Plotting ----
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
        "all_hiddens_shape": (L, n_pos, n_tok_max, n_tasks_total, B, D),
        "task_vecs": task_vecs_by_layer,
        "token_vecs": token_vecs_by_layer,
        "results": results,
        "fig_task": fig_task,
        "fig_additive": fig_add,
    }
