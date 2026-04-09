"""Analysis plots for linear regression (posterior, R², loss curves)."""

import json
import os
from typing import Optional

import numpy as np
import torch

from icl.linear.analysis._helpers import _show_or_close, _temporary_task_attributes
from icl.linear.analysis.posterior._compute import (
    task_posterior_over_time_linear_regression,
    task_posterior_with_gaussian_linear_regression,
)
from icl.linear.analysis.probes import train_linear_hidden_predictor
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)


def plot_task_posterior(
    exp_name: str,
    n_plots: int = 3,
    uniform_prior: bool = True,
    include_gaussian: bool = True,
    major_only: bool = False,
    max_positions: Optional[int] = None,
    figsize: tuple = (10, 3.5),
    title: Optional[str] = None,
    show: bool = True,
) -> dict:
    """Plot P(z=k | x₁:t, y₁:t₋₁) over time for random sequences.

    Optionally includes a Gaussian "new task" hypothesis (K+1-th component).
    """
    import matplotlib.pyplot as plt
    from icl.linear.linear_path_utils import load_model_task_config
    _, train_task, config = load_model_task_config(exp_name)
    device = config.device
    n_points = int(config.task.n_points)
    B = max(n_plots, 4)

    has_minor = (train_task.n_minor_tasks > 0
                 and train_task.minor_pool is not None)
    K_major = int(train_task.n_tasks)
    K_minor = int(train_task.n_minor_tasks) if has_minor else 0
    T_plot = max_positions if max_positions is not None else n_points

    # Temporarily override task state for sampling.
    temp_overrides = {"batch_size": B}
    if uniform_prior and has_minor:
        temp_overrides["p_minor"] = K_minor / (K_major + K_minor)
    with _temporary_task_attributes(train_task, **temp_overrides):
        demo_data, _, demo_target = train_task.sample_batch(step=42, is_eval=major_only)
        demo_data = demo_data.to(device)
        demo_target = demo_target.to(device)

    # ---- compute posterior ----
    if include_gaussian:
        K_total = K_major + K_minor + 1
        posterior_time = torch.zeros(B, T_plot, K_total, device=device)
        for t in range(T_plot):
            posterior_time[:, t, :] = task_posterior_with_gaussian_linear_regression(
                train_task, demo_data[:, :t+1], demo_target[:, :t+1],
                include_minor=has_minor,
            )
    else:
        post_all = task_posterior_over_time_linear_regression(
            train_task, demo_data, demo_target, include_minor=has_minor,
        )
        posterior_time = post_all[:, :T_plot, :]

    # ---- plot ----
    major_colors = [plt.cm.Blues(0.3 + 0.6 * i / max(K_major - 1, 1))
                    for i in range(K_major)]
    minor_colors = [plt.cm.Reds(0.3 + 0.6 * i / max(K_minor - 1, 1))
                    for i in range(K_minor)]
    ts = np.arange(T_plot)

    fig, axes = plt.subplots(n_plots, 1,
                             figsize=(figsize[0], figsize[1] * n_plots),
                             squeeze=False)

    for idx, ax in enumerate(axes.flat):
        post = posterior_time[idx].cpu().numpy()

        for k in range(K_major):
            ax.plot(ts, post[:, k], color=major_colors[k], alpha=0.8, lw=1.5,
                    label="Major" if idx == 0 and k == 0 else None)

        for k in range(K_minor):
            ax.plot(ts, post[:, K_major + k], color=minor_colors[k], alpha=0.6, lw=1.0,
                    label="Minor" if idx == 0 and k == 0 else None)

        if include_gaussian:
            ax.plot(ts, post[:, -1], color="black", ls="--", lw=2.0, alpha=0.9,
                    label="P(Gaussian)" if idx == 0 else None)

        ax.set(ylabel="P(Z=k | X,Y)", xlim=(0, max(T_plot - 1, 1)),
               ylim=(-0.02, 1.02), title=f"Sample {idx + 1}")
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")

    axes.flat[-1].set_xlabel("Position t")
    fig.suptitle("", fontsize=18, y=1.01)
    fig.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)

    posteriors = [posterior_time[i].cpu() for i in range(n_plots)]
    return {
        "posteriors": posteriors,
        "fig": fig,
        "axes": list(axes.flat),
        "demo_data": demo_data.cpu(),
        "demo_target": demo_target.cpu(),
    }


def plot_val_r2_across_layers(
    exp_name: str,
    layers: Optional[list] = None,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Sweep OLS probe h ≈ πW + b across layers; plot R² and partial-R² bars.

    Also prints design-matrix collinearity diagnostics (condition number,
    VIF, GVIF, pairwise R² between feature groups).
    """
    import matplotlib.pyplot as plt
    from icl.linear.linear_path_utils import load_model_task_config

    if layers is None:
        _, _, config = load_model_task_config(exp_name)
        layers = list(range(config.model.n_layer))

    def _diag(r, key, default=float("nan")):
        d = r.get("diagnostics")
        return d[key] if d is not None else default

    all_results = {}
    for layer in layers:
        logger.info(f"[sweep] running layer {layer} ...")
        all_results[layer] = train_linear_hidden_predictor(
            exp_name=exp_name, layer=layer, print_summary=False, **kwargs,
        )

    # Print design matrix diagnostics (layer-independent, use first layer)
    first_res = all_results[layers[0]]
    dd = first_res.get("diagnostics", {}).get("design_diagnostics")
    if dd is not None:
        _r2 = "\u00b2"
        print(f"\n{'=' * 65}")
        print(f"  Design Matrix Collinearity Summary (layer-independent)")
        print(f"{'=' * 65}")
        print(f"  Condition number: {dd['condition_number']:.2e}")
        print(f"  Features: posterior={dd['n_features']['posterior']}  "
              f"token={dd['n_features']['token']}  "
              f"logit={dd['n_features']['logit']}  "
              f"(total={dd['n_features']['total']})")
        print()
        print(f"  {'Group':<12} {'dims':>5} {'VIF':>10} "
              f"{'GVIF^(1/2p)':>12} {'R{r2} from rest':>14}".format(r2=_r2))
        print(f"  {'-' * 55}")
        for grp in ("posterior", "token", "logit"):
            ndim = dd["n_features"][grp]
            vif_val = dd["vif"][grp]
            gvif_val = dd["gvif_adj"][grp]
            r2_rest = dd["r2_from_rest"][grp]
            print(f"  {grp:<12} {ndim:>5d} {vif_val:>10.2f} "
                  f"{gvif_val:>12.4f} {r2_rest:>14.4f}")
        print()
        pw = dd["pairwise_r2"]
        _arrow = "\u2194"
        print(f"  Pairwise R{_r2} between feature groups:")
        print(f"    post{_arrow}tok   = {pw['post_tok']:.4f}")
        print(f"    post{_arrow}logit = {pw['post_logit']:.4f}")
        print(f"    tok{_arrow}logit  = {pw['tok_logit']:.4f}")
        print(f"{'=' * 65}\n")

    x = np.arange(len(layers))
    layer_labels = [str(l) for l in layers]

    marginal_metrics = {
        "Joint": lambda r: r["val_r2"],
        "Posterior only": lambda r: _diag(r, "r2_posterior_only"),
        "Token only": lambda r: _diag(r, "r2_token_only"),
        "Logit only": lambda r: _diag(r, "r2_logit_only"),
    }
    partial_metrics = {
        "Posterior | rest": lambda r: _diag(r, "partial_r2_posterior"),
        "Token | rest": lambda r: _diag(r, "partial_r2_token"),
        "Logit | rest": lambda r: _diag(r, "partial_r2_logit"),
    }

    panels = [marginal_metrics, partial_metrics]
    panel_titles = ["Val R\u00b2 (marginal)", "Partial R\u00b2 (unique contribution)"]
    panel_ylabels = ["R\u00b2", "Partial R\u00b2"]

    fig, axes = plt.subplots(1, 2, figsize=(max(5 * len(layers) / 4, 12), 5))

    for ax, metrics, ptitle, ylabel in zip(
        axes, panels, panel_titles, panel_ylabels,
    ):
        n_m = len(metrics)
        bw = 0.8 / n_m
        colors = plt.cm.Set2(np.linspace(0, 0.8, n_m))
        for i, (name, ext) in enumerate(metrics.items()):
            vals = [ext(all_results[l]) for l in layers]
            offset = (i - (n_m - 1) / 2) * bw
            bars = ax.bar(x + offset, vals, bw, label=name, color=colors[i])
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    txt = f"{v:.2f}".lstrip("0") if 0 < abs(v) < 1 else f"{v:.2f}"
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height(), txt,
                            ha="center", va="bottom", fontsize=9)
        ax.set(xlabel="Layer", ylabel=ylabel, title=ptitle)
        ax.set_xticks(x, layer_labels)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    if title:
        fig.suptitle("", fontsize=18, y=1.02)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    _show_or_close(fig, show)

    return fig, all_results


def plot_id_ood_loss(
    k_list=None,
    logx: bool = True,
    figsize: tuple = (10, 4),
    show: bool = True,
    pad: Optional[str] = "none",
    exp_name_kwargs: Optional[dict] = None,
    id_source: str = "id",
    ood_source: str = "ood",
    metric_key: str = "Transformer | True",
    noise_scale: Optional[float] = 0.5,
    n_major_list=None,
    legend_title: Optional[str] = None,
) -> dict:
    """Plot ID and OOD training loss vs step for multiple k values.

    The plotted loss is RMSE(model predictions, noisy targets). With additive
    Gaussian noise ε ~ N(0, σ²), the best achievable RMSE is σ (noise_scale).
    If ``noise_scale`` is set, a horizontal reference line is drawn at that level.

    Parameters
    ----------
    k_list : sequence of int, optional
        Each k → ``get_exp_name("linear", k, pad=pad, **exp_name_kwargs)``.
        Use **either** ``k_list`` or ``n_major_list``, not both.
    n_major_list : sequence of int, optional
        Major-only sweep: each value is ``n_tasks`` with
        ``get_exp_name(..., k=0, n_tasks=..., n_minor_tasks=1, p_minor=1e-12, ...)``.
    legend_title : str, optional
        Legend title override (default depends on k vs major sweep).
    exp_name_kwargs : dict, optional
        Extra kwargs passed to ``get_exp_name`` for linear (e.g. ``n_layer``,
        ``total_steps``, ``warmup_steps``, ``batch_size``). Must match the
        config used when training the experiments.
    id_source : str
        Eval block used for ID curve. Common choices:
        ``"id"`` (new format), ``"pretrain_false"``, ``"pretrain_true"``,
        ``"coin_id"``.  ``"id"`` maps to ``eval/IDLoss``.
    ood_source : str
        Eval block used for OOD curve. Common choices:
        ``"ood"`` (new format), ``"latent_false"``, ``"latent_true"``,
        ``"coin_ood"``.  ``"ood"`` maps to ``eval/OODLoss``.
    metric_key : str
        Metric name inside eval blocks (default ``"Transformer | True"``).
    noise_scale : float, optional
        Standard deviation of observation noise (default 0.5). Best achievable
        RMSE equals this value. If provided, a horizontal "noise floor" line is
        drawn. Set to None to disable.

    Returns
    -------
    dict
        ``{'fig', 'ax1', 'ax2', 'results'}``.
    """
    import matplotlib.pyplot as plt
    from icl.utils.unified_interface import get_exp_name

    if exp_name_kwargs is None:
        exp_name_kwargs = {}

    if n_major_list is not None and k_list is not None:
        raise ValueError("Pass only one of k_list or n_major_list.")
    if n_major_list is None and k_list is None:
        raise ValueError("Provide k_list or n_major_list.")
    use_n_major = n_major_list is not None

    def _resolve_linear_exp_dir(exp_name: str) -> str:
        candidates = [
            os.path.join("results", "linear", exp_name),
            os.path.join("..", "results", "linear", exp_name),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0]

    def _to_mean_curve(values):
        curve = []
        for v in values:
            arr = np.asarray(v, dtype=float)
            curve.append(float(arr.mean()) if arr.size > 0 else float("nan"))
        return np.asarray(curve, dtype=float)

    def _resolve_source_key(source: str) -> Optional[str]:
        s = str(source).strip().lower()
        mapping = {
            "pretrain_false": "eval/Pretrain_false",
            "pretrain_true": "eval/Pretrain_true",
            "latent_false": "eval/Latent_false",
            "latent_true": "eval/Latent_true",
            "coin_id": "eval/IDLoss",
            "coin_ood": "eval/OODLoss",
            "id": "eval/IDLoss",
            "ood": "eval/OODLoss",
            "minor": "eval/MinorLoss",
        }
        if s in mapping:
            return mapping[s]
        if source in mapping.values():
            return source
        return None

    def _extract_curve_from_block(eval_block: dict):
        if not isinstance(eval_block, dict):
            return None
        if metric_key in eval_block:
            return _to_mean_curve(eval_block[metric_key])
        if "Transformer | True" in eval_block:
            return _to_mean_curve(eval_block["Transformer | True"])
        for k_metric, vals in eval_block.items():
            if str(k_metric).startswith("Transformer |"):
                return _to_mean_curve(vals)
        return None

    _OLD_FORMAT_FALLBACKS = {
        "eval/IDLoss": ["eval/Pretrain_true", "eval/Pretrain_false"],
        "eval/OODLoss": ["eval/Latent_true", "eval/Latent_false"],
    }

    def _extract_curve(data_dict: dict, source: str):
        src_key = _resolve_source_key(source)
        if src_key is None:
            raise KeyError(
                f"Unknown source {source!r}. Use one of "
                f"id/ood/minor/pretrain_false/pretrain_true/latent_false/latent_true/coin_id/coin_ood."
            )

        if src_key in ("eval/IDLoss", "eval/OODLoss", "eval/MinorLoss"):
            vals = data_dict.get(src_key, None)
            if vals is not None:
                return np.asarray(vals, dtype=float), src_key
            for fallback_key in _OLD_FORMAT_FALLBACKS.get(src_key, []):
                block = data_dict.get(fallback_key, None)
                curve = _extract_curve_from_block(block)
                if curve is not None:
                    return curve, fallback_key
            return None, src_key

        block = data_dict.get(src_key, None)
        curve = _extract_curve_from_block(block)
        if curve is not None:
            return curve, src_key

        src_key_l = src_key.lower()
        for k_data, v_data in data_dict.items():
            if isinstance(k_data, str) and k_data.lower() == src_key_l:
                curve = _extract_curve_from_block(v_data)
                if curve is not None:
                    return curve, k_data

        return None, src_key

    results = {}
    iter_keys = list(n_major_list) if use_n_major else list(k_list)
    for key in iter_keys:
        if use_n_major:
            exp_name = get_exp_name(
                "linear",
                0,
                pad=pad,
                n_tasks=int(key),
                n_minor_tasks=1,
                p_minor=1e-12,
                **exp_name_kwargs,
            )
            n_minor_tasks = int(key)
        else:
            k = key
            exp_name = get_exp_name("linear", k, pad=pad, **exp_name_kwargs)
            n_minor_tasks = 2 ** k if k >= 0 else 0

        try:
            exp_dir = _resolve_linear_exp_dir(exp_name)
            log_path = os.path.join(exp_dir, "log.json")
            with open(log_path, "r") as f:
                data = json.load(f)

            train_steps = np.asarray(data.get("eval/step", data.get("train/step", [])), dtype=float)
            id_loss, id_key_used = _extract_curve(data, id_source)
            ood_loss, ood_key_used = _extract_curve(data, ood_source)

            if id_loss is None or ood_loss is None:
                raise KeyError(
                    f"Could not find requested ID/OOD curves in log.json "
                    f"(id_source={id_source!r}, ood_source={ood_source!r}, metric={metric_key!r})."
                )

            if train_steps.size == 0:
                L = min(len(id_loss), len(ood_loss))
                train_steps = np.arange(1, L + 1, dtype=float)
            L = min(len(train_steps), len(id_loss), len(ood_loss))
            train_steps = np.asarray(train_steps[:L], dtype=float)
            id_loss = np.asarray(id_loss[:L], dtype=float)
            ood_loss = np.asarray(ood_loss[:L], dtype=float)

            results[key] = dict(
                n_minor=n_minor_tasks,
                train_steps=train_steps,
                id_loss=id_loss,
                ood_loss=ood_loss,
                id_source_used=id_key_used,
                ood_source_used=ood_key_used,
                metric_key_used=metric_key,
            )
        except Exception as e:
            logger.warning(f"Could not load key={key}: {e}")

    ks_sorted = sorted(results.keys())
    if not ks_sorted:
        logger.warning("No experiments loaded successfully.")
        return {}

    nk = len(ks_sorted)
    cmap = plt.get_cmap("viridis")
    color_map = {}
    for i, k in enumerate(ks_sorted):
        t = 0.15 + 0.75 * (i / max(1, nk - 1))
        color_map[k] = cmap(t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, which="major", alpha=0.15, linestyle="-")
        ax.grid(True, which="minor", alpha=0.06, linestyle=":")
        if noise_scale is not None:
            ax.axhline(noise_scale, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="_nolegend_")

    lw, alpha = 1.5, 0.85
    fs_label, fs_tick = 14, 12

    for k in ks_sorted:
        d = results[k]
        c = color_map[k]
        xs, y_id = d["train_steps"], d["id_loss"]
        if logx:
            mask = xs > 0
            xs, y_id = xs[mask], y_id[mask]
        if xs.size == 0:
            continue
        ax1.plot(xs, y_id, color=c, linewidth=lw, alpha=alpha, label=str(k))

    ax1.set_title("In-distribution", fontsize=fs_label)
    if logx:
        ax1.set_xscale("log")
    ax1.set_xlabel("Training Step", fontsize=fs_label)
    ax1.set_ylabel("Loss (RMSE)", fontsize=fs_label)
    ax1.tick_params(labelsize=fs_tick)

    handles, labels = ax1.get_legend_handles_labels()
    _leg_title = legend_title
    if _leg_title is None:
        _leg_title = (
            r"$N_{\mathrm{major}}$"
            if use_n_major
            else r"$\log_2(n_{\mathrm{minor}})$"
        )
    ax2.legend(
        handles,
        labels,
        title=_leg_title,
        fontsize=fs_tick,
        title_fontsize=fs_label,
        frameon=True,
        framealpha=0.95,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    for k in ks_sorted:
        d = results[k]
        c = color_map[k]
        xs, y_ood = d["train_steps"], d["ood_loss"]
        if logx:
            mask = xs > 0
            xs, y_ood = xs[mask], y_ood[mask]
        if xs.size == 0:
            continue
        ax2.plot(xs, y_ood, color=c, linewidth=lw, alpha=alpha)

    ax2.set_title("Out-of-distribution", fontsize=fs_label)
    if logx:
        ax2.set_xscale("log")
    ax2.set_xlabel("Training Step", fontsize=fs_label)
    ax2.set_ylabel("")
    ax2.tick_params(labelsize=fs_tick)

    fig.subplots_adjust(wspace=0.05, right=0.78)
    fig.tight_layout()
    if show:
        plt.show()

    return {"fig": fig, "ax1": ax1, "ax2": ax2, "results": results}
