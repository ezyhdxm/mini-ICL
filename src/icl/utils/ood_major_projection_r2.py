"""
Plot maj_r2_ood vs training step for multiple experiments (average task subspace).

Uses process_ood_minor_metric (average-approach R²). Supports coin (k_list),
linear, and latent (list of exp_name or (exp_name, label)). Multi-GPU
parallelization happens at the *step level* inside process_ood_minor_metric
(ThreadPoolExecutor, one thread per GPU). Optional IQR band (q25–q75 over ``B``
samples) uses the same colors and alpha as the faint per-step lines. No
title/subtitle by default.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from icl.utils.unified_ood_analysis import process_ood_minor_metric
from icl.utils.unified_path_finder import get_exp_name
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_POSITION_BLOCKS: Tuple[Tuple[int, int], ...] = ((0, 15), (15, -1))
DEFAULT_BLOCK_LABELS: Tuple[str, ...] = ("0-15", "15-end")


def _ema_smooth(y: np.ndarray, alpha: float = 0.99) -> np.ndarray:
    """Exponential moving average smoothing."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
    return out


def _resolve_experiments(
    task_name: str,
    experiments: Union[Sequence[int], Sequence[Union[str, Tuple[str, str]]]],
    vocab_size: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """Return list of (exp_name, label) for plotting."""
    out: List[Tuple[str, str]] = []
    if task_name == "coin":
        for k in experiments:
            k = int(k)
            exp_name = get_exp_name("coin", k, vocab_size=vocab_size)
            label = str(k)
            out.append((exp_name, label))
    else:
        for item in experiments:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                exp_name, label = item[0], item[1]
            else:
                exp_name = str(item)
                label = exp_name
            out.append((exp_name, label))
    return out


def _extract_experiment_results(
    rd: Dict[str, Any],
    label: str,
    layer: int,
    position_blocks,
    n_blocks: int,
    results_by_block: List[
        List[Tuple[str, List[int], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]
    ],
    results_flat: List[
        Tuple[str, List[int], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
    ],
) -> None:
    """Unpack process_ood_minor_metric results into by-block / flat lists."""
    empty = (label, [], np.array([]), None, None)
    if not rd or layer not in rd["maj_r2_ood"]:
        if position_blocks is None:
            results_flat.append(empty)
        else:
            for b in range(n_blocks):
                results_by_block[b].append(empty)
        return

    if position_blocks is not None and "maj_r2_ood_by_block" in rd:
        by_block = rd["maj_r2_ood_by_block"][layer]
        by_iqr = rd.get("maj_r2_ood_by_block_iqr", {}).get(layer, {})
        m_steps = sorted(by_block.keys())
        for b in range(n_blocks):
            m_values = np.array(
                [by_block[s][b] for s in m_steps], dtype=float
            )
            q25 = q75 = None
            if by_iqr:
                q25_list: List[float] = []
                q75_list: List[float] = []
                for s in m_steps:
                    if s in by_iqr and b < len(by_iqr[s]):
                        q25_list.append(float(by_iqr[s][b]["q25"]))
                        q75_list.append(float(by_iqr[s][b]["q75"]))
                    else:
                        q25_list.append(float("nan"))
                        q75_list.append(float("nan"))
                q25 = np.asarray(q25_list, dtype=float)
                q75 = np.asarray(q75_list, dtype=float)
                if not np.any(np.isfinite(q25) & np.isfinite(q75)):
                    q25 = q75 = None
            results_by_block[b].append((label, m_steps, m_values, q25, q75))
    else:
        layer_metric = rd["maj_r2_ood"][layer]
        m_steps = sorted(layer_metric.keys())
        m_values = np.array(
            [layer_metric[s] for s in m_steps], dtype=float
        )
        iqr_by_step = rd.get("maj_r2_ood_iqr", {}).get(layer, {})
        q25 = q75 = None
        if iqr_by_step:
            q25 = np.array(
                [float(iqr_by_step.get(s, {}).get("q25", np.nan)) for s in m_steps],
                dtype=float,
            )
            q75 = np.array(
                [float(iqr_by_step.get(s, {}).get("q75", np.nan)) for s in m_steps],
                dtype=float,
            )
            if not np.any(np.isfinite(q25) & np.isfinite(q75)):
                q25 = q75 = None
        results_flat.append((label, m_steps, m_values, q25, q75))


def plot_maj_r2_ood_across_steps(
    task_name: str,
    experiments: Union[Sequence[int], Sequence[Union[str, Tuple[str, str]]]],
    layer: int,
    steps: Sequence[int],
    n_minor: int = 0,
    n_ood: int = 128,
    B: int = 8,
    n_gpus: Optional[int] = None,
    vocab_size: Optional[int] = None,
    position_blocks: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
    block_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    logx: bool = True,
    ema_alpha: float = 0.99,
    shadow_alpha: float = 0.15,
    shadow_lw: float = 1.0,
    smooth_lw: float = 1.4,
    scatter_alpha: float = 0.25,
    scatter_size: float = 10.0,
    figsize: Tuple[float, float] = (6, 4),
    show: bool = True,
    force_recompute: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
    legend_title: Optional[str] = None,
    show_ylabel: bool = True,
    show_title: bool = True,
    show_iqr_band: bool = True,
    band_alpha: Optional[float] = None,
    compute_minor_metrics: bool = False,
    extraction_point: str = "post_attn",
) -> Dict[str, Any]:
    """
    Plot maj_r2_ood vs training step for multiple experiments.

    Multi-GPU parallelization is handled at the **step level** inside
    ``process_ood_minor_metric``: uncached steps are distributed across GPUs
    via ``ThreadPoolExecutor`` (one thread per GPU, no process-spawn overhead).

    Parameters
    ----------
    task_name : str
        One of "coin", "linear", "latent".
    experiments : sequence
        For coin: list of k (int). For linear/latent: list of (exp_name,
        label) or list of exp_name.
    layer : int
        Layer index for maj_r2_ood.
    steps : sequence of int
        Training steps to evaluate.
    n_minor, n_ood, B : int
        Passed to process_ood_minor_metric.
    n_gpus : int, optional
        Number of GPUs for step-level parallelism.  Default: all available
        (``torch.cuda.device_count()``).  Use 1 for sequential.
    vocab_size : int, optional
        For coin, passed to get_exp_name.
    position_blocks : sequence, optional
        Position blocks for per-block R².
    block_labels : sequence of str, optional
        Labels for each block.
    title, subtitle : str, optional
        Figure title / subtitle.
    show_title : bool
        If False, suppress all titles.
    logx : bool
        Use log scale for x-axis.
    ema_alpha, shadow_alpha, shadow_lw, smooth_lw : float
        Smoothing and line styling.
    scatter_alpha : float
        Opacity of the raw-step scatter dots (default 0.25).
    scatter_size : float
        Marker size of the raw-step scatter dots in points² (default 10).
    figsize : tuple
        Figure size.
    show : bool
        Call plt.show().
    force_recompute : bool
        Bypass cache in process_ood_minor_metric.
    ylim : tuple, optional
        (ymin, ymax) for y-axis.
    show_iqr_band : bool
        If True, shade the IQR (q25–q75) over B forward passes when cached.
    band_alpha : float, optional
        Face alpha for the IQR band; defaults to ``shadow_alpha``.
    compute_minor_metrics : bool
        Passed to ``process_ood_minor_metric``.  If True, also compute minor-
        reference ``min_r2_*`` metrics (slower).  Default False.
    extraction_point : str
        Where in each transformer layer to extract hidden representations.
        ``"post_attn"`` (default) — after the attention block, before MLP.
        ``"post_mlp"`` — after the full block (attention + MLP), i.e. the
        residual stream.  Note: different extraction points use separate
        cache files so they never interfere.

    Returns
    -------
    dict
        fig, ax, results, and optional title/subtitle used.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    if band_alpha is None:
        band_alpha = shadow_alpha

    if position_blocks is None:
        position_blocks = DEFAULT_POSITION_BLOCKS
    elif position_blocks is not None and len(position_blocks) == 0:
        position_blocks = None

    steps_list = list(steps)
    resolved = _resolve_experiments(task_name, experiments, vocab_size=vocab_size)
    if not resolved:
        raise ValueError("No experiments resolved from experiments argument.")

    # Determine GPU count for step-level parallelism
    n_available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus is None:
        effective_gpus = max(n_available, 1)
    else:
        effective_gpus = min(n_gpus, n_available) if n_available > 0 else 1

    if position_blocks is not None and block_labels is None:
        block_labels = (
            list(DEFAULT_BLOCK_LABELS)
            if position_blocks == DEFAULT_POSITION_BLOCKS
            else [f"block {b}" for b in range(len(position_blocks))]
        )
    elif position_blocks is not None and block_labels is not None:
        if len(block_labels) != len(position_blocks):
            raise ValueError(
                "block_labels length must match position_blocks length"
            )

    n_blocks = len(position_blocks) if position_blocks is not None else 0
    results_by_block: List[
        List[Tuple[str, List[int], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]
    ] = [[] for _ in range(max(n_blocks, 1))]
    results_flat: List[
        Tuple[str, List[int], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
    ] = []

    for _i, (exp_name, label) in enumerate(resolved):
        try:
            rd = process_ood_minor_metric(
                task_name=task_name,
                exp_name=exp_name,
                steps=steps_list,
                n_minor=n_minor,
                n_ood=n_ood,
                B=B,
                force_recompute=force_recompute,
                device="cuda:0" if n_available > 0 else None,
                position_blocks=position_blocks,
                n_gpus=effective_gpus,
                compute_minor_metrics=compute_minor_metrics,
                extraction_point=extraction_point,
            )
        except Exception as e:
            logger.warning(f"[plot_maj_r2_ood] {exp_name} failed: {e}")
            rd = {}

        _extract_experiment_results(
            rd, label, layer, position_blocks, n_blocks,
            results_by_block, results_flat,
        )

    # ── Plotting ──────────────────────────────────────────────────────────────
    results: List[
        Tuple[str, List[int], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
    ] = []
    if position_blocks is not None:
        for b in range(n_blocks):
            bl = block_labels[b] if block_labels else f"block {b}"
            for lab, m_steps, m_vals, q25, q75 in results_by_block[b]:
                results.append((f"{lab} ({bl})", m_steps, m_vals, q25, q75))
    else:
        results = results_flat

    _leg_title = legend_title if legend_title is not None else r"$\log_2(n_{\mathrm{minor}})$"

    _PALETTE = [
        "#4477AA", "#EE6677", "#228833", "#CCBB44",
        "#66CCEE", "#AA3377", "#BBBBBB",
    ]
    unique_labels = [label for (_, label) in resolved]
    exp_color_map: Dict[str, Any] = {
        lab: _PALETTE[i % len(_PALETTE)] for i, lab in enumerate(unique_labels)
    }

    def _draw_ax(
        ax,
        curves: List[
            Tuple[str, List[int], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
        ],
        ax_title: Optional[str] = None,
    ) -> None:
        for label, m_steps, m_values, q25, q75 in curves:
            if len(m_steps) == 0:
                continue
            x = np.asarray(m_steps, dtype=float)
            y = m_values
            if logx:
                mask = x > 0
                x, y = x[mask], y[mask]
                if q25 is not None and q75 is not None:
                    q25 = np.asarray(q25, dtype=float)[mask]
                    q75 = np.asarray(q75, dtype=float)[mask]
            if x.size == 0:
                continue
            c = exp_color_map[label]
            if (
                show_iqr_band
                and q25 is not None
                and q75 is not None
                and np.any(np.isfinite(q25) & np.isfinite(q75))
            ):
                where = np.isfinite(q25) & np.isfinite(q75)
                ax.fill_between(
                    x,
                    q25,
                    q75,
                    where=where,
                    facecolor=to_rgba(c, band_alpha),
                    linewidth=0,
                    zorder=0,
                    clip_on=True,
                )
            ax.plot(x, y, color=c, alpha=shadow_alpha, linewidth=shadow_lw, zorder=1)
            ax.scatter(x, y, color=c, s=scatter_size, zorder=3, edgecolors="none", alpha=scatter_alpha)
            y_s = _ema_smooth(y, alpha=ema_alpha)
            ax.plot(
                x, y_s, color=c, linewidth=smooth_lw,
                label=label, zorder=2, solid_capstyle="round",
            )
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel("Training step", fontsize=11)
        if show_ylabel:
            ax.set_ylabel(r"$R^2$", fontsize=11)
        ax.tick_params(labelsize=10, direction="in", top=True, right=True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.4, color="gray")
        ax.set_axisbelow(True)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                title=_leg_title, fontsize=9, title_fontsize=9,
                framealpha=0.85, edgecolor="0.8", handlelength=1.8,
            )
        if show_title:
            title_parts = []
            if ax_title is not None:
                title_parts.append(ax_title)
            if title is not None:
                title_parts.append(title)
            if subtitle is not None:
                title_parts.append(subtitle)
            if title_parts:
                ax.set_title("\n".join(title_parts), fontsize=12)
        if ylim is not None:
            ax.set_ylim(ylim)

    if position_blocks is not None and n_blocks > 0:
        figs: List[Any] = []
        axes: List[Any] = []
        for b in range(n_blocks):
            bl = block_labels[b] if block_labels else f"block {b}"
            fig_b, ax_b = plt.subplots(figsize=figsize, dpi=150)
            _draw_ax(ax_b, results_by_block[b],
                     ax_title=f"Positions {bl}" if show_title else None)
            fig_b.tight_layout()
            if show:
                plt.show()
            figs.append(fig_b)
            axes.append(ax_b)
        return {
            "figs": figs, "axes": axes,
            "fig": figs[0] if figs else None,
            "ax": axes[0] if axes else None,
            "results": results, "title": title, "subtitle": subtitle,
        }
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        _draw_ax(ax, results_flat)
        fig.tight_layout()
        if show:
            plt.show()
        return {
            "figs": [fig], "axes": [ax], "fig": fig, "ax": ax,
            "results": results, "title": title, "subtitle": subtitle,
        }


def plot_maj_r2_ood_across_steps_coin(
    k_list: Sequence[int],
    steps: Sequence[int],
    layer: int = 5,
    n_minor: int = 64,
    n_ood: int = 30,
    B: int = 64,
    n_gpus: Optional[int] = None,
    vocab_size: int = 8,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Plot maj_r2_ood vs training step for multiple coin experiments (by k)."""
    kwargs.setdefault("legend_title", r"$\log_2(n_{\mathrm{minor}})$")
    return plot_maj_r2_ood_across_steps(
        task_name="coin",
        experiments=list(k_list),
        layer=layer,
        steps=steps,
        n_minor=n_minor,
        n_ood=n_ood,
        B=B,
        n_gpus=n_gpus,
        vocab_size=vocab_size,
        **kwargs,
    )


def plot_maj_r2_ood_across_steps_linear(
    k_list: Sequence[int],
    steps: Sequence[int],
    layer: int,
    n_minor: int = 64,
    n_ood: int = 30,
    B: int = 64,
    n_gpus: Optional[int] = None,
    exp_name_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Plot maj_r2_ood vs training step for multiple linear experiments (by k)."""
    if exp_name_kwargs is None:
        exp_name_kwargs = {}
    experiments = [
        (get_exp_name("linear", k, **exp_name_kwargs), str(k))
        for k in k_list
    ]
    return plot_maj_r2_ood_across_steps(
        task_name="linear",
        experiments=experiments,
        layer=layer,
        steps=steps,
        n_minor=n_minor,
        n_ood=n_ood,
        B=B,
        n_gpus=n_gpus,
        **kwargs,
    )


def plot_maj_r2_ood_across_steps_latent(
    k_list: Sequence[int],
    steps: Sequence[int],
    layer: int,
    n_minor: int = 64,
    n_ood: int = 30,
    B: int = 64,
    n_gpus: Optional[int] = None,
    exp_name_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Plot maj_r2_ood vs training step for multiple latent experiments (by k)."""
    if exp_name_kwargs is None:
        exp_name_kwargs = {}
    experiments = [
        (get_exp_name("latent", k, **exp_name_kwargs), str(k))
        for k in k_list
    ]
    return plot_maj_r2_ood_across_steps(
        task_name="latent",
        experiments=experiments,
        layer=layer,
        steps=steps,
        n_minor=n_minor,
        n_ood=n_ood,
        B=B,
        n_gpus=n_gpus,
        **kwargs,
    )
