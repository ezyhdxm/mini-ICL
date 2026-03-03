from typing import Tuple, Dict, Optional, Sequence, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from icl.utils.attention_scores import get_attention_score_at_step

# Optional: interactive hover tooltips (nice replacement for Plotly hover).
try:
    import mplcursors  # pip install mplcursors
except Exception:
    mplcursors = None


def _format_step_compact(n: int) -> str:
    n = int(n)
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1_000_000_000:
        v = n / 1_000_000_000
        s = f"{v:.2f}B" if v < 10 else (f"{v:.1f}B" if v < 100 else f"{v:.0f}B")
    elif n >= 1_000_000:
        v = n / 1_000_000
        s = f"{v:.2f}M" if v < 10 else (f"{v:.1f}M" if v < 100 else f"{v:.0f}M")
    elif n >= 1_000:
        v = n / 1_000
        s = f"{v:.2f}k" if v < 10 else (f"{v:.1f}k" if v < 100 else f"{v:.0f}k")
    else:
        s = str(n)
    return sign + s


def _sparse_ticks(labels: Sequence[str], max_ticks: int = 6) -> Tuple[List[int], List[str]]:
    """
    Select a sparse subset of categorical ticks for matplotlib.

    labels: list[str]
    returns: (tick_positions, tick_text)
    """
    n = len(labels)
    if n <= max_ticks:
        idx = list(range(n))
        return idx, list(labels)

    idx = np.linspace(0, n - 1, max_ticks).round().astype(int)
    idx = np.unique(idx).tolist()  # safety
    ticktext = [labels[i] for i in idx]
    return idx, ticktext


def _get_default_colors() -> List[str]:
    # Use matplotlib's default cycle if available, else fall back to a stable list.
    colors = plt.rcParams.get("axes.prop_cycle", None)
    if colors is not None:
        by_key = colors.by_key()
        if "color" in by_key and by_key["color"]:
            return list(by_key["color"])

    return [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]


# A stable K color list (matplotlib's default cycle is usually tab10 anyway)
_K_COLORS = _get_default_colors()

# Matplotlib line/marker cycles approximating the Plotly ones
_DASH_CYCLE = [
    "-",          # solid
    "--",         # dash
    ":",          # dot
    "-.",         # dashdot
    (0, (7, 3)),  # longdash-ish
    (0, (7, 2, 1.5, 2)),  # longdashdot-ish
]
_MARKER_CYCLE = ["o", "s", "D", "+", "x", "^", "v", "<", ">", "*", "P"]


def plot_attention_scores_over_steps_mpl(
    k: int,
    steps: Sequence[int],
    mode: str = "ood",
    force_recompute: bool = False,
    title_prefix: Optional[str] = None,
    show: bool = True,
) -> Dict[str, Figure]:
    """
    Matplotlib version of the Plotly plot.

    Notes:
      - x-axis uses evenly spaced positions (0..S-1) with compact step tick labels.
      - Hover is not native in static matplotlib; if `mplcursors` is installed, you get tooltips.

    Returns:
      dict with keys: "ih", "pth", "real_pth" mapping to matplotlib Figures.
    """
    if not isinstance(steps, (list, tuple)):
        steps = list(steps)
    if len(steps) == 0:
        raise ValueError("steps must be non-empty")

    steps_list = [int(s) for s in steps]
    step_labels = [_format_step_compact(s) for s in steps_list]

    # ---- compute scores per step ----
    ih_list, pth_list, rpth_list = [], [], []
    for st in steps_list:
        ih, pth, rpth = get_attention_score_at_step(
            k=k, step=st, mode=mode, force_recompute=force_recompute
        )
        ih_list.append(ih.detach().float().cpu())
        pth_list.append(pth.detach().float().cpu())
        rpth_list.append(rpth.detach().float().cpu())

    ih_arr = torch.stack(ih_list, dim=0).numpy()   # (S, L, H)
    pth_arr = torch.stack(pth_list, dim=0).numpy()
    rpth_arr = torch.stack(rpth_list, dim=0).numpy()

    S, L, H = ih_arr.shape
    x = np.arange(S)

    colors = _get_default_colors()

    def _attach_hover(ax: Axes, metric_key: str):
        """
        If mplcursors is available, attach hover tooltips to all lines in ax.
        """
        if mplcursors is None:
            return

        cursor = mplcursors.cursor(ax.lines, hover=True)

        @cursor.connect("add")
        def _on_add(sel):
            line = sel.artist
            idx = sel.index  # index of the selected data point
            meta = getattr(line, "_icl_meta", None)
            if not meta:
                return

            step_int = meta["steps"][idx]
            layer = meta["layer"]
            head = meta["head"]
            yval = float(line.get_ydata()[idx])

            sel.annotation.set_text(
                f"step={step_int:,}\nlayer={layer}\nhead={head}\n{metric_key}={yval:.4f}"
            )
            # Make the tooltip a little less intrusive
            sel.annotation.get_bbox_patch().set(alpha=0.9)

    def _make_fig(metric_name: str, metric_key: str, arr: np.ndarray) -> Figure:
        fig, ax = plt.subplots(figsize=(11, 6.5))  # ~1100x650 at 100dpi

        # Plot all traces
        for l in range(L):
            layer_color = colors[l % len(colors)]
            for h in range(H):
                y = arr[:, l, h]
                line, = ax.plot(
                    x,
                    y,
                    label=f"L{l} · H{h}",
                    color=layer_color,
                    linestyle=_DASH_CYCLE[h % len(_DASH_CYCLE)],
                    marker=_MARKER_CYCLE[h % len(_MARKER_CYCLE)],
                    linewidth=2,
                    markersize=5,
                )
                # Store metadata for optional hover
                line._icl_meta = {
                    "steps": steps_list,
                    "layer": l,
                    "head": h,
                }

        prefix = (title_prefix + " · ") if title_prefix else ""
        ax.set_title(f"{prefix}{metric_name} scores (mode={mode}, k={k})")
        ax.set_xlabel("step")
        ax.set_ylabel("attention score")

        # Sparse x ticks with compact labels
        tickpos, ticktext = _sparse_ticks(step_labels, max_ticks=6)
        ax.set_xticks(tickpos)
        ax.set_xticklabels(ticktext)

        ax.grid(True, which="major", axis="both", alpha=0.3)
        ax.set_ylim(bottom=0)

        # Legend handling:
        # Plotly can handle huge legends interactively; matplotlib can't.
        # If L*H is big, switch to a compact legend (layers + head styles).
        n_traces = L * H
        if n_traces <= 30:
            # Per-trace legend (can still be large, but manageable for small models)
            ax.legend(
                title="Layer · Head",
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize="small",
            )
            fig.subplots_adjust(right=0.78)
        else:
            # Compact legends: one for layers (colors), one for heads (styles)
            layer_handles = [
                Line2D([0], [0], color=colors[l % len(colors)], lw=2, label=f"Layer {l}")
                for l in range(L)
            ]
            head_handles = [
                Line2D(
                    [0], [0],
                    color="black",
                    lw=2,
                    linestyle=_DASH_CYCLE[h % len(_DASH_CYCLE)],
                    marker=_MARKER_CYCLE[h % len(_MARKER_CYCLE)],
                    markersize=5,
                    label=f"Head {h}",
                )
                for h in range(H)
            ]

            leg1 = ax.legend(
                handles=layer_handles,
                title="Layers (color)",
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize="small",
            )
            ax.add_artist(leg1)
            ax.legend(
                handles=head_handles,
                title="Heads (style)",
                loc="upper left",
                bbox_to_anchor=(1.02, 0.55),
                borderaxespad=0.0,
                fontsize="small",
            )
            fig.subplots_adjust(right=0.72)

        _attach_hover(ax, metric_key)
        return fig

    figs = {
        "ih": _make_fig("Induction (successor/pad union)", "ih", ih_arr),
        "pth": _make_fig("PTH (padded queries)", "pth", pth_arr),
        "real_pth": _make_fig("Real-PTH (real queries)", "real_pth", rpth_arr),
    }

    if show:
        plt.show()

    return figs


def plot_max_scores_over_k_steps_mpl(
    ks: Sequence[int],
    steps: Sequence[int],
    mode: str = "ood",
    force_recompute: bool = False,
    title_prefix: Optional[str] = None,
    max_xticks: int = 7,
    show: bool = True,
) -> Tuple[Dict[str, Figure], Dict[int, Dict[int, Dict[str, float]]]]:
    """
    Matplotlib version:
      - For each (k, step), compute ih/pth/real_pth and keep max over (layers, heads)
      - Create 3 matplotlib figures, one per metric, with one trace per k

    Returns:
      (figs, data_dict) where data_dict[k][step][metric] = max_score
    """
    ks_list = [int(x) for x in ks]
    steps_list = [int(s) for s in steps]
    if len(ks_list) == 0 or len(steps_list) == 0:
        raise ValueError("ks and steps must both be non-empty")

    step_labels = [_format_step_compact(s) for s in steps_list]
    tickpos, ticktext = _sparse_ticks(step_labels, max_ticks=max_xticks)
    x = np.arange(len(steps_list))

    data: Dict[int, Dict[int, Dict[str, float]]] = {k: {} for k in ks_list}

    # ---- compute ----
    for k in ks_list:
        for st in steps_list:
            ih, pth, rpth = get_attention_score_at_step(
                k=k, step=st, mode=mode, force_recompute=force_recompute
            )
            data[k][st] = {
                "ih": float(ih.detach().max().item()),
                "pth": float(pth.detach().max().item()),
                "real_pth": float(rpth.detach().max().item()),
            }

    def _attach_hover(ax: Axes, metric_key: str):
        if mplcursors is None:
            return

        cursor = mplcursors.cursor(ax.lines, hover=True)

        @cursor.connect("add")
        def _on_add(sel):
            line = sel.artist
            idx = sel.index
            meta = getattr(line, "_icl_meta", None)
            if not meta:
                return

            step_int = meta["steps"][idx]
            k_val = meta["k"]
            yval = float(line.get_ydata()[idx])

            sel.annotation.set_text(
                f"k={k_val}\nstep={step_int:,}\n{metric_key}={yval:.4f}"
            )
            sel.annotation.get_bbox_patch().set(alpha=0.9)

    def _make_metric_fig(metric_key: str, metric_title: str) -> Figure:
        fig, ax = plt.subplots(figsize=(11, 5.2))  # ~1100x520 @ 100dpi

        prefix = (title_prefix + " · ") if title_prefix else ""

        for i, k in enumerate(ks_list):
            color = _K_COLORS[i % len(_K_COLORS)]
            y = [data[k][st][metric_key] for st in steps_list]

            line, = ax.plot(
                x,
                y,
                label=f"k={k}",
                color=color,
                linewidth=2,
                marker="o",
                markersize=5,
            )
            line._icl_meta = {"steps": steps_list, "k": k}

        ax.set_title(f"{prefix}{metric_title} (max over layers & heads) · mode={mode}")
        ax.set_xlabel("step")
        ax.set_ylabel("max attention score")

        ax.set_xticks(tickpos)
        ax.set_xticklabels(ticktext, rotation=0 if len(tickpos) <= 8 else 35, ha="right")

        ax.grid(True, which="major", axis="both", alpha=0.3)
        ax.set_ylim(bottom=0)

        ax.legend(
            title="k",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
        fig.subplots_adjust(right=0.78)

        _attach_hover(ax, metric_key)
        return fig

    figs = {
        "ih": _make_metric_fig("ih", "Induction (successor/pad union)"),
        "pth": _make_metric_fig("pth", "PTH (padded queries)"),
        "real_pth": _make_metric_fig("real_pth", "Real-PTH (real queries)"),
    }

    if show:
        plt.show()

    return figs, data
