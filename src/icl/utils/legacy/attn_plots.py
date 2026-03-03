from typing import Tuple, List, Dict, Optional, Sequence
import numpy as np
import torch
import plotly.graph_objects as go

from icl.figures.head_view import head_view
from icl.utils.attention_scores import get_attention_score_at_step

try:
    import plotly.express as px
    _DEFAULT_COLORS = px.colors.qualitative.Plotly
except Exception:
    _DEFAULT_COLORS = None


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


def _sparse_ticks(labels, max_ticks=6):
    """
    Select a sparse subset of categorical ticks.

    labels: list[str]
    returns: (tickvals, ticktext)
    """
    n = len(labels)
    if n <= max_ticks:
        return labels, labels

    idx = np.linspace(0, n - 1, max_ticks).round().astype(int)
    idx = np.unique(idx)  # safety
    tickvals = [labels[i] for i in idx]
    ticktext = [labels[i] for i in idx]
    return tickvals, ticktext


try:
    import plotly.express as px
    _K_COLORS = px.colors.qualitative.Plotly
except Exception:
    _K_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                 "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

def plot_attention_scores_over_steps(
    k: int,
    steps: Sequence[int],
    mode: str = "ood",
    force_recompute: bool = False,
    title_prefix: Optional[str] = None,
    show: bool = True,
) -> Dict[str, go.Figure]:
    """
    Improved x-axis:
      - x is categorical (even spacing)
      - tick labels are compact (e.g. 120k, 1.05M)
      - hover shows the true step integer
    """
    if not isinstance(steps, (list, tuple)):
        steps = list(steps)
    if len(steps) == 0:
        raise ValueError("steps must be non-empty")

    steps_list = [int(s) for s in steps]
    step_labels = [_format_step_compact(s) for s in steps_list]  # x tick labels

    # ---- compute scores per step ----
    ih_list, pth_list, rpth_list = [], [], []
    for st in steps_list:
        ih, pth, rpth = get_attention_score_at_step(
            k=k, step=st, mode=mode, force_recompute=force_recompute
        )
        ih_list.append(ih.detach().float().cpu())
        pth_list.append(pth.detach().float().cpu())
        rpth_list.append(rpth.detach().float().cpu())

    ih_arr = torch.stack(ih_list, dim=0).numpy()       # (S, L, H)
    pth_arr = torch.stack(pth_list, dim=0).numpy()
    rpth_arr = torch.stack(rpth_list, dim=0).numpy()

    S, L, H = ih_arr.shape

    colors = _DEFAULT_COLORS or [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    dash_cycle = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
    marker_cycle = ["circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down"]

    def _make_fig(metric_name: str, arr: np.ndarray) -> go.Figure:
        fig = go.Figure()

        # x-values are categorical labels, but we also attach the true step via customdata for hover
        x = step_labels
        custom_steps = np.array(steps_list)

        for l in range(L):
            layer_color = colors[l % len(colors)]
            for h in range(H):
                y = arr[:, l, h]
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        customdata=custom_steps,
                        mode="lines+markers",
                        name=f"L{l} · H{h}",
                        legendgroup=f"layer_{l}",
                        line=dict(color=layer_color, dash=dash_cycle[h % len(dash_cycle)], width=2),
                        marker=dict(symbol=marker_cycle[h % len(marker_cycle)], size=7),
                        hovertemplate=(
                            "step=%{customdata:,}<br>"
                            f"layer={l}<br>"
                            f"head={h}<br>"
                            "score=%{y:.4f}<extra></extra>"
                        ),
                    )
                )

        prefix = (title_prefix + " · ") if title_prefix else ""
        fig.update_layout(
            title=f"{prefix}{metric_name} scores (mode={mode}, k={k})",
            xaxis_title="step",
            yaxis_title="attention score",
            template="plotly_white",
            width=1100,
            height=650,
            hovermode="x unified",  # nicer hover for many traces
            legend=dict(
                title="Layer · Head",
                orientation="v",
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.02,
            ),
            margin=dict(l=70, r=260, t=70, b=80),
        )

        tickvals, ticktext = _sparse_ticks(step_labels, max_ticks=6)

        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=step_labels,
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=0,
            showgrid=True,
        )

        fig.update_yaxes(showgrid=True, rangemode="tozero")

        return fig

    figs = {
        "ih": _make_fig("Induction (successor/pad union)", ih_arr),
        "pth": _make_fig("PTH (padded queries)", pth_arr),
        "real_pth": _make_fig("Real-PTH (real queries)", rpth_arr),
    }

    if show:
        for f in figs.values():
            f.show()

    return figs




def plot_max_scores_over_k_steps(
    ks: Sequence[int],
    steps: Sequence[int],
    mode: str = "ood",
    force_recompute: bool = False,
    title_prefix: Optional[str] = None,
    max_xticks: int = 7,
    show: bool = True,
) -> Tuple[Dict[str, go.Figure], Dict[int, Dict[int, Dict[str, float]]]]:
    """
    For each (k, step), compute metrics (ih, pth, real_pth) and keep max over (layers, heads).
    Then create 3 Plotly figures, one per metric, with one trace per k.

    Returns (figs, data_dict) where:
      data_dict[k][step][metric] = max_score  (float)
    """
    ks_list = [int(x) for x in ks]
    steps_list = [int(s) for s in steps]
    if len(ks_list) == 0 or len(steps_list) == 0:
        raise ValueError("ks and steps must both be non-empty")

    # Use categorical x for nicer spacing
    step_labels = [_format_step_compact(s) for s in steps_list]
    tickvals, ticktext = _sparse_ticks(step_labels, max_ticks=max_xticks)

    # Store results
    data: Dict[int, Dict[int, Dict[str, float]]] = {k: {} for k in ks_list}

    # ---- compute ----
    for k in ks_list:
        for st in steps_list:
            ih, pth, rpth = get_attention_score_at_step(
                k=k, step=st, mode=mode, force_recompute=force_recompute
            )
            # each is (n_layers, n_heads)
            ih_max = float(ih.detach().max().item())
            pth_max = float(pth.detach().max().item())
            rpth_max = float(rpth.detach().max().item())
            data[k][st] = {"ih": ih_max, "pth": pth_max, "real_pth": rpth_max}

    def _make_metric_fig(metric_key: str, metric_title: str) -> go.Figure:
        fig = go.Figure()
        prefix = (title_prefix + " · ") if title_prefix else ""

        for i, k in enumerate(ks_list):
            color = _K_COLORS[i % len(_K_COLORS)]
            y = [data[k][st][metric_key] for st in steps_list]

            fig.add_trace(
                go.Scatter(
                    x=step_labels,
                    y=y,
                    customdata=np.array(steps_list),
                    mode="lines+markers",
                    name=f"k={k}",
                    line=dict(color=color, width=2),
                    marker=dict(size=7),
                    hovertemplate=(
                        f"k={k}<br>"
                        "step=%{customdata:,}<br>"
                        f"{metric_key}=%{{y:.4f}}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=f"{prefix}{metric_title} (max over layers & heads) · mode={mode}",
            xaxis_title="step",
            yaxis_title="max attention score",
            template="plotly_white",
            width=1100,
            height=520,
            hovermode="x unified",
            legend=dict(
                title="k",
                orientation="v",
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.02,
            ),
            margin=dict(l=70, r=220, t=70, b=70),
        )
        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=step_labels,
            tickvals=tickvals,
            ticktext=ticktext,
            showgrid=True,
            tickangle=0 if len(tickvals) <= 8 else -35,
        )
        fig.update_yaxes(showgrid=True, rangemode="tozero")
        return fig

    figs = {
        "ih": _make_metric_fig("ih", "Induction (successor/pad union)"),
        "pth": _make_metric_fig("pth", "PTH (padded queries)"),
        "real_pth": _make_metric_fig("real_pth", "Real-PTH (real queries)"),
    }

    if show:
        figs["ih"].show()
        figs["pth"].show()
        figs["real_pth"].show()

    return figs, data






def visualize_attention_with_head_view(
    attn_map,
    seq,
    *,
    max_tokens=None,        # e.g. 129 or None
    layers=None,            # e.g. [0, 5, 10] or None
    heads=None,             # e.g. [0] or None
    html_action="view",
):
    """
    Wrapper around head_view for attention maps stored as:
      attn_map: dict[layer_idx -> (1, H, T, T)]
      seq: (1, T) integer tensor

    This function:
    - orders layers correctly
    - slices tokens/attention if max_tokens is given
    - converts seq to token strings
    - calls head_view()
    """

    import torch

    # ---- 1. Order layers ----
    layer_keys = sorted(attn_map.keys())
    attention_list = [attn_map[k] for k in layer_keys]

    # ---- 2. Slice sequence ----
    if hasattr(seq, "detach"):
        seq_ids = seq.detach().cpu()[0]
    else:
        seq_ids = torch.tensor(seq)[0]

    if max_tokens is not None:
        seq_ids = seq_ids[:max_tokens]

    tokens = [str(int(x)) for x in seq_ids.tolist()]
    T = len(tokens)

    # ---- 3. Slice attention tensors ----
    sliced_attention = []
    for A in attention_list:
        # A: (1, H, T_full, T_full)
        if max_tokens is not None:
            A = A[:, :, :T, :T]
        sliced_attention.append(A)

    # ---- 4. Layer indices for head_view ----
    # head_view expects layer indices relative to the LIST, not original keys
    if layers is None:
        include_layers = list(range(len(sliced_attention)))
    else:
        # map original layer ids -> list indices
        key_to_idx = {k: i for i, k in enumerate(layer_keys)}
        include_layers = [key_to_idx[l] for l in layers]

    # ---- 5. Call head_view ----
    return head_view(
        sliced_attention,
        tokens=tokens,
        include_layers=include_layers,
        heads=heads,
        html_action=html_action,
    )