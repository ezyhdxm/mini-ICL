from __future__ import annotations

from typing import Optional, Union, Sequence, List, Tuple
import numpy as np
import plotly.graph_objects as go
import torch
from scipy.optimize import curve_fit
from tqdm.notebook import trange


def plot_task_vector_variance_with_fit(task_vectors: torch.Tensor, normalize: bool = True,
                                       save_path: Union[str, None] = "tvar_plot.png") -> None:
    """
    Plot hidden vector variance vs. position across tasks, with a fitted power-law curve.

    Parameters:
    -----------
    task_vectors : torch.Tensor
        Tensor of shape (n_tasks, sequence_length, embedding_dim)
    save_path : str or None
        If provided, the plot is saved to this file (e.g., "tvar_plot.png"). If None, no file is saved.
    """

    if normalize:
        task_vectors = task_vectors / task_vectors.norm(dim=-1, keepdim=True)

    # Step 1: Compute per-task variance over embedding dim
    tvs_means = task_vectors.mean(dim=-2, keepdim=True)  # mean over sequence
    tsds = (task_vectors - tvs_means).norm(dim=-1)       # shape: (n_tasks, seq_len)
    tvars = (tsds**2).mean(dim=-1).cpu().numpy()         # shape: (n_tasks, seq_len)
    
    x = np.arange(1, tvars.shape[1] + 1)                 # Position index
    mean_tvar = tvars.mean(axis=0)

    # Step 2: Power-law model
    def power_law_model(x, a, c, b):
        return a * x**c + b

    fit_successful = False
    try:
        popt, _ = curve_fit(power_law_model, x[10:], mean_tvar[10:], p0=(1.0, -1.0, 0.0), maxfev=10000)
        a_fit, c_fit, b_fit = popt
        fitted_curve = power_law_model(x[10:], *popt)
        fit_successful = True
    except (RuntimeError, ValueError) as e:
        print(f"[Warning] curve_fit failed: {e}")
        fitted_curve = None

    # Step 3: Plotting
    fig = go.Figure()

    # Plot each task
    for i in range(tvars.shape[0]):
        fig.add_trace(go.Scatter(
            x=x,
            y=tvars[i],
            mode='lines',
            opacity=0.5,
            name=f"Task {i}" if i < 5 else None,
            showlegend=i < 5,
            line=dict(width=1)
        ))

    # Plot fitted curve
    if fit_successful and fitted_curve is not None:
        label_text = f"$\\mathrm{{Fit}}: {a_fit:.2f} \\cdot x^{{{c_fit:.2f}}} {'-' if b_fit < 0 else '+'} {abs(b_fit):.2f}$"
        fig.add_trace(go.Scatter(
            x=x[10:],
            y=fitted_curve,
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            opacity=0.8,
            name=label_text
        ))

    fig.update_layout(
        title="",
        xaxis_title="Position",
        yaxis_title="Variance",
        width=800,
        height=600,
        template="plotly_white"
    )

    if save_path:
        fig.write_image(save_path, scale=3)

    fig.show()




def plot_lambdas(lambdas, convex_combs=None):
    """
    Create an interactive Plotly plot with a dropdown to select task k,
    and plot λ^{(k)}_{j',t} over positions t with LaTeX title.
    
    Args:
        lambdas: np.array of shape (num_tasks, seq_len, num_tasks)
    """
    num_tasks, seq_len, num_fit_tasks = lambdas.shape
    x = list(range(seq_len))
    traces = []

    # Prepare traces for all k and j'
    for k in range(num_tasks):
        lambda_k = lambdas[k]  # shape: (seq_len, num_tasks)
        for j_prime in range(num_fit_tasks):
            trace = go.Scatter(
                x=x,
                y=lambda_k[:, j_prime],
                mode='lines+markers',
                name=f"$\\lambda_{{{j_prime}}}$",
                visible=(k == 0),  # only show k=0 initially
            )
            traces.append(trace)

    # Dropdown buttons for each task k
    dropdown_buttons = []
    for k in range(num_tasks):
        visibility = [False] * (num_tasks * num_fit_tasks)
        for i in range(num_fit_tasks):
            visibility[k * num_fit_tasks + i] = True

        if convex_combs is None:
            button = dict(
                label=f"Task k = {k}",
                method="update",
                args=[
                    {"visible": visibility},
                    {"title.text": f"$\\lambda^{{({k})}}_{{j', t}}$ over positions"}
                ]
            )
        else:
            a0, a1, a2 = convex_combs[k]
            button = dict(
                label=f"Task k = {k}",
                method="update",
                args=[
                    {"visible": visibility},
                    {"title.text": f"${a0:.2f}\\theta_{{0}} + {a1:.2f}\\theta_{{1}} + {a2:.2f}\\theta_{{2}}$ over positions"}
                ]
            )
        dropdown_buttons.append(button)

    # Create the figure
    if convex_combs is not None:
        a0, a1, a2 = convex_combs[0]
    init_title = "$\\lambda^{(0)}_{j', t}$ over positions" if convex_combs is None else f"${a0:.2f}\\theta_{{0}} + {a1:.2f}\\theta_{{1}} + {a2:.2f}\\theta_{{2}}$ over positions"
    fig = go.Figure(data=traces)
    fig.update_layout(
        updatemenus=[dict(
            buttons=dropdown_buttons,
            direction="down",
            showactive=True,
            x=0.5,
            y=1.15,
            xanchor="left",
            yanchor="top"
        )],
        title_text="",  # no title for paper
        xaxis_title="Position $t$",
        yaxis_title="Belief weight $\\lambda$",
        legend_title="$j'$ (latent task index)",
        height=500
    )

    fig.show()


def plot_task_vector_modes(
    task_vectors: torch.Tensor,
    normalize: bool = True,
    save_path: Union[str, None] = None,
    verbose: bool = False,
    trace_names: Optional[list[str]] = None,  # NEW: optional custom labels for each "task"
) -> None:
    """
    Expects 4D task_vectors: (n_tasks, seq_len, B, emb_dim).
    trace_names (optional): list of length n_tasks to name traces.
    """
    if task_vectors.ndim != 4:
        raise ValueError(f"plot_task_vector_modes expects 4D tensor, got shape {tuple(task_vectors.shape)}")

    if normalize:
        task_vectors = task_vectors / (task_vectors.norm(dim=-1, keepdim=True) + 1e-12)

    n_tasks, seq_len, _, _ = task_vectors.shape
    x = np.arange(1, seq_len + 1)

    if trace_names is None:
        trace_names = [f"Task {i}" for i in range(n_tasks)]
    else:
        if len(trace_names) != n_tasks:
            raise ValueError(f"trace_names must have length {n_tasks}, got {len(trace_names)}")

    # --- Mode 1: Task vector variance and power-law fit ---
    tvs_means = task_vectors.mean(dim=-2, keepdim=True)
    tsds = (task_vectors - tvs_means).norm(dim=-1)
    tvars = (tsds**2).mean(dim=-1).cpu().numpy()  # (n_tasks, seq_len)
    mean_tvar = tvars.mean(axis=0)

    def power_law_model(x, a, c, b):
        return a * x**c + b

    try:
        popt, _ = curve_fit(power_law_model, x[10:], mean_tvar[10:], p0=(1.0, -1.0, 0.0), maxfev=10000)
        fitted_curve = power_law_model(x[10:], *popt)
        a_fit, c_fit, b_fit = popt
        fit_label = f"Fit: {a_fit:.2f}·x^{c_fit:.2f} {'-' if b_fit < 0 else '+'} {abs(b_fit):.2f}"
    except Exception as e:
        if verbose:
            print(f"[Warning] curve_fit failed: {e}")
        fitted_curve = None
        fit_label = "Fit failed"

    # --- Mode 2: Mean of task vector differences ---
    avg_task_vector = task_vectors.mean(dim=0)
    diff_norms = []
    for i in range(n_tasks):
        diff = task_vectors[i] - avg_task_vector
        diff_mean = diff.mean(dim=1)
        diff_norm = diff_mean.norm(dim=-1)
        diff_norms.append(diff_norm.cpu().numpy())

    # --- Mode 3: Variance of task vector differences ---
    pairwise_tvars = []
    for i in range(n_tasks):
        diff = task_vectors[i] - avg_task_vector
        tsds_diff = (diff - diff.mean(dim=-2, keepdim=True)).norm(dim=-1)
        tvar_diff = (tsds_diff**2).mean(dim=-1).cpu().numpy()
        pairwise_tvars.append(tvar_diff)

    # --- Plotly Traces ---
    fig = go.Figure()

    # Traces for Mode 1
    for i in range(n_tasks):
        fig.add_trace(go.Scatter(
            x=x, y=tvars[i], mode="lines", opacity=0.5,
            line=dict(width=1),
            name=trace_names[i],
            visible=True
        ))
    if fitted_curve is not None:
        fig.add_trace(go.Scatter(
            x=x[10:], y=fitted_curve, mode="lines",
            line=dict(color="red", dash="dash", width=2),
            name=fit_label,
            visible=True
        ))

    # Traces for Mode 2
    for i in range(n_tasks):
        fig.add_trace(go.Scatter(
            x=x, y=diff_norms[i], mode="lines", opacity=0.5,
            line=dict(width=1),
            name=f"Diff {trace_names[i]}",
            visible=False
        ))

    # Traces for Mode 3
    for i in range(n_tasks):
        fig.add_trace(go.Scatter(
            x=x, y=pairwise_tvars[i], mode="lines", opacity=0.5,
            line=dict(width=1),
            name=f"VarDiff {trace_names[i]}",
            visible=False
        ))

    # --- Dropdown Menus ---
    n_mode1 = n_tasks + (1 if fitted_curve is not None else 0)
    n_mode2 = n_tasks
    n_mode3 = n_tasks
    total_traces = n_mode1 + n_mode2 + n_mode3

    def visibility_mask(n_total, start, length):
        return [start <= i < start + length for i in range(n_total)]

    dropdown_buttons = [
        dict(
            label="Task Vector Variance (Power-law fit)",
            method="update",
            args=[
                {"visible": visibility_mask(total_traces, 0, n_mode1)},
                {"title": {"text": "Hidden Vector Variance vs Position"},
                 "yaxis": {"title": "Variance"}}
            ],
        ),
        dict(
            label="Mean of Task Vector Differences",
            method="update",
            args=[
                {"visible": visibility_mask(total_traces, n_mode1, n_mode2)},
                {"title": {"text": "Mean of Hidden Vector Differences vs Position"},
                 "yaxis": {"title": "Norm"}}
            ],
        ),
        dict(
            label="Variance of Task Vector Differences",
            method="update",
            args=[
                {"visible": visibility_mask(total_traces, n_mode1 + n_mode2, n_mode3)},
                {"title": {"text": "Variance of Hidden Vector Differences vs Position"},
                 "yaxis": {"title": "Variance"}}
            ],
        ),
    ]

    # --- Final Layout ---
    fig.update_layout(
        updatemenus=[dict(
            active=0, buttons=dropdown_buttons,
            x=0.01, y=1.1, xanchor="left", yanchor="top"
        )],
        title="",
        xaxis_title="Position",
        yaxis_title="Variance",
        width=900,
        height=600,
        template="plotly_white",
    )

    if save_path:
        fig.write_image(save_path, scale=3)

    fig.show()


from plotly.subplots import make_subplots

def plot_task_vocab_vector_modes(
    task_vectors: torch.Tensor,
    normalize: bool = True,
    save_path: Union[str, None] = None,
    verbose: bool = False,
) -> None:
    """
    Wrapper:
      - If 4D: assumes (task, seq_len, *, *) and calls plot_task_vector_modes.
      - If 5D: assumes (task, vocab, seq_len, *, *), flattens (task,vocab)->tv,
               and calls plot_task_vector_modes with trace names "Task t | Vocab v".
    """
    if task_vectors.ndim == 4:
        return plot_task_vector_modes(
            task_vectors, normalize=normalize, save_path=save_path, verbose=verbose
        )

    if task_vectors.ndim != 5:
        raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(task_vectors.shape)}")

    n_task, n_vocab, seq_len, a, b = task_vectors.shape  # (task, vocab, seq_len, *, *)
    flat = task_vectors.reshape(n_task * n_vocab, seq_len, a, b)

    trace_names = [f"Task {t} | Vocab {v}" for t in range(n_task) for v in range(n_vocab)]

    return plot_task_vector_modes(
        flat,
        normalize=normalize,
        save_path=save_path,
        verbose=verbose,
        trace_names=trace_names,
    )


from plotly.subplots import make_subplots


def plot_task_vector_exp(
    task_vectors: torch.Tensor,
    n_minor: int,
    *,
    normalize: bool = True,
    save_path: Union[str, None] = None,
    verbose: bool = False,
    trace_names: Optional[Sequence[str]] = None,
    major_k: int = 3,
    layer_names: Optional[Sequence[str]] = None,
    show: bool = True,
    ignore_vocab: bool = False,
):
    """
    Supports:
      - 4D: (n_tasks, seq_len, B, emb_dim)              -> treated as 1 layer
      - 5D: (n_layers, n_tasks, seq_len, B, emb_dim)
      - 6D: (n_layers, n_tasks, n_vocab, seq_len, B, emb_dim) -> (n_layers, n_tasks, seq_len, B, emb_dim*n_vocab)

    One figure with ONE dropdown:
      - Select Layer (updates all three subplots simultaneously)

    The three modes are shown simultaneously (nice + simple):
      (1) Mode 1: task vector variance over batch (+ optional mean power-law fit)
      (2) Mode 2: || mean_B(task - avg_task) ||
      (3) Mode 3: Var_B(task - avg_task) (via norms)

    Legend shows ONLY: major / ood / minor.
    Individual task names appear in hover only.
    """

    # -------- normalize to 5D --------
    if task_vectors.ndim == 4:
        task_vectors = task_vectors.unsqueeze(0)  # (1, K, T, B, D)
    elif task_vectors.ndim == 6:
        L, K, V, T, B, D = task_vectors.shape
        if not ignore_vocab:
            task_vectors = task_vectors.permute(0, 1, 3, 4, 5, 2).reshape(L, K, T, B, D*V)
        else:
            task_vectors = task_vectors.permute(0, 1, 3, 2, 4, 5).reshape(L, K, T, V*B, D)
    if task_vectors.ndim != 5:
        raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(task_vectors.shape)}")

    L, K, T, B, D = task_vectors.shape

    if n_minor < 0:
        raise ValueError("n_minor must be >= 0")
    if major_k < 0:
        raise ValueError("major_k must be >= 0")
    if major_k + n_minor > K:
        raise ValueError(f"major_k + n_minor must be <= n_tasks, got {major_k}+{n_minor} > {K}")

    x = np.arange(1, T + 1)

    # -------- names --------
    if trace_names is None:
        trace_names = [f"Task {i}" for i in range(K)]
    else:
        if len(trace_names) != K:
            raise ValueError(f"trace_names must have length {K}, got {len(trace_names)}")
        trace_names = list(trace_names)

    if layer_names is None:
        layer_names = [f"Layer {l}" for l in range(L)]
    else:
        if len(layer_names) != L:
            raise ValueError(f"layer_names must have length {L}, got {len(layer_names)}")
        layer_names = list(layer_names)

    # -------- grouping --------
    def group_of(i: int) -> str:
        if i < major_k:
            return "major"
        if i >= K - n_minor:
            return "minor"
        return "ood"

    groups = [group_of(i) for i in range(K)]
    group_order = ["major", "ood", "minor"]
    group_color = {"major": "#1f77b4", "ood": "#ff7f0e", "minor": "#2ca02c"}

    # Within-group styling to distinguish tasks without changing group colors
    dash_cycle = ["solid", "dot", "dash", "dashdot"]
    width_cycle = [2, 2, 1, 1]

    group_to_indices = {g: [] for g in group_order}
    for i, g in enumerate(groups):
        group_to_indices[g].append(i)

    task_style: List[Tuple[str, int]] = [("solid", 1) for _ in range(K)]
    for g in group_order:
        for rank, i in enumerate(group_to_indices[g]):
            task_style[i] = (dash_cycle[rank % len(dash_cycle)], width_cycle[rank % len(width_cycle)])

    # -------- fit helper --------
    def power_law_model(x_, a, c, b):
        return a * (x_ ** c) + b

    # -------- build subplot figure (3 panels) --------
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Mode 1: Variance over batch (per task) + mean power-law fit",
            "Mode 2: || mean_B(task − avg_task) ||",
            "Mode 3: Var_B(task − avg_task) (via norms)",
        ],
    )

    # We'll add:
    # - 3 dummy traces for legend entries (major/ood/minor) always visible
    # - traces for each layer (all hidden except initial layer)
    #
    # Keep track of which trace indices belong to each layer, so the dropdown
    # can toggle exactly those traces.
    layer_to_trace_idxs: Dict[int, List[int]] = {l: [] for l in range(L)}

    # Dummy legend entries (always visible)
    for g in group_order:
        fig.add_trace(
            go.Scatter(
                x=[x[0], x[0] + 1],
                y=[np.nan, np.nan],
                mode="lines",
                name=g,
                legendgroup=g,
                showlegend=True,
                line=dict(color=group_color[g], width=3),
                hoverinfo="skip",
                visible=True,
            ),
            row=1, col=1,
        )
    n_dummy = len(group_order)

    def add_task_traces_for_layer(
        l: int,
        y_mat: np.ndarray,      # (K, T)
        row: int,
        yname: str,
        visible: bool,
    ) -> None:
        for i in range(K):
            g = groups[i]
            dash, width = task_style[i]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_mat[i],
                    mode="lines",
                    name=trace_names[i],       # not in legend
                    legendgroup=g,
                    showlegend=False,
                    visible=visible,
                    opacity=0.85,
                    line=dict(color=group_color[g], dash=dash, width=width),
                    hovertemplate=(
                        f"layer={layer_names[l]}<br>"
                        f"group={g}<br>"
                        f"task={trace_names[i]}<br>"
                        "pos=%{x}<br>"
                        f"{yname}=%{{y:.6g}}<extra></extra>"
                    ),
                ),
                row=row, col=1,
            )
            layer_to_trace_idxs[l].append(len(fig.data) - 1)

    # initial layer shown
    init_layer = 0

    # -------- add all layers’ traces (precomputed per layer) --------
    for l in range(L):
        tv = task_vectors[l]  # (K,T,B,D)
        if normalize:
            tv = tv / (tv.norm(dim=-1, keepdim=True) + 1e-12)

        # Mode 1: variance over batch
        tv_meanB = tv.mean(dim=-2, keepdim=True)           # (K,T,1,D)
        tsds = (tv - tv_meanB).norm(dim=-1)                # (K,T,B)
        tvars = (tsds ** 2).mean(dim=-1).cpu().numpy()     # (K,T)
        mean_tvar = tvars.mean(axis=0)                     # (T,)

        # Optional fit on mean curve
        fit_start = 10 if T > 12 else max(1, T // 4)
        fitted_curve = None
        fit_label = "Mean fit failed"
        try:
            popt, _ = curve_fit(
                power_law_model,
                x[fit_start:],
                mean_tvar[fit_start:],
                p0=(1.0, -1.0, 0.0),
                maxfev=10000,
            )
            fitted_curve = power_law_model(x[fit_start:], *popt)
            a_fit, c_fit, b_fit = popt
            fit_label = f"Mean fit: {a_fit:.2f}·x^{c_fit:.2f} {'-' if b_fit < 0 else '+'} {abs(b_fit):.2f}"
        except Exception as e:
            if verbose:
                print(f"[Layer {l}] curve_fit failed: {e}")

        # Mode 2: mean norm of (task - avg_task)
        avg_task = tv.mean(dim=0)                           # (T,B,D)
        diff = tv - avg_task.unsqueeze(0)                   # (K,T,B,D)
        diff_meanB = diff.mean(dim=-2)                      # (K,T,D)
        diff_norms = diff_meanB.norm(dim=-1).cpu().numpy()  # (K,T)

        # Mode 3: variance over batch of (task - avg_task)
        diff_mean_keepB = diff.mean(dim=-2, keepdim=True)   # (K,T,1,D)
        tsds_diff = (diff - diff_mean_keepB).norm(dim=-1)   # (K,T,B)
        vardiff = (tsds_diff ** 2).mean(dim=-1).cpu().numpy()  # (K,T)

        vis = (l == init_layer)

        # Add per-layer task traces to each subplot row
        add_task_traces_for_layer(l, tvars, row=1, yname="var",  visible=vis)
        add_task_traces_for_layer(l, diff_norms, row=2, yname="norm", visible=vis)
        add_task_traces_for_layer(l, vardiff, row=3, yname="var",  visible=vis)

        # Add fit curve (also layer-toggled)
        if fitted_curve is not None:
            fig.add_trace(
                go.Scatter(
                    x=x[fit_start:],
                    y=fitted_curve,
                    mode="lines",
                    name=fit_label,
                    legendgroup="fit",
                    showlegend=False,
                    visible=vis,
                    line=dict(color="red", dash="dash", width=3),
                    hovertemplate=(
                        f"layer={layer_names[l]}<br>"
                        "pos=%{x}<br>"
                        "mean_fit=%{y:.6g}<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )
            layer_to_trace_idxs[l].append(len(fig.data) - 1)

    # -------- dropdown: layer selection (toggle all three panels together) --------
    total_traces = len(fig.data)

    def visibility_for_layer(l: int) -> List[bool]:
        vis = [False] * total_traces
        # dummy legend entries always visible
        for i in range(n_dummy):
            vis[i] = True
        # this layer's real traces visible
        for idx in layer_to_trace_idxs[l]:
            vis[idx] = True
        return vis

    layer_buttons = []
    for l in range(L):
        layer_buttons.append(
            dict(
                label=layer_names[l],
                method="update",
                args=[
                    {"visible": visibility_for_layer(l)},
                    {"title": {"text": f"{layer_names[l]} · Task-vector diagnostics (3 modes)"}},
                ],
            )
        )

    # -------- layout polish --------
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=layer_buttons,
                direction="down",
                x=0.01,
                y=1.17,
                xanchor="left",
                yanchor="top",
                active=init_layer,
                showactive=True,
            )
        ],
        title="",
        template="plotly_white",
        width=1000,
        height=900,
        legend=dict(
            title="Group",
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
        ),
        margin=dict(l=70, r=220, t=110, b=60),
    )
    fig.update_xaxes(title_text="Position", row=3, col=1)
    fig.update_yaxes(title_text="Variance", row=1, col=1)
    fig.update_yaxes(title_text="Norm", row=2, col=1)
    fig.update_yaxes(title_text="Variance", row=3, col=1)

    # Save: exports only the currently visible layer state
    if save_path:
        fig.write_image(save_path, scale=3)

    if show:
        fig.show()
    
    return fig
