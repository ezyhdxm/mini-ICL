import numpy as np
import plotly.graph_objects as go
import torch
from scipy.optimize import curve_fit
from tqdm.notebook import trange
from typing import Union, Optional


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
        title="Hidden Vector Variance vs Position",
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
        title_text=init_title,  # initial title
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
    Expects 4D task_vectors: (n_tasks, seq_len, *, *).
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
        title="Hidden Vector Variance vs Position",
        xaxis_title="Position",
        yaxis_title="Variance",
        width=900,
        height=600,
        template="plotly_white",
    )

    if save_path:
        fig.write_image(save_path, scale=3)

    fig.show()


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

