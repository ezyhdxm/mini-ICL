from typing import Tuple
from typing import Optional, Sequence
import torch
from tqdm.notebook import trange
import numpy as np
import plotly.graph_objects as go
import os

# The following functions are used to compute hidden representations from the model at different granularity and different layers.

def compute_hiddens(config,
                    model: torch.nn.Module,
                    sampler,
                    layer_index: int = 1,
                    return_final: bool = False,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract hidden representations from the attention block of a specified layer.
    
    Args:
        config: Configuration object, must contain device and model.emb_dim
        model: Transformer model
        sampler: Data sampler
        layer_index: Index of the layer to extract from
        return_final: If True, only return the hidden at the final position; 
                     if False, return all positions
    
    Returns:
        Tensor shape depends on return_final:
        - return_final=True:  (n_tasks, B, n_embd) - final position only
        - return_final=False: (n_tasks, seq_len-1, B, n_embd) - all positions
    """
    device = config.device
    B = 256 
    max_tasks = 64
    n_tasks = min(max_tasks, sampler.n_major_tasks + sampler.n_minor_tasks)
    seq_len = sampler.seq_len
    n_embd = config.model.emb_dim

    if return_final:
        output_shape = (n_tasks, B, n_embd)
        task_pos = 2 * seq_len - 1
    else:
        output_shape = (n_tasks, seq_len-1, B, n_embd) 
        task_pos = 2 * torch.arange(seq_len-1, device=device) + 1  # (seq_len-1,)

    all_hiddens = torch.empty(output_shape, device=device)

    def run_and_extract(batch_data: torch.Tensor, pos):
        cache = {}

        def hook_fn(module, inp, out):
            # out: (B, L, d_model)
            if isinstance(pos, torch.Tensor):
                # (B, P, d)
                cache['vecs'] = out.index_select(dim=1, index=pos).detach()
            else:
                # (B, d)
                cache['vecs'] = out[:, pos, :].detach()

        handle = model.layers[layer_index].attn_block.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(batch_data)
        handle.remove()
        return cache['vecs']

    for i in range(n_tasks):
        demo_data, _ = sampler.generate(mode="testing", task=i, num_samples=B)  # (B, L_in)

        vecs = run_and_extract(demo_data, task_pos)
        #   return_final=True  -> (B, n_embd)
        #   return_final=False -> (B, P, n_embd)

        if return_final:
            all_hiddens[i, :, :] = vecs  # (B, d)
        else:
            all_hiddens[i, :, :, :] = vecs.permute(1, 0, 2)  # (P, B, d)
    
    return all_hiddens




# A core visualization function. 


def project_with_r2_size(
    task_vecs_over_all_time,
    final_task_vecs,
    r2_scores,
    lambdas,
    task_labels=None,
    final_labels=None,
    n_minors=0,
    size_min=6,
    size_max=20,
    # --- NEW: extra hover data ---
    hover_data=None,          # shape (K,) or None
    hover_name="hover",       # label shown on hover
    hover_fmt=".3f",          # formatting for numeric hover values
):
    """
    Project high-dimensional task vectors to 2D plane and adjust scatter point sizes by R² scores.
    Uses final_task_vecs to construct the projection plane (via SVD to get first two principal components).
    Supports time slider for dynamically viewing projections at different timesteps.

    Args:
        task_vecs_over_all_time: (K, T, D)
        final_task_vecs: (3, D)
        r2_scores: (K, T)
        lambdas: (K, T, 3)
        task_labels: list[str] length K
        final_labels: list[str] length 3
        n_minors: int
        size_min, size_max: marker size range

        hover_data: optional extra hover field of shape (K,), same ordering as tasks.
                   (Static across time; if you want (K,T) later, we can extend similarly.)
        hover_name: label used in hover tooltip for hover_data
        hover_fmt: numeric format (Plotly d3-format-style like ".3f", ".2e", etc.)

    Returns:
        fig: plotly.graph_objects.Figure
    """

    def to_np(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    X = to_np(task_vecs_over_all_time)   # (K,T,D)
    F = to_np(final_task_vecs)           # (3,D)
    R2 = to_np(r2_scores)                # (K,T)
    L = to_np(lambdas)                   # (K,T,3)
    H = to_np(hover_data)                # (K,) or None

    K, T, D = X.shape

    if H is not None:
        H = np.asarray(H)
        if H.shape != (K,):
            H = H.reshape(-1)
            if H.shape[0] != K:
                raise ValueError(f"hover_data must have shape (K,), got {hover_data.shape} (K={K})")
        H = H.reshape(K, 1)  # (K,1) for concatenation

    # --- build projection plane from final_task_vecs ---
    F_center = F.mean(axis=0, keepdims=True)
    F0 = F - F_center
    U, S, Vt = np.linalg.svd(F0, full_matrices=False)
    basis = Vt[:2].T  # (D,2)
    F_proj = (F - F_center) @ basis  # (3,2)

    # Separate major, OOD, and minor tasks
    if n_minors > 0:
        n_non_minor = K - n_minors
        n_major = min(3, n_non_minor)     # first 3 are major
        n_ood = max(0, n_non_minor - 3)   # rest before minors are OOD

        X_major = X[:n_major]  # (n_major,T,D)
        X_ood = X[n_major:n_major + n_ood] if n_ood > 0 else None
        X_minor = X[n_major + n_ood:]  # (n_minors,T,D)

        R2_major = R2[:n_major]
        R2_ood = R2[n_major:n_major + n_ood] if n_ood > 0 else None
        R2_minor = R2[n_major + n_ood:]

        L_major = L[:n_major]
        L_ood = L[n_major:n_major + n_ood] if n_ood > 0 else None
        L_minor = L[n_major + n_ood:]

        H_major = H[:n_major] if H is not None else None
        H_ood = H[n_major:n_major + n_ood] if (H is not None and n_ood > 0) else None
        H_minor = H[n_major + n_ood:] if H is not None else None

        # Precompute projections
        X_major_proj_list = [(X_major[:, t, :] - F_center) @ basis for t in range(T)]
        X_ood_proj_list = [(X_ood[:, t, :] - F_center) @ basis for t in range(T)] if n_ood > 0 else None
        X_minor_proj_list = [(X_minor[:, t, :] - F_center) @ basis for t in range(T)]
    else:
        n_major = min(3, K)
        n_ood = max(0, K - 3)

        X_major = X[:n_major]
        X_ood = X[n_major:] if n_ood > 0 else None
        X_minor = None

        R2_major = R2[:n_major]
        R2_ood = R2[n_major:] if n_ood > 0 else None
        R2_minor = None

        L_major = L[:n_major]
        L_ood = L[n_major:] if n_ood > 0 else None
        L_minor = None

        H_major = H[:n_major] if H is not None else None
        H_ood = H[n_major:] if (H is not None and n_ood > 0) else None
        H_minor = None

        X_major_proj_list = [(X_major[:, t, :] - F_center) @ basis for t in range(T)]
        X_ood_proj_list = [(X_ood[:, t, :] - F_center) @ basis for t in range(T)] if n_ood > 0 else None
        X_minor_proj_list = None

    # Generate labels
    if n_minors > 0:
        if task_labels is None:
            major_labels = [f"major_{i}" for i in range(3)]
            ood_labels = [f"ood_{i}" for i in range(3, K - n_minors)]
            minor_labels = [f"minor_{i}" for i in range(n_minors)]
        else:
            major_labels = task_labels[:3]
            ood_labels = task_labels[3:K - n_minors] if K - n_minors > 3 else []
            minor_labels = [f"minor_{i}" for i in range(n_minors)]
    else:
        if task_labels is None:
            major_labels = [f"major_{i}" for i in range(3)]
            ood_labels = [f"ood_{i}" for i in range(3, K)] if K > 3 else []
        else:
            major_labels = task_labels[:3]
            ood_labels = task_labels[3:K] if K > 3 else []
        minor_labels = None

    if final_labels is None:
        final_labels = [f"final_{i}" for i in range(3)]

    # Map R2 -> size (linear scaling)
    def scale_sizes(r2_values):
        r2_min, r2_max = R2.min(), R2.max()
        if r2_max == r2_min:
            return np.full_like(r2_values, (size_min + size_max) / 2.0)
        return size_min + (r2_values - r2_min) / (r2_max - r2_min) * (size_max - size_min)

    # customdata builder: [R2, lambda1, lambda2, lambda3, (optional hover)]
    def make_custom(r2_col, lam_mat, h_col=None):
        base = np.concatenate([r2_col.reshape(-1, 1), lam_mat], axis=1)  # (n,4)
        if h_col is None:
            return base
        return np.concatenate([base, h_col], axis=1)  # (n,5)

    # hover template builder
    def make_hovertemplate(has_extra: bool):
        extra_line = f"<br>{hover_name}=%{{customdata[4]:{hover_fmt}}}" if has_extra else ""
        return (
            "task=%{text}"
            "<br>R²=%{customdata[0]:.3f}"
            "<br>λ₁=%{customdata[1]:.3f}"
            "<br>λ₂=%{customdata[2]:.3f}"
            "<br>λ₃=%{customdata[3]:.3f}"
            + extra_line
            + "<extra></extra>"
        )

    has_extra = H is not None

    # Initial scatter at t=0
    t0 = 0
    fig = go.Figure()

    # --- major tasks trace ---
    X_major0 = X_major_proj_list[t0]
    sizes_major0 = scale_sizes(R2_major[:, t0])
    custom_major0 = make_custom(R2_major[:, t0], L_major[:, t0, :], H_major)

    fig.add_trace(go.Scatter(
        x=X_major0[:, 0],
        y=X_major0[:, 1],
        mode="markers",
        name="major tasks",
        marker=dict(size=sizes_major0, opacity=0.8, sizemode="diameter", color="blue"),
        text=major_labels,
        customdata=custom_major0,
        hovertemplate=make_hovertemplate(has_extra),
    ))

    # --- OOD tasks trace ---
    if X_ood_proj_list is not None and len(ood_labels) > 0:
        X_ood0 = X_ood_proj_list[t0]
        sizes_ood0 = scale_sizes(R2_ood[:, t0])
        custom_ood0 = make_custom(R2_ood[:, t0], L_ood[:, t0, :], H_ood)

        fig.add_trace(go.Scatter(
            x=X_ood0[:, 0],
            y=X_ood0[:, 1],
            mode="markers",
            name="OOD tasks",
            marker=dict(size=sizes_ood0, opacity=0.8, sizemode="diameter", color="green", symbol="square"),
            text=ood_labels,
            customdata=custom_ood0,
            hovertemplate=make_hovertemplate(has_extra),
        ))

    # --- minor tasks trace ---
    if n_minors > 0:
        X_minor0 = X_minor_proj_list[t0]
        sizes_minor0 = scale_sizes(R2_minor[:, t0])
        custom_minor0 = make_custom(R2_minor[:, t0], L_minor[:, t0, :], H_minor)

        fig.add_trace(go.Scatter(
            x=X_minor0[:, 0],
            y=X_minor0[:, 1],
            mode="markers",
            name="minor tasks",
            marker=dict(size=sizes_minor0, opacity=0.8, sizemode="diameter", color="red", symbol="diamond"),
            text=minor_labels,
            customdata=custom_minor0,
            hovertemplate=make_hovertemplate(has_extra),
        ))

    # --- final refs trace ---
    fig.add_trace(go.Scatter(
        x=F_proj[:, 0],
        y=F_proj[:, 1],
        mode="markers+text",
        name="final refs",
        marker=dict(size=12, symbol="star", line=dict(width=1)),
        text=final_labels,
        textposition="top center",
        hoverinfo="skip",
    ))

    # Slider steps
    steps = []
    for t in range(T):
        X_major_t = X_major_proj_list[t]
        sizes_major_t = scale_sizes(R2_major[:, t])
        custom_major_t = make_custom(R2_major[:, t], L_major[:, t, :], H_major)

        # Lists must match trace order: major, (ood), (minor), final refs
        x_coords = [X_major_t[:, 0]]
        y_coords = [X_major_t[:, 1]]
        custom_data = [custom_major_t]
        markers = [dict(size=sizes_major_t, opacity=0.8, sizemode="diameter", color="blue")]
        texts = [major_labels]

        if X_ood_proj_list is not None and len(ood_labels) > 0:
            X_ood_t = X_ood_proj_list[t]
            sizes_ood_t = scale_sizes(R2_ood[:, t])
            custom_ood_t = make_custom(R2_ood[:, t], L_ood[:, t, :], H_ood)

            x_coords.append(X_ood_t[:, 0])
            y_coords.append(X_ood_t[:, 1])
            custom_data.append(custom_ood_t)
            markers.append(dict(size=sizes_ood_t, opacity=0.8, sizemode="diameter", color="green", symbol="square"))
            texts.append(ood_labels)

        if n_minors > 0:
            X_minor_t = X_minor_proj_list[t]
            sizes_minor_t = scale_sizes(R2_minor[:, t])
            custom_minor_t = make_custom(R2_minor[:, t], L_minor[:, t, :], H_minor)

            x_coords.append(X_minor_t[:, 0])
            y_coords.append(X_minor_t[:, 1])
            custom_data.append(custom_minor_t)
            markers.append(dict(size=sizes_minor_t, opacity=0.8, sizemode="diameter", color="red", symbol="diamond"))
            texts.append(minor_labels)

        # final refs (unchanged over time)
        x_coords.append(F_proj[:, 0])
        y_coords.append(F_proj[:, 1])
        custom_data.append(None)  # not used for final refs
        markers.append(None)      # keep marker unchanged
        texts.append(final_labels)

        steps.append(dict(
            method="update",
            args=[
                {"x": x_coords,
                 "y": y_coords,
                 "customdata": custom_data,
                 "marker": markers,
                 "text": texts},
                {"title": f"Projection at t={t}"}
            ],
            label=str(t)
        ))

    fig.update_layout(
        title="Projection with R² (hover & size)" + (
            " - Major/OOD/Minor tasks" if (n_minors > 0 or len(ood_labels) > 0) else ""
        ),
        xaxis_title="axis 1",
        yaxis_title="axis 2",
        sliders=[dict(
            active=0,
            steps=steps,
            currentvalue=dict(prefix="t = "),
            pad=dict(t=10)
        )],
        width=800,
        height=650,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig







































# Currently not used, not working as expected, needs to be fixed

@torch.no_grad()
def induction_head_score_with_pad_i_minus_1(attn: torch.Tensor,
                                            tokens: torch.Tensor,
                                            pad_id: int,
                                            reduce: str = "mean_over_batch") -> torch.Tensor:
    """
    Compute induction head (IH) score considering special matching conditions for padding tokens.
    Matching condition: x[j-2] == x[i-1] and x[j-1] is a padding token.
    This function is used to analyze whether the model has learned the induction head mechanism.
    
    Args:
        attn: Attention weights of shape (B, H, T, T2)
        tokens: Token sequences of shape (B, T)
        pad_id: ID of the padding token
        reduce: Reduction mode, "mean_over_batch" averages over batch dimension, "none" does no reduction
    
    Returns:
        scores: Shape (B, H) if reduce="none", otherwise shape (H,)
    """
    assert attn.dim() == 4 and tokens.dim() == 2
    B, H, T, T2 = attn.shape
    assert T == T2 and tokens.shape[1] == T
    dev = attn.device
    
    idx = torch.arange(T, device=dev)
    I = idx.view(T, 1).expand(T, T)          
    J = idx.view(1, T).expand(T, T)          
    j_lt_i = (J < I)                          

    is_pad_q = (tokens == pad_id)        

    prev1_is_pad = torch.zeros((B, T), dtype=torch.bool, device=dev)
    prev1_is_pad[:, 1:] = (tokens[:, :-1] == pad_id)   

    prev2 = torch.full((B, T), fill_value=-1, dtype=tokens.dtype, device=dev)
    prev2[:, 2:] = tokens[:, :-2]                       

    i_minus1 = torch.full((B, T), fill_value=-1, dtype=tokens.dtype, device=dev)
    i_minus1[:, 1:] = tokens[:, :-1]                    

    cond_q_pad   = is_pad_q[:, :, None]                         
    cond_j_lt_i  = j_lt_i[None, :, :]                           
    cond_prev1   = prev1_is_pad[:, None, :]                     
    cond_match   = (prev2[:, None, :] == i_minus1[:, :, None])  # (B,T,T)

    M = cond_q_pad & cond_j_lt_i & cond_prev1 & cond_match     

    M_f = M.float()
    num = (attn * M_f[:, None, :, :]).sum(dim=(-1, -2))     
    den = M_f.sum(dim=(-1, -2)).clamp_min(1e-8)             
    scores_bh = num / den[:, None]                           

    if reduce == "none":
        return scores_bh
    return scores_bh.mean(dim=0) 





