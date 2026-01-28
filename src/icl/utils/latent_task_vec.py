from typing import Tuple
from typing import Optional, Sequence
import torch
from tqdm.notebook import trange
import numpy as np
import plotly.graph_objects as go
import os

# The following functions are used to compute hidden representations from the model at different granularity and different layers.

import torch

def compute_hiddens(
    config,
    model: torch.nn.Module,
    sampler,
    B: int = 64,
    return_data=False,
):
    """
    Extract hidden representations from the attention blocks of ALL layers.

    Returns:
        all_hiddens: (n_layers, n_tasks, seq_len-1, B, n_embd)
    """
    device = config.device
    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    seq_len = sampler.seq_len
    n_embd = config.model.emb_dim
    n_layers = len(model.layers)

    # positions corresponding to "real" tokens
    task_pos = 2 * torch.arange(seq_len - 1, device=device) + 1  # (P,)

    all_hiddens = torch.empty(
        (n_layers, n_tasks, seq_len - 1, B, n_embd),
        device=device,
    )

    def run_and_extract(batch_data):
        cache = {}
        handles = []

        for l in range(n_layers):
            def make_hook(layer_idx):
                def hook_fn(module, inp, out):
                    # out: (B, L, d)
                    cache[layer_idx] = out.index_select(1, task_pos).detach()
                return hook_fn

            h = model.layers[l].attn_block.register_forward_hook(
                make_hook(l)
            )
            handles.append(h)

        with torch.no_grad():
            _ = model(batch_data)

        for h in handles:
            h.remove()

        return cache  # layer -> (B, P, d)

    if return_data:
        data_collector = torch.empty((n_tasks, B, seq_len))

    for i in range(n_tasks):
        demo_data, _ = sampler.generate(
            mode="testing", task=i, num_samples=B
        )

        layer_vecs = run_and_extract(demo_data)

        for l in range(n_layers):
            # (B, P, d) -> (P, B, d)
            all_hiddens[l, i] = layer_vecs[l].permute(1, 0, 2)
        if return_data:
            data_collector[i] = demo_data[:, ::2]
    
    if return_data:
        return all_hiddens, data_collector # (n_layers, n_tasks, seq_len-1, B, n_embd)

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
    # --- flexible hover data ---
    hover_data=None,          # None, (...,), (...,Dh), or (...,T,Dh)
    hover_name="hover",       # str or list[str]
    hover_fmt=".3f",          # str or list[str]
    # --- NEW: mask annotation ---
    mask=None,                # None, (T,), (K,T), or (...,T) matching X leading dims
    show_mask_annotation=True,  # allow turning off even if mask is passed
    # --- Display options ---
    show_final_refs=True,     # whether to display final reference points and labels
):
    """
    Project high-dimensional task vectors to a 2D plane and adjust scatter marker sizes by R² scores.
    Uses final_task_vecs to construct the projection plane (via SVD to get first two principal components).
    Supports time slider for dynamically viewing projections at different timesteps.

    Supports flexible leading dims:
      - task_vecs_over_all_time: (..., T, D) -> flattened to (K, T, D)
      - r2_scores:              (..., T)    -> flattened to (K, T)
      - lambdas:                (..., T, 3) -> flattened to (K, T, 3)

    Hover data:
      - hover_data: None, (...,), (...,Dh), or (...,T,Dh)

    NEW: mask annotation (from your second function):
      - mask can be:
          * (T,)                 binary
          * (K,T)                binary
          * (...,T)              binary with same leading dims as X (will be flattened)
        We compute per-time counts and display cumulative sum + "legal prefixes".

    Minor handling:
      - n_minors refers to the last n_minors items after flattening.

    Display options:
      - show_final_refs: If False, hides the final reference points and labels from the plot.

    Returns:
        fig: plotly.graph_objects.Figure
    """
    import numpy as np
    import plotly.graph_objects as go
    from math import comb, floor

    # ----------------------------
    # Helpers
    # ----------------------------
    def to_np(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def prod(shape):
        out = 1
        for s in shape:
            out *= int(s)
        return int(out)

    def flatten_leading(X, *, name, tail_ndim):
        """
        Flatten leading dims of X so it becomes (K, *tail).
        tail_ndim: number of tail dims to preserve.
        Returns (X_flat, lead_shape, tail_shape)
        """
        X = np.asarray(X)
        if X.ndim < tail_ndim:
            raise ValueError(f"{name} must have at least {tail_ndim} dims, got shape {X.shape}")
        lead_shape = X.shape[: X.ndim - tail_ndim]
        tail_shape = X.shape[X.ndim - tail_ndim:]
        K = prod(lead_shape) if len(lead_shape) > 0 else 1
        return X.reshape((K, *tail_shape)), lead_shape, tail_shape

    def normalize_labels(task_labels, lead_shape, K):
        if task_labels is None:
            if len(lead_shape) == 2:
                a, b = lead_shape
                return [f"task{i}|vocab{j}" for i in range(a) for j in range(b)]
            return [f"item_{k}" for k in range(K)]

        LBL = to_np(task_labels)
        if isinstance(LBL, np.ndarray):
            if LBL.size == K:
                return [str(x) for x in LBL.reshape(-1).tolist()]
            try:
                flat = LBL.reshape(-1)
                if flat.shape[0] == K:
                    return [str(x) for x in flat.tolist()]
            except Exception:
                pass

        LBL_list = list(task_labels)
        if len(LBL_list) != K:
            raise ValueError(f"task_labels must have length K={K} after flattening; got {len(LBL_list)}")
        return [str(x) for x in LBL_list]

    def normalize_hover(H, lead_shape, T, K):
        """
        Returns:
          H_static: (K,Dh) or None
          H_time:   (K,T,Dh) or None
          Dh: int
          hover_names: list[str] length Dh
          hover_fmts: list[str] length Dh
        """
        if H is None:
            return None, None, 0, [], []

        H = np.asarray(H)

        # Allow either matching leading dims (...,*) or already-flattened (K,*)
        if H.shape[0] != K and H.shape[:len(lead_shape)] != lead_shape:
            raise ValueError(
                f"hover_data leading dims must match task_vecs leading dims {lead_shape} "
                f"or be flattened with first dim K={K}. Got hover_data shape {H.shape}."
            )

        if H.shape[0] != K:
            H = H.reshape((K, *H.shape[len(lead_shape):]))

        if H.ndim == 1:
            H_static, H_time = H.reshape(K, 1), None
        elif H.ndim == 2:
            H_static, H_time = H, None
        elif H.ndim == 3:
            if H.shape[1] != T:
                raise ValueError(f"hover_data time axis must have length T={T}; got {H.shape}")
            H_static, H_time = None, H
        else:
            raise ValueError(f"hover_data must be (K,), (K,Dh), or (K,T,Dh) after flattening; got {H.shape}")

        Dh = (H_static.shape[1] if H_static is not None else H_time.shape[2])

        hn = hover_name
        hf = hover_fmt

        if isinstance(hn, str):
            hover_names = [f"{hn}_{j}" for j in range(Dh)] if Dh > 1 else [hn]
        else:
            hover_names = list(hn)
            if len(hover_names) != Dh:
                raise ValueError(f"hover_name list length must equal Dh={Dh}, got {len(hover_names)}")

        if isinstance(hf, str):
            hover_fmts = [hf] * Dh
        else:
            hover_fmts = list(hf)
            if len(hover_fmts) != Dh:
                raise ValueError(f"hover_fmt list length must equal Dh={Dh}, got {len(hover_fmts)}")

        return H_static, H_time, Dh, hover_names, hover_fmts

    def normalize_mask(mask, lead_shape, T, K):
        """
        mask can be:
          - (T,)
          - (K,T)
          - (...,T) matching lead_shape
        Returns:
          per_t: (T,) int64 counts per time
          cum_mask: (T,) cumulative
        """
        if mask is None or (not show_mask_annotation):
            return None, None

        M = to_np(mask)
        M = np.asarray(M)

        if M.ndim == 1:
            if M.shape[0] != T:
                raise ValueError(f"1D mask must have length T={T}, got {M.shape[0]}")
            per_t = M.astype(np.int64)
        else:
            # Allow (...,T) or (K,T)
            if M.shape[-1] != T:
                raise ValueError(f"mask last dim must be T={T}; got shape {M.shape}")

            # If already (K,T) use it, else flatten leading dims
            if M.ndim == 2 and M.shape[0] == K:
                M_flat = M
            else:
                # Expect leading dims match lead_shape (or be flattenable to K)
                if M.shape[:len(lead_shape)] != lead_shape:
                    # still allow if it can be reshaped to (K,T) by total size
                    lead_prod = prod(M.shape[:-1])
                    if lead_prod != K:
                        raise ValueError(
                            f"mask leading dims must match task_vecs leading dims {lead_shape} "
                            f"or flatten to K={K}. Got mask shape {M.shape}."
                        )
                M_flat = M.reshape((K, T))

            per_t = M_flat.astype(np.int64).sum(axis=0)

        cum_mask = np.cumsum(per_t)
        return per_t, cum_mask

    # ----------------------------
    # Convert + flatten inputs
    # ----------------------------
    X_raw = to_np(task_vecs_over_all_time)
    F = to_np(final_task_vecs)
    R2_raw = to_np(r2_scores)
    L_raw = to_np(lambdas)
    H_raw = to_np(hover_data)

    if X_raw is None or R2_raw is None or L_raw is None or F is None:
        raise ValueError("task_vecs_over_all_time, final_task_vecs, r2_scores, and lambdas must be provided.")

    # X: (...,T,D) -> (K,T,D)
    X, lead_shape, _ = flatten_leading(X_raw, name="task_vecs_over_all_time", tail_ndim=2)
    if X.ndim != 3:
        raise ValueError(f"task_vecs_over_all_time must be shape (...,T,D). Got {X_raw.shape} -> {X.shape}")
    K, T, D = X.shape

    # R2: (...,T) -> (K,T)
    R2, _, _ = flatten_leading(R2_raw, name="r2_scores", tail_ndim=1)
    if R2.shape != (K, T):
        raise ValueError(f"r2_scores must be shape (...,T) matching X. Got {R2_raw.shape} -> {R2.shape}, expected {(K,T)}")

    # L: (...,T,3) -> (K,T,3)
    L, _, _ = flatten_leading(L_raw, name="lambdas", tail_ndim=2)
    if L.shape != (K, T, 3):
        raise ValueError(f"lambdas must be shape (...,T,3) matching X. Got {L_raw.shape} -> {L.shape}, expected {(K,T,3)}")

    # Hover
    H_static, H_time, Dh, hover_names, hover_fmts = normalize_hover(H_raw, lead_shape, T, K)

    # Mask
    _, cum_mask = normalize_mask(mask, lead_shape, T, K)

    # Labels
    all_labels = normalize_labels(task_labels, lead_shape, K)
    if final_labels is None:
        final_labels = [f"final_{i}" for i in range(3)]
    final_labels = list(final_labels)
    if len(final_labels) != 3:
        raise ValueError(f"final_labels must have length 3; got {len(final_labels)}")

    # ----------------------------
    # Projection plane from final_task_vecs
    # ----------------------------
    F = np.asarray(F)
    if F.ndim != 2 or F.shape[0] != 3:
        raise ValueError(f"final_task_vecs must have shape (3,D), got {F.shape}")
    if F.shape[1] != D:
        raise ValueError(f"final_task_vecs has D={F.shape[1]} but task_vecs have D={D}")

    F_center = F.mean(axis=0, keepdims=True)
    F0 = F - F_center
    _, _, Vt = np.linalg.svd(F0, full_matrices=False)
    basis = Vt[:2].T  # (D,2)
    F_proj = (F - F_center) @ basis  # (3,2)

    # Precompute projections over time
    X_proj_list = [(X[:, t, :] - F_center) @ basis for t in range(T)]  # list[(K,2)]

    # ----------------------------
    # Split indices into major / ood / minor
    # ----------------------------
    if n_minors < 0:
        raise ValueError("n_minors must be >= 0")
    if n_minors >= K and n_minors > 0:
        raise ValueError(f"n_minors={n_minors} must be < K={K}")

    if n_minors > 0:
        n_non_minor = K - n_minors
        n_major = min(3, n_non_minor)
        n_ood = max(0, n_non_minor - 3)

        idx_major = np.arange(0, n_major)
        idx_ood = np.arange(n_major, n_major + n_ood) if n_ood > 0 else None
        idx_minor = np.arange(n_major + n_ood, K)
    else:
        n_major = min(3, K)
        n_ood = max(0, K - 3)
        idx_major = np.arange(0, n_major)
        idx_ood = np.arange(n_major, K) if n_ood > 0 else None
        idx_minor = None
    
    # ----------------------------
    # NEW: Prepare OOD posterior colors (optional)
    # ----------------------------
    P_ood_plot = None
    major_palette_rgb = None

    if ood_posterior is not None and ood_count > 0:
        # palette for the K major anchors
        if major_palette is None:
            cmap = plt.get_cmap("tab10")
            major_palette = [cmap(i) for i in range(int(k_major))]
        major_palette_rgb = np.asarray([mcolors.to_rgb(c) for c in major_palette], dtype=float)

        P_ood_full = _normalize_ood_posterior(
            ood_posterior, T_full=T, Kmaj=int(k_major), ood_count=int(ood_count), b_ood=b_ood
        )
        # downsample with same indices as trajectories
        P_ood_plot = P_ood_full[:, t_idx, :]  # (ood_count, T_plot, k_major)


    major_labels = [all_labels[i] for i in idx_major.tolist()]
    ood_labels = [all_labels[i] for i in idx_ood.tolist()] if idx_ood is not None else []
    minor_labels = [all_labels[i] for i in idx_minor.tolist()] if idx_minor is not None else []

    # ----------------------------
    # Size scaling
    # ----------------------------
    def scale_sizes(r2_values):
        r2_min, r2_max = float(R2.min()), float(R2.max())
        if r2_max == r2_min:
            return np.full_like(r2_values, (size_min + size_max) / 2.0, dtype=float)
        return size_min + (r2_values - r2_min) / (r2_max - r2_min) * (size_max - size_min)

    # ----------------------------
    # customdata and hovertemplate
    # ----------------------------
    # customdata columns: [R2, lambda1, lambda2, lambda3, hover_0, ..., hover_{Dh-1}]
    def make_custom(idx, t):
        r2_col = R2[idx, t].reshape(-1, 1)      # (n,1)
        lam = L[idx, t, :]                      # (n,3)
        base = np.concatenate([r2_col, lam], axis=1)  # (n,4)

        if Dh == 0:
            return base

        if H_time is not None:
            h = H_time[idx, t, :]               # (n,Dh)
        else:
            h = H_static[idx, :]                # (n,Dh)
        return np.concatenate([base, h], axis=1)  # (n,4+Dh)

    def make_hovertemplate():
        parts = [
            "task=%{text}",
            "<br>R²=%{customdata[0]:.3f}",
            "<br>λ₁=%{customdata[1]:.3f}",
            "<br>λ₂=%{customdata[2]:.3f}",
            "<br>λ₃=%{customdata[3]:.3f}",
        ]
        for j in range(Dh):
            parts.append(f"<br>{hover_names[j]}=%{{customdata[{4+j}]:{hover_fmts[j]}}}")
        parts.append("<extra></extra>")
        return "".join(parts)

    hovertemplate = make_hovertemplate()

    # ----------------------------
    # Mask annotation (NEW)
    # ----------------------------
    def make_annotation(t):
        if cum_mask is None or (not show_mask_annotation):
            txt = ""
        else:
            total = int(cum_mask[-1])
            curr = int(cum_mask[t])
            # NOTE: comb can overflow for very large curr; that's inherent.
            legal_prefix_count = comb(curr, floor(curr / 2)) if curr >= 0 else 0
            txt = f"Cumsum mask up to t={t}: {curr} / {total} (legal prefixes: {legal_prefix_count})"
        return dict(
            xref="paper", yref="paper",
            x=0.18, y=1.00, showarrow=False,
            align="left",
            font=dict(size=12),
            text=txt,
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(0,0,0,0.2)", borderwidth=1
        )

    # ----------------------------
    # Build figure (t=0)
    # ----------------------------
    t0 = 0
    fig = go.Figure()

    # Major
    Xm0 = X_proj_list[t0][idx_major]
    fig.add_trace(go.Scatter(
        x=Xm0[:, 0], y=Xm0[:, 1],
        mode="markers",
        name="major tasks",
        marker=dict(size=scale_sizes(R2[idx_major, t0]), opacity=0.8, sizemode="diameter", color="blue"),
        text=major_labels,
        customdata=make_custom(idx_major, t0),
        hovertemplate=hovertemplate,
    ))

    # OOD
    if idx_ood is not None and idx_ood.size > 0:
        Xo0 = X_proj_list[t0][idx_ood]
        fig.add_trace(go.Scatter(
            x=Xo0[:, 0], y=Xo0[:, 1],
            mode="markers",
            name="OOD tasks",
            marker=dict(size=scale_sizes(R2[idx_ood, t0]), opacity=0.8, sizemode="diameter", color="green", symbol="square"),
            text=ood_labels,
            customdata=make_custom(idx_ood, t0),
            hovertemplate=hovertemplate,
        ))

    # Minor
    if idx_minor is not None and idx_minor.size > 0:
        Xn0 = X_proj_list[t0][idx_minor]
        fig.add_trace(go.Scatter(
            x=Xn0[:, 0], y=Xn0[:, 1],
            mode="markers",
            name="minor tasks",
            marker=dict(size=scale_sizes(R2[idx_minor, t0]), opacity=0.8, sizemode="diameter", color="red", symbol="diamond"),
            text=minor_labels,
            customdata=make_custom(idx_minor, t0),
            hovertemplate=hovertemplate,
        ))

    # Final refs (static)
    if show_final_refs:
        fig.add_trace(go.Scatter(
            x=F_proj[:, 0], y=F_proj[:, 1],
            mode="markers+text",
            name="final refs",
            marker=dict(size=12, symbol="star", line=dict(width=1)),
            text=final_labels,
            textposition="top center",
            hoverinfo="skip",
        ))

    # ----------------------------
    # Slider steps
    # ----------------------------
    steps = []
    for t in range(T):
        xs, ys, cds, markers, texts = [], [], [], [], []

        # major
        Xm = X_proj_list[t][idx_major]
        xs.append(Xm[:, 0]); ys.append(Xm[:, 1])
        cds.append(make_custom(idx_major, t))
        markers.append(dict(size=scale_sizes(R2[idx_major, t]), opacity=0.8, sizemode="diameter", color="blue"))
        texts.append(major_labels)

        # ood
        if idx_ood is not None and idx_ood.size > 0:
            Xo = X_proj_list[t][idx_ood]
            xs.append(Xo[:, 0]); ys.append(Xo[:, 1])
            cds.append(make_custom(idx_ood, t))
            markers.append(dict(size=scale_sizes(R2[idx_ood, t]), opacity=0.8, sizemode="diameter", color="green", symbol="square"))
            texts.append(ood_labels)

        # minor
        if idx_minor is not None and idx_minor.size > 0:
            Xn = X_proj_list[t][idx_minor]
            xs.append(Xn[:, 0]); ys.append(Xn[:, 1])
            cds.append(make_custom(idx_minor, t))
            markers.append(dict(size=scale_sizes(R2[idx_minor, t]), opacity=0.8, sizemode="diameter", color="red", symbol="diamond"))
            texts.append(minor_labels)

        # final refs unchanged
        if show_final_refs:
            xs.append(F_proj[:, 0]); ys.append(F_proj[:, 1])
            cds.append(None)
            markers.append(None)
            texts.append(final_labels)

        step_layout = {"title": f"Projection at t={t}"}
        if cum_mask is not None and show_mask_annotation:
            step_layout["annotations"] = [make_annotation(t)]

        steps.append(dict(
            method="update",
            args=[
                {"x": xs, "y": ys, "customdata": cds, "marker": markers, "text": texts},
                step_layout
            ],
            label=str(t),
        ))

    base_layout = dict(
        title="Projection with R² (hover & size)",
        xaxis_title="axis 1",
        yaxis_title="axis 2",
        sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix="t = "), pad=dict(t=10))],
        width=800,
        height=650,
    )
    if cum_mask is not None and show_mask_annotation:
        base_layout["annotations"] = [make_annotation(0)]

    fig.update_layout(**base_layout)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig
















def project_with_r2_trajectories_group_colors(
    task_vecs_over_all_time,
    final_task_vecs,
    r2_scores,
    lambdas,
    task_labels=None,
    n_minors=0,
    # --- size mapping (uses R²) ---
    size_min=6,
    size_max=18,
    # --- alpha ramp (early -> late) ---
    alpha_start=0.06,
    alpha_end=0.85,
    # --- group colors (ALL tasks in group share same color) ---
    major_color="#1f77b4",   # blue
    ood_color="#ff7f0e",     # orange
    minor_color="#2ca02c",   # green
    # --- trajectory styling ---
    line_alpha=0.22,
    line_width=1.1,
    marker_line_width=0.0,  # for non-endpoints (usually 0 looks cleaner)
    # --- endpoint emphasis (NO size change; just edge + full opacity) ---
    end_marker_alpha=1.0,
    end_marker_line_color="rgba(0,0,0,0.9)",
    end_marker_line_width=2.4,
    # --- layout ---
    width=900,
    height=700,
):
    """
    Overlapped trajectories on one plot (no slider, no final refs).

    Coloring rule (as requested):
      - First 3 (after flattening): MAJOR color
      - Last n_minors: MINOR color
      - Everything in-between: OOD color

    Endpoint visibility:
      - Does NOT change endpoint size (size reflects R²).
      - Adds a solid outline + full-opacity fill by drawing endpoints as an overlay trace.

    Shapes:
      task_vecs_over_all_time : (..., T, D)
      r2_scores               : (..., T)
      lambdas                 : (..., T, 3)
      final_task_vecs         : (3, D)
    """
    import re
    import numpy as np
    import plotly.graph_objects as go

    # ----------------------------
    # Helpers
    # ----------------------------
    def to_np(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def prod(shape):
        out = 1
        for s in shape:
            out *= int(s)
        return int(out)

    def flatten_leading(X, *, name, tail_ndim):
        X = np.asarray(X)
        if X.ndim < tail_ndim:
            raise ValueError(f"{name} must have ≥ {tail_ndim} dims, got {X.shape}")
        lead_shape = X.shape[: X.ndim - tail_ndim]
        tail_shape = X.shape[X.ndim - tail_ndim :]
        K = prod(lead_shape) if len(lead_shape) > 0 else 1
        return X.reshape((K, *tail_shape)), lead_shape, tail_shape

    def normalize_labels(task_labels, lead_shape, K):
        if task_labels is None:
            if len(lead_shape) == 2:
                a, b = lead_shape
                return [f"task{i}|vocab{j}" for i in range(a) for j in range(b)]
            return [f"item_{k}" for k in range(K)]
        arr = to_np(task_labels)
        if isinstance(arr, np.ndarray):
            flat = arr.reshape(-1)
            if flat.shape[0] == K:
                return [str(x) for x in flat.tolist()]
        task_labels = list(task_labels)
        if len(task_labels) != K:
            raise ValueError(f"task_labels must have length {K}")
        return [str(x) for x in task_labels]

    # robust color -> rgba
    _rgb_pat = re.compile(
        r"^\s*rgba?\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*(?:,\s*([0-9.]+)\s*)?\)\s*$",
        re.IGNORECASE,
    )

    def to_rgba(color, a):
        a = float(max(0.0, min(1.0, a)))
        if not isinstance(color, str):
            return color
        s = color.strip()

        m = _rgb_pat.match(s)
        if m:
            r, g, b = [int(round(float(m.group(i)))) for i in range(1, 4)]
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            return f"rgba({r},{g},{b},{a:.4f})"

        if s.startswith("#"):
            h = s[1:]
            if len(h) == 3:
                h = "".join(c * 2 for c in h)
            if len(h) != 6:
                raise ValueError(f"Unsupported hex color: {color}")
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            return f"rgba({r},{g},{b},{a:.4f})"

        return s  # named colors etc.

    # ----------------------------
    # Convert + flatten inputs
    # ----------------------------
    X_raw = to_np(task_vecs_over_all_time)
    F = to_np(final_task_vecs)
    R2_raw = to_np(r2_scores)
    L_raw = to_np(lambdas)

    if any(v is None for v in (X_raw, F, R2_raw, L_raw)):
        raise ValueError("task_vecs_over_all_time, final_task_vecs, r2_scores, lambdas are required.")

    X, lead_shape, _ = flatten_leading(X_raw, name="task_vecs_over_all_time", tail_ndim=2)  # (K,T,D)
    if X.ndim != 3:
        raise ValueError(f"task_vecs_over_all_time must flatten to (K,T,D). Got {X.shape}")
    K, T, D = X.shape
    if K < 1:
        raise ValueError("K=0: no trajectories to plot.")
    if T < 1:
        raise ValueError("T=0: no timesteps to plot.")

    R2, _, _ = flatten_leading(R2_raw, name="r2_scores", tail_ndim=1)  # (K,T)
    L, _, _ = flatten_leading(L_raw, name="lambdas", tail_ndim=2)      # (K,T,3)
    if R2.shape != (K, T):
        raise ValueError(f"r2_scores shape mismatch: expected {(K,T)}, got {R2.shape}")
    if L.shape != (K, T, 3):
        raise ValueError(f"lambdas shape mismatch: expected {(K,T,3)}, got {L.shape}")

    labels = normalize_labels(task_labels, lead_shape, K)

    if n_minors < 0 or n_minors > K:
        raise ValueError(f"n_minors must be in [0, K]. Got n_minors={n_minors}, K={K}.")

    # ----------------------------
    # Projection plane from final_task_vecs
    # ----------------------------
    F = np.asarray(F)
    if F.ndim != 2 or F.shape[0] != 3 or F.shape[1] != D:
        raise ValueError(f"final_task_vecs must be (3,{D}), got {F.shape}")

    F_center = F.mean(axis=0, keepdims=True)          # (1,D)
    F0 = F - F_center                                 # (3,D)
    _, _, Vt = np.linalg.svd(F0, full_matrices=False)

    basis = Vt[:2].T                                   # (D,2) or (D,1) if D==1
    if basis.shape[1] == 1:
        basis = np.concatenate([basis, np.zeros((D, 1), dtype=basis.dtype)], axis=1)
    if basis.shape != (D, 2):
        raise RuntimeError(f"Internal error: basis has shape {basis.shape}, expected {(D,2)}")

    # IMPORTANT: avoid accidental 4D broadcast; guarantee (K,T,2)
    X_centered = X - F_center.reshape(1, 1, D)         # (K,T,D)
    X_proj = np.tensordot(X_centered, basis, axes=([2], [0]))  # (K,T,2)
    if X_proj.shape != (K, T, 2):
        raise RuntimeError(f"Internal error: X_proj has shape {X_proj.shape}, expected {(K,T,2)}")

    # ----------------------------
    # Group indices (exactly as requested)
    # ----------------------------
    non_minor_end = K - n_minors if n_minors > 0 else K
    major_end = min(3, non_minor_end)

    idx_major = np.arange(0, major_end, dtype=int)
    idx_ood = np.arange(major_end, non_minor_end, dtype=int)
    idx_minor = np.arange(non_minor_end, K, dtype=int) if n_minors > 0 else np.array([], dtype=int)

    group_specs = [
        ("major tasks", idx_major, major_color, "circle"),
        ("OOD tasks",   idx_ood,   ood_color,   "square"),
        ("minor tasks", idx_minor, minor_color, "diamond"),
    ]

    # ----------------------------
    # Size scaling by R² (global)
    # ----------------------------
    r2_min, r2_max = float(R2.min()), float(R2.max())
    if r2_max == r2_min:
        sizes = np.full_like(R2, (size_min + size_max) / 2.0, dtype=float)
    else:
        sizes = size_min + (R2 - r2_min) / (r2_max - r2_min) * (size_max - size_min)

    # ----------------------------
    # Alpha ramp over time
    # ----------------------------
    alpha_start = max(0.0, min(1.0, float(alpha_start)))
    alpha_end = max(0.0, min(1.0, float(alpha_end)))
    alphas = np.array([alpha_end], dtype=float) if T == 1 else np.linspace(alpha_start, alpha_end, T)

    # ----------------------------
    # Build figure
    # ----------------------------
    fig = go.Figure()

    hovertemplate = (
        "task=%{text}"
        "<br>t=%{customdata[0]}"
        "<br>R²=%{customdata[1]:.3f}"
        "<br>λ₁=%{customdata[2]:.3f}"
        "<br>λ₂=%{customdata[3]:.3f}"
        "<br>λ₃=%{customdata[4]:.3f}"
        "<extra></extra>"
    )

    end_hovertemplate = (
        "task=%{text}"
        "<br><b>FINAL</b> (t=%{customdata[0]})"
        "<br>R²=%{customdata[1]:.3f}"
        "<br>λ₁=%{customdata[2]:.3f}"
        "<br>λ₂=%{customdata[3]:.3f}"
        "<br>λ₃=%{customdata[4]:.3f}"
        "<extra></extra>"
    )

    for legend_name, idxs, base_color, symbol in group_specs:
        if idxs.size == 0:
            continue

        # --- Lines: one trace per group (segments separated by None) ---
        x_line = []
        y_line = []
        for k in idxs.tolist():
            x_line.extend(X_proj[k, :, 0].tolist())
            y_line.extend(X_proj[k, :, 1].tolist())
            x_line.append(None)
            y_line.append(None)

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=legend_name,
                showlegend=True,
                hoverinfo="skip",
                line=dict(color=to_rgba(base_color, line_alpha), width=float(line_width)),
            )
        )

        # --- Markers: all points in group, alpha increases with time ---
        Kg = int(idxs.size)

        x_pts = X_proj[idxs, :, 0].reshape(-1)
        y_pts = X_proj[idxs, :, 1].reshape(-1)

        text_pts = np.repeat(np.asarray([labels[k] for k in idxs.tolist()], dtype=object), T)

        t_pts = np.tile(np.arange(T, dtype=int), Kg)
        r2_pts = R2[idxs, :].reshape(-1)
        lam_pts = L[idxs, :, :].reshape(-1, 3)
        custom_pts = np.column_stack([t_pts, r2_pts, lam_pts])  # (N,5)

        size_pts = sizes[idxs, :].reshape(-1).astype(float)

        rgba_by_t = [to_rgba(base_color, a) for a in alphas]     # len T
        color_pts = np.tile(np.asarray(rgba_by_t, dtype=object), Kg)

        fig.add_trace(
            go.Scatter(
                x=x_pts,
                y=y_pts,
                mode="markers",
                name=legend_name + " points",
                showlegend=False,
                marker=dict(
                    size=size_pts,
                    color=color_pts.tolist(),
                    symbol=symbol,
                    line=dict(width=float(marker_line_width)),
                ),
                text=text_pts.tolist(),
                customdata=custom_pts,
                hovertemplate=hovertemplate,
            )
        )

        # --- Endpoints overlay: SAME size as R² (no scaling), but solid edge + full opacity ---
        x_end = X_proj[idxs, -1, 0]
        y_end = X_proj[idxs, -1, 1]
        text_end = np.asarray([labels[k] for k in idxs.tolist()], dtype=object)

        t_end = np.full((Kg,), T - 1, dtype=int)
        r2_end = R2[idxs, -1].reshape(-1)
        lam_end = L[idxs, -1, :].reshape(-1, 3)
        custom_end = np.column_stack([t_end, r2_end, lam_end])  # (Kg,5)

        end_size = sizes[idxs, -1].astype(float)  # <-- NO SIZE MODIFICATION

        fig.add_trace(
            go.Scatter(
                x=x_end,
                y=y_end,
                mode="markers",
                name=legend_name + " endpoints",
                showlegend=False,
                marker=dict(
                    size=end_size,
                    color=to_rgba(base_color, end_marker_alpha),
                    symbol=symbol,
                    line=dict(color=end_marker_line_color, width=float(end_marker_line_width)),
                ),
                text=text_end.tolist(),
                customdata=custom_end,
                hovertemplate=end_hovertemplate,
            )
        )

    fig.update_layout(
        title="Overlapped trajectories (group colors, emphasized endpoints)",
        xaxis_title="axis 1",
        yaxis_title="axis 2",
        width=int(width),
        height=int(height),
        legend=dict(itemsizing="constant"),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig




