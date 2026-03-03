"""
Requires:
    pip install adjustText
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Wedge, Circle
from matplotlib.offsetbox import DrawingArea, AnnotationBbox

try:
    from adjustText import adjust_text
except Exception as e:
    raise ImportError(
        "This version uses adjustText for annotation placement. "
        "Install it with: pip install adjustText"
    ) from e


# ============================================================
# Helpers (self-contained)
# ============================================================

def _to_np(x):
    """Convert torch/np/list-like to numpy (no copy unless needed)."""
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _strip_b_suffix(s: str) -> str:
    return re.sub(r"\|b\d+$", "", str(s))


def _prettify_label(s: str) -> str:
    s = _strip_b_suffix(str(s))
    m = re.fullmatch(r"(item|task)[_\-]?(\d+)", s, flags=re.IGNORECASE)
    if m:
        idx = int(m.group(2))
        return f"Task {idx + 1}"
    s2 = s.replace("_", " ").strip()
    if s2.lower() == s2 and any(c.isalpha() for c in s2):
        s2 = s2.title()
    return s2


def _normalize_major_and_ood_labels(task_labels, *, default_ood="OOD"):
    """
    Always 3 major reference points (from final_task_vecs).
    One OOD trajectory.

    task_labels accepted formats:
      - None -> majors Task1..3, ood "OOD"
      - len==1 -> that is OOD label, majors default
      - len==3 -> majors labels, ood default
      - len>=4 -> first 3 are majors, last is OOD label
      - len==2 -> take last as OOD label, majors default
    """
    maj_default = [f"Task {i+1}" for i in range(3)]
    ood_default = default_ood

    if task_labels is None:
        return maj_default, ood_default

    arr = _to_np(task_labels)
    flat = arr.reshape(-1).tolist() if isinstance(arr, np.ndarray) else list(task_labels)
    flat = [_prettify_label(x) for x in flat if str(x).strip() != ""]

    if len(flat) == 0:
        return maj_default, ood_default
    if len(flat) == 1:
        return maj_default, flat[0]
    if len(flat) == 3:
        return flat, ood_default
    if len(flat) >= 4:
        return flat[:3], flat[-1]
    # len == 2
    return maj_default, flat[-1]


def _rgba(color, a):
    r, g, b, _ = mcolors.to_rgba(color)
    return (r, g, b, float(np.clip(a, 0.0, 1.0)))


def _pow2_time_indices(T, include_last=True):
    idx = []
    p = 1
    while p <= T:
        idx.append(p - 1)
        p *= 2
    if include_last and T > 0:
        idx.append(T - 1)
    idx = np.unique(np.array(idx, dtype=int))
    idx.sort()
    return idx


def _refine_midpoints(indices, levels, T):
    idx = np.unique(np.array(indices, dtype=int))
    idx.sort()
    for _ in range(int(max(0, levels))):
        mids = []
        for i in range(idx.size - 1):
            a = int(idx[i])
            b = int(idx[i + 1])
            if b - a >= 2:
                m = (a + b) // 2
                if m != a and m != b:
                    mids.append(m)
        if not mids:
            break
        idx = np.unique(np.concatenate([idx, np.array(mids, dtype=int)]))
        idx.sort()
    idx = idx[(idx >= 0) & (idx < T)]
    idx = np.unique(idx)
    idx.sort()
    if T > 0:
        idx = np.unique(np.concatenate([idx, np.array([0, T - 1], dtype=int)]))
        idx.sort()
    return idx


def _r2_to_sizes_area(R2, *, size_min, size_max):
    """Map R2 -> marker areas (scatter uses area in pt^2)."""
    R2 = np.asarray(R2, dtype=float)
    r2_min, r2_max = float(np.min(R2)), float(np.max(R2))
    if r2_max == r2_min:
        size_lin = np.full_like(R2, (size_min + size_max) / 2.0, dtype=float)
    else:
        size_lin = size_min + (R2 - r2_min) / (r2_max - r2_min) * (size_max - size_min)
    return (size_lin ** 2).astype(float)


def _project_to_plane(X, F):
    """
    Project (K,T,D) trajectories and (3,D) reference vectors into 2D using SVD on F.
    """
    X = np.asarray(X, dtype=float)
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[0] != 3 or F.shape[1] != X.shape[-1]:
        raise ValueError(f"final_task_vecs must be (3,D) with D={X.shape[-1]}, got {F.shape}")

    F_center = F.mean(axis=0, keepdims=True)
    F0 = F - F_center
    _, _, Vt = np.linalg.svd(F0, full_matrices=False)
    basis = Vt[:2].T  # (D,2)

    X_proj = np.tensordot(X - F_center.reshape(1, 1, -1), basis, axes=([2], [0]))  # (K,T,2)
    F_proj = (F - F_center) @ basis  # (3,2)
    return X_proj, F_proj


def _alpha_piecewise(u, a0, a_mid, a1):
    """Piecewise linear alpha: (0->0.5) a0->a_mid, (0.5->1) a_mid->a1."""
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    out = np.empty_like(u)
    m = u <= 0.5
    out[m] = a0 + (a_mid - a0) * (u[m] / 0.5)
    out[~m] = a_mid + (a1 - a_mid) * ((u[~m] - 0.5) / 0.5)
    return out


# ----------------------------
# Posterior-change selection helpers
# ----------------------------

def _mad(x):
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _posterior_change_score(P, metric="l1", eps=1e-12):
    """
    score[t] measures change from t-1 -> t. score[0] = 0.
    P is (T,3) distribution.
    """
    P = np.asarray(P, dtype=float)
    T = P.shape[0]
    if T <= 1:
        return np.zeros(T, dtype=float)

    metric = str(metric).lower()
    if metric in ("l1", "manhattan"):
        d = np.sum(np.abs(P[1:] - P[:-1]), axis=1)
    elif metric in ("l2", "euclidean"):
        d = np.sqrt(np.sum((P[1:] - P[:-1]) ** 2, axis=1))
    elif metric in ("max", "linf", "inf"):
        d = np.max(np.abs(P[1:] - P[:-1]), axis=1)
    elif metric in ("js", "jsd", "jensen-shannon"):
        # Jensen–Shannon divergence (symmetric; stable for distributions)
        P0 = np.clip(P[:-1], eps, 1.0)
        P1 = np.clip(P[1:],  eps, 1.0)
        M = 0.5 * (P0 + P1)
        d = 0.5 * np.sum(P0 * np.log(P0 / M), axis=1) + 0.5 * np.sum(P1 * np.log(P1 / M), axis=1)
    else:
        raise ValueError(f"Unknown metric={metric}. Use l1/l2/max/js.")

    return np.concatenate([[0.0], d.astype(float)])


def _select_posterior_change_points(
    P,
    *,
    metric="l1",
    threshold=0.1,
    mad_k=4.0,
    min_gap=0,
    max_points=20,
    include_argmax_switch=True,
):
    """
    Select "salient" indices t where posterior noticeably changes vs t-1.

    Strategy:
      1) compute change score[t] = d(P[t], P[t-1])
      2) choose LOCAL MAXIMA above threshold
         - if threshold is None: thr = median(d) + mad_k * MAD(d)
      3) optionally include argmax switches (dominant task changes)
      4) enforce min_gap between chosen indices (keep highest-scoring first)
      5) cap at max_points

    Returns: chosen_idx, score, threshold_used
    """
    P = np.asarray(P, dtype=float)
    T = P.shape[0]
    score = _posterior_change_score(P, metric=metric)

    if T <= 1:
        return np.array([], dtype=int), score, 0.0

    d = score[1:]
    if threshold is None:
        med = float(np.median(d)) if d.size else 0.0
        mad = _mad(d) + 1e-12
        thr = med + float(mad_k) * mad
    else:
        thr = float(threshold)

    cand = []
    for t in range(1, T):
        if score[t] < thr:
            continue
        left = score[t - 1]
        right = score[t + 1] if (t + 1) < T else -np.inf
        if score[t] >= left and score[t] >= right:
            cand.append(t)
    cand = np.asarray(cand, dtype=int)

    if include_argmax_switch and T >= 2:
        dom = np.argmax(P, axis=1)
        sw = np.where(dom[1:] != dom[:-1])[0] + 1
        if sw.size:
            cand = np.unique(np.concatenate([cand, sw.astype(int)]))

    if cand.size == 0:
        return cand, score, thr

    order = cand[np.argsort(-score[cand])]  # highest first
    chosen = []
    min_gap = int(max(0, min_gap))
    max_points = int(max(0, max_points))

    for t in order:
        if all(abs(int(t) - int(c)) >= min_gap for c in chosen):
            chosen.append(int(t))
        if max_points > 0 and len(chosen) >= max_points:
            break

    chosen = np.array(sorted(set(chosen)), dtype=int)
    return chosen, score, thr


# ----------------------------
# NEW: draw a per-point pie chart marker (size in points) at (x,y)
# ----------------------------

def _add_pie_marker(
    ax,
    x,
    y,
    probs,                     # (3,)
    colors,                    # 3 colors
    *,
    radius_pt,
    alpha=1.0,
    start_angle=90.0,
    slice_edgecolor=(1, 1, 1, 0.70),
    slice_lw=0.25,
    outer_edgecolor=(0, 0, 0, 0.25),
    outer_lw=0.60,
    zorder=2.6,
):
    """
    Draw a small pie chart whose size is specified in *points* (not data units),
    placed at the data coordinate (x, y) using an AnnotationBbox.

    This is ideal for "OOD points as pie charts" while keeping marker size tied to R².
    """
    probs = np.asarray(probs, dtype=float).reshape(-1)
    probs = np.clip(probs, 0.0, 1.0)
    s = float(probs.sum())
    if s <= 0.0:
        probs = np.full(3, 1.0 / 3.0, dtype=float)
    else:
        probs = probs / s

    r = float(max(0.1, radius_pt))
    da = DrawingArea(2.0 * r, 2.0 * r, 0.0, 0.0, clip=True)

    theta = float(start_angle)
    for p, c in zip(probs.tolist(), list(colors)):
        if p <= 0.0:
            continue
        dtheta = 360.0 * float(p)
        w = Wedge(
            (r, r), r,
            theta, theta + dtheta,
            facecolor=_rgba(c, alpha),
            edgecolor=slice_edgecolor,
            linewidth=float(slice_lw),
        )
        da.add_artist(w)
        theta += dtheta

    # Outer border
    circ = Circle(
        (r, r), r,
        facecolor=(0, 0, 0, 0),
        edgecolor=outer_edgecolor,
        linewidth=float(outer_lw),
    )
    da.add_artist(circ)

    ab = AnnotationBbox(
        da,
        (float(x), float(y)),
        xycoords="data",
        box_alignment=(0.5, 0.5),
        frameon=False,
        pad=0.0,
        zorder=float(zorder),
    )
    ax.add_artist(ab)
    return ab


# ============================================================
# Main function (OOD-only trajectory + adjustText annotations)
# ============================================================

def project_with_r2_ood_posterior_colors_mpl(
    task_vecs_over_all_time,
    final_task_vecs,
    r2_scores,
    ood_posterior,           # REQUIRED: (T,3) (or (3,T) or (1,T,3))

    task_labels=None,

    # Axis / figure
    ax=None,
    figsize=(9, 7),
    dpi=150,
    clear_ax=False,
    cleanup_previous=True,
    gid_prefix="proj_r2_traj_posterior",

    # R² -> size (ONLY meaning of marker size)
    size_min=1,
    size_max=6,

    # alpha schedule for OOD points
    alpha_start=0.55,
    alpha_mid=0.55,
    alpha_end=0.55,

    # Posterior-to-color mapping:
    corner_colors=("#1f77b4", "#2ca02c", "#d62728"),  # (Major1, Major2, Major3)

    # ------------------------------------------------------------
    # NEW: OOD point style
    #   - "pie"   : draw each OOD point as a pie chart (posterior proportions)
    #   - "blend" : original behavior (single blended RGB dot)
    # ------------------------------------------------------------
    ood_point_style="pie",
    pie_start_angle=90.0,
    pie_slice_edge_color=(1, 1, 1, 0.70),
    pie_slice_edge_width=0.25,
    pie_outer_edge_color=(0, 0, 0, 0.25),
    pie_outer_edge_width=0.60,
    pie_min_radius_pt=1.0,   # ensures very small dots are still visible as pies

    # Trajectory rendering
    include_last=True,
    refine_levels=0,

    # big final direction arrow
    show_direction_arrow=True,
    direction_arrow_alpha=0.85,
    direction_arrow_scale=14.0,
    direction_arrow_frac=0.12,
    direction_arrow_style="-|>",

    # arrows between consecutive shown points
    show_segment_arrows=True,
    segment_arrow_every=1,
    segment_arrow_alpha=0.85,
    segment_arrow_scale=10.0,
    segment_arrow_style="-|>",
    segment_arrow_pad_pt=0.6,
    segment_arrow_min_px=10.0,

    # colored path segments
    color_line_by_posterior=True,
    line_width=1.25,
    line_alpha=0.5,

    # faint curve underneath for readability
    draw_faint_curve=True,
    faint_curve_color=(0, 0, 0, 0.12),
    faint_curve_lw=1.1,

    # reference vectors (major task locations)
    show_final_task_vecs=True,
    show_ref_labels=True,
    final_marker="*",
    final_marker_size=130,
    final_edge_color=(0, 0, 0, 0.95),
    final_edge_width=1.5,

    # cosmetics
    title="OOD trajectory (points as posterior pies)",
    hide_ticks=True,
    despine=True,

    # ============================================================
    # Posterior annotations using adjustText
    # ============================================================
    annotation_names=None,
    annotate_posterior=True,

    # how to pick annotation points: "change" | "fixed" | "both"
    annotation_mode="fixed",

    # fixed-step annotations (1-indexed); only used if mode in {"fixed","both"}
    annotation_steps_1indexed=(1,),
    annotation_include_mid=False,
    annotation_include_end=True,

    # posterior-change selection controls
    annotation_change_metric="l1",        # "l1" | "l2" | "max" | "js"
    annotation_change_threshold=0.2,      # None => auto via median + mad_k*MAD
    annotation_change_mad_k=4.0,
    annotation_change_min_gap=0,
    annotation_change_max=8,
    annotation_change_include_argmax_switch=True,

    # annotation style
    annotation_fontsize=6,
    annotation_box_alpha=0.4,
    annotation_box_pad=0.50,
    annotation_box_lw=0.95,
    annotation_linespacing=0.95,

    annotation_arrow_alpha=0.60,
    annotation_arrow_lw=0.95,

    # leader lines longer + start them from the box edge
    annotation_initial_offset_frac=0.03,
    annotation_min_leader_px=80.0,
    annotation_leader_pad_px=3.0,
    annotation_leader_use_bbox=True,
    annotation_leader_shrinkA_pt=0.0,
    annotation_leader_shrinkB_extra_pt=1.5,
    annotation_leader_max_push_iter=12,

    # adjustText controls
    adjusttext_lim=250,

    # prune overlapping annotation boxes
    annotation_prune_overlaps=True,
    annotation_prune_pad_px=2.0,
    annotation_prune_keep_end=True,
    annotation_prune_keep_mid=True,
    annotation_prune_keep_first=True,
    annotation_prune_use_change_score=True,
):
    """
    OOD-only plotter:
      - task_vecs_over_all_time is the OOD trajectory only:
            (T,D) or (1,T,D) or (K,T,D) (last trajectory treated as OOD)
            also supports (K,T,B,D) by averaging over B.
      - final_task_vecs is (3,D) and defines the 2D plane + reference star locations.
      - ood_posterior is (T,3) and controls colors + annotations.

    NEW:
      - OOD points can be rendered as PIE CHART markers showing posterior proportions:
            ood_point_style="pie"
        (Set ood_point_style="blend" to revert to the old blended RGB dots.)
    """

    clear_ax = bool(clear_ax)
    cleanup_previous = bool(cleanup_previous)
    include_last = bool(include_last)

    show_direction_arrow = bool(show_direction_arrow)
    show_segment_arrows = bool(show_segment_arrows)
    segment_arrow_every = int(max(1, segment_arrow_every))

    color_line_by_posterior = bool(color_line_by_posterior)
    draw_faint_curve = bool(draw_faint_curve)
    show_final_task_vecs = bool(show_final_task_vecs)

    hide_ticks = bool(hide_ticks)
    despine = bool(despine)

    annotate_posterior = bool(annotate_posterior)
    annotation_include_mid = bool(annotation_include_mid)
    annotation_include_end = bool(annotation_include_end)

    annotation_mode = str(annotation_mode).lower().strip()
    ood_point_style = str(ood_point_style).lower().strip()

    def _bbox_padded(bbox, pad_px: float):
        # Matplotlib has bbox.padded in newer versions; keep a fallback.
        try:
            return bbox.padded(float(pad_px))
        except Exception:
            from matplotlib.transforms import Bbox
            return Bbox.from_extents(
                bbox.x0 - pad_px, bbox.y0 - pad_px,
                bbox.x1 + pad_px, bbox.y1 + pad_px,
            )

    def _prune_overlapping_texts(texts, priorities, *, renderer, pad_px=2.0):
        """
        Greedy: keep highest priority labels whose bboxes don't overlap.
        Returns keep_indices (in original order). Removes the others from the axes.
        """
        if len(texts) <= 1:
            return list(range(len(texts)))

        priorities = np.asarray(priorities, dtype=float)
        order = np.argsort(-priorities)  # highest first

        kept_bboxes = []
        keep = []

        for i in order:
            if texts[i] is None:
                continue
            bbox = texts[i].get_window_extent(renderer=renderer)
            bbox = _bbox_padded(bbox, float(pad_px))

            overlaps = any(bbox.overlaps(b) for b in kept_bboxes)
            if not overlaps:
                kept_bboxes.append(bbox)
                keep.append(int(i))

        keep_set = set(keep)
        for j, txt in enumerate(texts):
            if j not in keep_set and txt is not None:
                try:
                    txt.remove()
                except Exception:
                    try:
                        txt.set_visible(False)
                    except Exception:
                        pass

        keep_sorted = sorted(keep)  # preserve original order
        return keep_sorted

    def _gid(suffix: str) -> str:
        return f"{gid_prefix}:{suffix}"

    def _set_gid(artist, suffix: str):
        try:
            artist.set_gid(_gid(suffix))
        except Exception:
            pass
        return artist

    def _cleanup_previous_artists(ax_):
        for art in list(ax_.get_children()):
            gid = getattr(art, "get_gid", lambda: None)()
            if isinstance(gid, str) and gid.startswith(gid_prefix + ":"):
                try:
                    art.remove()
                except Exception:
                    pass
        if getattr(ax_, "legend_", None) is not None:
            try:
                ax_.legend_.remove()
            except Exception:
                pass

    # ----------------------------
    # Convert inputs
    # ----------------------------
    X_raw = _to_np(task_vecs_over_all_time)
    F = _to_np(final_task_vecs)
    R2_raw = _to_np(r2_scores)
    P_raw = _to_np(ood_posterior)

    if any(v is None for v in (X_raw, F, R2_raw, P_raw)):
        raise ValueError("task_vecs_over_all_time, final_task_vecs, r2_scores, ood_posterior are required.")

    Xnp = np.asarray(X_raw)
    if Xnp.ndim == 2:  # (T,D) -> (1,T,D)
        Xnp = Xnp[None, :, :]
    elif Xnp.ndim == 4:
        # assume (K,T,B,D) -> mean over B
        Xnp = Xnp.mean(axis=2)
    elif Xnp.ndim != 3:
        raise ValueError(f"task_vecs_over_all_time must be (T,D) or (K,T,D) or (K,T,B,D). Got {Xnp.shape}")

    K, T, D = Xnp.shape
    if T <= 0:
        raise ValueError("T must be >= 1.")

    # R2 normalize to (K,T)
    R2np = np.asarray(R2_raw, dtype=float)
    if R2np.ndim == 1:
        if R2np.shape[0] != T:
            raise ValueError(f"r2_scores length mismatch: expected {T}, got {R2np.shape[0]}")
        R2np = np.broadcast_to(R2np[None, :], (K, T))
    elif R2np.ndim == 2:
        if R2np.shape == (K, T):
            pass
        elif R2np.shape == (T, K):
            R2np = R2np.T
        elif K == 1 and R2np.shape == (T, 1):
            R2np = R2np.T
        elif K == 1 and R2np.shape == (1, T):
            pass
        else:
            raise ValueError(f"r2_scores shape mismatch: expected {(K, T)} (or (T,) ), got {R2np.shape}")
    elif R2np.ndim == 3:
        # supported if passed (K,T,B) or (K,B,T); average B
        if R2np.shape[0] == K and R2np.shape[1] == T:
            R2np = R2np.mean(axis=2)
        elif R2np.shape[0] == K and R2np.shape[2] == T:
            R2np = R2np.mean(axis=1)
        else:
            raise ValueError(f"Unsupported r2_scores shape: {R2np.shape}")
    else:
        raise ValueError(f"Unsupported r2_scores shape: {R2np.shape}")

    # Posterior normalize to (T,3)
    P = np.asarray(P_raw, dtype=float)
    if P.ndim == 3 and P.shape[0] == 1 and P.shape[1:] == (T, 3):
        P = P[0]
    elif P.ndim == 2 and P.shape == (T, 3):
        pass
    elif P.ndim == 2 and P.shape == (3, T):
        P = P.T
    else:
        raise ValueError(f"ood_posterior must be (T,3) (or (3,T) or (1,T,3)). Got {P.shape}")

    P = np.clip(P, 0.0, 1.0)
    P = P / (P.sum(axis=-1, keepdims=True) + 1e-12)

    maj_names, ood_name = _normalize_major_and_ood_labels(task_labels, default_ood="OOD")

    # ----------------------------
    # Projection using final_task_vecs only
    # ----------------------------
    X_proj, F_proj = _project_to_plane(Xnp, np.asarray(F, dtype=float))

    # ----------------------------
    # Times to display and annotate
    # ----------------------------
    t_base = _pow2_time_indices(T, include_last=include_last)
    t_plot = _refine_midpoints(t_base, levels=int(refine_levels), T=T)

    t_ann_list = []

    # fixed steps
    if annotation_mode in ("fixed", "both"):
        for s in list(annotation_steps_1indexed or ()):
            s = int(s)
            if s >= 1:
                t = s - 1
                if 0 <= t < T:
                    t_ann_list.append(t)

    # change-based points
    if annotation_mode in ("change", "both"):
        t_change, _, _ = _select_posterior_change_points(
            P,
            metric=annotation_change_metric,
            threshold=annotation_change_threshold,
            mad_k=annotation_change_mad_k,
            min_gap=annotation_change_min_gap,
            max_points=annotation_change_max,
            include_argmax_switch=annotation_change_include_argmax_switch,
        )
        t_ann_list.extend(t_change.tolist())

    if annotation_include_mid:
        t_ann_list.append(int((T - 1) // 2))
    if annotation_include_end:
        t_ann_list.append(int(T - 1))

    t_ann = np.unique(np.array(t_ann_list, dtype=int))

    # Ensure we draw + size points for: plotting indices + annotation indices + endpoints
    t_show = np.unique(np.concatenate([t_plot, t_ann, np.array([0, T - 1], dtype=int)]))
    t_show.sort()

    # ----------------------------
    # Sizes (ONLY meaning of size)
    # ----------------------------
    sizes_area = _r2_to_sizes_area(R2np, size_min=size_min, size_max=size_max)

    # ----------------------------
    # Posterior -> RGB blend (still used for line segments/arrows/labels)
    # ----------------------------
    if len(corner_colors) != 3:
        raise ValueError("corner_colors must have length 3.")
    corner_rgbs = np.stack([mcolors.to_rgb(c) for c in corner_colors], axis=0)  # (3,3)

    rgb_show = np.clip(P[t_show] @ corner_rgbs, 0.0, 1.0)  # (M,3)
    seg_rgb = 0.5 * (rgb_show[:-1] + rgb_show[1:]) if rgb_show.shape[0] >= 2 else rgb_show

    u = (t_show.astype(float) / max(1.0, float(T - 1)))
    alpha_pts = _alpha_piecewise(u, float(alpha_start), float(alpha_mid), float(alpha_end))
    alpha_pts = np.clip(alpha_pts, 0.0, 1.0)

    # ----------------------------
    # Figure / Axes
    # ----------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
        if clear_ax:
            ax.clear()
        elif cleanup_previous:
            _cleanup_previous_artists(ax)

    # ----------------------------
    # OOD trajectory (use last trajectory if K>1)
    # ----------------------------
    ood_idx = K - 1
    Pood = X_proj[ood_idx]          # (T,2)
    P_disp = Pood[t_show]           # (M,2)

    # Ensure data limits are correct even if lines are disabled (important for pie markers)
    try:
        ax.update_datalim(Pood)
        if show_final_task_vecs:
            ax.update_datalim(F_proj)
        ax.autoscale_view()
    except Exception:
        pass

    if draw_faint_curve:
        ln = ax.plot(
            Pood[:, 0], Pood[:, 1],
            color=faint_curve_color,
            lw=float(faint_curve_lw),
            linestyle="-",
            zorder=1.05,
        )[0]
        _set_gid(ln, "ood_faint_curve")

    if color_line_by_posterior and P_disp.shape[0] >= 2:
        segs = np.stack([P_disp[:-1], P_disp[1:]], axis=1)  # (M-1,2,2)
        seg_rgba = np.column_stack([
            np.clip(seg_rgb[:, 0], 0.0, 1.0),
            np.clip(seg_rgb[:, 1], 0.0, 1.0),
            np.clip(seg_rgb[:, 2], 0.0, 1.0),
            np.full(seg_rgb.shape[0], float(line_alpha)),
        ])
        lc = LineCollection(segs, colors=seg_rgba, linewidths=float(line_width), zorder=2.0)
        ax.add_collection(lc)
        _set_gid(lc, "ood_colored_segments")

    # ------------------------------------------------------------
    # OOD points:
    #   - "pie": posterior pie markers (NEW)
    #   - "blend": original scatter with blended colors
    # ------------------------------------------------------------
    if ood_point_style in ("pie", "pies"):
        # Convert scatter-area (pt^2) -> circle radius (pt) so visual size is comparable.
        r_pts_show = np.sqrt(np.clip(sizes_area[ood_idx, t_show], 1e-9, None) / np.pi)
        r_pts_show = np.maximum(r_pts_show, float(max(0.0, pie_min_radius_pt)))

        pie_artists = []
        for j, t in enumerate(t_show.tolist()):
            xj = float(P_disp[j, 0])
            yj = float(P_disp[j, 1])
            pj = P[int(t)]  # (3,)
            aj = float(alpha_pts[j])

            ab = _add_pie_marker(
                ax,
                xj,
                yj,
                pj,
                corner_colors,
                radius_pt=float(r_pts_show[j]),
                alpha=aj,
                start_angle=float(pie_start_angle),
                slice_edgecolor=pie_slice_edge_color,
                slice_lw=float(pie_slice_edge_width),
                outer_edgecolor=pie_outer_edge_color,
                outer_lw=float(pie_outer_edge_width),
                zorder=2.6,
            )
            _set_gid(ab, f"ood_pie_point:{int(t)}")
            pie_artists.append(ab)
        _set_gid(ax, "ood_pies_container")  # harmless tag

    elif ood_point_style in ("blend", "mix", "rgb"):
        fc = np.column_stack([rgb_show, alpha_pts])
        pts = ax.scatter(
            P_disp[:, 0], P_disp[:, 1],
            s=sizes_area[ood_idx, t_show],
            marker="o",
            facecolors=fc,
            edgecolors=(0, 0, 0, 0.25),
            linewidths=0.6,
            zorder=2.6,
        )
        _set_gid(pts, "ood_points")
    else:
        raise ValueError(f"Unknown ood_point_style={ood_point_style!r}. Use 'pie' or 'blend'.")

    # small arrows between consecutive shown points (order)
    if show_segment_arrows and P_disp.shape[0] >= 2:
        r_pts = np.sqrt(np.clip(sizes_area[ood_idx, t_show], 1e-9, None) / np.pi)  # radii in points
        if ood_point_style in ("pie", "pies"):
            r_pts = np.maximum(r_pts, float(max(0.0, pie_min_radius_pt)))

        for j in range(0, P_disp.shape[0] - 1, segment_arrow_every):
            p0 = P_disp[j]
            p1 = P_disp[j + 1]
            if np.allclose(p0, p1):
                continue

            if segment_arrow_min_px is not None and float(segment_arrow_min_px) > 0:
                p0d = ax.transData.transform(p0)
                p1d = ax.transData.transform(p1)
                if float(np.hypot(*(p1d - p0d))) < float(segment_arrow_min_px):
                    continue

            c = seg_rgb[j] if (seg_rgb.shape[0] >= j + 1) else rgb_show[min(j, rgb_show.shape[0] - 1)]
            lw_here = 0.0 if color_line_by_posterior else float(line_width)

            arr = FancyArrowPatch(
                posA=(float(p0[0]), float(p0[1])),
                posB=(float(p1[0]), float(p1[1])),
                arrowstyle=segment_arrow_style,
                mutation_scale=float(segment_arrow_scale),
                linewidth=float(lw_here),
                color=(float(c[0]), float(c[1]), float(c[2]),
                       float(np.clip(segment_arrow_alpha, 0.0, 1.0))),
                shrinkA=float(r_pts[j] + float(segment_arrow_pad_pt)),
                shrinkB=float(r_pts[j + 1] + float(segment_arrow_pad_pt)),
                zorder=3.1,
            )
            ax.add_patch(arr)
            _set_gid(arr, f"ood_seg_arrow:{j}")

    # big final arrow (end direction)
    if show_direction_arrow and P_disp.shape[0] >= 2:
        rgb_end = rgb_show[-1]
        n = P_disp.shape[0]
        frac = float(np.clip(direction_arrow_frac, 0.02, 0.8))
        i0 = int(max(0, np.floor((1.0 - frac) * (n - 1))))
        i1 = n - 1
        p0 = P_disp[i0]
        p1 = P_disp[i1]
        if not np.allclose(p0, p1):
            arr = FancyArrowPatch(
                posA=(p0[0], p0[1]),
                posB=(p1[0], p1[1]),
                arrowstyle=direction_arrow_style,
                mutation_scale=float(direction_arrow_scale),
                linewidth=0.0,
                color=(float(rgb_end[0]), float(rgb_end[1]), float(rgb_end[2]),
                       float(np.clip(direction_arrow_alpha, 0.0, 1.0))),
                zorder=3.6,
            )
            ax.add_patch(arr)
            _set_gid(arr, "ood_arrow")

    # ----------------------------
    # Reference major task locations (stars) + labels
    # ----------------------------
    if show_final_task_vecs:
        dx = 0.018 * (np.ptp(F_proj[:, 0]) + 1e-9)
        dy = 0.018 * (np.ptp(F_proj[:, 1]) + 1e-9)

        for i in range(3):
            rgb = corner_rgbs[i]
            ref = ax.scatter(
                [F_proj[i, 0]], [F_proj[i, 1]],
                s=float(final_marker_size),
                marker=final_marker,
                facecolors=[(rgb[0], rgb[1], rgb[2], 1.0)],
                edgecolors=[final_edge_color],
                linewidths=float(final_edge_width),
                zorder=4.2,
            )
            _set_gid(ref, f"ref_vec:{i}")

            if show_ref_labels:
                txt = ax.annotate(
                    maj_names[i] if i < len(maj_names) else f"Task {i + 1}",
                    xy=(F_proj[i, 0], F_proj[i, 1]),
                    xytext=(F_proj[i, 0] + dx, F_proj[i, 1] + dy),
                    fontsize=11,
                    ha="left",
                    va="bottom",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.75),
                    zorder=5,
                )
                _set_gid(txt, f"ref_label:{i}")

    # ----------------------------
    # Legend
    # ----------------------------
    handles = []
    for i in range(3):
        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="none",
                markersize=9,
                markerfacecolor=_rgba(corner_colors[i], 0.95),
                markeredgecolor=(0, 0, 0, 0.45),
                label=f"Posterior = 1.0 on {maj_names[i]}"
            )
        )
    handles.append(
        Line2D([0], [0], color=(0, 0, 0, 0.35), lw=1.8, linestyle="-",
               label=f"OOD trajectory ({ood_name})")
    )
    ax.legend(handles=handles, frameon=False, loc="best")

    # ----------------------------
    # Cosmetics
    # ----------------------------
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.06)
    ax.grid(False)

    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        ax.set_xlabel("axis 1")
        ax.set_ylabel("axis 2")

    if despine:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_alpha(0.45)
        ax.spines["bottom"].set_alpha(0.45)

    # ============================================================
    # Posterior annotations using adjustText
    # ============================================================
    if annotate_posterior and t_ann.size > 0:
        # Ensure renderer is ready (important for adjustText bboxes)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # Deterministic initial offsets so texts don't start identical
        span_x = float(np.ptp(Pood[:, 0]) + 1e-9)
        span_y = float(np.ptp(Pood[:, 1]) + 1e-9)

        frac = float(annotation_initial_offset_frac)
        dx0 = frac * span_x
        dy0 = frac * span_y

        offset_pattern = [
            (+dx0, +dy0),
            (+dx0, -dy0),
            (-dx0, +dy0),
            (-dx0, -dy0),
            (+0.0, +dy0),
            (+0.0, -dy0),
            (+dx0, +0.0),
            (-dx0, +0.0),
            (+1.2 * dx0, +0.6 * dy0),
            (-1.2 * dx0, -0.6 * dy0),
        ]

        def _fmt_step_label(t):
            if t == (T - 1):
                return f"End (Step {t + 1})"
            if t == int((T - 1) // 2):
                return f"Mid (Step {t + 1})"
            return f"Step {t + 1}"

        texts = []
        anchors_xy = []
        label_rgbs = []
        ann_ts = t_ann.tolist()

        for idx, t in enumerate(ann_ts):
            p = P[int(t)]
            rgb = np.clip(p @ corner_rgbs, 0.0, 1.0)

            ann_n = annotation_names if annotation_names is not None else maj_names
            text = (
                f"{_fmt_step_label(int(t))}\n"
                f"{ann_n[0]}: {p[0]:.2f}\n"
                f"{ann_n[1]}: {p[1]:.2f}\n"
                f"{ann_n[2]}: {p[2]:.2f}"
            )

            x = float(Pood[t, 0])
            y = float(Pood[t, 1])
            ox, oy = offset_pattern[idx % len(offset_pattern)]

            txt = ax.text(
                x + ox, y + oy,
                text,
                fontsize=int(annotation_fontsize),
                linespacing=float(annotation_linespacing),
                ha="left",
                va="bottom",
                color="black",
                bbox=dict(
                    boxstyle=f"round,pad={float(annotation_box_pad)}",
                    fc=(1, 1, 1, float(annotation_box_alpha)),
                    ec=(float(rgb[0]), float(rgb[1]), float(rgb[2]), 0.70),
                    lw=float(annotation_box_lw),
                ),
                zorder=6.0,
            )
            _set_gid(txt, f"posterior_text:{t}")

            texts.append(txt)
            anchors_xy.append((x, y))
            label_rgbs.append(rgb)

        # Objects/points to repel from: all shown OOD points + optional major stars
        repel_x = np.asarray(Pood[t_show, 0], dtype=float)
        repel_y = np.asarray(Pood[t_show, 1], dtype=float)
        if show_final_task_vecs:
            repel_x = np.concatenate([repel_x, np.asarray(F_proj[:, 0], dtype=float)])
            repel_y = np.concatenate([repel_y, np.asarray(F_proj[:, 1], dtype=float)])

        # Call adjustText (compatibility: some versions differ in kwargs)
        try:
            adjust_text(
                texts,
                x=repel_x, y=repel_y,
                ax=ax,
                lim=int(adjusttext_lim),
                ensure_inside_axes=True,
            )
        except TypeError:
            adjust_text(
                texts,
                x=repel_x, y=repel_y,
                ax=ax,
                lim=int(adjusttext_lim),
            )

        # ------------------------------------------------------------
        # Enforce a minimum leader length (in screen pixels)
        # by pushing the whole box away from its anchor if needed.
        # ------------------------------------------------------------
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_win = ax.get_window_extent(renderer=renderer)

        def _move_text_by_disp(txt_obj, dx_px, dy_px):
            """Move a data-coordinate text by a display-space delta (pixels)."""
            x_d, y_d = txt_obj.get_position()
            x_px, y_px = ax.transData.transform((x_d, y_d))
            new_d = ax.transData.inverted().transform((x_px + dx_px, y_px + dy_px))
            txt_obj.set_position((float(new_d[0]), float(new_d[1])))

        def _clamp_text_inside_axes(txt_obj, pad_px=2.0):
            """Keep text bbox inside axes window in display coords."""
            bbox = txt_obj.get_window_extent(renderer=renderer)
            dx = 0.0
            dy = 0.0
            if bbox.x0 < ax_win.x0 + pad_px:
                dx += (ax_win.x0 + pad_px) - bbox.x0
            if bbox.x1 > ax_win.x1 - pad_px:
                dx -= bbox.x1 - (ax_win.x1 - pad_px)
            if bbox.y0 < ax_win.y0 + pad_px:
                dy += (ax_win.y0 + pad_px) - bbox.y0
            if bbox.y1 > ax_win.y1 - pad_px:
                dy -= bbox.y1 - (ax_win.y1 - pad_px)
            if dx != 0.0 or dy != 0.0:
                _move_text_by_disp(txt_obj, dx, dy)

        def _closest_point_on_bbox(bbox, x_px, y_px):
            """Closest point on (or in) bbox to a point (x_px,y_px) in display coords."""
            cx = float(np.clip(x_px, bbox.x0, bbox.x1))
            cy = float(np.clip(y_px, bbox.y0, bbox.y1))
            return np.array([cx, cy], dtype=float)

        def _ensure_min_leader_len(txt_obj, anchor_xy_data, min_len_px, max_iter=10):
            anchor_px = np.array(ax.transData.transform(anchor_xy_data), dtype=float)
            for _ in range(int(max_iter)):
                bbox = txt_obj.get_window_extent(renderer=renderer)
                near_px = _closest_point_on_bbox(bbox, anchor_px[0], anchor_px[1])
                dist = float(np.hypot(*(near_px - anchor_px)))
                if dist >= float(min_len_px):
                    break

                # Direction: move box away from anchor along anchor->bbox_center
                center_px = np.array([(bbox.x0 + bbox.x1) * 0.5, (bbox.y0 + bbox.y1) * 0.5], dtype=float)
                v = center_px - anchor_px
                n = float(np.hypot(v[0], v[1]))
                if n < 1e-6:
                    v = np.array([1.0, 1.0], dtype=float)
                    n = float(np.hypot(v[0], v[1]))

                extra = float(min_len_px) - dist
                dx = (v[0] / n) * extra
                dy = (v[1] / n) * extra
                _move_text_by_disp(txt_obj, dx, dy)
                _clamp_text_inside_axes(txt_obj, pad_px=2.0)

        if annotation_min_leader_px is not None and float(annotation_min_leader_px) > 0:
            for txt, (ax0, ay0) in zip(texts, anchors_xy):
                _ensure_min_leader_len(
                    txt,
                    (float(ax0), float(ay0)),
                    min_len_px=float(annotation_min_leader_px),
                    max_iter=int(max(1, annotation_leader_max_push_iter)),
                )

        # ------------------------------------------------------------
        # Prune overlapping annotation boxes (drop some labels)
        # ------------------------------------------------------------
        if annotation_prune_overlaps:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

            # priority = keep End/Mid/Step1 + (optionally) posterior-change score
            if annotation_prune_use_change_score:
                change_score = _posterior_change_score(P, metric=annotation_change_metric)
            else:
                change_score = np.zeros(T, dtype=float)

            priorities = []
            for t in ann_ts:
                pr = float(change_score[int(t)])

                if annotation_prune_keep_end and int(t) == (T - 1):
                    pr += 1e6
                if annotation_prune_keep_mid and int(t) == int((T - 1) // 2):
                    pr += 5e5
                if annotation_prune_keep_first and int(t) == 0:
                    pr += 3e5

                priorities.append(pr)

            keep_idx = _prune_overlapping_texts(
                texts,
                priorities,
                renderer=renderer,
                pad_px=float(annotation_prune_pad_px),
            )

            texts      = [texts[i] for i in keep_idx]
            anchors_xy = [anchors_xy[i] for i in keep_idx]
            label_rgbs = [label_rgbs[i] for i in keep_idx]
            ann_ts     = [ann_ts[i] for i in keep_idx]

        # ------------------------------------------------------------
        # Draw per-label dashed leader lines (posterior-colored)
        # Start from bbox edge (optional), not from text anchor.
        # ------------------------------------------------------------
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        for txt, (ax0, ay0), rgb, t in zip(texts, anchors_xy, label_rgbs, ann_ts):
            anchor_data = (float(ax0), float(ay0))
            anchor_px = np.array(ax.transData.transform(anchor_data), dtype=float)

            # Approx marker radius in points, so leader line doesn't start inside the dot/pie
            s_anchor = float(sizes_area[ood_idx, int(t)])
            r_anchor_pt = float(np.sqrt(max(s_anchor, 1e-9) / np.pi))
            if ood_point_style in ("pie", "pies"):
                r_anchor_pt = float(max(r_anchor_pt, float(max(0.0, pie_min_radius_pt))))

            if annotation_leader_use_bbox:
                bbox = txt.get_window_extent(renderer=renderer)
                start_px = _closest_point_on_bbox(bbox, anchor_px[0], anchor_px[1])

                # add a small gap from the box edge, moving toward the anchor
                v = anchor_px - start_px
                d = float(np.hypot(v[0], v[1]))
                if d > 1e-6 and annotation_leader_pad_px is not None and float(annotation_leader_pad_px) > 0:
                    start_px = start_px + (v / d) * float(annotation_leader_pad_px)

                start_data = ax.transData.inverted().transform(start_px)
                posA = (float(start_data[0]), float(start_data[1]))
                shrinkA = float(annotation_leader_shrinkA_pt)
            else:
                # start at text anchor
                tx, ty = txt.get_position()
                posA = (float(tx), float(ty))
                shrinkA = float(annotation_leader_shrinkA_pt)

            leader = FancyArrowPatch(
                posA=posA,
                posB=anchor_data,
                arrowstyle="-",  # line only
                mutation_scale=1.0,
                linewidth=float(annotation_arrow_lw),
                linestyle=(0, (3, 2)),
                color=(float(rgb[0]), float(rgb[1]), float(rgb[2]), float(annotation_arrow_alpha)),
                shrinkA=shrinkA,
                shrinkB=float(r_anchor_pt + float(annotation_leader_shrinkB_extra_pt)),
                zorder=5.95,
                clip_on=False,
            )
            ax.add_patch(leader)
            _set_gid(leader, f"posterior_leader:{t}")

    return fig, ax
