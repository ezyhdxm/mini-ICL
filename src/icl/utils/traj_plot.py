import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

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


def _flatten_leading(X, *, name, tail_ndim):
    """Flatten all but the last `tail_ndim` dims into a single leading dim."""
    X = np.asarray(X)
    if X.ndim < tail_ndim:
        raise ValueError(f"{name} must have ≥ {tail_ndim} dims, got {X.shape}")

    lead_shape = X.shape[:-tail_ndim]
    tail_shape = X.shape[-tail_ndim:]
    K_ = int(np.prod(lead_shape)) if lead_shape else 1
    return X.reshape((K_, *tail_shape)), lead_shape, tail_shape


def _strip_b_suffix(s: str) -> str:
    return re.sub(r"\|b\d+$", "", str(s))


def _prettify_label(s: str) -> str:
    """Human-friendly labels. Keeps original capitalization if user provided it."""
    s = _strip_b_suffix(str(s))
    m = re.fullmatch(r"(item|task)[_\-]?(\d+)", s, flags=re.IGNORECASE)
    if m:
        idx = int(m.group(2))
        return f"Task {idx + 1}"

    s2 = s.replace("_", " ").strip()
    # Title-case only if it looks like a lowercase identifier
    if s2.lower() == s2 and any(c.isalpha() for c in s2):
        s2 = s2.title()
    return s2


def _normalize_labels(task_labels, K_):
    if task_labels is None:
        return [f"Task {k + 1}" for k in range(K_)]

    arr = _to_np(task_labels)
    if isinstance(arr, np.ndarray):
        flat = arr.reshape(-1)
        if flat.shape[0] == K_:
            return [_prettify_label(x) for x in flat.tolist()]

    task_labels_list = list(task_labels)
    if len(task_labels_list) != K_:
        raise ValueError(f"task_labels must have length {K_}")
    return [_prettify_label(x) for x in task_labels_list]


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


def _jitter_color(base_color, rng, hue_j=0.02, sat_j=0.06, val_j=0.06):
    r, g, b = mcolors.to_rgb(base_color)
    hsv = mcolors.rgb_to_hsv(np.array([[r, g, b]], dtype=float))[0]
    h, s, v = hsv
    h = (h + rng.uniform(-hue_j, hue_j)) % 1.0
    s = float(np.clip(s + rng.uniform(-sat_j, sat_j), 0.0, 1.0))
    v = float(np.clip(v + rng.uniform(-val_j, val_j), 0.0, 1.0))
    rgb = mcolors.hsv_to_rgb(np.array([[h, s, v]], dtype=float))[0]
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]))


def _r2_to_sizes_area(R2, *, size_min, size_max):
    """Map R2 to marker areas (scatter uses area, not radius)."""
    R2 = np.asarray(R2, dtype=float)
    r2_min, r2_max = float(np.min(R2)), float(np.max(R2))

    if r2_max == r2_min:
        size_lin = np.full_like(R2, (size_min + size_max) / 2.0, dtype=float)
    else:
        size_lin = size_min + (R2 - r2_min) / (r2_max - r2_min) * (size_max - size_min)

    return (size_lin ** 2).astype(float)


def _project_to_plane(X, F):
    """Project (K,T,D) trajectories and (3,D) reference vectors into 2D."""
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


def _r2_value_to_area(r2v, *, r2_min, r2_max, size_min, size_max):
    """Map a single R² value to marker area, consistent with _r2_to_sizes_area."""
    r2v = float(r2v)
    r2_min = float(r2_min)
    r2_max = float(r2_max)

    if r2_max == r2_min:
        size_lin = 0.5 * (float(size_min) + float(size_max))
    else:
        size_lin = float(size_min) + (r2v - r2_min) / (r2_max - r2_min) * (float(size_max) - float(size_min))

    return float(size_lin ** 2)  # scatter uses area


# ============================================================
# Main function (FULL, self-contained)
# ============================================================

def project_with_r2_trajectories_group_colors_mpl(
    task_vecs_over_all_time,
    final_task_vecs,
    r2_scores,
    task_labels=None,

    # --- grouping ---
    n_ood=None,
    n_minor=0,
    b_ood=1,
    b_minor=1,
    b_major=1,
    use_mean=True,

    # --- axis reuse behavior ---
    ax=None,
    figsize=(9, 7),
    dpi=150,
    clear_ax=False,
    cleanup_previous=True,
    gid_prefix="proj_r2_traj_clear",

    # --- R² -> marker size ---
    size_min=4,
    size_max=12,

    # --- POINTS: show more ---
    max_points=5,
    mid_size_factor=0.55,

    # --- points alpha: early a bit darker, late lighter (end marker emphasized separately) ---
    point_alpha_start=0.34,
    point_alpha_end=0.25,
    point_alpha_power=1.6,

    # --- highlight pow2 anchors lightly ---
    show_pow2_anchors=True,
    anchor_alpha_floor=0.25,
    anchor_skip_ends=True,

    # --- NEW: time indices to always include AND annotate ---
    # Accepts int or iterable of ints. Negative indices allowed (python-style): -1 = last.
    must_include_times=None,

    # --- jitter to reduce overlap ---
    jitter_points=True,
    jitter_scale=0.0022,
    jitter_seed=0,

    # --- group base colors ---
    major_colors=("#1f77b4", "#9467bd", "#17becf"),
    ood_base_color="#ff7f0e",
    minor_base_color="#2ca02c",

    # --- line styles ---
    major_linestyle="-",
    minor_linestyle=":",
    ood_vary_linestyle=True,
    ood_linestyle_cycle=None,

    # --- LINES: darker at the beginning, lighter towards the end ---
    use_gradient_line=True,
    line_width=1.7,
    line_alpha_end=0.30,           # start alpha (darker)
    line_alpha_start_factor=0.60,  # end alpha = line_alpha_end * factor (lighter)
    line_alpha_power=1.45,

    # --- end emphasis ---
    include_last=True,
    end_marker_alpha=0.98,
    end_marker_edge_color=(0, 0, 0, 0.35),
    end_marker_edge_width=1.0,

    # --- OOD style variety ---
    ood_style_seed=0,
    ood_hue_jitter=0.10,
    ood_sat_jitter=0.20,
    ood_val_jitter=0.15,

    # --- reference vectors ---
    show_final_task_vecs=True,
    final_marker="*",
    final_marker_size=80,
    final_edge_color=(0, 0, 0, 0.95),
    final_edge_width=1.5,

    # --- cosmetics ---
    title="Projected trajectories",
    hide_ticks=True,
    despine=True,

    # ============================================================
    # Annotations (boxes + leader lines)
    # ============================================================
    annotate=True,
    annotate_k=None,                  # None => annotate ALL OOD trajectories

    # NEW: allow start/end annotation toggles
    annotate_start=True,
    annotate_end=True,

    # IMPORTANT CHANGE:
    # Default no longer includes the 20th point (index 19).
    # If you want the 20th annotated, include 19 in must_include_times.
    annotate_times=(0, -1),           # Start + End only by default (can still add others)

    annotation_fontsize=9,
    annotation_box_alpha=0.45,
    annotation_box_pad=0.20,
    annotation_box_lw=0.90,
    annotation_line_alpha=0.60,
    annotation_line_width=0.95,
    annotation_pad_px=7,

    # leader band (display px)
    annotation_max_leader_px=42,
    annotation_min_gap_px=6,          # extra gap beyond marker radius
    annotation_prefer_length_scale=0.40,  # L0 = Lmin + scale * half_diag(label)

    # keep boxes separated (display px)
    annotation_sep_px=6,

    # point avoidance
    annotation_point_clear_px=None,   # None => derived from fontsize
    annotation_point_collision_cap=0, # max point-circle intersections allowed per label candidate (None => no cap)
    annotation_point_penalty=350.0,   # per point-circle intersection
    annotation_density_weight=14.0,   # weight on density field (dimensionless)

    # density field grid for "empty space" preference
    annotation_density_grid=(220, 220),
    annotation_density_blur_k=9,      # odd integer kernel size (cells)

    # solver controls
    annotation_candidate_angles_deg=(0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180),
    annotation_candidate_radii_n=8,    # radii samples between [Lmin, Lmax]
    annotation_beam_width=48,
    annotation_beam_expand=28,        # candidates tried per label per beam state
    annotation_max_tries=4,           # if infeasible, grow Lmax gradually and retry
    annotation_max_leader_growth=1.55,
    annotation_polish_passes=2,       # coordinate-descent on chosen candidates
    annotation_debug=False,

    # --- compatibility alias ---
    b_maj=None,  # alias for b_major
    step=None,
    show_legend=True,
    
    # --- NEW: rigid alignment in 2D after projection ---
    align_major_rigid=True,
    align_major_target=(0.0, 0.0),     # where to pin major-1 star
    align_major_use_rotation=True,     # also fix orientation using (major1->major2)
    align_major_ref_dir=(1.0, 0.0),    # desired direction in 2D for (major1->major2)
    align_major_rotate_about="major1", # "major1" or "origin"
):
    """
    Key behavior:
      - must_include_times: those time indices are included in the polyline AND annotated.
      - annotate_start / annotate_end: can disable Start/End annotation without touching annotate_times.
        (must_include_times still forces annotation if it includes 0 or T-1.)
      - The "20th point" (index 19) is NOT annotated unless 19 is in must_include_times.
      - If any annotation boxes overlap, drop boxes iteratively until no overlaps remain.
    """

    # ---- normalize bool-ish flags early ----
    clear_ax = bool(clear_ax)
    cleanup_previous = bool(cleanup_previous)
    use_mean = bool(use_mean)
    include_last = bool(include_last)
    show_pow2_anchors = bool(show_pow2_anchors)
    anchor_skip_ends = bool(anchor_skip_ends)
    jitter_points = bool(jitter_points)
    show_final_task_vecs = bool(show_final_task_vecs)
    hide_ticks = bool(hide_ticks)
    despine = bool(despine)
    annotate = bool(annotate)
    annotate_start = bool(annotate_start)
    annotate_end = bool(annotate_end)
    annotation_debug = bool(annotation_debug)

    # --- compatibility: accept b_maj as alias for b_major ---
    if b_maj is not None:
        try:
            b_maj_int = int(b_maj)
        except Exception as e:
            raise ValueError("b_maj must be int-like") from e
        try:
            b_major_int = int(b_major)
        except Exception:
            b_major_int = None
        if b_major_int is not None and b_major_int != 1 and b_major_int != b_maj_int:
            raise ValueError("Pass only one of b_major or b_maj (alias), or set them equal.")
        b_major = b_maj_int

    # ---- gid helpers ----
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
    
    def _rigid_align_2d(X2, F2, *, target=(0.0, 0.0), use_rotation=True,
                    ref_dir=(1.0, 0.0), rotate_about="major1", eps=1e-12):
        """
        X2: (K,T,2), F2: (3,2)
        Returns aligned (X2a, F2a)

        - Always translates so F2[0] lands on `target`.
        - If use_rotation=True, rotates so (F2[1]-F2[0]) aligns to `ref_dir`.
        This removes random rotation/flip across plots (up to your chosen ref_dir).
        """
        X2 = np.asarray(X2, dtype=float)
        F2 = np.asarray(F2, dtype=float)
        tgt = np.asarray(target, dtype=float).reshape(1, 2)

        # choose pivot point for rotation
        if rotate_about == "origin":
            pivot = np.zeros((1, 2), dtype=float)
        else:
            pivot = F2[0:1, :]  # major1

        # --- optional rotation ---
        if use_rotation:
            d = F2[1] - F2[0]          # current direction (major1->major2)
            rd = np.asarray(ref_dir, dtype=float).reshape(2,)
            nd = np.linalg.norm(d)
            nrd = np.linalg.norm(rd)

            if nd > eps and nrd > eps:
                d = d / nd
                rd = rd / nrd

                # angle from d to rd in 2D
                # cos = d·rd, sin = cross2(d,rd)
                c = float(np.clip(d[0]*rd[0] + d[1]*rd[1], -1.0, 1.0))
                s = float(d[0]*rd[1] - d[1]*rd[0])

                # rotation matrix that maps d -> rd
                R = np.array([[c, -s],
                            [s,  c]], dtype=float)

                # rotate around pivot
                Xr = (X2 - pivot.reshape(1, 1, 2)) @ R.T + pivot.reshape(1, 1, 2)
                Fr = (F2 - pivot) @ R.T + pivot
            else:
                # degenerate: can't define direction, skip rotation
                Xr, Fr = X2, F2
        else:
            Xr, Fr = X2, F2

        # --- translation so major1 goes to target ---
        shift = tgt - Fr[0:1, :]   # (1,2)
        Xa = Xr + shift.reshape(1, 1, 2)
        Fa = Fr + shift
        return Xa, Fa


    # ============================================================
    # Convert inputs
    # ============================================================

    X_raw = _to_np(task_vecs_over_all_time)
    F = _to_np(final_task_vecs)
    R2_raw = _to_np(r2_scores)

    if any(v is None for v in (X_raw, F, R2_raw)):
        raise ValueError("task_vecs_over_all_time, final_task_vecs, r2_scores are required.")

    Xnp = np.asarray(X_raw)
    X_in_ndim = int(Xnp.ndim)

    idx_major = idx_ood = idx_minor = None
    orig_task_idx = None
    maj_names_override = None
    labels = None

    # ============================================================
    # FIX: use_mean=True supports (K,T,B,D) by averaging over B
    #      and IGNORE b_ood/b_minor/b_major for grouping in this mode.
    # ============================================================

    if use_mean and Xnp.ndim == 4:
        # Expect (K,T,B,D)
        K0, T0, B0, D0 = Xnp.shape
        Xnp = Xnp.mean(axis=2)  # -> (K,T,D)

        R2np = np.asarray(R2_raw, dtype=float)
        if R2np.ndim == 2 and R2np.shape == (K0, T0):
            pass
        elif R2np.ndim == 3 and R2np.shape == (K0, B0, T0):
            R2np = R2np.mean(axis=1)  # -> (K,T)
        elif R2np.ndim == 3 and R2np.shape == (K0, T0, B0):
            R2np = R2np.mean(axis=2)  # -> (K,T)
        else:
            raise ValueError(
                f"Unsupported r2_scores shape for use_mean=True with X (K,T,B,D). "
                f"Expected (K,T), (K,B,T), or (K,T,B); got {R2np.shape} (K={K0}, T={T0}, B={B0})."
            )

        R2_raw = R2np
        X_in_ndim = int(Xnp.ndim)

    if use_mean and Xnp.ndim != 3:
        raise ValueError(
            "When use_mean=True, task_vecs_over_all_time must be (K,T,D) "
            "or (K,T,B,D) (which will be averaged over B)."
        )

    # ============================================================
    # Optional expansion for use_mean=False: (K,T,B,D) -> (K',T,D)
    # ============================================================

    if not use_mean and Xnp.ndim == 4:
        K0, T0, B0, D0 = Xnp.shape

        n_minor = int(n_minor)
        if n_ood is None:
            raise ValueError("When use_mean=False and X is (K,T,B,D), pass n_ood explicitly.")
        n_ood = int(n_ood)

        if K0 != 3 + n_ood + n_minor:
            raise ValueError(f"Expected K={3 + n_ood + n_minor}, got {K0}.")

        b_major = int(b_major)
        b_ood = int(b_ood)
        b_minor = int(b_minor)
        if b_major < 1 or b_ood < 1 or (n_minor > 0 and b_minor < 1):
            raise ValueError("b_major/b_ood/b_minor must be >= 1 for expanded plotting.")

        R2np = np.asarray(R2_raw, dtype=float)
        if R2np.ndim == 2 and R2np.shape == (K0, T0):
            R2np = np.repeat(R2np[:, None, :], repeats=B0, axis=1)  # (K0,B0,T0)
        elif R2np.ndim == 3 and R2np.shape == (K0, B0, T0):
            pass
        elif R2np.ndim == 3 and R2np.shape == (K0, T0, B0):
            R2np = np.transpose(R2np, (0, 2, 1))
        else:
            raise ValueError(f"Unsupported r2_scores shape for expanded case: got {R2np.shape}")

        idx_major0 = np.arange(0, 3, dtype=int)
        idx_ood0 = np.arange(3, 3 + n_ood, dtype=int)
        idx_minor0 = np.arange(3 + n_ood, K0, dtype=int) if n_minor > 0 else np.array([], dtype=int)

        def _choose_B(b, seed_offset):
            rng = np.random.default_rng(int(ood_style_seed) + int(seed_offset))
            b_eff = min(int(b), int(B0))
            if b_eff >= B0:
                return np.arange(B0, dtype=int)
            return np.sort(rng.choice(B0, size=b_eff, replace=False))

        B_major = _choose_B(b_major, 101)
        B_ood = _choose_B(b_ood, 202)
        B_minor = _choose_B(b_minor, 303) if n_minor > 0 else np.array([], dtype=int)

        labels0 = _normalize_labels(task_labels, K0)
        if len(labels0) >= 3:
            maj_names_override = [labels0[0], labels0[1], labels0[2]]

        def _take_group(Xg, R2g, idxs0, Bsel):
            Kg = Xg.shape[0]
            if Kg == 0 or len(Bsel) == 0:
                return (
                    np.zeros((0, T0, D0), dtype=Xg.dtype),
                    np.zeros((0, T0), dtype=float),
                    [],
                    [],
                )

            Xsel = Xg[:, :, Bsel, :]  # (Kg,T0,b,D)
            Xflat = np.transpose(Xsel, (0, 2, 1, 3)).reshape(Kg * len(Bsel), T0, D0)

            R2sel = R2g[:, Bsel, :]  # (Kg,b,T0)
            R2flat = R2sel.reshape(Kg * len(Bsel), T0)

            new_labels = []
            for k0 in idxs0.tolist():
                base = labels0[k0]
                for j in range(len(Bsel)):
                    new_labels.append(f"{base}|b{j}")

            orig = []
            for k0 in idxs0.tolist():
                for _ in range(len(Bsel)):
                    orig.append(int(k0))

            return Xflat, R2flat, new_labels, orig

        X_major, R2_major, lab_major, orig_major = _take_group(Xnp[idx_major0], R2np[idx_major0], idx_major0, B_major)
        X_ood, R2_ood, lab_ood, orig_ood = _take_group(Xnp[idx_ood0], R2np[idx_ood0], idx_ood0, B_ood)
        X_minor, R2_minor, lab_minor, orig_minor = _take_group(Xnp[idx_minor0], R2np[idx_minor0], idx_minor0, B_minor)

        Xnp = np.concatenate([X_major, X_ood, X_minor], axis=0)
        R2_raw = np.concatenate([R2_major, R2_ood, R2_minor], axis=0)
        labels = lab_major + lab_ood + lab_minor

        orig_task_idx = np.array(orig_major + orig_ood + orig_minor, dtype=int)

        major_count = X_major.shape[0]
        ood_count = X_ood.shape[0]
        minor_count = X_minor.shape[0]

        idx_major = np.arange(0, major_count, dtype=int)
        idx_ood = np.arange(major_count, major_count + ood_count, dtype=int)
        idx_minor = (
            np.arange(major_count + ood_count, major_count + ood_count + minor_count, dtype=int)
            if minor_count > 0
            else np.array([], dtype=int)
        )

    elif not use_mean and Xnp.ndim not in (3, 4):
        raise ValueError("When use_mean=False, task_vecs_over_all_time must be (K,T,B,D) or (K,T,D).")

    # ============================================================
    # Flatten X / R2
    # ============================================================

    X, _, _ = _flatten_leading(Xnp, name="task_vecs_over_all_time", tail_ndim=2)
    K, T, D = X.shape

    R2, _, _ = _flatten_leading(_to_np(R2_raw), name="r2_scores", tail_ndim=1)
    if R2.shape != (K, T):
        raise ValueError(f"r2_scores shape mismatch: expected {(K, T)}, got {R2.shape}")

    if labels is None:
        labels = _normalize_labels(task_labels, K)

    # ============================================================
    # Groups (if not already computed)
    # ============================================================

    if idx_major is None:
        n_minor = int(n_minor)
        if n_minor < 0 or n_minor > K:
            raise ValueError("n_minor out of range.")

        # FIX: use_mean=True must IGNORE b_ood/b_minor/b_major.
        expanded_along_K = False
        bM = bO = bN = 1

        if not use_mean:
            bM = int(b_major)
            bO = int(b_ood)
            bN = int(b_minor)

            expanded_along_K = (X_in_ndim == 3) and ((bM > 1) or (bO > 1) or (n_minor > 0 and bN > 1))

            if expanded_along_K:
                major_count = 3 * bM
                minor_count = n_minor * (bN if n_minor > 0 else 1)
                ood_count = K - major_count - minor_count
                if major_count > K or ood_count < 0:
                    expanded_along_K = False

        if not expanded_along_K:
            if K < 3:
                raise ValueError("Assumes 3 major tasks exist (K must be >= 3).")

            if use_mean:
                if n_ood is None:
                    ood_count = K - 3 - n_minor
                else:
                    ood_count = int(n_ood)
                    if 3 + ood_count + n_minor != K:
                        raise ValueError("Grouping mismatch: 3 + n_ood + n_minor must equal K.")
                major_count = 3
                minor_count = n_minor
            else:
                major_count = 3
                minor_count = n_minor
                ood_count = K - major_count - minor_count

            if ood_count < 0:
                raise ValueError("Grouping mismatch: K must be >= 3 + n_minor (and n_ood if provided).")

        idx_major = np.arange(0, major_count, dtype=int)
        idx_ood = np.arange(major_count, major_count + ood_count, dtype=int)
        idx_minor = (
            np.arange(major_count + ood_count, K, dtype=int)
            if minor_count > 0
            else np.array([], dtype=int)
        )

        if expanded_along_K and orig_task_idx is None:
            orig_task_idx = -np.ones((K,), dtype=int)
            for kk in range(min(major_count, K)):
                orig_task_idx[kk] = int(kk // max(1, bM))

    # Major base labels -> consistent coloring even when expanded
    if maj_names_override is not None:
        major_bases = [_strip_b_suffix(str(maj_names_override[i])) for i in range(3)]
        maj_names = [_prettify_label(b) for b in major_bases]
        major_base_to_color = {major_bases[i]: i for i in range(3)}
    else:
        major_bases = []
        for k in idx_major.tolist():
            base = _strip_b_suffix(labels[int(k)])
            if base not in major_bases:
                major_bases.append(base)
            if len(major_bases) == 3:
                break
        if len(major_bases) < 3:
            major_bases = [f"Task {i + 1}" for i in range(3)]
        maj_names = [_prettify_label(b) for b in major_bases]
        major_base_to_color = {b: i for i, b in enumerate(major_bases)}

    # ============================================================
    # Projection
    # ============================================================

    X_proj, F_proj = _project_to_plane(X, np.asarray(F, dtype=float))

    if align_major_rigid:
        X_proj, F_proj = _rigid_align_2d(
            X_proj, F_proj,
            target=align_major_target,
            use_rotation=align_major_use_rotation,
            ref_dir=align_major_ref_dir,
            rotate_about=align_major_rotate_about,
        )


    # ============================================================
    # Time indices shown (pow2 + dense + must_include + annotate)
    # ============================================================

    def _normalize_time_list(ts, T_):
        out = []
        if ts is None:
            return out
        if isinstance(ts, (int, np.integer)):
            ts = [int(ts)]
        for tt in list(ts):
            try:
                t = int(tt)
            except Exception:
                continue
            if t < 0:
                t = int(T_) + t  # python-style negatives
            if 0 <= t < int(T_):
                out.append(int(t))
        return out

    must_times = set(_normalize_time_list(must_include_times, T))

    # Annotation times = user annotate_times ∪ must_include_times,
    # BUT: exclude index 19 (20th) unless it's in must_include_times.
    ann_times = set(_normalize_time_list(annotate_times, T)) | set(must_times)
    if 19 in ann_times and 19 not in must_times:
        ann_times.discard(19)

    # NEW: optional start/end annotation (must_include_times still forces)
    if (not annotate_start) and (0 in ann_times) and (0 not in must_times):
        ann_times.discard(0)
    if (not annotate_end) and ((T - 1) in ann_times) and ((T - 1) not in must_times):
        ann_times.discard(T - 1)

    t_base_full = _pow2_time_indices(T, include_last=include_last)

    m = int(max(2, min(T, int(max_points))))
    t_dense = np.unique(np.round(np.linspace(0, T - 1, m)).astype(int))

    t_show = np.unique(np.concatenate([t_base_full, t_dense, np.array([0, T - 1], dtype=int)]))
    if must_times:
        t_show = np.unique(np.concatenate([t_show, np.array(sorted(must_times), dtype=int)]))
    if annotate and ann_times:
        t_show = np.unique(np.concatenate([t_show, np.array(sorted(ann_times), dtype=int)]))

    t_show.sort()
    time_to_show_idx = {int(t): i for i, t in enumerate(t_show.tolist())}

    # ============================================================
    # Point alpha schedule (early darker -> late lighter)
    # ============================================================

    u = np.linspace(0.0, 1.0, t_show.size)
    a_pts = float(point_alpha_end) + (float(point_alpha_start) - float(point_alpha_end)) * (
        (1.0 - u) ** float(point_alpha_power)
    )
    a_pts = np.clip(a_pts, 0.0, 1.0)

    # anchors
    if show_pow2_anchors and t_base_full.size > 0:
        t_base = t_base_full.copy()
        if anchor_skip_ends and T > 1:
            t_base = t_base[(t_base != 0) & (t_base != (T - 1))]

        if t_base.size > 0:
            uA = t_base.astype(float) / max(1.0, float(T - 1))
            aA = float(point_alpha_end) + (float(point_alpha_start) - float(point_alpha_end)) * (
                (1.0 - uA) ** float(point_alpha_power)
            )
            aA = np.clip(np.maximum(aA, float(anchor_alpha_floor)), 0.0, 1.0)
        else:
            aA = None
    else:
        t_base = np.array([], dtype=int)
        aA = None

    sizes_area = _r2_to_sizes_area(R2, size_min=size_min, size_max=size_max)

    # ============================================================
    # Figure / Axes
    # ============================================================

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
        if clear_ax:
            ax.clear()
        elif cleanup_previous:
            _cleanup_previous_artists(ax)

    ax.margins(0.10)

    # ============================================================
    # Jitter (ONLY for scatter points)
    # ============================================================

    x_all = X_proj[..., 0].ravel()
    y_all = X_proj[..., 1].ravel()
    x_rng = float(np.ptp(x_all) + 1e-9)
    y_rng = float(np.ptp(y_all) + 1e-9)

    if jitter_points:
        rng = np.random.default_rng(int(jitter_seed))
        J = rng.normal(size=(K, t_show.size, 2)).astype(float)
        J[:, :, 0] *= float(jitter_scale) * x_rng
        J[:, :, 1] *= float(jitter_scale) * y_rng
    else:
        J = np.zeros((K, t_show.size, 2), dtype=float)

    # ============================================================
    # Styles
    # ============================================================

    if len(major_colors) != 3:
        raise ValueError("major_colors must have length 3.")
    major_rgbs = [mcolors.to_rgb(c) for c in major_colors]

    if ood_linestyle_cycle is None:
        ood_linestyle_cycle = [
            (0, (6, 2)),
            (0, (4, 2, 1, 2)),
            (0, (2.5, 1.2)),
            (0, (7, 2, 2, 2)),
            (0, (3, 1, 1, 1)),
        ]

    rng_ood = np.random.default_rng(int(ood_style_seed))
    ood_style = {}
    for j, k in enumerate(idx_ood.tolist()):
        rgb = _jitter_color(
            ood_base_color,
            rng_ood,
            hue_j=float(ood_hue_jitter),
            sat_j=float(ood_sat_jitter),
            val_j=float(ood_val_jitter),
        )
        ls = (
            ood_linestyle_cycle[j % len(ood_linestyle_cycle)]
            if (ood_vary_linestyle and len(ood_linestyle_cycle) > 0)
            else "--"
        )
        ood_style[int(k)] = (rgb, ls)

    rng_minor = np.random.default_rng(int(ood_style_seed) + 999)
    minor_style = {}
    for j, k in enumerate(idx_minor.tolist()):
        rgb = _jitter_color(minor_base_color, rng_minor, hue_j=0.02, sat_j=0.06, val_j=0.06)
        minor_style[int(k)] = rgb

    # ============================================================
    # Plot helpers
    # ============================================================

    def _draw_gradient_polyline(Pxy, rgb, ls, gid_suffix, z=1.5):
        """Line alpha: DARK at start -> LIGHT at end."""
        if Pxy.shape[0] < 2:
            return None

        segs = np.stack([Pxy[:-1], Pxy[1:]], axis=1)
        nseg = segs.shape[0]
        uu = np.linspace(0.0, 1.0, nseg)

        a0 = float(line_alpha_end)
        a1 = float(line_alpha_end) * float(line_alpha_start_factor)
        al = a0 + (a1 - a0) * (uu ** float(line_alpha_power))
        al = np.clip(al, 0.0, 1.0)

        cols = np.column_stack(
            [
                np.full(nseg, rgb[0]),
                np.full(nseg, rgb[1]),
                np.full(nseg, rgb[2]),
                al,
            ]
        )

        lc = LineCollection(
            segs,
            colors=cols,
            linewidths=float(line_width),
            linestyles=ls,
            zorder=float(z),
        )
        ax.add_collection(lc)
        return _set_gid(lc, gid_suffix)

    def _draw_points(Pfull, k, rgb, gid_suffix):
        idxs = np.arange(t_show.size, dtype=int)
        Pxy = Pfull[t_show] + J[k, idxs]

        fc = np.column_stack(
            [
                np.full_like(a_pts, rgb[0], dtype=float),
                np.full_like(a_pts, rgb[1], dtype=float),
                np.full_like(a_pts, rgb[2], dtype=float),
                a_pts,
            ]
        )

        coll = ax.scatter(
            Pxy[:, 0],
            Pxy[:, 1],
            s=sizes_area[k, t_show] * float(mid_size_factor),
            marker="o",
            facecolors=fc,
            edgecolors="none",
            linewidths=0.0,
            zorder=2.2,
        )
        return _set_gid(coll, gid_suffix)

    def _draw_pow2_anchors(Pfull, k, rgb, gid_suffix):
        if (not show_pow2_anchors) or (t_base.size == 0) or (aA is None):
            return None

        ii = np.array([time_to_show_idx[int(t)] for t in t_base.tolist()], dtype=int)
        Pxy = Pfull[t_base] + J[k, ii]

        fc = np.column_stack(
            [
                np.full_like(aA, rgb[0], dtype=float),
                np.full_like(aA, rgb[1], dtype=float),
                np.full_like(aA, rgb[2], dtype=float),
                aA,
            ]
        )

        coll = ax.scatter(
            Pxy[:, 0],
            Pxy[:, 1],
            s=sizes_area[k, t_base] * float(mid_size_factor),
            marker="o",
            facecolors=fc,
            edgecolors="none",
            linewidths=0.0,
            zorder=2.55,
        )
        return _set_gid(coll, gid_suffix)

    def _draw_end(Pfull, k, rgb, gid_group):
        if (not include_last) or T <= 0:
            return None

        tlast = T - 1
        ii = time_to_show_idx.get(int(tlast), None)
        jxy = J[k, ii] if ii is not None else np.zeros(2, dtype=float)

        px, py = float(Pfull[tlast, 0] + jxy[0]), float(Pfull[tlast, 1] + jxy[1])

        en = ax.scatter(
            [px],
            [py],
            s=[sizes_area[k, -1] * float(mid_size_factor)],
            marker="o",
            facecolors=[(rgb[0], rgb[1], rgb[2], float(end_marker_alpha))],
            edgecolors=[end_marker_edge_color],
            linewidths=float(end_marker_edge_width),
            zorder=3.35,
        )
        return _set_gid(en, f"{gid_group}_end:{k}")

    def _draw_one(k, Pfull, rgb, ls, gid_group):
        Pxy_line = Pfull[t_show]  # keep the line un-jittered
        if use_gradient_line:
            _draw_gradient_polyline(Pxy_line, rgb, ls, f"{gid_group}_gline:{k}", z=1.6)
        else:
            ln = ax.plot(
                Pxy_line[:, 0],
                Pxy_line[:, 1],
                color=(rgb[0], rgb[1], rgb[2], float(line_alpha_end)),
                lw=float(line_width),
                linestyle=ls,
                zorder=1.6,
            )[0]
            _set_gid(ln, f"{gid_group}_line:{k}")

        _draw_points(Pfull, k, rgb, f"{gid_group}_pts:{k}")
        _draw_pow2_anchors(Pfull, k, rgb, f"{gid_group}_anchors:{k}")
        _draw_end(Pfull, k, rgb, gid_group)

    # ============================================================
    # Draw trajectories
    # ============================================================

    def _major_rgb_for_k(k):
        if orig_task_idx is not None:
            try:
                ti = int(orig_task_idx[int(k)])
            except Exception:
                ti = None
            if ti is not None and 0 <= ti < 3:
                return major_rgbs[ti]
        base = _strip_b_suffix(labels[int(k)])
        ci = int(major_base_to_color.get(base, 0))
        return major_rgbs[ci]

    for k in idx_major.tolist():
        _draw_one(int(k), X_proj[int(k)], _major_rgb_for_k(int(k)), major_linestyle, "major")

    for k in idx_ood.tolist():
        rgb, ls = ood_style[int(k)]
        _draw_one(int(k), X_proj[int(k)], rgb, ls, "ood")

    for k in idx_minor.tolist():
        rgb = minor_style[int(k)]
        _draw_one(int(k), X_proj[int(k)], rgb, minor_linestyle, "minor")

    # ============================================================
    # Legend + Cosmetics
    # ============================================================

    handles = []
    for i in range(3):
        handles.append(
            Line2D(
                [0],
                [0],
                color=_rgba(major_colors[i], 0.95),
                lw=2.4,
                linestyle=major_linestyle,
                label=f"Major: {maj_names[i]}",
            )
        )

    if idx_ood.size > 0:
        handles.append(
            Line2D(
                [0],
                [0],
                color=_rgba(ood_base_color, 0.95),
                lw=2.4,
                linestyle="--",
                label="OOD trajectories (varied styles)",
            )
        )

    if idx_minor.size > 0:
        handles.append(
            Line2D(
                [0],
                [0],
                color=_rgba(minor_base_color, 0.95),
                lw=2.4,
                linestyle=minor_linestyle,
                label="Minor trajectories",
            )
        )

        # ============================================================
    # Legend + Cosmetics
    #   - Legend 1: groups (lines)
    #   - Legend 2: R² -> marker size (a few reference sizes)
    # ============================================================

    handles = []
    for i in range(3):
        handles.append(
            Line2D(
                [0],
                [0],
                color=_rgba(major_colors[i], 0.95),
                lw=2.4,
                linestyle=major_linestyle,
                label=f"Major: {maj_names[i]}",
            )
        )

    if idx_ood.size > 0:
        handles.append(
            Line2D(
                [0],
                [0],
                color=_rgba(ood_base_color, 0.95),
                lw=2.4,
                linestyle="--",
                label="OOD trajectories (varied styles)",
            )
        )

    if idx_minor.size > 0:
        handles.append(
            Line2D(
                [0],
                [0],
                color=_rgba(minor_base_color, 0.95),
                lw=2.4,
                linestyle=minor_linestyle,
                label="Minor trajectories",
            )
        )
    
    if show_legend:
        # --- Legend 1 (groups) ---
        leg1 = ax.legend(handles=handles, frameon=False, loc="upper left")

        # --- Legend 2 (size ~ R²) ---
        r2_min = float(np.min(R2))
        r2_max = float(np.max(R2))

        # pick a few representative R² values (quantiles tend to look nice)
        r2_levels = np.quantile(R2, [0.2, 0.5, 0.8]).astype(float)

        # ensure uniqueness / stability (avoid duplicates if R² is concentrated)
        r2_levels = np.unique(np.round(r2_levels, 2))
        if r2_levels.size < 3:
            r2_levels = np.unique(np.round(np.array([r2_min, 0.5 * (r2_min + r2_max), r2_max]), 2))

        size_handles = []
        size_labels = []
        for v in r2_levels:
            area = _r2_value_to_area(
                v,
                r2_min=r2_min,
                r2_max=r2_max,
                size_min=size_min,
                size_max=size_max,
            )
            area *= float(mid_size_factor)  # match what you actually plot

            h = ax.scatter(
                [], [],                       # dummy artist
                s=area,
                marker="o",
                facecolors=[(0, 0, 0, 0.18)],
                edgecolors=[(0, 0, 0, 0.45)],
                linewidths=0.9,
            )
            size_handles.append(h)
            size_labels.append(f"R² = {float(v):.2f}")

        leg2 = ax.legend(
            handles=size_handles,
            labels=size_labels,
            title="Marker size",
            scatterpoints=1,
            frameon=False,
            loc="upper right",
            borderpad=0.3,
            labelspacing=0.6,
            handletextpad=0.8,
        )

        # Keep both legends
        ax.add_artist(leg1)
    else:
        if getattr(ax, "legend_", None) is not None:
            ax.legend_.remove()
        
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
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
    # Reference stars (NO text labels)
    # ============================================================

    if show_final_task_vecs:
        for i in range(3):
            rgb = major_rgbs[i]
            ref = ax.scatter(
                [F_proj[i, 0]],
                [F_proj[i, 1]],
                s=float(final_marker_size),
                marker=final_marker,
                facecolors=[(rgb[0], rgb[1], rgb[2], 1.0)],
                edgecolors=[final_edge_color],
                linewidths=float(final_edge_width),
                zorder=4.2,
            )
            _set_gid(ref, f"ref_vec:{i}")

    # ============================================================
    # Prepare renderer + obstacles + point cloud for label placement
    # ============================================================

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox_disp = ax.get_window_extent(renderer=renderer)

    fixed_obstacles = []
    if getattr(ax, "legend_", None) is not None:
        try:
            fixed_obstacles.append(ax.legend_.get_window_extent(renderer=renderer))
        except Exception:
            pass

    fixed_obstacle_rects = [
        (float(ob.x0), float(ob.y0), float(ob.x1), float(ob.y1))
        for ob in fixed_obstacles
    ]

    size_factor_t = np.full(t_show.size, float(mid_size_factor), dtype=float)
    P_sample = (X_proj[:, t_show, :] + J).reshape(-1, 2)
    S_sample = (sizes_area[:, t_show] * size_factor_t[None, :]).reshape(-1)

    max_point_obstacles = 14000
    if P_sample.shape[0] > max_point_obstacles:
        step = max(1, P_sample.shape[0] // max_point_obstacles)
        P_sample = P_sample[::step]
        S_sample = S_sample[::step]

    P_disp = ax.transData.transform(P_sample) if P_sample.size else np.zeros((0, 2), dtype=float)

    if S_sample.size:
        r_px = np.sqrt(S_sample / np.pi) * (fig.dpi / 72.0)
        r_px = np.clip(r_px, 1.5, 14.0)
    else:
        r_px = np.zeros((0,), dtype=float)

    if show_final_task_vecs and F_proj.size:
        P_star = ax.transData.transform(F_proj)
        r_star = np.sqrt(np.full((3,), float(final_marker_size)) / np.pi) * (fig.dpi / 72.0)
        r_star = np.clip(r_star, 2.0, 18.0)
        if P_disp.size:
            P_disp = np.vstack([P_disp, P_star])
            r_px = np.concatenate([r_px, r_star])
        else:
            P_disp = P_star
            r_px = r_star

    if annotation_point_clear_px is None:
        fontsize_px = float(annotation_fontsize) * (fig.dpi / 72.0)
        point_clear_px = max(2.0, 0.45 * fontsize_px)
    else:
        point_clear_px = float(annotation_point_clear_px)

    # ============================================================
    # Annotation placement solver (beam search over discrete candidates)
    # ============================================================

    def _rects_overlap(r1, r2):
        return (r1[0] < r2[2]) and (r1[2] > r2[0]) and (r1[1] < r2[3]) and (r1[3] > r2[1])

    def _expand_rect(r, m):
        return (r[0] - m, r[1] - m, r[2] + m, r[3] + m)

    def _point_to_rect_distance(px_, py_, r):
        dx = 0.0
        if px_ < r[0]:
            dx = r[0] - px_
        elif px_ > r[2]:
            dx = px_ - r[2]
        dy = 0.0
        if py_ < r[1]:
            dy = r[1] - py_
        elif py_ > r[3]:
            dy = py_ - r[3]
        return float(np.hypot(dx, dy))

    def _rect_inside_axes(r, axb, pad):
        return (r[0] >= axb.x0 + pad) and (r[2] <= axb.x1 - pad) and (r[1] >= axb.y0 + pad) and (r[3] <= axb.y1 - pad)

    def _rect_to_grid_bounds(r, axb, nx, ny):
        w = float(axb.width) + 1e-9
        h = float(axb.height) + 1e-9
        fx0 = (r[0] - axb.x0) / w * nx
        fx1 = (r[2] - axb.x0) / w * nx
        fy0 = (r[1] - axb.y0) / h * ny
        fy1 = (r[3] - axb.y0) / h * ny
        x0 = int(np.floor(np.clip(fx0, 0, nx)))
        x1 = int(np.ceil(np.clip(fx1, 0, nx)))
        y0 = int(np.floor(np.clip(fy0, 0, ny)))
        y1 = int(np.ceil(np.clip(fy1, 0, ny)))
        x0 = max(0, min(nx, x0))
        x1 = max(0, min(nx, x1))
        y0 = max(0, min(ny, y0))
        y1 = max(0, min(ny, y1))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return x0, x1, y0, y1

    def _box_blur2d(A, k):
        k = int(k)
        if k <= 1:
            return A
        if k % 2 == 0:
            k += 1
        pad = k // 2
        P = np.pad(A, ((pad, pad), (pad, pad)), mode="edge")
        S = np.pad(P, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
        S = np.cumsum(np.cumsum(S, axis=0), axis=1)
        out = S[k:, k:] - S[:-k, k:] - S[k:, :-k] + S[:-k, :-k]
        out = out / float(k * k)
        return out

    def _build_density_integral(P_disp_, axb, grid_shape, blur_k):
        ny, nx = int(grid_shape[1]), int(grid_shape[0])
        ny = max(ny, 16)
        nx = max(nx, 16)

        G = np.zeros((ny, nx), dtype=float)
        if P_disp_.size == 0:
            I = np.zeros((ny + 1, nx + 1), dtype=float)
            return I, (nx, ny), 0.0

        w = float(axb.width) + 1e-9
        h = float(axb.height) + 1e-9
        u = (P_disp_[:, 0] - float(axb.x0)) / w
        v = (P_disp_[:, 1] - float(axb.y0)) / h
        mask = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        if not np.any(mask):
            I = np.zeros((ny + 1, nx + 1), dtype=float)
            return I, (nx, ny), 0.0

        u = u[mask]
        v = v[mask]
        ix = np.floor(u * (nx - 1)).astype(int)
        iy = np.floor(v * (ny - 1)).astype(int)
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)

        np.add.at(G, (iy, ix), 1.0)
        Gs = _box_blur2d(G, int(blur_k))
        Gs = _box_blur2d(Gs, int(blur_k))

        ref = float(np.percentile(Gs, 70.0))
        if not np.isfinite(ref) or ref <= 1e-9:
            ref = float(np.mean(Gs) + 1e-9)

        I = np.pad(Gs, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
        I = np.cumsum(np.cumsum(I, axis=0), axis=1)
        return I, (nx, ny), ref

    def _density_sum(I, nxny, rect, axb):
        nx, ny = nxny
        x0, x1, y0, y1 = _rect_to_grid_bounds(rect, axb, nx, ny)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return float(I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0])

    I_rho, (nx_rho, ny_rho), rho_ref = _build_density_integral(
        P_disp, ax_bbox_disp, annotation_density_grid, annotation_density_blur_k
    )

    Px_all = P_disp[:, 0] if P_disp.size else np.zeros((0,), dtype=float)
    Py_all = P_disp[:, 1] if P_disp.size else np.zeros((0,), dtype=float)
    r_px_all = r_px
    r_px_max = float(np.max(r_px_all)) if r_px_all.size else 0.0

    def _count_point_collisions(rect, *, clear_px):
        if Px_all.size == 0:
            return 0
        x0, y0, x1, y1 = rect
        pad = r_px_max + float(clear_px)
        mask = (
            (Px_all >= (x0 - pad))
            & (Px_all <= (x1 + pad))
            & (Py_all >= (y0 - pad))
            & (Py_all <= (y1 + pad))
        )
        if not np.any(mask):
            return 0

        Pxm = Px_all[mask]
        Pym = Py_all[mask]
        rm = r_px_all[mask] if r_px_all.size else 0.0

        dx = np.maximum(np.maximum(x0 - Pxm, 0.0), Pxm - x1)
        dy = np.maximum(np.maximum(y0 - Pym, 0.0), Pym - y1)
        dist = np.hypot(dx, dy)
        thresh = rm + float(clear_px)
        return int(np.count_nonzero(dist < thresh))

    # ============================================================
    # Build annotation problem (anchors + texts)
    # ============================================================

    ann_items = []
    if annotate and idx_ood.size > 0 and len(ann_times) > 0:
        if annotate_k is None:
            ks = idx_ood.tolist()
        elif isinstance(annotate_k, (list, tuple, np.ndarray)):
            ks = [int(x) for x in list(annotate_k)]
        else:
            ks = [int(annotate_k)]

        def _ordinal(n: int) -> str:
            if 10 <= (n % 100) <= 20:
                suf = "th"
            else:
                suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
            return f"{n}{suf}"

        def _label_for_time(tt: int) -> str:
            if tt == 0:
                return "Start"
            if tt == T - 1:
                return "End"
            return _ordinal(tt + 1)

        idx_ood_set = set(idx_ood.tolist())
        times_ordered = sorted(ann_times)
        # keep End early in ordering (helps beam search sometimes)
        if (T - 1) in ann_times:
            times_ordered = [T - 1] + [t for t in times_ordered if t != (T - 1)]

        for k in ks:
            if k < 0 or k >= K:
                continue

            rgb_k = ood_style[int(k)][0] if (int(k) in idx_ood_set) else _major_rgb_for_k(k)

            bbox_kwargs = dict(
                boxstyle=f"round,pad={float(annotation_box_pad)}",
                fc="white",
                ec=(rgb_k[0], rgb_k[1], rgb_k[2], 0.26),
                lw=float(annotation_box_lw),
                alpha=float(annotation_box_alpha),
            )

            for tt in times_ordered:
                if tt < 0 or tt >= T:
                    continue

                ii = time_to_show_idx.get(int(tt), None)
                jxy = J[k, ii] if ii is not None else np.zeros(2, dtype=float)

                px_data = float(X_proj[k, tt, 0] + jxy[0])
                py_data = float(X_proj[k, tt, 1] + jxy[1])

                r2v = float(R2[k, tt])
                text = f"{_label_for_time(tt)}\nR²={r2v:.2f}"

                s_vis = float(sizes_area[k, tt]) * float(mid_size_factor)
                r_anchor_px = float(np.sqrt(s_vis / np.pi) * (fig.dpi / 72.0))

                ann_items.append(
                    dict(
                        k=int(k),
                        tt=int(tt),
                        anchor_data=(px_data, py_data),
                        anchor_disp=ax.transData.transform((px_data, py_data)),
                        text=text,
                        rgb=rgb_k,
                        bbox_kwargs=bbox_kwargs,
                        r_anchor_px=r_anchor_px,
                    )
                )

    # If nothing to annotate, return now.
    if not ann_items:
        return fig, ax

    # ============================================================
    # Measure label sizes (in display px)
    # ============================================================

    def _measure_text_box_px(text, bbox_kwargs):
        tmp = ax.text(
            0.0,
            0.0,
            text,
            fontsize=int(annotation_fontsize),
            linespacing=0.90,
            ha="left",
            va="bottom",
            bbox=bbox_kwargs,
            zorder=9,
        )
        fig.canvas.draw()
        bb = tmp.get_window_extent(renderer=renderer)
        tmp.remove()
        return float(bb.width), float(bb.height)

    for it in ann_items:
        wpx, hpx = _measure_text_box_px(it["text"], it["bbox_kwargs"])
        it["w_px"] = wpx
        it["h_px"] = hpx
        it["half_diag_px"] = 0.5 * float(np.hypot(wpx, hpx))

    # ============================================================
    # Candidate generation and scoring
    # ============================================================

    def _rect_from_ref(x_ref, y_ref, w_px, h_px, ha, va):
        if ha == "left":
            x0 = x_ref
            x1 = x_ref + w_px
        elif ha == "right":
            x0 = x_ref - w_px
            x1 = x_ref
        else:
            x0 = x_ref - 0.5 * w_px
            x1 = x_ref + 0.5 * w_px

        if va == "bottom":
            y0 = y_ref
            y1 = y_ref + h_px
        elif va == "top":
            y0 = y_ref - h_px
            y1 = y_ref
        else:
            y0 = y_ref - 0.5 * h_px
            y1 = y_ref + 0.5 * h_px

        return (float(x0), float(y0), float(x1), float(y1))

    def _candidate_set_for_item(item, Lmax_px, *, collision_cap):
        ax_cx = 0.5 * (ax_bbox_disp.x0 + ax_bbox_disp.x1)
        ax_cy = 0.5 * (ax_bbox_disp.y0 + ax_bbox_disp.y1)
        ax0, ay0 = float(item["anchor_disp"][0]), float(item["anchor_disp"][1])

        vx = ax0 - ax_cx
        vy = ay0 - ax_cy
        if float(np.hypot(vx, vy)) < 1e-6:
            vx, vy = 1.0, 0.0
        base_angle = float(np.arctan2(vy, vx))

        Lmin = float(item["r_anchor_px"]) + float(annotation_min_gap_px)
        Lmax = float(Lmax_px)
        if Lmax < Lmin + 2.0:
            Lmax = Lmin + 2.0

        L0 = Lmin + float(annotation_prefer_length_scale) * float(item["half_diag_px"])
        L0 = float(np.clip(L0, Lmin + 1.0, Lmax - 1.0 if Lmax > Lmin + 2.0 else Lmax))

        nR = int(max(4, annotation_candidate_radii_n))
        radii = np.linspace(Lmin, Lmax, nR)
        extra = np.array([L0, 0.5 * (Lmin + L0), 0.5 * (L0 + Lmax)], dtype=float)
        radii = np.unique(np.concatenate([radii, extra]))
        radii = radii[(radii >= Lmin) & (radii <= Lmax)]
        radii.sort()

        angles = np.deg2rad(np.array(list(annotation_candidate_angles_deg), dtype=float))

        cand = []
        wpx = float(item["w_px"])
        hpx = float(item["h_px"])
        Lspan = max(8.0, (Lmax - Lmin))

        for r in radii:
            for da in angles:
                ang = base_angle + float(da)
                dx = float(r * np.cos(ang))
                dy = float(r * np.sin(ang))

                ha = "left" if dx >= 0 else "right"
                va = "bottom" if dy >= 0 else "top"

                x_ref = ax0 + dx
                y_ref = ay0 + dy

                rect = _rect_from_ref(x_ref, y_ref, wpx, hpx, ha, va)
                if not _rect_inside_axes(rect, ax_bbox_disp, float(annotation_pad_px)):
                    continue

                rect_pad = _expand_rect(rect, float(annotation_sep_px))
                if fixed_obstacle_rects:
                    if any(_rects_overlap(rect_pad, ob_r) for ob_r in fixed_obstacle_rects):
                        continue

                L = _point_to_rect_distance(ax0, ay0, rect)
                if (L < Lmin) or (L > Lmax):
                    continue

                cnt = _count_point_collisions(rect, clear_px=float(point_clear_px))
                if collision_cap is not None and int(cnt) > int(collision_cap):
                    continue

                dens_sum = _density_sum(I_rho, (nx_rho, ny_rho), rect_pad, ax_bbox_disp)
                x0g, x1g, y0g, y1g = _rect_to_grid_bounds(rect_pad, ax_bbox_disp, nx_rho, ny_rho)
                area_cells = max(1.0, float((x1g - x0g) * (y1g - y0g)))
                dens_mean = dens_sum / area_cells
                dens_norm = dens_mean / (rho_ref + 1e-9)

                len_cost = ((L - L0) / Lspan) ** 2
                cost = (
                    1.0 * float(len_cost)
                    + float(annotation_density_weight) * float(dens_norm)
                    + float(annotation_point_penalty) * float(cnt)
                )

                cand.append(
                    dict(
                        cost=float(cost),
                        dx_px=float(dx),
                        dy_px=float(dy),
                        dx_pt=float(dx) * 72.0 / float(fig.dpi),
                        dy_pt=float(dy) * 72.0 / float(fig.dpi),
                        ha=ha,
                        va=va,
                        rect=rect,
                        rect_pad=rect_pad,
                        point_cnt=int(cnt),
                    )
                )

        cand.sort(key=lambda c: c["cost"])
        return cand

    def _beam_solve(items, cand_lists, *, beam_width, expand_k):
        order = list(range(len(items)))
        order.sort(key=lambda i: (len(cand_lists[i]), -(items[i]["w_px"] * items[i]["h_px"])))
        beam = [(0.0, {}, [])]

        for idx in order:
            cands = cand_lists[idx]
            if not cands:
                return None
            cands = cands[: int(max(1, expand_k))]

            new_beam = []
            for total_cost, chosen, rects in beam:
                for c in cands:
                    rpad = c["rect_pad"]
                    if any(_rects_overlap(rpad, rr) for rr in rects):
                        continue
                    new_chosen = dict(chosen)
                    new_chosen[idx] = c
                    new_rects = rects + [rpad]
                    new_beam.append((total_cost + c["cost"], new_chosen, new_rects))

            if not new_beam:
                return None
            new_beam.sort(key=lambda x: x[0])
            beam = new_beam[: int(max(1, beam_width))]

        best = min(beam, key=lambda x: x[0])
        return best[1]

    def _polish_assignment(items, cand_lists, chosen, *, passes):
        if chosen is None:
            return None
        n = len(items)

        def build_rects(except_i=None):
            rr = []
            for i in range(n):
                if i == except_i:
                    continue
                if i in chosen:
                    rr.append(chosen[i]["rect_pad"])
            return rr

        for _ in range(int(max(0, passes))):
            improved = False
            order = list(range(n))
            order.sort(key=lambda i: (-(chosen[i]["cost"] if i in chosen else 0.0),
                                     -(chosen[i]["point_cnt"] if i in chosen else 0)))
            for i in order:
                if i not in chosen:
                    continue
                current = chosen[i]
                rects_other = build_rects(except_i=i)

                for c in cand_lists[i]:
                    if c["cost"] >= current["cost"] - 1e-9:
                        break
                    if any(_rects_overlap(c["rect_pad"], rr) for rr in rects_other):
                        continue
                    chosen[i] = c
                    improved = True
                    break
            if not improved:
                break
        return chosen

    base_Lmax = float(annotation_max_leader_px)
    max_growth = float(annotation_max_leader_growth)
    ntries = int(max(1, annotation_max_tries))

    chosen = None
    for attempt in range(ntries):
        grow = 1.0 + (max_growth - 1.0) * (attempt / float(ntries - 1)) if ntries > 1 else 1.0
        Lmax_try = base_Lmax * grow
        cap_try = None if annotation_point_collision_cap is None else int(annotation_point_collision_cap) + int(attempt)

        cand_lists = [_candidate_set_for_item(it, Lmax_try, collision_cap=cap_try) for it in ann_items]
        if any(len(cl) == 0 for cl in cand_lists):
            chosen = None
            continue

        chosen = _beam_solve(
            ann_items,
            cand_lists,
            beam_width=int(annotation_beam_width),
            expand_k=int(annotation_beam_expand),
        )
        if chosen is None:
            continue

        chosen = _polish_assignment(ann_items, cand_lists, chosen, passes=int(annotation_polish_passes))
        if chosen is not None:
            break

    if chosen is None:
        if annotation_debug:
            print("[annotation] No feasible non-overlapping placement found; falling back to greedy.")
        chosen = {}
        used = []
        for i, it in enumerate(ann_items):
            cap_fb = None if annotation_point_collision_cap is None else int(annotation_point_collision_cap) + int(ntries)
            cand = _candidate_set_for_item(it, base_Lmax * max_growth, collision_cap=cap_fb)
            if not cand:
                continue
            picked = None
            for c in cand:
                if all(not _rects_overlap(c["rect_pad"], rr) for rr in used):
                    picked = c
                    break
            if picked is None:
                picked = cand[0]
            chosen[i] = picked
            used.append(picked["rect_pad"])

    # ============================================================
    # REQUIRED: remove one of overlapping boxes iteratively until none overlap
    # ============================================================

    def _drop_overlapping_boxes_until_clear(items, chosen_map):
        if not chosen_map:
            return chosen_map

        chosen_map = dict(chosen_map)

        def _importance(i):
            tt_i = int(items[i]["tt"])
            if tt_i == (T - 1):
                return 2  # keep End
            if tt_i == 0:
                return 1  # keep Start
            if tt_i in must_times:
                return 1  # treat must_include as important
            return 0

        def _keep_key(i):
            c = chosen_map[i]
            area = float(items[i]["w_px"] * items[i]["h_px"])
            # higher is better to keep
            return (int(_importance(i)), -float(c.get("cost", 0.0)), -area, -int(i))

        # Iteratively remove one from any overlapping pair until no overlap remains
        while True:
            inds = sorted(chosen_map.keys())
            removed_any = False
            for a in range(len(inds)):
                i = inds[a]
                ri = chosen_map[i]["rect"]
                for b in range(a + 1, len(inds)):
                    j = inds[b]
                    rj = chosen_map[j]["rect"]
                    if _rects_overlap(ri, rj):
                        drop = j if _keep_key(i) >= _keep_key(j) else i
                        chosen_map.pop(drop, None)
                        removed_any = True
                        break
                if removed_any:
                    break
            if not removed_any:
                break

        return chosen_map

    chosen = _drop_overlapping_boxes_until_clear(ann_items, chosen)

    # ============================================================
    # Draw annotations with chosen placements
    # ============================================================

    for i, it in enumerate(ann_items):
        if i not in chosen:
            continue
        c = chosen[i]

        k = int(it["k"])
        tt = int(it["tt"])
        bbox_kwargs = it["bbox_kwargs"]
        px_data, py_data = it["anchor_data"]

        shrinkB_px = float(it["r_anchor_px"]) + 0.5 * float(annotation_min_gap_px)
        shrinkB_pt = shrinkB_px * 72.0 / float(fig.dpi)

        ann = ax.annotate(
            it["text"],
            xy=(px_data, py_data),
            xycoords="data",
            xytext=(c["dx_pt"], c["dy_pt"]),
            textcoords="offset points",
            fontsize=int(annotation_fontsize),
            linespacing=0.90,
            ha=c["ha"],
            va=c["va"],
            color="black",
            bbox=bbox_kwargs,
            arrowprops=dict(
                arrowstyle="-",
                linestyle=(0, (3, 2)),
                color=(0, 0, 0, float(annotation_line_alpha)),
                lw=float(annotation_line_width),
                shrinkA=0.0,
                shrinkB=float(shrinkB_pt),
                connectionstyle="arc3,rad=0.0",
            ),
            zorder=6.0,
        )
        _set_gid(ann, f"annot:{k}:{tt}")

    return fig, ax
