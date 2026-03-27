"""Small, self-contained utility functions shared across the traj_plot package."""

import re
import numpy as np
import matplotlib.colors as mcolors


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


def _rigid_align_2d(X2, F2, *, target=(0.0, 0.0), use_rotation=True,
                    ref_dir=(1.0, 0.0), rotate_about="major1", eps=1e-12):
    """
    X2: (K,T,2), F2: (3,2)
    Returns aligned (X2a, F2a)

    - Always translates so F2[0] lands on `target`.
    - If use_rotation=True, rotates so (F2[1]-F2[0]) aligns to `ref_dir`.
    """
    X2 = np.asarray(X2, dtype=float)
    F2 = np.asarray(F2, dtype=float)
    tgt = np.asarray(target, dtype=float).reshape(1, 2)

    if rotate_about == "origin":
        pivot = np.zeros((1, 2), dtype=float)
    else:
        pivot = F2[0:1, :]  # major1

    if use_rotation:
        d = F2[1] - F2[0]
        rd = np.asarray(ref_dir, dtype=float).reshape(2,)
        nd = np.linalg.norm(d)
        nrd = np.linalg.norm(rd)

        if nd > eps and nrd > eps:
            d = d / nd
            rd = rd / nrd

            c = float(np.clip(d[0]*rd[0] + d[1]*rd[1], -1.0, 1.0))
            s = float(d[0]*rd[1] - d[1]*rd[0])

            R = np.array([[c, -s],
                          [s,  c]], dtype=float)

            Xr = (X2 - pivot.reshape(1, 1, 2)) @ R.T + pivot.reshape(1, 1, 2)
            Fr = (F2 - pivot) @ R.T + pivot
        else:
            Xr, Fr = X2, F2
    else:
        Xr, Fr = X2, F2

    shift = tgt - Fr[0:1, :]
    Xa = Xr + shift.reshape(1, 1, 2)
    Fa = Fr + shift
    return Xa, Fa


def _make_gid(gid_prefix, suffix):
    return f"{gid_prefix}:{suffix}"


def _set_gid(artist, gid_prefix, suffix):
    try:
        artist.set_gid(_make_gid(gid_prefix, suffix))
    except Exception:
        pass
    return artist


def _cleanup_previous_artists(ax_, gid_prefix):
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
            t = int(T_) + t
        if 0 <= t < int(T_):
            out.append(int(t))
    return out
