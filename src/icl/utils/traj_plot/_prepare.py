"""Input conversion, use_mean averaging, 4-D expansion, flattening, and group resolution."""

import numpy as np

from icl.utils.traj_plot._helpers import (
    _to_np,
    _flatten_leading,
    _strip_b_suffix,
    _prettify_label,
    _normalize_labels,
)


def _prepare_inputs(
    task_vecs_over_all_time,
    final_task_vecs,
    r2_scores,
    task_labels,
    *,
    n_ood, n_minor, b_ood, b_minor, b_major,
    use_mean, ood_style_seed,
):
    """
    Convert raw inputs, handle use_mean averaging and 4-D expansion,
    flatten arrays, compute group indices, and resolve label colours.

    Returns a dict with keys:
        X, R2, labels, F, K, T, D,
        idx_major, idx_ood, idx_minor,
        orig_task_idx, major_bases, maj_names, major_base_to_color
    """
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

    # --- use_mean=True: average over B dimension ---
    if use_mean and Xnp.ndim == 4:
        K0, T0, B0, D0 = Xnp.shape
        Xnp = Xnp.mean(axis=2)

        R2np = np.asarray(R2_raw, dtype=float)
        if R2np.ndim == 2 and R2np.shape == (K0, T0):
            pass
        elif R2np.ndim == 3 and R2np.shape == (K0, B0, T0):
            R2np = R2np.mean(axis=1)
        elif R2np.ndim == 3 and R2np.shape == (K0, T0, B0):
            R2np = R2np.mean(axis=2)
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

    # --- use_mean=False 4-D expansion: (K,T,B,D) -> (K',T,D) ---
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
            R2np = np.repeat(R2np[:, None, :], repeats=B0, axis=1)
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

            Xsel = Xg[:, :, Bsel, :]
            Xflat = np.transpose(Xsel, (0, 2, 1, 3)).reshape(Kg * len(Bsel), T0, D0)

            R2sel = R2g[:, Bsel, :]
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

    # --- Flatten X / R2 ---
    X, _, _ = _flatten_leading(Xnp, name="task_vecs_over_all_time", tail_ndim=2)
    K, T, D = X.shape

    R2, _, _ = _flatten_leading(_to_np(R2_raw), name="r2_scores", tail_ndim=1)
    if R2.shape != (K, T):
        raise ValueError(f"r2_scores shape mismatch: expected {(K, T)}, got {R2.shape}")

    if labels is None:
        labels = _normalize_labels(task_labels, K)

    # --- Groups (if not already computed from 4-D expansion) ---
    if idx_major is None:
        n_minor = int(n_minor)
        if n_minor < 0 or n_minor > K:
            raise ValueError("n_minor out of range.")

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

    # --- Major base labels -> consistent coloring ---
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

    return dict(
        X=X, R2=R2, labels=labels, F=F, K=K, T=T, D=D,
        idx_major=idx_major, idx_ood=idx_ood, idx_minor=idx_minor,
        orig_task_idx=orig_task_idx,
        major_bases=major_bases, maj_names=maj_names,
        major_base_to_color=major_base_to_color,
    )
