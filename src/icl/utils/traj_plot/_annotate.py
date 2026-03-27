"""Annotation placement solver (beam search over discrete candidates) and rendering."""

import numpy as np

from icl.utils.traj_plot._helpers import _set_gid, _strip_b_suffix


def _annotate_trajectories(
    fig, ax, *,
    X_proj, J, R2, sizes_area, F_proj,
    t_show, ann_times, must_times, T, K,
    idx_ood, ood_style, major_rgbs,
    orig_task_idx, labels, major_base_to_color,
    gid_prefix,
    mid_size_factor,
    show_final_task_vecs, final_marker_size,
    annotate, annotate_k,
    annotation_fontsize,
    annotation_box_alpha, annotation_box_pad, annotation_box_lw,
    annotation_line_alpha, annotation_line_width,
    annotation_pad_px,
    annotation_max_leader_px, annotation_min_gap_px,
    annotation_prefer_length_scale,
    annotation_sep_px,
    annotation_point_clear_px,
    annotation_point_collision_cap,
    annotation_point_penalty,
    annotation_density_weight,
    annotation_density_grid,
    annotation_density_blur_k,
    annotation_candidate_angles_deg,
    annotation_candidate_radii_n,
    annotation_beam_width, annotation_beam_expand,
    annotation_max_tries, annotation_max_leader_growth,
    annotation_polish_passes,
    annotation_debug,
):
    """
    Build annotation items, solve placement with beam search,
    and render annotation boxes with leader lines.

    Returns (fig, ax).
    """

    # --- geometry helpers ---

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

    # --- point cloud + density for label placement ---

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
    P_sample = (X_proj[:, t_show, :] + J[:, t_show, :]).reshape(-1, 2)
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

    # --- density integral ---

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

    # --- build annotation items ---

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

        idx_ood_set = set(idx_ood.tolist())
        times_ordered = sorted(ann_times)
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

                jxy = J[k, tt]
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

    if not ann_items:
        fig.tight_layout(pad=0.3)
        return fig, ax

    # --- measure label sizes ---

    def _measure_text_box_px(text, bbox_kw):
        tmp = ax.text(
            0.0,
            0.0,
            text,
            fontsize=int(annotation_fontsize),
            linespacing=0.90,
            ha="left",
            va="bottom",
            bbox=bbox_kw,
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

    # --- candidate generation ---

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

    # --- beam-search solver ---

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

    # --- solve placement ---

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

    # --- drop overlapping boxes ---

    def _drop_overlapping_boxes_until_clear(items, chosen_map):
        if not chosen_map:
            return chosen_map

        chosen_map = dict(chosen_map)

        def _importance(i):
            tt_i = int(items[i]["tt"])
            if tt_i == (T - 1):
                return 2
            if tt_i == 0:
                return 1
            if tt_i in must_times:
                return 1
            return 0

        def _keep_key(i):
            c = chosen_map[i]
            area = float(items[i]["w_px"] * items[i]["h_px"])
            return (int(_importance(i)), -float(c.get("cost", 0.0)), -area, -int(i))

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

    # --- render annotations ---

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
        _set_gid(ann, gid_prefix, f"annot:{k}:{tt}")

    fig.tight_layout(pad=0.3)
    return fig, ax
