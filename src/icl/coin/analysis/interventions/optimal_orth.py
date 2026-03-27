"""Find and ablate the optimal orth-complement direction for OOD loss.

Uses gradient ascent to find a low-rank subspace in the orthogonal
complement of the task subspace whose removal maximally degrades OOD
next-token prediction, then runs the causal intervention.
"""

import gc
from typing import Optional

import numpy as np
import torch

import icl.utils.notebook_utils as nu
from icl.coin.coin_ood_analysis import get_new_sampler
from icl.utils.logger import setup_logger
from icl.utils.orth_common import (
    ols_r2, trial_stats, iqr_err_norm, make_projection_hook,
    gen_and_cache, ORTH_COLORS, ORTH_BAR_WIDTH, ORTH_GROUP_STEP,
    orth_bar_offsets, ORTH_RANDOM_BAND_COLOR, ORTH_RANDOM_BAND_ALPHA,
    ORTH_RANDOM_BAND_HATCH, ORTH_REFERENCE_LINE_COLOR,
)

logger = setup_logger(__name__)


# ── Module-level utilities ─────────────────────────────────────────────

def _compute_unigram(samples, vocab_size, transform="clr", alpha=0.5):
    """Compute CLR / log1p / sqrt unigram features from token sequences."""
    onehot = torch.nn.functional.one_hot(
        samples.long(), num_classes=vocab_size,
    ).float()
    prefix_counts = onehot.cumsum(dim=1)
    prefix_len = torch.arange(
        1, samples.shape[1] + 1, device=samples.device, dtype=torch.float32,
    ).view(1, -1, 1)
    if transform == "clr":
        freq = (prefix_counts + alpha) / (prefix_len + alpha * vocab_size)
        logf = torch.log(freq.clamp_min(1e-12))
        return logf - logf.mean(dim=-1, keepdim=True)
    elif transform == "log1p":
        return torch.log1p(prefix_counts)
    freq = prefix_counts / prefix_len.clamp_min(1.0)
    return torch.sqrt(freq.clamp_min(0.0))


# ── Main entry point ───────────────────────────────────────────────────

def plot_optimal_orth_direction_across_layers_coin(
    exp_name: str,
    layers: Optional[list] = None,
    B: int = 64,
    n_opt_steps: int = 200,
    opt_lr: float = 0.01,
    opt_B: int = 32,
    patience: int = 30,
    grad_clip_norm: float = 1.0,
    n_samples_eval: int = 500,
    n_samples_probe: int = 1000,
    n_ood: int = 30,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    eval_positions: Optional[list] = None,
    center_task_vecs: bool = False,
    n_directions: int = 1,
    scale: float = 1.0,
    unigram_transform: str = "clr",
    unigram_alpha: float = 0.5,
    opt_target: str = "ood",
    extraction_point: str = "post_attn",
    probe_method: str = "ols",
    n_rand_int: int = 5,
    val_patience: int = 0,
    show_ylabel: bool = True,
    figsize: tuple = (14, 6),
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Sweep optimal-orth intervention across layers (monolithic).

    Loads the model **once**, fits probes for all layers in a single
    forward pass, caches eval sequences, and collects hidden states
    for all layers simultaneously.  Only the gradient-ascent
    optimisation and per-layer causal evaluations run in the per-layer
    loop.

    Parameters
    ----------
    opt_target : ``"ood"`` | ``"minor"``
        Which data to optimise on (gradient ascent target).
        ``"ood"`` finds directions whose removal hurts OOD prediction.
        ``"minor"`` finds directions whose removal hurts minor-task
        prediction.
    probe_method : ``"ols"`` | ``"averaging"``
        How to estimate task and token subspaces for the protected subspace.
        ``"ols"`` (default) — fits a joint OLS probe.
        ``"averaging"`` — collects token-conditioned (interventional) data
        and derives task/token vectors from ANOVA cell-means, the same
        approach used by ``plot_anova_separability`` / ``plot_averaging_r2``.

    Five figures:

    1. **Delta loss** bar chart (major + target) with information-gain
       reference lines.
    2. **Validation loss** history.
    3. **R² forward** (target): Unigram → V_opt projection.
    4. **R² reverse** (target): V_opt projection → Unigram.
    5. **Filtered barplot**: V_opt full vs unigram-explained causal
       effect (target).

    Returns ``(fig_delta, fig_loss, fig_r2_fwd, fig_r2_rev, fig_explained, all_results)``.
    """
    import matplotlib.pyplot as plt
    from icl.coin.analysis.probes import _collect_coin_probe_data, _fit_coin_probe

    # =================================================================
    #  Setup
    # =================================================================
    _, _, config = nu.load_everything("coin", exp_name)
    if step is None:
        step = config.training.num_epochs
    D = int(config.model.emb_dim)
    vocab_size = int(config.vocab_size)
    if layers is None:
        layers = list(range(config.model.num_layers))

    sampler_major, _ = get_new_sampler(exp_name, n_minor=0, n_ood=0)
    sampler_ood, _ = get_new_sampler(exp_name, n_minor=0, n_ood=n_ood)
    sampler_minor, _ = get_new_sampler(exp_name, n_minor=1_000_000, n_ood=0)
    sampler_tgt = sampler_minor if opt_target == "minor" else sampler_ood
    tgt_label = "Minor" if opt_target == "minor" else "OOD"
    tgt_abbr  = "Min."  if opt_target == "minor" else "OOD"
    seq_len = sampler_major.seq_len

    if fit_positions is None:
        upper = seq_len - 1 if probe_method == "averaging" else seq_len
        fit_positions = list(range(100, upper))
    if eval_positions is None:
        eval_positions = list(range(seq_len))

    # =================================================================
    #  Phase 1: Estimate task/token subspaces (all layers, one pass)
    # =================================================================
    n_major_tasks = int(sampler_major.n_major_tasks)

    if probe_method == "averaging":
        # ANOVA cell-mean approach: collect token-conditioned (interventional)
        # hidden states and derive task/token vectors from two-way ANOVA
        # marginals — same method as plot_anova_separability / plot_averaging_r2.
        logger.info("[opt-dir] Phase 1: Collecting token-conditioned hiddens (ANOVA) ...")
        from icl.coin.analysis._helpers import get_token_conditioned_hiddens_coin

        all_hiddens_anova, anova_info = get_token_conditioned_hiddens_coin(
            exp_name, layers=layers, batch_size=B,
            positions_of_interest=fit_positions,
            step=step, n_minor=0,
            extraction_point=extraction_point,
        )
        # all_hiddens_anova: (L, n_positions, V_max, n_major, B, D)
        pos_to_anova = {p: i for i, p in enumerate(anova_info["positions"])}
        n_uniq = anova_info["n_unique_tokens"]
        V_max = all_hiddens_anova.shape[2]

        probes = {}
        for li, l in enumerate(layers):
            parts = []
            for p in fit_positions:
                if p not in pos_to_anova:
                    continue
                pi = pos_to_anova[p]
                V_p = n_uniq.get(p, V_max)
                parts.append(
                    all_hiddens_anova[li, pi, :V_p, :n_major_tasks].float().mean(dim=-2)
                )  # (V_p, K, D)

            if not parts:
                raise ValueError(
                    "[opt-dir] probe_method='averaging': none of fit_positions "
                    "found in token-conditioned hiddens."
                )

            min_V = min(p.shape[0] for p in parts)
            parts = [p[:min_V] for p in parts]

            demeaned = [cm - cm.mean(dim=(0, 1), keepdim=True) for cm in parts]
            pooled = torch.stack(demeaned, dim=0).mean(dim=0)  # (V, K, D)
            grand = pooled.mean(dim=(0, 1))
            task_vecs = pooled.mean(dim=0) - grand   # (K, D)
            token_vecs = pooled.mean(dim=1) - grand  # (V, D)

            probes[l] = {
                "model_weight": task_vecs,
                "token_weight": token_vecs,
            }
        del all_hiddens_anova

    else:
        logger.info("[opt-dir] Phase 1: Fitting OLS probes ...")
        probe_data = _collect_coin_probe_data(
            exp_name, layers, B=B, n_samples=fit_n_samples,
            step=step, n_minor=-1, positions=fit_positions,
            sample_mode="major",
            extraction_point=extraction_point,
        )
        probes = {}
        for l in layers:
            probes[l] = _fit_coin_probe(
                probe_data["hiddens_by_layer"][l],
                probe_data["posteriors_all"],
                probe_data["real_tokens_all"],
                layer=l,
                n_tasks=probe_data["n_tasks"],
                positions=probe_data["positions"],
                skip_baselines=True,
                print_summary=False,
            )
        del probe_data

    # =================================================================
    #  Phase 2: Load model, cache eval data
    # =================================================================
    logger.info("[opt-dir] Phase 2: Loading model & caching eval data ...")
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    model.requires_grad_(False)
    device = config.device

    cache_maj = gen_and_cache(model, sampler_major, "major", n_samples_eval, B, device)
    cache_ood = gen_and_cache(model, sampler_ood, "minor", n_samples_eval, B, device)
    cache_minor = gen_and_cache(model, sampler_minor, "minor", n_samples_eval, B, device)
    cache_tgt = cache_minor if opt_target == "minor" else cache_ood

    # =================================================================
    #  Phase 3: Collect hidden states + unigram (all layers, one pass)
    # =================================================================
    logger.info("[opt-dir] Phase 3: Collecting hiddens + unigram ...")
    eval_pos_idx = torch.tensor(eval_positions, device=device, dtype=torch.long)

    # Feature layout: [unigram_clr (V) | one_hot_x_t (V-1) | position (1)]
    # Unigram CLR sums to zero, so full V dimensions are kept (no redundancy for OLS).
    # One-hot of current token is reference-encoded (drop last class) to remove collinearity.
    _enrich_feat_dim = vocab_size + (vocab_size - 1) + 1
    _coin_feat_groups = {}
    _s = 0
    for _name, _k in [
        ("unigram_clr",  vocab_size),
        ("one_hot_x_t",  vocab_size - 1),
        ("position",     1),
    ]:
        _coin_feat_groups[_name] = slice(_s, _s + _k)
        _s += _k

    def _collect_all_h_and_uni(sampler, gen_mode, n):
        h_acc = {l: [] for l in layers}
        uni_acc = []
        enrich_acc = []
        for _ in range(max(1, (n + B - 1) // B)):
            g = sampler.generate(mode=gen_mode, task=None, num_samples=B, epochs=1)
            s = (g[0] if isinstance(g, (tuple, list)) else g)
            if s.dim() == 3:
                s = s.squeeze(0)
            s = s.to(device)
            Bs, L = s.shape
            caches = {}
            handles = []
            for ll in layers:
                def _hook(mod, inp, out, _l=ll):
                    h = out if torch.is_tensor(out) else out[0]
                    caches[_l] = h.index_select(1, eval_pos_idx).detach()
                _cah_tgt = (
                    model.layers[ll] if extraction_point == "post_mlp"
                    else model.layers[ll].attn_block
                )
                handles.append(_cah_tgt.register_forward_hook(_hook))
            with torch.no_grad():
                model(s)
            for hh in handles:
                hh.remove()
            for ll in layers:
                h_acc[ll].append(caches[ll].cpu())
            uni = _compute_unigram(s, vocab_size, unigram_transform, unigram_alpha)
            one_hot = torch.nn.functional.one_hot(
                s.long(), num_classes=vocab_size,
            ).float()[..., :-1]  # (Bs, L, vocab_size-1)  reference-encoded
            pos_feat = (
                torch.arange(L, device=device, dtype=torch.float32)
                .view(1, L, 1).expand(Bs, -1, 1)
            ) / L  # (Bs, L, 1)
            enrich = torch.cat([uni, one_hot, pos_feat], dim=-1)  # (Bs, L, V + V-1 + 1)
            uni_acc.append(uni.index_select(1, eval_pos_idx).cpu())
            enrich_acc.append(enrich.index_select(1, eval_pos_idx).cpu())
            del s, caches, uni, one_hot, pos_feat, enrich
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return (
            {ll: torch.cat(h_acc[ll]).reshape(-1, D).float() for ll in layers},
            torch.cat(uni_acc).reshape(-1, vocab_size).float(),
            torch.cat(enrich_acc).reshape(-1, _enrich_feat_dim).float(),
        )

    h_tgt, uni_tgt, enrich_tgt = _collect_all_h_and_uni(sampler_tgt, "minor", n_samples_probe)

    # =================================================================
    #  Shared helpers
    # =================================================================
    ce_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def _eval_hooked(layer_idx, P, cached):
        """Run intervened forward pass using cached baseline logits.

        Returns (baseline_mean, intervened_mean, loss_at_pos0, delta_per_batch).
        delta_per_batch is a list of (intervened - baseline) averaged over
        eval_positions for each cached batch — used for IQR computation.
        """
        seqs, base_logits = cached
        b_all, i_all, bl_at_0 = [], [], []
        batch_deltas = []
        for s_cpu, bl_cpu in zip(seqs, base_logits):
            s = s_cpu.to(device)

            _eh_tgt = (
                model.layers[layer_idx] if extraction_point == "post_mlp"
                else model.layers[layer_idx].attn_block
            )
            handle = _eh_tgt.register_forward_hook(make_projection_hook(P, scale))
            try:
                with torch.no_grad():
                    li = model(s)
            finally:
                handle.remove()
            bl = bl_cpu.to(device)
            if s.shape[1] > 1:
                bl_at_0.append(
                    ce_fn(bl[:, 0], s[:, 1].long()).mean().item()
                )
            b_batch, i_batch = [], []
            for p in eval_positions:
                if p + 1 >= s.shape[1]:
                    continue
                tgt = s[:, p + 1].long()
                bv = ce_fn(bl[:, p], tgt).mean().item()
                iv = ce_fn(li[:, p], tgt).mean().item()
                b_all.append(bv)
                i_all.append(iv)
                b_batch.append(bv)
                i_batch.append(iv)
            if b_batch:
                batch_deltas.append(
                    float(np.mean(i_batch)) - float(np.mean(b_batch))
                )
            del s, bl, li
        ba = float(np.mean(b_all))
        ia = float(np.mean(i_all))
        l0 = float(np.mean(bl_at_0)) if bl_at_0 else float("nan")
        return ba, ia, l0, batch_deltas

    val_seq = cache_tgt[0][0].to(device)
    ce_mean = torch.nn.CrossEntropyLoss()

    def _val_loss(layer_idx, V_mat):
        """CE loss on cached validation batch with intervention h' = h - s*V*V^T*h."""
        P = V_mat @ V_mat.T

        _vl_tgt = (
            model.layers[layer_idx] if extraction_point == "post_mlp"
            else model.layers[layer_idx].attn_block
        )
        hnd = _vl_tgt.register_forward_hook(make_projection_hook(P, scale))
        try:
            with torch.no_grad():
                logits_v = model(val_seq)
        finally:
            hnd.remove()
        acc, cnt = 0.0, 0
        for p in eval_positions:
            if p + 1 >= val_seq.shape[1]:
                continue
            acc += ce_mean(logits_v[:, p], val_seq[:, p + 1].long()).item()
            cnt += 1
        return acc / cnt if cnt > 0 else float("nan")

    def _optimise_v(layer_idx, U_orth_l, val_every=20):
        """Gradient ascent for V_opt in span(U_orth_l).

        Early stopping:
          - Training patience  (``patience``):   stop after this many consecutive
            training steps with no improvement in smoothed training loss.
          - Validation patience (``val_patience``): stop after this many consecutive
            validation evaluations with no improvement in validation CE loss.
            The snapshot with the best validation loss is returned as V_opt.
            Set val_patience=0 to disable (default).
        """
        orth_dim = U_orth_l.shape[1]
        W_p = torch.randn(orth_dim, n_directions, device=device)
        W_p = (W_p / W_p.norm(dim=0, keepdim=True)).detach().requires_grad_(True)
        opt_v = torch.optim.Adam([W_p], lr=opt_lr)
        ema_decay = 0.995
        ema_W = W_p.detach().clone()
        # Training-loss best snapshot
        best_ema      = ema_W.clone()
        best_loss     = -float("inf")
        stale         = 0
        smoothed      = None
        # Validation-loss best snapshot
        best_val_loss = -float("inf")
        best_ema_val  = None
        val_stale     = 0
        val_history   = []

        for step_i in range(n_opt_steps):
            g = sampler_tgt.generate(
                mode="minor", task=None, num_samples=opt_B, epochs=1,
            )
            s = (g[0] if isinstance(g, (tuple, list)) else g)
            if s.dim() == 3:
                s = s.squeeze(0)
            s = s.to(device)

            def _hook(mod, inp, out):
                h = out if torch.is_tensor(out) else out[0]
                Vq, _ = torch.linalg.qr(U_orth_l @ W_p)
                hm = h - scale * (h @ Vq @ Vq.T)
                return hm if torch.is_tensor(out) else (hm,) + out[1:]

            _ov_tgt = (
                model.layers[layer_idx] if extraction_point == "post_mlp"
                else model.layers[layer_idx].attn_block
            )
            hnd = _ov_tgt.register_forward_hook(_hook)
            try:
                logits = model(s)
            finally:
                hnd.remove()

            acc = torch.tensor(0.0, device=device)
            cnt = 0
            for p in eval_positions:
                if p + 1 >= s.shape[1]:
                    continue
                acc = acc + ce_mean(logits[:, p], s[:, p + 1].long())
                cnt += 1
            if cnt == 0:
                del s, logits
                continue

            avg = acc / cnt
            cl = avg.item()
            smoothed = cl if smoothed is None else 0.1 * cl + 0.9 * smoothed

            opt_v.zero_grad()
            (-avg).backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([W_p], max_norm=grad_clip_norm)
            opt_v.step()

            with torch.no_grad():
                ema_W.mul_(ema_decay).add_(W_p.detach(), alpha=1 - ema_decay)
            if smoothed > best_loss:
                best_loss = smoothed
                best_ema = ema_W.clone()
                stale = 0
            else:
                stale += 1

            # Validation evaluation
            if step_i % val_every == 0 or step_i == n_opt_steps - 1:
                with torch.no_grad():
                    V_cur, _ = torch.linalg.qr(U_orth_l @ best_ema)
                vl = _val_loss(layer_idx, V_cur)
                val_history.append((step_i, vl))
                if val_patience > 0:
                    if vl > best_val_loss:
                        best_val_loss = vl
                        best_ema_val  = best_ema.clone()
                        val_stale     = 0
                    else:
                        val_stale += 1
                    if val_stale >= val_patience:
                        logger.info(
                            f"[opt-dir] layer {layer_idx}: val-patience "
                            f"({val_patience} evals) reached at step {step_i}"
                        )
                        break

            del s, logits, acc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if patience > 0 and stale >= patience:
                break

        final_ema = (
            best_ema_val if (val_patience > 0 and best_ema_val is not None)
            else best_ema
        )
        with torch.no_grad():
            V_opt, _ = torch.linalg.qr(U_orth_l @ final_ema)
        return V_opt, val_history

    # =================================================================
    #  Phase 4: Per-layer loop
    # =================================================================
    N_RAND_R2 = 3
    all_results = {}

    for l in layers:
        logger.info(f"[opt-dir] Layer {l} ...")

        # --- (a) Build orth complement from probe ---
        W_fit = probes[l]["model_weight"].float()
        if center_task_vecs:
            task_vecs = W_fit - W_fit.mean(0, keepdim=True)
        else:
            task_vecs = W_fit.clone()
        _, S_tv, Vt_tv = torch.linalg.svd(task_vecs, full_matrices=False)
        rank = int((S_tv > 1e-6 * S_tv[0]).sum().item())
        basis = Vt_tv[:rank].T
        P_orth = (torch.eye(D) - basis @ basis.T).to(device)
        evals, evecs = torch.linalg.eigh(P_orth)
        U_orth_l = evecs[:, evals > 0.5].to(device)
        orth_dim = U_orth_l.shape[1]

        # --- (b) Optimise V_opt ---
        V_opt, val_hist = _optimise_v(l, U_orth_l)
        V_cpu = V_opt.cpu().float()

        U_orth_cpu = U_orth_l.cpu().float()

        # --- (c) R² diagnostics (target data) ---
        proj_tgt = h_tgt[l] @ V_cpu

        # Marginal R² per individual feature group (feat_i → V_opt)
        feat_r2_marginal = {
            name: ols_r2(enrich_tgt[:, sl], proj_tgt)
            for name, sl in _coin_feat_groups.items()
        }
        # Combined (all enriched features together) R²
        r2_combined = ols_r2(enrich_tgt, proj_tgt)

        # Legacy scalar R² values kept for backward compatibility
        r2_u2v = feat_r2_marginal["unigram_clr"]
        r2_v2u = ols_r2(proj_tgt, uni_tgt)
        r2_p2v = r2_combined
        r2_v2p = r2_combined

        # --- (d) Combined-feature-explained decomposition ---
        proj_c = proj_tgt - proj_tgt.mean(0, keepdim=True)
        total_var = (proj_c ** 2).sum().item()
        uni_explained = {}
        for fname, feat in [("unigram_clr", uni_tgt), ("enriched", enrich_tgt)]:
            Nf = feat.shape[0]
            X_aug = torch.cat([feat, torch.ones(Nf, 1)], 1)
            W_fit_exp = torch.linalg.pinv(X_aug) @ proj_tgt
            Y_hat = X_aug @ W_fit_exp
            r2 = 1.0 - ((proj_tgt - Y_hat) ** 2).sum().item() / total_var \
                if total_var > 0 else 0.0
            _, Sp, Vtp = torch.linalg.svd(
                Y_hat - Y_hat.mean(0, keepdim=True), full_matrices=False,
            )
            frac = Sp ** 2 / max(total_var, 1e-12)
            rk = int((frac > 0.005).sum().item())
            V_expl = V_cpu @ Vtp[:rk].T if rk > 0 else torch.empty(D, 0)
            uni_explained[fname] = {
                "explained_rank": rk, "r2_feat_to_proj": r2,
                "V_explained": V_expl,
            }

        # --- (e) Causal evaluation (uses cached baseline logits) ---
        P_v = (V_opt @ V_opt.T).to(device)
        ba_maj, ia_maj, l0_maj, bd_maj = _eval_hooked(l, P_v, cache_maj)
        ba_ood, ia_ood, l0_ood, bd_ood = _eval_hooked(l, P_v, cache_ood)
        ba_minor, ia_minor, l0_minor, bd_minor = _eval_hooked(l, P_v, cache_minor)

        # Random same-rank subspace baseline (major, ood, minor)
        rand_acc = {"major": [], "ood": [], "minor": []}
        rk_r = min(n_directions, orth_dim)
        for _ in range(n_rand_int):
            Vr, _ = torch.linalg.qr(
                U_orth_cpu @ torch.randn(orth_dim, rk_r),
            )
            Pr = (Vr @ Vr.T).to(device)
            for key, cache in [
                ("major", cache_maj), ("ood", cache_ood), ("minor", cache_minor),
            ]:
                bo, io, _, _ = _eval_hooked(l, Pr, cache)
                rand_acc[key].append(io - bo)

        rand_stats = {k: trial_stats(v) for k, v in rand_acc.items()}
        rand_delta_tgt = rand_stats["minor" if opt_target == "minor" else "ood"]["mean"]

        for info in uni_explained.values():
            if info["explained_rank"] == 0:
                info["delta_tgt"] = 0.0
                info["delta_ood"] = 0.0
                continue
            Ve = info["V_explained"].to(device)
            P_e = Ve @ Ve.T
            bo, io, _, _ = _eval_hooked(l, P_e, cache_tgt)
            info["delta_tgt"] = io - bo
            bo_o, io_o, _, _ = _eval_hooked(l, P_e, cache_ood)
            info["delta_ood"] = io_o - bo_o
            del Ve, P_e

        # --- (f) Store ---
        d_maj = ia_maj - ba_maj
        d_ood = ia_ood - ba_ood
        d_minor = ia_minor - ba_minor
        d_tgt = d_minor if opt_target == "minor" else d_ood
        all_results[l] = {
            "opt_target": opt_target,
            "tgt_label": tgt_label,
            "baseline_loss_major": ba_maj,
            "delta_loss_major": d_maj,
            "baseline_loss_ood": ba_ood,
            "delta_loss_ood": d_ood,
            "baseline_loss_minor": ba_minor,
            "delta_loss_minor": d_minor,
            "baseline_loss_tgt": ba_minor if opt_target == "minor" else ba_ood,
            "delta_loss_tgt": d_tgt,
            "baseline_loss_at_0_major": l0_maj,
            "baseline_loss_at_0_ood": l0_ood,
            "baseline_loss_at_0_minor": l0_minor,
            "baseline_loss_at_0_tgt": l0_minor if opt_target == "minor" else l0_ood,
            "val_history": val_hist,
            "feat_r2_marginal": feat_r2_marginal,
            "r2_combined_tgt": r2_combined,
            "uni_to_proj_r2_tgt": r2_u2v,
            "proj_to_uni_r2_tgt": r2_v2u,
            "pred_to_proj_r2_tgt": r2_p2v,
            "proj_to_pred_r2_tgt": r2_v2p,
            "rand_int_delta_tgt": rand_delta_tgt,
            "rand_stats": rand_stats,
            "delta_per_batch_major": bd_maj,
            "delta_per_batch_ood": bd_ood,
            "delta_per_batch_minor": bd_minor,
            "uni_explained": {
                fn: {k: v for k, v in info.items() if k != "V_explained"}
                for fn, info in uni_explained.items()
            },
            "task_subspace_rank": rank,
            "n_directions": n_directions,
        }
        logger.info(
            f"  layer {l}: d_maj={d_maj:.4f}, d_ood={d_ood:.4f}, "
            f"d_minor={d_minor:.4f}, R²(u→V)={r2_u2v:.3f}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =================================================================
    #  Cleanup
    # =================================================================
    model.cpu()
    del model, h_tgt, uni_tgt, enrich_tgt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # =================================================================
    #  Plots
    # =================================================================
    x = np.arange(len(layers))

    # ---- 1. Delta loss bar chart — normalized by ICL gain ----
    # 3 solid bars per layer (Major / OOD / Minor), each Δ𝓛/g × 100 (%).
    # Black error bars = IQR across eval batches (also normalized).
    # Shaded band = random same-rank subspace (OOD, normalized).

    # Gains: pos-0 baseline minus mean baseline (use first layer; consistent across layers)
    ref = all_results[layers[0]]
    g_maj   = ref["baseline_loss_at_0_major"]  - ref["baseline_loss_major"]
    g_ood   = ref["baseline_loss_at_0_ood"]    - ref["baseline_loss_ood"]
    g_minor = ref["baseline_loss_at_0_minor"]  - ref["baseline_loss_minor"]
    g_tgt   = g_minor if opt_target == "minor" else g_ood

    # Normalized deltas: Δ𝓛 / g × 100 (% of ICL gain disrupted)
    norm_maj   = [all_results[l]["delta_loss_major"]  / g_maj   * 100 for l in layers]
    norm_ood   = [all_results[l]["delta_loss_ood"]    / g_ood   * 100 for l in layers]
    norm_minor = [all_results[l]["delta_loss_minor"]  / g_minor * 100 for l in layers]

    MC_C = orth_bar_offsets(has_minor=True)

    fig_delta, ax = plt.subplots(figsize=figsize, dpi=150)

    # Random baseline band: OOD rand normalized by g_ood (primary target metric)
    rand_ood_q75_norm = max(
        all_results[l]["rand_stats"]["ood"]["q75"] / g_ood * 100 for l in layers
    )
    ax.axhspan(0, rand_ood_q75_norm, color=ORTH_RANDOM_BAND_COLOR,
               alpha=ORTH_RANDOM_BAND_ALPHA, zorder=1,
               hatch=ORTH_RANDOM_BAND_HATCH, label="Random")
    ax.axhline(rand_ood_q75_norm, color=ORTH_REFERENCE_LINE_COLOR,
               lw=1.2, ls="-", zorder=2, alpha=0.85)

    for mode, norm_vals, (key_pb, g_m), label in [
        ("maj",   norm_maj,   ("delta_per_batch_major", g_maj),   "Maj."),
        ("ood",   norm_ood,   ("delta_per_batch_ood",   g_ood),   "OOD"),
        ("minor", norm_minor, ("delta_per_batch_minor", g_minor), "Min."),
    ]:
        c  = ORTH_COLORS[mode]
        xm = x + MC_C[mode]
        lo, hi = iqr_err_norm(all_results, layers, key_pb, g_m)
        ax.bar(xm, norm_vals, ORTH_BAR_WIDTH, color=c, linewidth=0, zorder=3, label=label)
        ax.errorbar(xm, norm_vals, yerr=[lo, hi], fmt="none",
                    ecolor="black", elinewidth=0.9, capsize=3, capthick=0.9, zorder=5)

    ax.axhline(100, color="grey", ls="--", lw=1.0, alpha=0.55,
               label="100%")

    ax.set_xlabel("Layer", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Fraction of ICL gain disrupted (%)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.5, color="grey")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=5, framealpha=0.9, edgecolor="lightgrey",
              columnspacing=0.5, handlelength=1.0, handletextpad=0.3, borderpad=0.4)
    plt.tight_layout(pad=2.0)
    if save_path:
        fig_delta.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig_delta)

    # ── Printed Table 1: V_opt Intervention (per-layer normalized) ────────
    _g_tgt_lab = "Min." if opt_target == "minor" else "OOD"
    _W = 7
    _hd = (f"  {'Layer':>5}  {'Maj.%':>{_W}}  {'OOD%':>{_W}}  {'Min.%':>{_W}}"
           f"  {'Rand.%':>{_W}}  |  {'Maj.Δ':>{_W}}  {'OOD Δ':>{_W}}  {'Min.Δ':>{_W}}")
    _ln = "  " + "─" * (len(_hd) - 2)
    print(f"\n  V_opt Intervention — Δ𝓛/g (% of ICL gain)  [CE nats]")
    print(_ln); print(_hd); print(_ln)
    for _l in layers:
        _r = all_results[_l]
        _nm = _r["delta_loss_major"] / g_maj   * 100
        _no = _r["delta_loss_ood"]   / g_ood   * 100
        _nn = _r["delta_loss_minor"]  / g_minor * 100
        _nr = _r["rand_stats"][_r["opt_target"]]["mean"] / g_tgt * 100
        print(f"  {_l:>5}  {_nm:>{_W}.1f}%  {_no:>{_W}.1f}%  {_nn:>{_W}.1f}%"
              f"  {_nr:>{_W}.1f}%  |  "
              f"{_r['delta_loss_major']:>{_W}.4f}  {_r['delta_loss_ood']:>{_W}.4f}"
              f"  {_r['delta_loss_minor']:>{_W}.4f}")
    print(_ln)
    print(f"  {'Gain':>5}  {'':>{_W}}   {'':>{_W}}   {'':>{_W}}   {'':>{_W}}   |  "
          f"{g_maj:{_W}.4f}  {g_ood:{_W}.4f}  {g_minor:{_W}.4f}")
    print()

    # ── Printed Table 2: Layer-averaged ───────────────────────────────────
    _rand_norm = [
        all_results[l]["rand_stats"][all_results[l]["opt_target"]]["mean"] / g_tgt * 100
        for l in layers
    ]
    _mean_m = float(np.mean(norm_maj));   _std_m = float(np.std(norm_maj))
    _mean_o = float(np.mean(norm_ood));   _std_o = float(np.std(norm_ood))
    _mean_n = float(np.mean(norm_minor)); _std_n = float(np.std(norm_minor))
    _mean_r = float(np.mean(_rand_norm)); _std_r = float(np.std(_rand_norm))
    _mean_dm = float(np.mean([all_results[l]["delta_loss_major"] for l in layers]))
    _mean_do = float(np.mean([all_results[l]["delta_loss_ood"]   for l in layers]))
    _mean_dn = float(np.mean([all_results[l]["delta_loss_minor"]  for l in layers]))
    _WA = 12
    _sep = "  " + "─" * 57
    print(_sep)
    print(f"  Layer-averaged (mean ± std across {len(layers)} layers) — CE nats")
    print(_sep)
    print(f"  {'Mode':<6}  {'Δ/g (%)':>{_WA}}  {'Raw Δ':>{_WA}}  {'g (CE nats)':>{_WA}}")
    print(_sep)
    print(f"  {'Maj.':<6}  {_mean_m:>7.1f}±{_std_m:<4.1f}  {_mean_dm:{_WA}.4f}  {g_maj:{_WA}.4f}")
    print(f"  {'OOD':<6}  {_mean_o:>7.1f}±{_std_o:<4.1f}  {_mean_do:{_WA}.4f}  {g_ood:{_WA}.4f}")
    print(f"  {'Min.':<6}  {_mean_n:>7.1f}±{_std_n:<4.1f}  {_mean_dn:{_WA}.4f}  {g_minor:{_WA}.4f}")
    print(f"  {'Rand.':<6}  {_mean_r:>7.1f}±{_std_r:<4.1f}  {'---':>{_WA}}  {'---':>{_WA}}")
    print(_sep)
    print()
    # ─────────────────────────────────────────────────────────────────────

    # ---- 2. Validation loss history ----
    cmap = plt.cm.tab10
    fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
    for i, l in enumerate(layers):
        vh = all_results[l]["val_history"]
        steps_v = [s for s, _ in vh]
        losses_v = [v for _, v in vh]
        ax_loss.plot(
            steps_v, losses_v,
            label=f"Layer {l}", color=cmap(i % 10), alpha=0.85,
        )
    ax_loss.set_xlabel("Optimisation Step", fontsize=14)
    ax_loss.set_ylabel(f"{tgt_label} Val Loss (intervened)", fontsize=14)
    ax_loss.set_title("", fontsize=18)
    ax_loss.legend(fontsize=12)
    ax_loss.grid(alpha=0.3)
    ax_loss.tick_params(labelsize=12)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig_loss)

    # ---- 3. R² plot — individual features + dashed combined ----
    _COIN_FEAT_DISPLAY = [
        ("unigram_clr",  "Unigram CLR",   "o-",  "#FF9800"),
        ("one_hot_x_t",  "Current token", "s-",  "#4CAF50"),
        ("position",     "Position",      "^-",  "#9C27B0"),
    ]
    fig_r2_fwd, ax_r2 = plt.subplots(figsize=(6, 4.5), dpi=150)
    for name, label, style, color in _COIN_FEAT_DISPLAY:
        vals = [all_results[l]["feat_r2_marginal"].get(name, float("nan"))
                for l in layers]
        ax_r2.plot(layers, vals, style, label=label, color=color, lw=2, ms=6)
    combined_r2_vals = [all_results[l]["r2_combined_tgt"] for l in layers]
    ax_r2.plot(layers, combined_r2_vals, ls="--", color="black", lw=1.8,
               label="Combined", zorder=2)
    ax_r2.set_xlabel("Layer", fontsize=9)
    ax_r2.set_ylabel(r"$R^2$", fontsize=9)
    ax_r2.set_xticks(layers)
    ax_r2.tick_params(labelsize=8)
    ax_r2.legend(fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18),
                 framealpha=0.9, edgecolor="lightgrey",
                 columnspacing=0.5, handlelength=1.0, handletextpad=0.3, borderpad=0.4)
    ax_r2.yaxis.grid(True, alpha=0.25, linewidth=0.5, color="grey")
    ax_r2.set_axisbelow(True)
    ax_r2.spines["top"].set_visible(False)
    ax_r2.spines["right"].set_visible(False)
    ax_r2.set_ylim(-0.05, 1.05)
    plt.tight_layout(pad=2.0)
    # fig_r2_rev kept for API compatibility
    fig_r2_rev = fig_r2_fwd
    if show:
        plt.figure(fig_r2_fwd.number); plt.show()
    else:
        plt.close(fig_r2_fwd)

    # ---- 4. Filtered barplot: V_opt vs Prediction-explained ----
    # 4 flush bars per layer, interleaved: [V_opt OOD | Pred OOD | V_opt tgt | Pred tgt]
    # Prediction bars share the V_opt colour at lower alpha — no gaps.
    C_OOD_e = ORTH_COLORS["ood"]
    C_TGT_e = ORTH_COLORS["minor"] if opt_target == "minor" else ORTH_COLORS["ood"]
    ALPHA_PRED_e = 0.42

    bw_p_e      = 0.17
    total_w_bi_e = 4 * bw_p_e      # = 0.68 (no gaps)
    left_e       = -total_w_bi_e / 2

    xo_vopt_ood_e = left_e + 0.5 * bw_p_e
    xo_pred_ood_e = left_e + 1.5 * bw_p_e
    xo_vopt_tgt_e = left_e + 2.5 * bw_p_e
    xo_pred_tgt_e = left_e + 3.5 * bw_p_e

    # Use the combined (enriched) feature-explained subspace for filtering
    fn = "enriched"
    vals_ood_e = [all_results[l]["uni_explained"][fn]["delta_ood"] / g_ood * 100
                  for l in layers]
    vals_tgt_e = [all_results[l]["uni_explained"][fn]["delta_tgt"] / g_tgt * 100
                  for l in layers]

    fig_explained, ax_e = plt.subplots(figsize=figsize, dpi=150)

    ax_e.bar(x + xo_vopt_ood_e,
             [all_results[l]["delta_loss_ood"] / g_ood * 100 for l in layers],
             bw_p_e, label=r"$V_{\rm opt}$ (OOD)",
             color=C_OOD_e, linewidth=0, zorder=3)

    ax_e.bar(x + xo_pred_ood_e, vals_ood_e, bw_p_e,
             label="Filtered (OOD)",
             color=C_OOD_e, alpha=ALPHA_PRED_e, linewidth=0, zorder=3)

    ax_e.bar(x + xo_vopt_tgt_e,
             [all_results[l]["delta_loss_tgt"] / g_tgt * 100 for l in layers],
             bw_p_e, label=rf"$V_{{\rm opt}}$ ({tgt_abbr})",
             color=C_TGT_e, linewidth=0, zorder=3)

    ax_e.bar(x + xo_pred_tgt_e, vals_tgt_e, bw_p_e,
             label=f"Filtered ({tgt_abbr})",
             color=C_TGT_e, alpha=ALPHA_PRED_e, linewidth=0, zorder=3)

    ax_e.axhline(100, color="grey", ls="--", lw=1.0, alpha=0.55,
                 label="100%")

    ax_e.set_xlabel("Layer", fontsize=9)
    if show_ylabel:
        ax_e.set_ylabel("Fraction of ICL gain disrupted (%)", fontsize=9)
    ax_e.set_xticks(x)
    ax_e.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax_e.tick_params(axis="y", labelsize=8)
    ax_e.yaxis.grid(True, alpha=0.25, linewidth=0.5, color="grey")
    ax_e.set_axisbelow(True)
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)
    ax_e.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14),
                ncol=5, framealpha=0.9, edgecolor="lightgrey",
                columnspacing=0.5, handlelength=1.0, handletextpad=0.3, borderpad=0.4)
    plt.tight_layout(pad=2.0)
    if show:
        plt.show()
    else:
        plt.close(fig_explained)

    # ── Printed Table 5: V_opt & Prediction ──────────────────────────────
    _W5 = 9
    _hd5 = (f"  {'Layer':>5}  {'Vopt OOD%':>{_W5}}  {'Pred OOD%':>{_W5}}  "
            f"{'Vopt '+tgt_abbr+'%':>{_W5}}  {'Pred '+tgt_abbr+'%':>{_W5}}")
    _ln5 = "  " + "─" * (len(_hd5) - 2)
    print(f"\n  V_opt & Prediction — Δ𝓛/g (% of ICL gain)  (feat: {fn})")
    print(_ln5); print(_hd5); print(_ln5)
    for _l in layers:
        _r = all_results[_l]
        _vo = _r["delta_loss_ood"] / g_ood * 100
        _po = _r["uni_explained"][fn]["delta_ood"] / g_ood * 100
        _vt = _r["delta_loss_tgt"] / g_tgt * 100
        _pt = _r["uni_explained"][fn]["delta_tgt"] / g_tgt * 100
        print(f"  {_l:>5}  {_vo:>{_W5}.1f}%  {_po:>{_W5}.1f}%  "
              f"{_vt:>{_W5}.1f}%  {_pt:>{_W5}.1f}%")
    print(_ln5)
    print()
    # ─────────────────────────────────────────────────────────────────────

    return fig_delta, fig_loss, fig_r2_fwd, fig_r2_rev, fig_explained, all_results
