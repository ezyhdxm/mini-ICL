"""Optimal orthogonal direction intervention (latent Markov, non-padded)."""

import gc
import numpy as np
import torch
from typing import Optional

import icl.utils.notebook_utils as nu
from icl.latent_markov.analysis.ood import get_latent_sampler
from icl.utils.logger import setup_logger
from icl.utils.orth_common import (
    ols_r2, trial_stats, iqr_err_norm, svd_basis, make_projection_hook,
    gen_and_cache, ORTH_COLORS, ORTH_BAR_WIDTH, ORTH_GROUP_STEP,
    orth_bar_offsets, ORTH_RANDOM_BAND_COLOR, ORTH_RANDOM_BAND_ALPHA,
    ORTH_RANDOM_BAND_HATCH, ORTH_REFERENCE_LINE_COLOR,
)
from icl.latent_markov.analysis.probes import train_linear_hidden_predictor

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
#  intervene_optimal_orth_direction  (single layer)
# ---------------------------------------------------------------------------

def intervene_optimal_orth_direction(
    exp_name: str,
    layer: int,
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
    token_var_threshold: float = 0.9,
    bigram_transform: str = "clr",
    bigram_alpha: float = 0.5,
    extraction_point: str = "post_attn",
    verbose: bool = False,
    print_summary: bool = True,
) -> dict:
    """Find and evaluate optimal orthogonal directions at a single layer.

    **Goal**: Find a rank-k subspace V_opt in the orthogonal complement of
    span(W_task) + span(W_tok) that, when ablated from hidden states via
    h' = h - scale * h * V_opt * V_opt^T, maximally increases OOD CE loss.

    **Steps**:
      1. Fit joint probe  h = posterior * W_task + onehot(x_t) * W_tok + b.
      2. Protected subspace = span(W_task) + span(W_tok), orthogonalised.
         U_orth = basis of orthogonal complement (I - P_prot).
      3. Parametrise V = QR(U_orth * W), optimise W by gradient ascent
         on OOD CE loss.  EMA of W tracked; best snapshot kept.
      4. Evaluate causal effect on major + OOD data.
      5. Diagnostics: R^2 between bigram features and h * V_opt,
         bigram-explained decomposition with causal test.

    Parameters
    ----------
    extraction_point : ``"post_attn"`` | ``"post_mlp"``
        Where to extract and intervene on hidden representations.
        ``"post_attn"`` (default) — after the attention block, before MLP.
        ``"post_mlp"`` — after the full transformer block (attention + MLP),
        i.e. the residual-stream state used as the standard extraction point
        in the mechanistic-interpretability literature.
    """
    from icl.latent_markov.analysis.interventions.bigram import bigram_prefix_counts

    # =====================================================================
    #  Setup: load model, samplers, fit probe
    # =====================================================================
    _, _, config = nu.load_everything("latent", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    model.requires_grad_(False)
    device = config.device

    sampler_major, _, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=0)
    sampler_ood, _, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=n_ood)
    seq_len = sampler_major.seq_len
    V = int(sampler_major.num_states)

    if fit_positions is None:
        fit_positions = list(range(100, seq_len))
    if eval_positions is None:
        eval_positions = list(range(seq_len))

    # Joint probe:  h = posterior * W_task + onehot(x_t) * W_tok + b
    fit_res = train_linear_hidden_predictor(
        exp_name=exp_name, layer=layer, n_samples=fit_n_samples,
        positions=fit_positions, sample_mode="major", step=step,
        n_minor=-1, print_summary=False, skip_baselines=True,
        extraction_point=extraction_point,
    )
    W_task = fit_res["model_weight"].float()   # (K, D)
    W_tok = fit_res["token_weight"].float()    # (V, D)
    D = W_task.shape[1]

    # =====================================================================
    #  Build protected subspace and orthogonal complement
    # =====================================================================
    # SVD(W_task) -> task basis (columns with sigma > 1e-6 * sigma_max)
    # SVD(W_tok) -> token basis (directions explaining token_var_threshold
    #               fraction of sum(sigma^2))
    # Concatenate, re-orthogonalise -> B_prot.
    # U_orth = eigenvectors of (I - B_prot * B_prot^T) with eigenvalue ~1.
    if center_task_vecs:
        W_task = W_task - W_task.mean(0, keepdim=True)

    B_task, rk_task = svd_basis(W_task)
    B_tok, rk_tok = svd_basis(W_tok, var_thresh=token_var_threshold)
    Uc, Sc, _ = torch.linalg.svd(torch.cat([B_task, B_tok], 1), full_matrices=False)
    rk_prot = int((Sc > 1e-6 * Sc[0]).sum().item())
    Bp = Uc[:, :rk_prot]

    P_orth = (torch.eye(D) - Bp @ Bp.T).to(device)
    evals, evecs = torch.linalg.eigh(P_orth)
    U_orth = evecs[:, evals > 0.5].to(device)
    orth_dim = U_orth.shape[1]

    if verbose:
        logger.info(
            f"[opt-dir] task_rank={rk_task}, token_rank={rk_tok}, "
            f"protected_rank={rk_prot}, orth_dim={orth_dim}, "
            f"probe R^2={fit_res['val_r2']:.4f}"
        )

    # =====================================================================
    #  Optimise V_opt via gradient ascent on OOD CE loss
    # =====================================================================
    # V = QR(U_orth * W), intervention: h' = h - scale * h * V * V^T
    # Maximise E_x[ CE(model(x; h')) ] by gradient ascent on W.
    # EMA tracks W for stability; best EMA snapshot kept.
    W_p = torch.randn(orth_dim, n_directions, device=device)
    W_p = (W_p / W_p.norm(dim=0, keepdim=True)).detach().requires_grad_(True)
    opt = torch.optim.Adam([W_p], lr=opt_lr)
    ce = torch.nn.CrossEntropyLoss()

    ema_decay = 0.995
    ema_W = W_p.detach().clone()
    best_ema = ema_W.clone()
    best_loss = -float("inf")
    stale = 0
    smoothed = None
    loss_history = []

    for t in range(n_opt_steps):
        g = sampler_ood.generate(mode="minor", task=None, num_samples=opt_B, epochs=1)
        s = (g[0] if isinstance(g, (tuple, list)) else g)
        if s.dim() == 3:
            s = s.squeeze(0)
        s = s.to(device)

        def _hook(mod, inp, out):
            h = out if torch.is_tensor(out) else out[0]
            Vq, _ = torch.linalg.qr(U_orth @ W_p)
            hm = h - scale * (h @ Vq @ Vq.T)
            return hm if torch.is_tensor(out) else (hm,) + out[1:]

        _hook_target = (
            model.layers[layer] if extraction_point == "post_mlp"
            else model.layers[layer].attn_block
        )
        hnd = _hook_target.register_forward_hook(_hook)
        try:
            logits = model(s)
        finally:
            hnd.remove()

        acc = torch.tensor(0.0, device=device)
        cnt = 0
        for p in eval_positions:
            if p + 1 >= s.shape[1]:
                continue
            acc = acc + ce(logits[:, p], s[:, p + 1].long())
            cnt += 1
        if cnt == 0:
            del s, logits
            continue

        avg = acc / cnt
        cl = avg.item()
        loss_history.append(cl)
        smoothed = cl if smoothed is None else 0.1 * cl + 0.9 * smoothed

        opt.zero_grad()
        (-avg).backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_([W_p], max_norm=grad_clip_norm)
        opt.step()

        with torch.no_grad():
            ema_W.mul_(ema_decay).add_(W_p.detach(), alpha=1 - ema_decay)
        if smoothed > best_loss:
            best_loss = smoothed
            best_ema = ema_W.clone()
            stale = 0
        else:
            stale += 1
        del s, logits, acc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if patience > 0 and stale >= patience:
            break

    with torch.no_grad():
        V_opt, _ = torch.linalg.qr(U_orth @ best_ema)
    V_cpu = V_opt.cpu().float()

    # =====================================================================
    #  R^2 diagnostics: does V_opt encode bigram info?
    # =====================================================================
    eval_pos_idx = torch.tensor(eval_positions, device=device, dtype=torch.long)

    def _bigram_clr(samples):
        """CLR-transformed bigram suffix counts: log P(.|x_t) centered."""
        Bs, L = samples.shape
        counts = bigram_prefix_counts(samples, V).float()
        cur = samples.long()
        bi = torch.arange(Bs, device=samples.device).unsqueeze(1).expand(-1, L)
        ti = torch.arange(L, device=samples.device).unsqueeze(0).expand(Bs, -1)
        rc = counts[bi, ti, cur, :]
        if bigram_transform == "clr":
            freq = (rc + bigram_alpha) / (rc.sum(-1, keepdim=True) + bigram_alpha * V)
            lf = torch.log(freq.clamp_min(1e-12))
            return lf - lf.mean(-1, keepdim=True)
        elif bigram_transform == "log1p":
            return torch.log1p(rc)
        return torch.sqrt((rc / rc.sum(-1, keepdim=True).clamp_min(1.0)).clamp_min(0.0))

    def _collect_h_and_bigram(sampler, mode, n):
        h_acc, bi_acc = [], []
        for _ in range((n + B - 1) // B):
            g = sampler.generate(mode=mode, task=None, num_samples=B, epochs=1)
            s = (g[0] if isinstance(g, (tuple, list)) else g)
            if s.dim() == 3:
                s = s.squeeze(0)
            s = s.to(device)
            cache = {}
            def _hk(mod, inp, out):
                h = out if torch.is_tensor(out) else out[0]
                cache["h"] = h.index_select(1, eval_pos_idx).detach()
            _hk_target = (
                model.layers[layer] if extraction_point == "post_mlp"
                else model.layers[layer].attn_block
            )
            handle = _hk_target.register_forward_hook(_hk)
            try:
                with torch.no_grad():
                    model(s)
            finally:
                handle.remove()
            h_acc.append(cache["h"].cpu())
            bi_acc.append(_bigram_clr(s).index_select(1, eval_pos_idx).cpu())
            del s
        return (
            torch.cat(h_acc).reshape(-1, D).float(),
            torch.cat(bi_acc).reshape(-1, V).float(),
        )

    h_ood, bi_ood = _collect_h_and_bigram(sampler_ood, "minor", n_samples_probe)

    # proj = h * V_opt  (N x k projection onto the optimised subspace)
    proj_ood = h_ood @ V_cpu

    # R^2(bigram -> proj): can bigram stats predict the V_opt projection?
    # R^2(proj -> bigram): does V_opt projection predict bigram stats?
    r2_bi2v = ols_r2(bi_ood, proj_ood)
    r2_v2bi = ols_r2(proj_ood, bi_ood)
    
    # Extended feature set R^2 (enriched with unigram, entropy, position)
    # Collect enriched features on OOD data
    def _collect_enriched_features(sampler, mode, n):
        """Collect enriched feature set (bigram + unigram + entropy + position)."""
        feat_acc = []
        for _ in range((n + B - 1) // B):
            g = sampler.generate(mode=mode, task=None, num_samples=B, epochs=1)
            s = (g[0] if isinstance(g, (tuple, list)) else g)
            if s.dim() == 3:
                s = s.squeeze(0)
            s = s.to(device)
            
            Bs, L = s.shape
            
            # Bigram CLR
            counts = bigram_prefix_counts(s, V).float()
            cur = s.long()
            bi = torch.arange(Bs, device=device).unsqueeze(1).expand(-1, L)
            ti = torch.arange(L, device=device).unsqueeze(0).expand(Bs, -1)
            rc = counts[bi, ti, cur, :]
            freq = (rc + bigram_alpha) / (rc.sum(-1, keepdim=True) + bigram_alpha * V)
            bigram_clr = torch.log(freq.clamp_min(1e-12)) - torch.log(freq.clamp_min(1e-12)).mean(-1, keepdim=True)
            
            # Unigram CLR (counts of each token seen in prefix)
            unigram_counts = torch.zeros(Bs, L, V, device=device)
            for t in range(1, L):
                unigram_counts[:, t, :] = torch.nn.functional.one_hot(
                    s[:, :t].long(), num_classes=V
                ).sum(dim=1).float()
            unigram_freq = (unigram_counts + bigram_alpha) / (
                unigram_counts.sum(-1, keepdim=True) + bigram_alpha * V
            )
            unigram_clr = torch.log(unigram_freq.clamp_min(1e-12)) - torch.log(
                unigram_freq.clamp_min(1e-12)
            ).mean(-1, keepdim=True)
            
            # Entropy of bigram distribution
            bigram_entropy = -(freq * torch.log(freq.clamp_min(1e-12))).sum(-1, keepdim=True)
            bigram_entropy_norm = bigram_entropy / np.log(V)
            
            # Position encoding (normalized)
            positions = torch.arange(L, device=device, dtype=torch.float32).view(1, -1, 1)
            pos_norm = (positions / L).expand(Bs, -1, 1)
            
            # Concatenate features
            features = torch.cat([
                bigram_clr,
                unigram_clr,
                bigram_entropy_norm.expand(-1, -1, 1),
                pos_norm,
            ], dim=-1)
            
            feat_acc.append(features.cpu())
            del s
        
        return torch.cat(feat_acc).reshape(-1, features.shape[-1]).float()
    
    enrich_ood = _collect_enriched_features(sampler_ood, "minor", n_samples_probe)
    
    # Compute R^2 with enriched features
    r2_enrich2v = ols_r2(enrich_ood, proj_ood)
    r2_v2enrich = ols_r2(proj_ood, enrich_ood)

    # Baseline: R^2 for random rank-k subspace in orth complement
    U_orth_cpu = U_orth.cpu().float()
    N_RAND = 3
    rand_bi2v, rand_v2bi = [], []
    for _ in range(N_RAND):
        Vr, _ = torch.linalg.qr(U_orth_cpu @ torch.randn(orth_dim, n_directions))
        rp = h_ood @ Vr
        rand_bi2v.append(ols_r2(bi_ood, rp))
        rand_v2bi.append(ols_r2(rp, bi_ood))
    r2_rand_bi2v = float(np.mean(rand_bi2v))
    r2_rand_v2bi = float(np.mean(rand_v2bi))

    # =====================================================================
    #  Bigram-explained decomposition
    # =====================================================================
    # Fit OLS:  proj_ood = bigram_feat * W + b  ->  predicted part Y_hat
    # SVD(Y_hat) -> keep directions with sigma^2 > 0.5% of total variance
    # Map back to R^D:  V_explained = V_opt * Vt[:r]^T
    # Causal test: remove V_explained, measure OOD loss increase
    proj_c = proj_ood - proj_ood.mean(0, keepdim=True)
    total_var = (proj_c ** 2).sum().item()
    dir_pred = torch.softmax(bi_ood, dim=-1)
    bi_explained = {}
    for fname, feat in [("bigram_clr", bi_ood), ("dirichlet_pred", dir_pred)]:
        Nf = feat.shape[0]
        X_aug = torch.cat([feat, torch.ones(Nf, 1)], 1)
        W_fit = torch.linalg.pinv(X_aug) @ proj_ood
        Y_hat = X_aug @ W_fit
        r2 = 1.0 - ((proj_ood - Y_hat) ** 2).sum().item() / total_var \
            if total_var > 0 else 0.0
        _, Sp, Vtp = torch.linalg.svd(
            Y_hat - Y_hat.mean(0, keepdim=True), full_matrices=False,
        )
        frac = Sp ** 2 / max(total_var, 1e-12)
        rk = int((frac > 0.005).sum().item())
        V_expl = V_cpu @ Vtp[:rk].T if rk > 0 else torch.empty(D, 0)
        bi_explained[fname] = {
            "explained_rank": rk, "r2_feat_to_proj": r2, "V_explained": V_expl,
        }

    del proj_ood, h_ood, bi_ood
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # =====================================================================
    #  Causal evaluation
    # =====================================================================
    # Intervention: h' = h - scale * h * P_V  where P_V = V_opt * V_opt^T
    P_v = (V_opt @ V_opt.T).to(device)
    ce_eval = torch.nn.CrossEntropyLoss(reduction="none")

    def _run_eval(sampler, mode, n, full=False):
        """Hooked forward pass. Returns (baseline, intervened) or full dict."""
        bp_pos = {p: [] for p in eval_positions} if full else None
        ip_pos = {p: [] for p in eval_positions} if full else None
        b_all, i_all = [], []
        ld_sum = torch.zeros(V) if full else None
        ld_n = 0
        for _ in range((n + B - 1) // B):
            g = sampler.generate(mode=mode, task=None, num_samples=B, epochs=1)
            s = (g[0] if isinstance(g, (tuple, list)) else g)
            if s.dim() == 3:
                s = s.squeeze(0)
            s = s.to(device)
            with torch.no_grad():
                bl = model(s)
            _eval_target = (
                model.layers[layer] if extraction_point == "post_mlp"
                else model.layers[layer].attn_block
            )
            handle = _eval_target.register_forward_hook(make_projection_hook(P_v, scale))
            try:
                with torch.no_grad():
                    li = model(s)
            finally:
                handle.remove()
            for p in eval_positions:
                if p + 1 >= s.shape[1]:
                    continue
                tgt = s[:, p + 1].long()
                bv = ce_eval(bl[:, p], tgt).mean().item()
                iv = ce_eval(li[:, p], tgt).mean().item()
                b_all.append(bv)
                i_all.append(iv)
                if full:
                    bp_pos[p].append(bv)
                    ip_pos[p].append(iv)
                    dl = (li[:, p] - bl[:, p]).cpu()
                    ld_sum += dl.sum(0)
                    ld_n += dl.shape[0]
            del s, bl, li
        ba, ia = float(np.mean(b_all)), float(np.mean(i_all))
        if not full:
            return ba, ia
        bpp = [float(np.mean(bp_pos[p])) for p in eval_positions if bp_pos[p]]
        ipp = [float(np.mean(ip_pos[p])) for p in eval_positions if ip_pos[p]]
        return {
            "baseline": ba, "intervened": ia, "delta": ia - ba,
            "baseline_per_pos": bpp, "intervened_per_pos": ipp,
            "positions": [p for p in eval_positions if bp_pos[p]],
            "mean_logit_delta": (ld_sum / max(ld_n, 1)).numpy(),
        }

    res_maj = _run_eval(sampler_major, "major", n_samples_eval, full=True)
    res_ood = _run_eval(sampler_ood, "minor", n_samples_eval, full=True)

    # Random rank-k baseline intervention (OOD only, major delta ~ 0)
    N_RAND_INT = 2
    rand_deltas = []
    for _ in range(N_RAND_INT):
        rk = min(n_directions, orth_dim)
        Vr, _ = torch.linalg.qr(U_orth_cpu @ torch.randn(orth_dim, rk))
        Pr = (Vr @ Vr.T).to(device)
        # Reuse _run_eval by temporarily swapping projector
        old_Pv = P_v.clone()
        P_v.copy_(Pr)
        bo, io = _run_eval(sampler_ood, "minor", n_samples_eval)
        P_v.copy_(old_Pv)
        rand_deltas.append(io - bo)
    rand_delta_ood = float(np.mean(rand_deltas))

    # Bigram-explained causal test (OOD only)
    for info in bi_explained.values():
        if info["explained_rank"] == 0:
            info["delta_ood"] = 0.0
            continue
        Ve = info["V_explained"].to(device)
        old_Pv = P_v.clone()
        P_v.copy_(Ve @ Ve.T)
        bo, io = _run_eval(sampler_ood, "minor", n_samples_eval)
        P_v.copy_(old_Pv)
        info["delta_ood"] = io - bo
        del Ve

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # =====================================================================
    #  Build result dict (aligned with plot_optimal_orth_direction_across_layers)
    # =====================================================================
    pct_m = 100 * res_maj["delta"] / res_maj["baseline"] \
        if res_maj["baseline"] > 0 else float("nan")
    pct_o = 100 * res_ood["delta"] / res_ood["baseline"] \
        if res_ood["baseline"] > 0 else float("nan")

    results = {
        "baseline_loss_major": res_maj["baseline"],
        "intervened_loss_major": res_maj["intervened"],
        "delta_loss_major": res_maj["delta"],
        "pct_increase_major": pct_m,
        "baseline_loss_ood": res_ood["baseline"],
        "intervened_loss_ood": res_ood["intervened"],
        "delta_loss_ood": res_ood["delta"],
        "pct_increase_ood": pct_o,
        "baseline_per_pos_major": res_maj["baseline_per_pos"],
        "intervened_per_pos_major": res_maj["intervened_per_pos"],
        "baseline_per_pos_ood": res_ood["baseline_per_pos"],
        "intervened_per_pos_ood": res_ood["intervened_per_pos"],
        "eval_positions": res_maj["positions"],
        "layer": layer, "scale": scale,
        "task_rank": rk_task, "token_rank": rk_tok,
        "protected_rank": rk_prot,
        "n_directions": n_directions,
        "directions": V_opt.cpu(),
        "loss_history": loss_history,
        "r2_bi2v_ood": r2_bi2v,
        "r2_v2bi_ood": r2_v2bi,
        "r2_rand_bi2v_ood": r2_rand_bi2v,
        "r2_rand_v2bi_ood": r2_rand_v2bi,
        "r2_enrich2v_ood": r2_enrich2v,
        "r2_v2enrich_ood": r2_v2enrich,
        "rand_int_delta_ood": rand_delta_ood,
        "bi_explained": {
            fn: {k: v for k, v in info.items() if k != "V_explained"}
            for fn, info in bi_explained.items()
        },
        "mean_logit_delta_ood": res_ood["mean_logit_delta"],
    }

    if print_summary:
        print(f"\n{'='*60}")
        print(f"Optimal Orth Direction (layer {layer}, rank-{n_directions}, "
              f"scale={scale})")
        print(f"{'='*60}")
        print(f"  task_rank={rk_task}  token_rank={rk_tok}  "
              f"protected_rank={rk_prot}  orth_dim={orth_dim}")
        print(f"  opt steps: {len(loss_history)}/{n_opt_steps}  "
              f"loss: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}")
        print()
        print(f"  {'Metric':<25} {'Major':>10} {'OOD':>10}")
        print(f"  {'-'*45}")
        print(f"  {'Baseline CE':<25} {res_maj['baseline']:>10.4f} "
              f"{res_ood['baseline']:>10.4f}")
        print(f"  {'Intervened CE':<25} {res_maj['intervened']:>10.4f} "
              f"{res_ood['intervened']:>10.4f}")
        print(f"  {'Delta CE':<25} {res_maj['delta']:>10.4f} "
              f"{res_ood['delta']:>10.4f}")
        print(f"  {'% increase':<25} {pct_m:>9.1f}% {pct_o:>9.1f}%")
        print(f"  {'Rand orth delta (OOD)':<25} {'':>10} {rand_delta_ood:>10.4f}")
        print()
        print(f"  R^2 (OOD):  bi->V={r2_bi2v:.3f}  V->bi={r2_v2bi:.3f}  "
              f"rand={r2_rand_bi2v:.3f}")
        print(f"  R^2 (OOD, enriched):  enrich->V={r2_enrich2v:.3f}  "
              f"V->enrich={r2_v2enrich:.3f}")
        print(f"    [enriched features: bigram_clr + unigram_clr + bigram_entropy + position]")
        print()
        for fn, info in bi_explained.items():
            print(f"  {fn}: d_ood={info['delta_ood']:.4f}  "
                  f"rank={info['explained_rank']}  "
                  f"R^2={info['r2_feat_to_proj']:.3f}")

    return results


# ---------------------------------------------------------------------------
#  plot_optimal_orth_direction_across_layers
# ---------------------------------------------------------------------------

def plot_optimal_orth_direction_across_layers(
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
    n_minor: int = 0,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    eval_positions: Optional[list] = None,
    center_task_vecs: bool = False,
    n_directions: int = 1,
    scale: float = 1.0,
    token_var_threshold: float = 0.9,
    bigram_transform: str = "clr",
    bigram_alpha: float = 0.5,
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
    """Find and evaluate optimal orthogonal directions across layers.

    **Goal**: Find a rank-k subspace V_opt, orthogonal to both task and token
    subspaces, whose removal maximally increases target-data cross-entropy loss.

    Parameters
    ----------
    opt_target : ``"ood"`` | ``"minor"``
        Which data to optimise on (gradient ascent target).
        ``"ood"`` finds directions whose removal hurts OOD prediction.
        ``"minor"`` finds directions whose removal hurts minor-task
        prediction.
    extraction_point : ``"post_attn"`` | ``"post_mlp"``
        Where to extract (and intervene on) hidden representations in each
        transformer layer.  ``"post_attn"`` (default) hooks after the
        attention block, before the MLP.  ``"post_mlp"`` hooks after the
        full block (attention + MLP), i.e. the residual-stream state used
        as the standard extraction point in the mechanistic-interpretability
        literature.
    probe_method : ``"ols"`` | ``"averaging"``
        How to estimate task and token subspaces for the protected subspace.
        ``"ols"`` (default) — fits a joint OLS probe
        ``h = posterior * W_task + onehot(x_t) * W_tok + b`` and extracts
        W_task / W_tok via Frisch-Waugh-Lovell.
        ``"averaging"`` — collects token-conditioned (interventional) data
        and derives task/token vectors from ANOVA cell-means, the same
        approach used by ``plot_anova_separability`` / ``plot_averaging_r2``.

    **Algorithm**:
      1. Fit joint probe  h = posterior * W_task + onehot(x_t) * W_tok + b
         to identify task and token subspaces at each layer.
      2. Protected subspace = span(W_task) + span(W_tok).
         Orthogonal complement U_orth = eigenvectors of (I - P_prot) with
         eigenvalue 1, where P_prot = B_prot * B_prot^T.
      3. Parametrise V = U_orth * W  (W is orth_dim x k, learnable).
         Gradient ascent on target CE loss under intervention h' = h - s*h*V*V^T.
         EMA tracks W for stability; best EMA snapshot kept.
      4. Evaluate causal effect and compare to baselines:
         - Random rank-k subspace in orth complement (sanity check)
         - Bigram-explained subspace of V_opt (interpretability test)

    **Outputs** (6 figures + result dict):
      fig_delta    :  CE loss increase bars (major + target)
      fig_loss     :  Optimisation loss history
      fig_r2_fwd   :  Features → V_opt projection R² (target)
      fig_r2_rev   :  V_opt projection → Features R² (target)
      fig_logit    :  Per-token logit change under intervention (target)
      fig_bigram   :  V_opt full vs enriched-explained causal effect (target)

    Returns (fig_delta, fig_loss, fig_r2_fwd, fig_r2_rev, fig_logit, fig_bigram, all_results).
    """
    import matplotlib.pyplot as plt
    from icl.utils.notebook_utils import bigram_prefix_counts
    from icl.latent_markov.analysis.probes import (
        _collect_multi_layer_data, _fit_probe,
    )

    # =====================================================================
    #  Setup
    # =====================================================================
    _, _, config = nu.load_everything("latent", exp_name)
    if step is None:
        step = config.training.num_epochs
    D = int(config.model.emb_dim)
    if layers is None:
        layers = list(range(config.model.num_layers))

    sampler_tmp, _, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=0)
    seq_len = sampler_tmp.seq_len
    V = int(sampler_tmp.num_states)
    n_major = int(sampler_tmp.n_major_tasks)
    del sampler_tmp

    if fit_positions is None:
        # seq_len - 1 excluded: get_token_conditioned_hiddens requires p < seq_len - 1
        fit_positions = list(range(100, seq_len - 1))
    if eval_positions is None:
        eval_positions = list(range(seq_len))
    eval_pos = list(eval_positions)
    if 0 not in eval_pos:
        eval_pos = [0] + eval_pos

    # =====================================================================
    #  Phase 1: Estimate task/token subspaces (all layers, one forward pass)
    # =====================================================================
    if probe_method == "averaging":
        # ANOVA cell-mean approach: collect token-conditioned (interventional)
        # hidden states, then compute task and token vectors from the two-way
        # ANOVA marginals — the same method used by plot_anova_separability /
        # plot_averaging_r2_latent.
        logger.info("[opt-dir] Phase 1: Collecting token-conditioned hiddens (ANOVA) ...")
        from icl.latent_markov.analysis.variance import get_token_conditioned_hiddens

        all_hiddens_anova, anova_info = get_token_conditioned_hiddens(
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
                # cell_means: (V_p, K, D) — average over the batch dimension
                parts.append(
                    all_hiddens_anova[li, pi, :V_p, :n_major].float().mean(dim=-2)
                )

            if not parts:
                raise ValueError(
                    f"[opt-dir] probe_method='averaging': none of fit_positions "
                    f"found in token-conditioned hiddens."
                )

            # Align to minimum V_p across positions (tokens seen at all positions)
            min_V = min(p.shape[0] for p in parts)
            parts = [p[:min_V] for p in parts]

            # Two-way ANOVA: remove per-position grand mean, pool, decompose
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
        # OLS joint probe:  h = posterior * W_task + onehot(x_t) * W_tok + b
        # FWL gives unbiased W_task orthogonal to token confounds.
        logger.info("[opt-dir] Phase 1: Fitting OLS probes ...")
        probe_data = _collect_multi_layer_data(
            exp_name, layers, B=B, n_samples=fit_n_samples,
            step=step, n_minor=-1, positions=fit_positions,
            sample_mode="major",
            extraction_point=extraction_point,
        )
        seq_perm = torch.randperm(probe_data["posteriors"].shape[0])
        probes = {}
        for l in layers:
            probes[l] = _fit_probe(
                probe_data["hiddens_by_layer"][l],
                probe_data["posteriors"], probe_data["logits"],
                probe_data["real_tokens"],
                n_major=probe_data["n_major"], n_tasks=probe_data["n_tasks"],
                layer=l, positions=probe_data["positions"],
                sample_mode="major", skip_baselines=True, seq_perm=seq_perm,
            )
        del probe_data

    # =====================================================================
    #  Phase 2: Load model, prepare samplers, cache eval data
    # =====================================================================
    logger.info("[opt-dir] Phase 2: Loading model & caching eval data ...")
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    model.requires_grad_(False)
    device = config.device

    sampler_major, _, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=0)
    sampler_ood, _, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=n_ood)
    tgt_n_minor = n_minor if n_minor > 0 else 1_000_000
    sampler_minor, _, _ = get_latent_sampler(
        exp_name, n_minor=tgt_n_minor, n_ood=0,
    )
    sampler_tgt = sampler_minor if opt_target == "minor" else sampler_ood
    tgt_label = "Minor" if opt_target == "minor" else "OOD"
    tgt_abbr  = "Min."  if opt_target == "minor" else "OOD"

    cache_maj = gen_and_cache(model, sampler_major, "major", n_samples_eval, B, device)
    cache_ood = gen_and_cache(model, sampler_ood, "minor", n_samples_eval, B, device)
    cache_minor = gen_and_cache(model, sampler_minor, "minor", n_samples_eval, B, device)
    cache_tgt = cache_minor if opt_target == "minor" else cache_ood

    # =====================================================================
    #  Phase 3: Collect hidden states + bigram features (all layers, one pass)
    # =====================================================================
    # Bigram CLR: centred log-ratio of empirical bigram counts P(.|x_t)
    # from the prefix x_1..x_t.  This captures the "token-counting" info
    # that the model can compute without any latent-task knowledge.
    eval_pos_idx = torch.tensor(eval_pos, device=device, dtype=torch.long)

    def _collect_h_and_bigram(sampler, mode, n):
        """Forward pass with hooks at all layers; compute bigram CLR and enriched features.

        All statistics are derived from the **same** sequences used for hidden-state
        collection, so ``bi`` / ``enrich`` are row-aligned with every ``h_dict[l]``.

        Returns
        -------
        h_dict : dict[int, Tensor]  shape (N, D)       hidden states per layer
        bi     : Tensor             shape (N, V)        bigram CLR (``bigram_transform`` applied)
        enrich : Tensor             shape (N, 2*V+2)    [bigram_clr | unigram_clr | entropy | pos]
            where N = n_batches * B * len(eval_pos).
        """
        h_acc     = {l: [] for l in layers}
        bi_acc    = []
        enrich_acc = []
        # Layout (reference-category encoding — one dimension dropped per CLR/one-hot group):
        #   bigram_clr (V-1) | one_hot_x_t (V-1) | position (1)
        #
        # full_bigram (V*(V-1)) was dropped: its partial R² was consistently tiny (0.006–0.02)
        # relative to bigram_clr, while consuming V*(V-1) parameters.  The current-token CLR
        # row (bigram_clr) is the linearly-accessible representation that drives R2(bi->V).
        # CLR sums to zero, so we drop the last column of every group to remove the dependency.
        feat_dim  = (V - 1) + (V - 1) + 1  # bigram_clr + one_hot_x_t + position

        for _ in range(max(1, (n + B - 1) // B)):
            g = sampler.generate(mode=mode, task=None, num_samples=B, epochs=1)
            s = (g[0] if isinstance(g, (tuple, list)) else g)
            if s.dim() == 3:
                s = s.squeeze(0)
            s = s.to(device)
            Bs, L = s.shape

            # --- model forward with hooks (all layers, one pass) ---
            caches  = {}
            handles = []
            for ll in layers:
                def _hook(mod, inp, out, _l=ll):
                    h = out if torch.is_tensor(out) else out[0]
                    caches[_l] = h.index_select(1, eval_pos_idx).detach()
                _tgt = (
                    model.layers[ll] if extraction_point == "post_mlp"
                    else model.layers[ll].attn_block
                )
                handles.append(_tgt.register_forward_hook(_hook))
            with torch.no_grad():
                model(s)
            for hh in handles:
                hh.remove()
            for ll in layers:
                h_acc[ll].append(caches[ll].cpu())

            # --- shared prefix statistics (computed once per batch) ---
            idx_b  = torch.arange(Bs, device=device).unsqueeze(1).expand(-1, L)
            idx_t  = torch.arange(L,  device=device).unsqueeze(0).expand(Bs, -1)
            counts = bigram_prefix_counts(s, V).float()   # (Bs, L, V, V)
            rc     = counts[idx_b, idx_t, s.long(), :]    # (Bs, L, V) — current-token row

            # --- current-token bigram CLR (reference-encoded, V-1 features) ---
            # This is the same quantity used by bi_tgt / R2(bi->V), exposed directly
            # so the linear probe can access it without a nonlinear selection step.
            bi_prob = (rc + bigram_alpha) / (rc.sum(-1, keepdim=True) + bigram_alpha * V)
            log_bi  = torch.log(bi_prob.clamp_min(1e-12))
            bi_clr  = log_bi - log_bi.mean(-1, keepdim=True)   # CLR, (Bs, L, V)
            bi_clr_ref = bi_clr[..., :-1]                       # reference-encoded, (Bs, L, V-1)

            # Returned ``bi`` tensor follows bigram_transform (for existing R² plots)
            if bigram_transform == "clr":
                bi_out = bi_clr
            elif bigram_transform == "log1p":
                bi_out = torch.log1p(rc)
            else:
                bi_out = torch.sqrt(
                    (rc / rc.sum(-1, keepdim=True).clamp_min(1.0)).clamp_min(0.0)
                )

            # --- one-hot of current token x_t (reference-encoded, V-1 features) ---
            one_hot  = torch.nn.functional.one_hot(s.long(), num_classes=V).float()
            x_t_ref  = one_hot[..., :-1]   # (Bs, L, V-1)

            # --- fractional position ---
            pos_feat = (
                torch.arange(L, device=device, dtype=torch.float32)
                .view(1, L, 1).expand(Bs, -1, 1)
            ) / L  # (Bs, L, 1)

            # Concatenate and select eval positions
            enrich = torch.cat(
                [bi_clr_ref, x_t_ref, pos_feat], dim=-1,
            )  # (Bs, L, (V-1) + (V-1) + 1)
            bi_acc.append(bi_out.index_select(1, eval_pos_idx).cpu())
            enrich_acc.append(enrich.index_select(1, eval_pos_idx).cpu())

            del s, caches, counts, rc, bi_prob, log_bi, bi_clr, bi_clr_ref, bi_out
            del one_hot, x_t_ref, pos_feat, enrich

        # Feature-group slice definitions (fixed for all layers / call sites)
        _s = 0
        feat_groups = {}
        for _name, _k in [
            ("bigram_clr",  V - 1),
            ("one_hot_x_t", V - 1),
            ("position",    1),
        ]:
            feat_groups[_name] = slice(_s, _s + _k)
            _s += _k
        assert _s == feat_dim, f"feat_dim mismatch: {_s} vs {feat_dim}"

        return (
            {ll: torch.cat(h_acc[ll]).reshape(-1, D).float() for ll in layers},
            torch.cat(bi_acc).reshape(-1, V).float(),
            torch.cat(enrich_acc).reshape(-1, feat_dim).float(),
            feat_groups,
        )

    logger.info("[opt-dir] Phase 3: Collecting hiddens + bigrams ...")
    h_maj, bi_maj, enrich_maj, _           = _collect_h_and_bigram(sampler_major, "major", n_samples_probe)
    h_tgt, bi_tgt, enrich_tgt, feat_groups = _collect_h_and_bigram(sampler_tgt,   "minor", n_samples_probe)

    # =====================================================================
    #  Shared helpers
    # =====================================================================
    ce_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def _eval_hooked(layer_idx, P, cached, full=False):
        """Forward pass with hook  h' = h - scale * h * P  at one layer.

        P is a D x D projector onto the subspace to ablate.
        Baseline logits come from cache (no redundant unhooked passes).
        If full=True, also returns per-position losses, logit deltas, and
        per-batch CE deltas (for IQR computation).
        """
        seqs, base_logits = cached
        bp_pos = {p: [] for p in eval_pos} if full else None
        ip_pos = {p: [] for p in eval_pos} if full else None
        b_all, i_all = [], []
        batch_deltas = []
        ld_sum = torch.zeros(V) if full else None
        ld_n = 0
        for s_cpu, bl_cpu in zip(seqs, base_logits):
            s = s_cpu.to(device)
            _eval_tgt = (
                model.layers[layer_idx] if extraction_point == "post_mlp"
                else model.layers[layer_idx].attn_block
            )
            handle = _eval_tgt.register_forward_hook(make_projection_hook(P, scale))
            try:
                with torch.no_grad():
                    li = model(s)
            finally:
                handle.remove()
            bl = bl_cpu.to(device)
            b_batch, i_batch = [], []
            for p in eval_pos:
                if p + 1 >= s.shape[1]:
                    continue
                tgt = s[:, p + 1].long()
                bv = ce_fn(bl[:, p], tgt).mean().item()
                iv = ce_fn(li[:, p], tgt).mean().item()
                b_all.append(bv)
                i_all.append(iv)
                b_batch.append(bv)
                i_batch.append(iv)
                if full:
                    bp_pos[p].append(bv)
                    ip_pos[p].append(iv)
                    dl = (li[:, p] - bl[:, p]).cpu()
                    ld_sum += dl.sum(0)
                    ld_n += dl.shape[0]
            if b_batch:
                batch_deltas.append(
                    float(np.mean(i_batch)) - float(np.mean(b_batch))
                )
            del s, bl, li
        ba, ia = float(np.mean(b_all)), float(np.mean(i_all))
        if not full:
            return ba, ia
        bpp = [float(np.mean(bp_pos[p])) for p in eval_pos if bp_pos[p]]
        ipp = [float(np.mean(ip_pos[p])) for p in eval_pos if ip_pos[p]]
        return {
            "baseline": ba, "intervened": ia, "delta": ia - ba,
            "baseline_per_pos": bpp, "intervened_per_pos": ipp,
            "positions": [p for p in eval_pos if bp_pos[p]],
            "mean_logit_delta": (ld_sum / max(ld_n, 1)).numpy(),
            "delta_per_batch": batch_deltas,
        }

    def _mlp_r2(X, Y, hid=64, ep=100, lr=1e-3):
        """Two-layer MLP R^2 on held-out 20%."""
        N = X.shape[0]
        nt = int(0.8 * N)
        pm = torch.randperm(N)
        Xtr, Ytr = X[pm[:nt]], Y[pm[:nt]]
        Xte, Yte = X[pm[nt:]], Y[pm[nt:]]
        net = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], hid), torch.nn.SiLU(),
            torch.nn.Linear(hid, Y.shape[1]),
        )
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(ep):
            loss = ((net(Xtr) - Ytr) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            ss_r = ((Yte - net(Xte)) ** 2).sum().item()
            ss_t = ((Yte - Yte.mean(0)) ** 2).sum().item()
        return 1.0 - ss_r / ss_t if ss_t > 0 else float("nan")

    def _build_protected(W_task, W_tok):
        """Compute protected subspace and its orthogonal complement.

        Steps:
          1. SVD(W_task) -> keep right-singular vectors with sigma > 1e-6*sigma_max
             These span the "task subspace" in R^D.
          2. SVD(W_tok) -> keep directions explaining >= token_var_threshold
             fraction of sum(sigma^2).  This is the "token subspace".
          3. Concatenate both bases, re-orthogonalise via SVD -> B_prot (D x r_prot)
             P_prot = B_prot * B_prot^T  is the projector onto the protected space.
          4. U_orth = eigenvectors of (I - P_prot) with eigenvalue ~1.
             V_opt will live in span(U_orth), so it's orthogonal to task+token.
        """
        if center_task_vecs:
            W_task = W_task - W_task.mean(0, keepdim=True)

        B_task, rk_task = svd_basis(W_task)
        B_tok, rk_tok = svd_basis(W_tok, var_thresh=token_var_threshold)
        Uc, Sc, _ = torch.linalg.svd(
            torch.cat([B_task, B_tok], 1), full_matrices=False,
        )
        rk_prot = int((Sc > 1e-6 * Sc[0]).sum().item())
        Bp = Uc[:, :rk_prot]
        P_orth = (torch.eye(D) - Bp @ Bp.T).to(device)
        evals, evecs = torch.linalg.eigh(P_orth)
        U_orth = evecs[:, evals > 0.5].to(device)
        return U_orth, rk_task, rk_tok, rk_prot

    val_seq = cache_tgt[0][0].to(device)
    ce_mean = torch.nn.CrossEntropyLoss()

    def _val_loss(layer_idx, V_mat):
        """CE loss on cached validation batch with intervention."""
        P = V_mat @ V_mat.T

        _val_tgt = (
            model.layers[layer_idx] if extraction_point == "post_mlp"
            else model.layers[layer_idx].attn_block
        )
        hnd = _val_tgt.register_forward_hook(make_projection_hook(P, scale))
        try:
            with torch.no_grad():
                logits_v = model(val_seq)
        finally:
            hnd.remove()
        acc, cnt = 0.0, 0
        for p in eval_pos:
            if p + 1 >= val_seq.shape[1]:
                continue
            acc += ce_mean(logits_v[:, p], val_seq[:, p + 1].long()).item()
            cnt += 1
        return acc / cnt if cnt > 0 else float("nan")

    def _optimise_v(layer_idx, U_orth, val_every=20):
        """Gradient ascent for V_opt in span(U_orth).

        Parametrise V = QR(U_orth * W) so V is always orthonormal.
        Maximise CE loss of intervened model:
            max_W  E_x [ CrossEntropy( model(x; h' = h - s*h*V*V^T) ) ]
        Use EMA of W for stability and save the best snapshot.

        Early stopping:
          - Training patience  (``patience``):   stop after this many consecutive
            training steps with no improvement in smoothed training loss.
          - Validation patience (``val_patience``): stop after this many consecutive
            validation evaluations with no improvement in validation CE loss.
            The snapshot with the best validation loss is returned as V_opt.
            Set val_patience=0 to disable (default).
        """
        orth_dim = U_orth.shape[1]
        W_p = torch.randn(orth_dim, n_directions, device=device)
        W_p = (W_p / W_p.norm(dim=0, keepdim=True)).detach().requires_grad_(True)
        opt = torch.optim.Adam([W_p], lr=opt_lr)
        ce = torch.nn.CrossEntropyLoss()
        ema_decay = 0.995
        ema_W = W_p.detach().clone()
        # Training-loss best snapshot
        best_ema       = ema_W.clone()
        best_loss      = -float("inf")
        stale          = 0
        smoothed       = None
        # Validation-loss best snapshot
        best_val_loss  = -float("inf")
        best_ema_val   = None
        val_stale      = 0
        history        = []
        val_history    = []

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
                Vq, _ = torch.linalg.qr(U_orth @ W_p)
                hm = h - scale * (h @ Vq @ Vq.T)
                return hm if torch.is_tensor(out) else (hm,) + out[1:]

            _opt_tgt = (
                model.layers[layer_idx] if extraction_point == "post_mlp"
                else model.layers[layer_idx].attn_block
            )
            hnd = _opt_tgt.register_forward_hook(_hook)
            try:
                logits = model(s)
            finally:
                hnd.remove()

            acc = torch.tensor(0.0, device=device)
            cnt = 0
            for p in eval_pos:
                if p + 1 >= s.shape[1]:
                    continue
                acc = acc + ce(logits[:, p], s[:, p + 1].long())
                cnt += 1
            if cnt == 0:
                del s, logits
                continue

            avg = acc / cnt
            cl = avg.item()
            history.append(cl)
            smoothed = cl if smoothed is None else 0.1 * cl + 0.9 * smoothed

            opt.zero_grad()
            (-avg).backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([W_p], max_norm=grad_clip_norm)
            opt.step()

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
                    V_cur, _ = torch.linalg.qr(U_orth @ best_ema)
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

        # Return the val-best snapshot when val patience is active, else train-best
        final_ema = (
            best_ema_val if (val_patience > 0 and best_ema_val is not None)
            else best_ema
        )
        with torch.no_grad():
            V_opt, _ = torch.linalg.qr(U_orth @ final_ema)
        return V_opt, val_history

    # =====================================================================
    #  Partial R² & design-matrix diagnostics helper
    # =====================================================================
    def _partial_r2_and_diag(X: torch.Tensor, Y: torch.Tensor,
                              groups: dict, n_rep: int = 5) -> dict:
        """Partial R², VIF and design-matrix statistics for feature groups.

        Parameters
        ----------
        X       : (N, F)  full feature matrix (CPU, float32)
        Y       : (N, k)  target projections (CPU, float32)
        groups  : dict name -> slice  — non-overlapping or overlapping feature groups
        n_rep   : number of repeated random splits for stable R² estimates

        Returns
        -------
        dict with keys:
          r2_full          float
          partial_r2       dict[name -> float]   R²(full) - R²(without group)
          semi_partial_r2  dict[name -> float]   FWL: R²(Y_res ~ X_i_res)
          vif              dict[name -> float]   1 / (1 - R²(X_i ~ X_{-i}))
          cond_num         float   condition number of standardised X
          eff_rank         float   entropy-based effective rank of X
          group_corr       Tensor  (G, G) between-group correlation matrix
          group_names      list[str]
        """
        N, F = X.shape
        names = list(groups.keys())
        G = len(names)

        def _r2_stable(A, B):
            """Mean R² over n_rep random 80/20 splits — reduces split noise."""
            vals = [ols_r2(A, B) for _ in range(n_rep)]
            return float(np.mean([v for v in vals if not np.isnan(v)] or [float("nan")]))

        def _X_without(name):
            keep = torch.ones(F, dtype=torch.bool)
            keep[groups[name]] = False
            return X[:, keep]

        # Full model R²
        r2_full = _r2_stable(X, Y)

        # Drop-one-group partial R²
        partial_r2 = {}
        for name in names:
            X_wo = _X_without(name)
            r2_wo = _r2_stable(X_wo, Y) if X_wo.shape[1] > 0 else 0.0
            partial_r2[name] = max(0.0, r2_full - r2_wo)

        # FWL semi-partial R²: residualise Y and X_i on X_{-i}, then regress
        semi_partial_r2 = {}
        for name in names:
            Xi  = X[:, groups[name]]
            X_o = _X_without(name)
            if X_o.shape[1] == 0:
                semi_partial_r2[name] = r2_full
                continue
            # Residualise both Y and X_i on X_{-i}
            X_o_aug = torch.cat([X_o, torch.ones(N, 1)], 1)
            Wo_y = torch.linalg.pinv(X_o_aug) @ Y
            Y_res = Y - X_o_aug @ Wo_y
            Wo_x = torch.linalg.pinv(X_o_aug) @ Xi
            Xi_res = Xi - X_o_aug @ Wo_x
            semi_partial_r2[name] = max(0.0, _r2_stable(Xi_res, Y_res))

        # VIF: R²(X_i ~ X_{-i}), then 1/(1 - R²)
        vif = {}
        for name in names:
            Xi  = X[:, groups[name]]
            X_o = _X_without(name)
            if X_o.shape[1] == 0 or Xi.shape[1] == 0:
                vif[name] = float("nan")
                continue
            r2_xi = _r2_stable(X_o, Xi)
            vif[name] = 1.0 / max(1.0 - r2_xi, 1e-6)

        # Design-matrix diagnostics on standardised X
        mu  = X.mean(0, keepdim=True)
        std = X.std(0, keepdim=True).clamp_min(1e-8)
        X_std = (X - mu) / std
        _, S, _ = torch.linalg.svd(X_std, full_matrices=False)
        cond_num = (S[0] / S[-1].clamp_min(1e-10)).item()
        S2 = S ** 2
        p  = S2 / S2.sum().clamp_min(1e-12)
        eff_rank = float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item())

        # Between-group correlation (using per-sample group means as scalar representatives)
        grp_scalars = torch.stack(
            [X[:, groups[n]].mean(1) for n in names], dim=1
        )  # (N, G)
        gm = grp_scalars - grp_scalars.mean(0)
        gstd = gm.std(0).clamp_min(1e-8)
        gm_z = gm / gstd
        group_corr = (gm_z.T @ gm_z) / N  # (G, G)

        return {
            "r2_full":         r2_full,
            "partial_r2":      partial_r2,
            "semi_partial_r2": semi_partial_r2,
            "vif":             vif,
            "cond_num":        cond_num,
            "eff_rank":        eff_rank,
            "group_corr":      group_corr,
            "group_names":     names,
        }

    def _print_r2_diag_table(diag: dict, layer: int, tgt_label: str) -> None:
        """Print a formatted partial-R² and diagnostics table."""
        names = diag["group_names"]
        _W = 10
        hdr = (f"  {'Feature':<14}  {'pR²':>{_W}}  {'FWL pR²':>{_W}}  "
               f"{'VIF':>{_W}}")
        sep = "  " + "─" * (len(hdr) - 2)
        print(f"\n  ── Feature partial R² (layer {layer}, target={tgt_label}) ──")
        print(f"  Full model R² = {diag['r2_full']:.4f}  |  "
              f"Cond# = {diag['cond_num']:.1f}  |  "
              f"Eff. rank = {diag['eff_rank']:.2f} / {len(names)}")
        print(sep); print(hdr); print(sep)
        for name in names:
            pr2 = diag["partial_r2"].get(name, float("nan"))
            sp  = diag["semi_partial_r2"].get(name, float("nan"))
            vi  = diag["vif"].get(name, float("nan"))
            pr2_s = f"{pr2:.4f}" if np.isfinite(pr2) else "  n/a"
            sp_s  = f"{sp:.4f}"  if np.isfinite(sp)  else "  n/a"
            vi_s  = f"{vi:.2f}"  if np.isfinite(vi)  else "  n/a"
            print(f"  {name:<14}  {pr2_s:>{_W}}  {sp_s:>{_W}}  {vi_s:>{_W}}")
        print(sep)
        # Between-group correlation heatmap (text)
        corr = diag["group_corr"]
        G = len(names)
        col_w = max(len(n) for n in names)
        print(f"\n  Between-group correlation matrix:")
        header_row = "  " + " " * (col_w + 2) + "  ".join(
            f"{n[:6]:>6}" for n in names
        )
        print(header_row)
        for i, ni in enumerate(names):
            row = "  " + f"{ni[:col_w]:<{col_w}}  " + "  ".join(
                f"{corr[i, j].item():>6.2f}" for j in range(G)
            )
            print(row)
        print()

    # =====================================================================
    #  Phase 4: Per-layer loop
    # =====================================================================
    N_RAND_R2 = 3
    all_results = {}

    # Pre-compute commonly used tensors and functions
    eval_pos_idx = torch.tensor(eval_pos, device=device, dtype=torch.long)
    
    for l in layers:
        logger.info(f"[opt-dir] Layer {l} ...")

        # --- (a) Build protected subspace & orthogonal complement ---
        W_task_l = probes[l]["model_weight"].float()
        W_tok_l = probes[l]["token_weight"].float()
        U_orth_l, rk_task, rk_tok, rk_prot = _build_protected(W_task_l, W_tok_l)
        orth_dim = U_orth_l.shape[1]

        # --- (b) Optimise V_opt via gradient ascent on target CE loss ---
        V_opt, val_hist = _optimise_v(l, U_orth_l)
        V_cpu = V_opt.cpu().float()

        # --- (c) R^2 diagnostics (CPU): does V_opt encode bigram / enriched info? ---
        proj_tgt = h_tgt[l] @ V_cpu

        # Bigram-only R² (kept for backward compatibility with plots)
        r2_bi2v = ols_r2(bi_tgt, proj_tgt)
        r2_v2bi = ols_r2(proj_tgt, bi_tgt)

        # Full enriched R² — enrich_tgt is row-aligned with h_tgt (same sequences)
        r2_enrich2v = ols_r2(enrich_tgt, proj_tgt)
        r2_v2enrich = ols_r2(proj_tgt, enrich_tgt)

        # Marginal R² per individual feature group (forward: feat_i → V_opt)
        feat_r2_marginal = {
            name: ols_r2(enrich_tgt[:, sl], proj_tgt)
            for name, sl in feat_groups.items()
        }

        # Partial R², FWL semi-partial R², VIF, and design diagnostics
        diag_enrich = _partial_r2_and_diag(enrich_tgt, proj_tgt, feat_groups)
        _print_r2_diag_table(diag_enrich, l, tgt_label)

        U_orth_cpu = U_orth_l.cpu().float()
        rand_bi2v, rand_v2bi = [], []
        rand_enrich2v, rand_v2enrich = [], []
        for _ in range(N_RAND_R2):
            Vr, _ = torch.linalg.qr(
                U_orth_cpu @ torch.randn(orth_dim, n_directions),
            )
            rp = h_tgt[l] @ Vr
            rand_bi2v.append(ols_r2(bi_tgt, rp))
            rand_v2bi.append(ols_r2(rp, bi_tgt))
            rand_enrich2v.append(ols_r2(enrich_tgt, rp))
            rand_v2enrich.append(ols_r2(rp, enrich_tgt))
        r2_rand_bi2v = float(np.mean(rand_bi2v))
        r2_rand_v2bi = float(np.mean(rand_v2bi))
        r2_rand_enrich2v = float(np.mean(rand_enrich2v))
        r2_rand_v2enrich = float(np.mean(rand_v2enrich))

        # --- (d) Feature-explained decomposition ---
        # Fit a linear model (features → proj_tgt), project V_opt onto the fitted subspace,
        # then evaluate the causal effect of removing only the "explained" part of V_opt.
        # Using enrich_tgt (bigram_clr + one_hot_x_t + position) gives a richer filter than
        # bigram alone, so the "enriched" bar in fig_bigram shows the maximum linearly
        # explainable fraction of V_opt's causal power.
        proj_c = proj_tgt - proj_tgt.mean(0, keepdim=True)
        total_var = (proj_c ** 2).sum().item()
        bi_explained = {}
        for fname, feat in [("bigram_clr", bi_tgt), ("enriched", enrich_tgt)]:
            Nf = feat.shape[0]
            X_aug = torch.cat([feat, torch.ones(Nf, 1)], 1)
            W_fit = torch.linalg.pinv(X_aug) @ proj_tgt
            Y_hat = X_aug @ W_fit
            r2 = 1.0 - ((proj_tgt - Y_hat) ** 2).sum().item() / total_var \
                if total_var > 0 else 0.0
            _, Sp, Vtp = torch.linalg.svd(
                Y_hat - Y_hat.mean(0, keepdim=True), full_matrices=False,
            )
            frac = Sp ** 2 / max(total_var, 1e-12)
            rk = int((frac > 0.005).sum().item())
            V_expl = V_cpu @ Vtp[:rk].T if rk > 0 else torch.empty(D, 0)
            bi_explained[fname] = {
                "explained_rank": rk, "r2_feat_to_proj": r2,
                "V_explained": V_expl,
            }

        # --- (e) Causal evaluation ---
        P_v = (V_opt @ V_opt.T).to(device)
        res_maj = _eval_hooked(l, P_v, cache_maj, full=True)
        res_ood = _eval_hooked(l, P_v, cache_ood, full=True)
        res_minor = _eval_hooked(l, P_v, cache_minor, full=True)
        res_tgt = res_minor if opt_target == "minor" else res_ood

        # Random subspace baseline: n_rand_int random directions of same rank
        # evaluated on major, ood, and minor separately.
        rand_acc = {"major": [], "ood": [], "minor": []}
        rk = min(n_directions, orth_dim)
        for _ in range(n_rand_int):
            Vr, _ = torch.linalg.qr(
                U_orth_cpu @ torch.randn(orth_dim, rk),
            )
            Pr = (Vr @ Vr.T).to(device)
            for key, cache in [
                ("major", cache_maj), ("ood", cache_ood), ("minor", cache_minor),
            ]:
                bo, io = _eval_hooked(l, Pr, cache)
                rand_acc[key].append(io - bo)

        rand_stats = {k: trial_stats(v) for k, v in rand_acc.items()}
        rand_delta_tgt = rand_stats["minor" if opt_target == "minor" else "ood"]["mean"]

        for info in bi_explained.values():
            if info["explained_rank"] == 0:
                info["delta_tgt"] = 0.0
                info["delta_ood"] = 0.0
                continue
            Ve = info["V_explained"].to(device)
            P_e = Ve @ Ve.T
            bo, io = _eval_hooked(l, P_e, cache_tgt)
            info["delta_tgt"] = io - bo
            bo_o, io_o = _eval_hooked(l, P_e, cache_ood)
            info["delta_ood"] = io_o - bo_o
            del Ve, P_e

        # --- (f) Store results ---
        pct_m = 100 * res_maj["delta"] / res_maj["baseline"] \
            if res_maj["baseline"] > 0 else float("nan")
        pct_t = 100 * res_tgt["delta"] / res_tgt["baseline"] \
            if res_tgt["baseline"] > 0 else float("nan")

        all_results[l] = {
            "opt_target": opt_target,
            "tgt_label": tgt_label,
            "baseline_loss_major": res_maj["baseline"],
            "intervened_loss_major": res_maj["intervened"],
            "delta_loss_major": res_maj["delta"],
            "pct_increase_major": pct_m,
            "baseline_loss_ood": res_ood["baseline"],
            "intervened_loss_ood": res_ood["intervened"],
            "delta_loss_ood": res_ood["delta"],
            "baseline_loss_minor": res_minor["baseline"],
            "intervened_loss_minor": res_minor["intervened"],
            "delta_loss_minor": res_minor["delta"],
            "baseline_loss_tgt": res_tgt["baseline"],
            "intervened_loss_tgt": res_tgt["intervened"],
            "delta_loss_tgt": res_tgt["delta"],
            "pct_increase_tgt": pct_t,
            "baseline_per_pos_major": res_maj["baseline_per_pos"],
            "intervened_per_pos_major": res_maj["intervened_per_pos"],
            "baseline_per_pos_ood": res_ood["baseline_per_pos"],
            "intervened_per_pos_ood": res_ood["intervened_per_pos"],
            "baseline_per_pos_minor": res_minor["baseline_per_pos"],
            "intervened_per_pos_minor": res_minor["intervened_per_pos"],
            "baseline_per_pos_tgt": res_tgt["baseline_per_pos"],
            "intervened_per_pos_tgt": res_tgt["intervened_per_pos"],
            "eval_positions": res_maj["positions"],
            "layer": l, "scale": scale,
            "task_rank": rk_task, "token_rank": rk_tok,
            "protected_rank": rk_prot,
            "n_directions": n_directions,
            "directions": V_opt.cpu(),
            "val_history": val_hist,
            "feat_r2_marginal": feat_r2_marginal,
            "r2_bi2v_tgt": r2_bi2v,
            "r2_v2bi_tgt": r2_v2bi,
            "r2_rand_bi2v_tgt": r2_rand_bi2v,
            "r2_rand_v2bi_tgt": r2_rand_v2bi,
            "r2_rand_enrich2v_tgt": r2_rand_enrich2v,
            "r2_rand_v2enrich_tgt": r2_rand_v2enrich,
            "r2_enrich2v_tgt": r2_enrich2v,
            "r2_v2enrich_tgt": r2_v2enrich,
            "feat_partial_r2":       diag_enrich["partial_r2"],
            "feat_semi_partial_r2":  diag_enrich["semi_partial_r2"],
            "feat_vif":              diag_enrich["vif"],
            "design_cond_num":       diag_enrich["cond_num"],
            "design_eff_rank":       diag_enrich["eff_rank"],
            "feat_group_corr":       diag_enrich["group_corr"],
            "rand_int_delta_tgt": rand_delta_tgt,
            "rand_stats": rand_stats,
            "delta_per_batch_major": res_maj["delta_per_batch"],
            "delta_per_batch_ood": res_ood["delta_per_batch"],
            "delta_per_batch_minor": res_minor["delta_per_batch"],
            "bi_explained": {
                fn: {k: v for k, v in info.items() if k != "V_explained"}
                for fn, info in bi_explained.items()
            },
            "mean_logit_delta_tgt": res_tgt["mean_logit_delta"],
        }
        logger.info(
            f"  layer {l}: d_maj={res_maj['delta']:.4f} ({pct_m:.1f}%), "
            f"d_ood={res_ood['delta']:.4f}, d_minor={res_minor['delta']:.4f}, "
            f"R2(bi->V)={r2_bi2v:.3f}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =====================================================================
    #  Cleanup
    # =====================================================================
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # =====================================================================
    #  Post-processing: separate position-0 loss as reference
    # =====================================================================
    # Position 0 has no context, so its loss is the "uninformative" baseline.
    # gain = loss(pos0) - loss(pos>0)  is the max CE increase from removing
    # all learned information — used as a reference line in the bar chart.
    for l in layers:
        positions = list(all_results[l]["eval_positions"])
        pos0_idx = next(
            (i for i, p in enumerate(positions) if int(p) == 0), None,
        )
        if pos0_idx is not None:
            all_results[l]["pos0_baseline_major"] = \
                all_results[l]["baseline_per_pos_major"][pos0_idx]
            all_results[l]["pos0_baseline_ood"] = \
                all_results[l]["baseline_per_pos_ood"][pos0_idx]
            all_results[l]["pos0_baseline_minor"] = \
                all_results[l]["baseline_per_pos_minor"][pos0_idx]
            all_results[l]["pos0_baseline_tgt"] = \
                all_results[l]["baseline_per_pos_tgt"][pos0_idx]
            for key in ("major", "ood", "minor", "tgt"):
                bpp = [v for i, v in enumerate(
                    all_results[l][f"baseline_per_pos_{key}"]) if i != pos0_idx]
                ipp = [v for i, v in enumerate(
                    all_results[l][f"intervened_per_pos_{key}"]) if i != pos0_idx]
                all_results[l][f"baseline_loss_{key}"] = float(np.mean(bpp))
                all_results[l][f"intervened_loss_{key}"] = float(np.mean(ipp))
                all_results[l][f"delta_loss_{key}"] = (
                    all_results[l][f"intervened_loss_{key}"]
                    - all_results[l][f"baseline_loss_{key}"]
                )

    has_pos0 = "pos0_baseline_major" in all_results[layers[0]]
    if has_pos0:
        gain_maj = (
            float(np.mean([all_results[l]["pos0_baseline_major"] for l in layers]))
            - float(np.mean([all_results[l]["baseline_loss_major"] for l in layers]))
        )
        gain_ood = (
            float(np.mean([all_results[l]["pos0_baseline_ood"] for l in layers]))
            - float(np.mean([all_results[l]["baseline_loss_ood"] for l in layers]))
        )
        gain_minor = (
            float(np.mean([all_results[l]["pos0_baseline_minor"] for l in layers]))
            - float(np.mean([all_results[l]["baseline_loss_minor"] for l in layers]))
        )
        gain_tgt = gain_minor if opt_target == "minor" else gain_ood
    else:
        gain_maj = gain_ood = gain_minor = gain_tgt = None

    # =====================================================================
    #  Plots
    # =====================================================================
    def _style_ax(ax, xlabel, ylabel, title_str, xticks=None):
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title("", fontsize=18)
        if xticks is not None:
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(t) for t in xticks])
        ax.tick_params(labelsize=12)
        ax.grid(axis="y", alpha=0.3)

    x = np.arange(len(layers))

    # -- Plot 1: Δ𝓛/g (%) bar chart — normalized by ICL gain --
    # 3 solid bars per layer (Major / OOD / Minor), each Δ𝓛/g × 100 (%).
    # Black error bars = IQR across eval batches (also normalized).
    # Shaded band = random same-rank subspace (OOD, normalized).

    MC = orth_bar_offsets(has_minor=True)

    # Normalized deltas (% of ICL gain disrupted); fall back to raw if gains unknown
    if gain_maj is not None:
        norm_maj_l   = [all_results[l]["delta_loss_major"]  / gain_maj   * 100 for l in layers]
        norm_ood_l   = [all_results[l]["delta_loss_ood"]    / gain_ood   * 100 for l in layers]
        norm_minor_l = [all_results[l]["delta_loss_minor"]  / gain_minor * 100 for l in layers]
        _ylabel = "Fraction of ICL gain disrupted (%)"
        _rand_q75_norm = max(
            all_results[l]["rand_stats"]["ood"]["q75"] / gain_ood * 100 for l in layers
        )
    else:
        norm_maj_l   = [all_results[l]["delta_loss_major"]  for l in layers]
        norm_ood_l   = [all_results[l]["delta_loss_ood"]    for l in layers]
        norm_minor_l = [all_results[l]["delta_loss_minor"]  for l in layers]
        _ylabel = "Cross-entropy loss increase"
        _rand_q75_norm = max(
            all_results[l]["rand_stats"][mk]["q75"]
            for l in layers for mk in ("major", "ood", "minor")
        )

    fig_delta, ax = plt.subplots(figsize=figsize, dpi=150)

    ax.axhspan(0, _rand_q75_norm, color=ORTH_RANDOM_BAND_COLOR,
               alpha=ORTH_RANDOM_BAND_ALPHA, zorder=1,
               hatch=ORTH_RANDOM_BAND_HATCH, label="Random")
    ax.axhline(_rand_q75_norm, color=ORTH_REFERENCE_LINE_COLOR, lw=1.2, ls="-",
               zorder=2, alpha=0.85)

    for mode, norm_vals, (key_pb, g_m), label in [
        ("maj",   norm_maj_l,   ("delta_per_batch_major", gain_maj   or 1.0), "Maj."),
        ("ood",   norm_ood_l,   ("delta_per_batch_ood",   gain_ood   or 1.0), "OOD"),
        ("minor", norm_minor_l, ("delta_per_batch_minor", gain_minor or 1.0), "Min."),
    ]:
        c  = ORTH_COLORS[mode]
        xm = x + MC[mode]
        lo, hi = iqr_err_norm(all_results, layers, key_pb, g_m)
        ax.bar(xm, norm_vals, ORTH_BAR_WIDTH, color=c, linewidth=0, zorder=3, label=label)
        ax.errorbar(xm, norm_vals, yerr=[lo, hi], fmt="none",
                    ecolor="black", elinewidth=0.9, capsize=3, capthick=0.9, zorder=5)

    if gain_maj is not None:
        ax.axhline(100, color="grey", ls="--", lw=1.0, alpha=0.55,
                   label="100%")

    ax.set_xlabel("Layer", fontsize=9)
    if show_ylabel:
        ax.set_ylabel(_ylabel, fontsize=9)
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
    _W = 7
    _hd = (f"  {'Layer':>5}  {'Maj.%':>{_W}}  {'OOD%':>{_W}}  {'Min.%':>{_W}}"
           f"  {'Rand.%':>{_W}}  |  {'Maj.Δ':>{_W}}  {'OOD Δ':>{_W}}  {'Min.Δ':>{_W}}")
    _ln = "  " + "─" * (len(_hd) - 2)
    print(f"\n  V_opt Intervention — Δ𝓛/g (% of ICL gain)  [CE nats]")
    print(_ln); print(_hd); print(_ln)
    for _l in layers:
        _r = all_results[_l]
        _nm = _r["delta_loss_major"] / (gain_maj or 1.0)   * 100
        _no = _r["delta_loss_ood"]   / (gain_ood or 1.0)   * 100
        _nn = _r["delta_loss_minor"]  / (gain_minor or 1.0) * 100
        _nr = _r["rand_stats"][_r["opt_target"]]["mean"] / (gain_tgt or 1.0) * 100
        print(f"  {_l:>5}  {_nm:>{_W}.1f}%  {_no:>{_W}.1f}%  {_nn:>{_W}.1f}%"
              f"  {_nr:>{_W}.1f}%  |  "
              f"{_r['delta_loss_major']:>{_W}.4f}  {_r['delta_loss_ood']:>{_W}.4f}"
              f"  {_r['delta_loss_minor']:>{_W}.4f}")
    print(_ln)
    _fmt = lambda v: f"{v:{_W}.4f}" if v is not None else f"{'n/a':>{_W}}"
    print(f"  {'Gain':>5}  {'':>{_W}}   {'':>{_W}}   {'':>{_W}}   {'':>{_W}}   |  "
          f"{_fmt(gain_maj)}  {_fmt(gain_ood)}  {_fmt(gain_minor)}")
    print()

    # ── Printed Table 2: Layer-averaged ───────────────────────────────────
    if gain_maj is not None:
        _rand_norm_list = [
            all_results[l]["rand_stats"][all_results[l]["opt_target"]]["mean"]
            / (gain_tgt or 1.0) * 100 for l in layers
        ]
        _mean_m = float(np.mean(norm_maj_l));   _std_m = float(np.std(norm_maj_l))
        _mean_o = float(np.mean(norm_ood_l));   _std_o = float(np.std(norm_ood_l))
        _mean_n = float(np.mean(norm_minor_l)); _std_n = float(np.std(norm_minor_l))
        _mean_r = float(np.mean(_rand_norm_list)); _std_r = float(np.std(_rand_norm_list))
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
        print(f"  {'Maj.':<6}  {_mean_m:>7.1f}±{_std_m:<4.1f}  {_mean_dm:{_WA}.4f}  {gain_maj:{_WA}.4f}")
        print(f"  {'OOD':<6}  {_mean_o:>7.1f}±{_std_o:<4.1f}  {_mean_do:{_WA}.4f}  {gain_ood:{_WA}.4f}")
        print(f"  {'Min.':<6}  {_mean_n:>7.1f}±{_std_n:<4.1f}  {_mean_dn:{_WA}.4f}  {gain_minor:{_WA}.4f}")
        print(f"  {'Rand.':<6}  {_mean_r:>7.1f}±{_std_r:<4.1f}  {'---':>{_WA}}  {'---':>{_WA}}")
        print(_sep)
        print()
    # ─────────────────────────────────────────────────────────────────────

    # -- Plot 2: Validation loss history --
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
    _style_ax(ax_loss, "Step", f"{tgt_label} Val Loss (intervened)", "")
    ax_loss.legend(fontsize=12)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig_loss)

    # -- Plot 3: Feature R² — individual components + dashed combined --
    _LATENT_FEAT_DISPLAY = [
        ("bigram_clr",  "Bigram CLR",    "o-",  "#FF9800"),
        ("one_hot_x_t", "Current token", "s-",  "#4CAF50"),
        ("position",    "Position",      "^-",  "#9C27B0"),
    ]

    fig_r2_fwd, ax_r = plt.subplots(figsize=(6, 4.5), dpi=150)
    for name, label, style, color in _LATENT_FEAT_DISPLAY:
        vals = [all_results[l]["feat_r2_marginal"].get(name, float("nan"))
                for l in layers]
        ax_r.plot(layers, vals, style, label=label, color=color, lw=2, ms=6)
    combined_r2 = [all_results[l]["r2_enrich2v_tgt"] for l in layers]
    ax_r.plot(layers, combined_r2, ls="--", color="black", lw=1.8,
              label="Combined", zorder=2)
    _style_ax(ax_r, "Layer", r"$R^2$", "", xticks=layers)
    ax_r.legend(fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18),
                framealpha=0.9, edgecolor="lightgrey",
                columnspacing=0.5, handlelength=1.0, handletextpad=0.3, borderpad=0.4)
    ax_r.set_ylim(-0.05, 1.05)
    plt.tight_layout(pad=2.0)

    # fig_r2_rev is kept for API compatibility but mirrors the forward figure
    fig_r2_rev = fig_r2_fwd
    if show:
        plt.figure(fig_r2_fwd.number); plt.show()
    else:
        plt.close(fig_r2_fwd)

    # -- Plot 4: Per-token logit change --
    fig_logit, ax_ld = plt.subplots(figsize=(10, 5))
    x_tok = np.arange(V)
    bw_tok = 0.8 / len(layers)
    for idx, l in enumerate(layers):
        ax_ld.bar(
            x_tok + (idx - len(layers) / 2 + 0.5) * bw_tok,
            all_results[l]["mean_logit_delta_tgt"],
            bw_tok, label=f"L{l}", alpha=0.75,
        )
    _style_ax(ax_ld, "Token", "Mean d logit", "", xticks=x_tok)
    ax_ld.legend(fontsize=10)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig_logit)

    # -- Plot 5: Prediction-explained intervention (normalized) --
    # Bar geometry: 4 flush bars per layer, interleaved by condition
    # Order: [V_opt OOD | Pred OOD | V_opt tgt | Pred tgt] — no gaps
    C_OOD = ORTH_COLORS["ood"]
    C_TGT = ORTH_COLORS["minor"] if opt_target == "minor" else ORTH_COLORS["ood"]
    ALPHA_PRED = 0.42

    bw_p       = 0.17
    total_w_bi = 4 * bw_p
    left_edge  = -total_w_bi / 2

    xo_vopt_ood = left_edge + 0.5 * bw_p
    xo_pred_ood = left_edge + 1.5 * bw_p
    xo_vopt_tgt = left_edge + 2.5 * bw_p
    xo_pred_tgt = left_edge + 3.5 * bw_p

    # Enriched-feature explained subspace
    fn = "enriched"
    _g_ood_n  = gain_ood  or 1.0
    _g_tgt_n  = gain_tgt  or 1.0
    vals_ood = [all_results[l]["bi_explained"][fn]["delta_ood"] / _g_ood_n * 100
                for l in layers]
    vals_tgt = [all_results[l]["bi_explained"][fn]["delta_tgt"] / _g_tgt_n * 100
                for l in layers]

    fig_bigram, ax_bi = plt.subplots(figsize=figsize, dpi=150)

    ax_bi.bar(x + xo_vopt_ood,
              [all_results[l]["delta_loss_ood"] / _g_ood_n * 100 for l in layers],
              bw_p, label=r"$V_{\rm opt}$ (OOD)",
              color=C_OOD, linewidth=0, zorder=3)

    ax_bi.bar(x + xo_pred_ood, vals_ood, bw_p,
              label="Filtered (OOD)",
              color=C_OOD, alpha=ALPHA_PRED, linewidth=0, zorder=3)

    ax_bi.bar(x + xo_vopt_tgt,
              [all_results[l]["delta_loss_tgt"] / _g_tgt_n * 100 for l in layers],
              bw_p, label=rf"$V_{{\rm opt}}$ ({tgt_abbr})",
              color=C_TGT, linewidth=0, zorder=3)

    ax_bi.bar(x + xo_pred_tgt, vals_tgt, bw_p,
              label=f"Filtered ({tgt_abbr})",
              color=C_TGT, alpha=ALPHA_PRED, linewidth=0, zorder=3)

    if gain_tgt is not None:
        ax_bi.axhline(100, color="grey", ls="--", lw=1.0, alpha=0.55,
                      label="100%")

    ax_bi.set_xlabel("Layer", fontsize=9)
    if show_ylabel:
        ax_bi.set_ylabel("Fraction of ICL gain disrupted (%)" if gain_tgt is not None
                         else "Cross-entropy loss increase", fontsize=9)
    ax_bi.set_xticks(x)
    ax_bi.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax_bi.tick_params(axis="y", labelsize=8)
    ax_bi.yaxis.grid(True, alpha=0.25, linewidth=0.5, color="grey")
    ax_bi.set_axisbelow(True)
    ax_bi.spines["top"].set_visible(False)
    ax_bi.spines["right"].set_visible(False)
    ax_bi.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14),
                 ncol=5, framealpha=0.9, edgecolor="lightgrey",
                 columnspacing=0.5, handlelength=1.0, handletextpad=0.3, borderpad=0.4)
    plt.tight_layout(pad=2.0)
    if show:
        plt.show()
    else:
        plt.close(fig_bigram)

    # ── Printed Table 5: V_opt & Prediction ──────────────────────────────
    _W5 = 9
    _hd5 = (f"  {'Layer':>5}  {'Vopt OOD%':>{_W5}}  {'Pred OOD%':>{_W5}}  "
            f"{'Vopt '+tgt_abbr+'%':>{_W5}}  {'Pred '+tgt_abbr+'%':>{_W5}}")
    _ln5 = "  " + "─" * (len(_hd5) - 2)
    print(f"\n  V_opt & Enriched filter — Δ𝓛/g (% of ICL gain)  (feat: {fn})")
    print(_ln5); print(_hd5); print(_ln5)
    for _l in layers:
        _r = all_results[_l]
        _vo = _r["delta_loss_ood"] / _g_ood_n * 100
        _po = _r["bi_explained"][fn]["delta_ood"] / _g_ood_n * 100
        _vt = _r["delta_loss_tgt"] / _g_tgt_n * 100
        _pt = _r["bi_explained"][fn]["delta_tgt"] / _g_tgt_n * 100
        print(f"  {_l:>5}  {_vo:>{_W5}.1f}%  {_po:>{_W5}.1f}%  "
              f"{_vt:>{_W5}.1f}%  {_pt:>{_W5}.1f}%")
    print(_ln5)
    print()
    # ─────────────────────────────────────────────────────────────────────

    return fig_delta, fig_loss, fig_r2_fwd, fig_r2_rev, fig_logit, fig_bigram, all_results
