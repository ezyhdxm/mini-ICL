"""Optimal orthogonal direction intervention (latent Markov, non-padded)."""

import gc
import numpy as np
import torch
from typing import Optional

import icl.utils.notebook_utils as nu
from icl.latent_markov.analysis.ood import get_latent_sampler
from icl.utils.logger import setup_logger
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

    def _svd_basis(M, var_thresh=None):
        _, S, Vt = torch.linalg.svd(M, full_matrices=False)
        r = int((S > 1e-6 * S[0]).sum().item())
        if var_thresh is not None and r > 0:
            cum = torch.cumsum(S[:r] ** 2, 0)
            r = min(r, int((cum < var_thresh * cum[-1]).sum().item()) + 1)
        return Vt[:r].T, r

    B_task, rk_task = _svd_basis(W_task)
    B_tok, rk_tok = _svd_basis(W_tok, var_thresh=token_var_threshold)
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

        hnd = model.layers[layer].attn_block.register_forward_hook(_hook)
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
            handle = model.layers[layer].attn_block.register_forward_hook(_hk)
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

    def _ols_r2(X, Y):
        """OLS R^2 on held-out 20%:  Y ~ X * W + b."""
        N = X.shape[0]
        nt = int(0.8 * N)
        pm = torch.randperm(N)
        Xa = torch.cat([X[pm[:nt]], torch.ones(nt, 1)], 1)
        W = torch.linalg.pinv(Xa) @ Y[pm[:nt]]
        pred = torch.cat([X[pm[nt:]], torch.ones(N - nt, 1)], 1) @ W
        Yte = Y[pm[nt:]]
        ss_r = ((Yte - pred) ** 2).sum().item()
        ss_t = ((Yte - Yte.mean(0)) ** 2).sum().item()
        return 1.0 - ss_r / ss_t if ss_t > 0 else float("nan")

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
        o = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(ep):
            loss = ((net(Xtr) - Ytr) ** 2).mean()
            o.zero_grad(); loss.backward(); o.step()
        with torch.no_grad():
            ss_r = ((Yte - net(Xte)) ** 2).sum().item()
            ss_t = ((Yte - Yte.mean(0)) ** 2).sum().item()
        return 1.0 - ss_r / ss_t if ss_t > 0 else float("nan")

    # R^2(bigram -> proj): can bigram stats predict the V_opt projection?
    # R^2(proj -> bigram): does V_opt projection predict bigram stats?
    r2_bi2v = _ols_r2(bi_ood, proj_ood)
    r2_v2bi = _ols_r2(proj_ood, bi_ood)
    r2_bi2v_mlp = _mlp_r2(bi_ood, proj_ood)
    r2_v2bi_mlp = _mlp_r2(proj_ood, bi_ood)

    # Baseline: R^2 for random rank-k subspace in orth complement
    U_orth_cpu = U_orth.cpu().float()
    N_RAND = 3
    rand_bi2v, rand_v2bi = [], []
    for _ in range(N_RAND):
        Vr, _ = torch.linalg.qr(U_orth_cpu @ torch.randn(orth_dim, n_directions))
        rp = h_ood @ Vr
        rand_bi2v.append(_ols_r2(bi_ood, rp))
        rand_v2bi.append(_ols_r2(rp, bi_ood))
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
            def _hook(mod, inp, out, _P=P_v):
                h = out if torch.is_tensor(out) else out[0]
                hm = h - scale * (h @ _P)
                return hm if torch.is_tensor(out) else (hm,) + out[1:]
            handle = model.layers[layer].attn_block.register_forward_hook(_hook)
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
        "r2_bi2v_mlp_ood": r2_bi2v_mlp,
        "r2_v2bi_mlp_ood": r2_v2bi_mlp,
        "r2_rand_bi2v_ood": r2_rand_bi2v,
        "r2_rand_v2bi_ood": r2_rand_v2bi,
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
              f"bi->V(mlp)={r2_bi2v_mlp:.3f}  rand={r2_rand_bi2v:.3f}")
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

    **Outputs** (5 figures + result dict):
      fig_delta  :  CE loss increase bars (major + target)
      fig_loss   :  Optimisation loss history
      fig_r2     :  Bigram <-> V_opt projection R^2 (target)
      fig_logit  :  Per-token logit change under intervention (target)
      fig_bigram :  V_opt full vs bigram-explained causal effect (target)

    Returns (fig_delta, fig_loss, fig_r2, fig_logit, fig_bigram, all_results).
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
    del sampler_tmp

    if fit_positions is None:
        fit_positions = list(range(100, seq_len))
    if eval_positions is None:
        eval_positions = list(range(seq_len))
    eval_pos = list(eval_positions)
    if 0 not in eval_pos:
        eval_pos = [0] + eval_pos

    # =====================================================================
    #  Phase 1: Fit joint probes (all layers, one forward pass)
    # =====================================================================
    # From the joint probe  h = posterior * W_task + onehot(x_t) * W_tok + b
    # we extract W_task and W_tok per layer.
    logger.info("[opt-dir] Phase 1: Fitting probes ...")
    probe_data = _collect_multi_layer_data(
        exp_name, layers, B=B, n_samples=fit_n_samples,
        step=step, n_minor=-1, positions=fit_positions,
        sample_mode="major",
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

    def _gen_and_cache(sampler, mode, n):
        """Generate sequences, run unhooked forward pass, cache logits."""
        seqs, logits = [], []
        for _ in range(max(1, (n + B - 1) // B)):
            g = sampler.generate(mode=mode, task=None, num_samples=B, epochs=1)
            s = (g[0] if isinstance(g, (tuple, list)) else g)
            if s.dim() == 3:
                s = s.squeeze(0)
            s = s.to(device)
            with torch.no_grad():
                logits.append(model(s).cpu())
            seqs.append(s.cpu())
        return seqs, logits

    cache_maj = _gen_and_cache(sampler_major, "major", n_samples_eval)
    cache_ood = _gen_and_cache(sampler_ood, "minor", n_samples_eval)
    cache_minor = _gen_and_cache(sampler_minor, "minor", n_samples_eval)
    cache_tgt = cache_minor if opt_target == "minor" else cache_ood

    # =====================================================================
    #  Phase 3: Collect hidden states + bigram features (all layers, one pass)
    # =====================================================================
    # Bigram CLR: centred log-ratio of empirical bigram counts P(.|x_t)
    # from the prefix x_1..x_t.  This captures the "token-counting" info
    # that the model can compute without any latent-task knowledge.
    eval_pos_idx = torch.tensor(eval_pos, device=device, dtype=torch.long)

    def _bigram_clr(samples):
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
        """Forward pass with hooks at all layers; also compute bigram CLR."""
        h_acc = {l: [] for l in layers}
        bi_acc = []
        for _ in range(max(1, (n + B - 1) // B)):
            g = sampler.generate(mode=mode, task=None, num_samples=B, epochs=1)
            s = (g[0] if isinstance(g, (tuple, list)) else g)
            if s.dim() == 3:
                s = s.squeeze(0)
            s = s.to(device)
            caches = {}
            handles = []
            for ll in layers:
                def _hook(mod, inp, out, _l=ll):
                    h = out if torch.is_tensor(out) else out[0]
                    caches[_l] = h.index_select(1, eval_pos_idx).detach()
                handles.append(
                    model.layers[ll].attn_block.register_forward_hook(_hook)
                )
            with torch.no_grad():
                model(s)
            for hh in handles:
                hh.remove()
            for ll in layers:
                h_acc[ll].append(caches[ll].cpu())
            bi_acc.append(_bigram_clr(s).index_select(1, eval_pos_idx).cpu())
            del s, caches
        return (
            {ll: torch.cat(h_acc[ll]).reshape(-1, D).float() for ll in layers},
            torch.cat(bi_acc).reshape(-1, V).float(),
        )

    logger.info("[opt-dir] Phase 3: Collecting hiddens + bigrams ...")
    h_maj, bi_maj = _collect_h_and_bigram(sampler_major, "major", n_samples_probe)
    h_tgt, bi_tgt = _collect_h_and_bigram(sampler_tgt, "minor", n_samples_probe)

    # =====================================================================
    #  Shared helpers
    # =====================================================================
    ce_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def _eval_hooked(layer_idx, P, cached, full=False):
        """Forward pass with hook  h' = h - scale * h * P  at one layer.

        P is a D x D projector onto the subspace to ablate.
        Baseline logits come from cache (no redundant unhooked passes).
        If full=True, also returns per-position losses and logit deltas.
        """
        seqs, base_logits = cached
        bp_pos = {p: [] for p in eval_pos} if full else None
        ip_pos = {p: [] for p in eval_pos} if full else None
        b_all, i_all = [], []
        ld_sum = torch.zeros(V) if full else None
        ld_n = 0
        for s_cpu, bl_cpu in zip(seqs, base_logits):
            s = s_cpu.to(device)
            def _hook(mod, inp, out, _P=P):
                h = out if torch.is_tensor(out) else out[0]
                hm = h - scale * (h @ _P)
                return hm if torch.is_tensor(out) else (hm,) + out[1:]
            handle = model.layers[layer_idx].attn_block.register_forward_hook(_hook)
            try:
                with torch.no_grad():
                    li = model(s)
            finally:
                handle.remove()
            bl = bl_cpu.to(device)
            for p in eval_pos:
                if p + 1 >= s.shape[1]:
                    continue
                tgt = s[:, p + 1].long()
                bv = ce_fn(bl[:, p], tgt).mean().item()
                iv = ce_fn(li[:, p], tgt).mean().item()
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
        bpp = [float(np.mean(bp_pos[p])) for p in eval_pos if bp_pos[p]]
        ipp = [float(np.mean(ip_pos[p])) for p in eval_pos if ip_pos[p]]
        return {
            "baseline": ba, "intervened": ia, "delta": ia - ba,
            "baseline_per_pos": bpp, "intervened_per_pos": ipp,
            "positions": [p for p in eval_pos if bp_pos[p]],
            "mean_logit_delta": (ld_sum / max(ld_n, 1)).numpy(),
        }

    def _ols_r2(X, Y):
        """OLS regression R^2 on held-out 20%:  Y ~ X * W + b."""
        N = X.shape[0]
        nt = int(0.8 * N)
        pm = torch.randperm(N)
        Xa = torch.cat([X[pm[:nt]], torch.ones(nt, 1)], 1)
        W = torch.linalg.pinv(Xa) @ Y[pm[:nt]]
        pred = torch.cat([X[pm[nt:]], torch.ones(N - nt, 1)], 1) @ W
        Yte = Y[pm[nt:]]
        ss_r = ((Yte - pred) ** 2).sum().item()
        ss_t = ((Yte - Yte.mean(0)) ** 2).sum().item()
        return 1.0 - ss_r / ss_t if ss_t > 0 else float("nan")

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

        def _svd_basis(M, var_thresh=None):
            _, S, Vt = torch.linalg.svd(M, full_matrices=False)
            r = int((S > 1e-6 * S[0]).sum().item())
            if var_thresh is not None and r > 0:
                cum = torch.cumsum(S[:r] ** 2, 0)
                r = min(r, int((cum < var_thresh * cum[-1]).sum().item()) + 1)
            return Vt[:r].T, r

        B_task, rk_task = _svd_basis(W_task)
        B_tok, rk_tok = _svd_basis(W_tok, var_thresh=token_var_threshold)
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

        def _hook(mod, inp, out, _P=P):
            h = out if torch.is_tensor(out) else out[0]
            return (h - scale * (h @ _P)) if torch.is_tensor(out) \
                else (h - scale * (h @ _P),) + out[1:]

        hnd = model.layers[layer_idx].attn_block.register_forward_hook(_hook)
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
        """
        orth_dim = U_orth.shape[1]
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
        history = []
        val_history = []

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

            hnd = model.layers[layer_idx].attn_block.register_forward_hook(_hook)
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

            if step_i % val_every == 0 or step_i == n_opt_steps - 1:
                with torch.no_grad():
                    V_cur, _ = torch.linalg.qr(U_orth @ best_ema)
                val_history.append((step_i, _val_loss(layer_idx, V_cur)))

            del s, logits, acc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if patience > 0 and stale >= patience:
                break

        with torch.no_grad():
            V_opt, _ = torch.linalg.qr(U_orth @ best_ema)
        return V_opt, val_history

    # =====================================================================
    #  Phase 4: Per-layer loop
    # =====================================================================
    N_RAND_R2 = 3
    N_RAND_INT = 2
    all_results = {}

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

        # --- (c) R^2 diagnostics (CPU): does V_opt encode bigram info? ---
        proj_tgt = h_tgt[l] @ V_cpu
        r2_bi2v = _ols_r2(bi_tgt, proj_tgt)
        r2_v2bi = _ols_r2(proj_tgt, bi_tgt)
        r2_bi2v_mlp = _mlp_r2(bi_tgt, proj_tgt)
        r2_v2bi_mlp = _mlp_r2(proj_tgt, bi_tgt)

        U_orth_cpu = U_orth_l.cpu().float()
        rand_bi2v, rand_v2bi = [], []
        for _ in range(N_RAND_R2):
            Vr, _ = torch.linalg.qr(
                U_orth_cpu @ torch.randn(orth_dim, n_directions),
            )
            rp = h_tgt[l] @ Vr
            rand_bi2v.append(_ols_r2(bi_tgt, rp))
            rand_v2bi.append(_ols_r2(rp, bi_tgt))
        r2_rand_bi2v = float(np.mean(rand_bi2v))
        r2_rand_v2bi = float(np.mean(rand_v2bi))

        # --- (d) Bigram-explained decomposition ---
        proj_c = proj_tgt - proj_tgt.mean(0, keepdim=True)
        total_var = (proj_c ** 2).sum().item()
        dir_pred = torch.softmax(bi_tgt, dim=-1)
        bi_explained = {}
        for fname, feat in [("bigram_clr", bi_tgt), ("dirichlet_pred", dir_pred)]:
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

        rand_deltas = []
        for _ in range(N_RAND_INT):
            rk = min(n_directions, orth_dim)
            Vr, _ = torch.linalg.qr(
                U_orth_cpu @ torch.randn(orth_dim, rk),
            )
            Pr = (Vr @ Vr.T).to(device)
            bo, io = _eval_hooked(l, Pr, cache_tgt)
            rand_deltas.append(io - bo)
        rand_delta_tgt = float(np.mean(rand_deltas))

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
            "r2_bi2v_tgt": r2_bi2v,
            "r2_v2bi_tgt": r2_v2bi,
            "r2_bi2v_mlp_tgt": r2_bi2v_mlp,
            "r2_v2bi_mlp_tgt": r2_v2bi_mlp,
            "r2_rand_bi2v_tgt": r2_rand_bi2v,
            "r2_rand_v2bi_tgt": r2_rand_v2bi,
            "rand_int_delta_tgt": rand_delta_tgt,
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

    # -- Plot 1: CE loss increase (major + OOD + minor bars) --
    delta_maj = [all_results[l]["delta_loss_major"] for l in layers]
    delta_ood = [all_results[l]["delta_loss_ood"] for l in layers]
    delta_minor = [all_results[l]["delta_loss_minor"] for l in layers]
    bw3 = 0.25

    fig_delta, ax = plt.subplots(figsize=figsize)
    ax.bar(x - bw3, delta_maj, bw3, label="Major", color="#2196F3", alpha=0.85)
    ax.bar(x, delta_ood, bw3, label="OOD", color="#FF9800", alpha=0.85)
    ax.bar(x + bw3, delta_minor, bw3, label="Minor", color="#4CAF50", alpha=0.85)
    if gain_maj is not None:
        ax.axhline(gain_maj, color="#1565C0", ls="--", lw=1.8, alpha=0.7,
                   label=f"Major $l_0 - l_t$ ({gain_maj:.3f})")
        ax.axhline(gain_ood, color="#E65100", ls=":", lw=1.8, alpha=0.7,
                   label=f"OOD $l_0 - l_t$ ({gain_ood:.3f})")
        ax.axhline(gain_minor, color="#2E7D32", ls="-.", lw=1.8, alpha=0.7,
                   label=f"Minor $l_0 - l_t$ ({gain_minor:.3f})")
    for i, (vm, vo, vn) in enumerate(zip(delta_maj, delta_ood, delta_minor)):
        ax.text(x[i] - bw3, vm, f"{vm:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(x[i], vo, f"{vo:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(x[i] + bw3, vn, f"{vn:.3f}", ha="center", va="bottom", fontsize=9)
    _style_ax(ax, "Layer", "CE Loss Increase", "")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.legend(fontsize=10, loc="best")
    fig_delta.suptitle("", fontsize=18, y=1.02)
    plt.tight_layout()
    if save_path:
        fig_delta.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig_delta)

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

    # -- Plot 3: Bigram <-> V_opt R² --
    fig_r2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figsize)
    for ax_r, direction, key_v, key_mlp, key_rand, title_str in [
        (ax_a, "Bigram -> Proj", "r2_bi2v_tgt", "r2_bi2v_mlp_tgt",
         "r2_rand_bi2v_tgt",
         f"Bigram → $V_{{opt}}$ Projection $R^2$ ({tgt_label})"),
        (ax_b, "Proj -> Bigram", "r2_v2bi_tgt", "r2_v2bi_mlp_tgt",
         "r2_rand_v2bi_tgt",
         f"$V_{{opt}}$ Projection → Bigram $R^2$ ({tgt_label})"),
    ]:
        ax_r.plot(layers, [all_results[l][key_v] for l in layers],
                  "s-", label="$V_{opt}$ Linear", color="#FF9800", lw=2, ms=7)
        ax_r.plot(layers, [all_results[l][key_mlp] for l in layers],
                  "s--", label="$V_{opt}$ MLP", color="#E91E63", lw=2, ms=7)
        ax_r.plot(layers, [all_results[l][key_rand] for l in layers],
                  "v:", label="Rand orth Linear", color="gray", lw=1.5,
                  alpha=0.6, ms=6)
        _style_ax(ax_r, "Layer", "$R^2$", "", xticks=layers)
        ax_r.legend(fontsize=11)
        ax_r.set_ylim(-0.05, 1.05)
    fig_r2.suptitle("", fontsize=18, y=1.02)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig_r2)

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

    # -- Plot 5: Bigram-explained intervention --
    print("=== Major CE Loss Summary (V_opt removal) ===")
    for l in layers:
        r = all_results[l]
        print(f"  Layer {l}: baseline={r['baseline_loss_major']:.4f}, "
              f"d={r['delta_loss_major']:.4f} ({r['pct_increase_major']:.1f}%)")

    feat_names = list(all_results[layers[0]]["bi_explained"].keys())
    feat_colors = ["#E91E63", "#9C27B0"]
    n_groups = 2 + 2 * len(feat_names)
    total_w = 0.8
    bw_bi = total_w / n_groups

    fig_bigram, ax_bi = plt.subplots(figsize=figsize)
    ax_bi.bar(
        x - total_w / 2 + bw_bi / 2,
        [all_results[l]["delta_loss_ood"] for l in layers],
        bw_bi, label="V_opt (OOD)", color="#FF9800", alpha=0.85,
    )
    ax_bi.bar(
        x - total_w / 2 + 1.5 * bw_bi,
        [all_results[l]["delta_loss_tgt"] for l in layers],
        bw_bi, label=f"V_opt ({tgt_label})", color="#FF9800", alpha=0.35,
        edgecolor="#E65100", linewidth=1.2,
    )
    for fi, fn in enumerate(feat_names):
        vals_ood = [all_results[l]["bi_explained"][fn]["delta_ood"]
                    for l in layers]
        vals_tgt = [all_results[l]["bi_explained"][fn]["delta_tgt"]
                    for l in layers]
        r2_avg = float(np.mean([
            all_results[l]["bi_explained"][fn]["r2_feat_to_proj"]
            for l in layers
        ]))
        rk_avg = int(round(np.mean([
            all_results[l]["bi_explained"][fn]["explained_rank"]
            for l in layers
        ])))
        base_idx = 2 + fi * 2
        c = feat_colors[fi % len(feat_colors)]
        ax_bi.bar(
            x - total_w / 2 + (base_idx + 0.5) * bw_bi, vals_ood, bw_bi,
            label=f"{fn} (OOD)",
            color=c, alpha=0.85,
        )
        ax_bi.bar(
            x - total_w / 2 + (base_idx + 1.5) * bw_bi, vals_tgt, bw_bi,
            label=f"{fn} ({tgt_label}, R\u00b2={r2_avg:.2f}, rk={rk_avg})",
            color=c, alpha=0.35, edgecolor=c, linewidth=1.2,
        )
    if gain_tgt is not None:
        tgt_line_color = "#2E7D32" if opt_target == "minor" else "#E65100"
        ax_bi.axhline(gain_tgt, color=tgt_line_color, ls=":", lw=1.8,
                      alpha=0.7,
                      label=f"{tgt_label} $l_0 - l_t$ ({gain_tgt:.3f})")
    if gain_ood is not None:
        ax_bi.axhline(gain_ood, color="#E65100", ls="--", lw=1.8, alpha=0.7,
                      label=f"OOD $l_0 - l_t$ ({gain_ood:.3f})")
    _style_ax(ax_bi, "Layer", "\u0394 Loss", "")
    ax_bi.set_xticks(x)
    ax_bi.set_xticklabels([str(l) for l in layers])
    ax_bi.legend(fontsize=9, loc="best")
    fig_bigram.suptitle("", fontsize=18, y=1.02)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig_bigram)

    return fig_delta, fig_loss, fig_r2, fig_logit, fig_bigram, all_results
