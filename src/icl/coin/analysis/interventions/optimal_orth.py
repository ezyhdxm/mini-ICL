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
from icl.coin.analysis.probes import train_linear_hidden_predictor_coin
from icl.coin.coin_ood_analysis import get_new_sampler
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)


def intervene_optimal_orth_direction_coin(
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
    unigram_transform: str = "clr",
    unigram_alpha: float = 0.5,
    extraction_point: str = "post_attn",
    verbose: bool = False,
    print_summary: bool = True,
) -> dict:
    """
    Find a rank-``n_directions`` subspace in the orthogonal complement of
    the task subspace that maximally increases OOD loss when removed, then
    run the causal intervention and report the effect on both major and OOD.

    Direction-finding strategy (direct optimisation):
        Parameterise ``V`` as a ``(D, n_directions)`` matrix with
        orthonormal columns in the orth complement.  For each optimisation
        step, sample a batch of OOD data, run the **actual** intervened
        forward pass ``h → h − scale · V Vᵀ h``, compute the OOD
        cross-entropy loss, and backpropagate through the intervention to
        ``V``.  Gradient *ascent* on the loss finds the subspace whose
        removal hurts OOD the most.  After each step, ``V`` is projected
        back to the orth complement and re-orthonormalised.

    Returns dict with baseline/intervened/delta losses and the found
    directions.
    """
    _, _, config = nu.load_everything("coin", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    model.requires_grad_(False)
    device = config.device

    sampler_major, _ = get_new_sampler(exp_name, n_minor=0, n_ood=0)
    sampler_ood, _ = get_new_sampler(exp_name, n_minor=0, n_ood=n_ood)
    seq_len = sampler_major.seq_len

    # ---- 1. Fit task subspace ----
    if fit_positions is None:
        fit_positions = list(range(100, seq_len))

    fit_res = train_linear_hidden_predictor_coin(
        exp_name=exp_name,
        layer=layer,
        n_samples=fit_n_samples,
        positions=fit_positions,
        sample_mode="major",
        step=step,
        n_minor=-1,
        print_summary=False,
        skip_baselines=True,
        extraction_point=extraction_point,
    )
    W_fit = fit_res["model_weight"].float()

    if center_task_vecs:
        task_vecs = W_fit - W_fit.mean(dim=0, keepdim=True)
    else:
        task_vecs = W_fit.clone()
    U_tv, S_tv, Vt_tv = torch.linalg.svd(task_vecs, full_matrices=False)
    rank = int((S_tv > 1e-6 * S_tv[0]).sum().item())
    basis = Vt_tv[:rank].T  # (D, rank)
    D_dim = W_fit.shape[1]

    P_task = basis @ basis.T
    P_orth = (torch.eye(D_dim) - P_task).to(device)

    if verbose:
        logger.info(
            f"[opt-dir] Task subspace rank={rank} "
            f"(centered={center_task_vecs}), "
            f"posterior fit R²={fit_res['val_r2']:.4f}"
        )

    # ---- 2. Direct optimisation of V via gradient ascent ----
    if eval_positions is None:
        eval_positions = list(range(seq_len))

    eig_vals, eig_vecs = torch.linalg.eigh(P_orth)
    U_orth = eig_vecs[:, eig_vals > 0.5].to(device)  # (D, D-rank)
    orth_dim = U_orth.shape[1]

    if verbose:
        logger.info(
            f"[opt-dir] Orth complement dim: {orth_dim}, "
            f"optimising {n_directions} direction(s)"
        )

    W_param = torch.randn(orth_dim, n_directions, device=device)
    W_param = W_param / W_param.norm(dim=0, keepdim=True)
    W_param = W_param.detach().requires_grad_(True)

    optimizer_v = torch.optim.Adam([W_param], lr=opt_lr)
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    loss_history = []
    best_loss = -float("inf")
    best_W = W_param.detach().clone()
    steps_no_improve = 0
    smoothed_loss = None
    smooth_alpha = 0.1
    avg_last = 100
    recent_Ws = []

    for opt_step in range(n_opt_steps):
        gen_out = sampler_ood.generate(
            mode="minor", task=None, num_samples=opt_B, epochs=1,
        )
        samples = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
        if samples.dim() == 3:
            samples = samples.squeeze(0)
        samples = samples.to(device)

        def intervention_hook(module, inp, out):
            if torch.is_tensor(out):
                h = out
            else:
                h = out[0]
            V = U_orth @ W_param                       # (D, k)
            V_orth, _ = torch.linalg.qr(V)             # (D, k)
            proj = h @ V_orth @ V_orth.T                # (B, L, D)
            h_modified = h - scale * proj
            if torch.is_tensor(out):
                return h_modified
            return (h_modified,) + out[1:]

        _opt_tgt = (
            model.layers[layer] if extraction_point == "post_mlp"
            else model.layers[layer].attn_block
        )
        handle = _opt_tgt.register_forward_hook(intervention_hook)
        try:
            logits = model(samples)
        finally:
            handle.remove()

        loss_accum = torch.tensor(0.0, device=device)
        count = 0
        for p in eval_positions:
            if p + 1 >= samples.shape[1]:
                continue
            target = samples[:, p + 1].long()
            loss_accum = loss_accum + ce_loss_fn(logits[:, p, :], target)
            count += 1

        if count == 0:
            del samples, logits
            continue

        avg_loss = loss_accum / count
        cur_loss = avg_loss.item()
        loss_history.append(cur_loss)

        if smoothed_loss is None:
            smoothed_loss = cur_loss
        else:
            smoothed_loss = smooth_alpha * cur_loss + (1 - smooth_alpha) * smoothed_loss

        if smoothed_loss > best_loss:
            best_loss = smoothed_loss
            best_W = W_param.detach().clone()
            steps_no_improve = 0
        else:
            steps_no_improve += 1

        optimizer_v.zero_grad()
        (-avg_loss).backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_([W_param], max_norm=grad_clip_norm)
        optimizer_v.step()

        recent_Ws.append(W_param.detach().clone())
        if len(recent_Ws) > avg_last:
            recent_Ws.pop(0)

        del samples, logits, loss_accum
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if verbose and (opt_step + 1) % 50 == 0:
            logger.info(
                f"[opt-dir] step {opt_step + 1}/{n_opt_steps}, "
                f"OOD loss (intervened): {cur_loss:.4f}, "
                f"smoothed: {smoothed_loss:.4f}"
            )

        if patience > 0 and steps_no_improve >= patience:
            if verbose:
                logger.info(
                    f"[opt-dir] Early stopping at step {opt_step + 1} "
                    f"(no improvement for {patience} steps)"
                )
            break

    W_avg = torch.stack(recent_Ws).mean(dim=0)

    with torch.no_grad():
        V_final = U_orth @ W_avg
        V_opt, _ = torch.linalg.qr(V_final)

    if verbose:
        logger.info(
            f"[opt-dir] Optimisation done. Final OOD loss: "
            f"{loss_history[-1]:.4f} (initial: {loss_history[0]:.4f})"
        )

    # ---- 2b. Fit unigram → V_opt projection (diagnostic) ----
    V_opt_dev = V_opt.to(device)
    vocab_size = int(config.vocab_size)
    eval_pos_idx = torch.tensor(eval_positions, device=device, dtype=torch.long)

    def _compute_unigram(samples):
        onehot = torch.nn.functional.one_hot(
            samples.long(), num_classes=vocab_size,
        ).float()
        prefix_counts = onehot.cumsum(dim=1)
        prefix_len = torch.arange(
            1, samples.shape[1] + 1, device=device, dtype=torch.float32,
        ).view(1, -1, 1)
        if unigram_transform == "clr":
            freq = (prefix_counts + unigram_alpha) / (
                prefix_len + unigram_alpha * vocab_size
            )
            logf = torch.log(freq.clamp_min(1e-12))
            return logf - logf.mean(dim=-1, keepdim=True)
        elif unigram_transform == "log1p":
            return torch.log1p(prefix_counts)
        else:
            freq = prefix_counts / prefix_len.clamp_min(1.0)
            return torch.sqrt(freq.clamp_min(0.0))

    def _collect_h_and_uni(sampler, gen_mode, n_samples):
        """Collect raw hidden states and unigram features."""
        all_h, all_uni = [], []
        n_batches = (n_samples + B - 1) // B
        for _ in range(n_batches):
            gen_out = sampler.generate(
                mode=gen_mode, task=None, num_samples=B, epochs=1,
            )
            samp = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
            if samp.dim() == 3:
                samp = samp.squeeze(0)
            samp = samp.to(device)

            cache = {}
            def hook_fn(module, inp, out):
                if torch.is_tensor(out):
                    cache["h"] = out.index_select(1, eval_pos_idx).detach()
                elif isinstance(out, tuple) and torch.is_tensor(out[0]):
                    cache["h"] = out[0].index_select(1, eval_pos_idx).detach()

            _hk_tgt = (
                model.layers[layer] if extraction_point == "post_mlp"
                else model.layers[layer].attn_block
            )
            handle = _hk_tgt.register_forward_hook(hook_fn)
            try:
                with torch.no_grad():
                    model(samp)
                h = cache["h"]  # (B, P, D)
            finally:
                handle.remove()

            uni = _compute_unigram(samp).index_select(1, eval_pos_idx)

            all_h.append(h.cpu())
            all_uni.append(uni.cpu())
            del samp, h, uni
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (
            torch.cat(all_h, 0).reshape(-1, D_dim).float(),
            torch.cat(all_uni, 0).reshape(-1, vocab_size).float(),
        )

    h_maj, uni_maj = _collect_h_and_uni(sampler_major, "major", n_samples_probe)
    h_ood, uni_ood = _collect_h_and_uni(sampler_ood, "minor", n_samples_probe)

    proj_maj = (h_maj @ V_opt.cpu().float())
    proj_ood = (h_ood @ V_opt.cpu().float())

    def _fit_r2(X, Y):
        N = X.shape[0]
        n_train = int(0.8 * N)
        perm = torch.randperm(N)
        X_tr, Y_tr = X[perm[:n_train]], Y[perm[:n_train]]
        X_te, Y_te = X[perm[n_train:]], Y[perm[n_train:]]
        ones_tr = torch.ones(n_train, 1)
        X_aug = torch.cat([X_tr, ones_tr], dim=1)
        W = torch.linalg.pinv(X_aug) @ Y_tr
        ones_te = torch.ones(N - n_train, 1)
        pred_te = torch.cat([X_te, ones_te], dim=1) @ W
        ss_res = ((Y_te - pred_te) ** 2).sum().item()
        ss_tot = ((Y_te - Y_te.mean(0)) ** 2).sum().item()
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    uni_to_proj_r2_major = _fit_r2(uni_maj, proj_maj)
    uni_to_proj_r2_ood = _fit_r2(uni_ood, proj_ood)

    proj_to_uni_r2_major = _fit_r2(proj_maj, uni_maj)
    proj_to_uni_r2_ood = _fit_r2(proj_ood, uni_ood)

    # ---- Baselines ----

    n_rand_trials = 5
    def _rand_orth_r2_trials():
        u2p_maj, u2p_ood, p2u_maj, p2u_ood = [], [], [], []
        U_orth_cpu = U_orth.cpu().float()
        for _ in range(n_rand_trials):
            W_rand = torch.randn(orth_dim, n_directions)
            V_rand = U_orth_cpu @ W_rand
            V_rand, _ = torch.linalg.qr(V_rand)
            rp_maj = h_maj @ V_rand
            rp_ood = h_ood @ V_rand
            u2p_maj.append(_fit_r2(uni_maj, rp_maj))
            u2p_ood.append(_fit_r2(uni_ood, rp_ood))
            p2u_maj.append(_fit_r2(rp_maj, uni_maj))
            p2u_ood.append(_fit_r2(rp_ood, uni_ood))
        return (
            float(np.mean(u2p_maj)), float(np.mean(u2p_ood)),
            float(np.mean(p2u_maj)), float(np.mean(p2u_ood)),
        )

    (rand_orth_u2p_major, rand_orth_u2p_ood,
     rand_orth_p2u_major, rand_orth_p2u_ood) = _rand_orth_r2_trials()

    basis_cpu = basis.cpu().float()  # (D, rank)
    task_proj_maj = h_maj @ basis_cpu  # (N, rank)
    task_proj_ood = h_ood @ basis_cpu
    task_u2p_major = _fit_r2(uni_maj, task_proj_maj)
    task_u2p_ood = _fit_r2(uni_ood, task_proj_ood)
    task_p2u_major = _fit_r2(task_proj_maj, uni_maj)
    task_p2u_ood = _fit_r2(task_proj_ood, uni_ood)

    def _rand_full_r2_trials():
        u2p_maj, u2p_ood, p2u_maj, p2u_ood = [], [], [], []
        for _ in range(n_rand_trials):
            V_rand = torch.randn(D_dim, n_directions)
            V_rand, _ = torch.linalg.qr(V_rand)
            rp_maj = h_maj @ V_rand
            rp_ood = h_ood @ V_rand
            u2p_maj.append(_fit_r2(uni_maj, rp_maj))
            u2p_ood.append(_fit_r2(uni_ood, rp_ood))
            p2u_maj.append(_fit_r2(rp_maj, uni_maj))
            p2u_ood.append(_fit_r2(rp_ood, uni_ood))
        return (
            float(np.mean(u2p_maj)), float(np.mean(u2p_ood)),
            float(np.mean(p2u_maj)), float(np.mean(p2u_ood)),
        )

    (rand_full_u2p_major, rand_full_u2p_ood,
     rand_full_p2u_major, rand_full_p2u_ood) = _rand_full_r2_trials()

    if verbose:
        logger.info(
            f"[opt-dir] Unigram → V_opt R²: "
            f"major={uni_to_proj_r2_major:.4f}, OOD={uni_to_proj_r2_ood:.4f}"
        )
        logger.info(
            f"[opt-dir] V_opt → Unigram R²: "
            f"major={proj_to_uni_r2_major:.4f}, OOD={proj_to_uni_r2_ood:.4f}"
        )
        logger.info(
            f"[opt-dir] Random orth baseline Uni→V: "
            f"major={rand_orth_u2p_major:.4f}, OOD={rand_orth_u2p_ood:.4f}"
        )
        logger.info(
            f"[opt-dir] Task subspace baseline Uni→V: "
            f"major={task_u2p_major:.4f}, OOD={task_u2p_ood:.4f}"
        )
        logger.info(
            f"[opt-dir] Random full-space baseline Uni→V: "
            f"major={rand_full_u2p_major:.4f}, OOD={rand_full_u2p_ood:.4f}"
        )

    # ---- 2c. Nonlinear (MLP) unigram probe ----
    def _fit_r2_mlp(X, Y, hidden_dim=64, n_epochs=200, lr=1e-3):
        """Train a 2-layer MLP and return R² on a held-out split."""
        N = X.shape[0]
        n_train = int(0.8 * N)
        perm = torch.randperm(N)
        X_tr, Y_tr = X[perm[:n_train]], Y[perm[:n_train]]
        X_te, Y_te = X[perm[n_train:]], Y[perm[n_train:]]
        in_dim, out_dim = X.shape[1], Y.shape[1]

        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )
        opt = torch.optim.Adam(mlp.parameters(), lr=lr)
        for _ in range(n_epochs):
            pred = mlp(X_tr)
            loss = ((Y_tr - pred) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred_te = mlp(X_te)
            ss_res = ((Y_te - pred_te) ** 2).sum().item()
            ss_tot = ((Y_te - Y_te.mean(0)) ** 2).sum().item()
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    mlp_u2p_major = _fit_r2_mlp(uni_maj, proj_maj)
    mlp_u2p_ood = _fit_r2_mlp(uni_ood, proj_ood)
    mlp_p2u_major = _fit_r2_mlp(proj_maj, uni_maj)
    mlp_p2u_ood = _fit_r2_mlp(proj_ood, uni_ood)

    def _rand_orth_mlp_r2_trials():
        u2p_maj, u2p_ood, p2u_maj, p2u_ood = [], [], [], []
        U_orth_cpu = U_orth.cpu().float()
        for _ in range(n_rand_trials):
            W_rand = torch.randn(orth_dim, n_directions)
            V_rand = U_orth_cpu @ W_rand
            V_rand, _ = torch.linalg.qr(V_rand)
            rp_maj = h_maj @ V_rand
            rp_ood = h_ood @ V_rand
            u2p_maj.append(_fit_r2_mlp(uni_maj, rp_maj))
            u2p_ood.append(_fit_r2_mlp(uni_ood, rp_ood))
            p2u_maj.append(_fit_r2_mlp(rp_maj, uni_maj))
            p2u_ood.append(_fit_r2_mlp(rp_ood, uni_ood))
        return (
            float(np.mean(u2p_maj)), float(np.mean(u2p_ood)),
            float(np.mean(p2u_maj)), float(np.mean(p2u_ood)),
        )

    (mlp_rorth_u2p_major, mlp_rorth_u2p_ood,
     mlp_rorth_p2u_major, mlp_rorth_p2u_ood) = _rand_orth_mlp_r2_trials()

    # ---- 2d. Enriched feature probes ----
    def _compute_enriched_features(uni_feat):
        """Build enriched feature matrix from unigram features.

        Returns dict of {name: (N, F) tensor}.
        Assumes uni_feat came from CLR transform.
        """
        feats = {}

        feats["log_count"] = uni_feat

        freq = torch.softmax(uni_feat, dim=-1)
        feats["dirichlet_pred"] = freq

        return feats

    enriched_maj = _compute_enriched_features(uni_maj)
    enriched_ood = _compute_enriched_features(uni_ood)

    enriched_r2 = {}
    for feat_name in enriched_maj:
        f_maj = enriched_maj[feat_name]
        f_ood = enriched_ood[feat_name]
        enriched_r2[feat_name] = {
            "u2p_major": _fit_r2(f_maj, proj_maj),
            "u2p_ood": _fit_r2(f_ood, proj_ood),
            "p2u_major": _fit_r2(proj_maj, f_maj),
            "p2u_ood": _fit_r2(proj_ood, f_ood),
        }

    # ---- 2e. Unigram-explained decomposition ----
    V_opt_cpu = V_opt.cpu().float()  # (D, n_directions)
    proj_c = proj_ood - proj_ood.mean(0, keepdim=True)
    total_var = (proj_c ** 2).sum().item()
    dir_pred = torch.softmax(uni_ood, dim=-1)
    uni_explained = {}
    for fname, feat in [("unigram_clr", uni_ood), ("dirichlet_pred", dir_pred)]:
        Nf = feat.shape[0]
        X_aug = torch.cat([feat, torch.ones(Nf, 1)], 1)
        W_fit_exp = torch.linalg.pinv(X_aug) @ proj_ood
        Y_hat = X_aug @ W_fit_exp
        r2 = 1.0 - ((proj_ood - Y_hat) ** 2).sum().item() / total_var \
            if total_var > 0 else 0.0
        _, Sp, Vtp = torch.linalg.svd(
            Y_hat - Y_hat.mean(0, keepdim=True), full_matrices=False,
        )
        frac = Sp ** 2 / max(total_var, 1e-12)
        rk = int((frac > 0.005).sum().item())
        V_expl = V_opt_cpu @ Vtp[:rk].T if rk > 0 else torch.empty(D_dim, 0)
        uni_explained[fname] = {
            "explained_rank": rk,
            "r2_feat_to_proj": r2,
            "V_explained": V_expl,
        }

    del proj_maj, proj_ood, h_maj, h_ood, uni_maj, uni_ood
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ---- 3. Run intervention with rank-k projector V V^T ----
    P_v = (V_opt @ V_opt.T).to(device)  # (D, D)

    ce_loss_eval = torch.nn.CrossEntropyLoss(reduction="none")

    def _run_experiment(sampler, gen_mode, n_samples):
        baseline_losses_by_pos = {p: [] for p in eval_positions}
        intervened_losses_by_pos = {p: [] for p in eval_positions}
        baseline_loss_at_0 = []

        n_batches = (n_samples + B - 1) // B
        for _ in range(n_batches):
            gen_out = sampler.generate(
                mode=gen_mode, task=None, num_samples=B, epochs=1,
            )
            samples = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
            if samples.dim() == 3:
                samples = samples.squeeze(0)
            samples = samples.to(device)

            with torch.no_grad():
                logits_base = model(samples)

            def intervention_hook(module, inp, out):
                if torch.is_tensor(out):
                    h = out
                else:
                    h = out[0]
                h_proj = h @ P_v
                h_modified = h - scale * h_proj
                if torch.is_tensor(out):
                    return h_modified
                return (h_modified,) + out[1:]

            _run_tgt = (
                model.layers[layer] if extraction_point == "post_mlp"
                else model.layers[layer].attn_block
            )
            handle = _run_tgt.register_forward_hook(intervention_hook)
            try:
                with torch.no_grad():
                    logits_int = model(samples)
            finally:
                handle.remove()

            if samples.shape[1] > 1:
                baseline_loss_at_0.append(
                    ce_loss_eval(logits_base[:, 0, :], samples[:, 1].long()).mean().item()
                )

            for p in eval_positions:
                if p + 1 >= samples.shape[1]:
                    continue
                target = samples[:, p + 1].long()
                loss_base = ce_loss_eval(
                    logits_base[:, p, :], target,
                ).mean().item()
                loss_int = ce_loss_eval(
                    logits_int[:, p, :], target,
                ).mean().item()
                baseline_losses_by_pos[p].append(loss_base)
                intervened_losses_by_pos[p].append(loss_int)

            del samples, logits_base, logits_int
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        baseline_per_pos, intervened_per_pos, valid_positions = [], [], []
        for p in eval_positions:
            if baseline_losses_by_pos[p]:
                baseline_per_pos.append(np.mean(baseline_losses_by_pos[p]))
                intervened_per_pos.append(np.mean(intervened_losses_by_pos[p]))
                valid_positions.append(p)

        baseline_avg = float(np.mean(baseline_per_pos)) if baseline_per_pos else float("nan")
        intervened_avg = float(np.mean(intervened_per_pos)) if intervened_per_pos else float("nan")

        return {
            "baseline": baseline_avg,
            "intervened": intervened_avg,
            "delta": intervened_avg - baseline_avg,
            "baseline_per_pos": baseline_per_pos,
            "intervened_per_pos": intervened_per_pos,
            "positions": valid_positions,
            "baseline_loss_at_0": float(np.mean(baseline_loss_at_0)) if baseline_loss_at_0 else float("nan"),
        }

    if verbose:
        logger.info("[opt-dir] Running major experiment ...")
    res_major = _run_experiment(sampler_major, "major", n_samples_eval)

    if verbose:
        logger.info("[opt-dir] Running OOD experiment ...")
    res_ood = _run_experiment(sampler_ood, "minor", n_samples_eval)

    # ---- 3b. Random orth intervention baselines ----
    n_rand_int_trials = 3
    U_orth_cpu = U_orth.cpu().float()

    def _run_rand_int_experiment(P_proj, sampler, gen_mode, n_samples):
        bl_by_pos = {p: [] for p in eval_positions}
        it_by_pos = {p: [] for p in eval_positions}
        n_batches = (n_samples + B - 1) // B
        for _ in range(n_batches):
            gen_out = sampler.generate(
                mode=gen_mode, task=None, num_samples=B, epochs=1,
            )
            samples = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
            if samples.dim() == 3:
                samples = samples.squeeze(0)
            samples = samples.to(device)
            with torch.no_grad():
                logits_base = model(samples)

            def rand_hook(module, inp, out, _P=P_proj):
                if torch.is_tensor(out):
                    h = out
                else:
                    h = out[0]
                h_modified = h - scale * (h @ _P)
                if torch.is_tensor(out):
                    return h_modified
                return (h_modified,) + out[1:]

            _rand_tgt = (
                model.layers[layer] if extraction_point == "post_mlp"
                else model.layers[layer].attn_block
            )
            handle = _rand_tgt.register_forward_hook(rand_hook)
            try:
                with torch.no_grad():
                    logits_int = model(samples)
            finally:
                handle.remove()
            for p in eval_positions:
                if p + 1 >= samples.shape[1]:
                    continue
                target = samples[:, p + 1].long()
                bl_by_pos[p].append(
                    ce_loss_eval(logits_base[:, p, :], target).mean().item()
                )
                it_by_pos[p].append(
                    ce_loss_eval(logits_int[:, p, :], target).mean().item()
                )
            del samples, logits_base, logits_int
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        bl_vals = [np.mean(bl_by_pos[p]) for p in eval_positions if bl_by_pos[p]]
        it_vals = [np.mean(it_by_pos[p]) for p in eval_positions if it_by_pos[p]]
        bl_avg = float(np.mean(bl_vals)) if bl_vals else float("nan")
        it_avg = float(np.mean(it_vals)) if it_vals else float("nan")
        return bl_avg, it_avg

    def _rand_orth_intervention_trials(rand_rank):
        """Run random orth intervention with given rank, average over trials."""
        d_maj, d_ood, p_maj, p_ood = [], [], [], []
        actual_rank = min(rand_rank, orth_dim)
        for _ in range(n_rand_int_trials):
            W_r = torch.randn(orth_dim, actual_rank)
            V_r = U_orth_cpu @ W_r
            V_r, _ = torch.linalg.qr(V_r)
            P_r = (V_r @ V_r.T).to(device)
            bl_m, it_m = _run_rand_int_experiment(P_r, sampler_major, "major", n_samples_eval)
            bl_o, it_o = _run_rand_int_experiment(P_r, sampler_ood, "minor", n_samples_eval)
            d_maj.append(it_m - bl_m)
            d_ood.append(it_o - bl_o)
            p_maj.append(100.0 * (it_m - bl_m) / bl_m if bl_m > 0 else float("nan"))
            p_ood.append(100.0 * (it_o - bl_o) / bl_o if bl_o > 0 else float("nan"))
        return (
            float(np.mean(d_maj)), float(np.mean(d_ood)),
            float(np.mean(p_maj)), float(np.mean(p_ood)),
        )

    (rand_int_delta_major, rand_int_delta_ood,
     rand_int_pct_major, rand_int_pct_ood) = _rand_orth_intervention_trials(n_directions)

    rand3x_rank = 3 * n_directions
    (rand3x_int_delta_major, rand3x_int_delta_ood,
     rand3x_int_pct_major, rand3x_int_pct_ood) = _rand_orth_intervention_trials(rand3x_rank)

    # ---- Causal test for unigram-explained directions ----
    for info in uni_explained.values():
        if info["explained_rank"] == 0:
            info["delta_ood"] = 0.0
            continue
        Ve = info["V_explained"].to(device)
        P_e = Ve @ Ve.T
        bl_o, it_o = _run_rand_int_experiment(P_e, sampler_ood, "minor", n_samples_eval)
        info["delta_ood"] = it_o - bl_o
        del Ve

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    pct_major = (
        100.0 * res_major["delta"] / res_major["baseline"]
        if res_major["baseline"] > 0 else float("nan")
    )
    pct_ood = (
        100.0 * res_ood["delta"] / res_ood["baseline"]
        if res_ood["baseline"] > 0 else float("nan")
    )

    results = {
        "baseline_loss_major": res_major["baseline"],
        "intervened_loss_major": res_major["intervened"],
        "delta_loss_major": res_major["delta"],
        "pct_increase_major": pct_major,
        "baseline_loss_ood": res_ood["baseline"],
        "intervened_loss_ood": res_ood["intervened"],
        "delta_loss_ood": res_ood["delta"],
        "pct_increase_ood": pct_ood,
        "baseline_per_pos_major": res_major["baseline_per_pos"],
        "intervened_per_pos_major": res_major["intervened_per_pos"],
        "baseline_per_pos_ood": res_ood["baseline_per_pos"],
        "intervened_per_pos_ood": res_ood["intervened_per_pos"],
        "eval_positions": res_major["positions"],
        "layer": layer,
        "scale": scale,
        "task_subspace_rank": rank,
        "n_directions": n_directions,
        "directions": V_opt.cpu(),
        "loss_history": loss_history,
        "uni_to_proj_r2_major": uni_to_proj_r2_major,
        "uni_to_proj_r2_ood": uni_to_proj_r2_ood,
        "proj_to_uni_r2_major": proj_to_uni_r2_major,
        "proj_to_uni_r2_ood": proj_to_uni_r2_ood,
        "rand_orth_u2p_major": rand_orth_u2p_major,
        "rand_orth_u2p_ood": rand_orth_u2p_ood,
        "rand_orth_p2u_major": rand_orth_p2u_major,
        "rand_orth_p2u_ood": rand_orth_p2u_ood,
        "task_u2p_major": task_u2p_major,
        "task_u2p_ood": task_u2p_ood,
        "task_p2u_major": task_p2u_major,
        "task_p2u_ood": task_p2u_ood,
        "rand_full_u2p_major": rand_full_u2p_major,
        "rand_full_u2p_ood": rand_full_u2p_ood,
        "rand_full_p2u_major": rand_full_p2u_major,
        "rand_full_p2u_ood": rand_full_p2u_ood,
        "rand_int_delta_major": rand_int_delta_major,
        "rand_int_delta_ood": rand_int_delta_ood,
        "rand_int_pct_major": rand_int_pct_major,
        "rand_int_pct_ood": rand_int_pct_ood,
        "rand3x_int_delta_major": rand3x_int_delta_major,
        "rand3x_int_delta_ood": rand3x_int_delta_ood,
        "rand3x_int_pct_major": rand3x_int_pct_major,
        "rand3x_int_pct_ood": rand3x_int_pct_ood,
        "rand3x_rank": min(rand3x_rank, orth_dim),
        "mlp_u2p_major": mlp_u2p_major,
        "mlp_u2p_ood": mlp_u2p_ood,
        "mlp_p2u_major": mlp_p2u_major,
        "mlp_p2u_ood": mlp_p2u_ood,
        "mlp_rorth_u2p_major": mlp_rorth_u2p_major,
        "mlp_rorth_u2p_ood": mlp_rorth_u2p_ood,
        "mlp_rorth_p2u_major": mlp_rorth_p2u_major,
        "mlp_rorth_p2u_ood": mlp_rorth_p2u_ood,
        "enriched_r2": enriched_r2,
        "baseline_loss_at_0_major": res_major["baseline_loss_at_0"],
        "baseline_loss_at_0_ood": res_ood["baseline_loss_at_0"],
        "uni_explained": {
            fn: {k: v for k, v in info.items() if k != "V_explained"}
            for fn, info in uni_explained.items()
        },
    }

    if print_summary:
        print(f"\n{'=' * 65}")
        print(
            f"Causal Intervention: Optimal Rank-{n_directions} Orth Subspace  "
            f"(layer {layer}, scale={scale})"
        )
        print(f"{'=' * 65}")
        print(
            f"  Task subspace rank: {rank}  |  "
            f"Opt steps: {len(loss_history)}/{n_opt_steps}  |  "
            f"Loss: {loss_history[0]:.4f} → {loss_history[-1]:.4f}"
        )
        print(f"  {'R² metric':<28} {'Major':>8} {'OOD':>8}")
        print(f"  {'-'*44}")
        print(f"  {'Uni → V_opt':<28} {uni_to_proj_r2_major:>8.4f} {uni_to_proj_r2_ood:>8.4f}")
        print(f"  {'  Random orth':<28} {rand_orth_u2p_major:>8.4f} {rand_orth_u2p_ood:>8.4f}")
        print(f"  {'  Task subspace':<28} {task_u2p_major:>8.4f} {task_u2p_ood:>8.4f}")
        print(f"  {'  Random full-space':<28} {rand_full_u2p_major:>8.4f} {rand_full_u2p_ood:>8.4f}")
        print(f"  {'V_opt → Uni':<28} {proj_to_uni_r2_major:>8.4f} {proj_to_uni_r2_ood:>8.4f}")
        print(f"  {'  Random orth':<28} {rand_orth_p2u_major:>8.4f} {rand_orth_p2u_ood:>8.4f}")
        print(f"  {'  Task subspace':<28} {task_p2u_major:>8.4f} {task_p2u_ood:>8.4f}")
        print(f"  {'  Random full-space':<28} {rand_full_p2u_major:>8.4f} {rand_full_p2u_ood:>8.4f}")
        print(f"  {'Uni → V_opt (MLP)':<28} {mlp_u2p_major:>8.4f} {mlp_u2p_ood:>8.4f}")
        print(f"  {'  Random orth (MLP)':<28} {mlp_rorth_u2p_major:>8.4f} {mlp_rorth_u2p_ood:>8.4f}")
        print(f"  {'V_opt → Uni (MLP)':<28} {mlp_p2u_major:>8.4f} {mlp_p2u_ood:>8.4f}")
        print(f"  {'  Random orth (MLP)':<28} {mlp_rorth_p2u_major:>8.4f} {mlp_rorth_p2u_ood:>8.4f}")
        print()
        print(f"  {'Enriched probes → V_opt':<28} {'Major':>8} {'OOD':>8}")
        print(f"  {'-'*44}")
        for fn, rv in enriched_r2.items():
            print(f"  {fn:<28} {rv['u2p_major']:>8.4f} {rv['u2p_ood']:>8.4f}")
        print()
        print(f"  {'Unigram-explained V_opt':<28} {'R²':>8} {'rank':>6} {'Δ OOD':>8}")
        print(f"  {'-'*50}")
        for fn, info in uni_explained.items():
            print(f"  {fn:<28} {info['r2_feat_to_proj']:>8.4f} "
                  f"{info['explained_rank']:>6d} {info['delta_ood']:>8.4f}")
        print()
        print(f"  Eval positions: {len(res_major['positions'])} positions\n")
        print(f"{'Metric':<30} {'Major':>12} {'OOD':>12}")
        print("-" * 54)
        print(
            f"{'Baseline loss':<30} "
            f"{res_major['baseline']:>12.4f} "
            f"{res_ood['baseline']:>12.4f}"
        )
        print(
            f"{'Intervened loss':<30} "
            f"{res_major['intervened']:>12.4f} "
            f"{res_ood['intervened']:>12.4f}"
        )
        print(
            f"{'Delta loss':<30} "
            f"{res_major['delta']:>12.4f} "
            f"{res_ood['delta']:>12.4f}"
        )
        print(
            f"{'Percent increase':<30} "
            f"{pct_major:>11.1f}% "
            f"{pct_ood:>11.1f}%"
        )
        print(
            f"{'Rand orth Δ loss':<30} "
            f"{rand_int_delta_major:>12.4f} "
            f"{rand_int_delta_ood:>12.4f}"
        )
        print(
            f"{'Rand orth % increase':<30} "
            f"{rand_int_pct_major:>11.1f}% "
            f"{rand_int_pct_ood:>11.1f}%"
        )
        print(
            f"{'Rand orth 3x Δ loss':<30} "
            f"{rand3x_int_delta_major:>12.4f} "
            f"{rand3x_int_delta_ood:>12.4f}"
        )
        print(
            f"{'Rand orth 3x % increase':<30} "
            f"{rand3x_int_pct_major:>11.1f}% "
            f"{rand3x_int_pct_ood:>11.1f}%"
        )

    return results


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

    def _gen_and_cache(sampler, mode, n):
        seqs, logits_list = [], []
        for _ in range(max(1, (n + B - 1) // B)):
            g = sampler.generate(mode=mode, task=None, num_samples=B, epochs=1)
            s = (g[0] if isinstance(g, (tuple, list)) else g)
            if s.dim() == 3:
                s = s.squeeze(0)
            s = s.to(device)
            with torch.no_grad():
                logits_list.append(model(s).cpu())
            seqs.append(s.cpu())
        return seqs, logits_list

    cache_maj = _gen_and_cache(sampler_major, "major", n_samples_eval)
    cache_ood = _gen_and_cache(sampler_ood, "minor", n_samples_eval)
    cache_minor = _gen_and_cache(sampler_minor, "minor", n_samples_eval)
    cache_tgt = cache_minor if opt_target == "minor" else cache_ood

    # =================================================================
    #  Phase 3: Collect hidden states + unigram (all layers, one pass)
    # =================================================================
    logger.info("[opt-dir] Phase 3: Collecting hiddens + unigram ...")
    eval_pos_idx = torch.tensor(eval_positions, device=device, dtype=torch.long)

    def _compute_unigram(samples):
        onehot = torch.nn.functional.one_hot(
            samples.long(), num_classes=vocab_size,
        ).float()
        prefix_counts = onehot.cumsum(dim=1)
        prefix_len = torch.arange(
            1, samples.shape[1] + 1, device=samples.device, dtype=torch.float32,
        ).view(1, -1, 1)
        if unigram_transform == "clr":
            freq = (prefix_counts + unigram_alpha) / (
                prefix_len + unigram_alpha * vocab_size
            )
            logf = torch.log(freq.clamp_min(1e-12))
            return logf - logf.mean(dim=-1, keepdim=True)
        elif unigram_transform == "log1p":
            return torch.log1p(prefix_counts)
        freq = prefix_counts / prefix_len.clamp_min(1.0)
        return torch.sqrt(freq.clamp_min(0.0))

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
            uni = _compute_unigram(s)  # (Bs, L, vocab_size)  CLR transform
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

            def _hook(mod, inp, out, _P=P):
                h = out if torch.is_tensor(out) else out[0]
                hm = h - scale * (h @ _P)
                return hm if torch.is_tensor(out) else (hm,) + out[1:]

            _eh_tgt = (
                model.layers[layer_idx] if extraction_point == "post_mlp"
                else model.layers[layer_idx].attn_block
            )
            handle = _eh_tgt.register_forward_hook(_hook)
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

    def _ols_r2(X, Y):
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
        N = X.shape[0]
        nt = int(0.8 * N)
        pm = torch.randperm(N)
        Xtr, Ytr = X[pm[:nt]], Y[pm[:nt]]
        Xte, Yte = X[pm[nt:]], Y[pm[nt:]]
        net = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], hid), torch.nn.SiLU(),
            torch.nn.Linear(hid, Y.shape[1]),
        )
        opt_mlp = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(ep):
            loss = ((net(Xtr) - Ytr) ** 2).mean()
            opt_mlp.zero_grad()
            loss.backward()
            opt_mlp.step()
        with torch.no_grad():
            ss_r = ((Yte - net(Xte)) ** 2).sum().item()
            ss_t = ((Yte - Yte.mean(0)) ** 2).sum().item()
        return 1.0 - ss_r / ss_t if ss_t > 0 else float("nan")

    val_seq = cache_tgt[0][0].to(device)
    ce_mean = torch.nn.CrossEntropyLoss()

    def _val_loss(layer_idx, V_mat):
        """CE loss on cached validation batch with intervention h' = h - s*V*V^T*h."""
        P = V_mat @ V_mat.T

        def _hook(mod, inp, out, _P=P):
            h = out if torch.is_tensor(out) else out[0]
            return (h - scale * (h @ _P)) if torch.is_tensor(out) \
                else (h - scale * (h @ _P),) + out[1:]

        _vl_tgt = (
            model.layers[layer_idx] if extraction_point == "post_mlp"
            else model.layers[layer_idx].attn_block
        )
        hnd = _vl_tgt.register_forward_hook(_hook)
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
            name: _ols_r2(enrich_tgt[:, sl], proj_tgt)
            for name, sl in _coin_feat_groups.items()
        }
        # Combined (all enriched features together) R²
        r2_combined = _ols_r2(enrich_tgt, proj_tgt)

        # Legacy scalar R² values kept for backward compatibility
        r2_u2v = feat_r2_marginal["unigram_clr"]
        r2_v2u = _ols_r2(proj_tgt, uni_tgt)
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

        def _rand_stats_coin(vals):
            arr = np.array(vals, dtype=float)
            return {
                "mean": float(arr.mean()),
                "q25": float(np.percentile(arr, 25)),
                "q75": float(np.percentile(arr, 75)),
            }

        rand_stats = {k: _rand_stats_coin(v) for k, v in rand_acc.items()}
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

    def _iqr_err_norm_coin(key_batch, g_m):
        """IQR error bars in normalized (%) units."""
        lo_arr, hi_arr = [], []
        for l in layers:
            vals = all_results[l][key_batch]
            if len(vals) < 2:
                lo_arr.append(0.0); hi_arr.append(0.0); continue
            arr = np.array(vals, dtype=float) / g_m * 100
            mn = arr.mean()
            lo_arr.append(abs(mn - np.percentile(arr, 25)))
            hi_arr.append(abs(np.percentile(arr, 75) - mn))
        return np.array(lo_arr), np.array(hi_arr)

    COLORS_C = {"maj": "#2166ac", "ood": "#d6604d", "minor": "#1a9850"}
    bw_bar_c = 0.22
    g_step_c = 0.24
    MC_C     = {"maj": -g_step_c, "ood": 0.0, "minor": +g_step_c}

    fig_delta, ax = plt.subplots(figsize=figsize, dpi=150)

    # Random baseline band: OOD rand normalized by g_ood (primary target metric)
    rand_ood_q75_norm = max(
        all_results[l]["rand_stats"]["ood"]["q75"] / g_ood * 100 for l in layers
    )
    ax.axhspan(0, rand_ood_q75_norm, color="#b0b8c8", alpha=0.35, zorder=1,
               hatch="///", label="Random")
    ax.axhline(rand_ood_q75_norm, color="#556070", lw=1.2, ls="-", zorder=2, alpha=0.85)

    for mode, norm_vals, (key_pb, g_m), label in [
        ("maj",   norm_maj,   ("delta_per_batch_major", g_maj),   "Maj."),
        ("ood",   norm_ood,   ("delta_per_batch_ood",   g_ood),   "OOD"),
        ("minor", norm_minor, ("delta_per_batch_minor", g_minor), "Min."),
    ]:
        c  = COLORS_C[mode]
        xm = x + MC_C[mode]
        lo, hi = _iqr_err_norm_coin(key_pb, g_m)
        ax.bar(xm, norm_vals, bw_bar_c, color=c, linewidth=0, zorder=3, label=label)
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
    C_OOD_e = COLORS_C["ood"]
    C_TGT_e = COLORS_C["minor"] if opt_target == "minor" else COLORS_C["ood"]
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
