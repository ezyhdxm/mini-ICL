"""Remove bigram directions from orth(task + token) subspace (latent Markov, non-padded)."""

import gc
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

import icl.utils.notebook_utils as nu
from icl.latent_markov.analysis.ood import get_latent_sampler
from icl.utils.logger import setup_logger
from icl.latent_markov.analysis.probes import train_linear_hidden_predictor

logger = setup_logger(__name__)


def bigram_prefix_counts(x: torch.Tensor, V: int, *, dtype=torch.int64) -> torch.Tensor:
    """
    Compute cumulative bigram counts at each position in sequences.
    
    For each position t, computes the count of all bigrams (prev, next) that occurred
    up to and including position t in the sequence. This creates a cumulative count
    matrix for each sequence.
    
    Args:
        x: Input sequences of shape (B, T) where B is batch size and T is sequence length.
           Values should be in [0, V-1].
        V: Vocabulary size (number of states)
        dtype: Output dtype (default: torch.int64)
    
    Returns:
        counts: Tensor of shape (B, T, V, V) where counts[b, t, u, v] is the count
                of bigram (u -> v) in sequence b up to position t.
                Position 0 is always zeros (no bigrams yet).
    
    Example:
        >>> x = torch.tensor([[0, 1, 2, 1]])  # Single sequence: 0->1->2->1
        >>> counts = bigram_prefix_counts(x, V=3)
        >>> counts[0, 1, 0, 1]  # At position 1, count of (0->1) is 1
        tensor(1)
        >>> counts[0, 2, 1, 2]  # At position 2, count of (1->2) is 1
        tensor(1)
    """
    assert x.dim() == 2, "x must be (B, T)"
    B, T = x.shape
    device = x.device

    if T == 1:
        return torch.zeros(B, T, V, V, dtype=dtype, device=device)

    prev = x[:, :-1]               # (B, T-1)
    nxt  = x[:,  1:]               # (B, T-1)

    prev_oh = F.one_hot(prev, num_classes=V).to(torch.float32)  # (B, T-1, V)
    nxt_oh  = F.one_hot(nxt,  num_classes=V).to(torch.float32)  # (B, T-1, V)

    edges = torch.einsum('btu,btv->btuv', prev_oh, nxt_oh)      # (B, T-1, V, V)

    zero_pad = torch.zeros(B, 1, V, V, dtype=edges.dtype, device=device)
    edges_padded = torch.cat([zero_pad, edges], dim=1)          # (B, T, V, V)

    out = torch.cumsum(edges_padded, dim=1)                     
    return out.to(dtype)


# ---------------------------------------------------------------------------
#  intervene_remove_bigram_subspace
# ---------------------------------------------------------------------------

@torch.no_grad()
def intervene_remove_bigram_subspace(
    exp_name: str,
    layer: int,
    n_ood: int = 30,
    B: int = 64,
    n_samples_eval: int = 500,
    n_samples_fit: int = 2000,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    center_task_vecs: bool = False,
    eval_positions: Optional[list] = None,
    token_protection_mode: str = "residual_pca",
    token_protection_rank: Optional[int] = 3,
    bigram_transform: str = "clr",
    bigram_alpha: float = 0.5,
    feature_type: str = "bigram_clr",
    method: str = "direct",
    ablation_max_k: int = 10,
    scale: float = 1.0,
    extraction_point: str = "post_attn",
    probe_method: str = "ols",
    show: bool = True,
    figsize: tuple = (10, 4.5),
    print_summary: bool = True,
) -> dict:
    """
    Find directions in orth(task + token) that linearly predict bigram
    features, then causally remove them.

    Procedure
    ---------
    1. Fit the task subspace (linear posterior probe).
    2. Build a protected subspace = task + token directions (same as
       ``intervene_optimal_orth_direction``).
    3. Project OOD hidden states into the orthogonal complement of the
       protected subspace.
    4. Fit a linear map  h_orth -> features  on OOD data, yielding
       weight matrix ``W`` of shape ``(orth_dim, V)``.
    5. The column space of ``W`` (rank ≤ V) is the subspace through
       which the orthogonal complement linearly encodes the features.
    6. Remove that subspace from hidden states and evaluate.

    The resulting directions are guaranteed orthogonal to both task and
    token subspaces, so removing them cannot trivially destroy task
    identity or token information.

    Parameters
    ----------
    n_samples_fit : int
        Number of OOD samples used to fit the regression.
    bigram_transform : str
        ``"clr"`` (centred log-ratio, default), ``"log1p"``, or ``"freq"``
        (square-root frequency).  Applied to raw bigram counts before
        the ``feature_type`` transform.
    bigram_alpha : float
        Dirichlet smoothing parameter for bigram counts.
    feature_type : str
        ``"bigram_clr"`` — CLR-transformed current-token bigram row
        (default).
        ``"dirichlet_pred"`` — Dirichlet posterior predictive
        ``softmax(bigram_clr)``, i.e. the smoothed transition
        probability estimate ``P(next | current, counts)``.
    method : str
        ``"direct"`` (default) — remove the entire column-space of the
        fitted weight matrix at once.  The rank equals
        ``min(orth_dim, V)`` (typically V).
        ``"ablation_sweep"`` — SVD the weight matrix and remove
        components one at a time (individual + cumulative).
    ablation_max_k : int
        Number of SVD components to sweep (only for ``method="ablation_sweep"``).
    """
    import matplotlib.pyplot as plt

    # ---- 1. Load config & model ------------------------------------------------
    _, _, config = nu.load_everything("latent", exp_name)
    if step is None:
        step = config.training.num_epochs
    D = int(config.model.emb_dim)

    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    model.requires_grad_(False)
    device = config.device

    sampler_major, _, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=0)
    sampler_ood, _, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=n_ood)

    seq_len = sampler_major.seq_len
    vocab_size = int(sampler_major.num_states)

    if fit_positions is None:
        # Averaging requires p < seq_len-1 (get_token_conditioned_hiddens constraint)
        upper = seq_len - 1 if probe_method == "averaging" else seq_len
        fit_positions = list(range(max(0, seq_len // 2), upper))
    if eval_positions is None:
        eval_positions = list(range(seq_len))
    if 0 not in eval_positions:
        eval_positions = [0] + list(eval_positions)

    # ---- Extraction-point helper -----------------------------------------------
    def _layer_module():
        return (model.layers[layer]
                if extraction_point == "post_mlp"
                else model.layers[layer].attn_block)

    # ---- 2+3. Task & token subspaces -------------------------------------------
    if probe_method == "averaging":
        logger.info("[bigram-removal] Fitting subspaces via ANOVA averaging ...")
        from icl.latent_markov.analysis.variance import get_token_conditioned_hiddens

        all_hiddens_anova, anova_info = get_token_conditioned_hiddens(
            exp_name, layers=[layer], batch_size=B,
            positions_of_interest=fit_positions,
            step=step, n_minor=0,
            extraction_point=extraction_point,
        )
        # all_hiddens_anova: (1, n_positions, V_max, n_major, B, D)
        pos_to_anova = {p: i for i, p in enumerate(anova_info["positions"])}
        n_uniq = anova_info["n_unique_tokens"]
        V_max = all_hiddens_anova.shape[2]
        n_major_anova = all_hiddens_anova.shape[3]

        parts = []
        for p in fit_positions:
            if p not in pos_to_anova:
                continue
            pi = pos_to_anova[p]
            V_p = n_uniq.get(p, V_max)
            # cell_means: (V_p, K, D) — average over the batch dimension
            parts.append(
                all_hiddens_anova[0, pi, :V_p, :n_major_anova].float().mean(dim=-2)
            )
        if not parts:
            raise ValueError(
                "[bigram-removal] probe_method='averaging': no fit_positions "
                "found in token-conditioned hiddens."
            )
        del all_hiddens_anova

        min_V = min(p.shape[0] for p in parts)
        parts = [p[:min_V] for p in parts]
        demeaned = [cm - cm.mean(dim=(0, 1), keepdim=True) for cm in parts]
        pooled = torch.stack(demeaned, dim=0).mean(dim=0)  # (V, K, D)
        grand = pooled.mean(dim=(0, 1))
        task_vecs  = pooled.mean(dim=0) - grand   # (K, D)
        token_vecs = pooled.mean(dim=1) - grand   # (V, D)

        if center_task_vecs:
            task_vecs = task_vecs - task_vecs.mean(dim=0, keepdim=True)
        _, S_tv, Vt_tv = torch.linalg.svd(task_vecs, full_matrices=False)
        rank = int((S_tv > 1e-6 * S_tv[0]).sum().item())
        basis = Vt_tv[:rank].T  # (D, rank)
        W_tok_dir = token_vecs.T  # (D, V)

    else:
        # OLS joint probe
        logger.info("[bigram-removal] Fitting task subspace via OLS ...")
        fit_res = train_linear_hidden_predictor(
            exp_name=exp_name,
            layer=layer,
            n_samples=fit_n_samples,
            positions=fit_positions,
            sample_mode="major",
            n_minor=-1,
            step=step,
            print_summary=False,
            skip_baselines=True,
        )
        W_fit = fit_res["model_weight"].float()  # (K_major, D)
        if center_task_vecs:
            task_vecs = W_fit - W_fit.mean(dim=0, keepdim=True)
        else:
            task_vecs = W_fit.clone()
        _, S_tv, Vt_tv = torch.linalg.svd(task_vecs, full_matrices=False)
        rank = int((S_tv > 1e-6 * S_tv[0]).sum().item())
        basis = Vt_tv[:rank].T  # (D, rank)

        # Token directions via hidden-state → one-hot regression
        tok_pos_idx = torch.tensor(fit_positions, device=device, dtype=torch.long)
        all_h_tok, all_oh_tok = [], []
        n_tok_batches = (min(fit_n_samples, 2000) + B - 1) // B
        for _ in range(n_tok_batches):
            gen_out = sampler_major.generate(
                mode="major", task=None, num_samples=B, epochs=1,
            )
            samp = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
            if samp.dim() == 3:
                samp = samp.squeeze(0)
            samp = samp.to(device)
            cache = {}
            def _tok_hook(module, inp, out, _cache=cache):
                h = out if torch.is_tensor(out) else out[0]
                _cache["h"] = h.index_select(1, tok_pos_idx).detach()
            handle = _layer_module().register_forward_hook(_tok_hook)
            try:
                with torch.no_grad():
                    model(samp)
            finally:
                handle.remove()
            tokens_at_pos = samp[:, tok_pos_idx].long()
            oh = torch.nn.functional.one_hot(
                tokens_at_pos, num_classes=vocab_size,
            ).float()
            all_h_tok.append(cache["h"].cpu())
            all_oh_tok.append(oh.cpu())
            del samp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        H_tok = torch.cat(all_h_tok, 0).reshape(-1, D).float()
        OH_tok = torch.cat(all_oh_tok, 0).reshape(-1, vocab_size).float()
        del all_h_tok, all_oh_tok
        OH_aug = torch.cat([OH_tok, torch.ones(OH_tok.shape[0], 1)], dim=1)
        W_tok_aug = torch.linalg.pinv(OH_aug) @ H_tok
        W_tok_dir = W_tok_aug[:-1].T  # (D, V)
        del H_tok, OH_tok, OH_aug, W_tok_aug

    # ---- Token protection (shared by both probe methods) -----------------------
    token_basis = torch.empty(D, 0, dtype=torch.float32)
    token_basis_rank = 0

    if token_protection_mode == "full":
        token_basis = W_tok_dir
        token_basis_rank = int(token_basis.shape[1])
    elif token_protection_mode == "residual_pca":
        W_tok_resid = W_tok_dir - basis @ (basis.T @ W_tok_dir)
        U_tok, S_tok, _ = torch.linalg.svd(W_tok_resid, full_matrices=False)
        if S_tok.numel() > 0 and S_tok[0] > 0:
            rank_svd = int((S_tok > 1e-6 * S_tok[0]).sum().item())
            if token_protection_rank is None:
                s2 = S_tok[:rank_svd] ** 2
                cum = torch.cumsum(s2, dim=0) / s2.sum().clamp_min(1e-12)
                k_var = int(torch.searchsorted(cum, torch.tensor(0.9)).item()) + 1
                token_basis_rank = max(1, min(rank_svd, k_var))
            else:
                token_basis_rank = max(0, min(int(token_protection_rank), rank_svd))
            if token_basis_rank > 0:
                token_basis = U_tok[:, :token_basis_rank]

    combined = basis if token_basis.shape[1] == 0 else torch.cat(
        [basis, token_basis], dim=1,
    )
    U_comb, S_comb, _ = torch.linalg.svd(combined, full_matrices=False)
    protected_rank = int((S_comb > 1e-6 * S_comb[0]).sum().item())
    basis_protected = U_comb[:, :protected_rank]  # (D, protected_rank)
    P_protected = basis_protected @ basis_protected.T

    eig_vals, eig_vecs = torch.linalg.eigh(
        torch.eye(D) - P_protected,
    )
    U_orth = eig_vecs[:, eig_vals > 0.5]  # (D, orth_dim)
    orth_dim = U_orth.shape[1]

    logger.info(
        f"[bigram-removal] Protected subspace: task={rank}, "
        f"token={token_basis_rank}, total={protected_rank}, "
        f"orth_dim={orth_dim}, mode={token_protection_mode}"
    )

    # ---- 4. Collect OOD hiddens + bigram features at eval positions ------------
    eval_pos_idx = torch.tensor(eval_positions, device=device, dtype=torch.long)

    def _compute_current_bigram(samples):
        B_s, L = samples.shape
        counts = bigram_prefix_counts(samples, vocab_size).float()  # (B, L, V, V)
        current_tok = samples.long()
        b_idx = torch.arange(B_s, device=samples.device).unsqueeze(1).expand(-1, L)
        t_idx = torch.arange(L, device=samples.device).unsqueeze(0).expand(B_s, -1)
        row_counts = counts[b_idx, t_idx, current_tok, :]  # (B, L, V)

        if bigram_transform == "clr":
            row_sum = row_counts.sum(dim=-1, keepdim=True)
            freq = (row_counts + bigram_alpha) / (
                row_sum + bigram_alpha * vocab_size
            )
            logf = torch.log(freq.clamp_min(1e-12))
            return logf - logf.mean(dim=-1, keepdim=True)
        elif bigram_transform == "log1p":
            return torch.log1p(row_counts)
        else:
            row_sum = row_counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
            return torch.sqrt((row_counts / row_sum).clamp_min(0.0))

    all_h_fit, all_bi_fit = [], []
    n_fit_batches = (n_samples_fit + B - 1) // B
    for _ in range(n_fit_batches):
        gen_out = sampler_ood.generate(
            mode="minor", task=None, num_samples=B, epochs=1,
        )
        samp = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
        if samp.dim() == 3:
            samp = samp.squeeze(0)
        samp = samp.to(device)

        cache = {}
        def _h_hook(module, inp, out, _cache=cache):
            h = out if torch.is_tensor(out) else out[0]
            _cache["h"] = h.index_select(1, eval_pos_idx).detach()

        handle = _layer_module().register_forward_hook(_h_hook)
        try:
            with torch.no_grad():
                model(samp)
        finally:
            handle.remove()

        bi = _compute_current_bigram(samp).index_select(1, eval_pos_idx)
        all_h_fit.append(cache["h"].cpu())
        all_bi_fit.append(bi.cpu())
        del samp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    H_ood = torch.cat(all_h_fit, 0).reshape(-1, D).float()
    Bi_ood = torch.cat(all_bi_fit, 0).reshape(-1, vocab_size).float()
    del all_h_fit, all_bi_fit

    if feature_type == "dirichlet_pred":
        Bi_ood = torch.softmax(Bi_ood, dim=-1)

    feat_label = feature_type

    # ---- 5. Fit linear regression h_orth -> features in the reduced space ------
    H_reduced = H_ood @ U_orth  # (N, orth_dim)

    H_aug = torch.cat([H_reduced, torch.ones(H_reduced.shape[0], 1)], dim=1)
    W_bigram_aug = torch.linalg.pinv(H_aug) @ Bi_ood
    W_bigram = W_bigram_aug[:-1]  # (orth_dim, V)

    Bi_pred = H_aug @ W_bigram_aug
    ss_res = ((Bi_ood - Bi_pred) ** 2).sum()
    ss_tot = ((Bi_ood - Bi_ood.mean(0, keepdim=True)) ** 2).sum()
    fit_r2 = 1.0 - (ss_res / ss_tot.clamp_min(1e-12))
    logger.info(f"[bigram-removal] {feat_label} regression R²={fit_r2:.4f}")

    del H_ood, Bi_ood, H_reduced, H_aug, Bi_pred

    U_bi, S_bi, _ = torch.linalg.svd(W_bigram, full_matrices=False)
    feat_rank = int((S_bi > 1e-6 * S_bi[0]).sum().item())
    V_bigram_full = U_orth @ U_bi  # (D, min(orth_dim, V))

    logger.info(
        f"[bigram-removal] W_bigram rank={feat_rank}, "
        f"SVs: {S_bi[:feat_rank].tolist()}"
    )

    # ---- 6. Intervention -------------------------------------------------------
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    sampler_eval_maj, _, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=0)
    sampler_eval_ood, _, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=n_ood)

    def _eval_with_proj(P_proj, eval_sampler, gen_mode, n_samples):
        baseline_losses = {p: [] for p in eval_positions}
        intervened_losses = {p: [] for p in eval_positions}
        n_batches = max(1, (n_samples + B - 1) // B)
        for _ in range(n_batches):
            gen_out = eval_sampler.generate(
                mode=gen_mode, task=None, num_samples=B, epochs=1,
            )
            samples = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
            if samples.dim() == 3:
                samples = samples.squeeze(0)
            samples = samples.to(device)

            with torch.no_grad():
                logits_base = model(samples)

            def hook_fn(module, inp, out, _P=P_proj, _s=scale):
                h = out if torch.is_tensor(out) else out[0]
                h_mod = h - _s * (h @ _P)
                if torch.is_tensor(out):
                    return h_mod
                return (h_mod,) + out[1:]

            handle = _layer_module().register_forward_hook(hook_fn)
            try:
                with torch.no_grad():
                    logits_int = model(samples)
            finally:
                handle.remove()

            for p in eval_positions:
                if p + 1 >= samples.shape[1]:
                    continue
                target = samples[:, p + 1].long()
                loss_b = ce_loss_fn(logits_base[:, p, :], target).mean().item()
                loss_i = ce_loss_fn(logits_int[:, p, :], target).mean().item()
                baseline_losses[p].append(loss_b)
                intervened_losses[p].append(loss_i)

            del samples, logits_base, logits_int
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        bp, ip, vp = [], [], []
        for p in eval_positions:
            if baseline_losses[p]:
                bp.append(np.mean(baseline_losses[p]))
                ip.append(np.mean(intervened_losses[p]))
                vp.append(p)
        return {
            "baseline_per_pos": np.array(bp),
            "intervened_per_pos": np.array(ip),
            "positions": np.array(vp),
        }

    def _split_pos0(res):
        positions = list(res["positions"])
        bp = list(res["baseline_per_pos"])
        ip = list(res["intervened_per_pos"])
        pos0_base = None
        bp_ex, ip_ex = [], []
        for idx, p in enumerate(positions):
            if int(p) == 0:
                pos0_base = bp[idx]
            else:
                bp_ex.append(bp[idx])
                ip_ex.append(ip[idx])
        base_ex = float(np.mean(bp_ex)) if bp_ex else float(np.mean(bp))
        intv_ex = float(np.mean(ip_ex)) if ip_ex else float(np.mean(ip))
        return pos0_base, base_ex, intv_ex

    # =========================================================================
    #  method = "direct": remove full column space of W at once
    # =========================================================================
    if method == "direct":
        V_feat = V_bigram_full[:, :feat_rank].to(device)  # (D, feat_rank)
        P_feat = V_feat @ V_feat.T

        logger.info(
            f"[bigram-removal] Direct removal: projecting out rank-{feat_rank} "
            f"{feat_label} subspace"
        )

        r_maj = _eval_with_proj(P_feat, sampler_eval_maj, "major", n_samples_eval)
        r_ood = _eval_with_proj(P_feat, sampler_eval_ood, "train", n_samples_eval)

        p0_m, base_m, intv_m = _split_pos0(r_maj)
        p0_o, base_o, intv_o = _split_pos0(r_ood)
        delta_m = intv_m - base_m
        delta_o = intv_o - base_o
        gain_maj = (p0_m - base_m) if p0_m is not None else None
        gain_ood = (p0_o - base_o) if p0_o is not None else None

        if print_summary:
            print(f"\n{'=' * 65}")
            print(f"Remove {feat_label} Subspace (direct) — Latent (layer {layer})")
            print(f"{'=' * 65}")
            print(f"  {feat_label} regression R² = {fit_r2:.4f}")
            print(f"  Feature subspace rank = {feat_rank}")
            print(f"  Protected: task={rank} + token={token_basis_rank} "
                  f"= {protected_rank}")
            print(f"\n{'Metric':<30} {'Major':>12} {'OOD':>12}")
            print("-" * 55)
            print(f"{'Baseline CE':<30} {base_m:>12.4f} {base_o:>12.4f}")
            print(f"{'Intervened CE':<30} {intv_m:>12.4f} {intv_o:>12.4f}")
            print(f"{'Delta CE':<30} {delta_m:>12.4f} {delta_o:>12.4f}")
            if gain_maj is not None:
                print(f"{'l_0':<30} {p0_m:>12.4f} {p0_o:>12.4f}")
                print(f"{'l_0 - l_t':<30} {gain_maj:>12.4f} {gain_ood:>12.4f}")
            print(f"{'=' * 65}")

        model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "method": "direct",
            "feature_type": feature_type,
            "feat_rank": feat_rank,
            "bigram_directions": V_bigram_full[:, :feat_rank].cpu(),
            "bigram_singular_values": S_bi[:feat_rank].cpu(),
            "bigram_fit_r2": float(fit_r2),
            "baseline_loss_major": base_m,
            "intervened_loss_major": intv_m,
            "delta_loss_major": delta_m,
            "baseline_loss_ood": base_o,
            "intervened_loss_ood": intv_o,
            "delta_loss_ood": delta_o,
            "baseline_per_pos_major": list(r_maj["baseline_per_pos"]),
            "intervened_per_pos_major": list(r_maj["intervened_per_pos"]),
            "baseline_per_pos_ood": list(r_ood["baseline_per_pos"]),
            "intervened_per_pos_ood": list(r_ood["intervened_per_pos"]),
            "eval_positions": list(r_maj["positions"]),
            "pos0_loss_major": p0_m,
            "pos0_loss_ood": p0_o,
            "gain_major": gain_maj,
            "gain_ood": gain_ood,
            "token_protection_mode": token_protection_mode,
            "token_protection_rank": token_basis_rank,
            "protected_rank": protected_rank,
            "task_subspace_rank": rank,
            "orth_dim": orth_dim,
            "layer": layer,
        }

    # =========================================================================
    #  method = "ablation_sweep": SVD component-by-component removal
    # =========================================================================
    if method != "ablation_sweep":
        raise ValueError(
            f"method must be 'direct' or 'ablation_sweep', got {method!r}"
        )

    max_k = min(ablation_max_k, V_bigram_full.shape[1])

    indiv_delta_maj, indiv_delta_ood = [], []
    cumul_delta_maj, cumul_delta_ood = [], []
    base_ce_maj = None
    base_ce_ood = None
    pos0_loss_maj = None
    pos0_loss_ood = None

    V_dev = V_bigram_full.to(device)

    for k in range(max_k):
        logger.info(f"[bigram-ablation] component {k} (SV={S_bi[k]:.3f}) ...")
        v_k = V_dev[:, k:k+1]  # (D, 1)
        P_k = v_k @ v_k.T

        r_maj = _eval_with_proj(P_k, sampler_eval_maj, "major", n_samples_eval)
        r_ood = _eval_with_proj(P_k, sampler_eval_ood, "train", n_samples_eval)

        p0_m, b_m, i_m = _split_pos0(r_maj)
        p0_o, b_o, i_o = _split_pos0(r_ood)

        if base_ce_maj is None:
            base_ce_maj = b_m
            base_ce_ood = b_o
            if p0_m is not None:
                pos0_loss_maj = p0_m
                pos0_loss_ood = p0_o

        indiv_delta_maj.append(i_m - b_m)
        indiv_delta_ood.append(i_o - b_o)

        V_cum = V_dev[:, :k+1]  # (D, k+1)
        P_cum = V_cum @ V_cum.T
        rc_maj = _eval_with_proj(P_cum, sampler_eval_maj, "major", n_samples_eval)
        rc_ood = _eval_with_proj(P_cum, sampler_eval_ood, "train", n_samples_eval)
        _, bc_m, ic_m = _split_pos0(rc_maj)
        _, bc_o, ic_o = _split_pos0(rc_ood)
        cumul_delta_maj.append(ic_m - bc_m)
        cumul_delta_ood.append(ic_o - bc_o)

    gain_maj = (pos0_loss_maj - base_ce_maj) if pos0_loss_maj is not None else None
    gain_ood = (pos0_loss_ood - base_ce_ood) if pos0_loss_ood is not None else None

    # ---- Plot (ablation sweep) ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(max(14, 1.2 * max_k), 6))
    x = np.arange(max_k)
    bar_w = 0.35

    for ax, delta_m, delta_o, panel_title, xlabel in [
        (axes[0], indiv_delta_maj, indiv_delta_ood,
         "Individual component removal", f"{feat_label} direction index"),
        (axes[1], cumul_delta_maj, cumul_delta_ood,
         "Cumulative component removal", "Remove directions 0..k"),
    ]:
        ax.bar(x - bar_w / 2, delta_m, bar_w,
               label="Major", color="#2196F3", alpha=0.85)
        ax.bar(x + bar_w / 2, delta_o, bar_w,
               label="OOD", color="#FF9800", alpha=0.85)
        if gain_maj is not None:
            ax.axhline(gain_maj, color="#2196F3", ls="--", lw=2.0,
                       label=f"Major $l_0 - l_t$ ({gain_maj:.3f})")
            ax.axhline(gain_ood, color="#FF9800", ls="--", lw=2.0,
                       label=f"OOD $l_0 - l_t$ ({gain_ood:.3f})")
        for i, (vm, vo) in enumerate(zip(delta_m, delta_o)):
            ax.text(x[i] - bar_w / 2, vm, f"{vm:.3f}",
                    ha="center", va="bottom", fontsize=9)
            ax.text(x[i] + bar_w / 2, vo, f"{vo:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("CE Loss Increase", fontsize=13)
        ax.set_title("", fontsize=18)
        ax.set_xticks(x)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    sv_str = ", ".join(f"{s:.3f}" for s in S_bi[:max_k].numpy())
    prot_tag = (f", token_prot={token_protection_mode}"
                if token_protection_mode != "none" else "")
    fig.suptitle("", fontsize=18, y=1.05)
    plt.tight_layout()
    if show:
        plt.show()

    if print_summary:
        print(f"\n{'=' * 75}")
        print(f"{feat_label} Direction Ablation — Latent (layer {layer})")
        print(f"{'=' * 75}")
        print(f"  {feat_label} regression R² = {fit_r2:.4f}")
        print(f"  Protected subspace: task={rank}, token={token_basis_rank}, "
              f"total={protected_rank}")
        print(f"  {'k':>3}  {'SV':>8}  "
              f"{'Δ Indiv Maj':>12} {'Δ Indiv OOD':>12}  "
              f"{'Δ Cumul Maj':>12} {'Δ Cumul OOD':>12}")
        print("-" * 75)
        for k in range(max_k):
            print(f"  {k:>3}  {S_bi[k].item():>8.3f}  "
                  f"{indiv_delta_maj[k]:>12.4f} {indiv_delta_ood[k]:>12.4f}  "
                  f"{cumul_delta_maj[k]:>12.4f} {cumul_delta_ood[k]:>12.4f}")
        print(f"\n  Baseline: Major={base_ce_maj:.4f}, OOD={base_ce_ood:.4f}")
        if gain_maj is not None:
            print(f"  l_0:      Major={pos0_loss_maj:.4f}, OOD={pos0_loss_ood:.4f}")
            print(f"  l_0-l_t:  Major={gain_maj:.4f}, OOD={gain_ood:.4f}")
        print(f"{'=' * 75}")

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "fig": fig,
        "method": "ablation_sweep",
        "feature_type": feature_type,
        "bigram_directions": V_bigram_full.cpu(),
        "bigram_singular_values": S_bi.cpu(),
        "bigram_fit_r2": float(fit_r2),
        "indiv_delta_major": indiv_delta_maj,
        "indiv_delta_ood": indiv_delta_ood,
        "cumul_delta_major": cumul_delta_maj,
        "cumul_delta_ood": cumul_delta_ood,
        "base_ce_major": base_ce_maj,
        "base_ce_ood": base_ce_ood,
        "pos0_loss_major": pos0_loss_maj,
        "pos0_loss_ood": pos0_loss_ood,
        "gain_major": gain_maj,
        "gain_ood": gain_ood,
        "token_protection_mode": token_protection_mode,
        "token_protection_rank": token_basis_rank,
        "protected_rank": protected_rank,
        "task_subspace_rank": rank,
        "orth_dim": orth_dim,
        "layer": layer,
    }


# ---------------------------------------------------------------------------
#  plot_remove_bigram_subspace_across_layers
# ---------------------------------------------------------------------------

def plot_remove_bigram_subspace_across_layers(
    exp_name: str,
    layers: Optional[list] = None,
    n_ood: int = 30,
    B: int = 64,
    n_samples_eval: int = 500,
    n_samples_fit: int = 2000,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    center_task_vecs: bool = False,
    eval_positions: Optional[list] = None,
    token_protection_mode: str = "residual_pca",
    token_protection_rank: Optional[int] = 3,
    bigram_transform: str = "clr",
    bigram_alpha: float = 0.5,
    feature_type: str = "bigram_clr",
    scale: float = 1.0,
    extraction_point: str = "post_attn",
    probe_method: str = "ols",
    figsize: tuple = (10, 6),
    show: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Sweep ``intervene_remove_bigram_subspace`` (direct)
    across layers and plot delta-loss bars with ``l_0 - l_t`` reference.

    Returns ``(fig, all_results)``.
    """
    import matplotlib.pyplot as plt

    if layers is None:
        _, _, config = nu.load_everything("latent", exp_name)
        layers = list(range(config.model.num_layers))

    all_results = {}
    for l in layers:
        logger.info(f"[bigram-removal sweep] layer {l} ...")
        res = intervene_remove_bigram_subspace(
            exp_name=exp_name,
            layer=l,
            n_ood=n_ood,
            B=B,
            n_samples_eval=n_samples_eval,
            n_samples_fit=n_samples_fit,
            step=step,
            fit_n_samples=fit_n_samples,
            fit_positions=fit_positions,
            center_task_vecs=center_task_vecs,
            eval_positions=eval_positions,
            token_protection_mode=token_protection_mode,
            token_protection_rank=token_protection_rank,
            bigram_transform=bigram_transform,
            bigram_alpha=bigram_alpha,
            feature_type=feature_type,
            method="direct",
            scale=scale,
            extraction_point=extraction_point,
            probe_method=probe_method,
            show=False,
            print_summary=True,
        )
        all_results[l] = res

    # ---- Compute uninformative gap l_0 - l_t (averaged across layers) --------
    has_pos0 = all_results[layers[0]].get("pos0_loss_major") is not None
    if has_pos0:
        pos0_loss_maj = float(np.mean([
            all_results[l]["pos0_loss_major"] for l in layers
        ]))
        pos0_loss_ood = float(np.mean([
            all_results[l]["pos0_loss_ood"] for l in layers
        ]))
        base_maj_avg = float(np.mean([
            all_results[l]["baseline_loss_major"] for l in layers
        ]))
        base_ood_avg = float(np.mean([
            all_results[l]["baseline_loss_ood"] for l in layers
        ]))
        gain_maj = pos0_loss_maj - base_maj_avg
        gain_ood = pos0_loss_ood - base_ood_avg
    else:
        gain_maj = gain_ood = None

    # ---- Delta loss bars — colours / style match plot_optimal_orth_direction ----
    COLORS = {"maj": "#2166ac", "ood": "#d6604d"}
    bw_bar = 0.22
    g_step = 0.24
    MC     = {"maj": -g_step, "ood": 0.0}

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    x = np.arange(len(layers))

    delta_maj = [all_results[l]["delta_loss_major"] for l in layers]
    delta_ood = [all_results[l]["delta_loss_ood"] for l in layers]

    ax.bar(x + MC["maj"], delta_maj, bw_bar,
           label="Major", color=COLORS["maj"], linewidth=0, zorder=3)
    ax.bar(x + MC["ood"], delta_ood, bw_bar,
           label="OOD", color=COLORS["ood"], linewidth=0, zorder=3)

    if gain_maj is not None:
        ax.axhline(gain_maj, color=COLORS["maj"], ls="--", lw=0.9, alpha=0.45,
                   label=f"Maj. gain ({gain_maj:.3f})")
        ax.axhline(gain_ood, color=COLORS["ood"], ls=":", lw=0.9, alpha=0.45,
                   label=f"OOD gain ({gain_ood:.3f})")

    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel("Cross-entropy loss increase", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.5, color="grey")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.9,
              edgecolor="lightgrey", columnspacing=0.6,
              handlelength=1.2, handletextpad=0.4, borderpad=0.5)
    plt.tight_layout(pad=0.5)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, all_results
