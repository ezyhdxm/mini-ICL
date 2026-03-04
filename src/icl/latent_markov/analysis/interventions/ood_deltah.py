"""Remove OOD delta_h SVD components (latent Markov, non-padded).

Pipeline
--------
1. Joint OLS probe  →  task (W_task) and token (W_tok) weight matrices.
2. SVD(W_tok), keep directions explaining ≥ ``token_var_threshold`` of variance.
3. SVD(W_task) → task basis.   Combine with token basis → protected subspace.
4. OOD Δh (= h_t − h_0(x_t))  projected orthogonal to protected subspace.
5. SVD on residual  →  ablation sweep (individual + cumulative removal).
"""

import gc
import numpy as np
import torch
from typing import Optional

import icl.utils.notebook_utils as nu
from icl.latent_markov.analysis.ood import get_latent_sampler
from icl.utils.logger import setup_logger
from icl.latent_markov.analysis.probes import train_linear_hidden_predictor

logger = setup_logger(__name__)


@torch.no_grad()
def intervene_remove_ood_deltah_subspace(
    exp_name: str,
    layer: int,
    n_ood: int = 30,
    B: int = 64,
    n_samples_eval: int = 500,
    step: Optional[int] = None,
    fit_n_samples: int = 5000,
    fit_positions: Optional[list] = None,
    center_task_vecs: bool = False,
    min_position: int = 20,
    ablation_max_k: int = 10,
    eval_positions: Optional[list] = None,
    token_var_threshold: float = 0.9,
    show: bool = True,
    figsize: tuple = (14, 6),
    print_summary: bool = True,
) -> dict:
    """SVD ablation sweep on OOD Δh after projecting out task + token subspaces.

    Parameters
    ----------
    token_var_threshold : float
        Fraction of W_tok singular-value energy to keep (default 0.9).
    ablation_max_k : int
        Number of SVD components to test in the ablation sweep.
    """
    import matplotlib.pyplot as plt
    from icl.latent_markov.legacy.coin_latent_task_vecs import (
        extract_hidden_multi_coin_latent,
    )

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

    sampler, _, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=n_ood)
    K_major = sampler.n_major_tasks
    seq_len = sampler.seq_len
    V = sampler.num_states
    n_total = K_major + n_ood

    if fit_positions is None:
        fit_positions = list(range(max(0, seq_len // 2), seq_len))
    if eval_positions is None:
        eval_positions = list(range(seq_len))
    if 0 not in eval_positions:
        eval_positions = [0] + list(eval_positions)

    # ---- 2. Joint probe → task & token subspaces --------------------------------
    fit_res = train_linear_hidden_predictor(
        exp_name=exp_name, layer=layer,
        n_samples=fit_n_samples, positions=fit_positions,
        sample_mode="major", n_minor=-1, step=step,
        print_summary=False, skip_baselines=True,
    )
    W_task = fit_res["model_weight"].float()   # (K_major, D)
    W_tok  = fit_res["token_weight"].float()   # (V, D)

    if center_task_vecs:
        W_task = W_task - W_task.mean(0, keepdim=True)

    def _basis_from_svd(M, var_threshold=None):
        """Return an orthonormal basis spanning M's row space (or top-k by variance)."""
        _, S, Vt = torch.linalg.svd(M, full_matrices=False)
        alive = (S > 1e-6 * S[0]).sum().item()
        if var_threshold is not None and alive > 0:
            cum = torch.cumsum(S[:alive] ** 2, 0)
            alive = min(alive, int((cum < var_threshold * cum[-1]).sum().item()) + 1)
        return Vt[:alive].T, alive    # (D, rank)

    task_basis, task_rank = _basis_from_svd(W_task)
    tok_basis,  tok_rank  = _basis_from_svd(W_tok, var_threshold=token_var_threshold)

    # Orthogonalise the union → protected projector
    combined = torch.cat([task_basis, tok_basis], dim=1)
    U_comb, S_comb, _ = torch.linalg.svd(combined, full_matrices=False)
    prot_rank = int((S_comb > 1e-6 * S_comb[0]).sum().item())
    P_prot = U_comb[:, :prot_rank]                 # (D, prot_rank)
    P_prot_mat = P_prot @ P_prot.T                 # (D, D)

    logger.info(
        f"[ood-deltah] Probe R²={fit_res['val_r2']:.4f}, "
        f"protected rank: task={task_rank} + token={tok_rank} = {prot_rank}"
    )

    # ---- 3. h_0(v) lookup for every vocab token v --------------------------------
    dummy = torch.zeros(V, seq_len, dtype=torch.long, device=device)
    dummy[:, 0] = torch.arange(V, device=device)
    pos0 = torch.tensor([0], device=device, dtype=torch.long)
    h0_lookup = extract_hidden_multi_coin_latent(
        model, dummy, layers=[layer], task_pos=pos0,
    )[0, :, 0, :].cpu().float()                     # (V, D)

    # ---- 4. Compute OOD Δh and SVD on residual ----------------------------------
    task_pos_all = torch.arange(seq_len, device=device, dtype=torch.long)
    all_tokens = torch.empty(n_total, B, seq_len, dtype=torch.long)
    hiddens    = torch.empty(n_total, seq_len, B, D, dtype=torch.float32)

    for t in range(n_total):
        samples = sampler.generate(mode="testing", task=t, num_samples=B)[0].to(device)
        all_tokens[t] = samples.cpu()
        h = extract_hidden_multi_coin_latent(
            model, samples, layers=[layer], task_pos=task_pos_all,
        )
        hiddens[t] = h[0].permute(1, 0, 2).cpu().float()
        del samples, h

    h_base = h0_lookup[all_tokens.reshape(-1)].reshape(
        n_total, B, seq_len, D,
    ).permute(0, 2, 1, 3)
    delta_h = hiddens - h_base                       # (n_total, T, B, D)

    late = torch.arange(seq_len) >= min_position
    dh_ood = delta_h[K_major:][:, late].reshape(-1, D).float()

    dh_resid = dh_ood - dh_ood @ P_prot_mat
    _, S_all, Vt_all = torch.linalg.svd(dh_resid, full_matrices=False)

    del hiddens, delta_h, h_base, all_tokens, dh_ood, dh_resid
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- 5. Ablation sweep -------------------------------------------------------
    ce_fn = torch.nn.CrossEntropyLoss(reduction="none")
    sampler_maj, _, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=0)
    sampler_ood, _, _ = get_latent_sampler(exp_name, n_minor=0, n_ood=n_ood)

    def _eval_proj(P, samp, mode, n):
        """CE with & without projecting out P from layer activations."""
        base = {p: [] for p in eval_positions}
        intv = {p: [] for p in eval_positions}
        for _ in range(max(1, (n + B - 1) // B)):
            seq = samp.generate(mode=mode, task=None, num_samples=B, epochs=1)
            seq = seq[0] if isinstance(seq, (tuple, list)) else seq
            if seq.dim() == 3:
                seq = seq.squeeze(0)
            seq = seq.to(device)

            logits_b = model(seq)

            def hook(mod, inp, out, _P=P):
                h = out if torch.is_tensor(out) else out[0]
                h_mod = h - h @ _P
                return h_mod if torch.is_tensor(out) else (h_mod,) + out[1:]

            handle = model.layers[layer].attn_block.register_forward_hook(hook)
            try:
                logits_i = model(seq)
            finally:
                handle.remove()

            for p in eval_positions:
                if p + 1 >= seq.shape[1]:
                    continue
                tgt = seq[:, p + 1].long()
                base[p].append(ce_fn(logits_b[:, p], tgt).mean().item())
                intv[p].append(ce_fn(logits_i[:, p], tgt).mean().item())

            del seq, logits_b, logits_i
        return {
            p: (float(np.mean(base[p])), float(np.mean(intv[p])))
            for p in eval_positions if base[p]
        }

    max_k = min(ablation_max_k, len(S_all))
    indiv_d_maj, indiv_d_ood = [], []
    cumul_d_maj, cumul_d_ood = [], []
    base_maj = base_ood = pos0_maj = pos0_ood = None

    for k in range(max_k):
        logger.info(f"[ablation] component {k} ...")
        v_k = Vt_all[k:k + 1].T.to(device)
        P_k = v_k @ v_k.T

        rm = _eval_proj(P_k, sampler_maj, "major", n_samples_eval)
        ro = _eval_proj(P_k, sampler_ood, "train",  n_samples_eval)

        def _agg(r):
            """(pos0_base, mean_base_excl0, mean_intv_excl0)"""
            b0 = r.get(0, (None, None))[0]
            bp = [v[0] for p, v in r.items() if p != 0]
            ip = [v[1] for p, v in r.items() if p != 0]
            return b0, float(np.mean(bp)), float(np.mean(ip))

        p0m, bm, im = _agg(rm)
        p0o, bo, io = _agg(ro)

        if base_maj is None:
            base_maj, base_ood = bm, bo
            pos0_maj, pos0_ood = p0m, p0o

        indiv_d_maj.append(im - bm)
        indiv_d_ood.append(io - bo)

        V_cum = Vt_all[:k + 1].T.to(device)
        P_cum = V_cum @ V_cum.T
        rcm = _eval_proj(P_cum, sampler_maj, "major", n_samples_eval)
        rco = _eval_proj(P_cum, sampler_ood, "train",  n_samples_eval)
        _, bcm, icm = _agg(rcm)
        _, bco, ico = _agg(rco)
        cumul_d_maj.append(icm - bcm)
        cumul_d_ood.append(ico - bco)

    gain_maj = (pos0_maj - base_maj) if pos0_maj is not None else None
    gain_ood = (pos0_ood - base_ood) if pos0_ood is not None else None

    # ---- 6. Plot -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    x = np.arange(max_k)
    w = 0.35

    for ax, dm, do, title, xlabel in [
        (axes[0], indiv_d_maj, indiv_d_ood,
         "Individual component removal", "SVD component index"),
        (axes[1], cumul_d_maj, cumul_d_ood,
         "Cumulative component removal", "Remove components 0..k"),
    ]:
        ax.bar(x - w / 2, dm, w, label="Major", color="#2196F3", alpha=0.85)
        ax.bar(x + w / 2, do, w, label="OOD",   color="#FF9800", alpha=0.85)
        if gain_maj is not None:
            ax.axhline(gain_maj, color="#2196F3", ls="--", lw=2,
                       label=f"Major $l_0-l_t$ ({gain_maj:.3f})")
            ax.axhline(gain_ood, color="#FF9800", ls="--", lw=2,
                       label=f"OOD $l_0-l_t$ ({gain_ood:.3f})")
        for i, (vm, vo) in enumerate(zip(dm, do)):
            ax.text(x[i] - w / 2, vm, f"{vm:.3f}", ha="center", va="bottom", fontsize=9)
            ax.text(x[i] + w / 2, vo, f"{vo:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("CE Loss Increase", fontsize=13)
        ax.set_title("", fontsize=18)
        ax.set_xticks(x)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    sv_str = ", ".join(f"{s:.3f}" for s in S_all[:max_k].numpy())
    fig.suptitle("", fontsize=18, y=1.05)
    plt.tight_layout()
    if show:
        plt.show()

    if print_summary:
        print(f"\n{'='*70}")
        print(f"SVD Ablation Sweep — Latent (layer {layer})")
        print(f"{'='*70}")
        print(f"  {'k':>3}  {'SV':>8}  "
              f"{'Δ Indiv Maj':>12} {'Δ Indiv OOD':>12}  "
              f"{'Δ Cumul Maj':>12} {'Δ Cumul OOD':>12}")
        print("-" * 70)
        for k in range(max_k):
            print(f"  {k:>3}  {S_all[k].item():>8.3f}  "
                  f"{indiv_d_maj[k]:>12.4f} {indiv_d_ood[k]:>12.4f}  "
                  f"{cumul_d_maj[k]:>12.4f} {cumul_d_ood[k]:>12.4f}")
        print(f"\n  Baseline: Major={base_maj:.4f}, OOD={base_ood:.4f}")
        if gain_maj is not None:
            print(f"  l_0:      Major={pos0_maj:.4f}, OOD={pos0_ood:.4f}")
            print(f"  l_0-l_t:  Major={gain_maj:.4f}, OOD={gain_ood:.4f}")
        print(f"{'='*70}")

    # ---- cleanup -----------------------------------------------------------------
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "fig": fig,
        "singular_values": S_all.cpu(),
        "svd_Vt": Vt_all.cpu(),
        "indiv_delta_major": indiv_d_maj,
        "indiv_delta_ood": indiv_d_ood,
        "cumul_delta_major": cumul_d_maj,
        "cumul_delta_ood": cumul_d_ood,
        "base_ce_major": base_maj,
        "base_ce_ood": base_ood,
        "pos0_loss_major": pos0_maj,
        "pos0_loss_ood": pos0_ood,
        "gain_major": gain_maj,
        "gain_ood": gain_ood,
        "protected_rank": prot_rank,
        "task_rank": task_rank,
        "token_rank": tok_rank,
        "token_var_threshold": token_var_threshold,
        "layer": layer,
    }
