"""Decomposition and direction analysis for the Coin task."""

import gc

import numpy as np
import torch
from typing import Optional

import icl.utils.notebook_utils as nu
from icl.coin.coin_ood_analysis import get_new_sampler
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)


def decompose_hidden_posterior_vs_unigram_coin(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples_fit: int = 1000,
    n_samples_eval: int = 500,
    step: Optional[int] = None,
    n_ood: int = 30,
    positions: Optional[list] = None,
    validation_split: float = 0.2,
    unigram_transform: str = "clr",
    unigram_alpha: float = 0.5,
    verbose: bool = False,
    print_summary: bool = True,
) -> dict:
    """
    Decompose hidden representations into posterior and unigram components.

    Fits linear probes on major-mode data:

    - ``posterior_logits -> hiddens``  (R² post)
    - ``unigram_features -> hiddens`` (R² uni)
    - ``[posterior, unigram] -> hiddens`` (R² combined)

    Then evaluates the *same* probes on OOD data (random coins not in pool).
    Measures orthogonality via cosine similarity and cross-R² between the
    posterior-predicted and unigram-predicted hidden components.

    If the two components are truly orthogonal:
      R²_combined ≈ R²_post + R²_uni  (additivity gap ≈ 0)
      cosine(H_post_pred, H_uni_pred) ≈ 0

    Parameters
    ----------
    exp_name : str
    layer : int
    B : int
    n_samples_fit : int
        Number of major-mode samples for fitting probes.
    n_samples_eval : int
        Number of OOD samples for evaluation.
    step : int, optional
    n_ood : int
        Number of random OOD coins to generate.
    positions : list, optional
        Token positions to analyse.  ``None`` → first 10.
    validation_split : float
    unigram_transform, unigram_alpha
        Forwarded to unigram feature computation (CLR / log1p / sqrt_freq).
    verbose, print_summary : bool

    Returns
    -------
    dict with keys:
        r2_post_major, r2_uni_major, r2_combined_major,
        r2_post_ood, r2_uni_ood, r2_combined_ood,
        cos_post_uni_major, cos_post_uni_ood,
        cross_r2_post_from_uni, cross_r2_uni_from_post,
        additivity_gap_major, additivity_gap_ood,
        layer
    """
    if unigram_transform not in {"clr", "log1p", "sqrt_freq"}:
        raise ValueError(
            f"Unsupported unigram_transform={unigram_transform!r}. "
            f"Use one of: 'clr', 'log1p', 'sqrt_freq'."
        )

    # ---- load model / samplers ----
    _, _, config = nu.load_everything("coin", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True,
    )
    model.eval().to(config.device)
    device = config.device

    sampler_major, _ = get_new_sampler(exp_name, n_minor=0, n_ood=0)

    sampler_ood, _ = get_new_sampler(exp_name, n_minor=0, n_ood=n_ood)

    seq_len = sampler_major.seq_len
    if positions is None:
        positions = list(range(min(10, seq_len)))
    position_indices = torch.tensor(positions, device=device, dtype=torch.long)

    # ---- posterior setup (position-specific) ----
    maj_p = sampler_major.major_p.to(device=device, dtype=torch.float32)
    Kmaj = maj_p.shape[0]
    K_post = int(sampler_major.num_states)
    _eps_post = 1e-30
    _log_prior = torch.full(
        (Kmaj,), -float(np.log(max(1, Kmaj))),
        device=device, dtype=torch.float32,
    )
    _log_maj_p = torch.log(maj_p.clamp(min=_eps_post))  # (Kmaj, K)

    # ---- data collection helper ----
    def _collect(sampler, mode, n_samples):
        all_H, all_P, all_U = [], [], []
        n_batches = (n_samples + B - 1) // B
        for _ in range(n_batches):
            gen_out = sampler.generate(
                mode=mode, task=None, num_samples=B, epochs=1,
            )
            samples = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
            if samples.dim() == 3:
                samples = samples.squeeze(0)
            samples = samples.to(device)

            cache = {}
            layer_mod = model.layers[layer].attn_block

            def hook_fn(module, inp, out):
                if torch.is_tensor(out):
                    cache["hidden"] = out.index_select(
                        dim=1, index=position_indices,
                    ).detach()
                elif (isinstance(out, tuple) and len(out) > 0
                      and torch.is_tensor(out[0])):
                    cache["hidden"] = out[0].index_select(
                        dim=1, index=position_indices,
                    ).detach()

            handle = layer_mod.register_forward_hook(hook_fn)
            try:
                with torch.no_grad():
                    logits_full = model(samples)
                hiddens = cache["hidden"]
            finally:
                handle.remove()

            V = int(logits_full.shape[-1])
            onehot_seq = torch.nn.functional.one_hot(
                samples.long(), num_classes=V,
            ).float()
            prefix_counts = onehot_seq.cumsum(dim=1).index_select(
                dim=1, index=position_indices,
            )

            counts_for_post = prefix_counts[:, :, :K_post].float()
            loglik = torch.einsum(
                "bpk,tk->bpt", counts_for_post, _log_maj_p,
            )
            unnorm = loglik + _log_prior.view(1, 1, -1)
            log_post = unnorm - torch.logsumexp(
                unnorm, dim=-1, keepdim=True,
            )
            posteriors_at_pos = torch.exp(log_post)  # (B, P, Kmaj)

            prefix_len = (position_indices + 1).float().view(1, -1, 1)

            if unigram_transform == "clr":
                freq = (prefix_counts + unigram_alpha) / (
                    prefix_len + unigram_alpha * V
                )
                logf = torch.log(freq.clamp_min(1e-12))
                unigram = logf - logf.mean(dim=-1, keepdim=True)
            elif unigram_transform == "log1p":
                unigram = torch.log1p(prefix_counts)
            else:
                freq = prefix_counts / prefix_len.clamp_min(1.0)
                unigram = torch.sqrt(freq.clamp_min(0.0))

            all_H.append(hiddens.cpu())
            all_P.append(posteriors_at_pos.cpu())
            all_U.append(unigram.cpu())

            del samples, posteriors_at_pos, hiddens, logits_full
            del onehot_seq, prefix_counts, unigram
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        H = torch.cat(all_H, dim=0)
        P = torch.cat(all_P, dim=0)
        U = torch.cat(all_U, dim=0)
        n = H.shape[0] * H.shape[1]
        return H.reshape(n, -1), P.reshape(n, -1), U.reshape(n, -1)

    # ---- closed-form helpers ----
    def _fit(X, Y):
        X_, Y_ = X.float(), Y.float()
        ones = torch.ones(X_.shape[0], 1)
        X_aug = torch.cat([X_, ones], dim=1)
        W_aug = torch.linalg.pinv(X_aug) @ Y_
        return W_aug[:-1], W_aug[-1]

    def _r2_pred(X, Y, W, b):
        Yp = X.float() @ W + b
        ss_res = ((Y.float() - Yp) ** 2).sum().item()
        ss_tot = ((Y.float() - Y.float().mean(0)) ** 2).sum().item()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return r2, Yp

    def _cos(a, b_):
        a_, b2 = a.reshape(-1).float(), b_.reshape(-1).float()
        return (torch.dot(a_, b2) / (a_.norm() * b2.norm() + 1e-10)).item()

    def _cross_r2(A, B_):
        N_ = B_.shape[0]
        n_tr_ = int(0.8 * N_)
        perm_ = torch.randperm(N_)
        tr_, va_ = perm_[:n_tr_], perm_[n_tr_:]
        ones_tr_ = torch.ones(n_tr_, 1, dtype=B_.dtype)
        Ba_tr = torch.cat([B_[tr_], ones_tr_], 1)
        Wc = torch.linalg.lstsq(Ba_tr, A[tr_]).solution
        ones_va_ = torch.ones(va_.shape[0], 1, dtype=B_.dtype)
        pred_va_ = torch.cat([B_[va_], ones_va_], 1) @ Wc
        ss_res = ((A[va_] - pred_va_) ** 2).sum().item()
        ss_tot = (A[va_] ** 2).sum().item()
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # ---- collect major data for fitting ----
    if verbose:
        logger.info(
            f"[decompose] Collecting major data (n={n_samples_fit}) ..."
        )
    H_maj, P_maj, U_maj = _collect(sampler_major, "major", n_samples_fit)

    n_total = H_maj.shape[0]
    n_train = int(n_total * (1 - validation_split))
    idx = torch.randperm(n_total)
    tr, va = idx[:n_train], idx[n_train:]

    H_tr, H_va = H_maj[tr], H_maj[va]
    P_tr, P_va = P_maj[tr], P_maj[va]
    U_tr, U_va = U_maj[tr], U_maj[va]

    # ---- fit probes ----
    W_post, b_post = _fit(P_tr, H_tr)
    W_uni, b_uni = _fit(U_tr, H_tr)

    PU_tr = torch.cat([P_tr, U_tr], dim=1)
    PU_va = torch.cat([P_va, U_va], dim=1)
    W_comb, b_comb = _fit(PU_tr, H_tr)

    # ---- evaluate on major validation ----
    r2_post_maj, Yp_post_maj = _r2_pred(P_va, H_va, W_post, b_post)
    r2_uni_maj, Yp_uni_maj = _r2_pred(U_va, H_va, W_uni, b_uni)
    r2_comb_maj, _ = _r2_pred(PU_va, H_va, W_comb, b_comb)

    Yp_post_maj_c = Yp_post_maj - Yp_post_maj.mean(0, keepdim=True)
    Yp_uni_maj_c = Yp_uni_maj - Yp_uni_maj.mean(0, keepdim=True)

    cos_maj = _cos(Yp_post_maj_c, Yp_uni_maj_c)
    cr2_post_from_uni = _cross_r2(Yp_post_maj_c, Yp_uni_maj_c)
    cr2_uni_from_post = _cross_r2(Yp_uni_maj_c, Yp_post_maj_c)

    # ---- collect OOD data and evaluate ----
    if verbose:
        logger.info(
            f"[decompose] Collecting OOD data "
            f"(n={n_samples_eval}, n_ood_coins={n_ood}) ..."
        )
    H_ood, P_ood, U_ood = _collect(sampler_ood, "minor", n_samples_eval)

    r2_post_ood, Yp_post_ood = _r2_pred(P_ood, H_ood, W_post, b_post)
    r2_uni_ood, Yp_uni_ood = _r2_pred(U_ood, H_ood, W_uni, b_uni)
    PU_ood = torch.cat([P_ood, U_ood], dim=1)
    r2_comb_ood, _ = _r2_pred(PU_ood, H_ood, W_comb, b_comb)

    Yp_post_ood_c = Yp_post_ood - Yp_post_ood.mean(0, keepdim=True)
    Yp_uni_ood_c = Yp_uni_ood - Yp_uni_ood.mean(0, keepdim=True)
    cos_ood = _cos(Yp_post_ood_c, Yp_uni_ood_c)

    # ---- cleanup ----
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    add_gap_maj = r2_comb_maj - (r2_post_maj + r2_uni_maj)
    add_gap_ood = r2_comb_ood - (r2_post_ood + r2_uni_ood)

    results = {
        "r2_post_major": r2_post_maj,
        "r2_uni_major": r2_uni_maj,
        "r2_combined_major": r2_comb_maj,
        "r2_post_ood": r2_post_ood,
        "r2_uni_ood": r2_uni_ood,
        "r2_combined_ood": r2_comb_ood,
        "cos_post_uni_major": cos_maj,
        "cos_post_uni_ood": cos_ood,
        "cross_r2_post_from_uni": cr2_post_from_uni,
        "cross_r2_uni_from_post": cr2_uni_from_post,
        "additivity_gap_major": add_gap_maj,
        "additivity_gap_ood": add_gap_ood,
        "layer": layer,
    }

    if print_summary:
        print(f"\n{'=' * 60}")
        print(f"Posterior vs Unigram Decomposition  (layer {layer})")
        print(f"{'=' * 60}")
        print(
            f"  Probes fitted on major data "
            f"({n_train} train, {len(va)} val)"
        )
        print(f"  OOD: {n_ood} random coins, {n_samples_eval} samples\n")
        print(f"{'Metric':<32} {'Major (val)':>12} {'OOD':>12}")
        print("-" * 56)
        print(
            f"{'R2 (posterior -> H)':<32} "
            f"{r2_post_maj:>12.4f} {r2_post_ood:>12.4f}"
        )
        print(
            f"{'R2 (unigram -> H)':<32} "
            f"{r2_uni_maj:>12.4f} {r2_uni_ood:>12.4f}"
        )
        print(
            f"{'R2 (combined -> H)':<32} "
            f"{r2_comb_maj:>12.4f} {r2_comb_ood:>12.4f}"
        )
        print(
            f"{'Sum(R2_post + R2_uni)':<32} "
            f"{r2_post_maj + r2_uni_maj:>12.4f} "
            f"{r2_post_ood + r2_uni_ood:>12.4f}"
        )
        print(
            f"{'Additivity gap':<32} "
            f"{add_gap_maj:>12.4f} {add_gap_ood:>12.4f}"
        )
        print(
            f"\n{'cos(H_post, H_uni)':<32} "
            f"{cos_maj:>12.4f} {cos_ood:>12.4f}"
        )
        print(f"{'CrossR2(post <- uni)':<32} {cr2_post_from_uni:>12.4f}")
        print(f"{'CrossR2(uni <- post)':<32} {cr2_uni_from_post:>12.4f}")
        if abs(add_gap_maj) < 0.05:
            print(
                f"\n  Small additivity gap ({add_gap_maj:.4f}) "
                f"-- components are near-orthogonal"
            )

    return results


def plot_decomposition_across_layers_coin(
    exp_name: str,
    layers: Optional[list] = None,
    B: int = 64,
    n_samples_fit: int = 1000,
    n_samples_eval: int = 500,
    step: Optional[int] = None,
    n_ood: int = 30,
    positions: Optional[list] = None,
    unigram_transform: str = "clr",
    unigram_alpha: float = 0.5,
    figsize: tuple = (18, 6),
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Sweep ``decompose_hidden_posterior_vs_unigram_coin`` across
    layers and plot:

    - Left:   Major (val) — R2_post, R2_uni, R2_combined per layer
    - Centre: OOD         — same metrics (same probes, OOD data)
    - Right:  Orthogonality — cosine similarity & additivity gap

    Returns ``(fig, all_results)`` where *all_results* maps layer -> dict.
    """
    import matplotlib.pyplot as plt

    if layers is None:
        _, _, config = nu.load_everything("coin", exp_name)
        layers = list(range(config.model.num_layers))

    all_results = {}
    for l in layers:
        logger.info(f"[decompose sweep] layer {l} ...")
        res = decompose_hidden_posterior_vs_unigram_coin(
            exp_name=exp_name,
            layer=l,
            B=B,
            n_samples_fit=n_samples_fit,
            n_samples_eval=n_samples_eval,
            step=step,
            n_ood=n_ood,
            positions=positions,
            unigram_transform=unigram_transform,
            unigram_alpha=unigram_alpha,
            verbose=False,
            print_summary=False,
        )
        all_results[l] = res

    # ---- plotting ----
    metrics = {
        "Posterior": "r2_post",
        "Unigram": "r2_uni",
        "Combined": "r2_combined",
    }
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    x = np.arange(len(layers))
    n_m = len(metrics)
    bar_w = 0.8 / n_m

    for ax, suffix, ax_title in [
        (ax1, "_major", "Major (val)"),
        (ax2, "_ood", "OOD"),
    ]:
        for i, (name, key_prefix) in enumerate(metrics.items()):
            key = key_prefix + suffix
            vals = [all_results[l][key] for l in layers]
            offset = (i - (n_m - 1) / 2) * bar_w
            bars = ax.bar(
                x + offset, vals, bar_w,
                label=name, color=colors[i], alpha=0.85,
            )
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    lbl = (
                        f"{v:.2f}".lstrip("0")
                        if 0 < abs(v) < 1 else f"{v:.2f}"
                    )
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        lbl, ha="center", va="bottom", fontsize=11,
                    )
        ax.set_xlabel("Layer", fontsize=16)
        ax.set_ylabel("R\u00b2", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in layers])
        ax.tick_params(labelsize=14)
        ax.set_title("", fontsize=18)
        ax.legend(fontsize=12)
        ax.grid(axis="y", alpha=0.3)

    # right panel: orthogonality
    cos_maj_vals = [all_results[l]["cos_post_uni_major"] for l in layers]
    cos_ood_vals = [all_results[l]["cos_post_uni_ood"] for l in layers]
    gap_maj_vals = [all_results[l]["additivity_gap_major"] for l in layers]
    gap_ood_vals = [all_results[l]["additivity_gap_ood"] for l in layers]

    ax3.plot(
        layers, cos_maj_vals, "o-", color="#2196F3",
        label="cos (major)", linewidth=2, markersize=7,
    )
    ax3.plot(
        layers, cos_ood_vals, "s--", color="#FF9800",
        label="cos (OOD)", linewidth=2, markersize=7,
    )
    ax3.plot(
        layers, gap_maj_vals, "^-", color="#9C27B0",
        label="add. gap (major)", linewidth=2, markersize=7,
    )
    ax3.plot(
        layers, gap_ood_vals, "v--", color="#E91E63",
        label="add. gap (OOD)", linewidth=2, markersize=7,
    )
    ax3.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax3.set_xlabel("Layer", fontsize=16)
    ax3.set_ylabel("Value", fontsize=16)
    ax3.set_title("Orthogonality", fontsize=16)
    ax3.tick_params(labelsize=14)
    ax3.legend(fontsize=11)
    ax3.grid(alpha=0.3)

    sup = title or "Hidden Decomposition: Posterior vs Unigram"
    fig.suptitle("", fontsize=18, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, all_results
