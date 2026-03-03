import torch
from typing import Optional

from icl.dyck.dyck_bayes import DyckBayes


@torch.no_grad()
def plot_kl_model_vs_two_bayes_dyck(
    exp_name: str,
    modes: tuple = ("train", "major", "minor", "ood"),
    num_samples: int = 512,
    step=None,
    positions=None,
    use_dyck_positions: bool = False,
    n_masks: int = 60,
    uniform_train_tasks: bool = False,
    eps: float = 1e-12,
    figsize=None,
    show: bool = True,
):
    """
    Compare Dyck transformer next-token distributions against two Dyck Bayes baselines.

    Baselines
    ---------
    1) Pool-aware Dyck Bayes (``DyckBayes(..., flag=False)``):
       uses a trie built from the known task pool.
    2) Pool-agnostic Dyck Bayes (``DyckBayes(..., flag=True)``):
       ignores the finite task pool and uses Dyck-combinatorial prior logic.

    For each requested mode, plots KL(model || baseline) vs position with mean +/- 1 std.

    Parameters
    ----------
    positions : list[int], optional
        If provided, only these indices are plotted/analyzed.
        - ``use_dyck_positions=False``: indices on real timeline positions.
        - ``use_dyck_positions=True``: indices on Dyck-token order (0..2*dyck_length-1).
    use_dyck_positions : bool, default=False
        If True, aggregate KL by Dyck-token index (across random masks) instead
        of real timeline positions.
    n_masks : int, default=60
        Number of random masks used when ``use_dyck_positions=True``.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import icl.utils.notebook_utils as nu
    from icl.dyck.dyck_utils import sample_binary_mask

    _, sampler, config = nu.load_everything("dyck", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval().to(config.device)

    p_minor_orig = float(getattr(sampler, "p_minor", 0.0))

    def _extract_samples_from_generate_output(out):
        if torch.is_tensor(out):
            s = out
        elif isinstance(out, (tuple, list)):
            s = out[0]
        else:
            raise ValueError(f"Unsupported generate output type: {type(out)}")
        if s.dim() == 3:
            s = s.squeeze(0)
        return s

    def _extract_samples_and_masks_from_generate_output(out):
        if torch.is_tensor(out):
            raise ValueError("Dyck generate output must include masks.")
        if not isinstance(out, (tuple, list)) or len(out) < 2:
            raise ValueError(f"Unsupported generate output type: {type(out)}")
        s, m = out[0], out[1]
        if s.dim() == 3:
            s = s.squeeze(0)
        if m.dim() == 3:
            m = m.squeeze(0)
        return s, m

    def _kl_model_to_ref(p_model: torch.Tensor, p_ref: torch.Tensor):
        p = p_model.clamp_min(eps)
        q = p_ref.clamp_min(eps)
        return (p * (torch.log(p) - torch.log(q))).sum(dim=-1)

    if isinstance(modes, str):
        modes = (modes,)
    else:
        modes = tuple(modes)

    ncols = 1 if len(modes) == 1 else 2
    nrows = (len(modes) + ncols - 1) // ncols
    if figsize is None:
        figsize = (7 * ncols, 4.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    results = {}
    for mi, mode in enumerate(modes):
        if (
            mode == "train"
            and uniform_train_tasks
            and int(getattr(sampler, "n_major_tasks", 0)) > 0
            and int(getattr(sampler, "n_minor_tasks", 0)) > 0
        ):
            n_major = int(getattr(sampler, "n_major_tasks", 0))
            n_minor = int(getattr(sampler, "n_minor_tasks", 0))
            p_minor_sample = float(n_minor / (n_major + n_minor))
        else:
            p_minor_sample = p_minor_orig

        sampler.p_minor = p_minor_sample

        pool_bayes = DyckBayes(config, sampler, flag=False)
        agnostic_bayes = DyckBayes(config, sampler, flag=True)

        if use_dyck_positions:
            n_masks_eff = max(1, int(n_masks))
            n_samples_per_mask = (int(num_samples) + n_masks_eff - 1) // n_masks_eff
            n_dyck = int(getattr(sampler, "dyck_length", 0)) * 2
            if n_dyck <= 0 and getattr(sampler, "major_task_pool", None) is not None:
                n_dyck = int(sampler.major_task_pool.shape[1])
            if n_dyck <= 0:
                raise ValueError("Could not infer number of Dyck tokens.")

            sum_pool = torch.zeros(n_dyck, dtype=torch.float64)
            sum2_pool = torch.zeros(n_dyck, dtype=torch.float64)
            sum_agn = torch.zeros(n_dyck, dtype=torch.float64)
            sum2_agn = torch.zeros(n_dyck, dtype=torch.float64)
            count = torch.zeros(n_dyck, dtype=torch.float64)

            for _ in range(n_masks_eff):
                current_mask = sample_binary_mask(config).to(config.device)
                out = sampler.generate(
                    mode=mode,
                    num_samples=n_samples_per_mask,
                    epochs=1,
                    dyck_mask=current_mask.clone(),
                )
                samples, masks = _extract_samples_and_masks_from_generate_output(out)
                samples = samples.to(config.device)
                masks = masks.to(config.device)

                logits = model(samples)  # (B, T, V)
                p_model = torch.softmax(logits, dim=-1)
                p_pool = pool_bayes.pos_prob(samples)
                p_agnostic = agnostic_bayes.pos_prob(samples)

                t_cmp = min(p_model.size(1), p_pool.size(1), p_agnostic.size(1), masks.size(1))
                v_cmp = min(p_model.size(2), p_pool.size(2), p_agnostic.size(2))
                p_model = p_model[:, :t_cmp, :v_cmp]
                p_pool = p_pool[:, :t_cmp, :v_cmp]
                p_agnostic = p_agnostic[:, :t_cmp, :v_cmp]

                kl_pool = _kl_model_to_ref(p_model, p_pool)          # (B, Tcmp)
                kl_agnostic = _kl_model_to_ref(p_model, p_agnostic)  # (B, Tcmp)

                m_cmp = masks[:, :t_cmp].to(torch.bool)
                dyck_idx = m_cmp.to(torch.long).cumsum(dim=1) - 1
                sel_pool = kl_pool[m_cmp]
                sel_agn = kl_agnostic[m_cmp]
                sel_idx = dyck_idx[m_cmp]
                valid = (sel_idx >= 0) & (sel_idx < n_dyck)
                sel_idx = sel_idx[valid].detach().cpu()
                sel_pool = sel_pool[valid].detach().cpu().to(torch.float64)
                sel_agn = sel_agn[valid].detach().cpu().to(torch.float64)
                if sel_idx.numel() > 0:
                    one = torch.ones(sel_idx.numel(), dtype=torch.float64)
                    count.scatter_add_(0, sel_idx, one)
                    sum_pool.scatter_add_(0, sel_idx, sel_pool)
                    sum2_pool.scatter_add_(0, sel_idx, sel_pool * sel_pool)
                    sum_agn.scatter_add_(0, sel_idx, sel_agn)
                    sum2_agn.scatter_add_(0, sel_idx, sel_agn * sel_agn)

            pos_all = torch.arange(n_dyck, dtype=torch.long)
            mean_pool_full = torch.full((n_dyck,), float("nan"), dtype=torch.float64)
            std_pool_full = torch.full((n_dyck,), float("nan"), dtype=torch.float64)
            mean_agn_full = torch.full((n_dyck,), float("nan"), dtype=torch.float64)
            std_agn_full = torch.full((n_dyck,), float("nan"), dtype=torch.float64)
            valid = count > 0
            mean_pool_full[valid] = sum_pool[valid] / count[valid]
            mean_agn_full[valid] = sum_agn[valid] / count[valid]
            var_pool = torch.zeros_like(mean_pool_full)
            var_agn = torch.zeros_like(mean_agn_full)
            var_pool[valid] = (sum2_pool[valid] / count[valid]) - mean_pool_full[valid] ** 2
            var_agn[valid] = (sum2_agn[valid] / count[valid]) - mean_agn_full[valid] ** 2
            std_pool_full[valid] = torch.sqrt(torch.clamp(var_pool[valid], min=0.0))
            std_agn_full[valid] = torch.sqrt(torch.clamp(var_agn[valid], min=0.0))

            if positions is not None:
                pos_req = torch.as_tensor(list(positions), dtype=torch.long)
                keep = (pos_req >= 0) & (pos_req < n_dyck)
                pos_sel = pos_req[keep]
                if pos_sel.numel() == 0:
                    raise ValueError(
                        f"No valid Dyck positions selected. Requested={list(positions)}, valid range=[0, {n_dyck - 1}]"
                    )
            else:
                pos_sel = pos_all

            pos = pos_sel.numpy()
            mean_pool = mean_pool_full[pos_sel].numpy()
            std_pool = std_pool_full[pos_sel].numpy()
            mean_agnostic = mean_agn_full[pos_sel].numpy()
            std_agnostic = std_agn_full[pos_sel].numpy()
            kl_pool_out = None
            kl_agnostic_out = None
            counts_out = count[pos_sel].numpy()
        else:
            out = sampler.generate(mode=mode, num_samples=int(num_samples), epochs=1)
            samples = _extract_samples_from_generate_output(out).to(config.device)

            logits = model(samples)  # (B, T, V)
            p_model = torch.softmax(logits, dim=-1)
            p_pool = pool_bayes.pos_prob(samples)        # (B, T-1, V)
            p_agnostic = agnostic_bayes.pos_prob(samples)  # (B, T-1, V)
            t_cmp = min(p_model.size(1), p_pool.size(1), p_agnostic.size(1))
            v_cmp = min(p_model.size(2), p_pool.size(2), p_agnostic.size(2))
            p_model = p_model[:, :t_cmp, :v_cmp]
            p_pool = p_pool[:, :t_cmp, :v_cmp]
            p_agnostic = p_agnostic[:, :t_cmp, :v_cmp]

            kl_pool = _kl_model_to_ref(p_model, p_pool)          # (B, Tcmp)
            kl_agnostic = _kl_model_to_ref(p_model, p_agnostic)  # (B, Tcmp)

            pos_all = torch.arange(t_cmp, device=kl_pool.device)
            if positions is not None:
                pos_req = torch.as_tensor(list(positions), device=kl_pool.device, dtype=torch.long)
                valid = (pos_req >= 0) & (pos_req < t_cmp)
                pos_sel = pos_req[valid]
                if pos_sel.numel() == 0:
                    raise ValueError(
                        f"No valid positions selected. Requested={list(positions)}, valid range=[0, {t_cmp - 1}]"
                    )
                kl_pool = kl_pool.index_select(dim=1, index=pos_sel)
                kl_agnostic = kl_agnostic.index_select(dim=1, index=pos_sel)
                pos = pos_sel.detach().cpu().numpy()
            else:
                pos = pos_all.detach().cpu().numpy()

            mean_pool = kl_pool.mean(dim=0).detach().cpu().numpy()
            std_pool = kl_pool.std(dim=0).detach().cpu().numpy()
            mean_agnostic = kl_agnostic.mean(dim=0).detach().cpu().numpy()
            std_agnostic = kl_agnostic.std(dim=0).detach().cpu().numpy()
            kl_pool_out = kl_pool.detach().cpu()
            kl_agnostic_out = kl_agnostic.detach().cpu()
            counts_out = None

        ax = axes_flat[mi]
        ax.plot(pos, mean_pool, color="#1f77b4", lw=2.0, label="Pool-aware Dyck Bayes")
        pool_lo = np.maximum(mean_pool - std_pool, 0.0)
        pool_hi = mean_pool + std_pool
        ax.fill_between(pos, pool_lo, pool_hi, color="#1f77b4", alpha=0.2)

        ax.plot(pos, mean_agnostic, color="#d62728", lw=2.0, label="Pool-agnostic Dyck Bayes")
        agn_lo = np.maximum(mean_agnostic - std_agnostic, 0.0)
        agn_hi = mean_agnostic + std_agnostic
        ax.fill_between(pos, agn_lo, agn_hi, color="#d62728", alpha=0.2)

        ax.set_title(f"Mode: {mode}", fontsize=12)
        ax.set_xlabel("Dyck Position" if use_dyck_positions else "Position", fontsize=11)
        ax.set_ylabel("KL(model || baseline)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        results[mode] = {
            "positions": pos,
            "positions_all": pos_all.detach().cpu().numpy(),
            "p_minor_sampling_used": float(p_minor_sample),
            "p_minor_bayes_used": float(p_minor_orig),
            "kl_pool_mean": mean_pool,
            "kl_pool_std": std_pool,
            "kl_agnostic_mean": mean_agnostic,
            "kl_agnostic_std": std_agnostic,
            "kl_pool": kl_pool_out,
            "kl_agnostic": kl_agnostic_out,
            "counts_per_position": counts_out,
        }

    for j in range(len(modes), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    sampler.p_minor = p_minor_orig

    return {
        "fig": fig,
        "axes": axes,
        "results": results,
        "modes": modes,
        "positions_requested": None if positions is None else list(positions),
        "use_dyck_positions": bool(use_dyck_positions),
        "n_masks": int(n_masks),
        "step": int(step),
    }


@torch.no_grad()
def plot_kl_model_vs_two_bayes_dyck_across_k(
    k_values,
    mode: str = "ood",
    num_samples: int = 1024,
    positions=None,
    use_dyck_positions: bool = False,
    n_masks: int = 60,
    eps: float = 1e-12,
    add_std_band: bool = False,
    same_y_axis: bool = False,
    figsize: tuple = (14, 5),
    show: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Visualize KL(model || baseline) across multiple k values for Dyck.

    For each k:
      1. build exp_name = get_exp_name("dyck", k),
      2. run ``plot_kl_model_vs_two_bayes_dyck`` in one mode (show=False),
      3. collect mean/std KL-vs-position curves.

    Plots two panels:
      - left: pool-aware Dyck Bayes
      - right: major-aware + minor-agnostic Dyck Bayes
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from icl.utils.unified_interface import get_exp_name

    curves = {}
    for k in k_values:
        exp_name = get_exp_name("dyck", k)
        try:
            out = plot_kl_model_vs_two_bayes_dyck(
                exp_name=exp_name,
                modes=mode,
                num_samples=num_samples,
                positions=positions,
                use_dyck_positions=use_dyck_positions,
                n_masks=n_masks,
                eps=eps,
                show=False,
            )
            d = out["results"][mode]
            curves[k] = {
                "positions": np.asarray(d["positions"], dtype=float),
                "pool_mean": np.asarray(d["kl_pool_mean"], dtype=float),
                "pool_std": np.asarray(d["kl_pool_std"], dtype=float),
                "agnostic_mean": np.asarray(d["kl_agnostic_mean"], dtype=float),
                "agnostic_std": np.asarray(d["kl_agnostic_std"], dtype=float),
                "p_minor_bayes_used": float(d["p_minor_bayes_used"]),
            }
        except Exception as e:
            if verbose:
                print(f"[warn] k={k} failed: {e}")

    ks = sorted(curves.keys())
    if len(ks) == 0:
        raise RuntimeError("No k succeeded. Check exp_name availability/checkpoints.")

    if len(ks) <= 10:
        palette = list(plt.get_cmap("tab10").colors)
    elif len(ks) <= 20:
        palette = list(plt.get_cmap("tab20").colors)
    else:
        base = list(plt.get_cmap("tab20").colors)
        reps = (len(ks) + len(base) - 1) // len(base)
        palette = (base * reps)[: len(ks)]
    color_map = {k: palette[i] for i, k in enumerate(ks)}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, sharex=False, sharey=same_y_axis
    )

    for k in ks:
        c = color_map[k]
        d = curves[k]
        x = d["positions"]

        ax1.plot(x, d["pool_mean"], color=c, lw=2.0, label=f"k={k}")
        if add_std_band:
            lo = np.maximum(d["pool_mean"] - d["pool_std"], 0.0)
            hi = d["pool_mean"] + d["pool_std"]
            ax1.fill_between(x, lo, hi, color=c, alpha=0.15)

        ax2.plot(x, d["agnostic_mean"], color=c, lw=2.0, label=f"k={k}")
        if add_std_band:
            lo = np.maximum(d["agnostic_mean"] - d["agnostic_std"], 0.0)
            hi = d["agnostic_mean"] + d["agnostic_std"]
            ax2.fill_between(x, lo, hi, color=c, alpha=0.15)

    xticks = sorted({int(v) for k in ks for v in curves[k]["positions"]})
    if len(xticks) > 0:
        ax1.set_xticks(xticks)
        ax2.set_xticks(xticks)
        labels = [str(int(t)) for t in xticks]
        ax1.set_xticklabels(labels)
        ax2.set_xticklabels(labels)

    ax1.set_title(f"Pool-aware Dyck Bayes | mode={mode}", fontsize=12)
    ax1.set_xlabel("Dyck Position" if use_dyck_positions else "Position", fontsize=11)
    ax1.set_ylabel("KL(model || pool-aware)", fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.set_title(f"Major-aware, minor-agnostic | mode={mode}", fontsize=12)
    ax2.set_xlabel("Dyck Position" if use_dyck_positions else "Position", fontsize=11)
    ax2.set_ylabel("KL(model || hybrid)", fontsize=11)
    ax2.grid(True, alpha=0.3)

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=9, frameon=True)
    fig.tight_layout(rect=(0.0, 0.0, 0.9, 1.0))

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "fig": fig,
        "axes": (ax1, ax2),
        "mode": mode,
        "k_values_loaded": ks,
        "curves_by_k": curves,
        "positions_requested": None if positions is None else list(positions),
        "use_dyck_positions": bool(use_dyck_positions),
        "n_masks": int(n_masks),
    }
