"""Historical perturbation interventions for the Coin task.

Randomly perturbs historical hidden states to measure how much the model
relies on past representations for next-token prediction.
"""

import gc
from typing import Optional

import torch

import icl.utils.notebook_utils as nu
from icl.coin.coin_ood_analysis import get_new_sampler


def _historical_injection_test_next_token_coin(
    config,
    model: torch.nn.Module,
    sampler,
    *,
    layer_idx: int,
    p_idx: int,
    task: int,
    B: int = 64,
    mode: str = "testing",
    seed: int = 0,
    p: float = 0.1,
    rand_mode: str = "standard_normal",
    eps: float = 1e-12,
    hook_attr: str = "attn_block",
    return_logits: bool = False,
) -> dict:
    """
    Historical perturbation on **non-padded** sequences.

    For a non-padded sequence ``[tok0, tok1, ..., tok_{n-1}]``:
      - Current real token is at position ``p_idx``.
      - We keep position ``p_idx`` unchanged.
      - We randomly perturb positions ``[0, ..., p_idx - 1]`` with
        probability ``p``.
      - We evaluate next-token prediction at ``p_idx``
        against target ``batch[:, p_idx + 1]``.

    Parameters
    ----------
    config, model, sampler : standard
    layer_idx : int
    p_idx : int
        Position index (must be >= 1 so there is history to perturb).
    task : int
    B : int
    mode : str
    seed : int
    p : float
        Probability of perturbing each historical position.
    rand_mode : str
    eps : float
    hook_attr : str
    return_logits : bool

    Returns
    -------
    dict  (same keys as ``historical_injection_test_next_token``)
    """
    import torch.nn.functional as F

    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")

    device = torch.device(
        getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    model.eval().to(device)

    n_layers = len(model.layers)
    seq_len = int(sampler.seq_len)
    P = seq_len - 1

    if not (1 <= p_idx < P):
        raise ValueError(f"p_idx out of range: {p_idx} not in [1, {P - 1}]")
    if not (0 <= layer_idx < n_layers):
        raise ValueError(f"layer_idx out of range: {layer_idx} not in [0, {n_layers - 1}]")

    token_pos = p_idx
    historical_end = p_idx

    out_raw = sampler.generate(mode=mode, task=task, num_samples=B)
    batch_data = (out_raw[0] if isinstance(out_raw, (tuple, list)) else out_raw).to(device)

    L = batch_data.size(1)
    if token_pos + 1 >= L:
        raise ValueError(f"token_pos+1 out of bounds: token_pos={token_pos}, L={L}")

    def _get_next_token_slice(logits):
        logits_at_pos = logits[:, token_pos, :]
        target = batch_data[:, token_pos + 1]
        return logits_at_pos, target

    def _run_baseline():
        return model(batch_data) if torch.is_tensor(model(batch_data)) else model(batch_data)

    base_out = model(batch_data)
    base_logits = base_out if torch.is_tensor(base_out) else base_out[0]

    def _replace_historical(hid, cache):
        if hid.dim() != 3:
            raise RuntimeError(f"Expected 3D hidden, got shape={tuple(hid.shape)}")

        B_local = batch_data.size(0)

        if hid.size(0) == B_local:
            out2 = hid.clone()
            cache["layout"] = "BLD"

            mask = torch.zeros((B_local, historical_end), dtype=torch.bool, device=device)
            for pos in range(historical_end):
                g_pos = torch.Generator(device=device)
                g_pos.manual_seed(seed + pos)
                mask[:, pos] = torch.rand(B_local, generator=g_pos, device=device) < p

            cache["perturb_mask"] = mask.detach()
            cache["n_perturbed"] = mask.sum().item()

            for pos in range(historical_end):
                if mask[:, pos].any():
                    h_pos = out2[:, pos, :]
                    h_perturbed = h_pos.clone()
                    mask_rows = mask[:, pos]
                    h_to_perturb = h_pos[mask_rows]

                    g_perturb = torch.Generator(device=device)
                    g_perturb.manual_seed(seed + pos + 1000)

                    if rand_mode == "standard_normal":
                        r = torch.randn(h_to_perturb.shape, device=device,
                                        dtype=h_to_perturb.dtype, generator=g_perturb)
                    elif rand_mode == "match_std":
                        std = h_pos.std(unbiased=False) + eps
                        r = torch.randn(h_to_perturb.shape, device=device,
                                        dtype=h_to_perturb.dtype, generator=g_perturb) * std
                    elif rand_mode == "match_mean_std":
                        mean = h_pos.mean()
                        std = h_pos.std(unbiased=False) + eps
                        r = torch.randn(h_to_perturb.shape, device=device,
                                        dtype=h_to_perturb.dtype, generator=g_perturb) * std + mean
                    else:
                        raise ValueError(f"Unknown rand_mode={rand_mode!r}")

                    h_perturbed[mask_rows] = r
                    out2[:, pos, :] = h_perturbed

        elif hid.size(1) == B_local:
            out2 = hid.clone()
            cache["layout"] = "LBD"

            mask = torch.zeros((historical_end, B_local), dtype=torch.bool, device=device)
            for pos in range(historical_end):
                g_pos = torch.Generator(device=device)
                g_pos.manual_seed(seed + pos)
                mask[pos, :] = torch.rand(B_local, generator=g_pos, device=device) < p

            cache["perturb_mask"] = mask.detach()
            cache["n_perturbed"] = mask.sum().item()

            for pos in range(historical_end):
                if mask[pos, :].any():
                    h_pos = out2[pos, :, :]
                    h_perturbed = h_pos.clone()
                    mask_cols = mask[pos, :]
                    h_to_perturb = h_pos[mask_cols, :]

                    g_perturb = torch.Generator(device=device)
                    g_perturb.manual_seed(seed + pos + 1000)

                    if rand_mode == "standard_normal":
                        r = torch.randn(h_to_perturb.shape, device=device,
                                        dtype=h_to_perturb.dtype, generator=g_perturb)
                    elif rand_mode == "match_std":
                        std = h_pos.std(unbiased=False) + eps
                        r = torch.randn(h_to_perturb.shape, device=device,
                                        dtype=h_to_perturb.dtype, generator=g_perturb) * std
                    elif rand_mode == "match_mean_std":
                        mean = h_pos.mean()
                        std = h_pos.std(unbiased=False) + eps
                        r = torch.randn(h_to_perturb.shape, device=device,
                                        dtype=h_to_perturb.dtype, generator=g_perturb) * std + mean
                    else:
                        raise ValueError(f"Unknown rand_mode={rand_mode!r}")

                    h_perturbed[mask_cols, :] = r
                    out2[pos, :, :] = h_perturbed

        else:
            raise RuntimeError(f"Cannot infer layout. hid={tuple(hid.shape)}, B={B_local}")

        return out2

    cache = {}
    layer = model.layers[layer_idx]
    mod = getattr(layer, hook_attr)

    def hook_fn(module, inp, out_tensor):
        if torch.is_tensor(out_tensor):
            return _replace_historical(out_tensor, cache)
        if isinstance(out_tensor, tuple) and len(out_tensor) > 0 and torch.is_tensor(out_tensor[0]):
            new0 = _replace_historical(out_tensor[0], cache)
            return (new0,) + out_tensor[1:]
        raise RuntimeError(f"Unsupported hook output type: {type(out_tensor)}")

    handle = mod.register_forward_hook(hook_fn)
    try:
        inj_out = model(batch_data)
        inj_logits = inj_out if torch.is_tensor(inj_out) else inj_out[0]
    finally:
        handle.remove()

    base_slice, target = _get_next_token_slice(base_logits)
    inj_slice, _ = _get_next_token_slice(inj_logits)

    base_logp = F.log_softmax(base_slice, dim=-1)
    inj_logp = F.log_softmax(inj_slice, dim=-1)

    base_tlp = base_logp.gather(1, target[:, None]).squeeze(1)
    inj_tlp = inj_logp.gather(1, target[:, None]).squeeze(1)
    delta_tlp = inj_tlp - base_tlp

    base_top1 = base_slice.argmax(dim=-1)
    inj_top1 = inj_slice.argmax(dim=-1)
    base_acc = (base_top1 == target).float().mean().item()
    inj_acc = (inj_top1 == target).float().mean().item()
    top1_flip_rate = (base_top1 != inj_top1).float().mean().item()

    base_p_dist = base_logp.exp()
    kl = (base_p_dist * (base_logp - inj_logp)).sum(dim=-1)

    results = {
        "layer_idx": int(layer_idx),
        "p_idx": int(p_idx),
        "token_pos": int(token_pos),
        "historical_end": int(historical_end),
        "task": int(task),
        "B": int(B),
        "mode": mode,
        "hook_attr": hook_attr,
        "rand_mode": rand_mode,
        "seed": int(seed),
        "p": float(p),
        "n_perturbed": int(cache.get("n_perturbed", 0)),
        "hidden_layout": cache.get("layout", None),
        "base_acc": float(base_acc),
        "inj_acc": float(inj_acc),
        "top1_flip_rate": float(top1_flip_rate),
        "delta_target_logprob_mean": float(delta_tlp.mean().item()),
        "delta_target_logprob_std": float(delta_tlp.std(unbiased=False).item()),
        "kl_mean": float(kl.mean().item()),
        "kl_std": float(kl.std(unbiased=False).item()),
    }

    if return_logits:
        results.update({
            "base_logits_at_pos": base_slice.detach(),
            "inj_logits_at_pos": inj_slice.detach(),
            "target_tokens": target.detach(),
            "perturb_mask": cache.get("perturb_mask", None),
            "delta_target_logprob": delta_tlp.detach(),
            "kl_per_example": kl.detach(),
        })

    return results


def historical_injection_coin(
    exp_name: str,
    n_minor: int = 64,
    n_ood: int = 0,
    B: int = 64,
    step: Optional[int] = None,
    layer_idx: int = 5,
    hook_attr: str = "attn_block",
    agg: str = "mean",
    mode: str = "testing",
    seed: int = 0,
    p: float = 0.1,
    rand_mode: str = "standard_normal",
    stride: int = 10,
    start: Optional[int] = None,
    end: Optional[int] = None,
    verbose: bool = False,
) -> tuple:
    """
    Historical perturbation analysis for Coin task on non-padded sequences.

    For each evaluated position ``p_idx``, randomly perturbs
    historical hidden states at positions ``[0, ..., p_idx - 1]``
    with probability ``p``, keeping the current real token at ``p_idx``
    unchanged, and measures the effect on next-token prediction at
    ``p_idx`` → ``p_idx + 1``.

    Non-padded counterpart of ``historical_injection_coin``.

    Parameters
    ----------
    exp_name : str
    n_minor : int
        Capped at ``sampler.n_minor_tasks``.
    n_ood : int
    B : int
    step : int, optional
    layer_idx : int
    hook_attr : str
    agg : str
        ``"mean"`` or ``"median"`` across tasks.
    mode : str
    seed : int
    p : float
        Perturbation probability per historical position.
    rand_mode : str
    stride : int, default 10
        Evaluate every ``stride``-th position within ``[start, end]``.
        Use ``stride=1`` for every position.
    start : int, optional
        First ``p_idx`` to evaluate (inclusive, >= 1).
        ``None`` → 1.
    end : int, optional
        Last ``p_idx`` to evaluate (inclusive, <= ``seq_len - 2``).
        ``None`` → ``seq_len - 2``.
    verbose : bool

    Returns
    -------
    out : list[float]
        Δ log-prob for each evaluated position.
    positions : list[int]
        The ``p_idx`` values that were evaluated.
    info : dict
        ``{'task_ids', 'per_task', 'p', 'stride', 'start', 'end'}``.
    """
    _, _sampler_orig, config = nu.load_everything("coin", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True
    )

    sampler, _ = get_new_sampler(exp_name, n_minor, n_ood)

    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    task_ids = list(range(n_tasks))
    P = int(sampler.seq_len) - 1

    if start is None:
        start = 1
    if end is None:
        end = P - 1
    start = max(1, start)
    end = min(P - 1, end)

    eval_positions = list(range(start, end + 1, stride))
    if eval_positions[-1] != end:
        eval_positions.append(end)

    out = []
    per_task = []

    for i, p_idx in enumerate(eval_positions):
        vals = []
        for tid in task_ids:
            r = _historical_injection_test_next_token_coin(
                config=config,
                model=model,
                sampler=sampler,
                layer_idx=layer_idx,
                p_idx=p_idx,
                task=tid,
                B=B,
                mode=mode,
                seed=seed,
                p=p,
                rand_mode=rand_mode,
                hook_attr=hook_attr,
            )
            vals.append(r["delta_target_logprob_mean"])

        vals_t = torch.tensor(vals, dtype=torch.float32)
        if agg == "mean":
            agg_val = float(vals_t.mean().item())
        elif agg == "median":
            agg_val = float(vals_t.median().item())
        else:
            raise ValueError(f"Unknown agg={agg!r}")

        out.append(agg_val)
        per_task.append(vals)

        if verbose:
            print(
                f"[coin-nonpadded] p_idx={p_idx:03d}/{P - 1:03d} "
                f"({i + 1}/{len(eval_positions)}) "
                f"Δlogp({agg} over {n_tasks} tasks, p={p})={agg_val:+.4f}"
            )

    info = {
        "task_ids": task_ids,
        "per_task": per_task,
        "p": float(p),
        "stride": stride,
        "start": start,
        "end": end,
    }
    return out, eval_positions, info


def plot_historical_injection_coin(
    exp_name: str,
    n_minor: int = 64,
    n_ood: int = 0,
    B: int = 64,
    step: Optional[int] = None,
    layer_idx=5,
    hook_attr: str = "attn_block",
    agg: str = "mean",
    mode: str = "testing",
    seed: int = 0,
    p: float = 0.1,
    rand_mode: str = "standard_normal",
    stride: int = 10,
    start: Optional[int] = None,
    end: Optional[int] = None,
    verbose: bool = False,
    figsize: tuple = (10, 5),
    log_x: bool = False,
    show: bool = True,
    title: Optional[str] = None,
) -> dict:
    """
    Run historical perturbation analysis and plot the result.

    Convenience wrapper that calls ``historical_injection_coin``
    and produces a plot of Δ log P(next token) vs position.

    ``layer_idx`` can be a single ``int`` **or a list of ints**.
    When a list is given, each layer is perturbed independently (one at a
    time) and the results are overlaid on the same plot.

    Parameters
    ----------
    exp_name : str
    n_minor : int
    n_ood : int
    B : int
    step : int, optional
    layer_idx : int or list of int
        Layer(s) to perturb.  Each layer is perturbed separately.
    hook_attr : str
    agg : str
    mode : str
    seed : int
    p : float
    rand_mode : str
    stride : int, default 10
        Evaluate every ``stride``-th position.  Use ``stride=1`` for all.
    start : int, optional
        First position (inclusive, >= 1).  ``None`` → 1.
    end : int, optional
        Last position (inclusive).  ``None`` → ``seq_len - 2``.
    verbose : bool
    figsize : tuple
    log_x : bool
    show : bool
    title : str, optional

    Returns
    -------
    dict
        ``{'out_by_layer', 'info_by_layer', 'layers', 'positions',
        'fig', 'ax'}``.
    """
    import matplotlib.pyplot as plt

    if isinstance(layer_idx, int):
        layers = [layer_idx]
    else:
        layers = list(layer_idx)

    out_by_layer = {}
    info_by_layer = {}
    positions = None

    for li in layers:
        if verbose:
            print(f"=== Perturbing layer {li} ===")

        out, pos, info = historical_injection_coin(
            exp_name=exp_name,
            n_minor=n_minor,
            n_ood=n_ood,
            B=B,
            step=step,
            layer_idx=li,
            hook_attr=hook_attr,
            agg=agg,
            mode=mode,
            seed=seed,
            p=p,
            rand_mode=rand_mode,
            stride=stride,
            start=start,
            end=end,
            verbose=verbose,
        )
        out_by_layer[li] = out
        info_by_layer[li] = info
        if positions is None:
            positions = pos

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.tab10

    for idx, li in enumerate(layers):
        color = cmap(idx % 10)
        ax.plot(
            positions, out_by_layer[li], 'o-',
            linewidth=2, markersize=7, color=color,
            label=f'Layer {li}',
        )

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Position', fontsize=18)
    ax.set_ylabel('Δ log P(next token)', fontsize=18)
    if log_x:
        ax.set_xscale('log')
    ax.tick_params(labelsize=16)

    if title:
        ax.set_title(title, fontsize=18)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close()

    return {
        'out_by_layer': out_by_layer,
        'info_by_layer': info_by_layer,
        'layers': layers,
        'positions': positions,
        'fig': fig,
        'ax': ax,
    }
