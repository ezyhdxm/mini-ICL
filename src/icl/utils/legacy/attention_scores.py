import torch
import os
from typing import List, Dict, Optional, Sequence, Union
import numpy as np
import torch
import plotly.graph_objects as go

from icl.utils.train_utils import get_attn_base
from icl.utils.unified_interface import get_exp_name
from icl.utils.unified_path_finder import _get_exp_dir
import icl.utils.notebook_utils as nu

def stack_attn_maps(attn_maps: dict) -> torch.Tensor:
    # Ensure layers are in order 0..n_layers-1
    layers = sorted(attn_maps.keys())
    attn = torch.stack([attn_maps[l] for l in layers], dim=1)  # (B, n_layers, H, T, T)
    return attn


def _ensure_blhll(attn: torch.Tensor, B: int, L: int) -> torch.Tensor:
    """
    Normalize attention to shape (B, n_layers, n_heads, L, L).

    Accepts:
      - (n_layers, n_heads, L, L)         -> broadcast across batch
      - (B, n_layers, n_heads, L, L)      -> batched
    """
    if attn.dim() == 4:
        n_layers, n_heads, La, Lb = attn.shape
        assert La == L and Lb == L
        return attn.unsqueeze(0).expand(B, -1, -1, -1, -1)
    elif attn.dim() == 5:
        Bb, n_layers, n_heads, La, Lb = attn.shape
        assert Bb == B and La == L and Lb == L
        return attn
    else:
        raise ValueError("attn must be (layers, heads, L, L) or (B, layers, heads, L, L)")


def previous_token_head_score_padded_queries(
    tokens: torch.Tensor,
    attn: torch.Tensor,
) -> torch.Tensor:
    """
    Generalized previous-token head score.

    Assumptions
    -----------
    - real tokens at positions 0,2,4,...
    - padded tokens at positions 1,3,5,...
    - queries ONLY on padded tokens u_k = 2k+1

    For each k >= 1:
        score_k = A[u_k, t_{k-1}] + A[u_k, u_{k-1}]

    Final score = average over all valid k.

    Args
    ----
    tokens : (B, L) tensor
        Only used for shape/device.
    attn : (layers, heads, L, L) or (B, layers, heads, L, L)
        Attention weights (post-softmax).

    Returns
    -------
    prev_score : (B, layers, heads)
    """
    assert tokens.dim() == 2
    B, L = tokens.shape
    device = tokens.device

    attn_b = _ensure_blhll(attn, B, L).to(device=device)
    n_layers, n_heads = attn_b.shape[1], attn_b.shape[2]

    # fixed alternating layout
    real_pos = torch.arange(0, L, 2, device=device)
    pad_pos  = torch.arange(1, L, 2, device=device)

    # only slots with a pad are valid queries
    K = min(real_pos.numel(), pad_pos.numel())
    if K < 2:
        return torch.zeros(B, n_layers, n_heads, device=device, dtype=attn_b.dtype)

    real_pos = real_pos[:K]   # t_0 ... t_{K-1}
    pad_pos  = pad_pos[:K]    # u_0 ... u_{K-1}

    # queries u_k for k = 1..K-1
    q_idx   = pad_pos[1:]       # (Q,)
    prev_t  = real_pos[:-1]     # (Q,)
    prev_u  = pad_pos[:-1]      # (Q,)
    Q = q_idx.numel()

    # flatten layers * heads
    LH = n_layers * n_heads
    attn_f = attn_b.reshape(B, LH, L, L)

    # attention rows for all queries
    rows = attn_f[:, :, q_idx, :]        # (B, LH, Q, L)

    # gather {t_{k-1}, u_{k-1}}
    idx = torch.stack([prev_t, prev_u], dim=-1) \
             .view(1, 1, Q, 2) \
             .expand(B, LH, Q, 2)

    gathered = torch.gather(rows, dim=-1, index=idx)  # (B, LH, Q, 2)

    # sum over targets, mean over queries
    prev_score_f = gathered.sum(dim=-1).mean(dim=-1)  # (B, LH)

    return prev_score_f.view(B, n_layers, n_heads).mean(dim=0)  # (n_layers, n_heads)





def successor_token_or_pad_induction_score(
    tokens: torch.Tensor,
    attn: torch.Tensor,
) -> torch.Tensor:
    """
    Successor induction score (union, no double count), querying ONLY on padded tokens.

    Fixed structure (starts at 0):
      t_k = 2k   (real token positions)
      u_k = 2k+1 (pad positions; queries)

    For each slot k (query u_k), for each previous match m<k with tokens[t_m]==tokens[t_k],
    add successor targets (if m+1 exists):
        { t_{m+1}, u_{m+1} }.
    Take UNION across all such m, then sum attention mass on this union set.

    Average over k where the union set is non-empty (per sample).

    Args
    ----
    tokens: (B, L)
    attn:   (layers, heads, L, L) or (B, layers, heads, L, L)

    Returns
    -------
    score: (B, layers, heads) in [0,1]
    """
    if tokens.dim() != 2:
        raise ValueError("tokens must be (B, L)")
    B, L = tokens.shape
    device = tokens.device

    attn_b = _ensure_blhll(attn, B, L).to(device=device)
    n_layers, n_heads = attn_b.shape[1], attn_b.shape[2]

    # positions for alternating pattern starting at 0
    real_pos = torch.arange(0, L, 2, device=device)  # t_k
    pad_pos  = torch.arange(1, L, 2, device=device)  # u_k

    # only slots with pads are valid queries
    K = min(real_pos.numel(), pad_pos.numel())
    if K < 2:
        return torch.zeros(B, n_layers, n_heads, device=device, dtype=attn_b.dtype)

    real_pos = real_pos[:K]
    pad_pos  = pad_pos[:K]

    # real token values per slot
    v = tokens.index_select(dim=1, index=real_pos)  # (B, K)

    # flatten layers*heads
    LH = n_layers * n_heads
    attn_f = attn_b.reshape(B, LH, L, L)  # (B, LH, L, L)

    sum_mass = torch.zeros(B, LH, device=device, dtype=attn_f.dtype)
    cnt = torch.zeros(B, device=device, dtype=torch.long)

    # loop over current slot k (query at u_k)
    for k in range(1, K):
        # matches to all previous m<k: (B, k)
        matches = (v[:, :k] == v[:, k:k+1])  # (B, k)

        # successor slot indices s = m+1 ∈ {1,2,...,k}
        succ_slots = torch.arange(1, k + 1, device=device)  # (k,)

        # Build union key mask over key positions (B, L)
        key_mask = torch.zeros(B, L, device=device, dtype=torch.bool)

        # For each s, whether m=s-1 matched is matches[:, s-1]
        matches_succ = matches  # (B, k), aligned with s-1 in 0..k-1

        # Add successor real token t_{m+1} = t_s and successor pad u_{m+1} = u_s
        key_mask[:, real_pos[succ_slots]] |= matches_succ
        key_mask[:, pad_pos[succ_slots]]  |= matches_succ

        has_target = key_mask.any(dim=1)  # (B,)
        if not has_target.any():
            continue

        q = int(pad_pos[k].item())           # query u_k
        rows = attn_f[:, :, q, :]            # (B, LH, L)
        mass = (rows * key_mask.to(rows.dtype).unsqueeze(1)).sum(dim=-1)  # (B, LH)

        sum_mass[has_target] += mass[has_target]
        cnt[has_target] += 1

    out = torch.zeros_like(sum_mass)
    nz = cnt > 0
    if nz.any():
        out[nz] = sum_mass[nz] / cnt[nz].to(out.dtype).unsqueeze(1)

    return out.view(B, n_layers, n_heads).mean(dim=0)  # (n_layers, n_heads)

def real_previous_token_head_score(
    tokens: torch.Tensor,
    attn: torch.Tensor,
) -> torch.Tensor:
    """
    Real-PTH: query on REAL tokens t_k=2k, credit attending to previous REAL token t_{k-1}=2(k-1).

    Assumes pattern starts at 0:
      real positions: 0,2,4,...
      pad  positions: 1,3,5,...

    Score per k>=1:
      A[t_k, t_{k-1}]
    Average over k.

    Args
    ----
    tokens: (B, L)  (used for shape/device)
    attn:   (layers, heads, L, L) or (B, layers, heads, L, L)

    Returns
    -------
    score: (B, layers, heads) in [0,1]
    """
    if tokens.dim() != 2:
        raise ValueError("tokens must be (B, L)")
    B, L = tokens.shape
    device = tokens.device

    attn_b = _ensure_blhll(attn, B, L).to(device=device)
    n_layers, n_heads = attn_b.shape[1], attn_b.shape[2]

    real_pos = torch.arange(0, L, 2, device=device)  # t_k
    K = real_pos.numel()
    if K < 2:
        return torch.zeros(B, n_layers, n_heads, device=device, dtype=attn_b.dtype)

    # queries are t_k for k=1..K-1, targets are t_{k-1}
    q_idx = real_pos[1:]    # (Q,)
    k_idx = real_pos[:-1]   # (Q,)
    Q = q_idx.numel()

    LH = n_layers * n_heads
    attn_f = attn_b.reshape(B, LH, L, L)           # (B, LH, L, L)
    rows = attn_f[:, :, q_idx, :]                  # (B, LH, Q, L)

    # gather A[t_k, t_{k-1}] for each k
    idx = k_idx.view(1, 1, Q, 1).expand(B, LH, Q, 1)   # (B, LH, Q, 1)
    gathered = torch.gather(rows, dim=-1, index=idx).squeeze(-1)  # (B, LH, Q)

    score_f = gathered.mean(dim=-1)                 # (B, LH)
    return score_f.view(B, n_layers, n_heads).mean(dim=0)  # (n_layers, n_heads)


def _get_score_path(
    config,
    exp_name, 
    step,
    mode="ood",
):
    exp_dir = _get_exp_dir(config, exp_name)
    if mode != "ood":
        exp_name += f"_{mode}"
    cache_paths = {}
    cache_paths["pth"] = os.path.join(exp_dir, f"pth_scores_{exp_name}_step{step}.pt")
    cache_paths["ih"] = os.path.join(exp_dir, f"ih_scores_{exp_name}_step{step}.pt")
    cache_paths["real_pth"] = os.path.join(exp_dir, f"real_pth_scores_{exp_name}_step{step}.pt")
    return cache_paths

def get_attention_score_at_step(
    k,
    step=None,
    mode="ood",
    force_recompute: bool = False,
):
    exp_name = get_exp_name("latent", k)
    _, sampler, config = nu.load_everything("latent", exp_name)
    if step is None:
        step = config.training.num_epochs
    model, actual_step = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    cache_paths = _get_score_path(config, exp_name, actual_step, mode)
    if (os.path.exists(cache_paths["ih"]) 
        and os.path.exists(cache_paths["pth"]) 
        and os.path.exists(cache_paths["real_pth"]) 
        and not force_recompute):
        ih_scores = torch.load(cache_paths["ih"], weights_only=True)
        pth_scores = torch.load(cache_paths["pth"], weights_only=True)
        real_pth_scores = torch.load(cache_paths["real_pth"], weights_only=True)
        return ih_scores, pth_scores, real_pth_scores

    batch, *_ = sampler.generate(mode=mode)
    model.eval()
    attn_maps = get_attn_base(model, batch)
    attn = stack_attn_maps(attn_maps)  # (B, n_layers, H, T, T)
    induction_score = successor_token_or_pad_induction_score(batch, attn)
    prev_token_score = previous_token_head_score_padded_queries(batch, attn)
    real_prev_token_score = real_previous_token_head_score(batch, attn)
    torch.save(induction_score, cache_paths["ih"])
    torch.save(prev_token_score, cache_paths["pth"])
    torch.save(real_prev_token_score, cache_paths["real_pth"])
    return induction_score, prev_token_score, real_prev_token_score












def get_attn_map(k, step, task_name="latent", mode="ood", vocab_size=8):
    exp_name = get_exp_name(task_name, k, vocab_size)
    _, sampler, config = nu.load_everything(task_name, exp_name)
    if step is None:
        step = config.training.num_epochs
    model, actual_step = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    batch, *_ = sampler.generate(mode=mode, num_samples=1)
    model.eval()
    attn_maps = get_attn_base(model, batch)
    return attn_maps, batch

























