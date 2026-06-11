"""Minimal pure-PyTorch Mamba (selective SSM / S6) for analysis.

Deliberately hand-written (not mamba-ssm): the official fused CUDA kernel hides
the per-timestep SSM state, but our analyses read internal states. Here every
intermediate is plain PyTorch, so:

  - each layer's output (B, T, D) is the residual stream -> the generic
    extractor (forward hook on self.layers) captures h_t exactly as for the
    Transformer/RNN/LSTM, so state="hidden" works unchanged;
  - the d_state recurrent memory is available from a stepped scan (analogous to
    the LSTM cell state) if we later want to probe the SSM memory directly.

Same contract as the other models: token ids (B, T) -> logits (B, T, vocab_size).
Sized (via matched_d_model) to the Transformer's parameter count, like RNN/LSTM.

The recurrence is the standard Mamba-1 selective scan with the common simplified
ZOH discretization (A_bar = exp(dt*A), B_bar = dt*B), run sequentially.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _dt_rank(d_model: int) -> int:
    return max(1, d_model // 16)


class MambaBlock(nn.Module):
    """Selective SSM block: in_proj -> causal conv -> selective scan -> gate -> out_proj."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = _dt_rank(d_model)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is a learned negative diagonal (per d_inner channel, per state), via log.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):  # x: (B, T, D)
        B, T, _ = x.shape
        x_and_z = self.in_proj(x)                      # (B, T, 2*d_inner)
        u, z = x_and_z.chunk(2, dim=-1)                # each (B, T, d_inner)

        # Causal depthwise conv (pad left, trim right to T).
        u = self.conv1d(u.transpose(1, 2))[..., :T].transpose(1, 2)
        u = F.silu(u)                                   # (B, T, d_inner)

        # Input-dependent (selective) SSM parameters.
        dbl = self.x_proj(u)                            # (B, T, dt_rank + 2*d_state)
        dt, Bm, Cm = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))               # (B, T, d_inner)
        A = -torch.exp(self.A_log)                      # (d_inner, d_state)

        y = self._selective_scan(u, dt, A, Bm, Cm)      # (B, T, d_inner)
        y = y * F.silu(z)                               # gated output
        return self.out_proj(y)                         # (B, T, D)

    def _selective_scan(self, u, dt, A, Bm, Cm):
        """Sequential selective scan. u,dt:(B,T,d_inner); A:(d_inner,d_state); Bm,Cm:(B,T,d_state)."""
        Bsz, T, d_inner = u.shape
        A_bar = torch.exp(dt.unsqueeze(-1) * A)                 # (B, T, d_inner, d_state)
        Bu = (dt * u).unsqueeze(-1) * Bm.unsqueeze(2)           # (B, T, d_inner, d_state)
        h = torch.zeros(Bsz, d_inner, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(T):
            h = A_bar[:, t] * h + Bu[:, t]                      # (B, d_inner, d_state)
            ys.append(torch.einsum("bdn,bn->bd", h, Cm[:, t]))  # (B, d_inner)
        y = torch.stack(ys, dim=1)                              # (B, T, d_inner)
        return y + u * self.D                                   # skip term


class MambaResidualBlock(nn.Module):
    """Pre-norm residual wrapper so each layer output is the residual stream (B,T,D)."""

    def __init__(self, d_model: int, **kw):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = MambaBlock(d_model, **kw)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


def _mamba_param_count(d_model, n_layers, vocab, d_state=16, d_conv=4, expand=2):
    D, I, N, K = d_model, expand * d_model, d_state, d_conv
    R = _dt_rank(d_model)
    block = (2 * D * I) + (I * K + I) + (I * (R + 2 * N)) + (R * I + I) + (I * N) + I + (I * D)
    block += 2 * D  # pre-norm LayerNorm (weight + bias)
    return vocab * D + n_layers * block + 2 * D + (D * vocab + vocab)  # embed + layers + norm_f + out


def matched_d_model(config) -> int:
    """d_model giving the closest parameter count to the Transformer from the same config."""
    from .base_models import Transformer

    target = sum(p.numel() for p in Transformer(config).parameters())
    n_layers, vocab = config.model.num_layers, config.vocab_size
    lo, hi = 8, 4096
    while lo < hi:
        mid = (lo + hi) // 2
        if _mamba_param_count(mid, n_layers, vocab) < target:
            lo = mid + 1
        else:
            hi = mid
    cand = [d for d in (lo - 1, lo) if d >= 8]
    return min(cand, key=lambda d: abs(_mamba_param_count(d, n_layers, vocab) - target))


class MambaModel(nn.Module):
    def __init__(self, config, d_model=None):
        super().__init__()
        if d_model is None:
            d_model = matched_d_model(config)
        self.d_model = self.hidden_size = d_model  # hidden_size: matches the RNN/LSTM convention
        self.embed = nn.Embedding(config.vocab_size, d_model)
        self.layers = nn.ModuleList(
            [MambaResidualBlock(d_model) for _ in range(config.model.num_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, config.vocab_size)
        self.to(config.device)

    def forward(self, x, **kwargs):
        # **kwargs absorbs Transformer kv_cache args so callers stay arch-agnostic.
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)                  # (B, T, D) residual stream; unidirectional => causal
        return self.output_layer(self.norm_f(h))
