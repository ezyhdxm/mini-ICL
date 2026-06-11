"""Adapter for the official `mamba-ssm` Mamba block.

Why this exists: the official `Mamba.forward` returns only the layer output
(B, L, D) and its fused CUDA kernel hides per-timestep internals. But the LAYER
OUTPUT (the residual stream h_t) is exactly what our generic hook extractor reads
and what ~all analyses use (variance, task-vector R2, beta/alpha, KL). So wrapping
the official block in our `self.layers` + (B,T,D) contract makes it plug into the
whole pipeline unchanged, while training uses the fast fused kernel.

The internal d_state SSM memory (the LSTM-cell analog) is NOT exposed by the fused
forward; `ssm_state_sequence()` recovers it via the official `step()` path (slow,
for a deeper probe only).

Requirements: `pip install mamba-ssm causal-conv1d` and a CUDA GPU. The kernels do
not build/run on CPU, so `mamba_ssm` is imported lazily here — the CPU-only env is
unaffected unless arch='mamba_official' is actually requested.

Param-matched to the Transformer via the same formula as the pure-PyTorch Mamba
(the block has the same parameterization: in_proj, conv1d, x_proj, dt_proj, A, D,
out_proj). The match is approximate (the official dt_rank uses ceil vs our floor);
verify/adjust on first GPU run.

NOTE: untested without a GPU. Verify on the box with the plan in
docs/LAMBDA_RUNBOOK or the module docstring test sketch below.
"""

import torch
from torch import nn

from .mamba import matched_d_model  # param-count match reused (no mamba_ssm import)


def _make_official_mamba(d_model, d_state=16, d_conv=4, expand=2):
    """Lazily construct an official mamba_ssm.Mamba block with a clear error if missing."""
    try:
        from mamba_ssm import Mamba
    except Exception as e:  # ImportError, or CUDA-load failure on CPU
        raise ImportError(
            "arch='mamba_official' needs the official package: "
            "`pip install mamba-ssm causal-conv1d` on a CUDA GPU. "
            f"(import failed: {e})"
        )
    return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


class _OfficialResidualBlock(nn.Module):
    """Pre-norm residual wrapper so the layer output is the residual stream (B,T,D)."""

    def __init__(self, d_model, **kw):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = _make_official_mamba(d_model, **kw)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class MambaOfficialModel(nn.Module):
    """Official-kernel Mamba with the same contract as the other models.

    self.layers is a ModuleList whose per-block output is (B, T, D); the generic
    forward-hook extractor therefore captures h_t exactly as for transformer/rnn/
    lstm/mamba. arch='mamba_official'.
    """

    def __init__(self, config, d_model=None):
        super().__init__()
        if d_model is None:
            d_model = matched_d_model(config)
        self.d_model = self.hidden_size = d_model
        self.embed = nn.Embedding(config.vocab_size, d_model)
        self.layers = nn.ModuleList(
            [_OfficialResidualBlock(d_model) for _ in range(config.model.num_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, config.vocab_size)
        self.to(config.device)

    def forward(self, x, **kwargs):
        # **kwargs absorbs Transformer kv_cache args so callers stay arch-agnostic.
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)              # (B, T, D) residual stream; captured by the hook extractor
        return self.output_layer(self.norm_f(h))

    @torch.no_grad()
    def ssm_state_sequence(self, x, layer_idx):
        """Per-timestep internal SSM state for one layer via the official step() path.

        Returns (B, T, d_inner, d_state). Slow (sequential step()); use only for the
        deeper internal-memory probe, not the standard h_t analyses. The standard
        analyses should use extract_layer_hiddens (the layer-output residual stream).
        """
        self.eval()
        B, T = x.shape
        # Run the stack up to (but not through) the target layer to get its input.
        h = self.embed(x)
        for i in range(layer_idx):
            h = self.layers[i](h)
        blk = self.layers[layer_idx]
        mixer = blk.mixer
        normed = blk.norm(h)                          # input the mixer sees
        conv_state, ssm_state = mixer.allocate_inference_cache(B, T)
        states = []
        for t in range(T):
            # step() consumes one token (B, D) and updates conv_state, ssm_state in place.
            mixer.step(normed[:, t], conv_state, ssm_state)
            states.append(ssm_state.detach().clone())  # (B, d_inner, d_state)
        return torch.stack(states, dim=1)              # (B, T, d_inner, d_state)
