"""Architecture-agnostic per-layer hidden-state extraction via forward hooks.

Works for Transformer, RNN, and LSTM -- any model whose ``self.layers`` is a
ModuleList. The captured quantity is the **full-layer output**: the residual
stream after layer ``l`` for the Transformer, and the cell output ``h_t`` for
recurrent models. That is the one extraction point with a clean analog across
all architectures, so it is what we use for cross-architecture comparison.

The only architecture difference handled here: ``nn.RNN`` / ``nn.LSTM`` return
an ``(output, state)`` tuple from their forward, whereas a Transformer block
returns a plain tensor. ``_layer_output`` normalizes both to the ``(B, T, D)``
activation.
"""

import torch
from torch import nn


def _layer_output(out):
    """Normalize a layer's forward output to its ``(B, T, D)`` activation tensor."""
    # nn.RNN / nn.LSTM return (output, hidden); a Transformer block returns a tensor.
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def extract_layer_hiddens(model, x, layers=None, state="hidden"):
    """Run ``model`` on ``x`` and capture each layer's per-timestep representation.

    Args:
        model: ``nn.Module`` exposing a ``.layers`` ModuleList.
        x: token ids of shape ``(B, T)``.
        layers: optional list of layer indices to capture (default: all layers).
        state: which representation to extract.
            ``"hidden"`` (default, all architectures) -- the layer output ``h_t``
            that feeds the next layer / readout, via forward hooks (fast path).
            ``"cell"`` (LSTM only) -- the per-timestep cell state ``c_t``, the
            additive memory, reconstructed with a stepped forward.

    Returns:
        Tensor of shape ``(n_layers, B, T, D)`` on the model's device, where
        ``n_layers == len(layers)`` and ``D`` is that model's hidden width.
    """
    if not hasattr(model, "layers"):
        raise AttributeError(
            f"{type(model).__name__} has no `.layers`; cannot extract per-layer hiddens."
        )
    layer_idxs = list(range(len(model.layers))) if layers is None else list(layers)

    if state == "cell":
        if not all(isinstance(model.layers[l], nn.LSTM) for l in layer_idxs):
            raise ValueError(
                "state='cell' is only defined for LSTM layers; this model's layers "
                f"are {type(model.layers[0]).__name__}."
            )
        return _lstm_cell_states(model, x, layer_idxs)
    if state != "hidden":
        raise ValueError(f"Unknown state={state!r}; expected 'hidden' or 'cell'.")

    cache = {}
    handles = []
    for l in layer_idxs:
        def make_hook(idx):
            def hook(module, inp, out):
                cache[idx] = _layer_output(out).detach()
            return hook
        handles.append(model.layers[l].register_forward_hook(make_hook(l)))

    was_training = model.training
    model.eval()
    try:
        model(x)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    return torch.stack([cache[l] for l in layer_idxs], dim=0)


@torch.no_grad()
def _lstm_cell_states(model, x, layer_idxs):
    """Per-timestep cell states ``c_t`` for selected LSTM layers via stepped forward.

    ``nn.LSTM`` only exposes the per-step ``h_t`` and the *final* ``c_n``, so we
    replay the model's forward one timestep at a time, carrying ``(h, c)`` and
    collecting ``c_t``. This is mathematically identical to the model's full-
    sequence forward (it is the same recurrence), but extraction-only -- training
    is untouched. Layer ``l``'s input is the post-dropout ``h`` sequence of layer
    ``l-1`` (``embed(x)`` for layer 0), exactly as in the model's forward.
    """
    was_training = model.training
    model.eval()
    try:
        inp = model.embed(x)  # (B, T, H)
        B, T, _ = inp.shape
        wanted = set(layer_idxs)
        collected = {}
        for l, layer in enumerate(model.layers):
            H = layer.hidden_size
            h = torch.zeros(1, B, H, device=inp.device, dtype=inp.dtype)
            c = torch.zeros(1, B, H, device=inp.device, dtype=inp.dtype)
            h_seq, c_seq = [], []
            for t in range(T):
                _, (h, c) = layer(inp[:, t:t + 1, :], (h, c))
                h_seq.append(h[0])  # (B, H)
                c_seq.append(c[0])
            h_out = torch.stack(h_seq, dim=1)  # (B, T, H)
            if l in wanted:
                collected[l] = torch.stack(c_seq, dim=1)  # (B, T, H)
            # next layer consumes this layer's (post-dropout) hidden output
            inp = model.dropout(h_out) if model.dropout is not None else h_out
    finally:
        if was_training:
            model.train()

    return torch.stack([collected[l] for l in layer_idxs], dim=0)
