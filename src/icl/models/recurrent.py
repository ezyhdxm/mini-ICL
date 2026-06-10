"""Recurrent sequence models (vanilla RNN and LSTM).

Same contract as :class:`icl.models.base_models.Transformer`:
token ids ``(B, T)`` -> next-token logits ``(B, T, vocab_size)``.

Design choices:
- We stack ``N`` single-layer cells in a ``self.layers`` ModuleList (mirroring
  the Transformer's ``self.layers``) so one per-layer hidden-state hook works on
  every architecture, and each layer's output ``(B, T, H)`` is the recurrent
  analogue of the residual stream. No residual connection between layers.
- Hidden width ``H`` defaults to whatever makes the model's parameter count
  closest to the Transformer built from the same config, so architectures are
  compared at matched capacity. Override with ``config.model.rnn_hidden``.
"""

from torch import nn


def _recurrent_param_count(vocab: int, hidden: int, n_layers: int, gates: int) -> int:
    """Parameter count of a stacked single-layer recurrent model of width ``hidden``.

    ``gates`` is the per-cell weight multiplier: 1 for vanilla RNN, 4 for LSTM.
    Embedding (vocab*H) + output (H*vocab + vocab) + per layer the cell's
    weight_ih (H*H) + weight_hh (H*H) + bias_ih (H) + bias_hh (H), times gates.
    """
    embed = vocab * hidden
    out = hidden * vocab + vocab
    per_layer = gates * (2 * hidden * hidden + 2 * hidden)
    return embed + out + n_layers * per_layer


def matched_hidden_size(config, gates: int) -> int:
    """Smallest-error hidden width so the recurrent model matches the Transformer.

    Builds the Transformer from the same config to get the target count, then
    binary-searches ``H`` (param count is monotonic in ``H``).
    """
    from .base_models import Transformer

    target = sum(p.numel() for p in Transformer(config).parameters())
    vocab, n_layers = config.vocab_size, config.model.num_layers

    lo, hi = 1, 8192
    while lo < hi:
        mid = (lo + hi) // 2
        if _recurrent_param_count(vocab, mid, n_layers, gates) < target:
            lo = mid + 1
        else:
            hi = mid
    candidates = [h for h in (lo - 1, lo) if h >= 1]
    return min(candidates, key=lambda h: abs(_recurrent_param_count(vocab, h, n_layers, gates) - target))


class _StackedRecurrent(nn.Module):
    """Shared embedding -> stacked cells -> output-projection skeleton."""

    cell_cls = None  # nn.RNN or nn.LSTM, set by subclass
    gates = None     # 1 (RNN) or 4 (LSTM), set by subclass

    def __init__(self, config):
        super().__init__()
        H = getattr(config.model, "rnn_hidden", None)
        if H is None:
            H = matched_hidden_size(config, self.gates)
        self.hidden_size = H

        dropout = config.model.dropout
        self.embed = nn.Embedding(config.vocab_size, H)
        self.layers = nn.ModuleList(
            [self.cell_cls(input_size=H, hidden_size=H, batch_first=True)
             for _ in range(config.model.num_layers)]
        )
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.output_layer = nn.Linear(H, config.vocab_size)
        self.to(config.device)

    def forward(self, x, **kwargs):
        # **kwargs absorbs the Transformer's kv_cache/cache_pos args so callers
        # stay architecture-agnostic; recurrent models are inherently sequential.
        x = self.embed(x)
        for layer in self.layers:
            out, _ = layer(x)  # (B, T, H); unidirectional => causal
            if self.dropout is not None:
                out = self.dropout(out)
            x = out
        return self.output_layer(x)


class LSTMModel(_StackedRecurrent):
    cell_cls = nn.LSTM
    gates = 4


class RNNModel(_StackedRecurrent):
    cell_cls = nn.RNN  # tanh nonlinearity (PyTorch default)
    gates = 1
