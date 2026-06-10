"""Model factory: build a sequence model from `config.model.arch`.

All architectures share the same contract as :class:`Transformer`:
they map token ids ``(B, T)`` to next-token logits ``(B, T, vocab_size)``,
so any of them drops into the existing training loop unchanged.

For now only "transformer" is registered; RNN/LSTM/Mamba are added in
later pieces. Defaulting to "transformer" keeps existing configs working.
"""

from torch import nn

from .base_models import Transformer
from .recurrent import LSTMModel, RNNModel


def build_model(config) -> nn.Module:
    """Construct the model named by ``config.model.arch`` (default "transformer")."""
    arch = getattr(config.model, "arch", "transformer")

    if arch == "transformer":
        return Transformer(config)
    if arch == "lstm":
        return LSTMModel(config)
    if arch == "rnn":
        return RNNModel(config)

    raise ValueError(
        f"Unknown config.model.arch={arch!r}. "
        f"Registered architectures: ['transformer', 'lstm', 'rnn']."
    )
