"""Parity tests for the generic per-layer hidden-state extractor.

For every architecture we independently replay the model's forward pass and
confirm ``extract_layer_hiddens`` captures exactly the same per-layer hidden
states. This is the guarantee that wiring the analyses onto this extractor does
not change the Transformer results before we trust it on RNN/LSTM.
"""

import pytest
import torch

from icl.latent_markov.latent_config import get_config_base
from icl.models.factory import build_model
from icl.models.hidden_extractor import extract_layer_hiddens

B, T = 3, 12


def _make_config(arch):
    cfg = get_config_base()
    cfg.device = "cpu"
    cfg.model.arch = arch
    cfg.model.dropout = None  # keep pre/post-dropout identical so replay is exact
    return cfg


@pytest.fixture(autouse=True)
def _seeded():
    torch.manual_seed(0)


def _manual_transformer_hiddens(model, x):
    """Replay Transformer.forward, collecting each block's output."""
    h = model.embed(x)
    outs = []
    for layer in model.layers:
        h = layer(h)  # TFBlock: (B, T, D) -> (B, T, D)
        outs.append(h)
    return torch.stack(outs, dim=0)


def _manual_recurrent_hiddens(model, x):
    """Replay _StackedRecurrent.forward, collecting each cell's output."""
    h = model.embed(x)
    outs = []
    for layer in model.layers:
        out, _ = layer(h)  # nn.RNN / nn.LSTM return (output, state)
        h = out
        outs.append(out)
    return torch.stack(outs, dim=0)


@pytest.mark.parametrize(
    "arch, manual_fn",
    [
        ("transformer", _manual_transformer_hiddens),
        ("lstm", _manual_recurrent_hiddens),
        ("rnn", _manual_recurrent_hiddens),
    ],
)
def test_extractor_matches_manual_forward(arch, manual_fn):
    cfg = _make_config(arch)
    model = build_model(cfg)
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (B, T))

    got = extract_layer_hiddens(model, x)
    with torch.no_grad():
        expected = manual_fn(model, x)

    n_layers = len(model.layers)
    D = model.hidden_size if arch != "transformer" else cfg.model.emb_dim
    assert got.shape == (n_layers, B, T, D)
    assert torch.allclose(got, expected, atol=1e-6)


def test_extractor_layer_subset():
    """Selecting a subset of layers returns those layers in order."""
    cfg = _make_config("lstm")
    model = build_model(cfg)
    x = torch.randint(0, cfg.vocab_size, (B, T))

    full = extract_layer_hiddens(model, x)
    subset = extract_layer_hiddens(model, x, layers=[0, 2, 4])
    assert subset.shape[0] == 3
    assert torch.allclose(subset, full[[0, 2, 4]], atol=1e-6)


def test_extractor_restores_training_mode():
    """The extractor must leave model.training untouched."""
    cfg = _make_config("rnn")
    model = build_model(cfg)
    model.train()
    x = torch.randint(0, cfg.vocab_size, (B, T))
    extract_layer_hiddens(model, x)
    assert model.training is True


def _manual_lstm_final_cell(model, x):
    """Per-layer final cell state c_n via nn.LSTM's full-sequence forward (ground truth)."""
    inp = model.embed(x)
    finals = []
    for layer in model.layers:
        out, (_, c_n) = layer(inp)
        finals.append(c_n[0])  # (B, H)
        inp = model.dropout(out) if model.dropout is not None else out
    return torch.stack(finals, dim=0)  # (n_layers, B, H)


def test_cell_state_matches_lstm_final_cell():
    """Stepped c_t at the last timestep == nn.LSTM's exposed final c_n, every layer."""
    cfg = _make_config("lstm")
    model = build_model(cfg)
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (B, T))

    cell = extract_layer_hiddens(model, x, state="cell")
    assert cell.shape == (len(model.layers), B, T, model.hidden_size)

    expected_final = _manual_lstm_final_cell(model, x)
    assert torch.allclose(cell[:, :, -1, :], expected_final, atol=1e-6)


@pytest.mark.parametrize("arch", ["rnn", "transformer"])
def test_cell_state_rejects_non_lstm(arch):
    """state='cell' is undefined for non-LSTM architectures."""
    cfg = _make_config(arch)
    model = build_model(cfg)
    x = torch.randint(0, cfg.vocab_size, (B, T))
    with pytest.raises(ValueError, match="only defined for LSTM"):
        extract_layer_hiddens(model, x, state="cell")
