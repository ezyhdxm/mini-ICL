"""Analysis 1 (conditional-variance) hidden extraction across architectures.

Exercises `compute_hiddens_token_conditioned_coin` directly (model + sampler in
memory, no checkpoint load) to confirm the token-conditioned extraction now
works for RNN/LSTM and the LSTM cell state, while the Transformer path and its
hidden width are unchanged.
"""

import pytest
import torch

from icl.latent_markov.latent_config import get_config_base
from icl.latent_markov.markov_latent import LatentMarkov
from icl.models.factory import build_model
from icl.coin.analysis._helpers import (
    compute_hiddens_token_conditioned_coin,
    _model_hidden_dim,
)

SEQ_LEN, N_TASKS, BATCH = 12, 3, 8
POSITIONS = [0, 5]


def _setup(arch):
    cfg = get_config_base()
    cfg.device = "cpu"
    cfg.model.arch = arch
    cfg.model.num_layers = 2  # smaller for test speed (head/mlp tuples stay len 6)
    cfg.model.dropout = None
    cfg.seq_len = SEQ_LEN
    cfg.batch_size = BATCH
    cfg.task.n_tasks = N_TASKS
    cfg.task.n_minor_tasks = 0
    cfg.task.p_minor = 0.0
    torch.manual_seed(0)
    sampler = LatentMarkov(cfg)
    model = build_model(cfg).eval()
    return cfg, model, sampler


@pytest.mark.parametrize("arch", ["transformer", "lstm", "rnn"])
def test_token_conditioned_shape_and_width(arch):
    cfg, model, sampler = _setup(arch)
    hiddens, info = compute_hiddens_token_conditioned_coin(
        config=cfg, model=model, sampler=sampler,
        batch_size=BATCH, positions_of_interest=POSITIONS,
    )
    L = cfg.model.num_layers
    D = _model_hidden_dim(model, cfg)
    # (L, n_positions, max_unique_tokens, n_tasks, batch, D)
    assert hiddens.shape[0] == L
    assert hiddens.shape[1] == len(POSITIONS)
    assert hiddens.shape[3] == N_TASKS
    assert hiddens.shape[4] == BATCH
    assert hiddens.shape[5] == D
    assert info["positions"] == POSITIONS
    # transformer keeps emb_dim width; recurrent uses its matched hidden_size
    assert D == (cfg.model.emb_dim if arch == "transformer" else model.hidden_size)


def test_lstm_cell_differs_from_hidden():
    cfg, model, sampler = _setup("lstm")
    common = dict(config=cfg, model=model, sampler=sampler,
                  batch_size=BATCH, positions_of_interest=POSITIONS)
    h_hidden, _ = compute_hiddens_token_conditioned_coin(**common, state="hidden")
    h_cell, _ = compute_hiddens_token_conditioned_coin(**common, state="cell")
    assert h_hidden.shape == h_cell.shape
    # cell state is a genuinely different representation from the hidden output
    assert not torch.allclose(h_hidden, h_cell, atol=1e-4)


def test_transformer_rejects_cell_state():
    cfg, model, sampler = _setup("transformer")
    with pytest.raises(ValueError, match="only valid for LSTM"):
        compute_hiddens_token_conditioned_coin(
            config=cfg, model=model, sampler=sampler,
            batch_size=BATCH, positions_of_interest=POSITIONS, state="cell",
        )
