"""Analysis 3 (model vs. Bayesian solutions) extraction across architectures.

The model-vs-Bayes comparison only needs model logits + the algebraic Bayesian
predictors, both already architecture-agnostic. This test covers the one coupled
piece -- the real-position hidden extractor used by the lambda/posterior probe --
confirming it now runs for RNN/LSTM (and the LSTM cell state) with the correct
width, while the Transformer path and width are unchanged.
"""

import pytest
import torch

from icl.latent_markov.latent_config import get_config_base
from icl.latent_markov.markov_latent import LatentMarkov
from icl.models.factory import build_model
from icl.utils.unified_hiddens import _compute_hiddens_at_real_tokens

SEQ_LEN, N_TASKS, B = 12, 3, 8


def _setup(arch):
    cfg = get_config_base()
    cfg.device = "cpu"
    cfg.model.arch = arch
    cfg.model.num_layers = 2
    cfg.model.dropout = None
    cfg.seq_len = SEQ_LEN
    cfg.batch_size = B
    cfg.task.n_tasks = N_TASKS
    cfg.task.n_minor_tasks = 0
    cfg.task.p_minor = 0.0
    torch.manual_seed(0)
    return cfg, build_model(cfg).eval(), LatentMarkov(cfg)


@pytest.mark.parametrize("arch", ["transformer", "lstm", "rnn"])
def test_real_token_hiddens_shape_and_width(arch):
    cfg, model, sampler = _setup(arch)
    hiddens = _compute_hiddens_at_real_tokens(cfg, model, sampler, B)
    D = cfg.model.emb_dim if arch == "transformer" else model.hidden_size
    # (n_layers, n_tasks, seq_len-1, B, D)
    assert hiddens.shape == (cfg.model.num_layers, N_TASKS, SEQ_LEN - 1, B, D)


def test_lstm_cell_vs_hidden_real_tokens():
    cfg, model, sampler = _setup("lstm")
    h = _compute_hiddens_at_real_tokens(cfg, model, sampler, B, state="hidden")
    c = _compute_hiddens_at_real_tokens(cfg, model, sampler, B, state="cell")
    assert h.shape == c.shape
    assert not torch.allclose(h, c, atol=1e-4)


def test_transformer_rejects_cell_state():
    cfg, model, sampler = _setup("transformer")
    with pytest.raises(ValueError, match="only valid for LSTM"):
        _compute_hiddens_at_real_tokens(cfg, model, sampler, B, state="cell")


def test_bayes_predictors_are_arch_agnostic():
    """The two Bayesian predictors consume only token samples, never the model."""
    import inspect
    from icl.latent_markov.analysis import bayes
    predict_methods = [
        obj.predict
        for _, obj in inspect.getmembers(bayes, inspect.isclass)
        if hasattr(obj, "predict")
    ]
    assert predict_methods, "expected Bayesian predictor classes with .predict"
    for predict in predict_methods:
        params = list(inspect.signature(predict).parameters)
        # signature is (self, samples, ...) -- no `model` argument anywhere
        assert "model" not in params
