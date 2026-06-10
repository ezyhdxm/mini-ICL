"""Analysis 2 (task-vector interpolation) end-to-end across architectures.

Reading the source showed the live interpolation analysis does NOT use the
dead KV-cache extractor; it runs on the real-position hidden tensor (the
piece-7 path) through pure tensor algebra: estimate per-task vectors, decompose
each hidden as a convex combination (lambda), project onto the simplex. This
test drives that exact chain on small RNN/LSTM/Transformer models (and the LSTM
cell state) to prove analysis 2 works for all architectures.
"""

import numpy as np
import pytest
import torch

from icl.latent_markov.latent_config import get_config_base
from icl.latent_markov.markov_latent import LatentMarkov
from icl.models.factory import build_model
from icl.utils.unified_hiddens import _compute_hiddens_at_real_tokens
from icl.utils.separability import estimate_task_vectors_by_averaging
from icl.linear.linear_utils import estimate_lambda_with_r2
from icl.utils.linear_algebra_utils import _project_onto_simplex_np

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


def _run_interpolation(cfg, model, sampler, state="hidden"):
    """The exact downstream chain from trajectory.py / posterior.py."""
    hiddens = _compute_hiddens_at_real_tokens(cfg, model, sampler, B, state=state)
    hiddens_layer = hiddens[-1].to(torch.float32)  # last layer: (K, T, B, D)
    K, T, _, D = hiddens_layer.shape

    est_positions = list(range(T - 4, T))  # late positions
    task_vecs, grand_mean = estimate_task_vectors_by_averaging(hiddens_layer, est_positions)
    targets = hiddens_layer.mean(dim=-2) - grand_mean  # (K, T, D)

    lambdas, r2_scores, _, _ = estimate_lambda_with_r2(
        task_vecs, targets, is_zero_mean=True,
    )
    return task_vecs, lambdas, r2_scores, (K, T, D)


@pytest.mark.parametrize("arch", ["transformer", "lstm", "rnn"])
def test_interpolation_pipeline_runs(arch):
    cfg, model, sampler = _setup(arch)
    D_expected = cfg.model.emb_dim if arch == "transformer" else model.hidden_size

    task_vecs, lambdas, r2_scores, (K, T, D) = _run_interpolation(cfg, model, sampler)

    assert D == D_expected
    assert task_vecs.shape == (K, D)
    lam = torch.as_tensor(lambdas).float()
    assert lam.shape == (K, T, K)          # (eval-task, position, mixing weights)
    assert torch.isfinite(lam).all()
    assert torch.isfinite(torch.as_tensor(r2_scores).float()).all()

    # Project onto the simplex -> valid mixing distributions over the K tasks.
    lam_np = lam.reshape(-1, K).numpy()
    proj = _project_onto_simplex_np(lam_np)
    assert np.all(proj >= -1e-6)
    assert np.allclose(proj.sum(axis=1), 1.0, atol=1e-5)


def test_interpolation_runs_on_lstm_cell_state():
    """The interpolation pipeline also flows through the LSTM cell state."""
    cfg, model, sampler = _setup("lstm")
    task_vecs, lambdas, _, (K, T, D) = _run_interpolation(cfg, model, sampler, state="cell")
    assert task_vecs.shape == (K, D)
    lam_np = torch.as_tensor(lambdas).float().reshape(-1, K).numpy()
    proj = _project_onto_simplex_np(lam_np)
    assert np.allclose(proj.sum(axis=1), 1.0, atol=1e-5)
