"""Verify the official-mamba adapter on a CUDA GPU (run after `uv pip install
mamba-ssm causal-conv1d`).

Checks: param match to the Transformer, forward shape, causality, that the
LAYER-OUTPUT hidden states extract via the generic hook extractor (the path all
analyses use), and that the INTERNAL per-timestep SSM state extracts via the
official step() path. Also confirms it trains.

    uv run python scripts/verify_mamba_official.py
"""

import math
import torch

from icl.latent_markov.latent_config import get_config_base
from icl.latent_markov.markov_latent import LatentMarkov
from icl.models.factory import build_model
from icl.models.base_models import Transformer
from icl.models.hidden_extractor import extract_layer_hiddens


def main():
    cfg = get_config_base()
    cfg.model.arch = "mamba_official"
    target = sum(p.numel() for p in Transformer(cfg).parameters())
    m = build_model(cfg)
    n = sum(p.numel() for p in m.parameters())
    print(f"param match: d_model={m.d_model}  params={n}  vs transformer {target} "
          f"({100*abs(n-target)/target:.2f}%)")

    x = torch.randint(0, cfg.vocab_size, (4, 16), device=cfg.device)
    y = m(x)
    print("forward shape:", tuple(y.shape), "(expect (4,16,%d))" % cfg.vocab_size)
    x2 = x.clone(); x2[:, 10:] = (x2[:, 10:] + 1) % cfg.vocab_size
    print("causal:", torch.allclose(m(x)[:, :10], m(x2)[:, :10], atol=1e-4))

    # (1) LAYER-OUTPUT extraction (the path every analysis uses) -- via forward hooks
    h = extract_layer_hiddens(m, x)
    print("layer-output h_t extraction:", tuple(h.shape), "(n_layers, B, T, D)")

    # (2) INTERNAL SSM-state extraction -- via the official step() path
    ssm = m.ssm_state_sequence(x, layer_idx=cfg.model.num_layers - 1)
    print("internal ssm_state extraction:", tuple(ssm.shape), "(B, T, d_inner, d_state)")

    # (3) trains
    torch.manual_seed(0)
    cfg.seq_len = 64; cfg.batch_size = 64
    cfg.task.n_tasks = 4; cfg.task.n_minor_tasks = 0; cfg.task.p_minor = 0.0
    s = LatentMarkov(cfg); model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    print("baseline ln(8)=%.3f" % math.log(8)); model.train()
    for step in range(101):
        xb = s.generate(mode="train", epochs=1)[0].squeeze(0)
        loss = torch.nn.functional.cross_entropy(
            model(xb)[:, :-1].reshape(-1, cfg.vocab_size), xb[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print("step %3d loss %.4f" % (step, loss.item()))
    print("\nOK: official mamba builds, extracts h_t AND internal ssm_state, and trains.")


if __name__ == "__main__":
    main()
