#!/usr/bin/env python3
"""Train latent Markov with k=10, vocab_size=6, longer seq_len and more epochs.

Changes from default latent config:
  - vocab_size:   8 → 6
  - seq_len:      192 → 384  (doubled)
  - num_epochs:   20_000 → 40_000
  - warmup_steps: 10_000 → 20_000
  - k=10  (n_minor_tasks = 2^10 = 1024)

Usage (from project root):
    PYTHONPATH=src python scripts/train_latent_k10_v6_long.py
"""

import os
import argparse
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=384)
    p.add_argument("--num-epochs", type=int, default=40_000)
    p.add_argument("--warmup-steps", type=int, default=20_000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Print config and exp_name without training")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    from icl.latent_markov.latent_config import get_config_base
    from icl.models.base_models import Transformer
    from icl.utils.train import train_model_with_plot
    from icl.utils.basic import canonicalize_config_for_exp, get_hash

    config = get_config_base()

    # ---- Task ----
    config.vocab_size = args.vocab_size
    config.seq_len = args.seq_len
    config.task.n_minor_tasks = 2 ** args.k
    # keep n_tasks=3 (major), p_minor=0.1, alpha=1, order=1

    # ---- Model ----
    config.model.pos_max_len = args.seq_len

    # ---- Training ----
    config.training.num_epochs = args.num_epochs
    config.training.warmup_steps = args.warmup_steps

    # ---- Device ----
    if args.device is not None:
        config.device = args.device

    # Compute exp_name
    canonicalize_config_for_exp(config)
    exp_name = f"train_{get_hash(config)}"

    log.info("=" * 60)
    log.info("  Latent Markov training")
    log.info("=" * 60)
    log.info(f"  vocab_size   = {config.vocab_size}")
    log.info(f"  seq_len      = {config.seq_len}")
    log.info(f"  n_tasks      = {config.task.n_tasks} (major)")
    log.info(f"  n_minor      = {config.task.n_minor_tasks}")
    log.info(f"  p_minor      = {config.task.p_minor}")
    log.info(f"  num_epochs   = {config.training.num_epochs}")
    log.info(f"  warmup_steps = {config.training.warmup_steps}")
    log.info(f"  num_layers   = {config.model.num_layers}")
    log.info(f"  emb_dim      = {config.model.emb_dim}")
    log.info(f"  batch_size   = {config.batch_size}")
    log.info(f"  exp_name     = {exp_name}")
    log.info(f"  work_dir     = {config.work_dir}")
    log.info("=" * 60)

    if args.dry_run:
        log.info("[dry-run] Exiting without training.")
        return

    # Check if already done
    exp_dir = os.path.join(config.work_dir, exp_name)
    if os.path.exists(os.path.join(exp_dir, "log.json")):
        log.info(f"Experiment already completed at {exp_dir}")
        return

    # Reset device for actual training
    if args.device is not None:
        config.device = args.device
    else:
        from icl.device_utils import get_default_device
        config.device = get_default_device()

    model = Transformer(config)
    model = model.to(config.device)

    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    log.info(f"Training on device: {config.device}")

    results = train_model_with_plot(model, config, show=False, verbose=False)

    log.info(f"Training completed in {time.time() - t0:.1f}s")
    log.info(f"Results saved to {exp_dir}")


if __name__ == "__main__":
    main()
