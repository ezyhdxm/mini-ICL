#!/usr/bin/env python3
"""Train Dyck (E4) experiments in parallel across GPUs.

Each k value corresponds to a different number of minor tasks
(k=-1 → 1 minor task with p_minor≈0, k=0 → 1 minor, k≥1 → 2^k minors).

Usage (from project root):
    uv run python scripts/run_dyck.py                    # k = -1 … 9 (default)
    uv run python scripts/run_dyck.py -k -1 0 2 4 10    # specific k values
    uv run python scripts/run_dyck.py -k -1 --n-gpus 1  # single GPU
"""

import argparse
import sys

from icl.utils.logger import setup_logger

logger = setup_logger("run_dyck")

DEFAULT_K_LIST = [-1, 10]


def run_dyck(k_list, *, n_gpus=2, num_epochs=30_000, warmup_steps=15_000,
             verbose=True):
    from icl.utils.unified_interface import unified_train_parallel

    logger.info(
        f"Starting Dyck training  "
        f"k_list={k_list}  num_epochs={num_epochs}  "
        f"warmup_steps={warmup_steps}  n_gpus={n_gpus}"
    )
    results = unified_train_parallel(
        task_name="dyck",
        k_list=k_list,
        n_gpus=n_gpus,
        verbose=verbose,
        num_epochs=num_epochs,
        warmup_steps=warmup_steps,
    )
    logger.info("Dyck training complete.")
    return results


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Dyck experiments in parallel across GPUs.")
    p.add_argument(
        "-k", "--k-list", type=int, nargs="+", default=DEFAULT_K_LIST,
        metavar="K",
        help=f"k values to train (default: {DEFAULT_K_LIST})",
    )
    p.add_argument(
        "--n-gpus", type=int, default=2,
        help="Number of GPUs to use (default: 2)",
    )
    p.add_argument(
        "--num-epochs", type=int, default=30_000,
        help="Training steps / epochs (default: 30000)",
    )
    p.add_argument(
        "--warmup-steps", type=int, default=15_000,
        help="LR warmup steps (default: 15000)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"=== Dyck training start (k_list={args.k_list}) ===")
    try:
        run_dyck(
            args.k_list,
            n_gpus=args.n_gpus,
            num_epochs=args.num_epochs,
            warmup_steps=args.warmup_steps,
        )
    except Exception as e:
        logger.error(f"Dyck training failed: {e}", exc_info=True)
        sys.exit(1)
    logger.info("=== Dyck training done ===")
