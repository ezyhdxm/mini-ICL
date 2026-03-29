"""Run coin experiments with k in [-1, 0, ..., 10] at vocab_size=16."""

import argparse
import sys

from icl.utils.logger import setup_logger

logger = setup_logger("coin_v16")

K_LIST = list(range(-1, 11))  # -1, 0, 1, ..., 10
VOCAB_SIZE = 16


def run(k_list, n_gpus, num_epochs, warmup_steps):
    from icl.utils.unified_interface import unified_train_parallel

    logger.info(
        f"Starting coin training: k_list={k_list}, vocab_size={VOCAB_SIZE}, "
        f"num_epochs={num_epochs}, warmup_steps={warmup_steps}, n_gpus={n_gpus}"
    )
    results = unified_train_parallel(
        task_name="coin",
        k_list=k_list,
        n_gpus=n_gpus,
        verbose=True,
        vocab_size=VOCAB_SIZE,
        num_epochs=num_epochs,
        warmup_steps=warmup_steps,
    )
    logger.info("Coin training complete.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train coin models for k in [-1..10] with vocab_size=16."
    )
    parser.add_argument(
        "-k", "--k-list", type=int, nargs="+", default=K_LIST,
        help=f"k values to train (default: {K_LIST})",
    )
    parser.add_argument(
        "--n-gpus", type=int, default=None,
        help="Number of GPUs to use in parallel (default: auto-detect)",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=30_000,
        help="Number of training epochs per run (default: 30000)",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=15_000,
        help="Warmup steps per run (default: 15000)",
    )
    args = parser.parse_args()

    logger.info(f"=== coin v16 run start (k_list={args.k_list}) ===")
    try:
        run(args.k_list, args.n_gpus, args.num_epochs, args.warmup_steps)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
    logger.info("=== coin v16 run complete ===")
