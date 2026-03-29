"""Sequential pipeline: run latent, coin, and linear training."""

import argparse
import sys

from icl.utils.logger import setup_logger

logger = setup_logger("pipeline")

DEFAULT_K_LIST = list(range(10))


def run_latent(k_list):
    from icl.utils.unified_interface import unified_train_parallel
    logger.info(f"Starting latent training (k_list={k_list}, num_epochs=30_000, warmup_steps=15_000)")
    results = unified_train_parallel(
        task_name="latent",
        k_list=k_list,
        n_gpus=2,
        verbose=True,
        warmup_steps=15_000,
        num_epochs=30_000,
    )
    logger.info("Latent training complete.")
    return results


def run_coin(k_list):
    from icl.utils.unified_interface import unified_train_parallel
    logger.info(f"Starting coin training (k_list={k_list}, num_epochs=30_000, warmup_steps=15_000)")
    results = unified_train_parallel(
        task_name="coin",
        k_list=k_list,
        n_gpus=2,
        verbose=True,
        warmup_steps=15_000,
        num_epochs=30_000,
    )
    logger.info("Coin training complete.")
    return results


def run_linear(k_list):
    from icl.utils.unified_interface import unified_train_parallel
    logger.info(f"Starting linear training (k_list={k_list}, total_steps=30_000, warmup_steps=15_000)")
    results = unified_train_parallel(
        task_name="linear",
        k_list=k_list,
        n_gpus=2,
        n_layer=16,
        total_steps=30_000,
        warmup_steps=15_000,
        batch_size=256,
        max_grad_norm=1.0,
        noise_scale=0.5,
        verbose=True,
    )
    logger.info("Linear training complete.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training pipeline for latent, coin, and linear tasks.")
    parser.add_argument(
        "-k", "--k-list", type=int, nargs="+", default=DEFAULT_K_LIST,
        help=f"List of k values to train (default: {DEFAULT_K_LIST})",
    )
    args = parser.parse_args()

    logger.info(f"=== Pipeline start (k_list={args.k_list}) ===")

    for name, runner in [("latent", run_latent), ("coin", run_coin), ("linear", run_linear)]:
        try:
            runner(args.k_list)
        except Exception as e:
            logger.error(f"ERROR in {name} training: {e}", exc_info=True)
            sys.exit(1)

    logger.info("=== Pipeline complete ===")
