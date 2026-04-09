"""Train latent, coin, and linear with major tasks only (no minor pool).

Sweeps ``n_tasks`` (major-task count) over ``2**min_exp .. 2**max_exp`` inclusive,
with ``n_minor_tasks=1`` and ``p_minor=1e-12``, matching the ``k=-1`` branch in
``unified_train`` (dummy minor pool, negligible minor probability).

Run from the ``mini-ICL`` repo root (same as ``run_pipeline.py``), e.g.::

    uv run python scripts/run_major_only_sweep.py
"""

from __future__ import annotations

import argparse
import sys

from icl.utils.logger import setup_logger

logger = setup_logger("major_only_sweep")

DEFAULT_TASK_ORDER = ("latent", "coin", "linear")


def _n_major_list(exp_min: int, exp_max: int) -> list[int]:
    if exp_min > exp_max:
        raise ValueError(f"exp_min ({exp_min}) must be <= exp_max ({exp_max})")
    return [2**e for e in range(exp_min, exp_max + 1)]


def run_sweep(
    task_names: tuple[str, ...],
    n_major_list: list[int],
    n_gpus: int,
) -> None:
    from icl.utils.unified_training import unified_train_major_only_parallel

    for name in task_names:
        if name == "latent":
            common = dict(num_epochs=30_000, warmup_steps=15_000)
        elif name == "coin":
            common = dict(num_epochs=30_000, warmup_steps=15_000)
        elif name == "linear":
            common = dict(
                n_layer=16,
                total_steps=30_000,
                warmup_steps=15_000,
                batch_size=256,
                max_grad_norm=1.0,
                noise_scale=0.5,
            )
        else:
            raise ValueError(f"Unknown task: {name!r}")

        logger.info(
            f"=== {name}: major-only sweep n_major in {n_major_list} "
            f"(n_gpus={n_gpus}) ==="
        )
        try:
            unified_train_major_only_parallel(
                [name],
                n_major_list,
                n_gpus=n_gpus,
                verbose=True,
                **common,
            )
        except Exception as e:
            logger.error(f"ERROR in {name} sweep: {e}", exc_info=True)
            sys.exit(1)

    logger.info("=== Major-only sweep complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train latent / coin / linear with major tasks only; "
            "sweep n_tasks from 2**exp_min to 2**exp_max."
        )
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=list(DEFAULT_TASK_ORDER),
        help=f"Subset/order of tasks (default: {' '.join(DEFAULT_TASK_ORDER)})",
    )
    parser.add_argument(
        "--exp-min",
        type=int,
        default=2,
        help="Minimum exponent m so smallest n_major = 2**m (default: 2 -> 4 tasks)",
    )
    parser.add_argument(
        "--exp-max",
        type=int,
        default=10,
        help="Maximum exponent M so largest n_major = 2**M (default: 10 -> 1024 tasks)",
    )
    parser.add_argument(
        "-g",
        "--n-gpus",
        type=int,
        default=2,
        help="Number of parallel workers / GPUs (default: 2, same as run_pipeline)",
    )
    args = parser.parse_args()

    for t in args.tasks:
        if t not in DEFAULT_TASK_ORDER:
            logger.error(
                f"Unknown task {t!r}; allowed: {', '.join(DEFAULT_TASK_ORDER)}"
            )
            sys.exit(2)

    majors = _n_major_list(args.exp_min, args.exp_max)
    logger.info(
        f"Major-only sweep: tasks={list(args.tasks)}, "
        f"n_major values = {majors}"
    )
    run_sweep(tuple(args.tasks), majors, args.n_gpus)


if __name__ == "__main__":
    main()
