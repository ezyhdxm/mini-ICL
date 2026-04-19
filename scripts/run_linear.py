"""Run a single linear-regression training job with uv-friendly CLI flags."""

import argparse
from pathlib import Path

from icl.utils.unified_interface import get_exp_name, unified_train


DEFAULT_TOTAL_STEPS = 30_000
DEFAULT_WARMUP_STEPS = 15_000
DEFAULT_BATCH_SIZE = 256
DEFAULT_N_LAYER = 16
DEFAULT_MAX_GRAD_NORM = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one linear-regression training job."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Minor-task exponent when log2=True. k=10 means 1024 minor tasks.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=2.0,
        help="Observation noise scale for the linear-regression task.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device override, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=DEFAULT_TOTAL_STEPS,
        help=f"Total training steps (default: {DEFAULT_TOTAL_STEPS}).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help=f"Warmup steps (default: {DEFAULT_WARMUP_STEPS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Task batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--n-layer",
        type=int,
        default=DEFAULT_N_LAYER,
        help=f"Transformer layer count (default: {DEFAULT_N_LAYER}).",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=DEFAULT_MAX_GRAD_NORM,
        help=f"Gradient clipping norm (default: {DEFAULT_MAX_GRAD_NORM}).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence wandb console output during training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_minor_tasks = 2 ** args.k if args.k >= 0 else 1

    exp_name = get_exp_name(
        "linear",
        args.k,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        n_layer=args.n_layer,
        max_grad_norm=args.max_grad_norm,
        noise_scale=args.noise_scale,
    )
    exp_dir = Path("results") / "linear" / exp_name

    print("Starting linear training with:")
    print(f"  k={args.k}")
    print(f"  n_minor_tasks={n_minor_tasks}")
    print(f"  noise_scale={args.noise_scale}")
    print(f"  total_steps={args.total_steps}")
    print(f"  warmup_steps={args.warmup_steps}")
    print(f"  batch_size={args.batch_size}")
    print(f"  n_layer={args.n_layer}")
    print(f"  max_grad_norm={args.max_grad_norm}")
    print(f"  device={args.device or 'default'}")
    print(f"  expected_exp_dir={exp_dir}")

    _, log = unified_train(
        "linear",
        args.k,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        n_layer=args.n_layer,
        max_grad_norm=args.max_grad_norm,
        noise_scale=args.noise_scale,
        quiet=args.quiet,
        device=args.device,
    )

    print("Training complete.")
    print(f"  checkpoint_path={exp_dir / 'checkpoint.pt'}")
    print(f"  log_path={exp_dir / 'log.json'}")
    print(f"  logged_eval_steps={len(log.get('train/step', []))}")


if __name__ == "__main__":
    main()
