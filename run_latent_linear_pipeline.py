"""Sequential pipeline: run latent training, then linear training."""

import sys
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_latent():
    from icl.utils.unified_interface import unified_train_parallel
    log("Starting latent training (k_list=[-1, 10], num_epochs=30_000, warmup_steps=15_000)")
    results = unified_train_parallel(
        task_name="latent",
        k_list=[-1, 10],
        n_gpus=2,
        verbose=True,
        warmup_steps=15_000,
        num_epochs=30_000,
    )
    log("Latent training complete.")
    return results


def run_linear():
    from icl.utils.unified_interface import unified_train_parallel
    log("Starting linear training (k_list=[-1, 10], total_steps=30_000, warmup_steps=15_000)")
    results = unified_train_parallel(
        task_name="linear",
        k_list=[-1, 10],
        n_gpus=2,
        n_layer=16,
        total_steps=30_000,
        warmup_steps=15_000,
        batch_size=256,
        max_grad_norm=1.0,
        noise_scale=0.5,
        verbose=True,
    )
    log("Linear training complete.")
    return results


if __name__ == "__main__":
    log("=== Pipeline start ===")

    try:
        latent_results = run_latent()
    except Exception as e:
        log(f"ERROR in latent training: {e}")
        sys.exit(1)

    try:
        linear_results = run_linear()
    except Exception as e:
        log(f"ERROR in linear training: {e}")
        sys.exit(1)

    log("=== Pipeline complete ===")
