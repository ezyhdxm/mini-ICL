import math
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Tuple, Callable

def get_optimizer_and_lr_schedule(
    optimizer: str,
    schedule: str,
    params: torch.nn.Parameter,
    **kwargs
) -> Tuple[torch.optim.Optimizer, Callable[[int], float]]:
    min_lr_ratio = kwargs.get("min_lr", 0.0) / kwargs["lr"] if kwargs.get("min_lr") else 0.0

    def warmup_cosine(step):
        if step < kwargs["warmup_steps"]:
            return step / kwargs["warmup_steps"]
        progress = (step - kwargs["warmup_steps"]) / (kwargs["total_steps"] - kwargs["warmup_steps"])
        cosine = 0.5 * (1 + math.cos(progress * math.pi))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    def triangle(step):
        if step < kwargs["warmup_steps"]:
            return step / kwargs["warmup_steps"]
        decay = max(0.0, (kwargs["total_steps"] - step) / (kwargs["total_steps"] - kwargs["warmup_steps"]))
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay

    def convex_decay(step):
        p = kwargs.get("decay_power", 2.0)
        if step < kwargs["warmup_steps"]:
            return step / kwargs["warmup_steps"]
        progress = (step - kwargs["warmup_steps"]) / (kwargs["total_steps"] - kwargs["warmup_steps"])
        decay = max(0.0, 1.0 - progress) ** p
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay

    if schedule == "warmup_cosine_decay":
        lr_lambda = warmup_cosine
    elif schedule == "triangle":
        lr_lambda = triangle
    elif schedule == "convex":
        lr_lambda = convex_decay
    else:
        raise NotImplementedError(f"Unsupported schedule: {schedule}")


    if optimizer == "adam":
        optim = Adam(params, lr=kwargs["lr"])
    elif optimizer == "adamw":
        optim = AdamW(params, lr=kwargs["lr"], weight_decay=kwargs["weight_decay"])
    else:
        raise NotImplementedError(f"Unsupported optimizer: {optimizer}")

    # Scheduler
    scheduler = LambdaLR(optim, lr_lambda=lr_lambda)

    return optim, scheduler
