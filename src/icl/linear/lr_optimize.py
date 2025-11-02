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
    # Learning rate schedule function
    def warmup_cosine(step):
        if step < kwargs["warmup_steps"]:
            return step / kwargs["warmup_steps"]
        progress = (step - kwargs["warmup_steps"]) / (kwargs["total_steps"] - kwargs["warmup_steps"])
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535)))

    def triangle(step):
        if step < kwargs["warmup_steps"]:
            return step / kwargs["warmup_steps"]
        return max(0.0, (kwargs["total_steps"] - step) / (kwargs["total_steps"] - kwargs["warmup_steps"]))

    if schedule == "warmup_cosine_decay":
        lr_lambda = warmup_cosine
    elif schedule == "triangle":
        lr_lambda = triangle
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
