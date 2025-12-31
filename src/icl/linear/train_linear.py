import torch
from typing import Tuple, Callable
import os
import wandb
from ml_collections import ConfigDict
import hashlib
import json

from icl.linear.lr_task import *
from icl.linear.lr_models import get_model
from icl.linear.lr_optimize import get_optimizer_and_lr_schedule
from icl.linear.lr_eval import get_bsln_preds, get_model_preds, mse
from icl.linear.lr_utils import tabulate_model

Preds = dict[str, dict[str, torch.Tensor]]

# Adapted from https://github.com/mansheej/icl-task-diversity/blob/main/icl/train.py

########################################################################################################################
# Utilities                                                                                                            #
########################################################################################################################



def get_hash(config: ConfigDict) -> str:
    """
    Generate a hash string from configuration dictionary.
    
    Creates an MD5 hash of the JSON representation of the config,
    used for creating unique experiment identifiers.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Hexadecimal hash string
    """
    return hashlib.md5(config.to_json(sort_keys=True).encode("utf-8")).hexdigest()


def get_sharded_batch_sampler(task: Task, is_eval: bool=False) -> Callable[[int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Create a batch sampler that reshapes data for multi-device training.
    
    Takes a task and returns a function that samples batches and reshapes them
    for distributed training across multiple devices. Currently uses single device.
    
    Args:
        task: Task object with sample_batch method
        is_eval: Whether to use evaluation mode, when is_eval is True, only the major tasks are sampled, otherwise minor tasks may be sampled as well. The name may be misleading. 
    
    Returns:
        Function that takes step number and returns (data, tasks, targets) shaped for devices
    """
    n_devices = 1 # torch.cuda.device_count() or 1  # fallback to 1 if no CUDA

    def sample_batch(step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, tasks, targets = task.sample_batch(step, is_eval=is_eval)
        batch_size = data.shape[0]

        assert batch_size % n_devices == 0, "Batch size must be divisible by number of devices"
        per_device = batch_size // n_devices

        def reshape(x):
            return x.view(n_devices, per_device, *x.shape[1:])

        return reshape(data), reshape(tasks), reshape(targets)

    return sample_batch

def _init_log(bsln_preds_false: Preds, bsln_preds_true: Preds, n_dims: int) -> dict:
    """
    Initialize log dictionary for evaluation metrics.
    
    Sets up the logging structure for tracking training and evaluation metrics,
    including baseline comparisons and transformer performance.
    
    Args:
        bsln_preds_false: Baseline predictions for false evaluation mode, that is, minor tasks may be included. 
        bsln_preds_true: Baseline predictions for true evaluation mode
        n_dims: Number of dimensions for normalization
    
    Returns:
        Dictionary with initialized log structure for all metrics
    """
    log = {"train/step": [], "train/lr": []}
    for _task_name, _task_preds in bsln_preds_false.items():
        log[f"eval/{_task_name}_false"] = {}
        for _bsln_name, _bsln_preds in _task_preds.items():
            log[f"eval/{_task_name}_false"][f"Transformer | {_bsln_name}"] = []
            if _bsln_name != "True":
                _errs = mse(_bsln_preds, _task_preds["True"]) / n_dims
                log[f"eval/{_task_name}_false"][f"{_bsln_name} | True"] = _errs.tolist()
    
    for _task_name, _task_preds in bsln_preds_true.items():
        log[f"eval/{_task_name}_true"] = {}
        for _bsln_name, _bsln_preds in _task_preds.items():
            log[f"eval/{_task_name}_true"][f"Transformer | {_bsln_name}"] = []
            if _bsln_name != "True":
                _errs = mse(_bsln_preds, _task_preds["True"]) / n_dims
                log[f"eval/{_task_name}_true"][f"{_bsln_name} | True"] = _errs.tolist()
        
    print(log.keys())
    return log

@torch.no_grad()
def eval_step(model, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Perform a single evaluation step with the model.
    
    Runs inference in eval mode without computing gradients,
    useful for validation and evaluation.
    
    Args:
        model: Model to evaluate
        data: Input data tensor
        targets: Target data for computing context
    
    Returns:
        Model predictions
    """
    model.eval()
    data = data.to(model.device)
    targets = targets.to(model.device)
    preds = model(data, targets)
    return preds

def generate_wandb_run_name(config: ConfigDict, exp_name: str) -> str:
    """
    Generate a human-readable run name for Weights & Biases logging.
    
    Creates a descriptive name encoding key hyperparameters and configuration
    settings for easy identification in the W&B dashboard.
    
    Args:
        config: Configuration dictionary with task, model, etc.
        exp_name: Unique experiment identifier
    
    Returns:
        Formatted string containing task and model parameters
    """
    task = config.task
    model = config.model
    name = (
        f"{task.name}_{model.name}"
        f"_L{model.n_layer}_D{task.n_dims}_P{task.n_points}_E{model.n_embd}_H{model.n_head}"
        f"_{model.activation}"
        f"_ts{task.task_seed}"
        f"_run{exp_name}"  # for uniqueness
    )
    return name


def train(config: ConfigDict, verbose=False) -> None:
    """
    Main training function for linear regression transformer models.
    
    Handles the complete training pipeline including:
    - Configuration management and experiment tracking
    - Model, optimizer, and scheduler initialization
    - Baseline evaluation
    - Training loop with periodic evaluation
    - Checkpointing and logging
    
    Args:
        config: Configuration dictionary with all hyperparameters
        verbose: Whether to print detailed information during training
    
    Returns:
        Trained model and training logs dictionary
    """
    exp_name = f"train_{get_hash(config)}"
    exp_dir = os.path.join(config.work_dir, exp_name)   

    cur_dir = os.getcwd()
    if cur_dir.endswith("notebooks"):
        exp_dir = os.path.join("..", exp_dir)
    
    print(exp_dir)

    # logging.info(f"Train Experiment\nNAME: {exp_name}\nCONFIG:\n{config}")

    data_type = getattr(torch, config.dtype)

    # Skip if already completed
    log_path = os.path.join(exp_dir, "log.json")
    if os.path.exists(log_path):
        print(f"{exp_name} already completed")
        checkpoint_path = os.path.join(exp_dir, "checkpoint.pt")
        log_path = os.path.join(exp_dir, "log.json")
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
        model = get_model(**config["model"], dtype=data_type)
        model.load_state_dict(checkpoint["model"])
        model = model.to(config.device)
        print(f"Loaded model from {checkpoint_path}")
        return model, (json.load(open(log_path, "r")), checkpoint_path)
    
    # Save config
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        f.write(config.to_json())

    # Model, optimizer, schedule
    model = get_model(**config["model"], dtype=data_type)
    model = model.to(config.device)
    if verbose:
        print(tabulate_model(model, config["task"]["n_dims"], config["task"]["n_points"], config["task"]["batch_size"]))

    optimizer, scheduler = get_optimizer_and_lr_schedule(**config.training, params=model.parameters())
    
    if verbose:
        print("Initialized model, optimizer, and train state")

    # Data samplers
    train_task = get_task(**config["task"], dtype=data_type)
    sample_train_batch = get_sharded_batch_sampler(train_task)

    samplers_eval_false = {
        get_task_name(task): get_sharded_batch_sampler(task, is_eval=False)
        for task in train_task.get_default_eval_tasks(**config["eval"])
    } 
    samplers_eval_true = {
        get_task_name(task): get_sharded_batch_sampler(task, is_eval=True)
        for task in train_task.get_default_eval_tasks(**config["eval"])
    }
    
    if verbose:
        print("Initialized data samplers")

    # Evaluate baselines
    if verbose:
        print("Evaluating baselines...")
    bsln_preds_false = get_bsln_preds(train_task, samplers_eval_false, config["eval"]["n_samples"], config["eval"]["batch_size"])
    bsln_preds_true = get_bsln_preds(train_task, samplers_eval_false, config["eval"]["n_samples"], config["eval"]["batch_size"])


    # Logging
    log = _init_log(bsln_preds_false, bsln_preds_true, config["task"]["n_dims"])
    wandb_name = generate_wandb_run_name(config, exp_name)
    wandb.init(config=config, name=wandb_name, **config["wandb"])
    step = 0

    scaler = torch.amp.GradScaler("cuda")

    # Training loop
    print("Start training...")
    for i in range(1, config["training"]["total_steps"] + 1):
        step += 1
        data, _, targets = sample_train_batch(i)
        data = data.to(config.device)
        targets = targets.to(config.device)
        model.train()
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            preds = model(data, targets)
            loss = torch.mean((preds - targets) ** 2)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Evaluation
        if i % config["eval"]["every"] == 0 or i == config["training"]["total_steps"]:
            # print(f"Step: {i}")
            log["train/step"].append(i)
            lr_val = scheduler.get_last_lr()[0]
            log["train/lr"].append(lr_val)
            wandb.log({"train/lr": lr_val}, step=i)

            eval_preds_false = get_model_preds(
                model, eval_step, samplers_eval_false, config["eval"]["n_samples"], config["eval"]["batch_size"]
            )

            eval_preds_true = get_model_preds(
                model, eval_step, samplers_eval_true, config["eval"]["n_samples"], config["eval"]["batch_size"]
            )

            for task_name, task_preds in bsln_preds_false.items():
                for bsln_name, bsln_target_preds in task_preds.items():
                    bsln_target_preds = bsln_target_preds.to(config.device)
                    errs = mse(eval_preds_false[task_name]["Transformer"], bsln_target_preds) / config["task"]["n_dims"]
                    log[f"eval/{task_name}_false"][f"Transformer | {bsln_name}"].append(errs.tolist())
                    wandb.log({f"eval/{task_name}_false/{bsln_name}": errs.mean().item()}, step=i)
                
            for task_name, task_preds in bsln_preds_true.items():
                for bsln_name, bsln_target_preds in task_preds.items():
                    bsln_target_preds = bsln_target_preds.to(config.device)
                    errs = mse(eval_preds_true[task_name]["Transformer"], bsln_target_preds) / config["task"]["n_dims"]
                    log[f"eval/{task_name}_true"][f"Transformer | {bsln_name}"].append(errs.tolist())
                    wandb.log({f"eval/{task_name}_true/{bsln_name}": errs.mean().item()}, step=i)
                    

            # attns = get_attn(model, data, targets)
            # attn_means_norm_sq = {layer_key: tensor.mean(dim=0).norm(dim=(-1,-2)).square().cpu().item() for layer_key, tensor in attns.items()}
            # attn_vars_sum = {layer_key: tensor.var(dim=0).sum(dim=(-1,-2)).cpu().item() for layer_key, tensor in attns.items()}
            # for layer_key, mean_norm_sq in attn_means_norm_sq.items():
            #    wandb.log({f"eval/attn/{layer_key}/mean_norm_sq": mean_norm_sq}, step=i)
            #    wandb.log({f"eval/attn/{layer_key}/vars_sum": attn_vars_sum[layer_key]}, step=i)
            #    wandb.log({f"eval/attn/{layer_key}/ratio": attn_vars_sum[layer_key] / mean_norm_sq}, step=i)
        
        if (i % config["eval"].get("save_every", 1000) == 0):
            torch.save({
                "model": model.state_dict(), 
                "step": step,
                }, os.path.join(exp_dir, f"model_{step}.pt"))



    # Save final checkpoint
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step
    }, os.path.join(exp_dir, "checkpoint.pt"))

    # Save logs
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print("Training complete.")

    return model, log