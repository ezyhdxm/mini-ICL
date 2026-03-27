import torch
from typing import Tuple, Callable
import os
import wandb
from ml_collections import ConfigDict
import hashlib
import json

from icl.linear.lr_task import Task, get_task, get_task_name
from icl.linear.lr_models import get_model
from icl.linear.lr_optimize import get_optimizer_and_lr_schedule
from icl.linear.lr_eval import get_bsln_preds, get_model_preds, mse
from icl.linear.lr_utils import tabulate_model
from icl.utils.basic import canonicalize_config_for_exp
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)

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


def get_sharded_batch_sampler(task: Task, is_eval: bool=False, minor_only: bool=False) -> Callable[[int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Create a batch sampler that reshapes data for multi-device training.
    
    Takes a task and returns a function that samples batches and reshapes them
    for distributed training across multiple devices. Currently uses single device.
    
    Args:
        task: Task object with sample_batch method
        is_eval: Whether to use evaluation mode, when is_eval is True, only the major tasks are sampled, otherwise minor tasks may be sampled as well. The name may be misleading.
        minor_only: If True, sample only from the minor task pool.
    
    Returns:
        Function that takes step number and returns (data, tasks, targets) shaped for devices
    """
    n_devices = 1 # torch.cuda.device_count() or 1  # fallback to 1 if no CUDA

    def sample_batch(step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, tasks, targets = task.sample_batch(step, is_eval=is_eval, minor_only=minor_only)
        batch_size = data.shape[0]

        assert batch_size % n_devices == 0, "Batch size must be divisible by number of devices"
        per_device = batch_size // n_devices

        def reshape(x):
            return x.view(n_devices, per_device, *x.shape[1:])

        return reshape(data), reshape(tasks), reshape(targets)

    return sample_batch

def _init_log(bsln_preds: Preds, n_dims: int) -> dict:
    """
    Initialize log dictionary for evaluation metrics.

    Tracks per eval step:
    - eval/IDLoss, eval/OODLoss, eval/MinorLoss: RMSE(model_pred, noisy_targets).
      Floor = task noise_scale (irreducible error). Same eval batches each step.
    - eval/ID_oracle, eval/OOD_oracle, eval/Minor_oracle: RMSE(model_pred, clean x@w).
    Per-position MSE under ``*_per_pos``; baseline comparisons in ``eval/baseline``.
    """
    log = {
        "train/step": [], "train/lr": [], "train/loss": [],
        "eval/IDLoss": [], "eval/OODLoss": [], "eval/MinorLoss": [],
        "eval/IDLoss_per_pos": [], "eval/OODLoss_per_pos": [], "eval/MinorLoss_per_pos": [],
        "eval/ID_oracle": [], "eval/OOD_oracle": [], "eval/Minor_oracle": [],
        "eval/baseline": {},
    }
    for _task_name, _task_preds in bsln_preds.items():
        for _bsln_name, _bsln_preds in _task_preds.items():
            if _bsln_name != "True":
                _errs = mse(_bsln_preds, _task_preds["True"]) / n_dims
                log["eval/baseline"][f"{_task_name}/{_bsln_name}"] = _errs.tolist()
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
    # Use canonical device for exp_name and saved config so hashes match get_exp_name()
    # (device is runtime-only; cuda:0 vs cuda:1 must not change experiment identity)
    runtime_device = config.device
    canonicalize_config_for_exp(config)
    exp_name = f"train_{get_hash(config)}"
    exp_dir = os.path.join(config.work_dir, exp_name)

    cur_dir = os.getcwd()
    if cur_dir.endswith("notebooks"):
        exp_dir = os.path.join("..", exp_dir)
    
    logger.debug(f"Experiment directory: {exp_dir}")

    data_type = getattr(torch, config.dtype)

    # Skip if already completed
    log_path = os.path.join(exp_dir, "log.json")
    if os.path.exists(log_path):
        logger.debug(f"{exp_name} already completed")
        checkpoint_path = os.path.join(exp_dir, "checkpoint.pt")
        log_path = os.path.join(exp_dir, "log.json")
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
        model = get_model(**config["model"], dtype=data_type)
        model.load_state_dict(checkpoint["model"])
        model = model.to(config.device)
        logger.debug(f"Loaded model from {checkpoint_path}")
        return model, (json.load(open(log_path, "r")), checkpoint_path)
    
    # Save config (with canonical device so downstream get_exp_name / load match)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        f.write(config.to_json())
    config.device = runtime_device  # restore for training on the requested GPU

    # Model, optimizer, schedule (pass device so model is built on correct GPU for parallel training)
    model = get_model(**config["model"], dtype=data_type, device=config.device)
    model = model.to(config.device)
    if verbose:
        logger.info(tabulate_model(model, config["task"]["n_dims"], config["task"]["n_points"], config["task"]["batch_size"]))

    optimizer, scheduler = get_optimizer_and_lr_schedule(**config.training, params=model.parameters())
    
    if verbose:
        logger.info("Initialized model, optimizer, and train state")

    # Data samplers
    train_task = get_task(**config["task"], dtype=data_type)
    sample_train_batch = get_sharded_batch_sampler(train_task)

    eval_tasks = train_task.get_default_eval_tasks(**config["eval"])
    samplers_eval = {
        get_task_name(task): get_sharded_batch_sampler(task, is_eval=False)
        for task in eval_tasks
    }
    # Minor-only sampler: reuse the Pretrain eval task but sample only from the minor pool
    has_minor = config["task"].get("n_minor_tasks", 0) > 0 and config["task"].get("n_tasks", 0) > 0
    if has_minor:
        pretrain_task = [t for t in eval_tasks if t.n_tasks > 0][0]
        samplers_eval["Minor"] = get_sharded_batch_sampler(pretrain_task, minor_only=True)

    if verbose:
        logger.info("Initialized data samplers")

    # Evaluate baselines (same eval samplers + seeds used later for get_model_preds at each eval step)
    if verbose:
        logger.info("Evaluating baselines...")
    bsln_preds = get_bsln_preds(train_task, samplers_eval, config["eval"]["n_samples"], config["eval"]["batch_size"])

    # Sanity: oracle vs noisy targets should have RMSE ≈ noise_scale (irreducible error)
    noise_scale = config["task"]["noise_scale"]
    for _name, _key in [("OOD", "Latent"), ("ID", "Pretrain")]:
        if _key not in bsln_preds:
            continue
        true_ = bsln_preds[_key]["True"]
        tgt_ = bsln_preds[_key]["targets"]
        baseline_rmse = mse(true_, tgt_).mean().item() ** 0.5
        if abs(baseline_rmse - noise_scale) > 0.15:
            logger.warning(
                f"eval/{_name}: RMSE(oracle, targets)={baseline_rmse:.4f} (expected ≈ noise_scale={noise_scale}); "
                "check that 'True' and 'targets' are clean and noisy from the same batches."
            )

    # Logging
    n_dims = config["task"]["n_dims"]
    log = _init_log(bsln_preds, n_dims)
    wandb_name = generate_wandb_run_name(config, exp_name)
    wandb.init(config=config, name=wandb_name, **config["wandb"])
    step = 0

    # Oracle RMSE
    ood_oracle_rmse = mse(bsln_preds["Latent"]["True"], bsln_preds["Latent"]["targets"]).mean().item() ** 0.5
    id_oracle_rmse = None
    if "Pretrain" in bsln_preds:
        id_oracle_rmse = mse(bsln_preds["Pretrain"]["True"], bsln_preds["Pretrain"]["targets"]).mean().item() ** 0.5
    log["eval/ID_oracle_baseline"] = id_oracle_rmse
    log["eval/OOD_oracle_baseline"] = ood_oracle_rmse
    wandb.log({"eval/OOD_oracle_baseline": ood_oracle_rmse}, step=0)
    if id_oracle_rmse is not None:
        logger.info(f"Oracle RMSE — ID: {id_oracle_rmse:.4f}, OOD: {ood_oracle_rmse:.4f}")
        wandb.log({"eval/ID_oracle_baseline": id_oracle_rmse}, step=0)
    else:
        logger.info(f"Oracle RMSE — OOD: {ood_oracle_rmse:.4f}")

    # Batch size schedule: list of (step, batch_size) transitions
    bs_schedule = config["training"].get("batch_size_schedule", None)
    if bs_schedule:
        bs_schedule = sorted(bs_schedule, key=lambda x: x[0])
        logger.info(f"Batch size schedule: {[(s, b) for s, b in bs_schedule]}")

    # p_minor schedule: list of (step, p_minor) transitions
    pm_schedule = config["training"].get("p_minor_schedule", None)
    if pm_schedule:
        pm_schedule = sorted(pm_schedule, key=lambda x: x[0])
        logger.info(f"p_minor schedule: {[(s, p) for s, p in pm_schedule]}")

    def _get_scheduled_value(schedule, default, step):
        if not schedule:
            return None
        current = default
        for s, v in schedule:
            if step >= s:
                current = v
            else:
                break
        return current

    # Training loop
    logger.info("Start training...")
    for i in range(1, config["training"]["total_steps"] + 1):
        step += 1
        scheduled_bs = _get_scheduled_value(bs_schedule, config["task"]["batch_size"], i)
        if scheduled_bs is not None and train_task.batch_size != scheduled_bs:
            logger.info(f"Step {i}: batch_size changed {train_task.batch_size} -> {scheduled_bs}")
            train_task.batch_size = scheduled_bs
        scheduled_pm = _get_scheduled_value(pm_schedule, config["task"].get("p_minor", 0.0), i)
        if scheduled_pm is not None and train_task.p_minor != scheduled_pm:
            logger.info(f"Step {i}: p_minor changed {train_task.p_minor} -> {scheduled_pm}")
            train_task.p_minor = scheduled_pm
        data, _, targets = sample_train_batch(i)
        data = data.to(config.device)
        targets = targets.to(config.device)
        model.train()
        optimizer.zero_grad()

        preds = model(data, targets)
        loss = torch.mean((preds - targets) ** 2)

        loss.backward()
        max_grad_norm = config["training"].get("max_grad_norm", None)
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        # Evaluation
        if i % config["eval"]["every"] == 0 or i == config["training"]["total_steps"]:
            log["train/step"].append(i)
            log["train/loss"].append(loss.item())
            lr_val = scheduler.get_last_lr()[0]
            log["train/lr"].append(lr_val)
            wandb.log({"train/lr": lr_val, "train/loss": loss.item()}, step=i)

            eval_preds = get_model_preds(
                model, eval_step, samplers_eval, config["eval"]["n_samples"], config["eval"]["batch_size"]
            )

            # IDLoss / OODLoss = RMSE(transformer predictions, noisy targets).
            # Best achievable RMSE = config["task"]["noise_scale"] (irreducible error).
            # Oracle metrics = RMSE(predictions, clean x@w) for the same batches (from bsln_preds).
            # Eval and baseline use the same deterministic eval samplers, so sample counts align.

            # OOD RMSE(transformer, noisy_targets) + oracle RMSE(transformer, x@w)
            ood_preds = eval_preds["Latent"]["Transformer"]
            ood_targets = eval_preds["Latent"]["targets"].to(config.device)
            ood_oracle = bsln_preds["Latent"]["True"].to(config.device)
            assert ood_preds.shape == ood_targets.shape == ood_oracle.shape, (
                "OOD shape mismatch: preds/targets/oracle must align (same eval sample count)."
            )
            ood_mse = mse(ood_preds, ood_targets)
            ood_rmse = ood_mse.mean().item() ** 0.5
            ood_oracle_mse_per_pos = mse(ood_preds, ood_oracle)
            ood_oracle_rmse = ood_oracle_mse_per_pos.mean().item() ** 0.5
            ood_oracle_rmse_skip0 = ood_oracle_mse_per_pos[1:].mean().item() ** 0.5
            ood_oracle_rmse_skip5 = ood_oracle_mse_per_pos[5:].mean().item() ** 0.5
            ood_oracle_rmse_skip20 = ood_oracle_mse_per_pos[20:].mean().item() ** 0.5
            ood_oracle_rmse_last15 = ood_oracle_mse_per_pos[-15:].mean().item() ** 0.5
            log["eval/OODLoss"].append(ood_rmse)
            log["eval/OODLoss_per_pos"].append(ood_mse.sqrt().tolist())
            log["eval/OOD_oracle"].append(ood_oracle_rmse)
            wandb.log({"eval/OODLoss": ood_rmse, "eval/OOD_oracle": ood_oracle_rmse}, step=i)

            # ID RMSE(transformer, noisy_targets) + oracle — only when discrete task pool exists
            has_id = "Pretrain" in eval_preds
            if has_id:
                id_preds = eval_preds["Pretrain"]["Transformer"]
                id_targets = eval_preds["Pretrain"]["targets"].to(config.device)
                id_oracle = bsln_preds["Pretrain"]["True"].to(config.device)
                assert id_preds.shape == id_targets.shape == id_oracle.shape, (
                    "ID shape mismatch: preds/targets/oracle must align (same eval sample count)."
                )
                id_mse = mse(id_preds, id_targets)
                id_rmse = id_mse.mean().item() ** 0.5
                id_oracle_mse_per_pos = mse(id_preds, id_oracle)
                id_oracle_rmse = id_oracle_mse_per_pos.mean().item() ** 0.5
                id_oracle_rmse_skip0 = id_oracle_mse_per_pos[1:].mean().item() ** 0.5
                id_oracle_rmse_skip5 = id_oracle_mse_per_pos[5:].mean().item() ** 0.5
                id_oracle_rmse_skip20 = id_oracle_mse_per_pos[20:].mean().item() ** 0.5
                id_oracle_rmse_last15 = id_oracle_mse_per_pos[-15:].mean().item() ** 0.5
                log["eval/IDLoss"].append(id_rmse)
                log["eval/IDLoss_per_pos"].append(id_mse.sqrt().tolist())
                log["eval/ID_oracle"].append(id_oracle_rmse)
                wandb.log({"eval/IDLoss": id_rmse, "eval/ID_oracle": id_oracle_rmse}, step=i)

            # Minor RMSE — only when minor pool exists
            has_minor_eval = "Minor" in eval_preds
            if has_minor_eval:
                minor_preds = eval_preds["Minor"]["Transformer"]
                minor_targets = eval_preds["Minor"]["targets"].to(config.device)
                minor_oracle = bsln_preds["Minor"]["True"].to(config.device)
                assert minor_preds.shape == minor_targets.shape == minor_oracle.shape, (
                    "Minor shape mismatch: preds/targets/oracle must align."
                )
                minor_mse = mse(minor_preds, minor_targets)
                minor_rmse = minor_mse.mean().item() ** 0.5
                minor_oracle_mse_per_pos = mse(minor_preds, minor_oracle)
                minor_oracle_rmse = minor_oracle_mse_per_pos.mean().item() ** 0.5
                minor_oracle_rmse_skip0 = minor_oracle_mse_per_pos[1:].mean().item() ** 0.5
                minor_oracle_rmse_skip5 = minor_oracle_mse_per_pos[5:].mean().item() ** 0.5
                minor_oracle_rmse_skip20 = minor_oracle_mse_per_pos[20:].mean().item() ** 0.5
                minor_oracle_rmse_last15 = minor_oracle_mse_per_pos[-15:].mean().item() ** 0.5
                log["eval/MinorLoss"].append(minor_rmse)
                log["eval/MinorLoss_per_pos"].append(minor_mse.sqrt().tolist())
                log["eval/Minor_oracle"].append(minor_oracle_rmse)
                wandb.log({"eval/MinorLoss": minor_rmse, "eval/Minor_oracle": minor_oracle_rmse}, step=i)

            # Baseline comparisons (verbose only)
            if verbose:
                for task_name in bsln_preds:
                    for bsln_name in bsln_preds[task_name]:
                        if bsln_name == "True":
                            continue
                        bsln_target = bsln_preds[task_name][bsln_name].to(config.device)
                        errs = mse(eval_preds[task_name]["Transformer"], bsln_target) / n_dims
                        wandb.log({f"eval/{task_name}/vs_{bsln_name}": errs.mean().item()}, step=i)

            log_every = config["eval"].get("log_every", 800)
            if i % log_every == 0 or i == config["training"]["total_steps"]:
                msg = f"Step {i}: train_loss={loss.item():.4f}"
                if has_id:
                    msg += f", ID={id_rmse:.4f}|{id_oracle_rmse:.4f}|{id_oracle_rmse_skip0:.4f}|{id_oracle_rmse_skip5:.4f}|{id_oracle_rmse_skip20:.4f}|{id_oracle_rmse_last15:.4f}"
                if has_minor_eval:
                    msg += f", Minor={minor_rmse:.4f}|{minor_oracle_rmse:.4f}|{minor_oracle_rmse_skip0:.4f}|{minor_oracle_rmse_skip5:.4f}|{minor_oracle_rmse_skip20:.4f}|{minor_oracle_rmse_last15:.4f}"
                msg += f", OOD={ood_rmse:.4f}|{ood_oracle_rmse:.4f}|{ood_oracle_rmse_skip0:.4f}|{ood_oracle_rmse_skip5:.4f}|{ood_oracle_rmse_skip20:.4f}|{ood_oracle_rmse_last15:.4f}"
                logger.info(msg)
        
        _save_every = config["eval"].get("save_every", 1000)
        _sparse_after    = 10_000   # hardcoded — not in config (would break hashing)
        _sparse_interval = 1_000
        _do_save = (i % _save_every == 0 and i <= _sparse_after) or \
                   (i > _sparse_after and i % _sparse_interval == 0)
        if _do_save:
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

    logger.info("Training complete.")
    try:
        wandb.finish()
    except Exception:
        pass
    return model, log