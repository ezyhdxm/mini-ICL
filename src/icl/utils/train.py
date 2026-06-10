import json
import math
import os
import pickle
import timeit
from collections import defaultdict
from contextlib import contextmanager, nullcontext

import nvtx
import torch
import torch.nn as nn
import wandb

from icl.coin.coin import Coins
from icl.dyck.dyck import DyckPathTask
from icl.figures.plot import get_loss_plots
from icl.latent_markov import LatentMarkov
from icl.models.attention import MultiHeadAttention
from icl.utils.logger import setup_logger

from .basic import get_config_hash_for_exp
from .train_utils import get_attn_base, get_train_result, tabulate_model

logger = setup_logger(__name__)


# use for profiling
@contextmanager
def maybe_nvtx_range(message, color="blue", enabled=True):
    ctx = nvtx.annotate(message, color=color) if enabled else nullcontext()
    with ctx:
        if enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = timeit.default_timer()
        yield
        if enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = timeit.default_timer()
            logger.debug(f"{message}: {end_time - start_time:.6f} s")
            
# turn off flash attention to obtain attention scores
def flash_off(model):
    for m in model.modules():
        if isinstance(m, MultiHeadAttention):
            m.flash = False
# turn on flash attention for faster computation
def flash_on(model):
    for m in model.modules():
        if isinstance(m, MultiHeadAttention):
            m.flash = True



def _init_log() -> dict:
    """
    Initialize log dictionary for evaluation metrics.
    """
    log = {"train/step": [], "train/lr": [], "train/loss": [], 
           "baseline": {},
           "eval/loss": [], "eval/step": [], 
           "eval/IDLoss": [], "eval/ICLLoss": [], "eval/OODLoss": [], "eval/CopyError": [],
           "eval/pth_score": [], "eval/ih_score": [], "eval/IDAcc": [], "eval/OODAcc": [],
           "eval/LengthLoss": [], "eval/LengthAcc": []}
    return log


# Checkpoint schedule thresholds — hardcoded here (not in config) so that
# changing these does not affect experiment hashing.
_CKPT_SPARSE_AFTER    = 10_000   # steps before this: use config interval
_CKPT_SPARSE_INTERVAL = 1_000    # steps from this point: save every N steps


def _should_checkpoint(step: int, interval: int) -> bool:
    """
    Return True if a checkpoint should be saved at this step.

    Phase 1 — dense early (step < min(interval, 200)):  every 5 steps
    Phase 2 — regular    (step <= _CKPT_SPARSE_AFTER):  every `interval` steps
    Phase 3 — sparse     (step >  _CKPT_SPARSE_AFTER):  every _CKPT_SPARSE_INTERVAL steps
    """
    if step < min(interval, 200) and step % 5 == 0:
        return True
    if step % interval == 0 and step <= _CKPT_SPARSE_AFTER:
        return True
    if step > _CKPT_SPARSE_AFTER and step % _CKPT_SPARSE_INTERVAL == 0:
        return True
    return False


class BaseTrainer: 
    def __init__(self, config):
        self.config = config
        self.mixed_precision = config.mixed_precision if hasattr(config, "mixed_precision") else True
        self.profile = config.profile if hasattr(config, "profile") else False
        # Non-mutating: compute identity hash on a canonicalized copy so the
        # runtime config.device (e.g. "cpu") is preserved while exp_name stays
        # identical to get_exp_name's.
        self.exp_name = f"train_{get_config_hash_for_exp(config)}"
        self.exp_dir = os.path.join(config.work_dir, self.exp_name)
        cur_dir = os.getcwd()
        if cur_dir.endswith("notebooks"):
            self.exp_dir = os.path.join("..", self.exp_dir)
        logger.info(f"Train Experiment\nNAME: {self.exp_name}\nCONFIG:\n{config}")
        self.MAX_SIZE = 1024
        self.log_path = os.path.join(self.exp_dir, "log.json")
        if os.path.exists(self.log_path):
            logger.info(f"{self.exp_name} already completed")
            return
        os.makedirs(self.exp_dir, exist_ok=True)
        with open(os.path.join(self.exp_dir, "config.json"), "w") as f: 
            f.write(config.to_json())
        self.log = _init_log()
        self.checkpoint_path = os.path.join(self.exp_dir, f"checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.attn_maps, self.probes = {}, defaultdict(list)
        self.criterion = nn.CrossEntropyLoss()
        self.step = 0

    def info_process(self, info):
        return None
    
    def get_task_loss(self, outputs, targets, info):
        return self.criterion(outputs, targets).item()
    
    def log_eval(self, model, data, infos):
        step = self.step
        with torch.no_grad():
            model.eval()
            outputs = model(data["test"])
            target = data["test"][:, 1:].reshape(-1)
            outputs = outputs[:, :-1, :].reshape(-1, self.config.vocab_size)
            
            eval_loss = self.criterion(outputs, target)
            self.log["eval/loss"].append(eval_loss.item())
            wandb.log({"eval/loss": eval_loss.item()}, step=step)
            eval_task_loss = self.get_task_loss(outputs, target, infos["test"]) 
            self.log["eval/IDLoss"].append(eval_task_loss)
            wandb.log({"eval/IDLoss": eval_task_loss}, step=step)
            self.log["eval/step"].append(step)
            if self.config.task.ood:
                ood_outputs = model(data["ood"])
                ood_outputs = ood_outputs[:, :-1, :].reshape(-1, self.config.vocab_size)
                ood_target = data["ood"][:, 1:].reshape(-1)
                ood_loss = self.get_task_loss(ood_outputs, ood_target, infos["ood"])
                self.log["eval/OODLoss"].append(ood_loss)
                wandb.log({"eval/OODLoss": ood_loss}, step=step)

    def save_checkpoint(self, model, optimizer, is_final=False):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        step = self.step
        if is_final:
            torch.save({
                "model": model.state_dict(), 
                "optimizer": optimizer.state_dict(),
                "step": step,
                }, os.path.join(self.checkpoint_path, f"model_final_{step}.pt"))
        else:
            torch.save({
                "model": model.state_dict(), 
                "optimizer": optimizer.state_dict(),
                "step": step,
                }, os.path.join(self.checkpoint_path, f"model_{step}.pt"))

    def train(self, model, verbose=False):
        sampler = get_sampler(self.config)
        if verbose:
            logger.info(f"Model architecture:\n{tabulate_model(model, self.config.seq_len, self.config.batch_size, self.config.device)}")

        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=self.config.training.learning_rate, 
                                    weight_decay=self.config.training.weight_decay)  # torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        grad_clip_norm = getattr(self.config.training, "grad_clip_norm", None)
        
        if self.config.training.scheduler is True:
            max_lr = float(self.config.training.learning_rate)
            min_lr = float(getattr(self.config.training, "min_learning_rate", 1e-5))
            warmup_steps = int(getattr(self.config.training, "warmup_steps", 0))
            total_steps = int(self.config.training.num_epochs)
            scheduler_type = str(getattr(self.config.training, "scheduler_type", "triangle")).lower()

            def _triangle_lr_lambda(step_idx):
                if warmup_steps > 0 and step_idx < warmup_steps:
                    return (min_lr + (max_lr - min_lr) * step_idx / warmup_steps) / max_lr
                decay_steps = max(1, total_steps - warmup_steps)
                prog = min(max((step_idx - warmup_steps) / decay_steps, 0.0), 1.0)
                return (max_lr - (max_lr - min_lr) * prog) / max_lr

            def _cosine_warmup_lr_lambda(step_idx):
                if warmup_steps > 0 and step_idx < warmup_steps:
                    return (min_lr + (max_lr - min_lr) * step_idx / warmup_steps) / max_lr
                decay_steps = max(1, total_steps - warmup_steps)
                prog = min(max((step_idx - warmup_steps) / decay_steps, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
                lr = min_lr + (max_lr - min_lr) * cosine
                return lr / max_lr

            if scheduler_type in ("cosine_warmup", "warmup_cosine", "cosine"):
                lr_lambda = _cosine_warmup_lr_lambda
            else:
                # Backward-compatible default behavior.
                lr_lambda = _triangle_lr_lambda

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            scheduler = None
        
        #scheduler = CosineAnnealingLR(optimizer, T_max=self.config.training.T_max) if self.config.training.scheduler is True else None

        data = {"test": None, "ood": None,}
        infos = {"test": None, "ood": None,}
        scaler = torch.amp.GradScaler('cuda') if self.mixed_precision else None

        data["test"], infos["test"] = sampler.generate(mode="test")
        infos["test"] = self.info_process(infos["test"])
            

        if self.config.task.ood:
            data["ood"], infos["ood"] = sampler.generate(mode="ood")
            infos["ood"] = self.info_process(infos["ood"])
                
        # eval_batch, eval_info = sampler.generate(mode="eval")
        # eval_info = self.info_process(eval_info)

        epochs = min(self.config.training.num_epochs, self.MAX_SIZE)
        while self.config.training.num_epochs % epochs != 0: epochs -= 1

        tot_iters = self.config.training.num_epochs // epochs

        wandb.init(config=self.config, name=self.exp_name, **self.config["wandb"])

        if verbose: logger.info("Starting training...")
        for iters in range(tot_iters): 
            with maybe_nvtx_range(f"Generate Training Samples {iters}", color="green", enabled=self.config.profile):
                train_data = sampler.generate(epochs=epochs)
            
            sample, sample_info = train_data
            for i in range(epochs): 
                if self.profile:
                    logger.debug("="*50)
                self.step += 1
                model.train()
                batch, _ = sample[i], sample_info[i]
                optimizer.zero_grad()

                if (self.config.training.get_attn) > 0 and (self.step % self.config.training.get_attn == 0): 
                    flash_off(model)
                    self.attn_maps[self.step] = get_attn_base(model, batch)
                    flash_on(model)

                with maybe_nvtx_range(f"Forward Pass {iters}:{i}", color="blue", enabled=self.config.profile):
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if self.mixed_precision else nullcontext():
                        outputs = model(batch)
                        outputs = outputs[:, :-1, :].reshape(-1, self.config.vocab_size)
                        targets = batch[:, 1:].reshape(-1)
                        loss = self.criterion(outputs, targets)
                        

                self.log["train/loss"].append(loss.item())
                wandb.log({"train/loss": loss.item()}, step=self.step)

                with maybe_nvtx_range(f"Backward Pass {iters}:{i}", color="red", enabled=self.config.profile):
                    if self.mixed_precision:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                if grad_clip_norm is not None and float(grad_clip_norm) > 0:
                    # For AMP we must unscale before clipping gradients.
                    if self.mixed_precision:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))

                with maybe_nvtx_range(f"Optimizer Step {iters}:{i}", color="orange", enabled=self.config.profile):
                    if self.mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                if scheduler: 
                    with maybe_nvtx_range(f"Scheduler Step {iters}:{i}", color="yellow", enabled=self.config.profile):
                        scheduler.step()

                if self.config.training.get_checkpoints > 0 and _should_checkpoint(
                        self.step, self.config.training.get_checkpoints):
                    self.save_checkpoint(model, optimizer, is_final=False)


                if (self.step % self.config.training.eval_iter == 0) or (self.step < min(self.config.training.eval_iter, 100) and self.step % 5 == 0):
                    if verbose: 
                        logger.info(f"Step: {self.step}")
                    self.log["train/step"].append(self.step)
                    lr_val = scheduler.get_last_lr()[0] if scheduler else self.config.training.learning_rate
                    self.log["train/lr"].append(lr_val)
                    wandb.log({"train/lr": lr_val}, step=self.step)
                    self.log_eval(model, data, infos)

        logger.info("Saving final model...")
        self.save_checkpoint(model, optimizer, is_final=True)
        with open(self.log_path, "w") as f:
            json.dump(self.log, f, indent=2)

        if verbose:
            logger.info("Training complete.")

        try:
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass

        return get_train_result(log=self.log, config=self.config, sampler=sampler, attn_maps=self.attn_maps, probes=self.probes)





def get_sampler(config):
    task_samplers = {
        "latent": LatentMarkov,
        "coin": Coins,
        "dyck": DyckPathTask,
    }
    if config.task.name in task_samplers: return task_samplers[config.task.name](config)
    raise NotImplementedError(f"Task '{config.task.name}' not implemented yet.")



# Train model based on task
def train_model(config):
    return BaseTrainer(config)



def train_model_with_plot(model, config, show=False, verbose=False):
    # Non-mutating identity hash (preserves runtime config.device).
    exp_name = f"train_{get_config_hash_for_exp(config)}"
    exp_dir = os.path.join(config.work_dir, exp_name)

    cur_dir = os.getcwd()
    if cur_dir.endswith("notebooks"):
        exp_dir = os.path.join("..", exp_dir)

    if verbose:
        logger.info(f"Experiment directory: {exp_dir}") 

    if os.path.exists(os.path.join(exp_dir, "log.json")):
        if verbose:
            logger.info(f"{exp_name} already completed")
        log_path = os.path.join(exp_dir, "log.json")
        return (json.load(open(log_path, "r")), exp_dir)
    
    trainer = train_model(config)

    with maybe_nvtx_range("Training"):
        train_results = trainer.train(model, verbose=verbose)
    
    plot_path = os.path.join(exp_dir, "plots")
    os.makedirs(plot_path, exist_ok=True)

    get_loss_plots(config, train_results, folder=plot_path, show=show)

    if len(train_results["attn_maps"]) > 0:
        last_key = sorted(list(train_results["attn_maps"].keys()))[-1]
        last_attn = train_results["attn_maps"][last_key]
        last_attn["steps"] = last_key
        train_results["attn_maps"] = last_attn

    result_file_name = os.path.join(exp_dir, "sampler.pkl")
    with open(result_file_name, "wb") as file:
        pickle.dump(train_results["sampler"], file)
    
    return train_results
    