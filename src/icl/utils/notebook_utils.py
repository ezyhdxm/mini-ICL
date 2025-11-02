import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from pprint import pprint
from itertools import product
from IPython.display import display, HTML
import os
import glob
import re
from ml_collections import ConfigDict
import json
import pickle
from datetime import datetime
import hashlib

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from . import linear_algebra_utils as lau
from .basic import get_hash
from icl.models import Transformer

def hash_array(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()


def get_all_trans_mat(sampler, return_all=False):
    """
    Helper function to get all transition matrices from a sampler.
    Works with LatentMarkov class that has major_trans_mat and minor_trans_mat.
    Returns: torch.Tensor of shape (total_trans, num_states_order, num_states)
    """
    if hasattr(sampler, 'major_trans_mat'):
        # New LatentMarkov structure
        if return_all:
            if sampler.n_major_tasks > 0 and sampler.n_minor_tasks > 0:
                return torch.cat([sampler.major_trans_mat, sampler.minor_trans_mat], dim=0)
            elif sampler.n_major_tasks > 0:
                return sampler.major_trans_mat
            elif sampler.n_minor_tasks > 0:
                return sampler.minor_trans_mat
            else:
                raise ValueError("No transition matrices available.")
        else:
            return sampler.major_trans_mat
    else:
        raise AttributeError("Sampler has no transition matrices.")



def extract_experiment_metadata(root_dir=os.path.join("results", "latent")):
    """
    Extracts experiment metadata from a list of filenames.
    
    Each filename is expected to encode the following parameters:
    - vocab_size: int
    - alpha: float
    - seq_len: int
    - hidden_dim: int
    - stationary: bool (represented as 'stationary' or 'nonstationary' in the filename)

    Returns:
        pd.DataFrame with columns:
        ['vocab_size', 'alpha', 'seq_len', 'hidden_dim', 'stationary', 'filename']
    """
    records = []

    for subdir in os.listdir(root_dir):
        path = os.path.join(root_dir, subdir)
        config_path = os.path.join(path, "config.json")

        if os.path.isdir(path) and os.path.isfile(config_path):
            try:
                with open(config_path, "r") as f:
                    config = ConfigDict(json.load(f))
                record = {
                    "vocab_size": config.vocab_size,
                    "alpha": config.task.alpha,
                    "total_transitions": config.task.total_trans,
                    "seq_len": config.seq_len,
                    "hidden_dim": config.model.emb_dim,
                    "stationary": config.task.get("stationary"),
                    "filename": subdir
                }
                records.append(record)
            except Exception as e:
                print(f"Failed to read {config_path}: {e}")
    
    return pd.DataFrame(records)


#################
# Load model
#################

def load_model(checkpoint_dir, config, step=None):
    device = config.device
    
    # Extract step number from each filename
    def extract_step(path):
        match = re.search(r"model_final_(\d+)\.pt", path)
        return int(match.group(1)) if match else -1
    
    if step is not None:
        model_path = os.path.join(checkpoint_dir, f"model_{step}.pt")
    else:
        pattern = "model_final_*.pt"
        files = glob.glob(os.path.join(checkpoint_dir, pattern))
        if len(files) == 0:
            raise ValueError(f"No model found in {checkpoint_dir} with pattern {pattern}")

        paths = sorted(files, key=extract_step)
        model_path = paths[-1]
        
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = Transformer(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model.to(device)


def list_checkpoints(checkpoint_dir):
    """
    List all available checkpoints in the checkpoint directory.
    
    Args:
        checkpoint_dir: Path to the checkpoints directory (e.g., "results/latent/train_xxx/checkpoints")
    
    Returns:
        dict with keys:
            - 'regular': list of tuples (step, filename) for regular checkpoints (model_{step}.pt)
            - 'final': list of tuples (step, filename) for final checkpoints (model_final_{step}.pt)
            - 'all_steps': sorted list of all step numbers available
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    # Find all checkpoint files
    regular_pattern = os.path.join(checkpoint_dir, "model_*.pt")
    final_pattern = os.path.join(checkpoint_dir, "model_final_*.pt")
    
    regular_files = glob.glob(regular_pattern)
    final_files = glob.glob(final_pattern)
    
    # Extract step numbers
    def extract_step_number(filename):
        """Extract step number from filename like 'model_100.pt' or 'model_final_100.pt'"""
        basename = os.path.basename(filename)
        # Match both model_{step}.pt and model_final_{step}.pt
        match = re.search(r"model(?:_final)?_(\d+)\.pt", basename)
        return int(match.group(1)) if match else None
    
    regular_checkpoints = []
    for f in regular_files:
        # Skip final checkpoints in regular list
        if "model_final_" not in os.path.basename(f):
            step = extract_step_number(f)
            if step is not None:
                regular_checkpoints.append((step, os.path.basename(f)))
    
    final_checkpoints = []
    for f in final_files:
        step = extract_step_number(f)
        if step is not None:
            final_checkpoints.append((step, os.path.basename(f)))
    
    # Sort by step number
    regular_checkpoints.sort(key=lambda x: x[0])
    final_checkpoints.sort(key=lambda x: x[0])
    
    # Get all unique step numbers
    all_steps = sorted(set([s for s, _ in regular_checkpoints] + [s for s, _ in final_checkpoints]))
    
    return {
        'regular': regular_checkpoints,
        'final': final_checkpoints,
        'all_steps': all_steps
    }


def load_checkpoint(config, step=None, checkpoint_dir=None, use_final=True, verbose=True):
    """
    Load a model checkpoint from the checkpoint directory.
    
    More flexible than load_model - can load any checkpoint by step number,
    and provides better error messages and checkpoint listing.
    
    Args:
        config: Configuration object (ConfigDict) used to initialize the model.
                If checkpoint_dir is None, the checkpoint directory will be automatically
                constructed from config.work_dir and config hash.
        step: Step number to load (None = load latest). Can be:
              - None: load latest checkpoint (final if use_final=True, otherwise regular)
              - int: specific step number
              - "latest": explicitly load latest
              - "final": load latest final checkpoint
        checkpoint_dir: Optional path to the checkpoints directory.
                        If None, will be auto-constructed from config.
                        (e.g., "results/latent/train_xxx/checkpoints")
        use_final: If True and step=None, prefer final checkpoints over regular ones
        verbose: If True, print information about loaded checkpoint
    
    Returns:
        model: Loaded Transformer model in eval mode, moved to config.device
    """
    # Auto-construct checkpoint_dir from config if not provided
    if checkpoint_dir is None:
        exp_name = f"train_{get_hash(config)}"
        exp_dir = os.path.join(config.work_dir, exp_name)
        
        # Handle notebooks directory case
        cur_dir = os.getcwd()
        if cur_dir.endswith("notebooks"):
            exp_dir = os.path.join("..", exp_dir)
        
        checkpoint_dir = os.path.join(exp_dir, "checkpoints")
        
        if verbose:
            print(f"Auto-detected checkpoint directory: {checkpoint_dir}")
    
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    # List available checkpoints
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if len(checkpoints['all_steps']) == 0:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    device = config.device
    
    # Determine which checkpoint to load
    if step is None or step == "latest":
        if use_final and len(checkpoints['final']) > 0:
            # Load latest final checkpoint
            step_num, filename = checkpoints['final'][-1]
            model_path = os.path.join(checkpoint_dir, filename)
            if verbose:
                print(f"Loading latest final checkpoint: {filename} (step {step_num})")
        else:
            # Load latest regular checkpoint
            if len(checkpoints['regular']) == 0:
                if len(checkpoints['final']) > 0:
                    step_num, filename = checkpoints['final'][-1]
                    model_path = os.path.join(checkpoint_dir, filename)
                    if verbose:
                        print(f"Only final checkpoints available. Loading: {filename} (step {step_num})")
                else:
                    raise ValueError("No checkpoints found")
            else:
                step_num, filename = checkpoints['regular'][-1]
                model_path = os.path.join(checkpoint_dir, filename)
                if verbose:
                    print(f"Loading latest checkpoint: {filename} (step {step_num})")
    
    elif step == "final":
        if len(checkpoints['final']) == 0:
            raise ValueError("No final checkpoints found")
        step_num, filename = checkpoints['final'][-1]
        model_path = os.path.join(checkpoint_dir, filename)
        if verbose:
            print(f"Loading latest final checkpoint: {filename} (step {step_num})")
    
    elif isinstance(step, int):
        # Try to find checkpoint with this step number
        model_path = None
        step_num = step
        
        # First try exact match in regular checkpoints
        for s, fname in checkpoints['regular']:
            if s == step:
                model_path = os.path.join(checkpoint_dir, fname)
                step_num = s
                if verbose:
                    print(f"Loading checkpoint: {fname} (step {step})")
                break
        
        # If not found, try exact match in final checkpoints
        if model_path is None:
            for s, fname in checkpoints['final']:
                if s == step:
                    model_path = os.path.join(checkpoint_dir, fname)
                    step_num = s
                    if verbose:
                        print(f"Loading final checkpoint: {fname} (step {step})")
                    break
        
        # If still not found, try direct file paths
        if model_path is None:
            # Try regular format
            candidate = os.path.join(checkpoint_dir, f"model_{step}.pt")
            if os.path.exists(candidate):
                model_path = candidate
                step_num = step
                if verbose:
                    print(f"Loading checkpoint: model_{step}.pt (step {step})")
            else:
                # Try final format
                candidate = os.path.join(checkpoint_dir, f"model_final_{step}.pt")
                if os.path.exists(candidate):
                    model_path = candidate
                    step_num = step
                    if verbose:
                        print(f"Loading final checkpoint: model_final_{step}.pt (step {step})")
        
        # If still not found, find the closest step
        if model_path is None or (model_path is not None and not os.path.exists(model_path)):
            all_checkpoints = checkpoints['regular'] + checkpoints['final']
            if len(all_checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {checkpoint_dir}")
            
            # Find the checkpoint with the closest step number
            closest_step, closest_filename = min(all_checkpoints, key=lambda x: abs(x[0] - step))
            model_path = os.path.join(checkpoint_dir, closest_filename)
            step_num = closest_step
            
            if verbose:
                diff = abs(closest_step - step)
                print(f"Step {step} not found. Loading closest checkpoint: {closest_filename} (step {closest_step}, diff={diff})")
    
    else:
        raise ValueError(f"Invalid step parameter: {step}. Must be None, int, 'latest', or 'final'")
    
    # Load the checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {model_path}: {e}")
    
    # Initialize model and load state
    try:
        model = Transformer(config)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model = model.to(device)
        
        if verbose and 'step' in checkpoint:
            print(f"Checkpoint info: step={checkpoint.get('step', 'N/A')}")
        
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model from checkpoint: {e}")

####################
# Load Config
####################

def load_config(config_path):
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    config = ConfigDict(config_dict)
    return config


# If you do not know the exact folder name, 
# you can use this function to get the list of folders 
# with the same number of transition matrices.

def get_config(total_trans=None, vocab_size=10, path=None, 
               seq_len=None, hidden_dim=None, alpha=None, 
               stationary=None):
    DEFAULT_PATH = os.path.join("results", "latent")
    if path is None:
        path = DEFAULT_PATH
    if total_trans is None:
        total_trans_list = []
        
        for folder in os.listdir(path):
            if os.path.isdir(os.path.join(path, folder)):
                config = load_config(os.path.join(path, folder, "config.json"))
                flag = config.vocab_size == vocab_size
                if seq_len is not None:
                    flag = flag and (config.seq_len == seq_len)
                if hidden_dim is not None:
                    flag = flag and (config.model.emb_dim == hidden_dim)
                if alpha is not None:
                    flag = flag and (config.task.alpha == alpha)
                if stationary is not None:
                    if "stationary" in config.task:
                        flag = flag and (config.task.stationary == stationary)
                    else:
                        flag = flag and (stationary is False)
                if flag:
                    total_trans_list.append(config.task.total_trans)
        pprint(f"You can choose from the following number of total transition matrices: \n {sorted(total_trans_list)}")
    else:
        folder_list = []
        for folder in os.listdir(path):
            if os.path.isdir(os.path.join(path, folder)):
                config = load_config(os.path.join(path, folder, "config.json"))
                flag = (config.task.total_trans == total_trans) and (config.vocab_size == vocab_size)
                if seq_len is not None:
                    flag = flag and (config.seq_len == seq_len)
                if hidden_dim is not None:
                    flag = flag and (config.model.emb_dim == hidden_dim)
                if alpha is not None:
                    flag = flag and (config.task.alpha == alpha)
                if stationary is not None:
                    if "stationary" in config.task:
                        flag = flag and (config.task.stationary == stationary)
                    else:
                        flag = flag and (stationary is False)
                if flag:
                    folder_list.append(folder)
        pprint(f"You can choose from the following folders: {folder_list}")
        
        return folder_list


####################
# Load Log
####################

def load_log(log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r") as f:
        log_data = json.load(f)
    return log_data

#####################
# Load Sampler
#####################

def load_sampler(sampler_path):
    sampler = pickle.load(open(sampler_path, "rb"))
    return sampler


def load_everything(task_name, train_folder, get_log=False):
    curr_dir = os.getcwd()
    if curr_dir.endswith("notebooks"):
        path_prefix = os.path.join("..", "results", task_name)
    else:
        path_prefix = os.path.join("results", task_name)
    train_folder = train_folder
    checkpoint_dir = os.path.join(path_prefix, train_folder, "checkpoints")
    config_path = os.path.join(path_prefix, train_folder, "config.json")
    sampler_path = os.path.join(path_prefix, train_folder, "sampler.pkl")
    config = load_config(config_path)
    model = load_model(checkpoint_dir, config)
    sampler = load_sampler(sampler_path)
    if get_log:
        log_path = os.path.join(path_prefix, train_folder, "log.json")
        log_data = load_log(log_path)
        return model, sampler, config, log_data
    return model, sampler, config



def get_cos_sim_plot(x: torch.Tensor):
    x_normalized = x / x.norm(dim=1, keepdim=True)

    # Step 2: Compute cosine similarity matrix
    cos_sim = x_normalized @ x_normalized.T  # shape: (64, 64)

    # Step 3: Visualize
    plt.figure(figsize=(6, 5))
    sns.heatmap(cos_sim.cpu().numpy(), cmap='coolwarm', center=0, square=True)
    plt.title("Cosine Similarity Matrix")
    plt.xlabel("Index")
    plt.ylabel("Index")
    plt.tight_layout()
    plt.show()




def lighten(color, amount=0.5):
    """Blend color with white."""
    white = torch.tensor([1.0, 1.0, 1.0])
    base = torch.tensor(color[:3])
    blended = base + (white - base) * amount
    return (*blended.tolist(), color[3] if len(color) > 3 else 1.0)

def get_pos_loss(model, sampler, mode, folder, n_sumples=1):
    cmap = cm.get_cmap('tab10')  # or 'Set1', 'tab20', etc.

    NUM_SAMPLES = n_sumples

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    vocab_size = sampler.num_states
    
    # Generate samples without return_triggers
    batch, trans_random = sampler.generate(num_samples=NUM_SAMPLES, mode=mode)
    
    logits = model(batch)[0]
    preds = F.softmax(logits, dim=-1)

    losses = F.kl_div(preds.log(), trans_random, reduction="none").sum(dim=-1)
    
    seqs = torch.arange(batch.size(1))

    for b in range(NUM_SAMPLES):
        # All positions are valid (no masking)
        valid_indices = torch.arange(batch.size(1) - 1)
        target = batch[b][1:]
        emp_counts = torch.ones(vocab_size)
        emp_losses = torch.zeros_like(valid_indices, dtype=torch.float32)
        for i in range(len(valid_indices)):
            emp_probs = emp_counts / emp_counts.sum()
            emp_losses[i] = F.kl_div(emp_probs.log(), trans_random[b].cpu(), reduction="none").sum()
            token = target[i]
            emp_counts[token] += 1.0
        
        base_color = cmap((2*b)%10)
        model_color = lighten(base_color, amount=0)
        empirical_color = lighten(base_color, amount=0.4)

        plt.plot(seqs[valid_indices], losses[b][valid_indices].detach().cpu(), marker='o', linestyle='-', 
                 label=f"Model {b}", markersize=3, alpha=0.6, color=model_color)
        
        plt.plot(seqs[valid_indices], emp_losses.detach().cpu(), marker='x', linestyle='dashdot', 
                 label=f"Empirical {b}", markersize=3, alpha=1, color=empirical_color)
    
    plt.title("KL Divergence over Positions")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    image_path = os.path.join(folder, f"loss_over_pos_{mode}_{timestamp}.png")
    plt.savefig(image_path)
    plt.show()
    

    return losses.detach().cpu()



#######################
# Latent Markov Chain #
#######################

def get_empirical_transition(model, sampler, task, pos=400, num_samples=1024):
    assert task < sampler.total_trans, "task id out of range"
    assert pos < sampler.seq_len, "position out of range"

    model.eval()
    device = sampler.device

    if num_samples > 1:
        trans_mat_est = torch.zeros((sampler.num_states_order, sampler.num_states), device=device)
        batch, prob = sampler.generate(num_samples=num_samples, mode="testing", task=task)
        logits, _ = model(batch)
        preds = torch.softmax(logits, dim=-1)
        probs = preds[:, pos] # (B, N)
        states = batch[:, (pos-sampler.order+1):(pos+1)] # (B, O)
        states_indices = torch.sum(states * sampler.powers, dim=1)  # (B,)
        trans_mat_est = trans_mat_est.scatter_add(0, states_indices.unsqueeze(1).expand(-1, sampler.num_states), probs)
        counts = torch.bincount(states_indices, minlength=sampler.num_states_order).clamp(min=1)
        trans_mat_est /= counts.unsqueeze(1)  # Normalize by the counts
        return trans_mat_est.detach().cpu()

    else:
        perms = list(product(range(sampler.num_states), repeat=sampler.order))
        batch, prob = sampler.generate(num_samples=len(perms), mode="testing", task=task)
        perms = torch.tensor(perms, device=device)
        
        batch[:, (pos-sampler.order+1):(pos+1)] = perms 
        logits, _ = model(batch)
        preds = torch.softmax(logits, dim=-1)
        probs = preds[:, pos]
        
        return probs.detach().cpu()



def kl_div_ave(P: torch.Tensor, Q: torch.Tensor) -> float:
    """
    Compute the KL divergence between two transition matrices P and Q.
    Q is the true transition matrix, and P is the estimated transition matrix.
    P and Q should be 2D tensors of the same size.
    """
    assert P.size() == Q.size(), "P and Q must have the same size."
    P = P.to(Q.device)  # Ensure P is on the same device as Q
    mu = lau.get_stationary(Q)
    kl = F.kl_div(P.log(), Q, reduction="none").sum(dim=-1)
    return (kl * mu).sum(dim=-1).cpu().item() # Average over the rows


def compute_stationary_distributions(P_batch):
    """
    P_batch: Tensor of shape (M, n, n), batch of transition matrices
    Returns:
        pi_batch: Tensor of shape (M, n), batch of stationary distributions
    """
    M, n, _ = P_batch.shape
    pi_batch = []

    for m in range(M):
        P = P_batch[m]  # (n, n)
        # Transpose because we solve right eigenvector of P^T
        eigenvalues, eigenvectors = torch.linalg.eig(P.T)
        # Find the eigenvector corresponding to eigenvalue 1
        idx = torch.argmin(torch.abs(eigenvalues - 1))
        pi = eigenvectors[:, idx].real  # take real part, just in case
        pi = pi / pi.sum()  # normalize to sum to 1
        pi = torch.clamp(pi, min=0.0)   # Remove tiny negative values due to numerical error
        pi = pi / pi.sum()              # Normalize again after clamping
        pi_batch.append(pi)

    pi_batch = torch.stack(pi_batch, dim=0)
    return pi_batch

def pairwise_kl_divergence(P_batch, pi_batch=None):
    """
    P_batch: Tensor of shape (M, n, n), transition matrices
    pi_batch: Tensor of shape (M, n), stationary distributions
    Returns:
        KL matrix of shape (M, M) where (i,j) is KL(P[i] || P[j]) weighted by pi[i]
    """
    if pi_batch is None:
        pi_batch = compute_stationary_distributions(P_batch)

    M, n, _ = P_batch.shape
    log_P_batch = torch.log(P_batch + 1e-12)  # To avoid log(0)
    
    kl_matrix = torch.zeros(M, M)

    for i in range(M):
        for j in range(M):
            kl_per_row = F.kl_div(
                log_P_batch[i],  # log(P^{(i)}(i,j))
                P_batch[j],      # P^{(j)}(i,j)
                reduction='none'
            ).sum(dim=-1)  # sum over j for each i
            kl = (pi_batch[i] * kl_per_row).sum()  # weighted by pi^{(i)}(i)
            kl_matrix[i, j] = kl.item()
    
    return kl_matrix

def pairwise_kl_divergence_stationary(pi_batch, P_batch=None):
    """
    pi_batch: Tensor of shape (M, n), batch of stationary distributions
    Returns:
        KL matrix of shape (M, M) where (i,j) = KL(pi[i] || pi[j])
    """
    if pi_batch is None and P_batch is not None:
        pi_batch = compute_stationary_distributions(P_batch)
    M, n = pi_batch.shape
    log_pi = torch.log(pi_batch + 1e-12)  # shape (M, n), add epsilon for numerical stability

    kl_matrix = torch.zeros(M, M)

    for i in range(M):
        for j in range(M):
            kl = (pi_batch[i] * (log_pi[i] - log_pi[j])).sum()
            kl_matrix[i, j] = kl.item()

    return kl_matrix


##########################
# Phase Transition Plots #
##########################

def get_loss_lineplot(task_name, vocab_size=20, task_ids=None, 
                      alpha=1.0, seq_len=512, hidden_dim=64, stationary=False):
    
    folder_path = os.path.join("results", task_name)
    folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

    # Create a 1x2 subplot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # adjust figsize as needed

    # Define normalization and colormap

    plot_ids = []
    plot_paths = []

    for i in range(len(folders)):
        result_path = os.path.join(folder_path, folders[i])
        log_path = os.path.join(result_path, "log.json")
        config_path = os.path.join(result_path, "config.json")
        log_data = load_log(log_path)
        if log_data is None: continue
        config = load_config(config_path)

        if (task_ids is not None) and (config.task.total_trans not in task_ids): continue

        if config.vocab_size != vocab_size: continue

        if config.task.alpha != alpha: continue

        if config.seq_len != seq_len: continue

        if config.model.emb_dim != hidden_dim: continue

        if "stationary" in config.task:
            if config.task.stationary != stationary: continue
        elif stationary:
            continue
        
        plot_ids.append(config.task.total_trans)
        plot_paths.append(result_path)
    
    plot_ids = np.array(plot_ids)
    if task_ids is None:
        norm = mcolors.Normalize(vmin=np.log2(min(plot_ids)), vmax=np.log2(max(plot_ids)))
    else:
        norm = mcolors.Normalize(vmin=min(plot_ids), vmax=max(plot_ids))
    cmap = plt.get_cmap('plasma')
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for path in plot_paths:
        log_path = os.path.join(path, "log.json")
        config_path = os.path.join(path, "config.json")
        log_data = load_log(log_path)
        config = load_config(config_path)

        if task_ids is None:
            value = np.log2(config.task.total_trans)
        else:
            value = config.task.total_trans
        color = cmap(norm(value))
        
        # Plot 1: OOD Loss
        ax2.plot(log_data["eval/step"], log_data["eval/OODLoss"], color=color, alpha=0.6)
        
        # Plot 2: Some other metric (e.g., ID Loss)
        ax1.plot(log_data["eval/step"], log_data["eval/loss"], color=color, alpha=0.6)



    # Customize each subplot
    ax2.set_xscale("log")
    ax2.set_title("OOD Loss", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Training Steps", fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=11)

    ax1.set_xscale("log")
    ax1.set_title("ID Loss", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Training Steps", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=11)
    ax1.tick_params(axis='y', labelsize=11)

    plt.subplots_adjust(wspace=0.02)  # Try 0 for zero gap
    # Add colorbar to the whole figure
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='vertical')
    if task_ids is None:
        cbar.set_label("Log Number of Mixtures", fontsize=12)
    else:
        cbar.set_label("Number of Mixtures", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    if task_ids is None:
        plt.savefig(os.path.join(folder_path, f"loss_lineplots_{vocab_size}_{alpha}.png"))
    else:
        plt.savefig(os.path.join(folder_path, f"loss_lineplots_{hash_array(np.array(plot_ids))}_{vocab_size}_{alpha}.png"))
    plt.show()

# TODO: outdated!
def get_loss_heatmap_data(task_name, measure, task_ids=None):
    measure_name = {"ood": "OODLoss", "id": "loss", "ih": "ih_score", "pth": "pth_score"}
    folder_path = os.path.join("results", task_name)
    folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

    # Collect all steps and transitions first
    all_steps = set()
    all_trans = set()
    data_dict = {}

    for folder in folders:
        result_path = os.path.join(folder_path, folder)
        log_path = os.path.join(result_path, "log.json")
        config_path = os.path.join(result_path, "config.json")

        log_data = load_log(log_path)
        config = load_config(config_path)

        trans = config.task.total_trans
        if (task_ids is not None) and (trans not in task_ids):
            continue

        steps = log_data["eval/step"]
        losses = log_data[f"eval/{measure_name[measure]}"]  # or "eval/loss" for ID

        for step, loss in zip(steps, losses):
            all_steps.add(step)
            all_trans.add(trans)
            data_dict[(trans, step)] = loss

    # Sort axes
    sorted_steps = sorted(all_steps)
    sorted_trans = sorted(all_trans)

    # Create heatmap matrix
    heatmap = np.full((len(sorted_trans), len(sorted_steps)), np.nan)

    trans_idx = {v: i for i, v in enumerate(sorted_trans)}
    step_idx = {v: i for i, v in enumerate(sorted_steps)}

    for (trans, step), loss in data_dict.items():
        i = trans_idx[trans]
        j = step_idx[step]
        heatmap[i, j] = loss
    return heatmap, sorted_steps, sorted_trans


# TODO: outdated!
def get_loss_heatmap(task_name, measure, task_ids=None, log_scale=False):
    measure_name = {"ood": "OODLoss", "id": "loss", "ih": "ih_score", "pth": "pth_score"}
    folder_path = os.path.join("results", task_name)
    measure_title = {"ood": "OOD Loss", "id": "ID Loss", "ih": "Induction Head Score", "pth": "Previous Token Head Score"}
    
    heatmap, sorted_steps, sorted_trans = get_loss_heatmap_data(task_name, measure, task_ids=task_ids)

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("plasma")
    im = ax.imshow(heatmap, aspect='auto', cmap=cmap, origin='lower')

    if log_scale:
        ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Step")
    ax.set_ylabel("Total Transitions")
    ax.set_xticks(np.arange(0, len(sorted_steps), 25))
    ax.set_yticks(np.arange(0, len(sorted_trans), 5))
    ax.set_xticklabels(sorted_steps[::25], rotation=45)
    ax.set_yticklabels(sorted_trans[::5])
    ax.set_title(f"{measure_title[measure]} Heatmap")

    cbar = fig.colorbar(im, ax=ax)
    if measure in ["ih", "pth"]:
        cbar.set_label("Attention Score")
    else:
        cbar.set_label("Loss")

    plt.tight_layout()
    if task_ids is None:
        plt.savefig(os.path.join(folder_path, f"{measure_name[measure]}_heatmap.png"))
    else:
        plt.savefig(os.path.join(folder_path, f"{measure_name[measure]}_heatmap_{hash_array(task_ids)}.png"))
    plt.show()


# TODO: outdated!
def get_loss_heatmap_dual(task_name, task_ids=None, log_scale=False):
    folder_path = os.path.join("results", task_name)
    heatmap_ood, sorted_steps, sorted_trans = get_loss_heatmap_data(task_name, "ood", task_ids=task_ids)
    heatmap_id, _, _ = get_loss_heatmap_data(task_name, "id", task_ids=task_ids)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    # Shared color scale: choose LogNorm or Normalize
    data_all = np.concatenate([heatmap_id, heatmap_ood])
    if log_scale:
        norm = LogNorm(vmin=np.nanmin(data_all[data_all > 0]), vmax=np.nanmax(data_all))
    else:
        norm = mcolors.Normalize(vmin=np.nanmin(data_all), vmax=np.nanmax(data_all))

    # Heatmap 1: ID Loss
    im1 = ax1.imshow(
        heatmap_id,
        aspect='auto',
        origin='lower',
        cmap='plasma',
        norm=norm
    )


    ax1.set_title("ID Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total Transitions")
    ax1.set_xticks(np.arange(0, len(sorted_steps), 50))
    ax1.set_yticks(np.arange(0, len(sorted_trans), 5))
    ax1.set_xticklabels(sorted_steps[::50], rotation=45)
    ax1.set_yticklabels(sorted_trans[::5])
    ax1.invert_yaxis()

    # Heatmap 2: OOD Loss
    im2 = ax2.imshow(
        heatmap_ood,
        aspect='auto',
        origin='lower',
        cmap='plasma',
        norm=norm
    )

    ax2.set_title("OOD Loss")
    ax2.set_xlabel("Step")
    ax2.set_xticks(np.arange(0, len(sorted_steps), 50))
    ax2.set_yticks(np.arange(0, len(sorted_trans), 5))
    ax2.set_xticklabels(sorted_steps[::50], rotation=45)
    ax2.set_yticklabels(sorted_trans[::5])
    ax2.invert_yaxis()

    plt.subplots_adjust(wspace=0.02)

    # Shared colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], orientation='vertical')
    cbar.set_label("Loss")

    

    if task_ids is None:
        plt.savefig(os.path.join(folder_path, f"losses_heatmap.png"))
    else:
        plt.savefig(os.path.join(folder_path, f"losses_heatmap_{hash_array(task_ids)}.png"))
    plt.show()



def get_attn_score_lineplot(task_name, vocab_size=20, task_ids=None,
                            alpha=1.0, seq_len=512, hidden_dim=64, stationary=False):
    folder_path = os.path.join("results", task_name)
    folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    # Create a 1x2 subplot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # adjust figsize as needed

    plot_ids = []
    plot_paths = []

    for i in range(len(folders)):
        result_path = os.path.join(folder_path, folders[i])
        log_path = os.path.join(result_path, "log.json")
        config_path = os.path.join(result_path, "config.json")
        log_data = load_log(log_path)
        config = load_config(config_path)

        if (task_ids is not None) and (config.task.total_trans not in task_ids): continue

        if config.vocab_size != vocab_size: continue

        if config.task.alpha != alpha: continue

        if config.seq_len != seq_len: continue

        if config.model.emb_dim != hidden_dim: continue

        if "stationary" in config.task:
            if config.task.stationary != stationary: continue
        elif stationary:
            continue
        
        plot_ids.append(config.task.total_trans)
        plot_paths.append(result_path)
    
    plot_ids = np.array(plot_ids)
    if task_ids is None:
        norm = mcolors.Normalize(vmin=np.log2(min(plot_ids)), vmax=np.log2(max(plot_ids)))
    else:
        norm = mcolors.Normalize(vmin=min(plot_ids), vmax=max(plot_ids))
    cmap = plt.get_cmap('plasma')
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for path in plot_paths:
        log_path = os.path.join(path, "log.json")
        config_path = os.path.join(path, "config.json")
        log_data = load_log(log_path)
        config = load_config(config_path)

        if task_ids is None:
            value = np.log2(config.task.total_trans)
        else:
            value = config.task.total_trans
        color = cmap(norm(value))
        
        # Plot 1: IH Score
        ax2.plot(log_data["eval/step"], log_data["eval/ih_score"], color=color, alpha=0.6)
        
        # Plot 2: PTH Score
        ax1.plot(log_data["eval/step"], log_data["eval/pth_score"], color=color, alpha=0.6)


    # Customize each subplot
    ax2.set_xscale("log")
    ax2.set_title("Induction Head Score vs Step")
    ax2.set_xlabel("Step")

    ax1.set_xscale("log")
    ax1.set_title("Previous Token Head Score vs Step")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Attention Score")

    plt.subplots_adjust(wspace=0.02)  # Try 0 for zero gap
    # Add colorbar to the whole figure
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='vertical')
    cbar.set_label("Total Transitions")

    if task_ids is None:
        plt.savefig(os.path.join(folder_path, f"attn_scores_lineplots_{vocab_size}_{alpha}.png"))
    else:
        plt.savefig(os.path.join(folder_path, f"attn_scores_lineplots_{hash_array(plot_ids)}_{vocab_size}_{alpha}.png"))
    plt.show()



########################   
# Loss along positions #
########################

def kl_plot(model, sampler, task=None, num_samples=100, pos=None):
    trans_mat = get_all_trans_mat(sampler)
    
    if task == None:
        batch, _, tasks = sampler.generate(mode="testing", task=task, num_samples=num_samples)
        B,T = batch.shape
        K = sampler.num_states
        tasks_exp = tasks[:, None].expand(B, T)        # shape: (B, T)
        batch_exp = batch                              # shape: (B, T)
        
        # Flatten for advanced indexing
        flat_tasks = tasks_exp.reshape(-1)             # (B*T,)
        flat_batch = batch_exp.reshape(-1)             # (B*T,)
        
        # Gather rows from trans_mat using tasks and batch
        selected = trans_mat[flat_tasks, flat_batch]   # shape: (B*T, K)
        
        trans_probs = selected.view(B, T, K)
    else:
        batch, _ = sampler.generate(mode="testing", task=task, num_samples=num_samples)
        trans_probs = trans_mat[task][batch]
        
    kl_losses = F.kl_div(nn.Softmax(dim=-1)(model(batch)[0]).log(), trans_probs, reduction="none").sum(dim=-1).detach().cpu().numpy() 
    
    if pos is not None:
        kl_losses = kl_losses[:, :pos]

    # Create a 1D NumPy array
    arr = np.arange(kl_losses.shape[1])
    
    mean_losses = np.mean(kl_losses, axis=0)
    std_losses = np.std(kl_losses, axis=0)
    
    # Plot the array
    plt.plot(arr, mean_losses)
    plt.fill_between(arr, np.maximum(mean_losses - 2*std_losses, 0), mean_losses + 2*std_losses, color='blue', alpha=0.3, label="Mean ± 2 std")
    
    plt.grid()
    if task is not None:
        plt.title(f"Average KL-divergence for task {task}")
    else:
        plt.title("Average KL-divergence over all tasks")
    plt.xlabel("Positions")
    plt.ylabel("KL-divergence")
    plt.legend()
    plt.tight_layout()
    
    plt.show()



def get_empirical_transition_matrix(batch, num_states, pseudocount=1.0, return_per_sample=False):
    """
    Compute empirical transition matrix from a batch of sequences.
    
    Counts observed consecutive transitions i -> j and normalizes to get P(j|i).
    Applies Laplace smoothing to avoid zero probabilities.
    Uses vectorized operations for efficiency.
    
    Args:
        batch: LongTensor of shape (B, T), batch of sequences
        num_states: int, number of states (vocabulary size)
        pseudocount: float, Laplace smoothing parameter (default: 1.0)
        return_per_sample: bool, if True returns (B, N, N) matrices, else returns (N, N) averaged
    
    Returns:
        If return_per_sample=False:
            trans_mat: FloatTensor of shape (N, N), empirical transition matrix P(j|i)
                      where trans_mat[i, j] = P(next_token=j | current_token=i)
        If return_per_sample=True:
            trans_mat_batch: FloatTensor of shape (B, N, N), per-sample transition matrices
    
    Examples:
        >>> # Aggregate across all sequences
        >>> batch = torch.tensor([[0, 1, 2, 0, 1], [1, 2, 0, 1, 2]])  # (2, 5)
        >>> trans_mat = get_empirical_transition_matrix(batch, num_states=3)
        >>> trans_mat.shape
        torch.Size([3, 3])
        
        >>> # Get per-sample matrices
        >>> trans_mat_batch = get_empirical_transition_matrix(batch, num_states=3, return_per_sample=True)
        >>> trans_mat_batch.shape
        torch.Size([2, 3, 3])
    """
    B, T = batch.shape
    N = num_states
    device = batch.device
    
    # Initialize count matrix with pseudocount (Laplace smoothing)
    if return_per_sample:
        counts = torch.ones((B, N, N), device=device, dtype=torch.float) * pseudocount
    else:
        counts = torch.ones((N, N), device=device, dtype=torch.float) * pseudocount
    
    # Get all consecutive transitions (vectorized)
    current_tokens = batch[:, :-1]  # (B, T-1)
    next_tokens = batch[:, 1:]      # (B, T-1)
    
    if return_per_sample:
        # Vectorized counting for per-sample matrices
        # Create batch indices for each transition
        batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, T-1)  # (B, T-1)
        
        # Flatten everything for scatter_add
        flat_batch_idx = batch_indices.reshape(-1)    # (B*(T-1),)
        flat_current = current_tokens.reshape(-1)     # (B*(T-1),)
        flat_next = next_tokens.reshape(-1)           # (B*(T-1),)
        
        # Use index_put with accumulate=True for counting
        # indices is a tuple of (batch_idx, current_token, next_token)
        counts.index_put_(
            (flat_batch_idx, flat_current, flat_next),
            torch.ones_like(flat_batch_idx, dtype=torch.float),
            accumulate=True
        )
    else:
        # Vectorized counting for aggregated matrix
        # Flatten the token pairs
        flat_current = current_tokens.reshape(-1)  # (B*(T-1),)
        flat_next = next_tokens.reshape(-1)        # (B*(T-1),)
        
        # Use index_put with accumulate=True for counting
        counts.index_put_(
            (flat_current, flat_next),
            torch.ones_like(flat_current, dtype=torch.float),
            accumulate=True
        )
    
    # Normalize to get probabilities: P(j|i) = counts[i,j] / sum_j counts[i,j]
    trans_mat = counts / counts.sum(dim=-1, keepdim=True)
    
    return trans_mat


def predictive_distribution_batched(x_seq_batch, transition_matrices):
    """
    Computes batched Pr(x_{t+1} | x_{1:t}) in log-space for multiple sequences.
    
    Uses Bayesian inference to compute the posterior predictive distribution over
    K possible transition matrices, weighting each by its likelihood given the observed sequence.
    
    IMPORTANT: This function assumes sequences contain only actual tokens (no padding).
    If your sequences have padding tokens at odd positions, you must filter them out
    before calling this function (e.g., using batch[:, ::2] to extract even positions).

    Args:
        x_seq_batch: LongTensor of shape (B, T), each row is a sequence x_{1:t}
                     Must contain only actual tokens, not padding tokens.
        transition_matrices: FloatTensor of shape (K, N, N), each is a transition matrix P^{(k)}

    Returns:
        pred_probs: Tensor of shape (B, N), predictive distribution for x_{t+1}
    """
    B, T = x_seq_batch.shape
    K, N, _ = transition_matrices.shape

    # Step 1: Compute log_weights[b, k] = log-likelihood of x_{1:T} under model k
    # We'll gather P_{x_{tau-1}, x_tau}^{(k)} for each tau and compute log-product
    log_P = torch.log(torch.clamp(transition_matrices, min=1e-40))  # (K, N, N)

    # Expand for indexing
    x_prev = x_seq_batch[:, :-1]  # (B, T-1)
    x_curr = x_seq_batch[:, 1:]   # (B, T-1)

    # Gather transition probabilities for each model k
    # log_probs: (B, K, T-1)
    log_probs = []
    for k in range(K):
        # log_P_k[x_prev, x_curr] → (B, T-1)
        log_Pk = log_P[k]  # (N, N)
        log_prob_k = log_Pk[x_prev, x_curr]  # batch-wise indexing
        log_probs.append(log_prob_k.unsqueeze(1))  # shape (B, 1, T-1)

    log_probs = torch.cat(log_probs, dim=1)  # (B, K, T-1)
    log_weights = torch.sum(log_probs, dim=2)  # (B, K)

    # Normalize: log_softmax over K models
    log_weights = F.log_softmax(log_weights, dim=1)  # (B, K)

    # Step 2: Compute log predictive probabilities for each next state j
    x_t = x_seq_batch[:, -1]  # (B,)
    log_pred = torch.full((B, N), -float('inf'), device=x_seq_batch.device)

    for k in range(K):
        # log_P_k[x_t, :] shape: (B, N)
        log_Pk = log_P[k]  # (N, N)
        log_Pk_xt = log_Pk[x_t]  # (B, N)
        # Add log weight
        log_term = log_weights[:, k].unsqueeze(1) + log_Pk_xt  # (B, N)
        log_pred = torch.logaddexp(log_pred, log_term)

    # Final probability
    pred_probs = torch.exp(log_pred)  # (B, N)
    return pred_probs


def bayes_emp_plot(file_path=None, 
                   task=None, num_samples=2000,
                   emp=True, bayes=True, unigram=True):
    assert emp or bayes or unigram, "At least one of emp or bayes or unigram should be True"

    assert file_path is not None, "file_path must be provided"
    folder_name = file_path
    _, sampler, _ = load_everything("latent", folder_name)

    trans_mat = get_all_trans_mat(sampler)
    
    if task is None:
        batch, _, latents = sampler.generate(mode="major", num_samples=num_samples, task=task)
    else:
        batch, _ = sampler.generate(mode="major", num_samples=num_samples, task=task)
        latents = task

    batch = batch[:, ::2]
    
    batch_size, seq_len = batch.shape
    trans_mat_est = torch.ones((batch_size, sampler.num_states, sampler.num_states), device=batch.device)
    uni_trans_mat_est = torch.ones((batch_size, sampler.num_states), device=batch.device)

    values = torch.ones(batch_size, dtype=torch.float, device=batch.device)

    emp_mean_losses = np.zeros(seq_len)
    emp_std_losses = np.zeros(seq_len)
    bayes_mean_losses = np.zeros(seq_len)
    bayes_std_losses = np.zeros(seq_len)

    if unigram:
        unigram_mean_losses = np.zeros(seq_len)
        unigram_std_losses = np.zeros(seq_len)
    
    next_states = batch[:, 1:]  # (B, T-1)

    for pos in range(seq_len):
        pred_probs = trans_mat_est / trans_mat_est.sum(dim=-1, keepdim=True)
        trans_mat_est.index_put_((torch.arange(batch_size), batch[:,pos], next_states[:,pos]), values, accumulate=True)
        uni_trans_mat_est.index_put_((torch.arange(batch_size), batch[:,pos]), values, accumulate=True)

        
        kl_div = F.kl_div(pred_probs[torch.arange(batch_size),
                                        batch[:,pos]].log(), trans_mat[latents, batch[:,pos]], reduction="none").sum(dim=-1)
        emp_mean_losses[pos] = kl_div.mean().detach().cpu().numpy()
        emp_std_losses[pos] = kl_div.std().detach().cpu().numpy()

        if unigram:
            pred_probs = uni_trans_mat_est / uni_trans_mat_est.sum(dim=-1, keepdim=True)

            if hasattr(sampler, 'stationary') and sampler.stationary is not None:
                kl_div = F.kl_div(pred_probs[torch.arange(batch_size)].log(), 
                                sampler.stationary[latents], reduction="none").sum(dim=-1)
            else:
                kl_div = torch.zeros(batch_size, device=batch.device)
        
            unigram_mean_losses[pos] = kl_div.mean().detach().cpu().numpy()
            unigram_std_losses[pos] = kl_div.std().detach().cpu().numpy()

        pred_probs = predictive_distribution_batched(batch[:,:pos], trans_mat)
        kl_div = F.kl_div(pred_probs.log(), trans_mat[latents, batch[:,pos-1]], reduction="none").sum(dim=-1)
        bayes_mean_losses[pos] = kl_div.mean().detach().cpu().numpy()
        bayes_std_losses[pos] = kl_div.std().detach().cpu().numpy()

    arr = np.arange(seq_len)
    if emp:
        plt.plot(arr, emp_mean_losses)
        plt.fill_between(arr, np.maximum(emp_mean_losses - 2*emp_std_losses, 0), emp_mean_losses + 2*emp_std_losses, color='blue', alpha=0.3, label="Emp Mean ± 2 std")

    if bayes:
        plt.plot(arr, bayes_mean_losses)
        plt.fill_between(arr, np.maximum(bayes_mean_losses - 2*bayes_std_losses, 0), bayes_mean_losses + 2*bayes_mean_losses, color='orange', alpha=0.3, label="Bayes Mean ± 2 std")
    
    if unigram:
        plt.plot(arr, unigram_mean_losses)
        plt.fill_between(arr, np.maximum(unigram_mean_losses - 2*unigram_std_losses, 0), unigram_mean_losses + 2*unigram_mean_losses, color='green', alpha=0.3, label="Unigram Mean ± 2 std")
    
    plt.grid()
    if task is not None:
        plt.title(f"Average KL-divergence for task {task}")
    else:
        plt.title("Average KL-divergence over all tasks")
    plt.xlabel("Positions")
    plt.ylabel("KL-divergence")
    plt.legend()
    plt.tight_layout()
    
    plt.show()


def bayes_emp_ood_plot(
        vocab_size, total_trans, 
        alpha=0.5, seq_len=512, hidden_dim=64, num_samples=2000, low=1, high=200, 
        emp=True, bayes=True):
    assert emp or bayes, "At least one of emp or bayes should be True"

    folder_name = get_config(vocab_size=vocab_size, total_trans=total_trans, 
                             alpha=alpha, seq_len=seq_len, hidden_dim=hidden_dim)

    if len(folder_name) > 0:
        _, sampler, _ = load_everything("latent", folder_name[0])
    else:
        print("The configuration does not exist.")
        return

    batch, _, trans_mat, _ = sampler.generate(mode="ood", num_samples=num_samples, return_trans_mat=True)

    # Check if padding is used
    use_padding = hasattr(sampler, 'pad') and sampler.pad
    
    batch_size, seq_len = batch.shape
    trans_mat_est = torch.ones((batch_size, sampler.num_states, sampler.num_states), device=batch.device)

    values = torch.ones(batch_size, dtype=torch.float, device=batch.device)

    emp_mean_losses = np.zeros(high-low)
    emp_std_losses = np.zeros(high-low)
    bayes_mean_losses = np.zeros(high-low)
    bayes_std_losses = np.zeros(high-low)
    
    if use_padding:
        # For padded sequences, only use even positions (non-padding tokens)
        for t in range(0, low, 2):
            next_t = t + 2
            if next_t < seq_len:
                trans_mat_est.index_put_((torch.arange(batch_size), batch[:,t], batch[:,next_t]), values, accumulate=True)

        for pos in range(low, high):
            if pos % 2 != 0:  # Skip odd positions (padding tokens)
                continue
                
            pred_probs = trans_mat_est / trans_mat_est.sum(dim=-1, keepdim=True)
            next_pos = pos + 2
            if next_pos < seq_len:
                trans_mat_est.index_put_((torch.arange(batch_size), batch[:,pos], batch[:,next_pos]), values, accumulate=True)
            
            kl_div = F.kl_div(pred_probs[torch.arange(batch_size),
                                         batch[:,pos]].log(), 
                                         trans_mat[torch.arange(batch_size), batch[:,pos]], reduction="none").sum(dim=-1)
            emp_mean_losses[pos-low] = kl_div.mean().detach().cpu().numpy()
            emp_std_losses[pos-low] = kl_div.std().detach().cpu().numpy()

            # Extract only non-padding tokens for Bayesian predictor
            non_pad_batch = batch[:, :pos+1:2]
            pred_probs = predictive_distribution_batched(non_pad_batch, get_all_trans_mat(sampler))
            kl_div = F.kl_div(pred_probs.log(), trans_mat[torch.arange(batch_size), batch[:,pos]], reduction="none").sum(dim=-1)
            bayes_mean_losses[pos-low] = kl_div.mean().detach().cpu().numpy()
            bayes_std_losses[pos-low] = kl_div.std().detach().cpu().numpy()
    else:
        # Non-padded case: original logic
        next_states = batch[:, 1:]  # (B, T-1)
        
        for t in range(low):
            trans_mat_est.index_put_((torch.arange(batch_size), batch[:,t], next_states[:,t]), values, accumulate=True)

        for pos in range(low, high):
            pred_probs = trans_mat_est / trans_mat_est.sum(dim=-1, keepdim=True)
            trans_mat_est.index_put_((torch.arange(batch_size), batch[:,pos], next_states[:,pos]), values, accumulate=True)
            
            kl_div = F.kl_div(pred_probs[torch.arange(batch_size),
                                         batch[:,pos]].log(), 
                                         trans_mat[torch.arange(batch_size), batch[:,pos]], reduction="none").sum(dim=-1)
            emp_mean_losses[pos-low] = kl_div.mean().detach().cpu().numpy()
            emp_std_losses[pos-low] = kl_div.std().detach().cpu().numpy()

            pred_probs = predictive_distribution_batched(batch[:,:pos], get_all_trans_mat(sampler))
            kl_div = F.kl_div(pred_probs.log(), trans_mat[torch.arange(batch_size), batch[:,pos-1]], reduction="none").sum(dim=-1)
            bayes_mean_losses[pos-low] = kl_div.mean().detach().cpu().numpy()
            bayes_std_losses[pos-low] = kl_div.std().detach().cpu().numpy()
    
    arr = np.arange(low, high)
    if emp:
        plt.plot(arr, emp_mean_losses)
        plt.fill_between(arr, np.maximum(emp_mean_losses - 2*emp_std_losses, 0), emp_mean_losses + 2*emp_std_losses, color='blue', alpha=0.3, label="Emp Mean ± 2 std")

    if bayes:
        plt.plot(arr, bayes_mean_losses)
        plt.fill_between(arr, np.maximum(bayes_mean_losses - 2*bayes_std_losses, 0), bayes_mean_losses + 2*bayes_mean_losses, color='orange', alpha=0.3, label="Bayes Mean ± 2 std")
    
    plt.grid()
    plt.title("Average KL-divergence over all tasks (OOD)")
    plt.xlabel("Positions")
    plt.ylabel("KL-divergence")
    plt.legend()
    plt.tight_layout()
    
    plt.show()


def all_kl_plot(file_path=None, task=None, num_samples=1000, compute_bayes=True):
    """
    Plot KL divergence between model predictions and baselines over sequence positions (Plotly interactive).
    
    Compares model predictions against:
    - Truth: ground truth transition probabilities from the sampler
    - Bayes: Bayesian posterior over transition matrices (optional, expensive for many tasks)
    
    Args:
        file_path: Direct path to experiment folder
        task: Specific task ID to evaluate (None = all tasks)
        num_samples: Number of samples to generate
        compute_bayes: Whether to compute Bayesian baseline (expensive, default: True)
    
    Returns:
        fig: Plotly figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is not installed. Install it with: pip install plotly")
    
    # Load model, sampler, and config
    if file_path is not None:
        model, sampler, _ = load_everything("latent", file_path)
    else:
        raise ValueError("file_path must be provided")
    
    # Get all transition matrices from sampler (handles both major and minor tasks)
    trans_mat = get_all_trans_mat(sampler)
    total_trans = trans_mat.shape[0]
    
    # Generate test samples and get ground truth transition probabilities
    if task is None:
        # Generate samples from all tasks and track which task each sample uses
        batch, _, tasks = sampler.generate(mode="major", task=task, num_samples=num_samples)
        B, T = batch.shape
        N = sampler.num_states

        raw_batch = batch[:, ::2] # seq[0], seq[2], seq[4], ... (non-padding tokens)
        T_raw = raw_batch.shape[1]
        
        # For each position and batch, get the correct transition probabilities
        tasks_exp = tasks[:, None].expand(B, T_raw)        # shape: (B, T_raw)
        flat_tasks = tasks_exp.reshape(-1)                 # (B*T_raw,)
        
        # Gather transition probabilities: trans_mat[task_id, current_token, :]
        selected = trans_mat[flat_tasks, raw_batch.reshape(-1)]   # shape: (B*T_raw, N)
        
        # Reshape back to (B, T_raw, N)
        trans_probs = selected.view(B, T_raw, N)
        trans_probs = trans_probs[:, :-1] # remove the last token
    else:
        raise NotImplementedError("Task-specific KL plot is not implemented yet.")
    
    # Move model to the same device as batch
    model = model.to(batch.device)
    
    # Get model predictions (inference mode to save memory)
    with torch.no_grad():
        pred = model(batch)
        model_pred_probs = nn.Softmax(dim=-1)(pred)  # (B, T, N)
        model_pred_probs = model_pred_probs[:, 1::2, :-1] # (B, T_raw-1, N)
        
        # KL(model || ground_truth): how well model matches true transition probabilities
        trans_kl_losses = F.kl_div(trans_probs.log(), model_pred_probs, reduction="none").sum(dim=-1).detach().cpu().numpy()  # (B, T_raw-1)

    # Compute mean and std across samples for ground truth baseline
    trans_mean_losses = np.mean(trans_kl_losses, axis=0)
    trans_std_losses = np.std(trans_kl_losses, axis=0)
    
    # Compute Bayesian baseline if requested and feasible
    bayes_mean_losses = None
    bayes_std_losses = None
    if compute_bayes and total_trans < 300:
        print(f"Computing Bayesian baseline for {T_raw-1} positions...")
        bayes_mean_losses = np.zeros(T_raw-1)
        bayes_std_losses = np.zeros(T_raw-1)
        
        for t in range(T_raw-1):
            # Extract only non-padding tokens up to current position
            non_pad_batch = raw_batch[:, :t+1]
            with torch.no_grad():
                bayes_probs = predictive_distribution_batched(non_pad_batch, trans_mat)
                kl_div = F.kl_div(bayes_probs.log(), trans_probs[:, t], reduction="none").sum(dim=-1)
                bayes_mean_losses[t] = kl_div.mean().detach().cpu().numpy()
                bayes_std_losses[t] = kl_div.std().detach().cpu().numpy()
            
            # Clear GPU cache periodically to prevent OOM
            if t % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # ===== Create Plotly figure =====
    fig = go.Figure()
    
    x_positions = np.arange(T_raw-1)
    
    # Plot ground truth baseline with confidence band
    fig.add_trace(go.Scatter(
        x=x_positions,
        y=trans_mean_losses,
        mode='lines',
        name='Model',
        line=dict(color='rgb(31, 119, 180)', width=2),
        showlegend=True
    ))
    
    # Add confidence band for ground truth
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_positions, x_positions[::-1]]),
        y=np.concatenate([
            np.maximum(trans_mean_losses - trans_std_losses, 0),
            (trans_mean_losses + trans_std_losses)[::-1]
        ]),
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip',
        name='Model ± std'
    ))
    
    # Plot Bayesian baseline if computed
    if bayes_mean_losses is not None:
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=bayes_mean_losses,
            mode='lines',
            name='Bayesian Posterior',
            line=dict(color='rgb(255, 127, 14)', width=2),
            showlegend=True
        ))
        
        # Add confidence band for Bayesian
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_positions, x_positions[::-1]]),
            y=np.concatenate([
                np.maximum(bayes_mean_losses - bayes_std_losses, 0),
                (bayes_mean_losses + bayes_std_losses)[::-1]
            ]),
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip',
            name='Bayesian ± std'
        ))
    
    # Update layout
    title_text = f"KL Divergence: Model vs Baselines"
    if task is not None:
        title_text += f" (Task {task})"
    title_text += f"<br><sub>Total transitions: {total_trans}, Samples: {num_samples}</sub>"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Token Position (non-padding)",
        yaxis_title="KL Divergence",
        hovermode='x unified',
        template='plotly_white',
        width=1000,
        height=600,
        font=dict(size=12),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0, 0, 0, 0.1)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0, 0, 0, 0.1)'
        )
    )
    
    # Return the figure (Jupyter will automatically display it)
    return fig


def view_mask(batch, info, padded=False, dyck=False):
    a = batch.squeeze(0) if batch.dim() == 3 else batch
    b = info.squeeze(0) if info.dim() == 3 else info

    assert a.shape == b.shape and a.dim() == 2

    # Compute max digit width
    max_width = max(len(str(int(x.item()))) for x in a.flatten())
    cell_width = max_width + 1  # add spacing between columns
    
    pad = a.max().item()

    html = "<pre style='font-family: monospace; font-size: 14px; line-height: 1.2;'>"

    for i in range(a.shape[0]):
        html += f"<span style='color:pink; font-weight:bold;'>Seq {i:>2}:\n</span>"
        row_html = ""
        for j in range(a.shape[1]):
            val = int(a[i, j].item())
            if padded:
                if val == pad:
                    val = "_"
                elif dyck:
                    if val == (pad - 1):
                        val = "("
                    elif val == (pad - 2):
                        val = ")"
                else:
                    val = str(val)
            else:
                if dyck:
                    if val == pad:
                        val = "("
                    elif val == (pad - 1):
                        val = ")"
                    else:
                        val = str(val)
                else:
                    val = str(val)
            cell = f"<span style='display:inline-block; width:{cell_width}ch; text-align:center;'>{val}</span>"
            if b[i, j].item() != 0:
                cell = f"<span style='color:red; font-weight:bold;'>{cell}</span>"
            row_html += cell
        html += row_html + "<br>"

    html += "</pre>"
    display(HTML(html))


########################
# Bigram Analysis      #
########################

def bigram_prefix_counts(x: torch.Tensor, V: int, *, dtype=torch.int64) -> torch.Tensor:
    """
    Compute cumulative bigram counts at each position in sequences.
    
    For each position t, computes the count of all bigrams (prev, next) that occurred
    up to and including position t in the sequence. This creates a cumulative count
    matrix for each sequence.
    
    Args:
        x: Input sequences of shape (B, T) where B is batch size and T is sequence length.
           Values should be in [0, V-1].
        V: Vocabulary size (number of states)
        dtype: Output dtype (default: torch.int64)
    
    Returns:
        counts: Tensor of shape (B, T, V, V) where counts[b, t, u, v] is the count
                of bigram (u -> v) in sequence b up to position t.
                Position 0 is always zeros (no bigrams yet).
    
    Example:
        >>> x = torch.tensor([[0, 1, 2, 1]])  # Single sequence: 0->1->2->1
        >>> counts = bigram_prefix_counts(x, V=3)
        >>> counts[0, 1, 0, 1]  # At position 1, count of (0->1) is 1
        tensor(1)
        >>> counts[0, 2, 1, 2]  # At position 2, count of (1->2) is 1
        tensor(1)
    """
    assert x.dim() == 2, "x must be (B, T)"
    B, T = x.shape
    device = x.device

    if T == 1:
        return torch.zeros(B, T, V, V, dtype=dtype, device=device)

    prev = x[:, :-1]               # (B, T-1)
    nxt  = x[:,  1:]               # (B, T-1)

    prev_oh = F.one_hot(prev, num_classes=V).to(torch.float32)  # (B, T-1, V)
    nxt_oh  = F.one_hot(nxt,  num_classes=V).to(torch.float32)  # (B, T-1, V)

    edges = torch.einsum('btu,btv->btuv', prev_oh, nxt_oh)      # (B, T-1, V, V)

    zero_pad = torch.zeros(B, 1, V, V, dtype=edges.dtype, device=device)
    edges_padded = torch.cat([zero_pad, edges], dim=1)          # (B, T, V, V)

    out = torch.cumsum(edges_padded, dim=1)                     
    return out.to(dtype)


def compute_hiddens_data(
    config,
    model: torch.nn.Module,
    x: torch.Tensor,
    layer_index: int = 1,
) -> torch.Tensor:
    """
    Extract hidden representations from a specific layer at odd positions.
    
    For sequences with padding, this extracts hidden states at the non-padding
    token positions (odd indices in the full sequence).
    
    Args:
        config: Configuration object (not directly used, but may be needed for compatibility)
        model: Transformer model
        x: Input sequences of shape (B, T_full) where T_full includes padding tokens.
           Odd positions (1, 3, 5, ...) are the actual tokens.
        layer_index: Which transformer layer to extract from (default: 1)
    
    Returns:
        hiddens: Tensor of shape (B, seq_len, d_model) where seq_len = (T_full - 1) // 2
                 Hidden representations at odd positions from the specified layer's
                 attention block output.
    """
    model_device = next(model.parameters()).device
    x = x.to(model_device)

    seq_len = x.shape[1]
    pos = torch.arange(1, seq_len, 2).to(model_device)  # Odd positions
    cache = {}
        
    def hook_fn(module, inp, out):
        cache["vec"] = out[:, pos, :].detach()

    handle = model.layers[layer_index].attn_block.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        _ = model(x)
    handle.remove()

    return cache["vec"]


def compute_bigram_r2_scores(
    model: torch.nn.Module,
    sampler,
    config,
    num_samples: int = 1024,
    T0: int = 10,
    max_layers: int | None = None,
    mode: str = "ood",
) -> np.ndarray:
    """
    Compute R² scores between hidden representations and bigram prefix counts.
    
    For each layer and time step, fits a linear regression to predict bigram counts
    from hidden representations. Returns the R² scores as a measure of how well
    hidden states encode bigram information.
    
    Args:
        model: Transformer model
        sampler: Data sampler with generate() method and num_states, seq_len attributes
        config: Configuration object
        num_samples: Number of samples to generate (default: 1024)
        T0: Starting position index (default: 10, to skip early positions)
        max_layers: Maximum number of layers to analyze. If None, uses all layers.
        mode: Generation mode for sampler (default: "ood")
    
    Returns:
        r2_scores: Array of shape (num_layers, num_time_steps) containing R² scores
                   for each layer at each time step.
    """
    from sklearn.linear_model import LinearRegression
    
    # Generate samples
    x, *_ = sampler.generate(num_samples=num_samples, mode=mode)
    
    # Compute bigram counts (only use non-padding tokens)
    bigram_count = bigram_prefix_counts(x[:, ::2], sampler.num_states)
    bigram_count = bigram_count.reshape(num_samples, sampler.seq_len, -1)
    bigram_count = bigram_count[:, T0:-1]
    Tdim = bigram_count.shape[1]
    
    # Determine number of layers
    if max_layers is None:
        num_layers = len(model.layers)
    else:
        num_layers = min(max_layers, len(model.layers))
    
    r2 = np.zeros((num_layers, Tdim))
    
    for layer in range(num_layers):
        hiddens = compute_hiddens_data(
            config,
            model,
            x,
            layer_index=layer,
        )
        hiddens = hiddens[:, T0:]
        
        for t in range(Tdim):
            X = hiddens[:, t].cpu().numpy()
            y = bigram_count[:, t].cpu().numpy()
            
            lr_model = LinearRegression()
            lr_model.fit(X, y)
            r2[layer, t] = lr_model.score(X, y)
    
    return r2


def plot_bigram_r2_scores(
    r2_scores: np.ndarray,
    T0: int = 10,
    title: str = "R² Score: Hidden States vs Bigram Counts",
    show: bool = True,
):
    """
    Plot R² scores over time for each layer.
    
    Args:
        r2_scores: Array of shape (num_layers, num_time_steps) from compute_bigram_r2_scores
        T0: Starting position offset (used for x-axis labeling)
        title: Plot title (default: "R² Score: Hidden States vs Bigram Counts")
        show: Whether to show the plot immediately (default: True)
    
    Returns:
        fig: Plotly figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is not installed. Install it with: pip install plotly")
    
    num_layers, num_time_steps = r2_scores.shape
    time_steps = T0 + np.arange(num_time_steps)
    
    fig = go.Figure()
    
    for layer in range(num_layers):
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=r2_scores[layer],
            mode='lines',
            name=f'Layer {layer}'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Position t',
        yaxis_title='R² Score',
        template='plotly_white'
    )
    
    if show:
        fig.show()
    
    return fig


def plot_sufficient_stat(
    model: torch.nn.Module,
    sampler,
    config,
    num_samples: int = 1024,
    T0: int = 10,
    max_layers: int | None = None,
    mode: str = "ood",
    title: str = "R² Score: Hidden States vs Bigram Counts",
    show: bool = True,
):
    """
    Complete sufficient statistics analysis pipeline: compute R² scores and plot results.
    
    This is a convenience function that combines compute_bigram_r2_scores and
    plot_bigram_r2_scores into a single call.
    
    Args:
        model: Transformer model
        sampler: Data sampler with generate() method and num_states, seq_len attributes
        config: Configuration object
        num_samples: Number of samples to generate (default: 1024)
        T0: Starting position index (default: 10, to skip early positions)
        max_layers: Maximum number of layers to analyze. If None, uses all layers.
        mode: Generation mode for sampler (default: "ood")
        title: Plot title (default: "R² Score: Hidden States vs Bigram Counts")
        show: Whether to show the plot immediately (default: True)
    
    Returns:
        tuple: (r2_scores, fig) where:
            - r2_scores: Array of shape (num_layers, num_time_steps) with R² scores
            - fig: Plotly figure object
    """
    r2_scores = compute_bigram_r2_scores(
        model=model,
        sampler=sampler,
        config=config,
        num_samples=num_samples,
        T0=T0,
        max_layers=max_layers,
        mode=mode,
    )
    
    fig = plot_bigram_r2_scores(
        r2_scores=r2_scores,
        T0=T0,
        title=title,
        show=show,
    )
    
    return r2_scores, fig


