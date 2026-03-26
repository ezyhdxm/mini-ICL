import torch
import os
import glob
import re
from ml_collections import ConfigDict
import json
import pickle

from icl.utils.logger import setup_logger

logger = setup_logger(__name__)

from .basic import get_hash, get_config_hash_for_exp
from icl.models import Transformer


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


def load_checkpoint(config, 
                    step=None, 
                    checkpoint_dir=None, 
                    use_final=True, 
                    verbose=False, 
                    exp_name=None,
                    return_actual_step=False,
                    ):
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
        if exp_name is None:
            # Use canonical hash for linear so exp_name matches get_exp_name() / train_linear
            if getattr(config.task, "name", None) == "noisy_linear_regression":
                exp_name = f"train_{get_config_hash_for_exp(config)}"
            else:
                exp_name = f"train_{get_hash(config)}" 
        exp_dir = os.path.join(config.work_dir, exp_name)
        
        # Handle notebooks directory case
        cur_dir = os.getcwd()
        if cur_dir.endswith("notebooks"):
            exp_dir = os.path.join("..", exp_dir)
        
        if config.task.name!="noisy_linear_regression":
            checkpoint_dir = os.path.join(exp_dir, "checkpoints")
        else:
            checkpoint_dir = exp_dir
        if verbose:
            logger.info(f"Auto-detected checkpoint directory: {checkpoint_dir}")
    
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
                logger.info(f"Loading latest final checkpoint: {filename} (step {step_num})")
        else:
            # Load latest regular checkpoint
            if len(checkpoints['regular']) == 0:
                if len(checkpoints['final']) > 0:
                    step_num, filename = checkpoints['final'][-1]
                    model_path = os.path.join(checkpoint_dir, filename)
                    if verbose:
                        logger.info(f"Only final checkpoints available. Loading: {filename} (step {step_num})")
                else:
                    raise ValueError("No checkpoints found")
            else:
                step_num, filename = checkpoints['regular'][-1]
                model_path = os.path.join(checkpoint_dir, filename)
                if verbose:
                    logger.info(f"Loading latest checkpoint: {filename} (step {step_num})")
    
    elif step == "final":
        if len(checkpoints['final']) == 0:
            raise ValueError("No final checkpoints found")
        step_num, filename = checkpoints['final'][-1]
        model_path = os.path.join(checkpoint_dir, filename)
        if verbose:
            logger.info(f"Loading latest final checkpoint: {filename} (step {step_num})")
    
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
                    logger.info(f"Loading checkpoint: {fname} (step {step})")
                break
        
        # If not found, try exact match in final checkpoints
        if model_path is None:
            for s, fname in checkpoints['final']:
                if s == step:
                    model_path = os.path.join(checkpoint_dir, fname)
                    step_num = s
                    if verbose:
                        logger.info(f"Loading final checkpoint: {fname} (step {step})")
                    break
        
        # If still not found, try direct file paths
        if model_path is None:
            # Try regular format
            candidate = os.path.join(checkpoint_dir, f"model_{step}.pt")
            if os.path.exists(candidate):
                model_path = candidate
                step_num = step
                if verbose:
                    logger.info(f"Loading checkpoint: model_{step}.pt (step {step})")
            else:
                # Try final format
                candidate = os.path.join(checkpoint_dir, f"model_final_{step}.pt")
                if os.path.exists(candidate):
                    model_path = candidate
                    step_num = step
                    if verbose:
                        logger.info(f"Loading final checkpoint: model_final_{step}.pt (step {step})")
        
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
                logger.warning(f"Step {step} not found. Loading closest checkpoint: {closest_filename} (step {closest_step}, diff={diff})")
    
    else:
        raise ValueError(f"Invalid step parameter: {step}. Must be None, int, 'latest', or 'final'")
    
    # Load the checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {model_path}: {e}")
    
    # Initialize model and load state
    try:
        if config.task.name == "noisy_linear_regression":
            from icl.linear.lr_models import get_model
            data_type = torch.float
            model = get_model(**config["model"], dtype=data_type)
        else:
            model = Transformer(config)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model = model.to(device)
        
        if verbose and 'step' in checkpoint:
            logger.info(f"Checkpoint info: step={checkpoint.get('step', 'N/A')}")
        
        if not return_actual_step:
            return model
        else:
            return model, step_num
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
    """
    Load a sampler from a pickle file.
    
    Parameters:
    -----------
    sampler_path : str
        Path to the sampler pickle file
    
    Returns:
    --------
    sampler : object
        The loaded sampler object
    
    Raises:
    -------
    FileNotFoundError
        If the sampler file doesn't exist
    EOFError
        If the file is empty or corrupted
    """
    import os
    
    # Check if file exists
    if not os.path.exists(sampler_path):
        raise FileNotFoundError(
            f"Sampler file not found: {sampler_path}\n"
            f"Please ensure the experiment has been trained and the sampler was saved."
        )
    
    # Check if file is empty
    file_size = os.path.getsize(sampler_path)
    if file_size == 0:
        raise EOFError(
            f"Sampler file is empty (0 bytes): {sampler_path}\n"
            f"This usually means the training was interrupted before the sampler could be saved."
        )
    
    # Try to load the sampler with better error handling.
    # Use a CPU-mapping unpickler so that any CUDA tensors stored in the pickle
    # are restored on CPU instead of their original device.  This prevents a
    # corrupted CUDA context (from a previous experiment's failure) from
    # cascading into sampler-load failures for subsequent experiments.
    import io

    class _CPUUnpickler(pickle.Unpickler):
        """Unpickler that remaps torch storage bytes to CPU."""
        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
            return super().find_class(module, name)

    try:
        with open(sampler_path, "rb") as f:
            sampler = _CPUUnpickler(f).load()
        # _CPUUnpickler remaps stored tensors to CPU, but plain string attributes
        # like `self.device` are not tensors and remain as-is (e.g. "cuda:1").
        # Update them so that any new tensors created inside sampler.generate()
        # are also on CPU and don't mismatch the stored CPU tensors.
        if hasattr(sampler, "device") and isinstance(sampler.device, str) and sampler.device != "cpu":
            sampler.device = "cpu"
        return sampler
    except EOFError as e:
        raise EOFError(
            f"Failed to load sampler from {sampler_path}: file appears to be corrupted or incomplete.\n"
            f"File size: {file_size} bytes\n"
            f"This usually means:\n"
            f"  1. The training was interrupted before the sampler could be fully saved\n"
            f"  2. The file was corrupted during save/transfer\n"
            f"  3. The file is from an incomplete experiment\n"
            f"Original error: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error loading sampler from {sampler_path}: {e}"
        ) from e


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


def load_config_and_sampler(task_name, train_folder):
    """Load config and sampler without allocating GPU memory for the model.

    This is a lightweight alternative to :func:`load_everything` for callers
    that only need the config and sampler (e.g. to build a sampler clone before
    loading a specific checkpoint).  Avoids a wasted GPU allocation.

    Returns
    -------
    sampler : object
    config : ConfigDict
    """
    curr_dir = os.getcwd()
    if curr_dir.endswith("notebooks"):
        path_prefix = os.path.join("..", "results", task_name)
    else:
        path_prefix = os.path.join("results", task_name)
    config_path = os.path.join(path_prefix, train_folder, "config.json")
    sampler_path = os.path.join(path_prefix, train_folder, "sampler.pkl")
    config = load_config(config_path)
    sampler = load_sampler(sampler_path)
    return sampler, config


# Legacy re-exports
from icl.latent_markov.analysis.interventions.bigram import bigram_prefix_counts  # noqa: F401, E402

# Legacy re-exports (moved to icl.utils.legacy.notebook_utils_legacy)
from icl.utils.legacy.notebook_utils_legacy import (  # noqa: F401, E402
    hash_array,
    get_all_trans_mat,
    extract_experiment_metadata,
    get_config,
    get_cos_sim_plot,
    lighten,
    get_pos_loss,
    get_empirical_transition,
    kl_div_ave,
    compute_stationary_distributions,
    pairwise_kl_divergence,
    pairwise_kl_divergence_stationary,
    get_loss_lineplot,
    get_loss_heatmap_data,
    get_loss_heatmap,
    get_loss_heatmap_dual,
    get_attn_score_lineplot,
    kl_plot,
    get_empirical_transition_matrix,
    predictive_distribution_batched,
    bayes_emp_plot,
    bayes_emp_ood_plot,
    view_mask,
    compute_hiddens_data,
    compute_bigram_r2_scores,
)

# Legacy plotly functions (moved to icl.utils.legacy.notebook_utils_plotly)
from icl.utils.legacy.notebook_utils_plotly import (  # noqa: F401, E402
    all_kl_plot,
    plot_bigram_r2_scores,
    plot_sufficient_stat,
)
