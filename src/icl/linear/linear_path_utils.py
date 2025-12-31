import os 
from ml_collections import ConfigDict
from icl.linear.lr_task import get_task
from icl.linear.lr_models import get_model
import torch
import json

def get_path_to_exp_dir(exp_name):
    """
    Get the path to an experiment's directory.
    
    Args:
        exp_name: Name of the experiment
    
    Returns:
        Path to the experiment directory
    """
    work_dir = os.path.join("..", "results", "linear")
    exp_dir = os.path.join(work_dir, exp_name)
    return exp_dir


def load_model_task_config(exp_name):
    """
    Load a trained model, task, and configuration from an experiment.
    
    Args:
        exp_name: Name of the experiment
    
    Returns:
        model: Trained model loaded from checkpoint
        train_task: Task object used for training
        config: Configuration dictionary
    """
    exp_dir = get_path_to_exp_dir(exp_name)
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "r") as f: config_dict = json.load(f)
    
    config = ConfigDict(config_dict)
    checkpoint_path = os.path.join(exp_dir, "checkpoint.pt")
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
    data_type = torch.float
    model = get_model(**config["model"], dtype=data_type)
    model.load_state_dict(checkpoint["model"])
    train_task = get_task(**config["task"], device=config.device)
    return model, train_task, config

def get_checkpoint_files(exp_name):
    """
    Get list of checkpoint files in an experiment directory.
    
    Args:
        exp_name: Name of the experiment
    
    Returns:
        List of checkpoint filenames
    """
    exp_dir = get_path_to_exp_dir(exp_name)
    checkpoint_files = [f for f in os.listdir(exp_dir) if f.startswith("model_") and f.endswith(".pt")]
    return checkpoint_files

def load_checkpoint(exp_name, checkpoint_file):
    """
    Load a checkpoint from the experiment directory.
    
    Args:
        exp_name: Name of the experiment
        checkpoint_file: Name of the checkpoint file to load
    
    Returns:
        model: The model loaded from the checkpoint
        config: The configuration used for the model
        train_task: The training task object
    """
    exp_dir = get_path_to_exp_dir(exp_name)
    checkpoint_path = os.path.join(exp_dir, checkpoint_file)
    _, train_task, config = load_model_task_config(exp_name)

    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    model = get_model(**config["model"], dtype=torch.float32)
    model.load_state_dict(checkpoint["model"])
    model = model.to("cuda")
    return model, config, train_task