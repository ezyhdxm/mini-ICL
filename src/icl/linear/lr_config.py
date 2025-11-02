from ml_collections import ConfigDict
import torch
import os


def get_config() -> ConfigDict:
    config = ConfigDict()
    NDIMS = 6
    NPOINTS = 64

    config.dtype = "float32"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.work_dir = os.path.join("results", "linear")  # Specify working directory

    config.task = ConfigDict()
    config.task.name = "noisy_linear_regression"
    config.task.n_tasks = 3
    config.task.n_dims = NDIMS
    config.task.n_points = NPOINTS
    config.task.batch_size = 128
    config.task.data_seed = 101
    config.task.task_seed = 102
    config.task.noise_seed = 103
    config.task.data_scale = 1.0
    config.task.minor_scale = 1.0
    config.task.task_scale = 1.0
    config.task.noise_scale = 0.5
    config.task.p_minor = 0.1  # Probability of tasks from the minor task pool
    config.task.n_minor_tasks = 0  # Number of minor tasks, if needed
    config.task.is_mixture = False  # Whether to use mixture of tasks

    config.model = ConfigDict()
    config.model.name = "transformer"
    config.model.activation = "gelu"  # Activation function for the model
    config.model.n_points = NPOINTS
    config.model.n_dims = NDIMS
    config.model.n_layer = 16
    config.model.n_embd = 128
    config.model.n_head = 2
    config.model.seed = 100
    config.model.pad = "mapsto"  # Padding strategy, can be "bos" or "mapsto"

    config.training = ConfigDict()
    config.training.optimizer = "adamw"
    config.training.lr = 2e-4
    config.training.schedule = "triangle"
    config.training.weight_decay = 1e-4
    config.training.warmup_steps = 10_000
    config.training.total_steps = 20_000

    config.eval = ConfigDict()
    config.eval.n_samples = 2**11
    config.eval.batch_size = 512
    config.eval.data_seed = 104
    config.eval.task_seed = 105
    config.eval.noise_seed = 106
    config.eval.every = 25
    config.eval.save_every = 100

    config.wandb = ConfigDict()
    config.wandb.project = "mini-ICL-linear"  # Specify wandb project

    return config