from dataclasses import dataclass, field
import torch
from typing import Tuple, Optional, Any
from ml_collections import ConfigDict # DeepMind style config library
import os


def get_config_coin() -> ConfigDict:
    config = ConfigDict()
    config.profile = False  # Default profiling flag, can be set to True for performance profiling
    config.mixed_precision = False  # Default mixed precision flag, can be set to True for mixed precision training
    config.seq_len = 129
    config.vocab_size = 3
    config.seed = 10086
    config.batch_size = 64
    config.eval_size = 128
    config.test_size = 512
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    TASKNAME = "coin"  # Default task name, can be overridden in config
    config.work_dir = os.path.join("results", TASKNAME)  # Specify working directory
    config.ngram = 3  # N-gram order for the n-gram learner
    config.wandb = ConfigDict()
    config.wandb.project = "mini-ICL-coin"  # Specify wandb project

    #####################  
    #    Tasks          #
    #####################
    
    config.task = ConfigDict()
    config.task.name = TASKNAME  # Name of the task, must be "latent"
    config.task.ood = True  # Out-of-distribution flag
    config.task.n_tasks = 3  # Total number of transitions to sample
    config.task.p_minor = 0.1  # Probability of tasks from the minor task pool
    config.task.n_minor_tasks = 0  # Number of minor tasks, if needed
    config.task.init_task_pool = None
    config.task.pad = True  # Whether to pad the sequences with a special token

    ######################
    #     Model          #
    ######################

    NUM_LAYERS = 6 # Default number of layers, can be overridden in config

    config.model = ConfigDict()
    config.model.emb_dim = 128
    config.model.bias = False
    config.model.mlp_bias = True
    config.model.ff_dim = 4*128
    config.model.num_layers = NUM_LAYERS
    config.model.num_heads = tuple([2]*NUM_LAYERS)  # Tuple of number of heads for each layer
    config.model.dropout = None  # Dropout rate, None means no dropout
    config.model.mask = True  # Whether to use masking in attention
    config.model.mlp = tuple([True]*NUM_LAYERS)  # Tuple indicating whether to use MLP in each layer
    config.model.layer_norm = True  # Whether to use layer normalization
    config.model.activation = tuple([True]*NUM_LAYERS)  # Tuple indicating whether to use activation in each layer
    config.model.pos_enc = "rotary"  # Type of positional encoding
    config.model.pos_max_len = config.seq_len  # Maximum length for positional encoding
    config.model.flash = True  # Whether to use flash attention for faster computation
    

    #######################
    #     Training        #
    #######################

    config.training = ConfigDict()
    config.training.num_epochs = 20_000
    config.training.weight_decay = 4e-4
    config.training.warmup_steps = 10_000
    config.training.learning_rate = 4e-4
    config.training.eval_iter = 50
    config.training.get_attn = 5000
    config.training.get_checkpoints = 100
    config.training.scheduler = True
    config.training.T_max = 20
    
    
    return config








        
    





