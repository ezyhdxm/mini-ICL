# Core imports for convenience
from icl.latent_markov import get_config_base  # Fixed: was icl.config (doesn't exist)
from icl.models import Transformer
from icl.utils import (
    get_attn_base,
    # visualize_attention,
    #extract_task_vector_markov,
    #predict_with_task_vector_markov,
    train_model_with_plot,
    train_model,
    load_everything,
)
from icl.linear import (
    extract_hidden,
    extract_hidden_multi,
    get_config,
    DiscreteMMSE,
    Ridge,
)

from icl.latent_markov import (
    LatentMarkov,
)

# Define what gets imported with "from icl import *"
__all__ = [
    "get_config_base",
    "Transformer",
    "get_attn_base",
    # "visualize_attention",
    # "extract_task_vector_markov",  
    # "predict_with_task_vector_markov",
    "train_model_with_plot",
    "train_model",
    "load_everything",
    "extract_hidden",
    "extract_hidden_multi",
    "DiscreteMMSE",
    "Ridge",
    "get_config",
    "LatentMarkov",
]