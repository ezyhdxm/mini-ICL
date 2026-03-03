"""
Variance analysis for Latent Markov task on **non-padded** sequences.

Extracted from ``icl.utils.latent_nonpadded``:
- Task variance (P2)
- Token-conditioned hiddens
- P1 variance
- W variance vs position
"""

import gc
import numpy as np
import torch
from typing import Optional, Sequence

import icl.utils.notebook_utils as nu
from icl.coin.analysis._helpers import (
    compute_hiddens_multi_coin,
    compute_hiddens_token_conditioned_coin,
)
from icl.latent_markov.analysis.ood import get_latent_sampler
from icl.utils.logger import setup_logger
from icl.linear.task_variance import (
    compute_task_variance_multi_layer,
    extract_plotting_data_multi_layer,
)
from icl.linear.p1_variance import (
    compute_p1_variance_multi_layer,
    extract_plotting_data_multi_layer as extract_p1_plotting_data_multi_layer,
)

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# 1.  get_task_variance
# ---------------------------------------------------------------------------

def get_task_variance(
    exp_name: str,
    layers: Optional[Sequence] = None,
    batch_size: int = 64,
    positions_of_interest: Optional[list] = None,
    n_minor: int = 64,
    step: Optional[int] = None,
    verbose: bool = False,
    eps: float = 1e-8,
) -> tuple:
    """
    Get hidden representations and compute task variance (P2) for Latent Markov task
    on **non-padded** sequences.

    Non-padded counterpart of task variance for latent. Always uses ``n_ood=0``.
    If ``n_minor`` exceeds the sampler's actual minor-task count it is silently capped.

    Steps:
      1. Load model and create sampler with ``n_minor`` minor tasks (no OOD).
      2. For each task, sample a new batch and extract hiddens at the
         requested real-token positions ``[0, 1, 2, ...]``.
      3. Compute task variance: variance of batch-averaged hiddens across tasks.

    Parameters
    ----------
    exp_name : str
        Experiment name (folder under results/latent/).
    layers : list, optional
        ``None`` → all layers.
    batch_size : int
    positions_of_interest : list, optional
        ``None`` → all positions.
    n_minor : int
        Number of minor tasks. Capped at ``sampler.n_minor_tasks``.
    step : int, optional
    verbose : bool
    eps : float

    Returns
    -------
    all_hiddens : torch.Tensor
        ``(L, n_tasks, n_positions, batch_size, n_embd)``
    position_info : dict
    results_dict : dict
        Layer number → ``TaskVarianceResults``
    plotting_data : dict
    """
    _, _sampler_orig, config = nu.load_everything("latent", exp_name)

    if step is None:
        step = config.training.num_epochs

    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True
    )
    model.eval().to(config.device)

    sampler, k_minor, _ = get_latent_sampler(exp_name, n_minor, n_ood=0)

    if layers is None:
        layers = list(range(len(model.layers)))

    if verbose:
        logger.info(f"Computing non-padded task variance for latent exp: {exp_name}")
        logger.info(
            f"Layers: {layers}, B: {batch_size}, "
            f"n_tasks: {sampler.n_major_tasks + sampler.n_minor_tasks} "
            f"(k_minor={k_minor})"
        )

    all_hiddens, position_info = compute_hiddens_multi_coin(
        config=config, model=model, sampler=sampler,
        layers=layers, batch_size=batch_size,
        positions_of_interest=positions_of_interest,
    )

    if verbose:
        logger.info(f"Hiddens shape: {all_hiddens.shape}. Computing task variance...")

    results_dict = compute_task_variance_multi_layer(
        all_hiddens=all_hiddens,
        positions_of_interest=positions_of_interest,
        layers=layers,
        eps=eps,
    )
    plotting_data = extract_plotting_data_multi_layer(results_dict)

    if verbose:
        logger.info(f"Computed variance for {len(results_dict)} layers")

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return all_hiddens, position_info, results_dict, plotting_data


# ---------------------------------------------------------------------------
# 2.  plot_task_variance
# ---------------------------------------------------------------------------

def plot_task_variance(
    exp_name: str,
    layers: Optional[Sequence] = None,
    batch_size: int = 64,
    positions_of_interest: Optional[list] = None,
    n_minor: int = 64,
    step: Optional[int] = None,
    verbose: bool = False,
    eps: float = 1e-8,
    figsize: tuple = (8, 6),
    log_x: bool = True,
    show: bool = True,
    title: Optional[str] = None,
) -> dict:
    """
    Compute and plot normalised task variance (P2) for the Latent Markov task on
    non-padded sequences.

    Convenience wrapper around ``get_task_variance``.

    Parameters
    ----------
    exp_name : str
    layers : list, optional
        ``None`` → all layers.
    batch_size : int
    positions_of_interest : list, optional
        ``None`` → all positions.
    n_minor : int
        Capped at ``sampler.n_minor_tasks``.
    step : int, optional
    verbose : bool
    eps : float
    figsize : tuple
    log_x : bool
    show : bool
    title : str, optional

    Returns
    -------
    dict
        ``{'all_hiddens', 'position_info', 'results_dict',
        'plotting_data', 'fig', 'ax'}``.
    """
    import matplotlib.pyplot as plt

    all_hiddens, position_info, results_dict, plotting_data = (
        get_task_variance(
            exp_name=exp_name,
            layers=layers,
            batch_size=batch_size,
            positions_of_interest=positions_of_interest,
            n_minor=n_minor,
            step=step,
            verbose=verbose,
            eps=eps,
        )
    )

    fig, ax = plt.subplots(figsize=figsize)

    for layer_idx in plotting_data['layers']:
        positions = plotting_data['positions']
        var_pos_norm = plotting_data['var_pos_norm'][layer_idx]
        ax.plot(
            positions, var_pos_norm, 'o-',
            label=f'Layer {layer_idx}',
            linewidth=2, markersize=6,
        )

    ax.set_xlabel('Position' + (' (log scale)' if log_x else ''), fontsize=16)
    ax.set_ylabel('Normalized Task Variance', fontsize=16)
    if log_x:
        ax.set_xscale('log')
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()

    return {
        'all_hiddens': all_hiddens,
        'position_info': position_info,
        'results_dict': results_dict,
        'plotting_data': plotting_data,
        'fig': fig,
        'ax': ax,
    }


# ---------------------------------------------------------------------------
# 3.  get_token_conditioned_hiddens
# ---------------------------------------------------------------------------

def get_token_conditioned_hiddens(
    exp_name: str,
    layers: Optional[Sequence] = None,
    batch_size: int = 16,
    positions_of_interest: Optional[list] = None,
    max_unique_tokens: Optional[int] = None,
    n_minor: int = 0,
    step: Optional[int] = None,
    verbose: bool = False,
) -> tuple:
    """
    Get token-conditioned hidden representations for Latent Markov task (non-padded).

    Non-padded counterpart of token-conditioned hiddens for latent. Fixes tokens
    at specific positions and extracts hiddens **at the same position** (the real
    token itself — no following PAD token).
    Always uses ``n_ood=0``. ``n_minor`` is capped at ``sampler.n_minor_tasks``.

    For order=1 latent Markov, tokens are state values in {0, ..., num_states-1}.

    Parameters
    ----------
    exp_name : str
    layers : list, optional
        ``None`` → all layers.
    batch_size : int
    positions_of_interest : list, optional
        Position indices ``[0, seq_len-2]``. ``None`` → all.
    max_unique_tokens : int, optional
    n_minor : int
        Capped at ``sampler.n_minor_tasks``.
    step : int, optional
    verbose : bool

    Returns
    -------
    all_hiddens : torch.Tensor
        ``(L, n_positions, n_unique_tokens, n_tasks, batch_size, n_embd)``
    token_info : dict
    """
    _, _sampler_orig, config = nu.load_everything("latent", exp_name)

    if step is None:
        step = config.training.num_epochs

    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True
    )
    model.eval().to(config.device)

    sampler, k_minor, _ = get_latent_sampler(exp_name, n_minor, n_ood=0)

    if layers is None:
        layers = list(range(len(model.layers)))

    if verbose:
        logger.info(
            f"Computing non-padded token-conditioned hiddens for latent exp: {exp_name}"
        )
        logger.info(
            f"Layers: {layers}, Batch size: {batch_size}, "
            f"n_tasks: {sampler.n_major_tasks + sampler.n_minor_tasks}"
        )

    all_hiddens, token_info = compute_hiddens_token_conditioned_coin(
        config=config,
        model=model,
        sampler=sampler,
        layers=layers,
        batch_size=batch_size,
        positions_of_interest=positions_of_interest,
        max_unique_tokens=max_unique_tokens,
    )

    if verbose:
        logger.info(f"Computed hiddens shape: {all_hiddens.shape}")
        logger.info(f"Token info - positions: {token_info['positions']}")
        logger.info(f"Token info - n_unique_tokens: {token_info['n_unique_tokens']}")

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return all_hiddens, token_info


# ---------------------------------------------------------------------------
# 4.  plot_p1_variance
# ---------------------------------------------------------------------------

def plot_p1_variance(
    exp_name: str,
    layers: Optional[Sequence] = None,
    batch_size: int = 16,
    positions_of_interest: Optional[list] = None,
    max_unique_tokens: Optional[int] = None,
    n_minor: int = 0,
    step: Optional[int] = None,
    verbose: bool = False,
    eps: float = 1e-8,
    figsize: tuple = (8, 6),
    log_x: bool = True,
    show: bool = True,
    title: Optional[str] = None,
) -> dict:
    """
    Compute and plot P1 variance (normalized conditional residual variance)
    for the Latent Markov task on non-padded sequences.

    All-in-one convenience function that:

    1. Loads model/sampler.
    2. Collects token-conditioned hiddens via
       ``get_token_conditioned_hiddens``.
    3. Computes P1 variance via ``compute_p1_variance_multi_layer``.
    4. Plots normalised P1 variance vs position for each layer.

    Always uses ``n_ood=0``. ``n_minor`` is capped at ``sampler.n_minor_tasks``.

    Parameters
    ----------
    exp_name : str
    layers : list, optional
        ``None`` → all layers.
    batch_size : int
    positions_of_interest : list, optional
        ``None`` → all positions ``[0, seq_len-2]``.
    max_unique_tokens : int, optional
    n_minor : int
        Capped at ``sampler.n_minor_tasks``.
    step : int, optional
    verbose : bool
    eps : float
    figsize : tuple
    log_x : bool
        Use log scale on x-axis.
    show : bool
    title : str, optional

    Returns
    -------
    dict
        ``{'all_hiddens', 'token_info', 'results_dict', 'plotting_data',
        'fig', 'ax'}``.
    """
    import matplotlib.pyplot as plt

    all_hiddens, token_info = get_token_conditioned_hiddens(
        exp_name=exp_name,
        layers=layers,
        batch_size=batch_size,
        positions_of_interest=positions_of_interest,
        max_unique_tokens=max_unique_tokens,
        n_minor=n_minor,
        step=step,
        verbose=verbose,
    )

    L = all_hiddens.shape[0]
    if layers is None:
        _, _, config = nu.load_everything("latent", exp_name)
        layers = list(range(config.model.num_layers))

    results_dict = compute_p1_variance_multi_layer(
        all_hiddens=all_hiddens,
        token_info=token_info,
        layers=layers,
        eps=eps,
    )

    plotting_data = extract_p1_plotting_data_multi_layer(results_dict)

    if verbose:
        logger.info(f"Computed P1 variance for {len(results_dict)} layers")

    fig, ax = plt.subplots(figsize=figsize)

    for layer_idx in plotting_data['layers']:
        positions = plotting_data['positions']
        var_pos_norm = plotting_data['var_pos_norm'][layer_idx]
        ax.plot(
            positions, var_pos_norm, 'o-',
            label=f'Layer {layer_idx}',
            linewidth=2, markersize=6,
        )

    ax.set_xlabel('Position' + (' (log scale)' if log_x else ''), fontsize=16)
    ax.set_ylabel('Normalized P1 Variance', fontsize=16)
    if log_x:
        ax.set_xscale('log')
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()

    return {
        'all_hiddens': all_hiddens,
        'token_info': token_info,
        'results_dict': results_dict,
        'plotting_data': plotting_data,
        'fig': fig,
        'ax': ax,
    }


# ---------------------------------------------------------------------------
# 5.  plot_w_variance_vs_position  →  moved to legacy
# ---------------------------------------------------------------------------

from icl.latent_markov.legacy.w_variance_plot import plot_w_variance_vs_position  # noqa: F401, E402
