"""Task and P1 variance analysis for the Coin task."""

import gc
import torch
from typing import Optional

import icl.utils.notebook_utils as nu
from icl.coin.coin_ood_analysis import get_new_sampler
from icl.utils.logger import setup_logger
from icl.linear.task_variance import compute_task_variance_multi_layer, extract_plotting_data_multi_layer
from icl.linear.p1_variance import compute_p1_variance_multi_layer
from icl.linear.p1_variance import extract_plotting_data_multi_layer as extract_p1_plotting_data_multi_layer
from icl.coin.analysis._helpers import (
    compute_hiddens_multi_coin,
    get_token_conditioned_hiddens_coin,
)

logger = setup_logger(__name__)


def get_task_variance_coin(
    exp_name: str,
    layers: Optional[list] = None,
    batch_size: int = 64,
    positions_of_interest: Optional[list] = None,
    n_minor: int = 64,
    step: Optional[int] = None,
    verbose: bool = False,
    eps: float = 1e-8,
) -> tuple:
    """
    Get hidden representations and compute task variance for Coin task
    on **non-padded** sequences.

    Always uses ``n_ood=0``.  If ``n_minor`` exceeds the sampler's actual
    minor-task count it is silently capped.

    Steps:
      1. Load model and create sampler with ``n_minor`` minor tasks (no OOD).
      2. For each task, sample a new batch and extract hiddens at the
         requested positions ``[0, 1, 2, ...]``.
      3. Compute task variance: variance of batch-averaged hiddens across tasks.

    Parameters
    ----------
    exp_name : str
    layers : list, optional
        ``None`` → all layers.
    batch_size : int
    positions_of_interest : list, optional
        ``None`` → all positions.
    n_minor : int
        Number of minor tasks.  Capped at ``sampler.n_minor_tasks``.
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
    _, _sampler_orig, config = nu.load_everything("coin", exp_name)

    if step is None:
        step = config.training.num_epochs

    model, _ = nu.load_checkpoint(
        config, step=step, exp_name=exp_name, return_actual_step=True
    )
    model.eval().to(config.device)

    sampler, k_minor = get_new_sampler(exp_name, n_minor, n_ood=0)

    if layers is None:
        layers = list(range(len(model.layers)))

    if verbose:
        logger.info(f"Computing task variance for coin exp: {exp_name}")
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

    model.cpu(); del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return all_hiddens, position_info, results_dict, plotting_data


def plot_task_variance_coin(
    exp_name: str,
    layers: Optional[list] = None,
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
    Compute and plot normalised task variance for the Coin task on
    non-padded sequences.

    Convenience wrapper around ``get_task_variance_coin``.

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

    all_hiddens, position_info, results_dict, plotting_data = \
        get_task_variance_coin(
            exp_name=exp_name,
            layers=layers,
            batch_size=batch_size,
            positions_of_interest=positions_of_interest,
            n_minor=n_minor,
            step=step,
            verbose=verbose,
            eps=eps,
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

    if title:
        ax.set_title(title, fontsize=16)

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


def plot_p1_variance_coin(
    exp_name: str,
    layers: Optional[list] = None,
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
    for the Coin task.

    All-in-one convenience function that:

    1. Loads model/sampler.
    2. Collects token-conditioned hiddens via
       ``get_token_conditioned_hiddens_coin``.
    3. Computes P1 variance via ``compute_p1_variance_multi_layer``.
    4. Plots normalised P1 variance vs position for each layer.

    Always uses ``n_ood=0``.  ``n_minor`` is capped at
    ``sampler.n_minor_tasks``.

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

    # Step 1-2: get token-conditioned hiddens
    all_hiddens, token_info = get_token_conditioned_hiddens_coin(
        exp_name=exp_name,
        layers=layers,
        batch_size=batch_size,
        positions_of_interest=positions_of_interest,
        max_unique_tokens=max_unique_tokens,
        n_minor=n_minor,
        step=step,
        verbose=verbose,
    )

    # Reconstruct layers list from hiddens shape if needed
    L = all_hiddens.shape[0]
    if layers is None:
        _, _, config = nu.load_everything("coin", exp_name)
        layers = list(range(config.model.num_layers))

    # Step 3: compute P1 variance
    results_dict = compute_p1_variance_multi_layer(
        all_hiddens=all_hiddens,
        token_info=token_info,
        layers=layers,
        eps=eps,
    )

    # Extract plotting data
    plotting_data = extract_p1_plotting_data_multi_layer(results_dict)

    if verbose:
        logger.info(f"Computed P1 variance for {len(results_dict)} layers")

    # Step 4: plot
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

    if title:
        ax.set_title(title, fontsize=16)

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
