import gc

import torch
from typing import Optional

from torch import nn

import icl.utils.notebook_utils as nu
from icl.utils.unified_path_finder import get_exp_name
from icl.dyck.legacy.dyck_task_vec import get_dyck_sampler
from icl.dyck.dyck_utils import sample_binary_mask
from icl.utils.logger import setup_logger

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = setup_logger(__name__)


def train_linear_softmax_posterior_predictor_dyck_padded(
    exp_name: str,
    layer: int,
    B: int = 64,
    n_samples: int = 1000,
    step: Optional[int] = None,
    n_minor: Optional[int] = None,
    n_ood: int = 0,
    learning_rate: float = 0.01,
    num_epochs: int = 100,
    verbose: bool = False,
    dyck_mask: Optional[torch.Tensor] = None,
    n_masks: int = 1,
    min_dyck_positions: int = 0,
    max_dyck_positions: Optional[int] = None,
    validation_split: float = 0.2,
    uniform_sampling: bool = True,
    skip_baselines: bool = False,
) -> dict:
    """
    Train a linear softmax model to predict task posteriors from hidden representations.

    For Dyck task:
    1. Generates padded Dyck samples using one or more dyck_masks
    2. Computes task posteriors using dyck_task_posterior_over_time
    3. For each Dyck token position t_dyck (in the padded sequence):
       - Extracts the hidden state at t_pad = t_dyck + 1 (the pad slot right after)
       - Uses the posterior at real-token index t_dyck // 2 as the training target
    4. Trains a linear + softmax model to map hidden states to posteriors

    Parameters
    ----------
    exp_name : str
        Experiment name (e.g., "train_...")
    layer : int
        Layer index to extract hidden representations from
    B : int, default=64
        Batch size for sampling
    n_samples : int, default=1000
        Total number of sequences to generate for training (split across masks)
    step : int, optional
        Checkpoint step to load. If None, uses the final checkpoint.
    n_minor : int, optional
        Number of minor tasks to use. If None, uses all available minor tasks.
        If -1, uses no minor tasks (only OOD if n_ood > 0).
    n_ood : int, default=0
        Number of OOD tasks to include
    learning_rate : float, default=0.01
        Learning rate for training the linear model
    num_epochs : int, default=100
        Number of training epochs
    verbose : bool, default=False
        Whether to print progress messages
    dyck_mask : torch.Tensor, optional
        1D binary mask of length (seq_len+1)//2 indicating which real-token positions
        receive Dyck tokens. If provided, only this single mask is used and n_masks
        is ignored. If None, n_masks independent masks are sampled via
        sample_binary_mask.
    n_masks : int, default=1
        Number of independently-sampled dyck_masks to use for data collection.
        Samples are distributed evenly across masks. Ignored when dyck_mask is
        provided explicitly.
    min_dyck_positions : int, default=0
        Number of leading Dyck-token positions to skip (per mask). Useful for
        discarding early positions where the posterior is still near-uniform and
        uninformative. 0-indexed: ``min_dyck_positions=5`` skips positions 0-4.
    max_dyck_positions : int, optional
        If set, only Dyck-token positions up to index ``max_dyck_positions``
        (exclusive, per mask) are used. Combined with ``min_dyck_positions``,
        the positions used are ``[min_dyck_positions : max_dyck_positions]``.
        If None, all positions from ``min_dyck_positions`` onward are used.
    validation_split : float, default=0.2
        Fraction of data to use for validation.
    uniform_sampling : bool, default=True
        If True, modifies p_minor to achieve uniform sampling across all tasks.
    skip_baselines : bool, default=False
        If True, skips training permutation and logits baselines to save time.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'final_loss': float, final training loss
        - 'final_val_loss': float, final validation loss
        - 'loss_history': list of training losses
        - 'val_loss_history': list of validation losses
        - 'model': the trained linear model
        - 'baseline_final_loss': float, final training loss for permutation baseline
        - 'baseline_final_val_loss': float, final validation loss for permutation baseline
        - 'baseline_loss_history': list of training losses for baseline
        - 'baseline_val_loss_history': list of validation losses for baseline
        - 'baseline_model': the trained baseline model
        - 'logits_baseline_final_loss': float, final training loss for logits baseline
        - 'logits_baseline_final_val_loss': float, final validation loss for logits baseline
        - 'logits_baseline_loss_history': list of training losses for logits baseline
        - 'logits_baseline_val_loss_history': list of validation losses for logits baseline
        - 'logits_baseline_model': the trained logits baseline model
        - 'layer': int, the layer index used
        - 'n_tasks': int, number of tasks
        - 'dyck_masks': list of dyck_masks used
    """
    from icl.dyck.dyck import dyck_task_posterior_over_time
    import torch.optim as optim

    _, _sampler_orig, config = nu.load_everything("dyck", exp_name)

    if step is None:
        step = config.training.num_epochs

    model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
    model.eval()
    model.to(config.device)

    if n_minor is None:
        n_minor = 1000000
    elif n_minor == -1:
        n_minor = 0

    sampler, k_minor = get_dyck_sampler(exp_name, n_minor, n_ood)
    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks

    device = config.device

    if dyck_mask is not None:
        masks_list = [dyck_mask.to(device)]
    else:
        assert n_masks >= 1, f"n_masks must be >= 1, got {n_masks}"
        masks_list = [sample_binary_mask(config).to(device) for _ in range(n_masks)]

    original_p_minor = sampler.p_minor
    if uniform_sampling and sampler.n_minor_tasks > 0:
        sampler.p_minor = sampler.n_minor_tasks / (sampler.n_major_tasks + sampler.n_minor_tasks)
        if verbose:
            logger.info(f"Modified p_minor from {original_p_minor:.6f} to {sampler.p_minor:.6f} for uniform task sampling")
    elif verbose and not uniform_sampling:
        logger.info(f"Using original p_minor: {original_p_minor:.6f} (not modified for uniform sampling)")

    if verbose:
        logger.info(f"Training linear softmax model to predict Dyck posteriors from layer {layer} hidden representations")
        logger.info(f"Number of tasks: {n_tasks} (major: {sampler.n_major_tasks}, minor: {sampler.n_minor_tasks}), Batch size: {B}, Total samples: {n_samples}")
        logger.info(f"Using {len(masks_list)} mask(s) for data collection")

    num_masks = len(masks_list)
    n_samples_per_mask = (n_samples + num_masks - 1) // num_masks

    all_hiddens = []
    all_logits = []
    all_posteriors = []

    seq_len_padded = sampler.seq_len

    for mask_idx, current_mask in enumerate(masks_list):
        dyck_real_positions = torch.nonzero(current_mask == 1, as_tuple=True)[0]
        padded_positions_raw = (2 * dyck_real_positions + 1)

        valid = padded_positions_raw < seq_len_padded
        padded_positions = padded_positions_raw[valid].to(device=device, dtype=torch.long)
        posterior_indices = dyck_real_positions[valid].to(device=device, dtype=torch.long)

        start = min_dyck_positions
        end = max_dyck_positions
        padded_positions = padded_positions[start:end]
        posterior_indices = posterior_indices[start:end]

        n_positions = len(padded_positions)

        if n_positions == 0:
            if verbose:
                logger.info(f"  Mask {mask_idx+1}/{num_masks}: 0 valid Dyck positions — skipping")
            continue

        if verbose:
            logger.info(f"  Mask {mask_idx+1}/{num_masks}: {n_positions} Dyck positions, "
                        f"real-token indices {posterior_indices.tolist()}")

        n_batches = (n_samples_per_mask + B - 1) // B

        for batch_idx in range(n_batches):
            samples_raw, masks_raw = sampler.generate(
                mode="train", task=None, num_samples=B, epochs=1,
                dyck_mask=current_mask.clone(),
            )
            if samples_raw.dim() == 3:
                samples_raw = samples_raw.squeeze(0)
                masks_raw = masks_raw.squeeze(0)

            posteriors = dyck_task_posterior_over_time(
                sampler, samples_raw, masks_raw
            )
            posteriors_batch = posteriors[:, posterior_indices.cpu(), :]
            del posteriors

            samples = samples_raw.to(device)
            del samples_raw, masks_raw

            cache = {}
            layer_module = model.layers[layer].attn_block

            def hook_fn(module, inp, out, _pp=padded_positions):
                if torch.is_tensor(out):
                    cache["hidden"] = out.index_select(dim=1, index=_pp).detach()
                elif isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                    cache["hidden"] = out[0].index_select(dim=1, index=_pp).detach()
                else:
                    raise RuntimeError(f"Unsupported hook output type: {type(out)}")

            handle = layer_module.register_forward_hook(hook_fn)
            try:
                with torch.no_grad():
                    logits_full = model(samples)
                    logits_batch = logits_full.index_select(dim=1, index=padded_positions).cpu()
                    del logits_full
                hiddens_batch = cache["hidden"].cpu()
            finally:
                handle.remove()
                cache.clear()

            del samples
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            bsz = hiddens_batch.shape[0]
            all_hiddens.append(hiddens_batch.reshape(bsz * n_positions, -1))
            all_logits.append(logits_batch.reshape(bsz * n_positions, -1))
            all_posteriors.append(posteriors_batch.reshape(bsz * n_positions, -1))
            del hiddens_batch, logits_batch, posteriors_batch

    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    hiddens_flat = torch.cat(all_hiddens, dim=0)
    logits_flat = torch.cat(all_logits, dim=0)
    posteriors_flat = torch.cat(all_posteriors, dim=0)
    del all_hiddens, all_logits, all_posteriors

    n_total = hiddens_flat.shape[0]
    D = hiddens_flat.shape[1]
    vocab_size = logits_flat.shape[1]
    T = posteriors_flat.shape[1]

    n_train = int(n_total * (1 - validation_split))
    indices = torch.randperm(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    hiddens_train = hiddens_flat[train_indices]
    logits_train = logits_flat[train_indices]
    posteriors_train = posteriors_flat[train_indices]
    hiddens_val = hiddens_flat[val_indices]
    logits_val = logits_flat[val_indices]
    posteriors_val = posteriors_flat[val_indices]
    del hiddens_flat, logits_flat, posteriors_flat, indices

    if verbose:
        logger.info(f"Total data: {n_total} samples")
        logger.info(f"Training data: {n_train} samples ({100*(1-validation_split):.1f}%)")
        logger.info(f"Validation data: {len(val_indices)} samples ({100*validation_split:.1f}%)")
        logger.info(f"Training data shape: hiddens {hiddens_train.shape}, logits {logits_train.shape}, posteriors {posteriors_train.shape}")

    linear_model = nn.Sequential(
        nn.Linear(D, T, bias=True),
        nn.Softmax(dim=-1)
    ).to(device)

    optimizer = optim.Adam(linear_model.parameters(), lr=learning_rate)
    criterion = nn.KLDivLoss(reduction='batchmean')

    train_batch_size = min(2048, n_train)

    loss_history = []
    val_loss_history = []

    if verbose:
        logger.info(f"Training linear model for {num_epochs} epochs (mini-batch size {train_batch_size})...")

    for epoch in range(num_epochs):
        linear_model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_chunks = 0

        for start in range(0, n_train, train_batch_size):
            end = min(start + train_batch_size, n_train)
            idx = perm[start:end]
            h_batch = hiddens_train[idx].to(device)
            p_batch = posteriors_train[idx].to(device)

            optimizer.zero_grad()
            pred = linear_model(h_batch)
            log_pred = torch.log(pred + 1e-10)
            loss = criterion(log_pred, p_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * (end - start)
            n_chunks += 1
            del h_batch, p_batch, pred, log_pred, loss

        train_loss = epoch_loss / n_train

        linear_model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for start in range(0, len(val_indices), train_batch_size):
                end = min(start + train_batch_size, len(val_indices))
                h_val = hiddens_val[start:end].to(device)
                p_val = posteriors_val[start:end].to(device)
                pred_val = linear_model(h_val)
                log_pred_val = torch.log(pred_val + 1e-10)
                val_loss_sum += criterion(log_pred_val, p_val).item() * (end - start)
                del h_val, p_val, pred_val, log_pred_val

        val_loss = val_loss_sum / len(val_indices)

        loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if verbose and (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    final_loss = loss_history[-1]
    final_val_loss = val_loss_history[-1]

    if verbose:
        logger.info(f"Training completed. Final train loss: {final_loss:.6f}, Final val loss: {final_val_loss:.6f}")

    final_baseline_loss = float('nan')
    final_baseline_val_loss = float('nan')
    baseline_loss_history = []
    baseline_val_loss_history = []
    baseline_model = None

    final_logits_baseline_loss = float('nan')
    final_logits_baseline_val_loss = float('nan')
    logits_baseline_loss_history = []
    logits_baseline_val_loss_history = []
    logits_baseline_model = None

    if not skip_baselines:
        if verbose:
            logger.info("Training permutation baseline (shuffled posteriors)...")

        posteriors_train_shuffled = posteriors_train[torch.randperm(n_train)]
        posteriors_val_shuffled = posteriors_val[torch.randperm(len(val_indices))]

        baseline_model = nn.Sequential(
            nn.Linear(D, T, bias=True),
            nn.Softmax(dim=-1)
        ).to(device)

        baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=learning_rate)
        baseline_loss_history = []
        baseline_val_loss_history = []

        for epoch in range(num_epochs):
            baseline_model.train()
            perm = torch.randperm(n_train)
            epoch_loss_bl = 0.0
            for start in range(0, n_train, train_batch_size):
                end = min(start + train_batch_size, n_train)
                idx = perm[start:end]
                h_b = hiddens_train[idx].to(device)
                p_b = posteriors_train_shuffled[idx].to(device)
                baseline_optimizer.zero_grad()
                pred_bl = baseline_model(h_b)
                log_pred_bl = torch.log(pred_bl + 1e-10)
                loss_bl = criterion(log_pred_bl, p_b)
                loss_bl.backward()
                baseline_optimizer.step()
                epoch_loss_bl += loss_bl.item() * (end - start)
                del h_b, p_b, pred_bl, log_pred_bl, loss_bl

            baseline_model.eval()
            val_loss_bl_sum = 0.0
            with torch.no_grad():
                for start in range(0, len(val_indices), train_batch_size):
                    end = min(start + train_batch_size, len(val_indices))
                    h_v = hiddens_val[start:end].to(device)
                    p_v = posteriors_val_shuffled[start:end].to(device)
                    pred_v = baseline_model(h_v)
                    log_pred_v = torch.log(pred_v + 1e-10)
                    val_loss_bl_sum += criterion(log_pred_v, p_v).item() * (end - start)
                    del h_v, p_v, pred_v, log_pred_v

            baseline_loss_history.append(epoch_loss_bl / n_train)
            baseline_val_loss_history.append(val_loss_bl_sum / len(val_indices))

            if verbose and (epoch + 1) % 20 == 0:
                logger.info(f"  Baseline Epoch {epoch+1}/{num_epochs}, Train Loss: {baseline_loss_history[-1]:.6f}, Val Loss: {baseline_val_loss_history[-1]:.6f}")

        final_baseline_loss = baseline_loss_history[-1]
        final_baseline_val_loss = baseline_val_loss_history[-1]

        if verbose:
            logger.info(f"Baseline completed. Final train loss: {final_baseline_loss:.6f}, Final val loss: {final_baseline_val_loss:.6f}")
            logger.info(f"Improvement over baseline - Train: {final_loss - final_baseline_loss:.6f}, Val: {final_val_loss - final_baseline_val_loss:.6f}")

        del posteriors_train_shuffled, posteriors_val_shuffled

        if verbose:
            logger.info("Training logits baseline (predicting posteriors from logits)...")

        logits_baseline_model = nn.Sequential(
            nn.Linear(vocab_size, T, bias=True),
            nn.Softmax(dim=-1)
        ).to(device)

        logits_baseline_optimizer = optim.Adam(logits_baseline_model.parameters(), lr=learning_rate)
        logits_baseline_loss_history = []
        logits_baseline_val_loss_history = []

        for epoch in range(num_epochs):
            logits_baseline_model.train()
            perm = torch.randperm(n_train)
            epoch_loss_lg = 0.0
            for start in range(0, n_train, train_batch_size):
                end = min(start + train_batch_size, n_train)
                idx = perm[start:end]
                l_b = logits_train[idx].to(device)
                p_b = posteriors_train[idx].to(device)
                logits_baseline_optimizer.zero_grad()
                pred_lg = logits_baseline_model(l_b)
                log_pred_lg = torch.log(pred_lg + 1e-10)
                loss_lg = criterion(log_pred_lg, p_b)
                loss_lg.backward()
                logits_baseline_optimizer.step()
                epoch_loss_lg += loss_lg.item() * (end - start)
                del l_b, p_b, pred_lg, log_pred_lg, loss_lg

            logits_baseline_model.eval()
            val_loss_lg_sum = 0.0
            with torch.no_grad():
                for start in range(0, len(val_indices), train_batch_size):
                    end = min(start + train_batch_size, len(val_indices))
                    l_v = logits_val[start:end].to(device)
                    p_v = posteriors_val[start:end].to(device)
                    pred_v = logits_baseline_model(l_v)
                    log_pred_v = torch.log(pred_v + 1e-10)
                    val_loss_lg_sum += criterion(log_pred_v, p_v).item() * (end - start)
                    del l_v, p_v, pred_v, log_pred_v

            logits_baseline_loss_history.append(epoch_loss_lg / n_train)
            logits_baseline_val_loss_history.append(val_loss_lg_sum / len(val_indices))

            if verbose and (epoch + 1) % 20 == 0:
                logger.info(f"  Logits Baseline Epoch {epoch+1}/{num_epochs}, Train Loss: {logits_baseline_loss_history[-1]:.6f}, Val Loss: {logits_baseline_val_loss_history[-1]:.6f}")

        final_logits_baseline_loss = logits_baseline_loss_history[-1]
        final_logits_baseline_val_loss = logits_baseline_val_loss_history[-1]

        if verbose:
            logger.info(f"Logits baseline completed. Final train loss: {final_logits_baseline_loss:.6f}, Final val loss: {final_logits_baseline_val_loss:.6f}")
            logger.info(f"Comparison - Hiddens vs Logits - Train: {final_loss - final_logits_baseline_loss:.6f}, Val: {final_val_loss - final_logits_baseline_val_loss:.6f}")

    if hasattr(sampler, 'p_minor') and 'original_p_minor' in locals():
        sampler.p_minor = original_p_minor

    linear_model = linear_model.cpu()
    if baseline_model is not None:
        baseline_model = baseline_model.cpu()
    if logits_baseline_model is not None:
        logits_baseline_model = logits_baseline_model.cpu()

    return {
        'final_loss': final_loss,
        'final_val_loss': final_val_loss,
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'model': linear_model,
        'baseline_final_loss': final_baseline_loss,
        'baseline_final_val_loss': final_baseline_val_loss,
        'baseline_loss_history': baseline_loss_history,
        'baseline_val_loss_history': baseline_val_loss_history,
        'baseline_model': baseline_model,
        'logits_baseline_final_loss': final_logits_baseline_loss,
        'logits_baseline_final_val_loss': final_logits_baseline_val_loss,
        'logits_baseline_loss_history': logits_baseline_loss_history,
        'logits_baseline_val_loss_history': logits_baseline_val_loss_history,
        'logits_baseline_model': logits_baseline_model,
        'layer': layer,
        'n_tasks': n_tasks,
        'hidden_dim': D,
        'vocab_size': vocab_size,
        'n_samples': n_total,
        'n_train': n_train,
        'n_val': len(val_indices),
        'dyck_masks': [m.cpu() for m in masks_list],
    }


def plot_posterior_predictor_loss_vs_k_dyck_padded(
    k_values: list,
    layer: int,
    B: int = 64,
    n_samples: int = 3000,
    step: Optional[int] = None,
    n_minor: Optional[int] = None,
    n_ood: int = 0,
    learning_rate: float = 0.01,
    num_epochs: int = 200,
    verbose: bool = False,
    n_masks: int = 60,
    min_dyck_positions: int = 5,
    max_dyck_positions: Optional[int] = 15,
    validation_split: float = 0.2,
    uniform_sampling: bool = True,
    skip_baselines: bool = True,
    backend: str = "matplotlib",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> dict:
    """
    Train posterior predictors for different k values (log2 of number of minor tasks)
    and plot training and validation losses against k, for the Dyck task.

    Parameters
    ----------
    k_values : list
        List of k values where number of minor tasks = 2^k.
    layer : int
        Layer index to extract hidden representations from.
    B : int, default=64
        Batch size for sampling.
    n_samples : int, default=3000
        Total number of sequences to generate for training.
    step : int, optional
        Checkpoint step. If None, uses final checkpoint.
    n_minor : int, optional
        Number of minor tasks to use. None = all available, -1 = none.
    n_ood : int, default=0
        Number of OOD tasks to include.
    learning_rate : float, default=0.01
        Learning rate for training.
    num_epochs : int, default=200
        Number of training epochs.
    verbose : bool, default=False
        Whether to print progress messages.
    n_masks : int, default=60
        Number of independently-sampled dyck masks for data collection.
    min_dyck_positions : int, default=5
        Skip the first N Dyck positions (near-uniform, uninformative).
    max_dyck_positions : int, optional, default=15
        Use Dyck positions up to this index (exclusive).
    validation_split : float, default=0.2
        Fraction of data for validation.
    uniform_sampling : bool, default=True
        If True, uses uniform prior across all tasks.
    skip_baselines : bool, default=True
        If True, skips permutation and logits baselines.
    backend : str, default="matplotlib"
        Plotting backend: "matplotlib" or "plotly".
    figsize : tuple, default=(10, 6)
        Figure size for matplotlib.
    save_path : str, optional
        Path to save the figure.
    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    results : dict
        - 'k_values': list of k values
        - 'train_losses': list of final training losses
        - 'val_losses': list of final validation losses
        - 'fig': matplotlib or plotly Figure
    """
    train_losses = []
    val_losses = []

    for k in k_values:
        if verbose:
            logger.info(f"Processing k={k} (n_minor_tasks = {2**k})...")

        exp_name = get_exp_name("dyck", k)

        try:
            results = train_linear_softmax_posterior_predictor_dyck_padded(
                exp_name=exp_name,
                layer=layer,
                B=B,
                n_samples=n_samples,
                step=step,
                n_minor=n_minor,
                n_ood=n_ood,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                verbose=verbose,
                n_masks=n_masks,
                min_dyck_positions=min_dyck_positions,
                max_dyck_positions=max_dyck_positions,
                validation_split=validation_split,
                uniform_sampling=uniform_sampling,
                skip_baselines=skip_baselines,
            )

            train_losses.append(results['final_loss'])
            val_losses.append(results['final_val_loss'])

            if verbose:
                logger.info(f"  k={k}: Train Loss: {results['final_loss']:.6f}, Val Loss: {results['final_val_loss']:.6f}")

            del results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing k={k}: {e}")
            train_losses.append(float('nan'))
            val_losses.append(float('nan'))

    if backend == "matplotlib":
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(k_values, train_losses, 'o-', label='Training Loss', linewidth=2, markersize=8)
        ax.plot(k_values, val_losses, 's-', label='Validation Loss', linewidth=2, markersize=8)
        ax.set_xlabel('k (log2 of number of minor tasks)', fontsize=12)
        ax.set_ylabel('KL Divergence Loss', fontsize=12)
        ax.set_title(f'Posterior Predictor Loss vs k (Dyck Task, Layer {layer})', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    elif backend == "plotly":
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_values, y=train_losses,
            mode='lines+markers',
            name='Training Loss',
            line=dict(width=2),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=k_values, y=val_losses,
            mode='lines+markers',
            name='Validation Loss',
            line=dict(width=2),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title=f'Posterior Predictor Loss vs k (Dyck Task, Layer {layer})',
            xaxis_title='k (log2 of number of minor tasks)',
            yaxis_title='KL Divergence Loss',
            width=figsize[0]*100,
            height=figsize[1]*100,
        )

        if save_path:
            fig.write_image(save_path)
        if show:
            fig.show()
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return {
        'k_values': k_values,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'fig': fig,
    }


# ── backward-compat re-exports ──────────────────────────────────────
from icl.dyck.analysis.loss_plots import plot_id_ood_loss_dyck  # noqa: F401,E402
from icl.dyck.analysis.variance import (  # noqa: F401,E402
    compute_p1_variance_dyck_padded,
    compute_p1_variance_dyck,
    plot_p1_variance_dyck,
)
from icl.dyck.analysis.posterior_plots import (  # noqa: F401,E402
    plot_dyck_task_posterior_padded,
    plot_dyck_task_posterior,
)
