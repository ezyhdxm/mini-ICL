import torch
from typing import Optional

import icl.utils.notebook_utils as nu
from icl.utils.basic import get_hash
from icl.latent_markov.analysis.ood import get_all_samples, get_latent_sampler
from icl.linear.linear_ood_analysis import (
    _create_eval_task_pool,
    _setup_eval_task,
    setup_device,
)
from icl.latent_markov.task_vecs import compute_hiddens_onepos_all_layers_kvcache_beta
from icl.linear.linear_path_utils import load_model_task_config
from icl.linear.task_vecs import extract_hidden_multi
from icl.linear.legacy.task_vecs_padded import compute_hiddens_multi
from icl.utils.logger import setup_logger
from icl.utils.unified_path_finder import unified_get_config, get_exp_name  # noqa: F401
from icl.latent_markov.legacy.latent_task_vec import compute_hiddens
from icl.coin.coin_ood_analysis import get_new_sampler
from icl.dyck.legacy.dyck_task_vec import get_dyck_sampler, compute_hiddens_dyck
from icl.dyck.dyck_utils import sample_binary_mask

logger = setup_logger(__name__)


def _get_hiddens(
        task_name, 
        exp_name, 
        n_minor=64, 
        n_ood=30, 
        B=64,
        step: Optional[int] = None,
        force_recompute=False,
        verbose=False,
        **kwargs,
        ):

    if task_name == "latent":
        _, sampler, config = nu.load_everything("latent", exp_name)
        k_minor = min(n_minor, sampler.n_minor_tasks)
        if step is None:
            step = config.training.num_epochs

        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
        if verbose:
            logger.info("Getting samples...")
        all_samples, k_minor = get_all_samples(exp_name, n_minor=n_minor, n_ood=n_ood, B=B)
        if verbose:
            logger.info("Computing hiddens...")
        hiddens = compute_hiddens_onepos_all_layers_kvcache_beta(
                config, 
                model, 
                all_samples, 
                k_step = 32,
                b_step = 32,
                t_step = 4
            ).permute(0, 1, 3, 2, 4, 5) # (n_layers, n_tasks, num_states, T, B, D)
    
    elif task_name == "coin":
        _, sampler, config = nu.load_everything("coin", exp_name)
        k_minor = min(n_minor, sampler.n_minor_tasks)
        if step is None:
            step = config.training.num_epochs

        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
        if verbose:
            logger.info("Getting samples...")
        sampler_clone, k_minor = get_new_sampler(exp_name, n_minor, n_ood)
        if kwargs.get("return_data", False):
            hiddens, demo_data = compute_hiddens(config, model, sampler_clone, B, return_data=True)
        else:
            hiddens = compute_hiddens(config, model, sampler_clone, B)
        
        if kwargs.get("return_p", False):
            return hiddens, k_minor, torch.concat([sampler_clone.major_p, sampler_clone.minor_p])
        if kwargs.get("return_data", False):
            return hiddens, k_minor, demo_data, sampler_clone

    
    elif task_name == "dyck":
        _, sampler, config = nu.load_everything("dyck", exp_name)
        k_minor = min(n_minor, sampler.n_minor_tasks)
        if step is None:
            step = config.training.num_epochs
        
        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
        if verbose:
            logger.info("Getting samples...")
        sampler_clone, k_minor = get_dyck_sampler(exp_name, n_minor, n_ood)
        mask = sample_binary_mask(config)
        hiddens = compute_hiddens_dyck(config,
                    model,
                    sampler = sampler_clone,
                    dyck_mask = mask,
                    batch_size = B,
                    )
        return hiddens, k_minor, mask

    elif task_name == "linear":
        _, train_task, config = load_model_task_config(exp_name)
        k_minor = min(n_minor, train_task.n_minor_tasks)
        if step is None:
            step = config.training.total_steps
        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
        
        device = setup_device(None)
        if verbose:
            logger.info("Creating eval task pool...")
        
        eval_task_pool, k_minor = _create_eval_task_pool(
            train_task, 
            K=n_ood, 
            include_minor=True, 
             
            device=device,
            n_minor=n_minor,
        )
        eval_task = _setup_eval_task(config, eval_task_pool, B, device)
        if verbose:
            logger.info("Computing hiddens...")
        
        # Use smaller chunk_size for compute_hiddens_multi to reduce memory usage
        # Calculate chunk_size based on number of tasks and batch size to avoid OOM
        # For linear tasks, eval_task_pool is a tensor with shape (n_tasks, n_dims)
        n_tasks_estimate = eval_task_pool.shape[0] if isinstance(eval_task_pool, torch.Tensor) else (train_task.n_tasks + min(n_minor, train_task.n_minor_tasks) + n_ood)
        # Use smaller chunks for large numbers of tasks or large batch sizes
        # For B=96 and many tasks, use very small chunks
        if B >= 64 and n_tasks_estimate >= 64:
            chunk_size_hiddens = max(2, min(8, 32 // max(1, B // 32)))
        elif n_tasks_estimate >= 256:
            chunk_size_hiddens = max(2, min(8, 64 // max(1, n_tasks_estimate // 64)))
        else:
            chunk_size_hiddens = min(16, max(4, 64 // max(1, n_tasks_estimate // 64)))
        if verbose:
            logger.info(f"Using chunk_size={chunk_size_hiddens} for compute_hiddens_multi (n_tasks={n_tasks_estimate}, B={B})")
        
        hiddens, _ = compute_hiddens_multi(
            config, model, eval_task, 
            chunk_size=chunk_size_hiddens
        ) # (n_layers, n_tasks, T, B, D)
        
        # hiddens is already on CPU from compute_hiddens_multi
        
        # Clean up model and other GPU objects
        del model, eval_task, eval_task_pool, train_task, config
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return hiddens, k_minor


@torch.no_grad()
def _compute_hiddens_at_real_tokens(
    config,
    model,
    sampler,
    B,
    *,
    n_tasks=None,
    dyck_mask=None,
    return_data=False,
    verbose=False,
):
    """
    Extract hidden representations at real-token positions from non-padded sequences.

    Whereas ``compute_hiddens`` (latent_task_vec.py) extracts at **padding**
    positions ``2*i + 1`` in a padded sequence, this function extracts at
    positions ``[0, 1, ..., seq_len-2]`` – the real tokens that immediately
    precede where a padding token would have been inserted.

    Works for latent, coin, and dyck tasks whose models expose
    ``model.layers[l].attn_block``.

    Parameters
    ----------
    config : ConfigDict
    model : nn.Module
        Must have ``model.layers[l].attn_block``.
    sampler : object
        Must support ``.generate(mode, task, num_samples)`` and ``.seq_len``.
    B : int
        Batch size.
    n_tasks : int, optional
        Total number of tasks. Defaults to
        ``sampler.n_major_tasks + sampler.n_minor_tasks``.
    dyck_mask : torch.Tensor, optional
        Binary mask for dyck tasks (passed to ``sampler.generate``).
    return_data : bool
        If True, also return the raw (non-padded) sequences.
    verbose : bool

    Returns
    -------
    all_hiddens : torch.Tensor
        Shape ``(n_layers, n_tasks, seq_len-1, B, n_embd)``.
    data_collector : torch.Tensor  *(only when return_data=True)*
        Shape ``(n_tasks, B, seq_len)``.
    """
    device = config.device
    if n_tasks is None:
        n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    seq_len = sampler.seq_len
    n_embd = config.model.emb_dim
    n_layers = len(model.layers)

    # Key difference from padded version:
    #   padded   task_pos = 2 * arange(seq_len - 1) + 1   →  [1, 3, 5, ...]
    #   nonpadded task_pos = arange(seq_len - 1)           →  [0, 1, 2, ...]
    task_pos = torch.arange(seq_len - 1, device=device)

    all_hiddens = torch.empty(
        (n_layers, n_tasks, seq_len - 1, B, n_embd),
        device=device,
    )

    # ---- hook-based extraction (same pattern as compute_hiddens) ----
    def run_and_extract(batch_data):
        cache = {}
        handles = []
        for l in range(n_layers):
            def make_hook(layer_idx):
                def hook_fn(module, inp, out):
                    # out: (B, L, d)
                    cache[layer_idx] = out.index_select(1, task_pos).detach()
                return hook_fn
            h = model.layers[l].attn_block.register_forward_hook(make_hook(l))
            handles.append(h)

        _ = model(batch_data)

        for h in handles:
            h.remove()
        return cache  # layer_idx -> (B, P, d)

    if return_data:
        data_collector = torch.empty((n_tasks, B, seq_len), device=device)

    for i in range(n_tasks):
        gen_kwargs = dict(mode="testing", task=i, num_samples=B)
        if dyck_mask is not None:
            gen_kwargs["dyck_mask"] = dyck_mask.clone()

        demo_data, _ = sampler.generate(**gen_kwargs)

        if demo_data.device != device:
            demo_data = demo_data.to(device, non_blocking=True)

        cache = run_and_extract(demo_data)

        for l in range(n_layers):
            # (B, P, d) -> (P, B, d)
            all_hiddens[l, i] = cache[l].permute(1, 0, 2)

        if return_data:
            data_collector[i] = demo_data

        if verbose and (i == 0 or (i + 1) % 10 == 0 or i == n_tasks - 1):
            logger.info(f"[_compute_hiddens_at_real_tokens] task {i+1}/{n_tasks} done")

    if return_data:
        return all_hiddens, data_collector

    return all_hiddens


def _get_hiddens_at_real_positions(
        task_name,
        exp_name,
        n_minor=64,
        n_ood=30,
        B=64,
        step: Optional[int] = None,
        force_recompute=False,
        verbose=False,
        **kwargs,
        ):
    """
    Extract hidden representations at **real-token** positions (non-padded).

    This is the non-padded counterpart of :func:`_get_hiddens`.  Where
    ``_get_hiddens`` extracts at padding-token positions, this function
    extracts at the real token immediately **before** where a padding token
    would have been in the padded sequence.

    Sequence-position mapping
    -------------------------
    latent / coin / dyck (discrete-token tasks):
        padded:     ``[tok0, pad, tok1, pad, ..., tok_{n-1}]``
                    ``_get_hiddens`` extracts at pad positions ``[1, 3, 5, ...]``
        non-padded: ``[tok0, tok1, ..., tok_{n-1}]``
                    this fn extracts at real positions ``[0, 1, 2, ..., n-2]``

    linear (continuous-token task, ``pad="mapsto"``):
        padded:     ``[data0, PAD, tgt0, data1, PAD, tgt1, ...]``
                    ``_get_hiddens`` extracts at PAD positions ``[1, 4, 7, ...]``
        non-padded: this fn extracts at data positions ``[0, 3, 6, ...]``

    linear (``pad="none"``):
        sequence:   ``[data0, tgt0, data1, tgt1, ...]``
                    this fn extracts at data positions ``[0, 2, 4, ...]``

    Parameters / Returns
    --------------------
    Same signature and return convention as :func:`_get_hiddens`.

    For latent/coin/dyck the output hiddens shape is
    ``(n_layers, n_tasks, seq_len-1, B, D)`` (no ``V`` / ``num_states`` axis).

    For linear the output hiddens shape is
    ``(n_layers, n_tasks, n_points, B, D)`` (same as ``_get_hiddens``).
    """

    device_override = kwargs.get("device", None)

    if task_name == "latent":
        _, sampler, config = nu.load_everything("latent", exp_name)
        if device_override is not None:
            config.device = device_override
        k_minor = min(n_minor, sampler.n_minor_tasks)
        if step is None:
            step = config.training.num_epochs

        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)

        if verbose:
            logger.info("Getting latent sampler...")
        sampler_clone, k_minor, _ = get_latent_sampler(exp_name, n_minor, n_ood)

        if verbose:
            logger.info("Computing non-padded hiddens for latent task...")
        if kwargs.get("return_data", False):
            hiddens, demo_data = _compute_hiddens_at_real_tokens(
                config, model, sampler_clone, B,
                return_data=True, verbose=verbose,
            )
            return hiddens, k_minor, demo_data, sampler_clone
        else:
            hiddens = _compute_hiddens_at_real_tokens(
                config, model, sampler_clone, B, verbose=verbose,
            )

    elif task_name == "coin":
        _, sampler, config = nu.load_everything("coin", exp_name)
        if device_override is not None:
            config.device = device_override
        k_minor = min(n_minor, sampler.n_minor_tasks)
        if step is None:
            step = config.training.num_epochs

        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)

        if verbose:
            logger.info("Getting coin sampler...")
        sampler_clone, k_minor = get_new_sampler(exp_name, n_minor, n_ood)

        if verbose:
            logger.info("Computing non-padded hiddens for coin task...")
        if kwargs.get("return_data", False):
            hiddens, demo_data = _compute_hiddens_at_real_tokens(
                config, model, sampler_clone, B,
                return_data=True, verbose=verbose,
            )
        else:
            hiddens = _compute_hiddens_at_real_tokens(
                config, model, sampler_clone, B, verbose=verbose,
            )

        if kwargs.get("return_p", False):
            return hiddens, k_minor, torch.concat([sampler_clone.major_p, sampler_clone.minor_p])
        if kwargs.get("return_data", False):
            return hiddens, k_minor, demo_data, sampler_clone

    elif task_name == "dyck":
        _, sampler, config = nu.load_everything("dyck", exp_name)
        if device_override is not None:
            config.device = device_override
        k_minor = min(n_minor, sampler.n_minor_tasks)
        if step is None:
            step = config.training.num_epochs

        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)

        if verbose:
            logger.info("Getting dyck sampler...")
        sampler_clone, k_minor = get_dyck_sampler(exp_name, n_minor, n_ood)
        mask = sample_binary_mask(config)

        if verbose:
            logger.info("Computing non-padded hiddens for dyck task...")
        hiddens = _compute_hiddens_at_real_tokens(
            config, model, sampler_clone, B,
            dyck_mask=mask, verbose=verbose,
        )
        return hiddens, k_minor, mask

    elif task_name == "linear":
        _, train_task, config = load_model_task_config(exp_name)
        if device_override is not None:
            config.device = device_override
        k_minor = min(n_minor, train_task.n_minor_tasks)
        if step is None:
            step = config.training.total_steps
        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)

        device = setup_device(device_override)
        if verbose:
            logger.info("Creating eval task pool...")

        eval_task_pool, k_minor = _create_eval_task_pool(
            train_task,
            K=n_ood,
            include_minor=True,
            
            device=device,
            n_minor=n_minor,
        )
        eval_task = _setup_eval_task(config, eval_task_pool, B, device)

        if verbose:
            logger.info("Computing non-padded hiddens for linear task...")

        # --- Determine task_pos at the DATA token (real token before PAD) ---
        n_points = config.task.n_points
        pad_mode = getattr(model, "pad", "mapsto")
        if pad_mode == "mapsto":
            # mapsto: [data0, PAD, tgt0, data1, PAD, tgt1, ...]
            #          pos 0   1    2     3      4    5
            # PAD is at 3*i+1;  data (real token before PAD) is at 3*i
            task_pos = 3 * torch.arange(n_points, device=device)
        elif pad_mode == "none":
            # none: [data0, tgt0, data1, tgt1, ...]
            #        pos 0   1     2      3
            # data positions are 2*i
            task_pos = 2 * torch.arange(n_points, device=device)
        elif pad_mode == "bos":
            # bos: [BOS, data0, tgt0, data1, tgt1, ...]
            #       pos 0  1     2     3      4
            # data positions are 2*i+1
            task_pos = 2 * torch.arange(n_points, device=device) + 1
        else:
            raise ValueError(f"Unknown pad mode: {pad_mode}")

        # --- Chunked extraction (mirrors compute_hiddens_multi logic) ---
        n_tasks = eval_task.task_pool.shape[0]
        batch_size = eval_task.batch_size
        n_embd = config.model.n_embd
        layers = list(range(config.model.n_layer))
        L = len(layers)

        # Chunk-size heuristic (same as _get_hiddens)
        n_tasks_estimate = n_tasks
        if B >= 64 and n_tasks_estimate >= 64:
            chunk_size_hiddens = max(2, min(8, 32 // max(1, B // 32)))
        elif n_tasks_estimate >= 256:
            chunk_size_hiddens = max(2, min(8, 64 // max(1, n_tasks_estimate // 64)))
        else:
            chunk_size_hiddens = min(16, max(4, 64 // max(1, n_tasks_estimate // 64)))

        if verbose:
            logger.info(
                f"Using chunk_size={chunk_size_hiddens} for non-padded linear "
                f"(n_tasks={n_tasks_estimate}, B={B}, pad={pad_mode}, "
                f"task_pos={task_pos.tolist()[:4]}...)"
            )

        output_shape = (L, n_tasks, n_points, batch_size, n_embd)
        all_hiddens = torch.empty(output_shape, dtype=torch.float32, device="cpu")

        demo_data = eval_task.sample_data(step=step)  # (batch_size, n_points, n_dims)

        for i in range(0, n_tasks, chunk_size_hiddens):
            chunk_end = min(i + chunk_size_hiddens, n_tasks)
            chunk_size_actual = chunk_end - i

            demo_data_repeated = (
                demo_data.unsqueeze(0)
                .expand(chunk_size_actual, batch_size, n_points, -1)
                .reshape(-1, n_points, demo_data.size(-1))
            )  # (chunk*B, n_points, n_dims)

            demo_target = eval_task.evaluate(
                demo_data,
                eval_task.task_pool[i:chunk_end].squeeze(-1).T,
                step=step,
            )
            if demo_target.ndim == 3:
                demo_target = demo_target.permute(2, 0, 1).reshape(-1, n_points)

            chunk_hiddens = extract_hidden_multi(
                model=model,
                demo_data=demo_data_repeated,
                demo_target=demo_target,
                layers=layers,
                task_pos=task_pos,
            )
            # (L, chunk*B, n_points, D) -> (L, chunk, B, n_points, D)
            #                           -> (L, chunk, n_points, B, D)
            chunk_hiddens = chunk_hiddens.reshape(
                L, chunk_size_actual, batch_size, n_points, n_embd
            )
            chunk_hiddens = chunk_hiddens.permute(0, 1, 3, 2, 4)
            all_hiddens[:, i:chunk_end] = chunk_hiddens.cpu()

            del chunk_hiddens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        hiddens = all_hiddens.detach()

        # Clean up GPU objects
        del model, eval_task, eval_task_pool, train_task, config
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return hiddens, k_minor


from icl.utils.legacy.unified_interface_padded import get_task_variance_coin  # noqa: F401

from icl.latent_markov.legacy.unified_latent import get_task_variance_latent  # noqa: F401

from icl.utils.legacy.unified_interface_padded import compute_stable_rank_at_padded_positions  # noqa: F401
from icl.utils.legacy.unified_interface_padded import compute_logits_multi_coin_latent  # noqa: F401
from icl.utils.legacy.unified_interface_padded import compute_stable_rank_logits_at_padded_positions  # noqa: F401

from icl.utils.legacy.unified_interface_padded import plot_coin_task_posterior  # noqa: F401
from icl.latent_markov.legacy.unified_latent import plot_latent_task_posterior  # noqa: F401
from icl.latent_markov.legacy.unified_latent import plot_posterior_predictor_loss_vs_k_latent  # noqa: F401

# ── backward-compat re-exports ──────────────────────────────────────
from icl.coin.analysis._helpers import get_token_conditioned_hiddens_coin  # noqa: F401,E402
from icl.coin.legacy.softmax_probes import train_linear_softmax_posterior_predictor_coin  # noqa: F401,E402
from icl.coin.legacy.padded_probes import plot_posterior_predictor_loss_vs_k_coin  # noqa: F401,E402
from icl.dyck.analysis.probes import (  # noqa: F401,E402
    train_linear_softmax_posterior_predictor_dyck_padded,
    plot_id_ood_loss_dyck,
    compute_p1_variance_dyck_padded,
    compute_p1_variance_dyck,
    plot_p1_variance_dyck,
    plot_dyck_task_posterior_padded,
    plot_dyck_task_posterior,
    plot_posterior_predictor_loss_vs_k_dyck_padded,
)
from icl.dyck.legacy.dyck_analysis_legacy import (  # noqa: F401,E402
    train_linear_softmax_posterior_predictor_dyck,
    plot_posterior_predictor_loss_vs_k_dyck,
    plot_mi_vs_k_dyck,
    plot_max_stable_rank_vs_k_dyck,
)
from icl.latent_markov.analysis.probes import (  # noqa: F401,E402
    train_linear_softmax_posterior_predictor,
    train_linear_hidden_predictor,
    train_mlp_hidden_predictor,
)
from icl.linear.analysis.probes import (  # noqa: F401,E402
    get_token_conditioned_hiddens,
    get_task_variance,
    train_linear_softmax_posterior_predictor_linear,
    plot_posterior_predictor_loss_vs_k_linear,
)
from icl.utils.legacy.unified_stable_rank import (  # noqa: F401,E402
    plot_stable_rank_vs_positions,
    plot_max_stable_rank_vs_k,
    plot_max_stable_rank_all_layers_vs_k,
    plot_max_stable_rank_logits_vs_k,
    plot_stable_rank_final_layer_weight_vs_k,
)


    

from icl.models.base_models import Transformer
from icl.utils.train import train_model_with_plot
from icl.linear.train_linear import train

def unified_train(
    task_name,
    k: int,
    vocab_size: int = 8,
    log2: bool = True,
    pad = None,
    major_pool_type: str = None,
    major_means = None,
    quiet: bool = True,
):
    import os
    _prev_wandb_silent = os.environ.get("WANDB_SILENT")
    if quiet:
        os.environ["WANDB_SILENT"] = "true"

    try:
        config = unified_get_config(task_name)
        if k >= 0:
            if log2:
                config.task.n_minor_tasks = 2 ** k
            else:
                config.task.n_minor_tasks = k
        else:
            config.task.n_minor_tasks = 1
            config.task.p_minor = 1e-12  # Practically no minor tasks
        if pad is not None:
            if task_name == "linear":
                config.model.pad = pad  # "bos", "mapsto", or "none"
        if major_pool_type is not None:
            config.task.major_pool_type = major_pool_type
        if major_means is not None:
            config.task.major_means = list(major_means)
        if task_name == "linear":
            return train(config)
        else:
            config.vocab_size = vocab_size
            model = Transformer(config)
            model = model.to(config.device)
            return train_model_with_plot(model, config, show=False, verbose=False)
    finally:
        if _prev_wandb_silent is None:
            os.environ.pop("WANDB_SILENT", None)
        else:
            os.environ["WANDB_SILENT"] = _prev_wandb_silent
