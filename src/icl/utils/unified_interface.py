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
from icl.utils.logger import setup_logger
from icl.utils.unified_path_finder import unified_get_config, get_exp_name  # noqa: F401
from icl.coin.coin_ood_analysis import get_new_sampler
from icl.dyck.legacy.dyck_task_vec import get_dyck_sampler, compute_hiddens_dyck
from icl.dyck.dyck_utils import sample_binary_mask

logger = setup_logger(__name__)


def _make_linear_eval_generator(device: str, train_task, step: int) -> torch.Generator:
    """Deterministic generator for linear eval-pool construction."""
    base_seed = int(getattr(train_task, "task_seed", 0))
    gen = torch.Generator(device=device)
    gen.manual_seed(base_seed + 1_000_003 + int(step))
    return gen


def _get_hiddens(
        task_name,
        exp_name,
        n_minor=64,
        n_ood=30,
        B=64,
        step: Optional[int] = None,
        force_recompute=False,
        verbose=False,
        device: Optional[str] = None,
        **kwargs,
):

    if task_name == "latent":
        # Load config + sampler without wasting GPU on the final-checkpoint model
        sampler, config = nu.load_config_and_sampler("latent", exp_name)
        k_minor = min(n_minor, sampler.n_minor_tasks)
        if step is None:
            step = config.training.num_epochs

        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
        if device is not None:
            config.device = device
            model.to(device)
        if verbose:
            logger.info("Getting samples...")
        # Pass the already-loaded sampler to avoid a second load_everything call
        all_samples, k_minor = get_all_samples(exp_name, n_minor=n_minor, n_ood=n_ood, B=B, sampler=sampler)
        if verbose:
            logger.info("Computing hiddens...")
        hiddens = compute_hiddens_onepos_all_layers_kvcache_beta(
                config,
                model,
                all_samples,
                k_step=32,
                b_step=32,
                t_step=4,
            ).permute(0, 1, 3, 2, 4, 5)  # (n_layers, n_tasks, num_states, T, B, D) — already on CPU

        # Free the model from GPU now that hiddens are on CPU
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elif task_name == "coin":
        # Load config + sampler without wasting GPU on the final-checkpoint model
        sampler, config = nu.load_config_and_sampler("coin", exp_name)
        k_minor = min(n_minor, sampler.n_minor_tasks)
        if step is None:
            step = config.training.num_epochs

        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
        if device is not None:
            config.device = device
            model.to(device)
        if verbose:
            logger.info("Getting samples...")
        # Pass the already-loaded sampler to avoid a second load_everything call
        sampler_clone, k_minor = get_new_sampler(exp_name, n_minor, n_ood, sampler=sampler)
        if device is not None:
            sampler_clone.to(device)
        if kwargs.get("return_data", False):
            hiddens, demo_data = _compute_hiddens_at_real_tokens(
                config, model, sampler_clone, B, return_data=True
            )
            demo_data = demo_data.cpu()
        else:
            hiddens = _compute_hiddens_at_real_tokens(config, model, sampler_clone, B)

        # Move hiddens to CPU and free the model from GPU
        hiddens = hiddens.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if kwargs.get("return_p", False):
            return hiddens, k_minor, torch.concat([sampler_clone.major_p.cpu(), sampler_clone.minor_p.cpu()])
        if kwargs.get("return_data", False):
            return hiddens, k_minor, demo_data, sampler_clone

    
    elif task_name == "dyck":
        _, sampler, config = nu.load_everything("dyck", exp_name)
        k_minor = min(n_minor, sampler.n_minor_tasks)
        if step is None:
            step = config.training.num_epochs

        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
        if device is not None:
            config.device = device
            model.to(device)
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

        # Determine target device BEFORE loading the checkpoint so the model
        # is mapped directly onto it.  Also move train_task tensors so that
        # _create_eval_task_pool doesn't mix devices (e.g. task_scale).
        dev = device if device is not None else setup_device(None)
        config.device = dev
        if hasattr(train_task, "to"):
            train_task.to(dev)
        else:
            for attr in ("task_pool", "minor_pool", "task_scale"):
                val = getattr(train_task, attr, None)
                if isinstance(val, torch.Tensor):
                    setattr(train_task, attr, val.to(dev))

        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)
        model.to(dev)
        # TransformerLin stores self.device as a plain string set at construction
        # time; model.to() moves weights but does NOT update it, causing the forward
        # pass to send inputs to the wrong device.  Patch it explicitly here.
        if hasattr(model, "device"):
            model.device = dev
        if verbose:
            logger.info("Creating eval task pool...")

        eval_gen = _make_linear_eval_generator(dev, train_task, step)
        eval_task_pool, k_minor = _create_eval_task_pool(
            train_task,
            K=n_ood,
            include_minor=True,
            device=dev,
            n_minor=n_minor,
            generator=eval_gen,
        )
        eval_task = _setup_eval_task(config, eval_task_pool, B, dev)
        if verbose:
            logger.info("Computing hiddens...")

        # Determine real-token positions based on pad mode (no legacy padded logic)
        n_points = config.task.n_points
        n_tasks = eval_task.task_pool.shape[0]
        batch_size = eval_task.batch_size
        n_embd = config.model.n_embd
        layers = list(range(config.model.n_layer))
        L = len(layers)

        _ep = kwargs.get("extraction_point", "post_attn")

        pad_mode = getattr(model, "pad", "mapsto")
        if pad_mode == "mapsto":
            # [data, PAD, tgt, ...] — extract at data positions (real input tokens)
            task_pos = 3 * torch.arange(n_points, device=dev)
        elif pad_mode == "none":
            # [data, tgt, data, tgt, ...] — extract at data positions
            task_pos = 2 * torch.arange(n_points, device=dev)
        elif pad_mode == "bos":
            # [BOS, data, tgt, ...] — extract at data positions (after BOS)
            task_pos = 2 * torch.arange(n_points, device=dev) + 1
        else:
            raise ValueError(f"Unknown pad mode for linear model: {pad_mode!r}")

        # Chunk size heuristic (same as before, avoids OOM)
        n_tasks_estimate = n_tasks
        if B >= 64 and n_tasks_estimate >= 64:
            chunk_size_hiddens = max(2, min(8, 32 // max(1, B // 32)))
        elif n_tasks_estimate >= 256:
            chunk_size_hiddens = max(2, min(8, 64 // max(1, n_tasks_estimate // 64)))
        else:
            chunk_size_hiddens = min(16, max(4, 64 // max(1, n_tasks_estimate // 64)))
        if verbose:
            logger.info(
                f"Linear hidden extraction: pad_mode={pad_mode!r}, "
                f"task_pos={task_pos.tolist()[:4]}..., chunk_size={chunk_size_hiddens}"
            )

        output_shape = (L, n_tasks, n_points, batch_size, n_embd)
        all_hiddens = torch.empty(output_shape, dtype=torch.float32, device="cpu")
        demo_data = eval_task.sample_data(step=step)

        for i in range(0, n_tasks, chunk_size_hiddens):
            chunk_end = min(i + chunk_size_hiddens, n_tasks)
            chunk_size_actual = chunk_end - i

            demo_data_repeated = (
                demo_data.unsqueeze(0)
                .expand(chunk_size_actual, batch_size, n_points, -1)
                .reshape(-1, n_points, demo_data.size(-1))
            )
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
                extraction_point=_ep,
            )  # (L, chunk*B, n_points, D)
            chunk_hiddens = chunk_hiddens.reshape(
                L, chunk_size_actual, batch_size, n_points, n_embd
            ).permute(0, 1, 3, 2, 4)  # (L, chunk, n_points, B, D)
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


@torch.no_grad()
def pregenerate_task_sequences(
    sampler,
    B: int,
    n_tasks: Optional[int] = None,
    dyck_mask=None,
) -> torch.Tensor:
    """Pre-generate all task sequences once on CPU for reuse across checkpoints.

    Parameters
    ----------
    sampler : object
        Must support ``.generate(mode, task, num_samples)`` and ``.seq_len``.
    B : int
        Number of sequences per task.
    n_tasks : int, optional
        Total number of tasks. Defaults to
        ``sampler.n_major_tasks + sampler.n_minor_tasks``.
    dyck_mask : torch.Tensor, optional
        Binary mask for dyck tasks.

    Returns
    -------
    precomputed_data : torch.Tensor, shape ``(n_tasks, B, seq_len)`` on CPU.
        Memory footprint: n_tasks × B × seq_len × 8 bytes (int64).
        For typical settings (97 tasks, B=64, seq_len=128) ≈ 6 MB.
    """
    if n_tasks is None:
        n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    seq_len = sampler.seq_len
    precomputed = torch.empty((n_tasks, B, seq_len), dtype=torch.long)
    for i in range(n_tasks):
        gen_kwargs = dict(mode="testing", task=i, num_samples=B)
        if dyck_mask is not None:
            gen_kwargs["dyck_mask"] = dyck_mask.clone()
        demo_data, _ = sampler.generate(**gen_kwargs)
        precomputed[i] = demo_data.cpu()
    return precomputed


def _compute_hiddens_at_real_tokens(
    config,
    model,
    sampler,
    B,
    *,
    n_tasks=None,
    dyck_mask=None,
    return_data=False,
    post_layernorm=False,
    extraction_point: str = "post_attn",
    verbose=False,
    task_batch_size=None,
    precomputed_data: Optional[torch.Tensor] = None,
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
    task_batch_size : int or None
        Number of tasks to batch together per forward pass for better GPU
        utilization.  ``None`` (default) batches all tasks at once.  Use 1
        for the legacy one-task-per-forward-pass behaviour.
    precomputed_data : torch.Tensor, optional
        Pre-generated sequences of shape ``(n_tasks, B, seq_len)`` on CPU,
        produced by :func:`pregenerate_task_sequences`.  When provided,
        ``sampler.generate`` is skipped entirely, hiding its CPU overhead
        from the GPU forward pass.  ``dyck_mask`` is ignored when this is set.

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

    task_pos = torch.arange(seq_len - 1, device=device)

    all_hiddens = torch.empty(
        (n_layers, n_tasks, seq_len - 1, B, n_embd),
        device=device,
    )

    _ep = extraction_point

    def _apply_next_ln(h, l):
        if _ep == "post_attn":
            mlp_block = model.layers[l].mlp
            if hasattr(mlp_block, "ln2"):
                return mlp_block.ln2(h)
        elif _ep == "post_mlp":
            if l + 1 < n_layers:
                next_blk = model.layers[l + 1]
                if hasattr(next_blk, "attn_block") and hasattr(next_blk.attn_block, "ln1"):
                    return next_blk.attn_block.ln1(h)
        return h

    def run_and_extract(batch_data):
        cache = {}
        handles = []
        for l in range(n_layers):
            def make_hook(layer_idx):
                def hook_fn(module, inp, out):
                    h = out.index_select(1, task_pos).detach()
                    if post_layernorm:
                        h = _apply_next_ln(h, layer_idx)
                    cache[layer_idx] = h
                return hook_fn

            if _ep == "post_mlp":
                h = model.layers[l].register_forward_hook(make_hook(l))
            else:
                h = model.layers[l].attn_block.register_forward_hook(make_hook(l))
            handles.append(h)

        with torch.no_grad():
            _ = model(batch_data)

        for h in handles:
            h.remove()
        return cache

    if return_data:
        data_collector = torch.empty((n_tasks, B, seq_len), device=device)

    if task_batch_size is None:
        task_batch_size = n_tasks
    task_batch_size = max(1, task_batch_size)

    for chunk_start in range(0, n_tasks, task_batch_size):
        chunk_end = min(chunk_start + task_batch_size, n_tasks)
        chunk_size = chunk_end - chunk_start

        chunk_data = []
        for i in range(chunk_start, chunk_end):
            if precomputed_data is not None:
                demo_data = precomputed_data[i].to(device, non_blocking=True)
            else:
                gen_kwargs = dict(mode="testing", task=i, num_samples=B)
                if dyck_mask is not None:
                    gen_kwargs["dyck_mask"] = dyck_mask.clone()
                demo_data, _ = sampler.generate(**gen_kwargs)
                if demo_data.device != device:
                    demo_data = demo_data.to(device, non_blocking=True)
            chunk_data.append(demo_data)

            if return_data:
                data_collector[i] = demo_data

        if chunk_size == 1:
            cache = run_and_extract(chunk_data[0])
            for l in range(n_layers):
                all_hiddens[l, chunk_start] = cache[l].permute(1, 0, 2)
        else:
            big_batch = torch.cat(chunk_data, dim=0)
            cache = run_and_extract(big_batch)
            for l in range(n_layers):
                # (chunk_size*B, P, d) → (chunk_size, B, P, d) → (chunk_size, P, B, d)
                reshaped = cache[l].view(chunk_size, B, seq_len - 1, n_embd)
                all_hiddens[l, chunk_start:chunk_end] = reshaped.permute(0, 2, 1, 3)
            del big_batch
        del chunk_data, cache

        if verbose and (chunk_end == n_tasks or chunk_start == 0
                        or chunk_end % (task_batch_size * 5) == 0):
            logger.info(
                f"[_compute_hiddens_at_real_tokens] "
                f"tasks {chunk_start+1}-{chunk_end}/{n_tasks} done"
            )

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
    _post_ln = kwargs.get("post_layernorm", False)
    _ep = kwargs.get("extraction_point", "post_attn")

    if task_name == "latent":
        # Load config + sampler without wasting GPU on the final-checkpoint model
        sampler, config = nu.load_config_and_sampler("latent", exp_name)
        if device_override is not None:
            config.device = device_override
        k_minor = min(n_minor, sampler.n_minor_tasks)
        if step is None:
            step = config.training.num_epochs

        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)

        if verbose:
            logger.info("Getting latent sampler...")
        # Pass the already-loaded sampler to avoid a second load_everything call
        sampler_clone, k_minor, _ = get_latent_sampler(exp_name, n_minor, n_ood, sampler=sampler)

        if verbose:
            logger.info("Computing non-padded hiddens for latent task...")
        if kwargs.get("return_data", False):
            hiddens, demo_data = _compute_hiddens_at_real_tokens(
                config, model, sampler_clone, B,
                return_data=True, post_layernorm=_post_ln,
                extraction_point=_ep, verbose=verbose,
            )
            hiddens = hiddens.cpu()
            demo_data = demo_data.cpu()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return hiddens, k_minor, demo_data, sampler_clone
        else:
            hiddens = _compute_hiddens_at_real_tokens(
                config, model, sampler_clone, B,
                post_layernorm=_post_ln, extraction_point=_ep,
                verbose=verbose,
            )
            hiddens = hiddens.cpu()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elif task_name == "coin":
        # Load config + sampler without wasting GPU on the final-checkpoint model
        sampler, config = nu.load_config_and_sampler("coin", exp_name)
        if device_override is not None:
            config.device = device_override
        k_minor = min(n_minor, sampler.n_minor_tasks)
        if step is None:
            step = config.training.num_epochs

        model, _ = nu.load_checkpoint(config, step=step, exp_name=exp_name, return_actual_step=True)

        if verbose:
            logger.info("Getting coin sampler...")
        # Pass the already-loaded sampler to avoid a second load_everything call
        sampler_clone, k_minor = get_new_sampler(exp_name, n_minor, n_ood, sampler=sampler)

        if verbose:
            logger.info("Computing non-padded hiddens for coin task...")
        if kwargs.get("return_data", False):
            hiddens, demo_data = _compute_hiddens_at_real_tokens(
                config, model, sampler_clone, B,
                return_data=True, post_layernorm=_post_ln,
                extraction_point=_ep, verbose=verbose,
            )
            hiddens = hiddens.cpu()
            demo_data = demo_data.cpu()
        else:
            hiddens = _compute_hiddens_at_real_tokens(
                config, model, sampler_clone, B,
                post_layernorm=_post_ln, extraction_point=_ep,
                verbose=verbose,
            )
            hiddens = hiddens.cpu()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if kwargs.get("return_p", False):
            return hiddens, k_minor, torch.concat([sampler_clone.major_p.cpu(), sampler_clone.minor_p.cpu()])
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
            dyck_mask=mask, post_layernorm=_post_ln,
            extraction_point=_ep, verbose=verbose,
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

        eval_gen = _make_linear_eval_generator(device, train_task, step)
        eval_task_pool, k_minor = _create_eval_task_pool(
            train_task,
            K=n_ood,
            include_minor=True,
            
            device=device,
            n_minor=n_minor,
            generator=eval_gen,
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
                post_layernorm=_post_ln,
                extraction_point=_ep,
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
    total_steps: int = None,
    warmup_steps: int = None,
    lr: float = None,
    schedule: str = None,
    max_grad_norm: float = None,
    batch_size: int = None,
    noise_scale: float = None,
    p_minor: float = None,
    n_layer: int = None,
    n_points: int = None,
    min_lr: float = None,
    decay_power: float = None,
    batch_size_schedule: list = None,
    p_minor_schedule: list = None,
    final_layernorm: bool = None,
    quiet: bool = True,
    device: Optional[str] = None,
):
    import os
    _prev_wandb_silent = os.environ.get("WANDB_SILENT")
    if quiet:
        os.environ["WANDB_SILENT"] = "true"

    try:
        config = unified_get_config(task_name)
        if device is not None:
            config.device = device
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
        if total_steps is not None:
            if task_name == "linear":
                config.training.total_steps = total_steps
            else:
                config.training.num_epochs = total_steps
        if warmup_steps is not None:
            config.training.warmup_steps = warmup_steps
        if lr is not None:
            config.training.lr = lr
        if schedule is not None:
            config.training.schedule = schedule
        if max_grad_norm is not None:
            config.training.max_grad_norm = max_grad_norm
        if batch_size is not None:
            config.task.batch_size = batch_size
        if noise_scale is not None:
            config.task.noise_scale = noise_scale
        if p_minor is not None:
            config.task.p_minor = p_minor
        if n_layer is not None:
            config.model.n_layer = n_layer
        if n_points is not None:
            if task_name == "linear":
                config.task.n_points = n_points
                config.model.n_points = n_points
        if min_lr is not None:
            config.training.min_lr = min_lr
        if decay_power is not None:
            config.training.decay_power = decay_power
        if batch_size_schedule is not None:
            config.training.batch_size_schedule = list(batch_size_schedule)
        if p_minor_schedule is not None:
            config.training.p_minor_schedule = list(p_minor_schedule)
        if final_layernorm is not None:
            if task_name == "linear":
                config.model.final_layernorm = final_layernorm
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


def _unified_train_worker(args):
    """Picklable worker for ProcessPoolExecutor: run one unified_train on a given device."""
    task_name, k, device, kwargs = args
    verbose = kwargs.pop("_parallel_verbose", False)
    if verbose:
        import sys
        print(f"[unified_train_parallel] Training k={k} ({task_name}) on {device} ...", flush=True)
        sys.stdout.flush()
    return unified_train(task_name, k, device=device, **kwargs)


def unified_train_parallel(
    task_name: str,
    k_list: list,
    n_gpus: Optional[int] = None,
    verbose: bool = False,
    **kwargs,
) -> list:
    """Run multiple k experiments in parallel across GPUs.

    Each k is run in a separate process with config.device = f"cuda:{i % n_gpus}".
    With 2 GPUs and k_list=[0, 1, 2, 3], k=0 and k=2 use cuda:0, k=1 and k=3 use cuda:1.

    Parameters
    ----------
    task_name : str
        Same as unified_train (e.g. "linear", "coin", "latent", "dyck").
    k_list : list of int
        List of k values to train (e.g. [0, 1, 2, 3]).
    n_gpus : int, optional
        Number of GPUs to use. Default: min(2, torch.cuda.device_count()) or 1.
    verbose : bool, default False
        If True, log which k is being trained and when each run completes.
    **kwargs
        Passed through to unified_train for each run (e.g. total_steps, quiet, pad).

    Returns
    -------
    list
        Results from unified_train for each k, in the same order as k_list.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_workers = n_gpus if n_gpus is not None else min(2, n_available or 1)
    n_workers = min(n_workers, len(k_list))
    if n_workers <= 0:
        n_workers = 1
    use_gpu = n_available > 0

    # Drop device if caller passed it; we set it per worker
    kwargs = {k: v for k, v in kwargs.items() if k != "device"}
    # Pass verbose to worker only (worker pops it; unified_train does not accept it)
    worker_kwargs = {**kwargs, "_parallel_verbose": verbose}

    def _device(i: int) -> str:
        return f"cuda:{i % n_available}" if use_gpu else "cpu"

    args_list = [
        (task_name, k, _device(i), worker_kwargs)
        for i, k in enumerate(k_list)
    ]

    if n_workers == 1:
        # Avoid process spawn overhead when only one worker
        out = []
        for k in k_list:
            if verbose:
                print(f"[unified_train_parallel] Training k={k} ({task_name}) on {_device(0)} ...", flush=True)
            out.append(unified_train(task_name, k, device=_device(0), **kwargs))
        return out

    if verbose:
        print(
            f"[unified_train_parallel] Starting parallel training for {task_name} k_list={k_list} (n_workers={n_workers})",
            flush=True,
        )
        for i, (_, k, dev, _) in enumerate(args_list):
            print(f"[unified_train_parallel] Submitted k={k} on {dev}", flush=True)

    results = [None] * len(k_list)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        future_to_idx = {ex.submit(_unified_train_worker, args): i for i, args in enumerate(args_list)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            k = k_list[idx]
            try:
                results[idx] = future.result()
                if verbose:
                    print(f"[unified_train_parallel] Completed k={k}", flush=True)
            except Exception as e:
                logger.exception("unified_train_parallel failed for k=%s: %s", k, e)
                results[idx] = e  # store exception so order is preserved
    return results
