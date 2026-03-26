"""Shared low-level helpers for Coin analysis."""

import gc
import torch
import numpy as np
from typing import Sequence, Tuple, Optional
from torch import nn

import icl.utils.notebook_utils as nu
from icl.latent_markov.legacy.coin_latent_task_vecs import extract_hidden_multi_coin_latent
from icl.utils.linear_algebra_utils import stable_rank
from icl.coin.coin_ood_analysis import get_new_sampler
from icl.utils.logger import setup_logger

logger = setup_logger(__name__)


def compute_hiddens_multi_coin(
    config,
    model: nn.Module,
    sampler,
    layers: Sequence[int] = None,
    batch_size: int = 64,
    positions_of_interest: Sequence[int] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute hidden representations for Coin/Latent tasks on non-padded sequences.

    Non-padded counterpart of ``compute_hiddens_multi_coin_latent``.
    The sampler is forced to ``pad = False`` so that ``generate()`` returns
    sequences of shape ``(B, seq_len)`` containing only real tokens.
    Hiddens are extracted directly at the requested position indices.

    Parameters
    ----------
    config : ConfigDict
        Configuration object.
    model : nn.Module
        Model with ``model.layers[l].attn_block``.
    sampler : Coins or LatentMarkov
        Sampler with ``.generate()`` and ``.seq_len``.
    layers : sequence of int, optional
        Layers to extract from.  ``None`` → all layers.
    batch_size : int
        Batch size for sampling.
    positions_of_interest : sequence of int, optional
        Position indices ``[0, seq_len-2]`` to analyse.
        ``None`` → all positions ``[0, 1, ..., seq_len-2]``.

    Returns
    -------
    all_hiddens : torch.Tensor
        Shape ``(L, n_tasks, n_positions, batch_size, n_embd)``.
    position_info : dict
        ``{'positions': [...], 'seq_len': int}``.
    """
    device = config.device
    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    seq_len = sampler.seq_len
    n_embd = config.model.emb_dim

    if layers is None:
        layers = list(range(len(model.layers)))
    layers = list(layers)
    L = len(layers)

    # Determine positions of interest
    if positions_of_interest is None:
        positions_of_interest = list(range(seq_len - 1))
    else:
        positions_of_interest = list(positions_of_interest)
        if not all(0 <= p < seq_len - 1 for p in positions_of_interest):
            raise ValueError(f"All positions must be in [0, {seq_len - 2}]")

    n_positions = len(positions_of_interest)

    # Positions ARE the sequence positions (no 2*p+1 mapping)
    task_pos = torch.tensor(positions_of_interest, device=device, dtype=torch.long)

    output_shape = (L, n_tasks, n_positions, batch_size, n_embd)
    all_hiddens = torch.empty(output_shape, dtype=torch.float32, device=device)

    for task_idx in range(n_tasks):
        demo_data, _ = sampler.generate(
            mode="testing", task=task_idx, num_samples=batch_size
        )
        demo_data = demo_data.to(device)

        # (L, B, P, D)
        chunk_hiddens = extract_hidden_multi_coin_latent(
            model=model,
            batch_data=demo_data,
            layers=layers,
            task_pos=task_pos,
        )

        # (L, B, P, D) -> (L, P, B, D)
        chunk_hiddens = chunk_hiddens.permute(0, 2, 1, 3)
        all_hiddens[:, task_idx] = chunk_hiddens

    position_info = {
        'positions': positions_of_interest,
        'seq_len': seq_len,
    }

    return all_hiddens.detach().cpu(), position_info


@torch.no_grad()
def compute_hiddens_token_conditioned_coin(
    config,
    model: nn.Module,
    sampler,
    layers: Sequence[int] = None,
    batch_size: int = 64,
    positions_of_interest: Sequence[int] = None,
    max_unique_tokens: int = None,
    task_batch_size: int = 8,
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
) -> Tuple[torch.Tensor, dict]:
    """
    Token-conditioned hidden extraction on non-padded sequences.

    Non-padded counterpart of ``compute_hiddens_token_conditioned_coin_latent``.

    For each position of interest this function:

    1. Samples batches for every task to collect the token values at that
       position.
    2. Finds the set of unique token values.
    3. For each unique token value, creates modified batches where all
       sequences share the same token at that position.
    4. Extracts hidden representations **at that same position** (the real
       token itself — there is no following pad token).

    Position mapping (compared with the padded version):

    - Padded:  fix at ``2*pos_idx``, extract at ``2*pos_idx + 1``
    - Non-padded: fix at ``pos_idx``, extract at ``pos_idx``

    Parameters
    ----------
    config : ConfigDict
    model : nn.Module
        Model with ``model.layers[l].attn_block``.
    sampler : Coins or LatentMarkov
    layers : sequence of int, optional
        ``None`` → all layers.
    batch_size : int
    positions_of_interest : sequence of int, optional
        Position indices ``[0, seq_len-2]``.  ``None`` → all.
    max_unique_tokens : int, optional
        Cap on unique tokens per position.
    task_batch_size : int
        Number of tasks to batch into a single forward pass.
        Higher = faster (better GPU utilisation) but uses more VRAM.

    Returns
    -------
    all_hiddens : torch.Tensor
        ``(L, n_positions, max_unique_tokens, n_tasks, batch_size, n_embd)``
    token_info : dict
        ``{'positions', 'unique_tokens', 'token_type', 'n_unique_tokens'}``
    """
    device = config.device
    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    seq_len = sampler.seq_len
    n_embd = config.model.emb_dim

    if layers is None:
        layers = list(range(len(model.layers)))
    layers = list(layers)
    L = len(layers)

    if positions_of_interest is None:
        positions_of_interest = list(range(seq_len - 1))
    else:
        positions_of_interest = list(positions_of_interest)
        if not all(0 <= p < seq_len - 1 for p in positions_of_interest):
            raise ValueError(f"All positions must be in [0, {seq_len - 2}]")

    n_positions = len(positions_of_interest)

    # ------------------------------------------------------------------
    # Step 1: collect tokens at each position across all tasks (CPU only)
    # ------------------------------------------------------------------
    all_tokens_by_position = {p: [] for p in positions_of_interest}

    for task_idx in range(n_tasks):
        demo_data, _ = sampler.generate(
            mode="testing", task=task_idx, num_samples=batch_size
        )
        for pos_idx in positions_of_interest:
            all_tokens_by_position[pos_idx].append(demo_data[:, pos_idx])

    # ------------------------------------------------------------------
    # Step 2: find unique tokens per position
    # ------------------------------------------------------------------
    unique_tokens_by_position = {}
    for pos_idx in positions_of_interest:
        all_tokens = torch.cat(all_tokens_by_position[pos_idx], dim=0)
        unique_tokens = torch.unique(all_tokens, sorted=True)

        if max_unique_tokens is not None and len(unique_tokens) > max_unique_tokens:
            indices = torch.randperm(len(unique_tokens))[:max_unique_tokens]
            unique_tokens = unique_tokens[indices]

        unique_tokens_by_position[pos_idx] = unique_tokens

    del all_tokens_by_position

    max_unique_tokens_actual = max(len(ut) for ut in unique_tokens_by_position.values())

    # ------------------------------------------------------------------
    # Pre-allocate output on CPU to avoid GPU OOM
    # ------------------------------------------------------------------
    output_shape = (L, n_positions, max_unique_tokens_actual, n_tasks, batch_size, n_embd)
    all_hiddens = torch.zeros(output_shape, dtype=torch.float32, device="cpu")

    # ------------------------------------------------------------------
    # Step 3-4: fix token, batch tasks, extract hidden → CPU
    # ------------------------------------------------------------------
    for pos_idx_idx, pos_idx in enumerate(positions_of_interest):
        unique_tokens = unique_tokens_by_position[pos_idx]

        if pos_idx >= seq_len:
            continue

        extract_pos_tensor = torch.tensor(
            [pos_idx], device=device, dtype=torch.long
        )

        for token_idx, fixed_token_value in enumerate(unique_tokens):
            for t0 in range(0, n_tasks, task_batch_size):
                t1 = min(t0 + task_batch_size, n_tasks)

                batch_parts = []
                for task_idx in range(t0, t1):
                    demo_data, _ = sampler.generate(
                        mode="testing", task=task_idx, num_samples=batch_size
                    )
                    demo_data[:, pos_idx] = fixed_token_value
                    batch_parts.append(demo_data)

                big_batch = torch.cat(batch_parts, dim=0).to(device)
                del batch_parts

                chunk_hiddens = extract_hidden_multi_coin_latent(
                    model=model,
                    batch_data=big_batch,
                    layers=layers,
                    task_pos=extract_pos_tensor,
                    post_layernorm=post_layernorm,
                    extraction_point=extraction_point,
                )  # (L, (t1-t0)*batch_size, 1, n_embd)

                h_cpu = chunk_hiddens[:, :, 0, :].cpu()  # (L, (t1-t0)*B, D)
                del big_batch, chunk_hiddens

                n_in_batch = t1 - t0
                h_cpu = h_cpu.view(L, n_in_batch, batch_size, n_embd)
                all_hiddens[:, pos_idx_idx, token_idx, t0:t1] = h_cpu

                del h_cpu

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    token_info = {
        'positions': positions_of_interest,
        'unique_tokens': {
            pos: tokens.cpu().numpy().tolist()
            for pos, tokens in unique_tokens_by_position.items()
        },
        'token_type': 'data',
        'n_unique_tokens': {
            pos: len(tokens)
            for pos, tokens in unique_tokens_by_position.items()
        },
    }

    return all_hiddens.detach(), token_info


def _stable_rank_from_gram(gram: torch.Tensor) -> float:
    """Compute stable rank from a pre-accumulated Gram matrix ``A^T A``.

    stable_rank = trace(A^T A) / λ_max(A^T A) = ||A||_F^2 / σ_max^2
    """
    tr = torch.trace(gram).item()
    if tr == 0.0:
        return 0.0
    # Largest eigenvalue of the PSD Gram matrix = σ_max^2
    eigvals = torch.linalg.eigvalsh(gram)  # ascending order
    lam_max = eigvals[-1].item()
    if lam_max <= 0.0:
        return 0.0
    return tr / lam_max


def compute_stable_rank_at_real_positions(
    exp_name: str,
    task_name: Optional[str] = None,
    B: int = 64,
    step: Optional[int] = None,
    n_minor: Optional[int] = None,
    n_ood: int = 0,
    verbose: bool = False,
    task_batch_size: int = 16,
    position_stride: int = 3,
) -> dict:
    """
    Compute stable rank of hidden representations at real-token positions.

    Non-padded counterpart of ``compute_stable_rank_at_padded_positions``.

    **Memory-efficient, batched streaming implementation** — for coin and
    latent tasks, multiple tasks are batched into a single forward pass for
    speed, and Gram matrices ``A^T A`` of shape ``(D, D)`` are accumulated
    per (layer, position) without ever materialising the full hiddens tensor.

    Parameters
    ----------
    exp_name : str
    task_name : str, optional
        ``"coin"``, ``"latent"``, ``"linear"``, ``"dyck"``.
    B : int
        Batch size per task.
    step : int, optional
    n_minor : int, optional
        ``None`` → all.  ``-1`` → none.
    n_ood : int
    verbose : bool
    task_batch_size : int, default 16
        Number of tasks to batch into a single GPU forward pass.
        Higher = faster (better GPU utilisation) but uses more VRAM.
    position_stride : int, default 3
        Evaluate every ``position_stride``-th position to reduce
        computation.  Use 1 for all positions.

    Returns
    -------
    dict
        ``{'stable_ranks': (n_layers, n_eval_positions),
        'eval_positions': list[int], 'task_name',
        'n_layers', 'n_positions', 'k_minor'}``
    """
    from icl.latent_markov.analysis.ood import get_latent_sampler

    # ---- infer task_name ----
    if task_name is None:
        from icl.linear.linear_path_utils import load_model_task_config
        for tn in ["linear", "coin", "latent", "dyck"]:
            try:
                if tn == "linear":
                    load_model_task_config(exp_name)
                else:
                    nu.load_everything(tn, exp_name)
                task_name = tn
                if verbose:
                    logger.info(f"Inferred task_name: {task_name}")
                break
            except Exception:
                continue
        if task_name is None:
            raise ValueError(
                f"Could not infer task_name from exp_name: {exp_name}. "
                "Please specify task_name."
            )

    if n_minor is None:
        n_minor_effective = 1_000_000
    elif n_minor == -1:
        n_minor_effective = 0
    else:
        n_minor_effective = n_minor

    k_minor = 0

    if verbose:
        logger.info(
            f"Computing non-padded stable ranks (batched streaming): "
            f"exp={exp_name}, task={task_name}, "
            f"n_minor={n_minor_effective}, n_ood={n_ood}, "
            f"task_batch_size={task_batch_size}, position_stride={position_stride}"
        )

    # ==================================================================
    # coin / latent — batched streaming Gram accumulation
    # ==================================================================
    if task_name in ("coin", "latent"):
        if task_name == "latent":
            _, _, config = nu.load_everything("latent", exp_name)
            if step is None:
                step = config.training.num_epochs
            model, _ = nu.load_checkpoint(
                config, step=step, exp_name=exp_name, return_actual_step=True
            )
            sampler, k_minor, _ = get_latent_sampler(
                exp_name, n_minor=n_minor_effective, n_ood=n_ood
            )
        else:  # coin
            _, _, config = nu.load_everything("coin", exp_name)
            if step is None:
                step = config.training.num_epochs
            model, _ = nu.load_checkpoint(
                config, step=step, exp_name=exp_name, return_actual_step=True
            )
            sampler, k_minor = get_new_sampler(
                exp_name, n_minor_effective, n_ood
            )

        model.eval().to(config.device)

        device = config.device
        seq_len = sampler.seq_len
        D = int(config.model.emb_dim)
        layers = list(range(len(model.layers)))
        L = len(layers)

        # Subsampled positions
        all_positions = list(range(seq_len - 1))
        eval_positions = all_positions[::position_stride]
        if eval_positions[-1] != all_positions[-1]:
            eval_positions.append(all_positions[-1])
        n_eval = len(eval_positions)
        task_pos = torch.tensor(eval_positions, device=device, dtype=torch.long)

        n_tasks_total = sampler.n_major_tasks + sampler.n_minor_tasks

        gram_mem_mb = L * n_eval * D * D * 8 / 1e6
        if verbose:
            logger.info(
                f"{task_name}: {n_tasks_total} tasks, D={D}, "
                f"eval_positions={n_eval}/{seq_len - 1}, "
                f"task_batch_size={task_batch_size}, "
                f"Gram: ({L}, {n_eval}, {D}, {D}) = {gram_mem_mb:.1f} MB"
            )

        # Gram accumulators on CPU in float64
        grams = torch.zeros(
            (L, n_eval, D, D), dtype=torch.float64, device="cpu"
        )

        # Batched: generate for multiple tasks, concatenate, one forward pass
        for t0 in range(0, n_tasks_total, task_batch_size):
            t1 = min(t0 + task_batch_size, n_tasks_total)
            n_in_batch = t1 - t0

            # Generate sequences for each task in this batch
            batch_parts = []
            for task_idx in range(t0, t1):
                demo_data, _ = sampler.generate(
                    mode="testing", task=task_idx, num_samples=B
                )
                batch_parts.append(demo_data)

            # Concatenate: (n_in_batch * B, seq_len)
            big_batch = torch.cat(batch_parts, dim=0).to(device)
            del batch_parts

            # ONE forward pass for all tasks in this batch
            ch = extract_hidden_multi_coin_latent(
                model=model, batch_data=big_batch,
                layers=layers, task_pos=task_pos,
            )  # (L, n_in_batch*B, n_eval, D)

            ch_cpu = ch.cpu().to(torch.float64)
            del big_batch, ch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Accumulate Gram: h is (n_in_batch*B, D) per (layer, position)
            for li in range(L):
                for pi in range(n_eval):
                    h = ch_cpu[li, :, pi, :]  # (n_in_batch*B, D)
                    grams[li, pi].addmm_(h.t(), h)

            del ch_cpu

            if verbose:
                logger.info(
                    f"  tasks {t0}..{t1 - 1}  "
                    f"({t1}/{n_tasks_total}, batch={n_in_batch * B})"
                )

        # Free model
        model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Compute stable rank from Gram matrices
        n_layers = L
        stable_ranks = torch.zeros(n_layers, n_eval)
        for li in range(n_layers):
            for pi in range(n_eval):
                stable_ranks[li, pi] = _stable_rank_from_gram(grams[li, pi])

        del grams
        gc.collect()
        n_positions = n_eval

    # ==================================================================
    # dyck / linear — fall back to full hiddens (moderate task counts)
    # ==================================================================
    elif task_name == "dyck":
        from icl.utils.unified_interface import _get_hiddens_at_real_positions
        hiddens, k_minor, _mask = _get_hiddens_at_real_positions(
            task_name=task_name, exp_name=exp_name,
            n_minor=n_minor_effective, n_ood=n_ood, B=B,
            step=step, verbose=verbose,
        )
        L_h, K, T, B_actual, D = hiddens.shape
        n_layers = L_h
        eval_positions = list(range(0, T, position_stride))
        if eval_positions[-1] != T - 1:
            eval_positions.append(T - 1)
        n_positions = len(eval_positions)
        hiddens = hiddens.cpu()
        stable_ranks = torch.zeros(n_layers, n_positions)
        for l in range(n_layers):
            for pi, t in enumerate(eval_positions):
                h = hiddens[l, :, t, :, :].reshape(-1, D).float().numpy()
                stable_ranks[l, pi] = stable_rank(h)
        del hiddens

    else:
        # linear
        from icl.utils.unified_interface import _get_hiddens_at_real_positions
        hiddens, k_minor = _get_hiddens_at_real_positions(
            task_name=task_name, exp_name=exp_name,
            n_minor=n_minor_effective, n_ood=n_ood, B=B,
            step=step, verbose=verbose,
        )
        L_h, K, T, B_actual, D = hiddens.shape
        n_layers = L_h
        eval_positions = list(range(0, T, position_stride))
        if eval_positions[-1] != T - 1:
            eval_positions.append(T - 1)
        n_positions = len(eval_positions)
        hiddens = hiddens.cpu()
        stable_ranks = torch.zeros(n_layers, n_positions)
        for l in range(n_layers):
            for pi, t in enumerate(eval_positions):
                h = hiddens[l, :, t, :, :].reshape(-1, D).float().numpy()
                stable_ranks[l, pi] = stable_rank(h)
        del hiddens

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if verbose:
        logger.info(f"Stable ranks shape: {stable_ranks.shape}")

    return {
        'stable_ranks': stable_ranks,
        'eval_positions': eval_positions,
        'task_name': task_name,
        'n_layers': n_layers,
        'n_positions': n_positions,
        'k_minor': k_minor,
    }


def get_token_conditioned_hiddens_coin(
    exp_name: str,
    layers: Optional[list] = None,
    batch_size: int = 64,
    positions_of_interest: Optional[list] = None,
    max_unique_tokens: Optional[int] = None,
    n_minor: int = 0,
    step: Optional[int] = None,
    verbose: bool = False,
    task_batch_size: int = 8,
    post_layernorm: bool = False,
    extraction_point: str = "post_attn",
) -> tuple:
    """
    Get token-conditioned hidden representations for Coin task (non-padded).

    Non-padded counterpart of ``get_token_conditioned_hiddens_coin``.
    Fixes tokens at specific positions and extracts hiddens **at the same
    position** (the real token itself — no following PAD token).
    Always uses ``n_ood=0``.  ``n_minor`` is capped at
    ``sampler.n_minor_tasks``.

    Parameters
    ----------
    exp_name : str
    layers : list, optional
        ``None`` → all layers.
    batch_size : int
    positions_of_interest : list, optional
        Position indices ``[0, seq_len-2]``.  ``None`` → all.
    max_unique_tokens : int, optional
    n_minor : int
        Capped at ``sampler.n_minor_tasks``.
    step : int, optional
    verbose : bool
    task_batch_size : int
        Number of tasks to batch into a single forward pass.

    Returns
    -------
    all_hiddens : torch.Tensor
        ``(L, n_positions, n_unique_tokens, n_tasks, batch_size, n_embd)``
    token_info : dict
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
        logger.info(
            f"Computing token-conditioned hiddens for coin exp: {exp_name}"
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
        task_batch_size=task_batch_size,
        post_layernorm=post_layernorm,
        extraction_point=extraction_point,
    )

    if verbose:
        logger.info(f"Computed hiddens shape: {all_hiddens.shape}")
        logger.info(f"Token info - positions: {token_info['positions']}")
        logger.info(f"Token info - n_unique_tokens: {token_info['n_unique_tokens']}")

    # Cleanup
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return all_hiddens, token_info
