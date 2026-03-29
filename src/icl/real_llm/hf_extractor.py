"""
HuggingFace model wrapper for ICL hidden-state extraction.

Supports any causal LM with a transformer stack (Qwen2.5, LLaMA, Mistral, GPT-2, …).
Hidden states are extracted at the **last token** of each prompt (the query position)
using forward hooks.  Left-padding ensures the last token is always the real query
regardless of sequence length variation across a batch.

Key functions
-------------
load_model_and_tokenizer(model_name, device, dtype)
    Load a HuggingFace model and tokenizer.

get_module_getter(model)
    Return a  lambda l: <module>  callable that resolves layer l's nn.Module.

collect_hiddens(model, tokenizer, prompts, layers, batch_size, device)
    Batch forward passes; returns (L, N, D) hidden state tensor.

get_shot_positions(prompt, tokenizer, n_shots)
    Return 0-indexed token positions of the last token of each k-shot prefix.

collect_hiddens_trajectory(model, tokenizer, prompts, shot_positions_list, layers, ...)
    Extract hidden states at multiple positions per prompt; returns (L, T, N, D).

compute_task_vectors(hiddens_by_task, layers)
    Average support hiddens per task and centre → task vectors τ_k(l).

task_subspace_r2_per_layer(task_vecs_per_layer, hiddens_by_task)
    Compute AveragingR2Result for every layer.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
):
    """Load a HuggingFace causal LM and its tokenizer.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID, e.g. ``"Qwen/Qwen2.5-7B"`` or
        ``"meta-llama/Llama-3.2-3B"``.
    device : str
    dtype : torch.dtype
        Use ``torch.bfloat16`` on modern GPUs for memory efficiency.
    trust_remote_code : bool
        Required for Qwen models.

    Returns
    -------
    (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    # Left-padding: ensures position -1 is always the last real (query) token.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name}  (dtype={dtype}, device={device})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    print(
        f"Model loaded.  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B"
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Module resolution for different HF architectures
# ---------------------------------------------------------------------------

def get_module_getter(model: nn.Module) -> Callable[[int], nn.Module]:
    """Return a  lambda l: <decoder_layer_l>  for the given model.

    The returned module's output (after the full block: attn + MLP + residuals)
    is the residual-stream vector at that depth.

    Supports Qwen2.5, LLaMA 2/3, Mistral, Falcon, GPT-2, OPT, Bloom.
    Falls back to scanning for common attribute patterns.
    """
    # Try common attribute paths in order of likelihood
    candidates = [
        ("model", "layers"),         # Qwen2, LLaMA, Mistral, Falcon
        ("transformer", "h"),        # GPT-2, GPT-J, Falcon-alt
        ("model", "decoder", "layers"),  # OPT
        ("transformer", "blocks"),   # custom (this project's own models)
    ]
    for *attrs, leaf in candidates:
        obj = model
        try:
            for a in attrs:
                obj = getattr(obj, a)
            layers_obj = getattr(obj, leaf)
            # verify it is indexable
            _ = layers_obj[0]
            n_layers = len(layers_obj)
            print(
                f"Detected architecture: model.{'.'.join(attrs + [leaf])}  "
                f"({n_layers} layers)"
            )
            return lambda l, _lo=layers_obj: _lo[l]
        except (AttributeError, TypeError, KeyError, IndexError):
            continue

    raise RuntimeError(
        "Could not auto-detect transformer layer list.  "
        "Pass a custom module_getter to collect_hiddens()."
    )


def _get_n_layers(module_getter: Callable, max_search: int = 512) -> int:
    """Probe how many layers the module_getter supports."""
    for l in range(max_search):
        try:
            module_getter(l)
        except (IndexError, KeyError):
            return l
    return max_search


# ---------------------------------------------------------------------------
# Core hidden-state extraction via forward hooks
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_hiddens(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    layers: Sequence[int],
    module_getter: Optional[Callable[[int], nn.Module]] = None,
    batch_size: int = 8,
    device: str = "cuda",
    show_progress: bool = True,
) -> torch.Tensor:
    """Extract hidden states at the separator (":") token for every prompt.

    Following Hendel et al. (2023), extraction is at the **":" separator
    token** — the position that immediately follows the query input and
    precedes the (unobserved) answer.  Since our prompts end with
    ``"<query_x>: "`` (colon then space), this is at ``h[:, -2, :]``
    in the left-padded sequence.

    Parameters
    ----------
    model : nn.Module
        Causal LM in eval mode.
    tokenizer
        HuggingFace tokenizer with ``padding_side="left"``.
    prompts : list of str
        N ICL prompts, each ending with ``"<query_x>: "`` (no answer).
    layers : sequence of int
        Layer indices to extract from (0-indexed).
    module_getter : callable, optional
        ``lambda l: <nn.Module>``.  Auto-detected from model if None.
    batch_size : int
        Prompts processed per forward pass.
    device : str
    show_progress : bool

    Returns
    -------
    torch.Tensor  shape ``(L, N, D)``
        Float32, on CPU.  L = len(layers), N = len(prompts), D = hidden dim.
    """
    if module_getter is None:
        module_getter = get_module_getter(model)

    if layers is None:
        n_layers = _get_n_layers(module_getter)
        layers = list(range(n_layers))
        print(f"  Auto-detected {n_layers} layers")

    layers = list(layers)
    n_prompts = len(prompts)
    n_batches = math.ceil(n_prompts / batch_size)

    all_hiddens: List[torch.Tensor] = []  # each entry: (L, B, D)

    for bi in range(n_batches):
        if show_progress:
            print(f"  batch {bi + 1}/{n_batches}", end="\r")

        batch = prompts[bi * batch_size : (bi + 1) * batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Register hooks on each requested layer
        collected: Dict[int, torch.Tensor] = {}
        handles = []

        def _make_hook(l: int):
            def _hook(module, inp, output):
                # HF decoder layers return a tuple; first element is hidden states
                h = output[0] if isinstance(output, (tuple, list)) else output
                # h: (B, seq_len, D)
                # Extract at position -2: the ":" separator token (Hendel et al.
                # convention).  Our prompts end with "<query_x>: " so:
                #   position -1 = trailing space " "
                #   position -2 = colon ":"  ← task-relevant position
                collected[l] = h[:, -2, :].detach().float().cpu()
            return _hook

        for l in layers:
            handle = module_getter(l).register_forward_hook(_make_hook(l))
            handles.append(handle)

        try:
            model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for h in handles:
                h.remove()

        # Stack: (L, B, D)
        batch_tensor = torch.stack([collected[l] for l in layers], dim=0)
        all_hiddens.append(batch_tensor)

    if show_progress:
        print(f"  Done ({n_prompts} prompts, {len(layers)} layers)")

    return torch.cat(all_hiddens, dim=1)  # (L, N, D)


# ---------------------------------------------------------------------------
# Shot-position helpers for trajectory extraction
# ---------------------------------------------------------------------------

def get_shot_positions(
    prompt: str,
    tokenizer,
    n_shots: int,
    include_query: bool = True,
) -> List[int]:
    """Return 0-indexed token positions of the last token of each k-shot prefix.

    For a full N-shot ICL prompt whose lines are separated by ``"\\n"``, this
    function tokenizes the prefix after the first k examples (k = 1 … n_shots)
    and records the position of the last token in each prefix.  These positions
    mark *where in the full tokenized sequence* the model has just processed the
    k-th in-context example — the natural "time" axis for a trajectory plot.

    If ``include_query=True`` (the default), one extra position is appended:
    the last token of the *full* prompt (the query position).  This is the same
    position used by :func:`collect_hiddens` to estimate task vectors, so the
    final trajectory point will coincide with the task-vector stars in the plot.

    Parameters
    ----------
    prompt : str
        The full N-shot ICL prompt string (newline-separated examples, query
        at the end without an answer).
    tokenizer
        HuggingFace tokenizer.  Called with ``add_special_tokens=False`` so
        that the token count is relative to the prompt body (BOS offset is
        handled separately in ``collect_hiddens_trajectory``).
    n_shots : int
        Number of in-context examples in the prompt.
    include_query : bool
        If True (default), append the position of the query token (last token
        of the full prompt) so that T = n_shots + 1 and the trajectory endpoint
        aligns with the task vector stars.

    Returns
    -------
    list of int, length ``n_shots`` or ``n_shots + 1``
        ``positions[k]`` is the 0-indexed token index (in the *unpadded*,
        *no-BOS* tokenization) of the last token of the (k+1)-th example.
        ``positions[-1]`` is the query position when ``include_query=True``.
    """
    # Follow Hendel et al.: extract at the ":" separator token that immediately
    # follows the input x_k (and precedes the answer y_k).  At this position
    # the model has fully processed x_k and is about to generate y_k — the
    # most task-informative position in the sequence.
    #
    # Each line has the format  "x_k: y_k"  so we strip off ": y_k" to build
    # the prefix that ends exactly at ":".
    lines = prompt.split("\n")
    positions: List[int] = []
    prefix = ""
    for k in range(n_shots):
        # prefix = "x1:y1\n ... x_{k-1}:y_{k-1}\n x_k:"
        input_word = lines[k].split(": ")[0]   # everything before ": "
        prefix_to_colon = prefix + input_word + ":"
        n_toks = len(tokenizer(prefix_to_colon, add_special_tokens=False)["input_ids"])
        positions.append(n_toks - 1)   # 0-indexed position of ":" token
        # advance prefix past the full line (for the next iteration)
        prefix += lines[k] + "\n"
    if include_query:
        # Query position: the ":" after the query input word.
        # collect_hiddens extracts at h[:, -2, :] which is also the ":" token
        # (since prompts end with "<query_x>: " — colon then space).
        # Build prefix ending at ":" to get the same token index.
        prompt_to_colon = prompt.rstrip()   # strips trailing " ", ends at ":"
        n_full = len(tokenizer(prompt_to_colon, add_special_tokens=False)["input_ids"])
        positions.append(n_full - 1)   # position of ":" of query
    return positions


@torch.no_grad()
def collect_hiddens_trajectory(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    shot_positions_list: List[List[int]],
    layers: Sequence[int],
    module_getter: Optional[Callable[[int], nn.Module]] = None,
    batch_size: int = 4,
    device: str = "cuda",
    show_progress: bool = True,
) -> torch.Tensor:
    """Extract hidden states at multiple positions within each prompt.

    Unlike :func:`collect_hiddens` (which extracts only at the last / query
    token), this function extracts at the ``T`` shot-boundary positions
    returned by :func:`get_shot_positions` for each prompt.  The result is a
    4-D tensor that can be fed directly to
    ``project_with_r2_trajectories_group_colors_mpl`` with ``use_mean=False``.

    Parameters
    ----------
    model : nn.Module
    tokenizer
        Must have ``padding_side="left"`` (left-padding keeps position -1 at
        the query token, consistent with :func:`collect_hiddens`).
    prompts : list of str
        N ICL prompts.
    shot_positions_list : list of list of int
        One list of T positions per prompt (unpadded, no-BOS space).
        Produced by :func:`get_shot_positions`.
    layers : sequence of int
        Layer indices to extract (0-indexed).
    module_getter : callable, optional
    batch_size : int
        Smaller batches are safer when extracting many positions at once.
    device : str
    show_progress : bool

    Returns
    -------
    torch.Tensor  shape ``(L, T, N, D)``
        Float32, on CPU.
        L = number of layers, T = number of shot positions, N = number of
        prompts, D = hidden dimension.
    """
    if module_getter is None:
        module_getter = get_module_getter(model)

    if layers is None:
        n_layers = _get_n_layers(module_getter)
        layers = list(range(n_layers))
        print(f"  Auto-detected {n_layers} layers")

    layers = list(layers)
    n_prompts = len(prompts)
    T = len(shot_positions_list[0])   # number of shot positions (same for all)
    n_batches = math.ceil(n_prompts / batch_size)

    # Accumulate: list of (L, T, B, D) tensors, one per batch
    all_hiddens: List[torch.Tensor] = []

    for bi in range(n_batches):
        if show_progress:
            print(f"  batch {bi + 1}/{n_batches}", end="\r")

        batch_prompts = prompts[bi * batch_size : (bi + 1) * batch_size]
        batch_pos     = shot_positions_list[bi * batch_size : (bi + 1) * batch_size]
        B = len(batch_prompts)

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        input_ids      = enc["input_ids"].to(device)       # (B, S)
        attention_mask = enc["attention_mask"].to(device)  # (B, S)

        seq_len = input_ids.shape[1]

        # For each prompt b, convert unpadded shot positions to padded positions.
        # Left-padding: unpadded token k sits at padded index (S - n_full + k)
        # where n_full = number of real tokens in prompt b.
        n_full = attention_mask.sum(dim=1).tolist()   # (B,)  real token counts

        # padded_positions[b][t] = index in the padded sequence
        padded_positions = [
            [int(seq_len - n_full[b] + shot_positions_list[bi * batch_size + b][t])
             for t in range(T)]
            for b in range(B)
        ]

        # Clamp to valid range (safety against rounding in tokenization)
        padded_positions = [
            [max(0, min(p, seq_len - 1)) for p in plist]
            for plist in padded_positions
        ]

        # Hook: for each layer capture (B, T, D)
        collected: Dict[int, torch.Tensor] = {}
        handles = []

        def _make_hook(l: int, _pp=padded_positions, _B=B, _T=T):
            def _hook(module, inp, output):
                h = output[0] if isinstance(output, (tuple, list)) else output
                # h: (B, S, D) — extract at per-prompt positions
                extracted = torch.zeros(_B, _T, h.shape[-1],
                                        dtype=torch.float32, device="cpu")
                for b in range(_B):
                    for t in range(_T):
                        extracted[b, t] = h[b, _pp[b][t]].float().cpu()
                collected[l] = extracted   # (B, T, D)
            return _hook

        for l in layers:
            handle = module_getter(l).register_forward_hook(_make_hook(l))
            handles.append(handle)

        try:
            model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for h in handles:
                h.remove()

        # Stack over layers: (L, B, T, D) → transpose → (L, T, B, D)
        batch_tensor = torch.stack(
            [collected[l] for l in layers], dim=0
        )                                          # (L, B, T, D)
        batch_tensor = batch_tensor.permute(0, 2, 1, 3)  # (L, T, B, D)
        all_hiddens.append(batch_tensor)

    if show_progress:
        print(f"  Done ({n_prompts} prompts × {T} positions, {len(layers)} layers)")

    # Concatenate over the N dimension: (L, T, N, D)
    return torch.cat(all_hiddens, dim=2)


# ---------------------------------------------------------------------------
# Task vector estimation
# ---------------------------------------------------------------------------

def compute_task_vectors(
    support_hiddens: Dict[str, torch.Tensor],
    layers: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate task vectors by averaging support hidden states per task.

    Parameters
    ----------
    support_hiddens : dict  {task_name: (L, N_support, D)}
        Collected support hidden states for each ID task.
    layers : sequence of int
        Layer indices (used only for shape annotation).

    Returns
    -------
    task_vecs : torch.Tensor  shape ``(L, K, D)``
        Centred task vectors at each layer (sum to zero across K).
    grand_mean : torch.Tensor  shape ``(L, D)``
        Mean across all tasks and support samples at each layer.
    """
    task_names = list(support_hiddens.keys())
    K = len(task_names)

    # (L, K, D) — mean per (layer, task)
    task_means = torch.stack(
        [support_hiddens[name].mean(dim=1) for name in task_names],
        dim=1,
    )  # (L, K, D)

    grand_mean = task_means.mean(dim=1)          # (L, D)
    task_vecs = task_means - grand_mean.unsqueeze(1)  # (L, K, D) centred

    return task_vecs, grand_mean


# ---------------------------------------------------------------------------
# R² computation across layers
# ---------------------------------------------------------------------------

def task_subspace_r2_per_layer(
    task_vecs: torch.Tensor,           # (L, K, D)  centred task vectors
    eval_hiddens: Dict[str, torch.Tensor],  # {task_name: (L, N_eval, D)}
    grand_mean: torch.Tensor,          # (L, D)
    simplex: bool = True,
    eps: float = 1e-10,
) -> Dict[str, torch.Tensor]:
    """Compute task-subspace R² at every layer for each eval group.

    Uses the simplex-projection R² from ``separability.task_subspace_r2_at_position``.

    Parameters
    ----------
    task_vecs : (L, K, D)  centred ID task vectors
    eval_hiddens : {group_name: (L, N_eval, D)}
        May include both ID eval groups and OOD eval groups.
    grand_mean : (L, D)
        Grand mean from the support set (used to centre eval hiddens).
    simplex : bool
        If True, use simplex-constrained projection (β ≥ 0, Σβ = 1).
    eps : float

    Returns
    -------
    dict  {group_name: r2_tensor shape (L,)}
        Task-subspace R² at each layer for every group.
    """
    from icl.utils.separability import task_subspace_r2_at_position

    L, K, D = task_vecs.shape
    results: Dict[str, torch.Tensor] = {}

    for group_name, hiddens in eval_hiddens.items():
        # hiddens: (L, N_eval, D) — treat all N_eval samples as one group
        N_eval = hiddens.shape[1]
        r2_by_layer = []

        for l in range(L):
            tau_l = task_vecs[l]          # (K, D)
            mu_l  = grand_mean[l]         # (D,)

            # task_subspace_r2_at_position expects (n_tasks, B, D)
            # We pass all eval samples as one "task" with B=N_eval
            h_l = hiddens[l].unsqueeze(0)  # (1, N_eval, D)

            res = task_subspace_r2_at_position(
                task_vecs=tau_l,
                hiddens=h_l,
                fit_token="none",
                grand_mean=mu_l,
                simplex=simplex,
                eps=eps,
            )
            r2_by_layer.append(res.r2_task)

        results[group_name] = torch.tensor(r2_by_layer)

    return results


# ---------------------------------------------------------------------------
# Simplex coefficient trajectory
# ---------------------------------------------------------------------------

def simplex_trajectory(
    task_vecs: torch.Tensor,        # (L, K, D) centred
    eval_hiddens: torch.Tensor,     # (L, N, D)
    grand_mean: torch.Tensor,       # (L, D)
    constrain_to_simplex: bool = False,  # if True, also enforce β ≥ 0 (not recommended)
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute task coefficients λ(l) for each eval sample at each layer.

    Solves the affine OLS problem:
        min_λ  ‖h_centred − Σ λ_k V_k‖²   s.t.   Σ λ_k = 1

    Coefficients can be negative (indicating the sample is outside the task
    simplex, e.g. OOD samples), which is the geometrically correct result.

    Parameters
    ----------
    task_vecs : (L, K, D)  centred task vectors
    eval_hiddens : (L, N, D)  eval hidden states
    grand_mean : (L, D)  support grand mean
    constrain_to_simplex : bool
        If True, additionally project λ onto the probability simplex
        (β ≥ 0, Σβ = 1), clipping negative values.  Not recommended:
        this distorts OOD geometry by forcing them into the simplex.

    Returns
    -------
    lam : torch.Tensor  shape ``(L, N, K)``
        Affine OLS coefficients; each row sums to 1 (may be negative).
    """
    L, N, D = eval_hiddens.shape
    K = task_vecs.shape[1]
    lam_all = torch.zeros(L, N, K, dtype=torch.float32)

    for l in range(L):
        V  = task_vecs[l].float()          # (K, D)
        mu = grand_mean[l].float()         # (D,)
        h  = eval_hiddens[l].float() - mu  # (N, D) centred

        # Affine OLS: min ‖h − Vλ‖²  s.t. Σλ = 1
        # Parameterise λ = [γ, 1 − Σγ],  solve for γ via pinv of (V[:-1] − V[-1])
        anchor = V[-1]                           # (D,)
        diff   = V[:-1] - anchor                 # (K-1, D)
        pinv   = torch.linalg.pinv(diff)         # (D, K-1)
        gamma  = (h - anchor.unsqueeze(0)) @ pinv  # (N, K-1)
        lam_last = 1.0 - gamma.sum(dim=1, keepdim=True)
        lam_l = torch.cat([gamma, lam_last], dim=1)  # (N, K), sums to 1

        if constrain_to_simplex:
            from icl.utils.linear_algebra_utils import _project_onto_simplex_np
            lam_l = torch.from_numpy(_project_onto_simplex_np(lam_l.numpy()))

        lam_all[l] = lam_l

    return lam_all


# ---------------------------------------------------------------------------
# Trajectory tensor for traj_plot
# ---------------------------------------------------------------------------

def build_trajectory_tensor(
    task_vecs: torch.Tensor,         # (L, K, D) centred
    id_eval_hiddens: Dict[str, torch.Tensor],   # {task: (L, N, D)}
    ood_eval_hiddens: Dict[str, torch.Tensor],  # {ood_task: (L, N, D)}
    grand_mean: torch.Tensor,        # (L, D)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build tensors for ``project_with_r2_trajectories_group_colors_mpl``.

    Layout of the K axis:
      - first K_id entries  : one mean trajectory per ID task (blue shades)
      - next  K_ood entries : one mean trajectory per OOD task (orange shades)

    Parameters
    ----------
    task_vecs : (L, K_id, D)
    id_eval_hiddens : {task_name: (L, N_eval, D)} — K_id tasks
    ood_eval_hiddens : {ood_name: (L, N_ood, D)} — K_ood tasks
    grand_mean : (L, D)

    Returns
    -------
    traj : torch.Tensor  shape ``(K_id + K_ood, L, D)``
        Mean centred hidden states for each group at each layer.
    r2 : torch.Tensor  shape ``(K_id + K_ood, L)``
        Task-subspace R² for each group at each layer.
    final_task_vecs : torch.Tensor  shape ``(K_id, D)``
        ID task vectors at the last layer (used as the simplex reference frame).
    """
    from icl.utils.separability import task_subspace_r2_at_position

    L = task_vecs.shape[0]
    K_id = len(id_eval_hiddens)
    K_ood = len(ood_eval_hiddens)
    K_total = K_id + K_ood

    all_groups = list(id_eval_hiddens.values()) + list(ood_eval_hiddens.values())

    D = task_vecs.shape[2]
    traj = torch.zeros(K_total, L, D)
    r2   = torch.zeros(K_total, L)

    for k, hiddens in enumerate(all_groups):
        # Mean centred hidden state across eval batch at each layer
        mu = grand_mean.unsqueeze(1)           # (L, 1, D)
        traj[k] = (hiddens - mu).mean(dim=1)  # (L, D)

    # R² per group per layer
    for l in range(L):
        tau_l = task_vecs[l]         # (K_id, D)
        mu_l  = grand_mean[l]        # (D,)

        for k, hiddens in enumerate(all_groups):
            h_l = hiddens[l].unsqueeze(0)   # (1, N, D)
            res = task_subspace_r2_at_position(
                task_vecs=tau_l,
                hiddens=h_l,
                fit_token="none",
                grand_mean=mu_l,
                simplex=True,
            )
            r2[k, l] = res.r2_task

    final_task_vecs = task_vecs[-1]  # (K_id, D)  — last layer

    return traj, r2, final_task_vecs


# ---------------------------------------------------------------------------
# ICL performance evaluation (next-token accuracy & cross-entropy loss)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_icl_performance(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    answers: List[str],
    batch_size: int = 8,
    device: str = "cuda",
    show_progress: bool = True,
) -> dict:
    """Evaluate ICL next-token accuracy and cross-entropy loss.

    For each (prompt, answer) pair the model is given the prompt and we check
    whether the first generated token matches the first token of the
    ground-truth answer.  Cross-entropy loss is computed over the *first
    answer token* only (the most informative signal for ICL evaluation).

    Parameters
    ----------
    model : nn.Module
        Causal LM in eval mode.
    tokenizer
        HuggingFace tokenizer (``padding_side="left"``).
    prompts : list of str
        ICL prompts ending with ``"<query_x>: "`` (no answer appended).
    answers : list of str
        Ground-truth answer strings corresponding to each prompt.
    batch_size : int
    device : str
    show_progress : bool

    Returns
    -------
    dict with keys:
        ``"accuracy"``          – float, fraction of prompts where greedy
                                  prediction matches the first answer token.
        ``"mean_loss"``         – float, mean cross-entropy loss over all
                                  prompts (first answer token only).
        ``"per_sample_correct"``– list of bool, length N.
        ``"per_sample_loss"``   – list of float, length N.
    """
    import torch.nn.functional as F

    n = len(prompts)
    n_batches = math.ceil(n / batch_size)

    per_correct: List[bool] = []
    per_loss: List[float] = []

    for bi in range(n_batches):
        if show_progress:
            print(f"  eval batch {bi + 1}/{n_batches}", end="\r")

        batch_prompts = prompts[bi * batch_size : (bi + 1) * batch_size]
        batch_answers = answers[bi * batch_size : (bi + 1) * batch_size]
        B = len(batch_prompts)

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, S, V)

        # With left-padding the last real token is always at the absolute last
        # position (seq_len - 1) for every sequence in the batch.
        # Using attention_mask.sum()-1 would give real_len-1, which is wrong.
        last_pos = input_ids.shape[1] - 1  # scalar, same for all sequences

        for b in range(B):
            logit_vec = logits[b, last_pos]  # (V,)

            # Tokenize with a leading space to match the model's context:
            # prompts end with "<query_x>: " so the model predicts " answer"
            # (space fused into the subword), not "answer" in isolation.
            answer_tokens = tokenizer(
                " " + batch_answers[b], add_special_tokens=False
            )["input_ids"]
            if len(answer_tokens) == 0:
                per_correct.append(False)
                per_loss.append(float("inf"))
                continue

            target_id = answer_tokens[0]
            pred_id = logit_vec.argmax().item()

            per_correct.append(pred_id == target_id)

            loss = F.cross_entropy(
                logit_vec.unsqueeze(0).float(),
                torch.tensor([target_id], device=logit_vec.device),
            )
            per_loss.append(loss.item())

    if show_progress:
        print(f"  Done ({n} prompts evaluated)                ")

    accuracy = sum(per_correct) / n if n > 0 else 0.0
    mean_loss = sum(per_loss) / n if n > 0 else 0.0

    return {
        "accuracy": accuracy,
        "mean_loss": mean_loss,
        "per_sample_correct": per_correct,
        "per_sample_loss": per_loss,
    }
