"""Reusable analysis computations for the Real-LLM ICL notebook."""

from __future__ import annotations

import numpy as np
import torch

from icl.utils.separability import task_subspace_r2_at_position


def compute_r2_per_layer(
    hiddens_dict: dict,
    task_vecs: torch.Tensor,
    grand_mean: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Compute task-subspace R² at every layer for each group.

    Args:
        hiddens_dict: {name: tensor (L, N, D)} — hidden states per task/group.
        task_vecs:    tensor (L, K, D) — centred task vectors per layer.
        grand_mean:   tensor (L, D)   — grand mean per layer.

    Returns:
        {name: np.ndarray of shape (L,)} — per-layer R² for each group.
    """
    L = task_vecs.shape[0]
    result: dict[str, np.ndarray] = {}
    for name, hiddens in hiddens_dict.items():
        r2_l = []
        for l in range(L):
            tau_l = task_vecs[l]          # (K, D)
            mu_l  = grand_mean[l]         # (D,)
            h_l   = hiddens[l].unsqueeze(0)  # (1, N, D)
            res = task_subspace_r2_at_position(
                task_vecs=tau_l, hiddens=h_l,
                fit_token="none", grand_mean=mu_l, simplex=True,
            )
            r2_l.append(res.r2_task)
        result[name] = np.array(r2_l)
    return result


def compute_lambda_trajectories(
    task_names: list[str],
    ood_task_names: list[str],
    id_traj_hiddens: dict,
    ood_traj_hiddens: dict,
    task_vecs: torch.Tensor,
    grand_mean: torch.Tensor,
    layers: list[int],
    T_traj: int,
) -> tuple[dict, dict]:
    """Compute unconstrained affine coefficients λ_k along the prompt for each layer.

    For each layer ``l`` in ``layers`` and each position ``t`` in ``[0, T_traj)``,
    solves ``h_{l,t} - μ_l ≈ Σ_k λ_k τ_k(l)`` with the sum constraint but
    without non-negativity (affine, not simplex).

    Args:
        task_names:        ID task names (keys of id_traj_hiddens).
        ood_task_names:    OOD task names (keys of ood_traj_hiddens).
        id_traj_hiddens:   {name: tensor (L, T, N_eval, D)}.
        ood_traj_hiddens:  {name: tensor (L, T, N_eval, D)}.
        task_vecs:         tensor (L, K, D).
        grand_mean:        tensor (L, D).
        layers:            layer indices to compute.
        T_traj:            number of time steps (N_SHOTS + 1).

    Returns:
        beta_pos_id  : {layer_idx: {task_name: np.ndarray (T, N_eval, K)}}
        beta_pos_ood : {layer_idx: {ood_name:  np.ndarray (T, N_eval, K)}}
    """
    from icl.real_llm.hf_extractor import simplex_trajectory  # noqa: PLC0415

    beta_pos_id: dict  = {}
    beta_pos_ood: dict = {}

    for l_idx in layers:
        tv_l = task_vecs[l_idx].unsqueeze(0).expand(T_traj, -1, -1)   # (T, K, D)
        gm_l = grand_mean[l_idx].unsqueeze(0).expand(T_traj, -1)      # (T, D)

        beta_pos_id[l_idx] = {}
        for name in task_names:
            h = id_traj_hiddens[name][l_idx]                                   # (T, N_eval, D)
            lam = simplex_trajectory(tv_l, h, gm_l, constrain_to_simplex=False)
            beta_pos_id[l_idx][name] = lam.numpy()                             # (T, N_eval, K)

        beta_pos_ood[l_idx] = {}
        for name in ood_task_names:
            h = ood_traj_hiddens[name][l_idx]                                  # (T, N_eval, D)
            lam = simplex_trajectory(tv_l, h, gm_l, constrain_to_simplex=False)
            beta_pos_ood[l_idx][name] = lam.numpy()                            # (T, N_eval, K)

    return beta_pos_id, beta_pos_ood


def show_prediction_examples(
    task_name: str,
    prompts: list[str],
    answers: list[str],
    model,
    tokenizer,
    device: str,
    n: int = 5,
) -> None:
    """Print model predictions vs expected answers for debugging.

    Displays the prompt tail, expected answer, first-token IDs (with and without
    a leading space), the model's top-1 prediction, and the top-5 tokens.
    """
    print(f"\n{'=' * 60}")
    print(f"Task: {task_name}")
    print(f"{'=' * 60}")

    for idx in range(min(n, len(prompts))):
        prompt = prompts[idx]
        answer = answers[idx]

        enc = tokenizer(
            [prompt], return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        last_pos  = attention_mask.sum(dim=1) - 1
        logit_vec = logits[0, last_pos[0]]

        top5_ids  = logit_vec.topk(5).indices.tolist()
        top5_toks = [repr(tokenizer.decode([t])) for t in top5_ids]

        ans_ids_plain = tokenizer(answer,       add_special_tokens=False)["input_ids"]
        ans_ids_space = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
        first_plain   = ans_ids_plain[0] if ans_ids_plain else None
        first_space   = ans_ids_space[0] if ans_ids_space else None
        pred_id       = logit_vec.argmax().item()

        print(f"  prompt tail       : ...{prompt[-60:]!r}")
        print(f"  expected          : {answer!r}")
        print(f"  first tok (plain) : {first_plain} → {repr(tokenizer.decode([first_plain]))}")
        print(f"  first tok (space) : {first_space} → {repr(tokenizer.decode([first_space]))}")
        print(f"  model top-1       : {pred_id} → {repr(tokenizer.decode([pred_id]))}")
        print(f"  model top-5       : {top5_toks}")
        print()
