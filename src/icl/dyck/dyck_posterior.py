import torch


@torch.no_grad()
def dyck_task_posterior_over_time(
    task,                       # DyckPathTask instance
    samples: torch.Tensor,      # (B,L) or (E,B,L)
    masks: torch.Tensor,        # same shape as samples; 1 where Dyck token was planted
    *,
    return_log: bool = False,
    eps: float = 1e-30,
) -> torch.Tensor:
    """
    Compute filtering posterior P(Z=k | s_{0:t}) over Dyck-path identity Z for each t.

    Args:
        task: DyckPathTask.
        samples: Long tensor (B,L) or (E,B,L).
        masks:   Long/bool tensor same shape as samples.
        return_log: return log posterior if True.
        eps: numerical floor for probabilities (used when normalizing / degenerate rows).

    Returns:
        post: Tensor shape (B,L,T) or (E,B,L,T),
              where L = task.seq_len,
              and T = task.total_trans.
              post[..., t, k] = P(Z=k | s_{0:t}) (or log if return_log).
    """
    device = samples.device
    dtype = torch.float32

    if samples.dim() == 2:
        samples_ = samples.unsqueeze(0)  # (1,B,L_obs)
        masks_ = masks.unsqueeze(0)
        squeeze_E = True
    elif samples.dim() == 3:
        samples_ = samples
        masks_ = masks
        squeeze_E = False
    else:
        raise ValueError(f"samples must have shape (B,L) or (E,B,L), got {samples.shape}")

    if masks_.shape != samples_.shape:
        raise ValueError(f"masks must match samples shape; got masks={masks_.shape}, samples={samples_.shape}")

    x = samples_
    m = masks_

    E, B, L = x.shape
    T = task.total_trans
    assert T > 0, "No tasks available: total_trans == 0"

    # Build task pool of Dyck strings: (T, L_dyck)
    if task.n_minor_tasks > 0:
        dyck_all = torch.cat([task.major_task_pool, task.minor_task_pool], dim=0).to(device=device)
    else:
        dyck_all = task.major_task_pool.to(device=device)
    assert dyck_all.shape[0] == T

    # Prior over tasks (match generate() mixture)
    if task.n_minor_tasks == 0:
        prior = torch.full((T,), 1.0 / max(1, task.n_major_tasks), device=device, dtype=dtype)
    else:
        prior_major = (1.0 - float(task.p_minor)) / max(1, task.n_major_tasks)
        prior_minor = float(task.p_minor) / max(1, task.n_minor_tasks)
        prior = torch.cat([
            torch.full((task.n_major_tasks,), prior_major, device=device, dtype=dtype),
            torch.full((task.n_minor_tasks,), prior_minor, device=device, dtype=dtype),
        ], dim=0)
    prior = torch.clamp(prior, min=eps)
    log_prior = prior.log()  # (T,)

    # Output: (E,B,L,T)
    log_post = torch.empty((E, B, L, T), device=device, dtype=dtype)

    # Running unnormalized log-belief
    running = log_prior.view(1, 1, T).expand(E, B, T).clone()

    # planted index j = cumsum(m==1)-1, computed on the REAL timeline
    m_bool = m.to(torch.bool)
    planted_idx = m_bool.to(torch.long).cumsum(dim=-1) - 1  # (E,B,L)

    neg_inf = torch.tensor(-float("inf"), device=device, dtype=dtype)

    for t in range(L):
        is_planted = m_bool[:, :, t]  # (E,B)
        if is_planted.any():
            eb = torch.nonzero(is_planted, as_tuple=False)  # (N,2)
            e_idx = eb[:, 0]
            b_idx = eb[:, 1]

            j = planted_idx[e_idx, b_idx, t].long()   # (N,)
            obs = x[e_idx, b_idx, t].long()           # (N,)

            # expected tokens for each task: (N,T) where expected[n,k]=dyck_all[k, j[n]]
            expected = dyck_all.transpose(0, 1).index_select(0, j)  # (N,T)

            mismatch = expected != obs.unsqueeze(1)  # (N,T)
            running[e_idx, b_idx, :] = torch.where(mismatch, neg_inf, running[e_idx, b_idx, :])

        # Normalize to posterior at time t
        maxv = torch.max(running, dim=-1, keepdim=True).values  # (E,B,1)
        all_neg_inf = torch.isneginf(maxv)                      # (E,B,1)

        stabilized = running - maxv
        probs = torch.exp(stabilized)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)

        if all_neg_inf.any():
            probs = torch.where(all_neg_inf.expand_as(probs), torch.full_like(probs, 1.0 / T), probs)

        log_post[:, :, t, :] = torch.log(probs.clamp_min(eps))

        # carry forward log posterior
        running = log_post[:, :, t, :].clone()

    out = log_post if return_log else torch.exp(log_post)
    return out.squeeze(0) if squeeze_E else out
