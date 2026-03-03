import torch
from typing import Optional


@torch.no_grad()
def task_posterior_coins(
    sampler,                     # Coins instance
    samples: torch.Tensor,        # (N,L) or (E,N,L)
    probs_major: Optional[torch.Tensor] = None,  # optional override: (Kmaj, K)
    probs_minor: Optional[torch.Tensor] = None,  # optional override: (Kmin, K)
    *,
    include_minor: bool = True,
    return_log: bool = False,
    eps: float = 1e-30,
) -> torch.Tensor:
    """
    Compute posterior P(Z=k | x_{1:T}) for Coins (0th-order Markov / i.i.d. categorical).

    Prior matches sampler.generate():
      - If n_minor_tasks == 0: uniform over major tasks.
      - Else: P(minor)=p_minor, P(major)=1-p_minor, then uniform within pool.

    Args:
        sampler: Coins instance.
        samples: Long tensor (N,L) or (E,N,L).
        probs_major/probs_minor: optional probability pools to override sampler.major_p/minor_p.
        include_minor: whether to include minor pool in posterior (if available).
        return_log: if True, return log posterior.
        eps: numerical floor.

    Returns:
        posterior: (..., Ktot) where ... is (N,) or (E,N,),
                   Ktot = n_major_tasks + (n_minor_tasks if included and available).
    """
    device = samples.device
    dtype = torch.float32

    if samples.dim() == 2:
        samples_ = samples.unsqueeze(0)  # (1,N,L)
        squeeze0 = True
    elif samples.dim() == 3:
        samples_ = samples
        squeeze0 = False
    else:
        raise ValueError(f"samples must have shape (N,L) or (E,N,L), got {samples.shape}")

    E, N, _ = samples_.shape

    x = samples_
    E, N, T = x.shape

    K = int(sampler.num_states)

    maj = probs_major if probs_major is not None else sampler.major_p
    minp = probs_minor if probs_minor is not None else sampler.minor_p

    if maj is None or int(sampler.n_major_tasks) <= 0:
        raise ValueError("Posterior requires a finite major pool (n_major_tasks > 0).")

    maj = maj.to(device=device, dtype=dtype)  # (Kmaj, K)
    Kmaj = maj.shape[0]
    if maj.shape[1] != K:
        raise ValueError(f"major_p has wrong num_states: {maj.shape[1]} != {K}")

    use_minor_pool = (
        include_minor
        and int(sampler.n_minor_tasks) > 0
        and (minp is not None)
        and (minp.shape[0] == int(sampler.n_minor_tasks))
    )
    if use_minor_pool:
        minp = minp.to(device=device, dtype=dtype)  # (Kmin, K)
        Kmin = minp.shape[0]
        if minp.shape[1] != K:
            raise ValueError(f"minor_p has wrong num_states: {minp.shape[1]} != {K}")
        P = torch.cat([maj, minp], dim=0)  # (Ktot, K)
    else:
        Kmin = 0
        P = maj  # (Kmaj, K)

    Ktot = P.shape[0]

    if Kmin == 0:
        prior = torch.full((Ktot,), 1.0 / max(1, Kmaj), device=device, dtype=dtype)
    else:
        p0 = float(sampler.p_minor)
        prior_major = (1.0 - p0) / max(1, Kmaj)
        prior_minor = p0 / max(1, Kmin)
        prior = torch.cat([
            torch.full((Kmaj,), prior_major, device=device, dtype=dtype),
            torch.full((Kmin,), prior_minor, device=device, dtype=dtype),
        ], dim=0)

    prior = torch.clamp(prior, min=eps)
    log_prior = prior.log()  # (Ktot,)

    counts = torch.zeros((E, N, K), device=device, dtype=dtype)
    counts.scatter_add_(
        dim=-1,
        index=x.long(),
        src=torch.ones((E, N, T), device=device, dtype=dtype),
    )  # (E,N,K)

    logP = torch.log(torch.clamp(P, min=eps))         # (Ktot, K)
    loglik = torch.einsum("enk,tk->ent", counts, logP)  # (E,N,Ktot)

    unnorm = loglik + log_prior.view(1, 1, Ktot)      # (E,N,Ktot)
    log_post = unnorm - torch.logsumexp(unnorm, dim=-1, keepdim=True)

    out = log_post if return_log else torch.exp(log_post)
    return out.squeeze(0) if squeeze0 else out
