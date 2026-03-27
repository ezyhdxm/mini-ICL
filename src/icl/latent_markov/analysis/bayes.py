import torch
import torch.nn as nn
from typing import Optional


class GroupUniformKnownBayes(nn.Module):
    """
    Bayesian predictor when the true transition matrix is one of M known candidates.

    Prior:
        total mass p_common on first 3 chains (indices 0,1,2), uniform within them;
        total mass 1-p_common on remaining K chains (indices 3..M-1), uniform within them.
    """
    def __init__(self, trans_mat: torch.Tensor, p_common: float = 0.9, device: str = "cpu"):
        """
        Args:
            trans_mat: (M, N, N) row-stochastic transition matrices
            p_common: prior mass on first 3 chains
        """
        super().__init__()
        assert trans_mat.dim() == 3
        M, N, N2 = trans_mat.shape
        assert N == N2
        assert M >= 4, "Need M=3+K with K>=1 so M>=4."

        self.M = M
        self.N = N
        self.device = device

        self.register_buffer("log_trans_mat", torch.log(trans_mat.to(device).clamp_min(1e-30)))

        K = M - 3
        pi = torch.empty(M, device=device, dtype=torch.float)
        pi[:3] = p_common / 3.0
        pi[3:] = (1.0 - p_common) / float(K)
        self.register_buffer("log_pi", torch.log(pi.clamp_min(1e-30)))

    @torch.no_grad()
    def predict(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Args:
            samples: (B, T) integer states in {0,...,N-1}

        Returns:
            preds: (B, T, N) posterior predictive distribution for each position.
        """
        samples = samples.to(self.device)
        B, T = samples.shape
        M, N, _ = self.log_trans_mat.shape

        preds = torch.zeros((B, T, N), device=self.device, dtype=torch.float)

        m_idx = torch.arange(M, device=self.device).view(M, 1)

        cum_logw = self.log_pi.view(M, 1).expand(M, B).clone()

        s0 = samples[:, 0]
        row0 = self.log_trans_mat[:, s0, :]
        log_num = torch.logsumexp(row0 + cum_logw.unsqueeze(-1), dim=0)
        log_den = torch.logsumexp(cum_logw, dim=0).unsqueeze(-1)
        preds[:, 0] = (log_num - log_den).exp()

        log_trans_flat = self.log_trans_mat.view(M, N * N)

        for t in range(T - 1):
            s_t = samples[:, t]
            s_tp1 = samples[:, t + 1]
            flat_idx = s_t * N + s_tp1

            obs_logp = log_trans_flat.gather(dim=1, index=flat_idx.view(1, B).expand(M, B))
            cum_logw = cum_logw + obs_logp

            row = self.log_trans_mat[:, s_tp1, :]
            log_num = torch.logsumexp(row + cum_logw.unsqueeze(-1), dim=0)
            log_den = torch.logsumexp(cum_logw, dim=0).unsqueeze(-1)
            preds[:, t + 1] = (log_num - log_den).exp()

        return preds


class ThreeKnownPlusNewDirichletBayes(nn.Module):
    """
    Bayesian predictor over z in {0,1,2,new}.
      - z=0,1,2: known transition matrices P3[z]
      - z=3: a new (unknown) transition matrix; each row ~ Dirichlet(alpha=1,...,1)
    Prior:
      P(z in {0,1,2}) = p_common_total, uniform over the 3
      P(z=new) = 1 - p_common_total
    """
    def __init__(self, trans_mat_3: torch.Tensor, p_common_total: float = 0.9, alpha: float = 1.0, device: str = "cpu"):
        super().__init__()
        assert trans_mat_3.dim() == 3 and trans_mat_3.size(0) == 3
        _, N, N2 = trans_mat_3.shape
        assert N == N2

        self.N = N
        self.device = device
        self.alpha = float(alpha)

        self.register_buffer("log_trans_3", torch.log(trans_mat_3.to(device).clamp_min(1e-30)))

        pi = torch.tensor([p_common_total/3.0, p_common_total/3.0, p_common_total/3.0, 1.0 - p_common_total],
                          device=device, dtype=torch.float)
        self.register_buffer("log_pi", torch.log(pi.clamp_min(1e-30)))

    @torch.no_grad()
    def predict(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Args:
            samples: (B, T) states in {0,...,N-1}

        Returns:
            preds: (B, T, N) posterior predictive distribution at each position.
        """
        samples = samples.to(self.device)
        B, T = samples.shape
        N = self.N

        preds = torch.zeros((B, T, N), device=self.device, dtype=torch.float)

        cum_logw = self.log_pi.view(4, 1).expand(4, B).clone()

        counts = torch.zeros((B, N, N), device=self.device, dtype=torch.float)
        row_sums = torch.zeros((B, N), device=self.device, dtype=torch.float)
        b_idx = torch.arange(B, device=self.device)

        log_trans_3_flat = self.log_trans_3.view(3, N * N)

        s0 = samples[:, 0]
        known_row0 = self.log_trans_3[:, s0, :]
        new_row0 = torch.full((B, N), fill_value=-torch.log(torch.tensor(float(N), device=self.device)), device=self.device)

        row0 = torch.cat([known_row0, new_row0.unsqueeze(0)], dim=0)
        log_num = torch.logsumexp(row0 + cum_logw.unsqueeze(-1), dim=0)
        log_den = torch.logsumexp(cum_logw, dim=0).unsqueeze(-1)
        preds[:, 0] = (log_num - log_den).exp()

        for t in range(T - 1):
            s_t = samples[:, t]
            s_tp1 = samples[:, t + 1]
            flat_idx = s_t * N + s_tp1

            obs_logp_known = log_trans_3_flat.gather(
                dim=1, index=flat_idx.view(1, B).expand(3, B)
            )
            cum_logw[0:3] = cum_logw[0:3] + obs_logp_known

            num = self.alpha + counts[b_idx, s_t, s_tp1]
            den = N * self.alpha + row_sums[b_idx, s_t]
            pred_new = (num / den).clamp_min(1e-30)
            cum_logw[3] = cum_logw[3] + torch.log(pred_new)

            counts[b_idx, s_t, s_tp1] += 1.0
            row_sums[b_idx, s_t] += 1.0

            s_curr = s_tp1

            known_row = self.log_trans_3[:, s_curr, :]

            new_num = self.alpha + counts[b_idx, s_curr, :]
            new_den = (N * self.alpha + row_sums[b_idx, s_curr]).unsqueeze(-1)
            new_row = torch.log((new_num / new_den).clamp_min(1e-30))

            row = torch.cat([known_row, new_row.unsqueeze(0)], dim=0)

            log_num = torch.logsumexp(row + cum_logw.unsqueeze(-1), dim=0)
            log_den = torch.logsumexp(cum_logw, dim=0).unsqueeze(-1)
            preds[:, t + 1] = (log_num - log_den).exp()

        return preds


class ThreeKnownUniformBayes(nn.Module):
    """
    Bayesian predictor assuming the true transition matrix is one of 3 known candidates,
    with a uniform prior over the 3 tasks/chains.
    """
    def __init__(self, trans_mat_3: torch.Tensor, device: str = "cpu", eps: float = 1e-30):
        super().__init__()
        assert trans_mat_3.dim() == 3 and trans_mat_3.size(0) == 3, "Need (3, N, N)."
        _, N, N2 = trans_mat_3.shape
        assert N == N2

        self.N = N
        self.device = device
        self.eps = eps

        logT = torch.log(trans_mat_3.to(device).clamp_min(eps))
        self.register_buffer("log_trans", logT)

        log_pi = torch.log(torch.full((3,), 1.0 / 3.0, device=device))
        self.register_buffer("log_pi", log_pi)

    @torch.no_grad()
    def predict(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Args:
            samples: (B, T) integer tokens in {0,...,N-1}

        Returns:
            preds: (B, T, N) where preds[:, t] = p(s_{t+1} | s_{1:t})
        """
        samples = samples.to(self.device)
        B, T = samples.shape
        N = self.N

        preds = torch.zeros((B, T, N), device=self.device, dtype=torch.float)

        cum_logw = self.log_pi.view(3, 1).expand(3, B).clone()

        s0 = samples[:, 0]
        row0 = self.log_trans[:, s0, :]

        log_num = torch.logsumexp(row0 + cum_logw.unsqueeze(-1), dim=0)
        log_den = torch.logsumexp(cum_logw, dim=0).unsqueeze(-1)
        preds[:, 0] = (log_num - log_den).exp()

        log_trans_flat = self.log_trans.view(3, N * N)

        for t in range(T - 1):
            s_t = samples[:, t]
            s_tp1 = samples[:, t + 1]
            flat_idx = s_t * N + s_tp1

            obs_logp = log_trans_flat.gather(
                dim=1, index=flat_idx.view(1, B).expand(3, B)
            )
            cum_logw = cum_logw + obs_logp

            curr = s_tp1
            row = self.log_trans[:, curr, :]

            log_num = torch.logsumexp(row + cum_logw.unsqueeze(-1), dim=0)
            log_den = torch.logsumexp(cum_logw, dim=0).unsqueeze(-1)
            preds[:, t + 1] = (log_num - log_den).exp()

        return preds


@torch.no_grad()
def task_posterior_over_time(
    sampler,
    samples: torch.Tensor,
    *,
    include_minor: bool = True,
    return_log: bool = False,
    eps: float = 1e-30,
) -> torch.Tensor:
    """
    Compute filtering posterior P(Z=k | X_{0:t}) for each t, for sequences generated by LatentMarkov.

    Args:
        sampler: an instance of LatentMarkov.
        samples: Long tensor of shape (B, L_obs) or (E, B, L_obs).
        include_minor: whether to include minor task pool in the posterior.
        return_log: if True, return log posterior instead of posterior.
        eps: numerical floor for probabilities.

    Returns:
        post: Tensor of shape (..., L_real, T) where
              T = sampler.total_trans, L_real = sampler.seq_len.
    """
    device = samples.device
    dtype = torch.float32

    orig_shape = samples.shape
    if samples.dim() == 2:
        prefix_shape = (samples.shape[0],)
        samples_ = samples.unsqueeze(0)
    elif samples.dim() == 3:
        prefix_shape = samples.shape[:2]
        samples_ = samples
    else:
        raise ValueError(f"samples must have shape (B,L) or (E,B,L), got {samples.shape}")

    E, B, L_obs = samples_.shape

    x = samples_
    E, B, L = x.shape
    assert L == sampler.seq_len, f"Expected real length {sampler.seq_len}, got {L}"

    if include_minor:
        T = sampler.total_trans
    else:
        T = sampler.n_major_tasks
    if T <= 0:
        raise ValueError("Posterior requires at least one task in the selected pool.")
    order = sampler.order
    S = sampler.num_states
    powers = sampler.powers.to(device=device)

    if include_minor and sampler.n_minor_tasks > 0:
        trans_all = torch.cat([sampler.major_trans_mat, sampler.minor_trans_mat], dim=0).to(device=device)
    else:
        trans_all = sampler.major_trans_mat.to(device=device)
    assert trans_all.shape[0] == T

    if (not include_minor) or sampler.n_minor_tasks == 0:
        prior = torch.full((T,), 1.0 / max(1, sampler.n_major_tasks), device=device, dtype=dtype)
    else:
        prior_major = (1.0 - float(sampler.p_minor)) / max(1, sampler.n_major_tasks)
        prior_minor = float(sampler.p_minor) / max(1, sampler.n_minor_tasks)
        prior = torch.cat([
            torch.full((sampler.n_major_tasks,), prior_major, device=device, dtype=dtype),
            torch.full((sampler.n_minor_tasks,), prior_minor, device=device, dtype=dtype),
        ], dim=0)
    prior = torch.clamp(prior, min=eps)
    log_prior = prior.log()

    log_post = torch.empty((E, B, L, T), device=device, dtype=dtype)

    log_post[:, :, :min(order, L), :] = log_prior.view(1, 1, 1, T).expand(E, B, min(order, L), T)

    running = log_prior.view(1, 1, T).expand(E, B, T).clone()

    if L > order:
        state_hist = x[:, :, :order].clone()

        base = (S ** (order - 1)) if order > 0 else 1

        for t in range(order, L):
            idx = torch.sum(state_hist * powers.view(1, 1, -1), dim=-1).long()

            xt = x[:, :, t].long()

            probs_row = trans_all[:, idx]
            probs = probs_row.gather(dim=-1, index=xt.unsqueeze(0).unsqueeze(-1).expand(T, E, B, 1)).squeeze(-1)

            probs = torch.clamp(probs, min=eps)
            running = running + probs.permute(1, 2, 0).log()

            log_norm = torch.logsumexp(running, dim=-1, keepdim=True)
            log_post[:, :, t, :] = running - log_norm

            if order > 0:
                state_hist = torch.roll(state_hist, shifts=-1, dims=-1)
                state_hist[:, :, -1] = xt

    if return_log:
        out = log_post
    else:
        out = torch.exp(log_post)

    if len(prefix_shape) == 1:
        return out.squeeze(0)
    return out
