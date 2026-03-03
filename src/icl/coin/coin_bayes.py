import torch
from typing import Optional
import torch.nn as nn


class KnownPoolBayesCoins(nn.Module):
    """
    Exact Bayesian predictor for coin sequences with a finite known task pool.

    Hypotheses are all known coins:
      - major pool probabilities (required)
      - minor pool probabilities (optional)

    This is the exact Bayes model when the true generating coin is assumed to lie
    in that finite pool. Supports non-padded sequences directly.
    """
    def __init__(
        self,
        probs_major: torch.Tensor,
        probs_minor: Optional[torch.Tensor] = None,
        p_minor: float = 0.1,
        include_minor: bool = True,
        device: str = "cpu",
        eps: float = 1e-30,
    ):
        super().__init__()
        if probs_major.dim() != 2 or probs_major.size(0) <= 0:
            raise ValueError("probs_major must have shape (Kmaj, V) with Kmaj > 0.")

        major = probs_major.to(device=device, dtype=torch.float32)
        Kmaj, V = major.shape

        use_minor = (
            include_minor
            and probs_minor is not None
            and probs_minor.dim() == 2
            and probs_minor.size(0) > 0
        )
        if use_minor:
            minor = probs_minor.to(device=device, dtype=torch.float32)
            if minor.size(1) != V:
                raise ValueError(f"probs_minor vocab mismatch: {minor.size(1)} != {V}")
            pool = torch.cat([major, minor], dim=0)  # (Ktot, V)
            Kmin = int(minor.size(0))
        else:
            pool = major
            Kmin = 0

        if Kmin == 0:
            prior = torch.full((Kmaj,), 1.0 / Kmaj, device=device, dtype=torch.float32)
        else:
            p0 = float(p_minor)
            prior_major = (1.0 - p0) / Kmaj
            prior_minor = p0 / Kmin
            prior = torch.cat(
                [
                    torch.full((Kmaj,), prior_major, device=device, dtype=torch.float32),
                    torch.full((Kmin,), prior_minor, device=device, dtype=torch.float32),
                ],
                dim=0,
            )

        prior = prior.clamp_min(eps)
        pool = pool.clamp_min(eps)

        self.eps = float(eps)
        self.device = device
        self.Kmaj = int(Kmaj)
        self.Kmin = int(Kmin)
        self.V = int(V)
        self.Ktot = int(pool.size(0))

        self.register_buffer("pool_probs", pool)                    # (Ktot, V)
        self.register_buffer("log_pool_probs", torch.log(pool))     # (Ktot, V)
        self.register_buffer("log_prior", torch.log(prior))         # (Ktot,)

    @torch.no_grad()
    def predict(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Posterior predictive distribution over next token at each position.

        Args:
            samples: (B, T_real) non-padded token ids.

        Returns:
            preds: (B, T_real, V), where preds[:, t] = p(x_{t+1} | x_{1:t+1}).
        """
        if samples.dim() != 2:
            raise ValueError(f"samples must be (B,T), got {samples.shape}")

        x = samples.to(self.device)

        B, T = x.shape
        V = self.V
        preds = torch.zeros((B, T, V), device=self.device, dtype=torch.float32)

        counts = torch.zeros((B, V), device=self.device, dtype=torch.float32)
        ones = torch.ones((B, 1), device=self.device, dtype=torch.float32)

        for t in range(T):
            tok = x[:, t : t + 1].long()  # (B,1)
            counts.scatter_add_(dim=-1, index=tok, src=ones)

            loglik = counts @ self.log_pool_probs.T                       # (B,Ktot)
            unnorm = loglik + self.log_prior.unsqueeze(0)                # (B,Ktot)
            log_post = unnorm - torch.logsumexp(unnorm, dim=-1, keepdim=True)
            post = torch.exp(log_post)                                   # (B,Ktot)

            preds[:, t, :] = post @ self.pool_probs                      # (B,V)

        return preds


class ThreeKnownPlusNewDirichletCoinBayes(nn.Module):
    """
    Hybrid Bayesian predictor for coin sequences:
      - 3 known major coins (exact Bayesian update)
      - 1 aggregated "new" coin with Dirichlet prior (minor+OOD merged)

    This matches "three-known + new Dirichlet" in spirit, adapted to i.i.d.
    categorical observations. Supports non-padded sequences directly.
    """
    def __init__(
        self,
        probs_major_3: torch.Tensor,
        p_common_total: float = 0.9,
        alpha: float = 1.0,
        device: str = "cpu",
        eps: float = 1e-30,
    ):
        super().__init__()
        if probs_major_3.dim() != 2 or probs_major_3.size(0) != 3:
            raise ValueError("probs_major_3 must have shape (3, V).")

        major = probs_major_3.to(device=device, dtype=torch.float32).clamp_min(eps)
        _, V = major.shape

        pi = torch.tensor(
            [
                p_common_total / 3.0,
                p_common_total / 3.0,
                p_common_total / 3.0,
                1.0 - p_common_total,
            ],
            device=device,
            dtype=torch.float32,
        ).clamp_min(eps)

        self.device = device
        self.eps = float(eps)
        self.V = int(V)
        self.alpha = float(alpha)

        self.register_buffer("major_probs", major)                  # (3,V)
        self.register_buffer("log_major_probs", torch.log(major))   # (3,V)
        self.register_buffer("log_pi", torch.log(pi))               # (4,)

    @torch.no_grad()
    def predict(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Posterior predictive over next token at each position.

        Args:
            samples: (B, T_real) non-padded token ids.

        Returns:
            preds: (B, T_real, V)
        """
        if samples.dim() != 2:
            raise ValueError(f"samples must be (B,T), got {samples.shape}")

        x = samples.to(self.device)

        B, T = x.shape
        V = self.V
        alpha = self.alpha
        alpha0 = V * alpha

        preds = torch.zeros((B, T, V), device=self.device, dtype=torch.float32)

        counts = torch.zeros((B, V), device=self.device, dtype=torch.float32)
        total = torch.zeros((B,), device=self.device, dtype=torch.float32)
        ones = torch.ones((B, 1), device=self.device, dtype=torch.float32)

        for t in range(T):
            tok = x[:, t : t + 1].long()  # (B,1)
            counts.scatter_add_(dim=-1, index=tok, src=ones)
            total = total + 1.0

            ll_known = counts @ self.log_major_probs.T                        # (B,3)

            ll_new = (
                torch.lgamma(torch.tensor(alpha0, device=self.device))
                - torch.lgamma(alpha0 + total)
                + (torch.lgamma(alpha + counts) - torch.lgamma(torch.tensor(alpha, device=self.device))).sum(dim=-1)
            )  # (B,)

            unnorm = torch.cat([ll_known, ll_new.unsqueeze(-1)], dim=-1) + self.log_pi.unsqueeze(0)  # (B,4)
            log_post = unnorm - torch.logsumexp(unnorm, dim=-1, keepdim=True)
            post = torch.exp(log_post)                                        # (B,4)

            pred_known = post[:, :3] @ self.major_probs                       # (B,V)
            pred_new = (counts + alpha) / (total.unsqueeze(-1) + alpha0)      # (B,V)
            preds[:, t, :] = pred_known + post[:, 3:4] * pred_new             # (B,V)

        return preds
