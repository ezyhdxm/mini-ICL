"""PrefixProbe model: shared projection + per-length MLP heads."""

import torch.nn as nn


class PrefixProbe(nn.Module):
    """
    Shared linear projection + separate MLP head per prefix length.

    The linear projection is trained jointly across all prefix lengths.
    Each prefix length has its own full MLP classifier on top.
    """

    def __init__(self, d_model, proj_dim, n_classes_per_length, mlp_hidden=64):
        super().__init__()
        self.proj = nn.Linear(d_model, proj_dim)
        self.heads = nn.ModuleDict()
        for l, n_cls in n_classes_per_length.items():
            self.heads[str(l)] = nn.Sequential(
                nn.Linear(proj_dim, mlp_hidden),
                nn.SiLU(),
                nn.Linear(mlp_hidden, n_cls),
            )

    def project(self, h):
        return self.proj(h)

    def classify(self, z, prefix_len):
        return self.heads[str(prefix_len)](z)

    def forward(self, h, prefix_len):
        z = self.project(h)
        return self.classify(z, prefix_len)
