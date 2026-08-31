"""Compact affect head used with a truncated emotion2vec+ encoder."""

from __future__ import annotations

import torch
from torch import nn


class TruncatedEmotion2VecHead(nn.Module):
    def __init__(self, feature_count: int, label_count: int, *, kind: str = "mlp"):
        super().__init__()
        if kind == "linear":
            self.network = nn.Linear(feature_count, label_count)
        elif kind == "mlp":
            self.network = nn.Sequential(
                nn.Linear(feature_count, 128),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(64, label_count),
            )
        else:
            raise ValueError(f"unsupported affect head: {kind}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def pool_layer(
    layer: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool emotion2vec frames after its ten non-audio prefix tokens."""
    features = layer[:, 10:]
    valid = (~padding_mask).unsqueeze(-1)
    count = valid.sum(dim=1).clamp_min(1)
    mean = (features * valid).sum(dim=1) / count
    variance = ((features - mean.unsqueeze(1)).square() * valid).sum(dim=1) / count
    return torch.cat((mean, variance.clamp_min(1e-6).sqrt()), dim=-1)
