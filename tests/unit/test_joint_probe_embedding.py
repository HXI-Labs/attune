from __future__ import annotations

import torch

from attune.models.joint import _pooled_probe_embedding
from attune.models.sensevoice_probe import pool_sensevoice_frames


def test_probe_embedding_matches_frozen_encoder_pooling_with_padding() -> None:
    torch.manual_seed(7)
    frames = torch.randn(2, 17, 512)
    lengths = torch.tensor([17, 11])
    mask = torch.arange(frames.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    frames[1, 11:] = 1_000

    actual = _pooled_probe_embedding(frames, mask)
    expected = torch.stack(
        [
            pool_sensevoice_frames(frames[index, :length], torch)
            for index, length in enumerate(lengths.tolist())
        ]
    )

    torch.testing.assert_close(actual, expected)
