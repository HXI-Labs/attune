from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from attune.inference.probe_head import LinearProbeHead
from attune.models.sensevoice_probe import SENSEVOICE_EN_EMBEDDING


def write_probe(path: Path, *, threshold: float = 0.2) -> None:
    np.savez_compressed(
        path,
        format=np.asarray("attune_linear_probe_v1"),
        embedding=np.asarray(SENSEVOICE_EN_EMBEDDING),
        labels=np.asarray(["cough"]),
        channels=np.asarray(["event"]),
        target_labels=np.asarray(["cough"]),
        weight=np.asarray([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float32),
        bias=np.zeros(2, dtype=np.float32),
        feature_mean=np.zeros(2, dtype=np.float32),
        feature_scale=np.ones(2, dtype=np.float32),
        abstention_method=np.asarray("none_logit"),
        abstention_threshold=np.asarray(threshold, dtype=np.float32),
        temperature=np.asarray(2.0, dtype=np.float32),
    )


def test_linear_probe_emits_only_above_none_logit_margin(tmp_path: Path) -> None:
    artifact = tmp_path / "probe.npz"
    write_probe(artifact)
    head = LinearProbeHead(artifact)

    accepted = head.predict(np.asarray([1.0, 0.0], dtype=np.float32))
    rejected = head.predict(np.asarray([0.0, 0.0], dtype=np.float32))

    assert accepted.label == "cough"
    assert accepted.abstained is False
    assert accepted.confidence == pytest.approx(0.880797, rel=1e-5)
    assert rejected.abstained is True
    assert head.parameter_count == 6


def test_linear_probe_rejects_unvalidated_abstention(tmp_path: Path) -> None:
    artifact = tmp_path / "probe.npz"
    write_probe(artifact)
    with np.load(artifact, allow_pickle=False) as source:
        values = {name: source[name] for name in source.files}
    values["abstention_method"] = np.asarray("max_softmax")
    np.savez_compressed(artifact, **values)

    with pytest.raises(ValueError, match="none-logit"):
        LinearProbeHead(artifact)
