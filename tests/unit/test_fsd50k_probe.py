from __future__ import annotations

from pathlib import Path

import pytest

from attune.models.frozen_event_probe import ProbeDataError
from attune.models.fsd50k_probe import (
    FSD50K_PROBE_LABELS,
    FSD50KProbeExample,
    classification_metrics,
    validate_clip_disjoint,
)


def example(clip_id: str, partition: str, label: str) -> FSD50KProbeExample:
    return FSD50KProbeExample(
        path=Path(f"{clip_id}.wav"),
        clip_id=clip_id,
        partition=partition,
        source_class=label,
        label=label,
    )


def partition(name: str, suffix: str) -> tuple[FSD50KProbeExample, ...]:
    return tuple(example(f"{label}-{suffix}", name, label) for label in FSD50K_PROBE_LABELS)


def test_clip_disjoint_contract_accepts_all_four_separate_classes() -> None:
    validate_clip_disjoint(
        partition("train", "train"),
        partition("validation", "validation"),
        partition("inspection_test", "test"),
    )


def test_clip_disjoint_contract_rejects_inspection_leakage() -> None:
    train = partition("train", "shared")
    validation = partition("validation", "validation")
    test = partition("inspection_test", "shared")

    with pytest.raises(ProbeDataError, match="crosses"):
        validate_clip_disjoint(train, validation, test)


def test_four_way_metrics_keep_scream_separate_from_shout() -> None:
    metrics = classification_metrics(
        targets=[0, 1, 2, 3],
        predictions=[0, 1, 2, 0],
    )

    assert metrics["per_class"]["shout"]["f1"] == pytest.approx(2 / 3)
    assert metrics["per_class"]["whisper"]["f1"] == 1.0
    assert metrics["per_class"]["sob"]["f1"] == 1.0
    assert metrics["per_class"]["scream"]["f1"] == 0.0
