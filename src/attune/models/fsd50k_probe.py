"""Clip-disjoint FSD50K data contract for the frozen SenseVoice probe."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from attune.models.frozen_event_probe import ProbeDataError

FSD50K_PROBE_LABELS = ("shout", "whisper", "sob", "scream")
SOURCE_TO_PROBE_LABEL = {
    "Shout": "shout",
    "Whispering": "whisper",
    "Crying_and_sobbing": "sob",
    "Screaming": "scream",
}


@dataclass(frozen=True)
class FSD50KProbeExample:
    """One source-labelled clip in the bounded four-way probe protocol."""

    path: Path
    clip_id: str
    partition: str
    source_class: str
    label: str


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a non-empty JSONL manifest with actionable errors."""
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ProbeDataError(f"cannot read FSD50K probe manifest {path}: {error}") from error
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ProbeDataError(f"FSD50K probe manifest is empty or invalid: {path}")
    return rows


def training_examples(manifest: Path, cache_root: Path) -> tuple[FSD50KProbeExample, ...]:
    """Load deterministic train and validation rows from the probe manifest."""
    examples = tuple(_example(row, cache_root) for row in load_jsonl(manifest))
    invalid = [
        example.partition
        for example in examples
        if example.partition not in {"train", "validation"}
    ]
    if invalid:
        raise ProbeDataError("probe manifest may contain only train and validation rows")
    return examples


def inspection_examples(
    manifest: Path, cache_root: Path
) -> tuple[FSD50KProbeExample, ...]:
    """Load the immutable 100-row FSD50K inspection set as probe test data."""
    rows = [
        row
        for row in load_jsonl(manifest)
        if row.get("source_dataset") == "FSD50K"
    ]
    if not rows:
        raise ProbeDataError(f"no FSD50K rows found in inspection manifest {manifest}")
    return tuple(
        FSD50KProbeExample(
            path=cache_root / row["cache_path"],
            clip_id=str(row["clip_id"]),
            partition="inspection_test",
            source_class=str(row["intended_attune_labels"]["source_class"]),
            label=_probe_label(row["intended_attune_labels"]["source_class"]),
        )
        for row in rows
    )


def validate_clip_disjoint(
    train: tuple[FSD50KProbeExample, ...],
    validation: tuple[FSD50KProbeExample, ...],
    test: tuple[FSD50KProbeExample, ...],
) -> None:
    """Fail closed on clip leakage, duplicate IDs, or missing partition labels."""
    partitions = {"train": train, "validation": validation, "inspection_test": test}
    ids = {
        name: [example.clip_id for example in examples]
        for name, examples in partitions.items()
    }
    if any(len(values) != len(set(values)) for values in ids.values()):
        raise ProbeDataError("a probe partition contains duplicate clip IDs")
    sets = {name: set(values) for name, values in ids.items()}
    if (
        sets["train"] & sets["validation"]
        or sets["train"] & sets["inspection_test"]
        or sets["validation"] & sets["inspection_test"]
    ):
        raise ProbeDataError("an FSD50K clip crosses train, validation, and/or inspection test")
    expected = set(FSD50K_PROBE_LABELS)
    for name, examples in partitions.items():
        present = {example.label for example in examples}
        if present != expected:
            missing = sorted(expected - present)
            raise ProbeDataError(f"{name} partition lacks labels: {', '.join(missing)}")


def classification_metrics(
    targets: list[int], predictions: list[int]
) -> dict[str, Any]:
    """Compute dependency-free four-way metrics for the source-label diagnostic."""
    if not targets or len(targets) != len(predictions):
        raise ValueError("targets and predictions must be aligned and non-empty")
    per_class: dict[str, Any] = {}
    for index, label in enumerate(FSD50K_PROBE_LABELS):
        true_positive = sum(
            target == prediction == index
            for target, prediction in zip(targets, predictions, strict=True)
        )
        false_positive = sum(
            target != index and prediction == index
            for target, prediction in zip(targets, predictions, strict=True)
        )
        false_negative = sum(
            target == index and prediction != index
            for target, prediction in zip(targets, predictions, strict=True)
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(target == index for target in targets),
        }
    return {
        "accuracy": sum(
            target == prediction
            for target, prediction in zip(targets, predictions, strict=True)
        )
        / len(targets),
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(per_class),
        "per_class": per_class,
    }


def _example(row: dict[str, Any], cache_root: Path) -> FSD50KProbeExample:
    source_class = str(row["source_class"])
    return FSD50KProbeExample(
        path=cache_root / row["cache_path"],
        clip_id=str(row["clip_id"]),
        partition=str(row["partition"]),
        source_class=source_class,
        label=_probe_label(source_class),
    )


def _probe_label(source_class: Any) -> str:
    try:
        return SOURCE_TO_PROBE_LABEL[str(source_class)]
    except KeyError as error:
        raise ProbeDataError(f"unsupported FSD50K probe source class: {source_class}") from error
