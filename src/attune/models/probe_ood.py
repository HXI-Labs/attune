"""Data contract for bounded, speaker-disjoint CREMA probe negatives."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from attune.models.frozen_event_probe import ProbeDataError


@dataclass(frozen=True)
class ProbeNegative:
    """One genuine negative clip for both event/style probe ontologies."""

    path: Path
    clip_id: str
    partition: str
    speaker_id: str
    source_dataset: str


def crema_probe_negatives(
    manifest: Path,
    cache_root: Path,
    *,
    excluded_speakers: set[str],
) -> tuple[ProbeNegative, ...]:
    """Load and validate the bounded CREMA train/validation negative pool."""
    try:
        rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ProbeDataError(f"cannot read CREMA OOD manifest {manifest}: {error}") from error
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ProbeDataError(f"CREMA OOD manifest is empty or invalid: {manifest}")

    examples = tuple(_example(row, cache_root) for row in rows)
    clip_ids = [example.clip_id for example in examples]
    if len(clip_ids) != len(set(clip_ids)):
        raise ProbeDataError("CREMA OOD manifest contains duplicate clip IDs")
    partitions = {example.partition for example in examples}
    if partitions != {"train", "validation"}:
        raise ProbeDataError("CREMA OOD manifest requires train and validation partitions")
    speakers = {
        partition: {
            example.speaker_id
            for example in examples
            if example.partition == partition
        }
        for partition in partitions
    }
    if speakers["train"] & speakers["validation"]:
        raise ProbeDataError("a CREMA OOD speaker crosses train and validation")
    if (speakers["train"] | speakers["validation"]) & excluded_speakers:
        raise ProbeDataError("a CREMA OOD speaker appears in the 310-clip inspection")
    if any(example.source_dataset != "CREMA-D" for example in examples):
        raise ProbeDataError("CREMA OOD manifest contains another source dataset")
    return examples


def partition_negatives(
    examples: tuple[ProbeNegative, ...],
    partition: str,
) -> tuple[ProbeNegative, ...]:
    """Select one validated partition."""
    return tuple(example for example in examples if example.partition == partition)


def source_speakers(manifests: tuple[Path, ...], source_dataset: str) -> set[str]:
    """Collect namespaced source speakers from one or more inspection manifests."""
    speakers: set[str] = set()
    for manifest in manifests:
        try:
            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as error:
            raise ProbeDataError(f"cannot read inspection manifest {manifest}: {error}") from error
        speakers.update(
            str(row["speaker_id"])
            for row in rows
            if row.get("source_dataset") == source_dataset
        )
    return speakers


def _example(row: dict[str, Any], cache_root: Path) -> ProbeNegative:
    required = {
        "cache_path",
        "clip_id",
        "partition",
        "speaker_id",
        "source_dataset",
    }
    missing = required - row.keys()
    if missing:
        raise ProbeDataError(
            f"CREMA OOD manifest row lacks fields: {', '.join(sorted(missing))}"
        )
    partition = str(row["partition"])
    if partition not in {"train", "validation"}:
        raise ProbeDataError(f"unsupported CREMA OOD partition: {partition}")
    return ProbeNegative(
        path=cache_root / str(row["cache_path"]),
        clip_id=str(row["clip_id"]),
        partition=partition,
        speaker_id=str(row["speaker_id"]),
        source_dataset=str(row["source_dataset"]),
    )
