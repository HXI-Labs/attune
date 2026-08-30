"""Reproducible integrity audit for a prepared joint-training manifest."""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

import torch

from attune.training.data import JointFeatureDataset, file_sha256, manifest_sha256


def audit_joint_manifest(
    manifest: Path,
    *,
    expected_feature_size: int | None = 560,
    require_evaluation_supervision: bool = True,
) -> dict[str, Any]:
    datasets = {
        split: JointFeatureDataset(manifest, split=split)
        for split in ("train", "development", "sealed_test")
    }
    rows = [row for dataset in datasets.values() for row in dataset.rows]
    clip_ids = [row.clip_id for row in rows]
    if len(clip_ids) != len(set(clip_ids)):
        raise ValueError("clip IDs are not globally unique across splits")
    feature_paths = [row.feature_path for row in rows]
    if len(feature_paths) != len(set(feature_paths)):
        raise ValueError("feature paths are not globally unique")

    feature_sizes: collections.Counter[int] = collections.Counter()
    for row in rows:
        if not row.feature_path.is_file():
            raise FileNotFoundError(row.feature_path)
        if file_sha256(row.feature_path) != row.feature_sha256:
            raise ValueError(f"feature hash mismatch for {row.clip_id}")
        value = torch.load(row.feature_path, map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(f"invalid feature tensor for {row.clip_id}")
        feature_sizes[int(value.shape[-1])] += 1
    if expected_feature_size is not None and set(feature_sizes) != {expected_feature_size}:
        raise ValueError(
            f"expected only {expected_feature_size}-wide features; found {dict(feature_sizes)}"
        )

    split_units: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        split_units[row.dataset_id].add(row.split_unit)
    mixed_units = {
        dataset_id: sorted(units) for dataset_id, units in split_units.items() if len(units) > 1
    }
    if mixed_units:
        raise ValueError(f"datasets mix split-unit policies: {mixed_units}")

    speakers = {
        split: {
            row.speaker_id for row in dataset.rows if row.speaker_id and row.split_unit == "speaker"
        }
        for split, dataset in datasets.items()
    }
    overlaps = {
        f"{left}/{right}": len(speakers[left] & speakers[right])
        for left, right in (
            ("train", "development"),
            ("train", "sealed_test"),
            ("development", "sealed_test"),
        )
    }
    if any(overlaps.values()):
        raise ValueError(f"known speakers cross partitions: {overlaps}")

    pair_splits: dict[tuple[str, int], str] = {}
    crossed_pairs = []
    for row in rows:
        if row.split_unit != "sentence":
            continue
        key = (row.dataset_id, row.pair_id)
        previous = pair_splits.setdefault(key, row.split)
        if previous != row.split:
            crossed_pairs.append((row.dataset_id, row.pair_id, previous, row.split))
    if crossed_pairs:
        raise ValueError(f"sentence pairs cross partitions: {crossed_pairs[:5]}")

    dataset_counts = collections.Counter(row.dataset_id for row in rows)

    def supervision_counts(source_rows):
        counts = collections.Counter()
        for row in source_rows:
            counts.update(
                {
                    "asr": row.token_ids is not None,
                    "strong_events": row.events is not None,
                    "event_presence": row.event_presence is not None,
                    "styles": row.styles is not None,
                    "affect": row.affect_distribution is not None,
                    "vad": row.vad is not None,
                    "known_ood": row.is_ood,
                    "in_distribution": not row.is_ood,
                }
            )
        return dict(counts)

    supervision = supervision_counts(rows)
    supervision_by_split = {
        split: supervision_counts(dataset.rows) for split, dataset in datasets.items()
    }
    required = (
        "asr",
        "strong_events",
        "event_presence",
        "styles",
        "affect",
        "known_ood",
        "in_distribution",
    )
    if require_evaluation_supervision:
        missing = [
            f"{split}:{task}"
            for split in ("development", "sealed_test")
            for task in required
            if supervision_by_split[split].get(task, 0) == 0
        ]
        if missing:
            raise ValueError(f"evaluation splits lack required supervision: {missing}")
    return {
        "schema_version": "1.0",
        "manifest_sha256": manifest_sha256(manifest),
        "rows": len(rows),
        "splits": {split: len(dataset) for split, dataset in datasets.items()},
        "datasets": dict(sorted(dataset_counts.items())),
        "feature_sizes": {str(key): value for key, value in sorted(feature_sizes.items())},
        "unique_feature_paths": len(set(feature_paths)),
        "known_speaker_overlap": overlaps,
        "sentence_disjoint_pairs": len(pair_splits),
        "supervision_rows": dict(supervision),
        "supervision_by_split": supervision_by_split,
        "duration_hours": sum(row.duration_ms for row in rows) / 3_600_000.0,
    }
