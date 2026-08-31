#!/usr/bin/env python3
"""Merge validated joint manifests while preserving feature-path provenance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from attune.integrity import file_digest
from attune.training.data import JointManifestRow


def merge_manifests(
    inputs: list[Path],
    output: Path,
    *,
    speaker_overlap_allowed: set[str] | None = None,
    pair_overlap_allowed: set[str] | None = None,
    minimum_duration_ms: int | None = None,
    maximum_duration_ms: int | None = None,
    training_maximum_duration_ms: int | None = None,
    excluded_datasets: set[str] | None = None,
) -> list[JointManifestRow]:
    if len(inputs) < 2:
        raise ValueError("merge requires at least two input manifests")
    output = output.resolve()
    speaker_overlap_allowed = speaker_overlap_allowed or set()
    pair_overlap_allowed = pair_overlap_allowed or set()
    excluded_datasets = excluded_datasets or set()
    if minimum_duration_ms is not None and minimum_duration_ms <= 0:
        raise ValueError("minimum duration must be positive")
    if maximum_duration_ms is not None and maximum_duration_ms <= 0:
        raise ValueError("maximum duration must be positive")
    if training_maximum_duration_ms is not None and training_maximum_duration_ms <= 0:
        raise ValueError("training maximum duration must be positive")
    if (
        minimum_duration_ms is not None
        and maximum_duration_ms is not None
        and maximum_duration_ms < minimum_duration_ms
    ):
        raise ValueError("maximum duration must not be below minimum duration")
    rows: list[JointManifestRow] = []
    excluded_rows: list[dict[str, object]] = []
    dataset_excluded_rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    speaker_splits: dict[tuple[str, str], str] = {}
    pair_splits: dict[tuple[str, int], str] = {}
    for source in inputs:
        source = source.resolve()
        for line_number, line in enumerate(source.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = JointManifestRow.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
            if row.dataset_id in excluded_datasets:
                dataset_excluded_rows.append(
                    {
                        "dataset_id": row.dataset_id,
                        "clip_id": row.clip_id,
                        "duration_ms": row.duration_ms,
                        "reason": "excluded_dataset",
                    }
                )
                continue
            if minimum_duration_ms is not None and row.duration_ms < minimum_duration_ms:
                excluded_rows.append(
                    {
                        "dataset_id": row.dataset_id,
                        "clip_id": row.clip_id,
                        "duration_ms": row.duration_ms,
                        "reason": "below_minimum_duration",
                    }
                )
                continue
            if maximum_duration_ms is not None and row.duration_ms > maximum_duration_ms:
                excluded_rows.append(
                    {
                        "dataset_id": row.dataset_id,
                        "clip_id": row.clip_id,
                        "duration_ms": row.duration_ms,
                        "reason": "above_maximum_duration",
                    }
                )
                continue
            if (
                training_maximum_duration_ms is not None
                and row.split == "train"
                and row.duration_ms > training_maximum_duration_ms
            ):
                excluded_rows.append(
                    {
                        "dataset_id": row.dataset_id,
                        "clip_id": row.clip_id,
                        "duration_ms": row.duration_ms,
                        "reason": "above_training_maximum_duration",
                    }
                )
                continue
            key = (row.dataset_id, row.clip_id)
            if key in seen:
                raise ValueError(f"duplicate merged clip: {key}")
            seen.add(key)
            if row.speaker_id is not None and row.split_unit == "speaker":
                speaker_key = (row.dataset_id, row.speaker_id)
                previous = speaker_splits.setdefault(speaker_key, row.split)
                if previous != row.split and row.dataset_id not in speaker_overlap_allowed:
                    raise ValueError(f"speaker crosses merged partitions: {speaker_key}")
            if row.pair_id >= 0 and row.split_unit == "sentence":
                pair_key = (row.dataset_id, row.pair_id)
                previous = pair_splits.setdefault(pair_key, row.split)
                if previous != row.split and row.dataset_id not in pair_overlap_allowed:
                    raise ValueError(f"pair crosses merged partitions: {pair_key}")
            feature_path = row.feature_path
            if not feature_path.is_absolute():
                feature_path = (source.parent / feature_path).resolve()
            if not feature_path.is_file():
                raise ValueError(f"missing feature file: {feature_path}")
            row.feature_path = Path(os.path.relpath(feature_path, output.parent))
            rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(row.model_dump_json() + "\n" for row in rows))
    provenance = {
        "schema_version": "1.0",
        "inputs": [
            {
                "path": str(path),
                "sha256": file_digest(path),
                "rows": len(path.read_text().splitlines()),
            }
            for path in inputs
        ],
        "speaker_overlap_allowed": sorted(speaker_overlap_allowed),
        "pair_overlap_allowed": sorted(pair_overlap_allowed),
        "dataset_filter": {
            "excluded_datasets": sorted(excluded_datasets),
            "excluded_count": len(dataset_excluded_rows),
            "excluded_rows": dataset_excluded_rows,
        },
        "duration_filter": {
            "minimum_duration_ms": minimum_duration_ms,
            "maximum_duration_ms": maximum_duration_ms,
            "training_maximum_duration_ms": training_maximum_duration_ms,
            "excluded_count": len(excluded_rows),
            "excluded_rows": excluded_rows,
        },
        "output": str(output),
        "rows": len(rows),
        "output_sha256": file_digest(output),
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-speaker-overlap-dataset", action="append", default=[])
    parser.add_argument("--allow-pair-overlap-dataset", action="append", default=[])
    parser.add_argument("--minimum-duration-ms", type=int)
    parser.add_argument("--maximum-duration-ms", type=int)
    parser.add_argument("--training-maximum-duration-ms", type=int)
    parser.add_argument("--exclude-dataset", action="append", default=[])
    arguments = parser.parse_args()
    rows = merge_manifests(
        arguments.input,
        arguments.output,
        speaker_overlap_allowed=set(arguments.allow_speaker_overlap_dataset),
        pair_overlap_allowed=set(arguments.allow_pair_overlap_dataset),
        minimum_duration_ms=arguments.minimum_duration_ms,
        maximum_duration_ms=arguments.maximum_duration_ms,
        training_maximum_duration_ms=arguments.training_maximum_duration_ms,
        excluded_datasets=set(arguments.exclude_dataset),
    )
    print(f"Merged {len(rows)} rows into {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
