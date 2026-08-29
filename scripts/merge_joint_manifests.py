#!/usr/bin/env python3
"""Merge validated joint manifests while preserving feature-path provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from attune.training.data import JointManifestRow


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def merge_manifests(
    inputs: list[Path],
    output: Path,
    *,
    speaker_overlap_allowed: set[str] | None = None,
    pair_overlap_allowed: set[str] | None = None,
) -> list[JointManifestRow]:
    if len(inputs) < 2:
        raise ValueError("merge requires at least two input manifests")
    output = output.resolve()
    speaker_overlap_allowed = speaker_overlap_allowed or set()
    pair_overlap_allowed = pair_overlap_allowed or set()
    rows: list[JointManifestRow] = []
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
            key = (row.dataset_id, row.clip_id)
            if key in seen:
                raise ValueError(f"duplicate merged clip: {key}")
            seen.add(key)
            if row.speaker_id is not None:
                speaker_key = (row.dataset_id, row.speaker_id)
                previous = speaker_splits.setdefault(speaker_key, row.split)
                if previous != row.split and row.dataset_id not in speaker_overlap_allowed:
                    raise ValueError(f"speaker crosses merged partitions: {speaker_key}")
            if row.pair_id >= 0:
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
                "sha256": file_sha256(path),
                "rows": len(path.read_text().splitlines()),
            }
            for path in inputs
        ],
        "speaker_overlap_allowed": sorted(speaker_overlap_allowed),
        "pair_overlap_allowed": sorted(pair_overlap_allowed),
        "output": str(output),
        "rows": len(rows),
        "output_sha256": file_sha256(output),
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
    arguments = parser.parse_args()
    rows = merge_manifests(
        arguments.input,
        arguments.output,
        speaker_overlap_allowed=set(arguments.allow_speaker_overlap_dataset),
        pair_overlap_allowed=set(arguments.allow_pair_overlap_dataset),
    )
    print(f"Merged {len(rows)} rows into {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
