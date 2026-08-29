#!/usr/bin/env python3
"""Prepare speaker-disjoint weak VocalSound presence rows for joint training."""

from __future__ import annotations

import argparse
import hashlib
import wave
from pathlib import Path

from attune.models.frozen_event_probe import (
    discover_vocalsound,
    load_inspection_rows,
)
from attune.training.data import file_sha256
from attune.training.source_adapters import write_source_rows


def _partition(speaker_id: str) -> str:
    bucket = int(hashlib.sha256(speaker_id.encode()).hexdigest()[:8], 16) % 5
    return "development" if bucket == 0 else "train"


def _duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as audio:
        if audio.getframerate() != 16_000 or audio.getnchannels() != 1:
            raise ValueError(f"VocalSound audio must be 16 kHz mono: {path}")
        return round(audio.getnframes() / audio.getframerate() * 1000)


def source_rows(
    *,
    dataset_root: Path,
    inspection_manifest: Path,
    inspection_cache: Path,
) -> list[dict[str, object]]:
    inspection = load_inspection_rows(inspection_manifest)
    excluded = {str(row["speaker_id"]) for row in inspection}
    rows: list[dict[str, object]] = []
    for example in discover_vocalsound(dataset_root):
        if example.speaker_id in excluded:
            continue
        rows.append(
            {
                "clip_id": f"vocalsound-joint-{example.path.stem.lower()}",
                "dataset_id": "vocalsound_v0.1",
                "split": _partition(example.speaker_id),
                "speaker_id": example.speaker_id,
                "audio_path": str(example.path.resolve()),
                "audio_sha256": file_sha256(example.path),
                "duration_ms": _duration_ms(example.path),
                "transcript": None,
                "events": None,
                "event_presence": [example.label.value],
                "styles": None,
                "affect_distribution": None,
                "vad": None,
                "pair_id": -1,
                "is_ood": True,
                "lexical_affect_label": None,
            }
        )
    for row in inspection:
        path = inspection_cache / row["cache_path"]
        rows.append(
            {
                "clip_id": row["clip_id"],
                "dataset_id": "vocalsound_v0.1",
                "split": "sealed_test",
                "speaker_id": row["speaker_id"],
                "audio_path": str(path.resolve()),
                "audio_sha256": row["sha256"],
                "duration_ms": round(float(row["duration_s"]) * 1000),
                "transcript": None,
                "events": None,
                "event_presence": row["intended_attune_labels"]["events"],
                "styles": None,
                "affect_distribution": None,
                "vad": None,
                "pair_id": -1,
                "is_ood": True,
                "lexical_affect_label": None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/raw/vocalsound-16k"))
    parser.add_argument(
        "--inspection-manifest",
        type=Path,
        default=Path("data/manifests/inspection-set.jsonl"),
    )
    parser.add_argument("--inspection-cache", type=Path, default=Path("data/raw/inspection-set"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/vocalsound-joint-source-v0.1.jsonl"),
    )
    arguments = parser.parse_args()
    rows = source_rows(
        dataset_root=arguments.dataset_dir,
        inspection_manifest=arguments.inspection_manifest,
        inspection_cache=arguments.inspection_cache,
    )
    write_source_rows(arguments.output, rows)
    print(f"Wrote {len(rows)} speaker-disjoint VocalSound source rows")


if __name__ == "__main__":
    main()
