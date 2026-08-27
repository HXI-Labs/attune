#!/usr/bin/env python3
"""Build bounded strong-label DCASE clips after the committed licence review."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path
from typing import Any

WINDOW_MS = 10_000
INSPECTION_CLIPS = 100
LABEL_MAP = {
    "clearthroat": "throat_clear",
    "cough": "cough",
    "laughter": "laugh",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def annotations(path: Path) -> list[tuple[float, float, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            onset, offset, label = line.split("\t")
            rows.append((float(onset), float(offset), label))
    return rows


def cut(source: Path, target: Path, start_ms: int, end_ms: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), "rb") as reader:
        rate = reader.getframerate()
        reader.setpos(round(start_ms * rate / 1000))
        frames = reader.readframes(round((end_ms - start_ms) * rate / 1000))
        parameters = reader.getparams()
    with wave.open(str(target), "wb") as writer:
        writer.setparams(parameters)
        writer.writeframes(frames)


def source_rows(
    source_root: Path,
    *,
    source_partition: str,
    cache_root: Path,
) -> list[dict[str, Any]]:
    sound_root = source_root / "sound"
    annotation_root = source_root / "annotation"
    rows = []
    for source in sorted(sound_root.glob("*.wav")):
        events = annotations(annotation_root / f"{source.stem}.txt")
        with wave.open(str(source), "rb") as reader:
            duration_ms = round(reader.getnframes() / reader.getframerate() * 1000)
        for start_ms in range(0, duration_ms, WINDOW_MS):
            end_ms = min(start_ms + WINDOW_MS, duration_ms)
            if end_ms - start_ms < WINDOW_MS:
                continue
            relevant = [
                {
                    "label": LABEL_MAP[label],
                    "start_ms": max(0, round(onset * 1000) - start_ms),
                    "end_ms": min(WINDOW_MS, round(offset * 1000) - start_ms),
                    "source_label": label,
                }
                for onset, offset, label in events
                if label in LABEL_MAP
                and round(offset * 1000) > start_ms
                and round(onset * 1000) < end_ms
            ]
            clip_id = f"dcase2016-{source_partition}-{source.stem}-{start_ms:06d}"
            cache_path = f"{source_partition}/{clip_id}.wav"
            target = cache_root / cache_path
            cut(source, target, start_ms, end_ms)
            rows.append(
                {
                    "clip_id": clip_id,
                    "cache_path": cache_path,
                    "sha256": digest(target),
                    "duration_ms": WINDOW_MS,
                    "partition": source_partition,
                    "source_recording": source.name,
                    "source_window_start_ms": start_ms,
                    "events": relevant,
                    "annotation_type": "strong synthetic onset/offset",
                    "label_status": "synthetic strong label; not human-reviewed Attune gold",
                    "licence": "CC-BY-3.0" if source_partition == "development" else "CC-BY-4.0",
                    "attribution": (
                        "IEEE DCASE 2016 Task 2 by Grégoire Lafay, IRCCYN / "
                        "Ecole Centrale de Nantes."
                    ),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/dcase2016-task2"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/dcase2016-localization"),
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=Path("data/manifests/dcase2016-localization-training.jsonl"),
    )
    parser.add_argument(
        "--inspection-manifest",
        type=Path,
        default=Path("data/manifests/dcase2016-localization-inspection.jsonl"),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("data/manifests/dcase2016-localization.provenance.json"),
    )
    arguments = parser.parse_args()
    development_root = (
        arguments.root
        / "train-dev/dcase2016_task2_train_dev/dcase2016_task2_dev"
    )
    test_root = arguments.root / "public-test/dcase2016_task2_test_public"
    if not development_root.is_dir() or not test_root.is_dir():
        raise SystemExit(
            "error: extract the reviewed DCASE train/dev and public-test archives first"
        )
    training = source_rows(
        development_root,
        source_partition="development",
        cache_root=arguments.cache_dir,
    )
    test_candidates = [
        row
        for row in source_rows(
            test_root,
            source_partition="inspection_test",
            cache_root=arguments.cache_dir,
        )
        if row["events"]
    ]
    inspection = test_candidates[:INSPECTION_CLIPS]
    if len(inspection) != INSPECTION_CLIPS:
        raise SystemExit(
            f"error: only {len(inspection)} event-bearing test windows were available"
        )
    for path, rows in (
        (arguments.training_manifest, training),
        (arguments.inspection_manifest, inspection),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    provenance = {
        "schema_version": 1,
        "licence_review": "data/provenance/dcase2016_task2.yaml",
        "training_manifest": str(arguments.training_manifest),
        "training_rows": len(training),
        "training_manifest_sha256": digest(arguments.training_manifest),
        "inspection_manifest": str(arguments.inspection_manifest),
        "inspection_rows": len(inspection),
        "inspection_manifest_sha256": digest(arguments.inspection_manifest),
        "selection": (
            "All non-overlapping 10-second development windows for fitting; first "
            "100 lexically sorted event-bearing public-test windows for untouched inspection."
        ),
        "source_disjoint": (
            "DCASE public test uses source event examples separate from train/development."
        ),
        "labels": LABEL_MAP,
        "audio_committed": False,
    }
    arguments.provenance.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(training)} training and {len(inspection)} inspection rows")


if __name__ == "__main__":
    main()
