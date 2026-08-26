from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_dataset import PreparationError, load_manifest, prepare


def manifest_row(
    *,
    clip_id: str,
    speaker_id: str,
    split: str,
    cache_path: str,
    sha256: str,
) -> dict:
    return {
        "clip_id": clip_id,
        "source_dataset": "CREMA-D",
        "source_filename": f"{clip_id}.wav",
        "speaker_id": speaker_id,
        "split": split,
        "duration_s": 1.0,
        "sha256": sha256,
        "licence": {"identifier": "ODbL-1.0"},
        "attribution": "CREMA-D test attribution",
        "intended_attune_labels": {
            "events": [],
            "styles": [],
            "affect": ["neutral"],
            "label_status": "weak_source_label",
        },
        "acted_status": "acted",
        "notes": "test row",
        "cache_path": cache_path,
        "fetch": {"type": "url", "url": "https://example.invalid/audio.wav"},
    }


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")


def test_load_manifest_rejects_speaker_crossing_splits(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        manifest_row(
            clip_id="one",
            speaker_id="crema-d:1001",
            split="inspect",
            cache_path="crema_d/one.wav",
            sha256="0" * 64,
        ),
        manifest_row(
            clip_id="two",
            speaker_id="crema-d:1001",
            split="held_out_speakers",
            cache_path="crema_d/two.wav",
            sha256="1" * 64,
        ),
    ]
    write_manifest(manifest, rows)

    with pytest.raises(PreparationError, match="crosses splits"):
        load_manifest(manifest)


def test_prepare_verifies_existing_local_file(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    audio = cache / "crema_d/example.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"local inspection fixture")
    expected_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        [
            manifest_row(
                clip_id="example",
                speaker_id="crema-d:1001",
                split="inspect",
                cache_path="crema_d/example.wav",
                sha256=expected_hash,
            )
        ],
    )

    assert prepare(manifest, cache, allow_downloads=False) == 1


def test_prepare_reports_missing_files_without_downloading(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        [
            manifest_row(
                clip_id="missing",
                speaker_id="crema-d:1001",
                split="inspect",
                cache_path="crema_d/missing.wav",
                sha256="0" * 64,
            )
        ],
    )

    with pytest.raises(PreparationError, match="Rerun with --download"):
        prepare(manifest, tmp_path / "cache", allow_downloads=False)


def test_committed_inspection_manifest_invariants() -> None:
    repository = Path(__file__).parents[2]
    rows = load_manifest(repository / "data/manifests/inspection-set.jsonl")

    assert 120 <= len(rows) <= 180
    assert {row["source_dataset"] for row in rows} == {"VocalSound", "CREMA-D"}
    assert {row["split"] for row in rows} == {"inspect", "held_out_speakers"}
    assert all(len(row["sha256"]) == 64 for row in rows)

    inspect_speakers = {
        row["speaker_id"] for row in rows if row["split"] == "inspect"
    }
    held_out_speakers = {
        row["speaker_id"] for row in rows if row["split"] == "held_out_speakers"
    }
    assert inspect_speakers.isdisjoint(held_out_speakers)

    events = {
        event for row in rows for event in row["intended_attune_labels"]["events"]
    }
    assert events == {"laugh", "sigh", "cough", "throat_clear", "sneeze"}
