#!/usr/bin/env python3
"""Prepare the pinned Apache-2.0 DisfluencySpeech corpus for weak event training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import wave
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

from attune.integrity import file_digest
from attune.training.prepare import SourceRow

DATASET_ID = "disfluency_speech_v0.1"
HF_REPOSITORY = "amaai-lab/DisfluencySpeech"
HF_REVISION = "b7da294fe3a70dd96df6640893f2a5dfc2c87638"
LICENCE = "Apache-2.0"
ATTRIBUTION = (
    "Kyra Wang and Dorien Herremans (2024), DisfluencySpeech -- Single-Speaker "
    "Conversational Speech Dataset with Paralanguage, arXiv:2406.08820."
)
README_SHA256 = "1b6c1503faf82a18afe330276f5e22033a31d2e02b3168241bd5a9a1f6fb3313"
SHARDS = {
    "data/train-00000-of-00003.parquet": {
        "sha256": "6285da585435e76adefe43c3142852fd92a6be0c8c820486d941421a8638b4ae",
        "size": 438_310_456,
        "rows": 1_500,
        "split": "train",
    },
    "data/train-00001-of-00003.parquet": {
        "sha256": "bfb907061b59e14fe9f9c2f6e9a00e13fe1cf531e50e6987c7cd34aecad81887",
        "size": 448_788_413,
        "rows": 1_500,
        "split": "train",
    },
    "data/train-00002-of-00003.parquet": {
        "sha256": "16bc5a61a9ff54923b78ca86d17c7028e5f36d9b3655637f3ffd64e73cf37426",
        "size": 450_679_315,
        "rows": 1_500,
        "split": "train",
    },
    "data/validation-00000-of-00001.parquet": {
        "sha256": "693ba2a06301d9cbafea52eae0664f8c776467e9b559ad9e25b6756f236f6a9c",
        "size": 74_193_762,
        "rows": 250,
        "split": "development",
    },
    "data/test-00000-of-00001.parquet": {
        "sha256": "e78837be393206fc0ae6a0cefb4ec8c942c3e26401640eb8af67cf0dac21b4b0",
        "size": 70_868_626,
        "rows": 250,
        "split": "sealed_test",
    },
}
EVENT_TAG = re.compile(r"<\s*([^>]+?)\s*>")
EVENT_MAP = {
    "laughter": "laugh",
    "laugh": "laugh",
    "throat_clearing": "throat_clear",
    "throatclear": "throat_clear",
    "sigh": "sigh",
    "cough": "cough",
    "sneeze": "sneeze",
    "scream": "scream",
    "cry": "sob",
    "sobbing": "sob",
}


class DisfluencySpeechPreparationError(RuntimeError):
    """Raised when the pinned source violates its preparation contract."""


def _parquet() -> Any:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise DisfluencySpeechPreparationError(
            "install the dataset-tools extra to prepare DisfluencySpeech"
        ) from error
    return parquet


def event_presence(transcript: str) -> list[str]:
    labels = []
    for raw_label in EVENT_TAG.findall(transcript):
        normalized = re.sub(r"[\s-]+", "_", raw_label.strip().lower())
        label = EVENT_MAP.get(normalized)
        if label is not None and label not in labels:
            labels.append(label)
    return labels


def _audio_metadata(audio_bytes: bytes) -> tuple[int, int, int, int]:
    try:
        with wave.open(BytesIO(audio_bytes), "rb") as audio:
            sample_rate = audio.getframerate()
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            duration_ms = round(audio.getnframes() / sample_rate * 1000)
    except (EOFError, wave.Error) as error:
        raise DisfluencySpeechPreparationError(f"invalid source WAV: {error}") from error
    if channels != 1 or sample_width != 2 or sample_rate != 22_050:
        raise DisfluencySpeechPreparationError(
            "DisfluencySpeech audio must be 22.05 kHz mono PCM16"
        )
    return duration_ms, sample_rate, channels, sample_width


def verify_download(dataset_dir: Path) -> list[Path]:
    readme = dataset_dir / "README.md"
    if not readme.is_file() or file_digest(readme) != README_SHA256:
        raise DisfluencySpeechPreparationError(
            "DisfluencySpeech README is missing or does not match the pinned revision"
        )
    paths = []
    for relative_path, expected in SHARDS.items():
        path = dataset_dir / relative_path
        if not path.is_file() or path.stat().st_size != expected["size"]:
            raise DisfluencySpeechPreparationError(f"missing or incomplete shard: {path}")
        if file_digest(path) != expected["sha256"]:
            raise DisfluencySpeechPreparationError(f"shard checksum mismatch: {path}")
        paths.append(path)
    return paths


def _source_rows(
    dataset_dir: Path,
    audio_dir: Path,
    manifest_path: Path,
) -> tuple[list[SourceRow], dict[str, Any]]:
    parquet = _parquet()
    rows: list[SourceRow] = []
    audio_hashes: set[str] = set()
    duplicate_audio = 0
    event_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    source_index = 0
    for relative_path, expected in SHARDS.items():
        shard = parquet.ParquetFile(dataset_dir / relative_path)
        row_index = 0
        for batch in shard.iter_batches(batch_size=32):
            for raw_row in batch.to_pylist():
                audio = raw_row.get("audio")
                audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
                if not isinstance(audio_bytes, bytes):
                    raise DisfluencySpeechPreparationError("source row is missing embedded audio")
                audio_sha256 = hashlib.sha256(audio_bytes).hexdigest()
                if audio_sha256 in audio_hashes:
                    duplicate_audio += 1
                    row_index += 1
                    source_index += 1
                    continue
                audio_hashes.add(audio_sha256)
                annotated = raw_row.get("transcript_annotated")
                transcript = raw_row.get("transcript_a")
                if not isinstance(annotated, str) or not isinstance(transcript, str):
                    raise DisfluencySpeechPreparationError("source row is missing transcripts")
                labels = event_presence(annotated)
                event_counts.update(labels)
                split = str(expected["split"])
                split_counts[split] += 1
                duration_ms, _rate, _channels, _width = _audio_metadata(audio_bytes)
                shard_id = Path(relative_path).stem.removesuffix(".parquet")
                clip_id = f"disfluency-{shard_id}-{row_index:04d}"
                audio_path = audio_dir / split / f"{clip_id}.wav"
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                if audio_path.is_file() and file_digest(audio_path) != audio_sha256:
                    raise DisfluencySpeechPreparationError(
                        f"existing extracted audio differs: {audio_path}"
                    )
                if not audio_path.is_file():
                    audio_path.write_bytes(audio_bytes)
                rows.append(
                    SourceRow(
                        clip_id=clip_id,
                        dataset_id=DATASET_ID,
                        split=split,
                        speaker_id=None,
                        audio_path=Path(
                            os.path.relpath(audio_path.resolve(), manifest_path.parent.resolve())
                        ),
                        audio_sha256=audio_sha256,
                        duration_ms=duration_ms,
                        transcript=" ".join(transcript.split()),
                        asr_supervised=False,
                        events=None,
                        event_presence=labels,
                        styles=None,
                        affect_distribution=None,
                        vad=None,
                        pair_id=source_index,
                        is_ood=True,
                        lexical_affect_label=None,
                        split_unit="sentence",
                    )
                )
                row_index += 1
                source_index += 1
        if row_index != expected["rows"]:
            raise DisfluencySpeechPreparationError(
                f"expected {expected['rows']} rows in {relative_path}, found {row_index}"
            )
    return rows, {
        "source_rows": sum(int(value["rows"]) for value in SHARDS.values()),
        "retained_unique_audio": len(rows),
        "duplicate_audio_rows_excluded": duplicate_audio,
        "split_counts": dict(sorted(split_counts.items())),
        "event_counts": dict(sorted(event_counts.items())),
    }


def prepare(
    dataset_dir: Path,
    audio_dir: Path,
    manifest_path: Path,
    provenance_path: Path,
) -> dict[str, Any]:
    verify_download(dataset_dir)
    rows, audit = _source_rows(dataset_dir, audio_dir, manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("".join(row.model_dump_json() + "\n" for row in rows))
    provenance = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "source_repository": HF_REPOSITORY,
        "source_revision": HF_REVISION,
        "licence": LICENCE,
        "attribution": ATTRIBUTION,
        "source_shards": {
            path: {
                "sha256": values["sha256"],
                "size": values["size"],
                "rows": values["rows"],
                "split": values["split"],
            }
            for path, values in SHARDS.items()
        },
        "split_policy": (
            "The official clip partitions are retained. The corpus has one speaker, so it is "
            "declared sentence-disjoint and is never used as the final external evaluation."
        ),
        "supervision": (
            "Human-recorded utterances with word-position event tags are converted to weak "
            "utterance-presence targets; no timestamp is invented."
        ),
        "manifest": str(manifest_path),
        "manifest_sha256": file_digest(manifest_path),
        **audit,
        "audio_committed": False,
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/raw/disfluency-speech"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/raw/disfluency-speech-audio"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/disfluency-speech-v0.1.jsonl"),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("data/manifests/disfluency-speech-v0.1.provenance.json"),
    )
    arguments = parser.parse_args()
    report = prepare(
        arguments.dataset_dir,
        arguments.audio_dir,
        arguments.manifest,
        arguments.provenance,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
