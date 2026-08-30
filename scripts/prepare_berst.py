#!/usr/bin/env python3
"""Verify and prepare the CC BY 4.0 BERSt speech corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import wave
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from attune.integrity import file_digest
from attune.schema.output import AffectCategory
from attune.training.prepare import SourceRow

DATASET_ID = "berst_v1"
HF_REPOSITORY = "Rosie-Lab/BERSt"
HF_REVISION = "fed35c477427cff44206464b7f85e68d1fc99062"
LICENCE = "CC-BY-4.0"
ATTRIBUTION = (
    "Tuttösí P et al. (2026), BERSting at the Screams: A Benchmark for "
    "Distanced, Emotional and Shouted Speech Recognition, Computer Speech & Language 95."
)
README_SHA256 = "111f91fc9736aa0e07477de2746f432cca3822351321dc752259156743c8e600"
SHARDS = {
    "data/test-00000-of-00001.parquet": {
        "sha256": "f68cdfde5e5003f1c457f0a2d651300647b5c6031dd9487b4124ae200a86a976",
        "size": 116_203_074,
        "rows": 532,
        "source_split": "test",
        "split": "sealed_test",
        "speakers": 10,
    },
    "data/train-00000-of-00002.parquet": {
        "sha256": "ae0a26f8cb9ee4b4653f3d15240ec50d067fe653e43334894284a16fb154a206",
        "size": 410_896_159,
        "rows": 1_752,
        "source_split": "train",
        "split": "train",
        "speakers": 78,
    },
    "data/train-00001-of-00002.parquet": {
        "sha256": "b67800f5dfb54c067184d8dc8567c22c99c0dc4c13ecec21f39892ba2c8eac3a",
        "size": 416_283_048,
        "rows": 1_751,
        "source_split": "train",
        "split": "train",
        "speakers": 78,
    },
    "data/validation-00000-of-00001.parquet": {
        "sha256": "733e61551eaffc648bfb86c0d46d129bc0676548534496052efa6072303179d2",
        "size": 112_213_896,
        "rows": 488,
        "source_split": "validation",
        "split": "development",
        "speakers": 10,
    },
}
EXPECTED_SPLIT_ROWS = {"train": 3_503, "development": 488, "sealed_test": 532}
EXPECTED_SPLIT_SPEAKERS = {"train": 78, "development": 10, "sealed_test": 10}
EXPECTED_AFFECTS = {"anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"}
EXPECTED_SHOUT_LEVELS = {"shout", "no-shout", "n/a"}
AFFECT_MAP = {
    "anger": "anger",
    "disgust": "other",
    "fear": "fear",
    "joy": "joy",
    "neutral": "neutral",
    "sadness": "distress",
    "surprise": "surprise",
}
METADATA_COLUMNS = (
    "user_id",
    "age",
    "current_language",
    "first_language",
    "gender",
    "phone_model",
    "audio_id",
    "affect",
    "last_modified",
    "phone_position",
    "script",
    "shout_level",
)


class BerstPreparationError(RuntimeError):
    """Raised when the pinned BERSt source violates the preparation contract."""


def _pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise BerstPreparationError("install the dataset-tools extra to prepare BERSt") from error
    return parquet


def verify_download(cache_dir: Path) -> list[Path]:
    readme = cache_dir / "README.md"
    if not readme.is_file() or file_digest(readme) != README_SHA256:
        raise BerstPreparationError("BERSt README is missing or does not match the pinned revision")
    paths = []
    for relative_path, expected in SHARDS.items():
        path = cache_dir / relative_path
        if not path.is_file() or path.stat().st_size != expected["size"]:
            raise BerstPreparationError(f"missing or incomplete BERSt shard: {path}")
        if file_digest(path) != expected["sha256"]:
            raise BerstPreparationError(f"BERSt shard checksum mismatch: {path}")
        paths.append(path)
    return paths


def _metadata_rows(cache_dir: Path) -> list[dict[str, Any]]:
    parquet = _pyarrow_parquet()
    rows: list[dict[str, Any]] = []
    for relative_path, expected in SHARDS.items():
        path = cache_dir / relative_path
        table = parquet.read_table(path, columns=list(METADATA_COLUMNS))
        if table.num_rows != expected["rows"]:
            raise BerstPreparationError(f"unexpected row count in {path}")
        for source_row_index, row in enumerate(table.to_pylist()):
            rows.append(
                {
                    **row,
                    "source_shard": relative_path,
                    "source_row_index": source_row_index,
                    "source_split": expected["source_split"],
                    "split": expected["split"],
                }
            )
    return rows


def validate_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != sum(EXPECTED_SPLIT_ROWS.values()):
        raise BerstPreparationError(f"expected 4,523 rows, found {len(rows)}")
    identities = [(row["source_shard"], row["source_row_index"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise BerstPreparationError("duplicate Parquet row identity")
    affect_labels = {str(row["affect"]) for row in rows}
    shout_levels = {str(row["shout_level"]) for row in rows}
    if affect_labels != EXPECTED_AFFECTS:
        raise BerstPreparationError(f"unexpected BERSt affect labels: {sorted(affect_labels)}")
    if shout_levels != EXPECTED_SHOUT_LEVELS:
        raise BerstPreparationError(f"unexpected BERSt shout labels: {sorted(shout_levels)}")

    split_rows = Counter(str(row["split"]) for row in rows)
    if dict(split_rows) != EXPECTED_SPLIT_ROWS:
        raise BerstPreparationError(f"unexpected BERSt split counts: {dict(split_rows)}")
    split_speakers: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_speakers[str(row["split"])].add(str(row["user_id"]))
    speaker_counts = {split: len(speakers) for split, speakers in split_speakers.items()}
    if speaker_counts != EXPECTED_SPLIT_SPEAKERS:
        raise BerstPreparationError(f"unexpected BERSt speaker counts: {speaker_counts}")
    splits = sorted(split_speakers)
    overlaps = {
        f"{left}/{right}": len(split_speakers[left] & split_speakers[right])
        for index, left in enumerate(splits)
        for right in splits[index + 1 :]
    }
    if any(overlaps.values()):
        raise BerstPreparationError(f"BERSt speakers cross partitions: {overlaps}")

    pair_splits: dict[str, str] = {}
    for row in rows:
        audio_id = str(row["audio_id"])
        previous = pair_splits.setdefault(audio_id, str(row["split"]))
        if previous != row["split"]:
            raise BerstPreparationError(f"BERSt intensity pair crosses partitions: {audio_id}")
    return {
        "rows": len(rows),
        "split_rows": dict(split_rows),
        "split_speakers": speaker_counts,
        "speaker_overlaps": overlaps,
        "affect_counts": dict(sorted(Counter(str(row["affect"]) for row in rows).items())),
        "shout_level_counts": dict(
            sorted(Counter(str(row["shout_level"]) for row in rows).items())
        ),
        "intensity_pair_count": len(pair_splits),
    }


def affect_distribution(source_affect: str) -> dict[str, float]:
    target = AFFECT_MAP[source_affect]
    return {category.value: float(category.value == target) for category in AffectCategory}


def style_target(shout_level: str) -> tuple[list[str] | None, list[str]]:
    if shout_level == "shout":
        return ["shouting"], []
    if shout_level == "no-shout":
        return [], ["styles"]
    if shout_level == "n/a":
        return None, []
    raise BerstPreparationError(f"unsupported BERSt shout level: {shout_level}")


def clip_id(audio_id: str, embedded_path: str) -> str:
    source_stem = Path(audio_id).stem
    chunk = PurePosixPath(embedded_path)
    if chunk.is_absolute() or ".." in chunk.parts or len(chunk.parts) != 1:
        raise BerstPreparationError(f"unsafe embedded audio path: {embedded_path!r}")
    raw = f"berst-{source_stem}-{chunk.stem}"
    return "".join(
        character if character.isalnum() or character in "-_" else "_" for character in raw
    )


def _wav_metadata(content: bytes | Path) -> tuple[int, int, int, int]:
    source: BytesIO | str = BytesIO(content) if isinstance(content, bytes) else str(content)
    try:
        with wave.open(source, "rb") as audio:
            return (
                round(audio.getnframes() / audio.getframerate() * 1_000),
                audio.getframerate(),
                audio.getnchannels(),
                audio.getsampwidth() * 8,
            )
    except (OSError, wave.Error) as error:
        raise BerstPreparationError(f"invalid BERSt WAV: {error}") from error


def normalize_audio(content: bytes, target: Path) -> dict[str, int]:
    source_duration, source_rate, source_channels, source_bits = _wav_metadata(content)
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.part.wav")
        process = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                "pipe:0",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                "-fflags",
                "+bitexact",
                "-flags:a",
                "+bitexact",
                str(temporary),
            ],
            input=content,
            capture_output=True,
        )
        if process.returncode:
            temporary.unlink(missing_ok=True)
            message = process.stderr.decode(errors="replace").strip()
            raise BerstPreparationError(f"FFmpeg failed for {target.name}: {message}")
        temporary.replace(target)
    duration, rate, channels, bits = _wav_metadata(target)
    if rate != 16_000 or channels != 1 or bits != 16 or abs(duration - source_duration) > 20:
        raise BerstPreparationError(f"invalid normalized BERSt audio: {target}")
    return {
        "source_duration_ms": source_duration,
        "source_sample_rate_hz": source_rate,
        "source_channels": source_channels,
        "source_sample_width_bits": source_bits,
        "duration_ms": duration,
        "sample_rate_hz": rate,
        "channels": channels,
        "sample_width_bits": bits,
    }


def prepare_rows(
    cache_dir: Path,
    metadata_rows: list[dict[str, Any]],
    *,
    source_manifest: Path,
) -> tuple[list[dict[str, Any]], list[SourceRow]]:
    parquet = _pyarrow_parquet()
    audio_ids = sorted({str(row["audio_id"]) for row in metadata_rows})
    pair_ids = {audio_id: index for index, audio_id in enumerate(audio_ids)}
    metadata_by_identity = {
        (str(row["source_shard"]), int(row["source_row_index"])): row for row in metadata_rows
    }
    provenance_rows: list[dict[str, Any]] = []
    source_rows: list[SourceRow] = []
    seen_clip_ids: set[str] = set()
    processed = 0
    for relative_path in SHARDS:
        parquet_file = parquet.ParquetFile(cache_dir / relative_path)
        source_row_index = 0
        for batch in parquet_file.iter_batches(batch_size=32):
            for raw_row in batch.to_pylist():
                metadata = metadata_by_identity[(relative_path, source_row_index)]
                source_row_index += 1
                audio = raw_row.get("audio")
                if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
                    raise BerstPreparationError("BERSt row is missing embedded audio bytes")
                identifier = clip_id(str(metadata["audio_id"]), str(audio.get("path")))
                if identifier in seen_clip_ids:
                    raise BerstPreparationError(f"duplicate BERSt clip ID: {identifier}")
                seen_clip_ids.add(identifier)
                target = cache_dir / "pcm16_16k" / str(metadata["split"]) / f"{identifier}.wav"
                audio_metadata = normalize_audio(audio["bytes"], target)
                audio_sha256 = file_digest(target)
                styles, auxiliary_negative_tasks = style_target(str(metadata["shout_level"]))
                source_rows.append(
                    SourceRow(
                        clip_id=identifier,
                        dataset_id=DATASET_ID,
                        split=str(metadata["split"]),
                        speaker_id=f"berst:{metadata['user_id']}",
                        audio_path=Path(
                            os.path.relpath(target.resolve(), source_manifest.parent.resolve())
                        ),
                        audio_sha256=audio_sha256,
                        duration_ms=audio_metadata["duration_ms"],
                        transcript=str(metadata["script"]).strip(),
                        asr_supervised=True,
                        events=None,
                        event_presence=None,
                        styles=styles,
                        affect_distribution=affect_distribution(str(metadata["affect"])),
                        vad=None,
                        pair_id=pair_ids[str(metadata["audio_id"])],
                        is_ood=False,
                        lexical_affect_label="neutral",
                        split_unit="speaker",
                        auxiliary_negative_tasks=auxiliary_negative_tasks,
                    )
                )
                provenance_rows.append(
                    {
                        **metadata,
                        "clip_id": identifier,
                        "dataset_id": DATASET_ID,
                        "hf_repository": HF_REPOSITORY,
                        "hf_revision": HF_REVISION,
                        "embedded_audio_path": str(audio["path"]),
                        "source_audio_sha256": hashlib.sha256(audio["bytes"]).hexdigest(),
                        "audio_sha256": audio_sha256,
                        **audio_metadata,
                        "pair_id": pair_ids[str(metadata["audio_id"])],
                        "target_affect": AFFECT_MAP[str(metadata["affect"])],
                        "target_styles": styles,
                        "licence": LICENCE,
                        "attribution": ATTRIBUTION,
                        "release_weight_training_permitted": True,
                        "affect_label_status": (
                            "actor prompt mapped to the Attune ontology; not listener-validated"
                        ),
                        "style_label_status": (
                            "source vocal-intensity label mapped only when shout/no-shout is known"
                        ),
                    }
                )
                processed += 1
                if processed % 100 == 0:
                    print(f"Prepared {processed}/{len(metadata_rows)} BERSt clips", flush=True)
    if processed != len(metadata_rows):
        raise BerstPreparationError(f"prepared {processed} rows, expected {len(metadata_rows)}")
    provenance_rows.sort(key=lambda row: str(row["clip_id"]))
    source_rows.sort(key=lambda row: row.clip_id)
    return provenance_rows, source_rows


def write_manifests(
    provenance_rows: list[dict[str, Any]],
    source_rows: list[SourceRow],
    *,
    provenance_manifest: Path,
    source_manifest: Path,
) -> None:
    provenance_manifest.parent.mkdir(parents=True, exist_ok=True)
    provenance_manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in provenance_rows
        ),
        encoding="utf-8",
    )
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    source_manifest.write_text(
        "".join(row.model_dump_json() + "\n" for row in source_rows), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/berst"))
    parser.add_argument(
        "--provenance-manifest",
        type=Path,
        default=Path("data/manifests/berst-v0.1.jsonl"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("artifacts/manifests/berst-v0.1-source.jsonl"),
    )
    parser.add_argument(
        "--inspection-output",
        type=Path,
        default=Path("artifacts/dataset-inspection/berst-v0.1.json"),
    )
    parser.add_argument("--metadata-only", action="store_true")
    arguments = parser.parse_args()

    verify_download(arguments.cache_dir)
    metadata_rows = _metadata_rows(arguments.cache_dir)
    report = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "hf_repository": HF_REPOSITORY,
        "hf_revision": HF_REVISION,
        "licence": LICENCE,
        **validate_metadata(metadata_rows),
    }
    arguments.inspection_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.inspection_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if arguments.metadata_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    provenance_rows, source_rows = prepare_rows(
        arguments.cache_dir,
        metadata_rows,
        source_manifest=arguments.source_manifest,
    )
    write_manifests(
        provenance_rows,
        source_rows,
        provenance_manifest=arguments.provenance_manifest,
        source_manifest=arguments.source_manifest,
    )
    print(
        f"Verified and prepared {len(source_rows)} CC BY clips from "
        f"{report['split_speakers']} speakers",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BerstPreparationError as error:
        raise SystemExit(f"error: {error}") from error
