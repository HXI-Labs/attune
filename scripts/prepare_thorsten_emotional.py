#!/usr/bin/env python3
"""Fetch, verify, normalize, and manifest the CC0 Thorsten emotional corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path, PurePosixPath
from typing import Any

from attune.schema.output import AffectCategory
from attune.training.prepare import SourceRow

RECORD_ID = "5525023"
ARCHIVE_NAME = "thorsten-emotional_v02.tgz"
ARCHIVE_URL = f"https://zenodo.org/records/{RECORD_ID}/files/{ARCHIVE_NAME}?download=1"
ARCHIVE_MD5 = "f19585f3759c6d026757c46024a37aa4"
ARCHIVE_SHA256 = "ff17a96a8c11742a7798e7800386159ada2ceb682ebef6402ecc68abf8c6e9bf"
DATASET_ID = "thorsten_voice_2021_06_emotional"
SOURCE_VERSION = "2.0"
ROOT_NAME = "thorsten-emotional_v02"
METADATA_NAME = "thorsten-emotional-metadata.csv"
STYLES = (
    "amused",
    "angry",
    "disgusted",
    "drunk",
    "neutral",
    "sleepy",
    "surprised",
    "whisper",
)
MISSING_WHISPER_ID = "2cc2cc4a34b961ef1657cc82dbd18875"
AFFECT_MAP = {
    "amused": "joy",
    "angry": "anger",
    "disgusted": "other",
    "neutral": "neutral",
    "surprised": "surprise",
}


class ThorstenPreparationError(RuntimeError):
    """Raised when the CC0 source archive or extraction contract is invalid."""


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download_with_retry(url: str, target: Path, *, retries: int = 4) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": "attune-thorsten/1"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response, temporary.open(
                "wb"
            ) as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            temporary.replace(target)
            return
        except (OSError, urllib.error.URLError) as error:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise ThorstenPreparationError(f"archive download failed: {error}") from error
            time.sleep(min(2**attempt, 16))


def _safe_member(member: tarfile.TarInfo) -> PurePosixPath:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise ThorstenPreparationError(f"unsafe archive path: {member.name!r}")
    if member.issym() or member.islnk() or member.isdev():
        raise ThorstenPreparationError(f"archive links/devices are prohibited: {member.name!r}")
    return path


def extract_archive(archive: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            _safe_member(member)
        bundle.extractall(output_dir, members=members, filter="data")


def inspect_extraction(output_dir: Path) -> dict[str, object]:
    wavs = sorted(output_dir.rglob("*.wav"))
    if not wavs:
        raise ThorstenPreparationError("verified archive extraction contains no WAV files")
    metadata = sorted(
        path
        for pattern in ("*.csv", "*.tsv", "*.txt", "*.json", "*.jsonl")
        for path in output_dir.rglob(pattern)
        if path.is_file()
    )
    first_components = sorted(
        {
            path.relative_to(output_dir).parts[0]
            for path in [*wavs, *metadata]
            if path.relative_to(output_dir).parts
        }
    )
    return {
        "wav_count": len(wavs),
        "metadata_files": [str(path.relative_to(output_dir)) for path in metadata],
        "top_level_components": first_components,
        "first_wav_paths": [str(path.relative_to(output_dir)) for path in wavs[:30]],
    }


def read_metadata(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            sentence_id, transcript = line.split("|", 1)
        except ValueError as error:
            raise ThorstenPreparationError(
                f"{path}:{line_number}: expected id|transcript"
            ) from error
        if len(sentence_id) != 32 or sentence_id in rows or not transcript.strip():
            raise ThorstenPreparationError(f"{path}:{line_number}: invalid metadata row")
        rows[sentence_id] = transcript.strip()
    if len(rows) != 300:
        raise ThorstenPreparationError(f"expected 300 metadata rows, found {len(rows)}")
    return rows


def sentence_partitions(sentence_ids: list[str]) -> dict[str, tuple[str, int]]:
    """Assign paired deliveries to deterministic 210/45/45 sentence partitions."""
    if len(sentence_ids) != 300 or len(set(sentence_ids)) != 300:
        raise ThorstenPreparationError("sentence split requires exactly 300 unique IDs")
    ordered = sorted(sentence_ids, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    result: dict[str, tuple[str, int]] = {}
    for pair_id, sentence_id in enumerate(ordered):
        split = "train" if pair_id < 210 else "development" if pair_id < 255 else "sealed_test"
        result[sentence_id] = (split, pair_id)
    return result


def discover_rows(corpus_root: Path) -> list[dict[str, Any]]:
    metadata = read_metadata(corpus_root / METADATA_NAME)
    partitions = sentence_partitions(list(metadata))
    rows: list[dict[str, Any]] = []
    for style in STYLES:
        directory = corpus_root / style
        observed = {path.stem: path for path in directory.glob("*.wav") if path.is_file()}
        expected = set(metadata)
        if style == "whisper":
            expected.remove(MISSING_WHISPER_ID)
        if set(observed) != expected:
            missing = sorted(expected - set(observed))[:5]
            extra = sorted(set(observed) - expected)[:5]
            raise ThorstenPreparationError(
                f"{style}: source membership mismatch; missing={missing}, extra={extra}"
            )
        for sentence_id, source_path in sorted(observed.items()):
            split, pair_id = partitions[sentence_id]
            rows.append(
                {
                    "clip_id": f"thorsten-{style}-{sentence_id}",
                    "sentence_id": sentence_id,
                    "pair_id": pair_id,
                    "split": split,
                    "source_style": style,
                    "transcript": metadata[sentence_id],
                    "source_path": source_path,
                }
            )
    if len(rows) != 2_399:
        raise ThorstenPreparationError(f"expected 2,399 WAVs, found {len(rows)}")
    return rows


def _wav_metadata(path: Path) -> tuple[int, int, int]:
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getsampwidth() != 2:
                raise ThorstenPreparationError(f"{path}: expected PCM16 WAV")
            return (
                round(audio.getnframes() / audio.getframerate() * 1000),
                audio.getframerate(),
                audio.getnchannels(),
            )
    except (OSError, wave.Error) as error:
        raise ThorstenPreparationError(f"{path}: invalid WAV: {error}") from error


def normalize_rows(cache_root: Path, rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows, 1):
        source = Path(row["source_path"])
        target = cache_root / "pcm16_16k" / str(row["source_style"]) / source.name
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
                    str(source),
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
                capture_output=True,
                text=True,
            )
            if process.returncode:
                temporary.unlink(missing_ok=True)
                raise ThorstenPreparationError(
                    f"FFmpeg conversion failed for {source}: {process.stderr.strip()}"
                )
            temporary.replace(target)
        duration_ms, sample_rate_hz, channels = _wav_metadata(target)
        if duration_ms < 250 or sample_rate_hz != 16_000 or channels != 1:
            raise ThorstenPreparationError(f"{target}: invalid normalized audio contract")
        row["audio_path"] = target
        row["duration_ms"] = duration_ms
        if index % 100 == 0:
            print(f"Normalized {index}/{len(rows)} Thorsten clips", flush=True)


def _affect_distribution(style: str) -> dict[str, float] | None:
    target = AFFECT_MAP.get(style)
    if target is None:
        return None
    return {
        category.value: float(category.value == target) for category in AffectCategory
    }


def build_manifests(
    rows: list[dict[str, Any]],
    *,
    source_manifest: Path,
    provenance_manifest: Path,
) -> tuple[list[dict[str, Any]], list[SourceRow]]:
    provenance_rows: list[dict[str, Any]] = []
    source_rows: list[SourceRow] = []
    for row in rows:
        source_path = Path(row["source_path"])
        audio_path = Path(row["audio_path"])
        source_duration_ms, source_sample_rate_hz, source_channels = _wav_metadata(source_path)
        audio_sha256 = digest(audio_path)
        style = str(row["source_style"])
        styles = ["whispering"] if style == "whisper" else [] if style == "neutral" else None
        auxiliary_negative_tasks: list[str] = ["event_presence"]
        if style == "neutral":
            auxiliary_negative_tasks.append("styles")
        label_status = (
            "source delivery label mapped to Attune affect ontology"
            if style in AFFECT_MAP
            else "source delivery retained without categorical affect supervision"
        )
        if style == "whisper":
            label_status = "source delivery mapped to Attune whispering style"
        provenance_rows.append(
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"source_path", "audio_path"}
                },
                "dataset_id": DATASET_ID,
                "source_version": SOURCE_VERSION,
                "record_id": RECORD_ID,
                "archive_name": ARCHIVE_NAME,
                "archive_md5": ARCHIVE_MD5,
                "archive_sha256": ARCHIVE_SHA256,
                "source_audio_path": str(source_path.relative_to(source_path.parents[2])),
                "source_audio_sha256": digest(source_path),
                "source_duration_ms": source_duration_ms,
                "source_sample_rate_hz": source_sample_rate_hz,
                "source_channels": source_channels,
                "audio_sha256": audio_sha256,
                "duration_ms": int(row["duration_ms"]),
                "sample_rate_hz": 16_000,
                "channels": 1,
                "licence": "CC0-1.0",
                "release_weight_training_permitted": True,
                "asr_supervised": False,
                "split_unit": "sentence",
                "single_speaker_limitation": True,
                "label_status": label_status,
            }
        )
        source_rows.append(
            SourceRow(
                clip_id=str(row["clip_id"]),
                dataset_id=DATASET_ID,
                split=str(row["split"]),
                speaker_id="thorsten:thorsten",
                audio_path=Path(
                    os.path.relpath(audio_path.resolve(), source_manifest.parent.resolve())
                ),
                audio_sha256=audio_sha256,
                duration_ms=int(row["duration_ms"]),
                transcript=str(row["transcript"]),
                asr_supervised=False,
                events=None,
                event_presence=[],
                styles=styles,
                affect_distribution=_affect_distribution(style),
                vad=None,
                pair_id=int(row["pair_id"]),
                is_ood=True,
                lexical_affect_label=None,
                split_unit="sentence",
                auxiliary_negative_tasks=auxiliary_negative_tasks,
            )
        )
    provenance_manifest.parent.mkdir(parents=True, exist_ok=True)
    provenance_manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in provenance_rows
        ),
        encoding="utf-8",
    )
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    source_manifest.write_text(
        "".join(row.model_dump_json() + "\n" for row in source_rows),
        encoding="utf-8",
    )
    return provenance_rows, source_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/thorsten-emotional"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--inspection-output",
        type=Path,
        default=Path("artifacts/dataset-inspection/thorsten-emotional.json"),
    )
    parser.add_argument(
        "--provenance-manifest",
        type=Path,
        default=Path("data/manifests/thorsten-emotional-v0.1.jsonl"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("artifacts/manifests/thorsten-emotional-v0.1-source.jsonl"),
    )
    arguments = parser.parse_args()
    archive = arguments.cache_dir / "_archives" / ARCHIVE_NAME
    if arguments.download and not archive.is_file():
        download_with_retry(ARCHIVE_URL, archive)
    if not archive.is_file():
        raise SystemExit(f"error: missing {archive}; rerun with --download")
    if digest(archive, "md5") != ARCHIVE_MD5:
        raise SystemExit(f"error: official MD5 mismatch for {archive}")
    if digest(archive) != ARCHIVE_SHA256:
        raise SystemExit(f"error: pinned SHA-256 mismatch for {archive}")
    extracted = arguments.cache_dir / "source"
    needs_extraction = not extracted.is_dir() or not any(extracted.iterdir())
    if needs_extraction:
        extract_archive(archive, extracted)
    report = {
        "schema_version": "1.0",
        "record_id": RECORD_ID,
        "archive": ARCHIVE_NAME,
        "archive_md5": ARCHIVE_MD5,
        "archive_sha256": digest(archive),
        "inspection": inspect_extraction(extracted),
    }
    arguments.inspection_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.inspection_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    corpus_root = extracted / ROOT_NAME
    rows = discover_rows(corpus_root)
    normalize_rows(arguments.cache_dir, rows)
    provenance, source = build_manifests(
        rows,
        source_manifest=arguments.source_manifest,
        provenance_manifest=arguments.provenance_manifest,
    )
    print(json.dumps(report["inspection"], indent=2), flush=True)
    print(
        f"Verified {len(source)} CC0 clips across "
        f"{len({row['pair_id'] for row in provenance})} sentence pairs",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except ThorstenPreparationError as error:
        raise SystemExit(f"error: {error}") from error
