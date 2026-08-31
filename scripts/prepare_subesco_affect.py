#!/usr/bin/env python3
"""Verify, normalize, and manifest the CC BY 4.0 SUBESCO corpus."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import wave
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from attune.integrity import file_digest as digest
from attune.schema.output import AffectCategory
from attune.training.prepare import SourceRow

RECORD_ID = "4526477"
DOI = "10.5281/zenodo.4526477"
ARCHIVE_NAME = "SUBESCO.zip"
ARCHIVE_MD5 = "13d702cf944bf54d70d66bffc036ed7a"
ARCHIVE_SHA256 = "7505111bc864193d1906c05f83ca50d950250c65b62ae9b1740dee44f5cdb113"
ARCHIVE_SIZE = 1_664_512_929
DATASET_ID = "subesco_v1_1"
LICENCE = "CC-BY-4.0"
ATTRIBUTION = (
    "Sultana S, Rahman MS, Selim MR, Iqbal MZ (2021), SUST Bangla Emotional "
    "Speech Corpus (SUBESCO), doi:10.5281/zenodo.4526477."
)
MALFORMED_MEMBER = "SUBESCO/F_02_MONIKA_S_2_SURPRISE_3].wav"
FILENAME = re.compile(
    r"^SUBESCO/([FM])_(\d{2})_([A-Z]+)_S_(10|[1-9])_"
    r"(ANGRY|DISGUST|FEAR|HAPPY|NEUTRAL|SAD|SURPRISE)_([1-5])\.wav$"
)
AFFECT_MAP = {
    "ANGRY": "anger",
    "DISGUST": "other",
    "FEAR": "fear",
    "HAPPY": "joy",
    "NEUTRAL": "neutral",
    "SAD": "distress",
    "SURPRISE": "surprise",
}


class SubescoPreparationError(RuntimeError):
    """Raised when a pinned source or manifest invariant is violated."""


def parse_member(name: str) -> dict[str, Any]:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 2:
        raise SubescoPreparationError(f"unsafe archive member: {name!r}")
    canonical = name
    source_name_anomaly = False
    if name == MALFORMED_MEMBER:
        canonical = name.replace("3].wav", "3.wav")
        source_name_anomaly = True
    match = FILENAME.fullmatch(canonical)
    if match is None:
        raise SubescoPreparationError(f"unexpected archive member: {name!r}")
    gender, speaker_number, speaker_name, sentence, emotion, take = match.groups()
    speaker_id = f"{gender}_{speaker_number}_{speaker_name}"
    return {
        "archive_member": name,
        "canonical_filename": PurePosixPath(canonical).name,
        "speaker_id": speaker_id,
        "gender_presentation": gender,
        "speaker_number": int(speaker_number),
        "speaker_name": speaker_name,
        "sentence_id": int(sentence),
        "source_emotion": emotion,
        "take": int(take),
        "source_name_anomaly": source_name_anomaly,
    }


def speaker_partitions(speakers: list[str]) -> dict[str, str]:
    """Return deterministic gender-stratified 14/2/4 actor partitions."""
    if len(speakers) != 20 or len(set(speakers)) != 20:
        raise SubescoPreparationError("speaker split requires exactly 20 unique actors")
    result: dict[str, str] = {}
    for gender in ("F", "M"):
        members = [speaker for speaker in speakers if speaker.startswith(f"{gender}_")]
        if len(members) != 10:
            raise SubescoPreparationError(f"expected ten {gender} actors")
        ordered = sorted(members, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
        for index, speaker in enumerate(ordered):
            if index < 7:
                result[speaker] = "train"
            elif index == 7:
                result[speaker] = "development"
            else:
                result[speaker] = "sealed_test"
    return result


def protocol_rows(bundle: zipfile.ZipFile) -> list[dict[str, Any]]:
    infos = bundle.infolist()
    if len(infos) != 7_000 or sum(info.file_size for info in infos) != 2_823_631_746:
        raise SubescoPreparationError("archive membership or uncompressed-size mismatch")
    if any(info.is_dir() or info.file_size > 1_000_000 for info in infos):
        raise SubescoPreparationError("archive contains an unexpected member type or size")
    rows = [parse_member(info.filename) for info in infos]
    partitions = speaker_partitions(sorted({str(row["speaker_id"]) for row in rows}))
    seen: set[tuple[str, int, str, int]] = set()
    for row in rows:
        identity = (
            str(row["speaker_id"]),
            int(row["sentence_id"]),
            str(row["source_emotion"]),
            int(row["take"]),
        )
        if identity in seen:
            raise SubescoPreparationError(f"duplicate logical clip: {identity}")
        seen.add(identity)
        row["split"] = partitions[str(row["speaker_id"])]
        speaker_number = sorted(partitions).index(str(row["speaker_id"]))
        row["pair_id"] = speaker_number * 10 + int(row["sentence_id"]) - 1
        row["clip_id"] = f"subesco-{Path(str(row['canonical_filename'])).stem.lower()}"
    return sorted(rows, key=lambda row: str(row["clip_id"]))


def _wav_metadata(content: bytes | Path) -> tuple[int, int, int, int]:
    source = io.BytesIO(content) if isinstance(content, bytes) else str(content)
    try:
        with wave.open(source, "rb") as audio:
            return (
                round(audio.getnframes() / audio.getframerate() * 1000),
                audio.getframerate(),
                audio.getnchannels(),
                audio.getsampwidth() * 8,
            )
    except (OSError, wave.Error) as error:
        raise SubescoPreparationError(f"invalid WAV: {error}") from error


def normalize_rows(bundle: zipfile.ZipFile, cache_root: Path, rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows, 1):
        content = bundle.read(str(row["archive_member"]))
        source_duration, source_rate, source_channels, source_bits = _wav_metadata(content)
        target = cache_root / "pcm16_16k" / str(row["canonical_filename"])
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
                raise SubescoPreparationError(f"FFmpeg failed for {target.name}: {message}")
            temporary.replace(target)
        duration, rate, channels, bits = _wav_metadata(target)
        if rate != 16_000 or channels != 1 or bits != 16 or abs(duration - source_duration) > 20:
            raise SubescoPreparationError(f"{target}: invalid normalized audio contract")
        row.update(
            {
                "source_audio_sha256": hashlib.sha256(content).hexdigest(),
                "source_duration_ms": source_duration,
                "source_sample_rate_hz": source_rate,
                "source_channels": source_channels,
                "source_sample_width_bits": source_bits,
                "audio_path": target,
                "audio_sha256": digest(target),
                "duration_ms": duration,
            }
        )
        if index % 200 == 0:
            print(f"Normalized {index}/{len(rows)} SUBESCO clips", flush=True)


def _distribution(source_emotion: str) -> dict[str, float]:
    target = AFFECT_MAP[source_emotion]
    return {category.value: float(category.value == target) for category in AffectCategory}


def build_manifests(
    rows: list[dict[str, Any]],
    *,
    source_manifest: Path,
    provenance_manifest: Path,
) -> tuple[list[dict[str, Any]], list[SourceRow]]:
    provenance_rows: list[dict[str, Any]] = []
    source_rows: list[SourceRow] = []
    for row in rows:
        audio_path = Path(row["audio_path"])
        provenance_rows.append(
            {
                **{key: value for key, value in row.items() if key != "audio_path"},
                "dataset_id": DATASET_ID,
                "record_id": RECORD_ID,
                "doi": DOI,
                "archive_name": ARCHIVE_NAME,
                "archive_md5": ARCHIVE_MD5,
                "archive_sha256": ARCHIVE_SHA256,
                "target_affect": AFFECT_MAP[str(row["source_emotion"])],
                "sample_rate_hz": 16_000,
                "channels": 1,
                "sample_width_bits": 16,
                "licence": LICENCE,
                "attribution": ATTRIBUTION,
                "release_weight_training_permitted": True,
                "asr_supervised": False,
                "label_status": "source acted label mapped to Attune affect ontology",
            }
        )
        source_rows.append(
            SourceRow(
                clip_id=str(row["clip_id"]),
                dataset_id=DATASET_ID,
                split=str(row["split"]),
                speaker_id=f"subesco:{row['speaker_id']}",
                audio_path=Path(
                    os.path.relpath(audio_path.resolve(), source_manifest.parent.resolve())
                ),
                audio_sha256=str(row["audio_sha256"]),
                duration_ms=int(row["duration_ms"]),
                transcript=None,
                asr_supervised=False,
                events=None,
                event_presence=None,
                styles=None,
                affect_distribution=_distribution(str(row["source_emotion"])),
                vad=None,
                pair_id=int(row["pair_id"]),
                is_ood=True,
                lexical_affect_label=None,
                auxiliary_negative_tasks=[],
            )
        )
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
    return provenance_rows, source_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/subesco"))
    parser.add_argument(
        "--provenance-manifest",
        type=Path,
        default=Path("data/manifests/subesco-affect-v0.1.jsonl"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("artifacts/manifests/subesco-affect-v0.1-source.jsonl"),
    )
    arguments = parser.parse_args()
    archive = arguments.cache_dir / "_archives" / ARCHIVE_NAME
    if not archive.is_file() or archive.stat().st_size != ARCHIVE_SIZE:
        raise SystemExit(f"error: missing or incomplete official archive: {archive}")
    if digest(archive, "md5") != ARCHIVE_MD5 or digest(archive) != ARCHIVE_SHA256:
        raise SystemExit(f"error: official archive checksum mismatch: {archive}")
    with zipfile.ZipFile(archive) as bundle:
        rows = protocol_rows(bundle)
        normalize_rows(bundle, arguments.cache_dir, rows)
    provenance, source = build_manifests(
        rows,
        source_manifest=arguments.source_manifest,
        provenance_manifest=arguments.provenance_manifest,
    )
    speakers = {row["speaker_id"] for row in provenance}
    print(f"Verified {len(source)} CC BY clips from {len(speakers)} actors", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SubescoPreparationError as error:
        raise SystemExit(f"error: {error}") from error
