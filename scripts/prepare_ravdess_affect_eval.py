#!/usr/bin/env python3
"""Prepare an evaluation-only, actor-disjoint RAVDESS affect/control slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import wave
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from attune.schema.output import AffectCategory
from attune.training.prepare import SourceRow

RECORD_ID = "1188976"
ARCHIVE_NAME = "Audio_Speech_Actors_01-24.zip"
ARCHIVE_URL = f"https://zenodo.org/records/{RECORD_ID}/files/{ARCHIVE_NAME}?download=1"
ARCHIVE_MD5 = "bc696df654c87fed845eb13823edef8a"
LICENCE = "CC-BY-NC-SA-4.0"
ATTRIBUTION = (
    "Livingstone SR, Russo FA (2018), RAVDESS v1.0.0, "
    "doi:10.5281/zenodo.1188976."
)
DEFAULT_ACTORS = tuple(range(17, 25))
STATEMENTS = {
    1: "Kids are talking by the door",
    2: "Dogs are sitting by the door",
}
EMOTIONS = {
    1: ("neutral", "neutral"),
    2: ("calm", "neutral"),
    3: ("happy", "joy"),
    4: ("sad", "distress"),
    5: ("angry", "anger"),
    6: ("fearful", "fear"),
    7: ("disgust", "other"),
    8: ("surprised", "surprise"),
}
FILENAME_PATTERN = re.compile(r"^03-01-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav$")


class RavdessPreparationError(RuntimeError):
    """Raised when RAVDESS provenance or the selected archive is invalid."""


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def protocol_rows(actors: tuple[int, ...] = DEFAULT_ACTORS) -> list[dict[str, Any]]:
    """Return the fixed 60-trial speech protocol for each selected actor."""
    invalid = not actors or len(actors) != len(set(actors)) or any(
        actor not in range(1, 25) for actor in actors
    )
    if invalid:
        raise RavdessPreparationError("actors must be unique integers in the inclusive range 1..24")
    rows: list[dict[str, Any]] = []
    for actor in actors:
        for emotion_id, (source_emotion, target_affect) in EMOTIONS.items():
            intensities = (1,) if emotion_id == 1 else (1, 2)
            for intensity in intensities:
                for statement_id, transcript in STATEMENTS.items():
                    for repetition in (1, 2):
                        filename = (
                            f"03-01-{emotion_id:02d}-{intensity:02d}-"
                            f"{statement_id:02d}-{repetition:02d}-{actor:02d}.wav"
                        )
                        rows.append(
                            {
                                "clip_id": f"ravdess-{Path(filename).stem}",
                                "filename": filename,
                                "archive_member": f"Actor_{actor:02d}/{filename}",
                                "actor_id": actor,
                                "speaker_id": f"ravdess:{actor:02d}",
                                "source_emotion": source_emotion,
                                "target_affect": target_affect,
                                "intensity": "normal" if intensity == 1 else "strong",
                                "statement_id": statement_id,
                                "repetition": repetition,
                                "transcript": transcript,
                            }
                        )
    return rows


def _download(url: str, target: Path, *, retries: int = 4) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": "attune-ravdess-eval/1"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
                "wb"
            ) as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            temporary.replace(target)
            return
        except (OSError, urllib.error.URLError) as error:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise RavdessPreparationError(f"archive download failed: {error}") from error
            time.sleep(min(2**attempt, 16))


def _validate_member(member: str) -> PurePosixPath:
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 2:
        raise RavdessPreparationError(f"unsafe or unexpected archive member: {member!r}")
    if not re.fullmatch(r"Actor_\d{2}", path.parts[0]) or not FILENAME_PATTERN.fullmatch(
        path.name
    ):
        raise RavdessPreparationError(f"unexpected RAVDESS speech member: {member!r}")
    return path


def extract_selected(archive: Path, cache_root: Path, rows: list[dict[str, Any]]) -> None:
    expected = {str(row["archive_member"]): row for row in rows}
    extracted: set[str] = set()
    with zipfile.ZipFile(archive) as bundle:
        members = set(bundle.namelist())
        missing = sorted(set(expected) - members)
        if missing:
            raise RavdessPreparationError(
                f"archive is missing {len(missing)} selected member(s), including {missing[0]}"
            )
        for index, member in enumerate(sorted(expected), 1):
            path = _validate_member(member)
            target = cache_root / "clips" / path.parts[0] / path.name
            if not target.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".wav.part")
                with bundle.open(member) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                temporary.replace(target)
            extracted.add(member)
            if index % 60 == 0:
                print(f"Prepared {index}/{len(expected)} RAVDESS clips", flush=True)
    if extracted != set(expected):
        raise RavdessPreparationError("selected archive extraction was incomplete")


def _wav_metadata(path: Path) -> tuple[int, int, int]:
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getsampwidth() != 2:
                raise RavdessPreparationError(f"{path}: expected PCM16 WAV")
            return (
                round(audio.getnframes() / audio.getframerate() * 1000),
                audio.getframerate(),
                audio.getnchannels(),
            )
    except (OSError, wave.Error) as error:
        raise RavdessPreparationError(f"{path}: invalid WAV: {error}") from error


def convert_selected(cache_root: Path, rows: list[dict[str, Any]]) -> None:
    """Normalize all official 48 kHz sources to the Attune 16 kHz mono contract."""
    for index, row in enumerate(rows, 1):
        relative = Path(str(row["archive_member"]))
        source = cache_root / "clips" / relative
        target = cache_root / "pcm16_16k" / relative
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
                raise RavdessPreparationError(
                    f"FFmpeg conversion failed for {source}: {process.stderr.strip()}"
                )
            temporary.replace(target)
        duration_ms, sample_rate_hz, channels = _wav_metadata(target)
        if sample_rate_hz != 16_000 or channels != 1 or duration_ms < 500:
            raise RavdessPreparationError(f"{target}: invalid normalized audio contract")
        if index % 60 == 0:
            print(f"Normalized {index}/{len(rows)} RAVDESS clips", flush=True)


def build_manifests(
    rows: list[dict[str, Any]],
    *,
    cache_root: Path,
    provenance_manifest: Path,
    source_manifest: Path,
) -> tuple[list[dict[str, Any]], list[SourceRow]]:
    provenance_rows: list[dict[str, Any]] = []
    source_rows: list[SourceRow] = []
    categories = [category.value for category in AffectCategory]
    for row in rows:
        source_audio_path = cache_root / "clips" / str(row["archive_member"])
        audio_path = cache_root / "pcm16_16k" / str(row["archive_member"])
        if not source_audio_path.is_file() or not audio_path.is_file():
            raise RavdessPreparationError(f"missing selected audio: {audio_path}")
        source_duration_ms, source_sample_rate_hz, source_channels = _wav_metadata(
            source_audio_path
        )
        duration_ms, sample_rate_hz, channels = _wav_metadata(audio_path)
        if sample_rate_hz != 16_000 or channels != 1:
            raise RavdessPreparationError(f"{audio_path}: expected normalized 16 kHz mono WAV")
        if abs(duration_ms - source_duration_ms) > 20:
            raise RavdessPreparationError(f"{audio_path}: normalized duration drift")
        audio_sha256 = digest(audio_path)
        target = str(row["target_affect"])
        distribution = {category: float(category == target) for category in categories}
        provenance_rows.append(
            {
                **row,
                "source_dataset": "RAVDESS",
                "source_version": "1.0.0",
                "record_id": RECORD_ID,
                "archive_name": ARCHIVE_NAME,
                "archive_md5": ARCHIVE_MD5,
                "source_audio_sha256": digest(source_audio_path),
                "source_duration_ms": source_duration_ms,
                "source_sample_rate_hz": source_sample_rate_hz,
                "source_channels": source_channels,
                "audio_sha256": audio_sha256,
                "duration_ms": duration_ms,
                "sample_rate_hz": sample_rate_hz,
                "channels": channels,
                "licence": LICENCE,
                "attribution": ATTRIBUTION,
                "research_only": True,
                "release_weight_training_prohibited": True,
                "label_status": "acted source label; external evaluation only",
            }
        )
        source_rows.append(
            SourceRow(
                clip_id=str(row["clip_id"]),
                dataset_id="ravdess_v1_external_affect",
                split="sealed_test",
                speaker_id=str(row["speaker_id"]),
                audio_path=Path(
                    os.path.relpath(audio_path.resolve(), source_manifest.parent.resolve())
                ),
                audio_sha256=audio_sha256,
                duration_ms=duration_ms,
                transcript=str(row["transcript"]),
                events=None,
                event_presence=[],
                styles=None,
                affect_distribution=distribution,
                vad=None,
                pair_id=-1,
                is_ood=False,
                lexical_affect_label="neutral",
                auxiliary_negative_tasks=["event_presence"],
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
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/ravdess-affect-eval"))
    parser.add_argument(
        "--provenance-manifest",
        type=Path,
        default=Path("data/manifests/ravdess-affect-external-v0.1.jsonl"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("artifacts/manifests/ravdess-affect-external-v0.1-source.jsonl"),
    )
    parser.add_argument("--download", action="store_true")
    arguments = parser.parse_args()

    archive = arguments.cache_dir / "_archives" / ARCHIVE_NAME
    if arguments.download and not archive.is_file():
        _download(ARCHIVE_URL, archive)
    if not archive.is_file():
        raise SystemExit(f"error: missing {archive}; rerun with --download")
    if digest(archive, "md5") != ARCHIVE_MD5:
        raise SystemExit(f"error: official MD5 mismatch for {archive}")
    rows = protocol_rows()
    extract_selected(archive, arguments.cache_dir, rows)
    convert_selected(arguments.cache_dir, rows)
    provenance, source = build_manifests(
        rows,
        cache_root=arguments.cache_dir,
        provenance_manifest=arguments.provenance_manifest,
        source_manifest=arguments.source_manifest,
    )
    actor_count = len({row["actor_id"] for row in provenance})
    print(f"Verified {len(source)} evaluation-only clips from {actor_count} actors", flush=True)


if __name__ == "__main__":
    try:
        main()
    except RavdessPreparationError as error:
        raise SystemExit(f"error: {error}") from error
