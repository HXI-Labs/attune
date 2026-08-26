#!/usr/bin/env python3
"""Stream, convert, and verify the NC research-only Ghanaian-English WER slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

DATASET_ID = "ghananlpcommunity/ghana-english-asr-2700hrs"
DATASET_REVISION = "893a08082ec0f34b5d2fbec56f1ab2230ebea1e7"
DEFAULT_MANIFEST = Path("data/manifests/ghana-english-wer.jsonl")
DEFAULT_CACHE = Path("data/raw/ghana-english-wer")
DEFAULT_CLIP_COUNT = 100
MIN_DURATION_S = 0.5
MAX_DURATION_S = 30.0
LICENCE = "CC-BY-NC-4.0"
ATTRIBUTION = "Ghana NLP Community, Ghana English ASR Dataset (2026)."
FETCH_SCRIPT = "scripts/prepare_ghana_english.py"

REQUIRED_FIELDS = {
    "clip_id",
    "source_dataset",
    "dataset_revision",
    "stream_position",
    "transcript",
    "duration_s",
    "sample_rate_hz",
    "channels",
    "sha256",
    "source_audio_sha256",
    "licence",
    "attribution",
    "research_only",
    "commercial_redistribution_prohibited",
    "speaker_id",
    "speaker_disjoint",
    "cache_path",
    "fetch_script",
}


class GhanaPreparationError(RuntimeError):
    """Raised when the NC-only slice cannot be prepared safely."""


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load a Ghana manifest and enforce its non-commercial-use invariants."""
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise GhanaPreparationError(f"{path}:{line_number}: invalid JSON: {error}") from error
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise GhanaPreparationError(
                f"{path}:{line_number}: missing fields: {', '.join(sorted(missing))}"
            )
        if row["clip_id"] in seen_ids:
            raise GhanaPreparationError(f"{path}:{line_number}: duplicate clip_id")
        seen_ids.add(row["clip_id"])
        if (
            row["source_dataset"] != DATASET_ID
            or row["dataset_revision"] != DATASET_REVISION
            or row["licence"] != LICENCE
            or row["research_only"] is not True
            or row["commercial_redistribution_prohibited"] is not True
            or row["fetch_script"] != FETCH_SCRIPT
        ):
            raise GhanaPreparationError(
                f"{path}:{line_number}: NC research-only provenance invariant failed"
            )
        if row["speaker_id"] is not None or row["speaker_disjoint"] is not False:
            raise GhanaPreparationError(
                f"{path}:{line_number}: dataset has no speaker IDs; do not claim disjointness"
            )
        if not MIN_DURATION_S <= float(row["duration_s"]) <= MAX_DURATION_S:
            raise GhanaPreparationError(f"{path}:{line_number}: duration is outside contract")
        if row["sample_rate_hz"] != 16_000 or row["channels"] != 1:
            raise GhanaPreparationError(f"{path}:{line_number}: audio contract is not 16 kHz mono")
        rows.append(row)
    if not rows:
        raise GhanaPreparationError(f"manifest {path} contains no rows")
    return rows


def safe_target(cache_root: Path, relative_path: str) -> Path:
    root = cache_root.resolve()
    target = (root / relative_path).resolve()
    if not target.is_relative_to(root):
        raise GhanaPreparationError(f"cache_path escapes cache root: {relative_path!r}")
    return target


def verify_audio(path: Path, expected_duration_s: float) -> None:
    try:
        with wave.open(str(path), "rb") as audio:
            duration_s = audio.getnframes() / audio.getframerate()
            if (
                audio.getframerate() != 16_000
                or audio.getnchannels() != 1
                or audio.getsampwidth() != 2
            ):
                raise GhanaPreparationError(f"{path}: expected 16 kHz mono PCM16 WAV")
    except (OSError, wave.Error) as error:
        raise GhanaPreparationError(f"{path}: invalid WAV: {error}") from error
    if abs(duration_s - expected_duration_s) > 0.05:
        raise GhanaPreparationError(
            f"{path}: manifest duration {expected_duration_s:.3f}s != WAV {duration_s:.3f}s"
        )


def verify(manifest: Path, cache_root: Path) -> int:
    rows = load_manifest(manifest)
    missing: list[Path] = []
    for row in rows:
        target = safe_target(cache_root, row["cache_path"])
        if not target.exists():
            missing.append(target)
            continue
        if file_digest(target) != row["sha256"]:
            raise GhanaPreparationError(f"{row['clip_id']}: SHA-256 mismatch for {target}")
        verify_audio(target, float(row["duration_s"]))
    if missing:
        examples = "\n".join(f"  - {path}" for path in missing[:10])
        raise GhanaPreparationError(
            f"{len(missing)} manifest files are missing:\n{examples}\n"
            f"Rerun {FETCH_SCRIPT} --download to stream only the selected clips."
        )
    return len(rows)


def _stream_rows() -> Any:
    try:
        from datasets import Audio, load_dataset
    except ImportError as error:
        raise GhanaPreparationError(
            "streaming requires the dataset-tools extra: uv sync --extra dataset-tools"
        ) from error
    dataset = load_dataset(
        DATASET_ID,
        split="train",
        streaming=True,
        revision=DATASET_REVISION,
    )
    return dataset.cast_column("audio", Audio(decode=False))


def _audio_bytes(row: dict[str, Any]) -> bytes:
    audio = row.get("audio")
    value = audio.get("bytes") if isinstance(audio, dict) else None
    if not isinstance(value, bytes) or not value:
        raise GhanaPreparationError("streamed row did not contain embedded audio bytes")
    return value


def _clip_id(source_audio_sha256: str) -> str:
    return f"ghana-en-{source_audio_sha256[:16]}"


def _convert(source: bytes, target: Path) -> tuple[float, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".audio") as source_file:
        source_file.write(source)
        source_file.flush()
        temporary = target.with_name(f".{target.name}.part.wav")
        process = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                source_file.name,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(temporary),
            ],
            capture_output=True,
            text=True,
        )
        if process.returncode:
            temporary.unlink(missing_ok=True)
            raise GhanaPreparationError(f"ffmpeg conversion failed: {process.stderr.strip()}")
        temporary.replace(target)
    with wave.open(str(target), "rb") as audio:
        duration_s = audio.getnframes() / audio.getframerate()
    return duration_s, file_digest(target)


def create_manifest(manifest: Path, cache_root: Path, clip_count: int) -> list[dict[str, Any]]:
    """Select the first duration-valid rows from the pinned streaming revision."""
    if not 80 <= clip_count <= 120:
        raise GhanaPreparationError("clip_count must remain between 80 and 120")
    rows: list[dict[str, Any]] = []
    seen_source_hashes: set[str] = set()
    for stream_position, source_row in enumerate(_stream_rows()):
        duration_s = float(source_row["duration_ss"])
        transcript = str(source_row["corrected_text"]).strip()
        if not MIN_DURATION_S <= duration_s <= MAX_DURATION_S or not transcript:
            continue
        source = _audio_bytes(source_row)
        source_hash = hashlib.sha256(source).hexdigest()
        if source_hash in seen_source_hashes:
            continue
        seen_source_hashes.add(source_hash)
        clip_id = _clip_id(source_hash)
        relative_path = f"clips/{clip_id}.wav"
        target = safe_target(cache_root, relative_path)
        converted_duration, converted_hash = _convert(source, target)
        if not MIN_DURATION_S <= converted_duration <= MAX_DURATION_S:
            target.unlink(missing_ok=True)
            continue
        rows.append(
            {
                "clip_id": clip_id,
                "source_dataset": DATASET_ID,
                "dataset_revision": DATASET_REVISION,
                "stream_position": stream_position,
                "transcript": transcript,
                "duration_s": round(converted_duration, 6),
                "sample_rate_hz": 16_000,
                "channels": 1,
                "sha256": converted_hash,
                "source_audio_sha256": source_hash,
                "licence": LICENCE,
                "attribution": ATTRIBUTION,
                "research_only": True,
                "commercial_redistribution_prohibited": True,
                "speaker_id": None,
                "speaker_disjoint": False,
                "speaker_limitation": (
                    "The published dataset has no speaker identifier; speaker-disjoint "
                    "sampling and speaker leakage checks are impossible."
                ),
                "cache_path": relative_path,
                "fetch_script": FETCH_SCRIPT,
            }
        )
        print(f"[{len(rows)}/{clip_count}] prepared {clip_id}", flush=True)
        if len(rows) == clip_count:
            break
    if len(rows) != clip_count:
        raise GhanaPreparationError(f"stream ended after selecting only {len(rows)} clips")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def download_manifest_rows(manifest: Path, cache_root: Path) -> int:
    """Re-fetch missing committed rows without downloading any full shard."""
    rows = load_manifest(manifest)
    missing = {
        row["source_audio_sha256"]: row
        for row in rows
        if not safe_target(cache_root, row["cache_path"]).exists()
    }
    if not missing:
        return verify(manifest, cache_root)
    maximum_position = max(int(row["stream_position"]) for row in missing.values())
    for stream_position, source_row in enumerate(_stream_rows()):
        if stream_position > maximum_position:
            break
        source = _audio_bytes(source_row)
        source_hash = hashlib.sha256(source).hexdigest()
        row = missing.get(source_hash)
        if row is None:
            continue
        target = safe_target(cache_root, row["cache_path"])
        duration_s, digest = _convert(source, target)
        if digest != row["sha256"] or abs(duration_s - float(row["duration_s"])) > 0.05:
            target.unlink(missing_ok=True)
            raise GhanaPreparationError(f"{row['clip_id']}: reproduced WAV does not match")
        del missing[source_hash]
        print(f"prepared {row['clip_id']}", flush=True)
        if not missing:
            break
    if missing:
        raise GhanaPreparationError(
            f"stream did not reproduce {len(missing)} selected clips at pinned revision"
        )
    return verify(manifest, cache_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--create-manifest", action="store_true")
    parser.add_argument("--clip-count", type=int, default=DEFAULT_CLIP_COUNT)
    arguments = parser.parse_args()
    if arguments.create_manifest:
        rows = create_manifest(arguments.manifest, arguments.cache_dir, arguments.clip_count)
        print(
            f"Wrote {len(rows)} NC research-only rows to {arguments.manifest}",
            flush=True,
        )
    elif arguments.download:
        count = download_manifest_rows(arguments.manifest, arguments.cache_dir)
        print(f"Prepared and verified {count} NC research-only clips", flush=True)
    else:
        count = verify(arguments.manifest, arguments.cache_dir)
        print(f"Verified {count} NC research-only clips")
    if arguments.create_manifest or arguments.download:
        # Arrow's remote Parquet reader can retain a native I/O thread until
        # interpreter finalization (apache/arrow#45214). All files are closed
        # and verified above, so avoid that upstream shutdown-only abort.
        os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except GhanaPreparationError as error:
        raise SystemExit(f"error: {error}") from error
