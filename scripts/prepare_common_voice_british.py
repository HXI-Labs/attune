#!/usr/bin/env python3
"""Stream, convert, and verify a speaker-disjoint British Common Voice WER slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Any

DATASET_NAME = "Mozilla Common Voice Corpus 17.0 English"
MIRROR_ID = "fixie-ai/common_voice_17_0"
MIRROR_REVISION = "34f78a43893414e7b6e271ba94c1d5e05f18b239"
SOURCE_SPLIT = "train"
ROWS_API = "https://datasets-server.huggingface.co/rows"
DEFAULT_MANIFEST = Path("data/manifests/common-voice-british-wer.jsonl")
DEFAULT_CACHE = Path("data/raw/common-voice-british-wer")
DEFAULT_CLIP_COUNT = 100
MIN_DURATION_S = 0.5
MAX_DURATION_S = 30.0
ACCENT_FIELD = "accent"
ACCENT_VALUES = ("England English", "Scottish English", "Welsh English")
LICENCE = "CC0-1.0"
ATTRIBUTION = "Mozilla Common Voice Corpus 17.0 English, released under CC0."
FETCH_SCRIPT = "scripts/prepare_common_voice_british.py"
PAGE_SIZE = 100

REQUIRED_FIELDS = {
    "clip_id",
    "source_dataset",
    "source_mirror",
    "dataset_revision",
    "source_split",
    "source_row_index",
    "source_filename",
    "transcript",
    "accent_field",
    "accent_value",
    "client_id",
    "duration_s",
    "sample_rate_hz",
    "channels",
    "sha256",
    "source_audio_sha256",
    "licence",
    "attribution",
    "speaker_disjoint",
    "cache_path",
    "fetch_script",
}


class CommonVoicePreparationError(RuntimeError):
    """Raised when the British Common Voice slice cannot be prepared safely."""


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load a manifest and enforce its CC0, accent, and speaker invariants."""
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_clients: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise CommonVoicePreparationError(
                f"{path}:{line_number}: invalid JSON: {error}"
            ) from error
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise CommonVoicePreparationError(
                f"{path}:{line_number}: missing fields: {', '.join(sorted(missing))}"
            )
        if row["clip_id"] in seen_ids:
            raise CommonVoicePreparationError(f"{path}:{line_number}: duplicate clip_id")
        if not row["client_id"] or row["client_id"] in seen_clients:
            raise CommonVoicePreparationError(
                f"{path}:{line_number}: client_id must be present and unique"
            )
        seen_ids.add(row["clip_id"])
        seen_clients.add(row["client_id"])
        if (
            row["source_dataset"] != DATASET_NAME
            or row["source_mirror"] != MIRROR_ID
            or row["dataset_revision"] != MIRROR_REVISION
            or row["source_split"] != SOURCE_SPLIT
            or row["accent_field"] != ACCENT_FIELD
            or row["accent_value"] not in ACCENT_VALUES
            or row["licence"] != LICENCE
            or row["speaker_disjoint"] is not True
            or row["fetch_script"] != FETCH_SCRIPT
        ):
            raise CommonVoicePreparationError(
                f"{path}:{line_number}: British CC0 provenance invariant failed"
            )
        if not MIN_DURATION_S <= float(row["duration_s"]) <= MAX_DURATION_S:
            raise CommonVoicePreparationError(f"{path}:{line_number}: duration is outside contract")
        if row["sample_rate_hz"] != 16_000 or row["channels"] != 1:
            raise CommonVoicePreparationError(
                f"{path}:{line_number}: audio contract is not 16 kHz mono"
            )
        rows.append(row)
    if not 80 <= len(rows) <= 120:
        raise CommonVoicePreparationError(f"manifest {path} must contain 80–120 rows")
    return rows


def safe_target(cache_root: Path, relative_path: str) -> Path:
    root = cache_root.resolve()
    target = (root / relative_path).resolve()
    if not target.is_relative_to(root):
        raise CommonVoicePreparationError(f"cache_path escapes cache root: {relative_path!r}")
    return target


def verify_audio(path: Path, expected_duration_s: float) -> None:
    try:
        with wave.open(str(path), "rb") as audio:
            duration_s = audio.getnframes() / audio.getframerate()
            contract = (
                audio.getframerate() == 16_000
                and audio.getnchannels() == 1
                and audio.getsampwidth() == 2
            )
    except (OSError, wave.Error) as error:
        raise CommonVoicePreparationError(f"{path}: invalid WAV: {error}") from error
    if not contract:
        raise CommonVoicePreparationError(f"{path}: expected 16 kHz mono PCM16 WAV")
    if abs(duration_s - expected_duration_s) > 0.05:
        raise CommonVoicePreparationError(
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
            raise CommonVoicePreparationError(f"{row['clip_id']}: SHA-256 mismatch for {target}")
        verify_audio(target, float(row["duration_s"]))
    if missing:
        examples = "\n".join(f"  - {path}" for path in missing[:10])
        raise CommonVoicePreparationError(
            f"{len(missing)} manifest files are missing:\n{examples}\n"
            f"Rerun {FETCH_SCRIPT} --download to fetch only the selected clips."
        )
    return len(rows)


def _request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "attune-common-voice-slice/1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except (OSError, json.JSONDecodeError) as error:
        raise CommonVoicePreparationError(f"metadata request failed: {error}") from error


def _page(offset: int, length: int = PAGE_SIZE) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "dataset": MIRROR_ID,
            "config": "en",
            "split": SOURCE_SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    payload = _request_json(f"{ROWS_API}?{query}")
    if "rows" not in payload:
        raise CommonVoicePreparationError(f"rows API returned no rows: {payload.get('error')}")
    return payload


def _asset_url(api_row: dict[str, Any]) -> str:
    assets = api_row.get("audio")
    source_url = assets[0].get("src") if isinstance(assets, list) and assets else None
    if not isinstance(source_url, str) or not source_url:
        raise CommonVoicePreparationError("rows API returned no audio asset URL")
    encoded_revision = f"/--/{MIRROR_REVISION}/--/"
    if encoded_revision not in source_url:
        raise CommonVoicePreparationError(
            "rows API asset is not from the pinned mirror revision; refusing mutable data"
        )
    return source_url


def _download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "attune-common-voice-slice/1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    except OSError as error:
        target.unlink(missing_ok=True)
        raise CommonVoicePreparationError(f"audio fetch failed: {error}") from error


def _convert(source_url: str, target: Path) -> tuple[float, str, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav") as source_file:
        _download(source_url, Path(source_file.name))
        source_hash = file_digest(Path(source_file.name))
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
            raise CommonVoicePreparationError(f"ffmpeg conversion failed: {process.stderr.strip()}")
        temporary.replace(target)
    with wave.open(str(target), "rb") as audio:
        duration_s = audio.getnframes() / audio.getframerate()
    return duration_s, file_digest(target), source_hash


def _source_row(
    row_index: int, page_cache: dict[int, dict[int, dict[str, Any]]] | None = None
) -> dict[str, Any]:
    page_offset = row_index - (row_index % PAGE_SIZE)
    if page_cache is not None and page_offset in page_cache:
        indexed = page_cache[page_offset]
    else:
        payload = _page(page_offset)
        indexed = {int(item["row_idx"]): item["row"] for item in payload["rows"]}
        if page_cache is not None:
            page_cache[page_offset] = indexed
    if row_index not in indexed:
        raise CommonVoicePreparationError(f"rows API did not reproduce source row {row_index}")
    return indexed[row_index]


def _stream_metadata() -> Any:
    """Stream metadata columns only; selected audio is fetched separately."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise CommonVoicePreparationError(
            "streaming metadata requires the dataset-tools extra: "
            "uv sync --extra dataset-tools"
        ) from error
    dataset = load_dataset(
        MIRROR_ID,
        "en",
        split=SOURCE_SPLIT,
        streaming=True,
        revision=MIRROR_REVISION,
    )
    return dataset.select_columns(["client_id", "path", "sentence", ACCENT_FIELD])


def create_manifest(manifest: Path, cache_root: Path, clip_count: int) -> list[dict[str, Any]]:
    """Select one exact-accent clip per client while downloading only candidates."""
    if not 80 <= clip_count <= 120:
        raise CommonVoicePreparationError("clip_count must remain between 80 and 120")
    rows: list[dict[str, Any]] = []
    seen_clients: set[str] = set()
    page_cache: dict[int, dict[int, dict[str, Any]]] = {}
    for row_index, metadata in enumerate(_stream_metadata()):
        client_id = str(metadata.get("client_id", "")).strip()
        transcript = str(metadata.get("sentence", "")).strip()
        accent = str(metadata.get(ACCENT_FIELD, "")).strip()
        if accent not in ACCENT_VALUES or not client_id or not transcript:
            continue
        if client_id in seen_clients:
            continue
        source_row = _source_row(row_index, page_cache)
        if (
            str(source_row.get("client_id", "")).strip() != client_id
            or str(source_row.get("sentence", "")).strip() != transcript
            or str(source_row.get(ACCENT_FIELD, "")).strip() != accent
        ):
            raise CommonVoicePreparationError(
                f"streamed metadata does not match rows API at row {row_index}"
            )
        relative_path = f"clips/cv17-en-train-{row_index:07d}.wav"
        target = safe_target(cache_root, relative_path)
        duration_s, converted_hash, source_hash = _convert(_asset_url(source_row), target)
        if not MIN_DURATION_S <= duration_s <= MAX_DURATION_S:
            target.unlink(missing_ok=True)
            continue
        seen_clients.add(client_id)
        clip_id = f"cv17-british-{source_hash[:16]}"
        rows.append(
            {
                "clip_id": clip_id,
                "source_dataset": DATASET_NAME,
                "source_mirror": MIRROR_ID,
                "dataset_revision": MIRROR_REVISION,
                "source_split": SOURCE_SPLIT,
                "source_row_index": row_index,
                "source_filename": Path(str(source_row["path"])).name,
                "transcript": transcript,
                "accent_field": ACCENT_FIELD,
                "accent_value": accent,
                "client_id": client_id,
                "duration_s": round(duration_s, 6),
                "sample_rate_hz": 16_000,
                "channels": 1,
                "sha256": converted_hash,
                "source_audio_sha256": source_hash,
                "licence": LICENCE,
                "attribution": ATTRIBUTION,
                "speaker_disjoint": True,
                "cache_path": relative_path,
                "fetch_script": FETCH_SCRIPT,
            }
        )
        print(f"[{len(rows)}/{clip_count}] prepared row {row_index}", flush=True)
        if len(rows) == clip_count:
            break
    if len(rows) != clip_count:
        raise CommonVoicePreparationError(f"stream ended after selecting only {len(rows)} clips")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def download_manifest_rows(manifest: Path, cache_root: Path) -> int:
    """Re-fetch missing committed rows without downloading a corpus or shard."""
    rows = load_manifest(manifest)
    page_cache: dict[int, dict[int, dict[str, Any]]] = {}
    for row in rows:
        target = safe_target(cache_root, row["cache_path"])
        if target.exists():
            continue
        source_row = _source_row(int(row["source_row_index"]), page_cache)
        if (
            str(source_row.get("client_id")) != row["client_id"]
            or str(source_row.get("sentence", "")).strip() != row["transcript"]
            or str(source_row.get(ACCENT_FIELD, "")).strip() != row["accent_value"]
        ):
            raise CommonVoicePreparationError(
                f"{row['clip_id']}: pinned source metadata no longer matches"
            )
        duration_s, digest, source_hash = _convert(_asset_url(source_row), target)
        if (
            digest != row["sha256"]
            or source_hash != row["source_audio_sha256"]
            or abs(duration_s - float(row["duration_s"])) > 0.05
        ):
            target.unlink(missing_ok=True)
            raise CommonVoicePreparationError(f"{row['clip_id']}: reproduced WAV does not match")
        print(f"prepared {row['clip_id']}", flush=True)
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
        print(f"Wrote {len(rows)} CC0 British-English rows to {arguments.manifest}")
    elif arguments.download:
        count = download_manifest_rows(arguments.manifest, arguments.cache_dir)
        print(f"Prepared and verified {count} CC0 British-English clips")
    else:
        count = verify(arguments.manifest, arguments.cache_dir)
        print(f"Verified {count} CC0 British-English clips")
    if arguments.create_manifest:
        # Arrow's remote reader can retain a native I/O thread until
        # interpreter finalization; all selected files are closed above.
        os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except CommonVoicePreparationError as error:
        raise SystemExit(f"error: {error}") from error
