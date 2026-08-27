#!/usr/bin/env python3
"""Build and verify bounded, licence-clean FSD50K and CREMA-D slices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import wave
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

FSD50K_REVISION = "812caa9897ee9e0e9a3b0ce075f7d70f14fa6460"
FSD50K_RECORD = "4060432"
CREMA_D_REVISION = "1658cd342dff90010aa843eaeebd53610a08b1dc"
DEFAULT_CACHE = Path("data/raw/licence-clean-inspection")
DEFAULT_MANIFEST = Path("data/manifests/licence-clean-inspection.jsonl")
INSPECTION_MANIFEST = Path("data/manifests/inspection-set.jsonl")
FSD_CLASSES = {
    "Crying_and_sobbing": {
        "affect": [],
        "events": ["sob"],
        "styles": [],
        "note": "Standalone sob event; not crying_speech without speech-with-crying review.",
    },
    "Screaming": {
        "affect": [],
        "events": [],
        "styles": [],
        "note": "Screaming is retained as a separate source class and is not mapped to shouting.",
    },
    "Shout": {
        "affect": [],
        "events": [],
        "styles": ["shouting"],
        "note": "Standalone Freesound shout; not evidence of speech-embedded shouting.",
    },
    "Whispering": {
        "affect": [],
        "events": [],
        "styles": ["whispering"],
        "note": "Source-labelled whispering weakly maps to whispering.",
    },
}
FSD_CLASS_QUOTA = 25
CREMA_SPEAKER_QUOTA = 20
CREMA_AFFECT = {
    "ANG": "anger",
    "DIS": "other",
    "FEA": "fear",
}
CREMA_SENTENCE = "It's eleven o'clock"
ALLOWED_CLIP_LICENCES = {
    "http://creativecommons.org/licenses/by/3.0/": ("CC-BY-3.0", True),
    "http://creativecommons.org/publicdomain/zero/1.0/": ("CC0-1.0", False),
}
ARCHIVES = {
    "FSD50K.metadata.zip": {
        "url": (
            f"https://zenodo.org/records/{FSD50K_RECORD}/files/"
            "FSD50K.metadata.zip?download=1"
        ),
        "sha256": "9a738e032546f9a2c6e3d04566928d04a65fb79b422cc9d78bb781723537bd19",
    },
    "FSD50K.ground_truth.zip": {
        "url": (
            f"https://zenodo.org/records/{FSD50K_RECORD}/files/"
            "FSD50K.ground_truth.zip?download=1"
        ),
        "sha256": "db2396260a7b1fb06feb6ccac71685794fc89917d71c3a8d64a1de85c2cc0ebf",
    },
}
TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}


class PreparationError(RuntimeError):
    """Raised when the bounded preparation contract cannot be satisfied."""


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download_with_retry(
    url: str,
    target: Path,
    *,
    timeout_s: float,
    retries: int,
    token: str | None = None,
) -> None:
    """Download atomically with finite exponential backoff for transient failures."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part")
    headers = {"User-Agent": "attune-licence-inspection/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with (
                urllib.request.urlopen(request, timeout=timeout_s) as response,
                temporary.open("wb") as handle,
            ):
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            temporary.replace(target)
            return
        except urllib.error.HTTPError as error:
            temporary.unlink(missing_ok=True)
            if error.code not in TRANSIENT_HTTP_CODES or attempt == retries:
                raise PreparationError(f"download failed ({error.code}) for {url}") from error
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise PreparationError(f"download failed for {url}: {error}") from error
        time.sleep(min(2**attempt, 16))
    raise AssertionError("unreachable")


def ensure_fsd_metadata(cache_root: Path, timeout_s: float, retries: int) -> Path:
    metadata_root = cache_root / "_metadata"
    for filename, source in ARCHIVES.items():
        archive = metadata_root / filename
        if not archive.exists():
            download_with_retry(
                source["url"], archive, timeout_s=timeout_s, retries=retries
            )
        if digest(archive) != source["sha256"]:
            raise PreparationError(f"metadata archive checksum mismatch: {archive}")
        expected_directory = metadata_root / filename.removesuffix(".zip")
        if not expected_directory.is_dir():
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(metadata_root)
    return metadata_root


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreparationError(f"cannot read metadata {path}: {error}") from error


def _fsd_candidates(metadata_root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for split in ("dev", "eval"):
        clip_info = _read_json(
            metadata_root / "FSD50K.metadata" / f"{split}_clips_info_FSD50K.json"
        )
        csv_path = metadata_root / "FSD50K.ground_truth" / f"{split}.csv"
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = csv.DictReader(handle)
            for row in rows:
                labels = set(row["labels"].split(","))
                selected_labels = labels & FSD_CLASSES.keys()
                if len(selected_labels) != 1:
                    continue
                info = clip_info[row["fname"]]
                if info["license"] not in ALLOWED_CLIP_LICENCES:
                    continue
                for source_label in sorted(selected_labels):
                    candidates.append(
                        {
                            "freesound_id": row["fname"],
                            "source_label": source_label,
                            "source_split": split,
                            "labels": sorted(labels),
                            "title": info["title"],
                            "uploader": info["uploader"],
                            "licence_url": info["license"],
                        }
                    )
    return candidates


def select_fsd(metadata_root: Path) -> list[dict[str, Any]]:
    """Select a stable balanced slice only after applying clip-level licence filters."""
    selected: list[dict[str, Any]] = []
    candidates = _fsd_candidates(metadata_root)
    for source_label in FSD_CLASSES:
        class_rows = sorted(
            (row for row in candidates if row["source_label"] == source_label),
            key=lambda row: (int(row["freesound_id"]), row["source_split"]),
        )
        # Keep a Freesound clip in one source class to avoid duplicate audio and scoring.
        class_rows = [
            row
            for row in class_rows
            if row["freesound_id"] not in {item["freesound_id"] for item in selected}
        ]
        if len(class_rows) < FSD_CLASS_QUOTA:
            raise PreparationError(
                f"{source_label} has only {len(class_rows)} licence-clean candidates"
            )
        selected.extend(class_rows[:FSD_CLASS_QUOTA])
    return selected


def _existing_crema_speakers(path: Path) -> set[str]:
    speakers: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("source_dataset") == "CREMA-D":
            speakers.add(str(row["source_filename"]).split("_", 1)[0])
    return speakers


def select_crema() -> list[dict[str, str]]:
    """Select ANG/DIS/FEA high-intensity files from actors absent from the old set."""
    excluded = _existing_crema_speakers(INSPECTION_MANIFEST)
    selected: list[dict[str, str]] = []
    # CREMA-D actor IDs are 1001..1091. Every actor has the IEO sentence in these classes.
    for speaker in (str(value) for value in range(1001, 1092)):
        if speaker in excluded:
            continue
        for source_label, affect in CREMA_AFFECT.items():
            selected.append(
                {
                    "speaker": speaker,
                    "source_label": source_label,
                    "affect": affect,
                    "filename": f"{speaker}_IEO_{source_label}_HI.wav",
                }
            )
        if len({row["speaker"] for row in selected}) == CREMA_SPEAKER_QUOTA:
            break
    if len(selected) != CREMA_SPEAKER_QUOTA * len(CREMA_AFFECT):
        raise PreparationError("could not select the requested CREMA-D speaker-disjoint slice")
    return selected


def convert_to_pcm16(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part.wav")
    command = [
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
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, timeout=60)
        temporary.replace(target)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        temporary.unlink(missing_ok=True)
        raise PreparationError(f"audio conversion failed for {source}: {error}") from error


def wav_metadata(path: Path) -> tuple[float, int, int, int]:
    try:
        with wave.open(str(path), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
            return duration, audio.getframerate(), audio.getnchannels(), audio.getsampwidth()
    except (OSError, wave.Error) as error:
        raise PreparationError(f"invalid WAV {path}: {error}") from error


def fetch_fsd_row(
    selected: dict[str, Any],
    cache_root: Path,
    timeout_s: float,
    retries: int,
    hf_token: str | None,
) -> dict[str, Any]:
    clip_id = selected["freesound_id"]
    split = selected["source_split"]
    target = cache_root / "fsd50k" / f"{clip_id}.wav"
    if not target.exists():
        source = cache_root / "_downloads" / f"fsd50k-{clip_id}.wav"
        url = (
            "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/"
            f"{FSD50K_REVISION}/clips/{split}/{clip_id}.wav"
        )
        download_with_retry(
            url, source, timeout_s=timeout_s, retries=retries, token=hf_token
        )
        convert_to_pcm16(source, target)
        source.unlink(missing_ok=True)
    duration, sample_rate, channels, sample_width = wav_metadata(target)
    licence_id, attribution_required = ALLOWED_CLIP_LICENCES[selected["licence_url"]]
    mapping = FSD_CLASSES[selected["source_label"]]
    return {
        "acted_status": "unknown",
        "attribution": (
            f'Freesound clip {clip_id} "{selected["title"]}" uploaded by '
            f'{selected["uploader"]}; {licence_id}. FSD50K curation by Fonseca et al.; '
            "dataset annotations CC BY 4.0."
        ),
        "cache_path": f"fsd50k/{clip_id}.wav",
        "clip_id": f"fsd50k-{clip_id}",
        "duration_s": round(duration, 6),
        "fetch": {
            "type": "huggingface_file",
            "repository": "Fhrozen/FSD50k",
            "revision": FSD50K_REVISION,
            "path": f"clips/{split}/{clip_id}.wav",
        },
        "intended_attune_labels": {
            "affect": mapping["affect"],
            "events": mapping["events"],
            "styles": mapping["styles"],
            "label_status": "weak_source_label",
            "source_class": selected["source_label"],
        },
        "licence": {
            "clip_identifier": licence_id,
            "clip_url": selected["licence_url"],
            "attribution_required": attribution_required,
            "dataset_curation": "CC-BY-4.0",
        },
        "notes": (
            f'{mapping["note"]} FSD50K labels: {", ".join(selected["labels"])}. '
            "Weak source label, not reviewed gold."
        ),
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sha256": digest(target),
        "source_dataset": "FSD50K",
        "source_filename": f"{clip_id}.wav",
        "source_split": split,
        "speaker_id": f"freesound:{selected['uploader']}",
        "split": "inspect_events",
        "uploader": selected["uploader"],
    }


def fetch_crema_row(
    selected: dict[str, str],
    cache_root: Path,
    timeout_s: float,
    retries: int,
) -> dict[str, Any]:
    filename = selected["filename"]
    target = cache_root / "crema_d" / filename
    if not target.exists():
        source = cache_root / "_downloads" / filename
        url = (
            "https://media.githubusercontent.com/media/"
            f"CheyneyComputerScience/CREMA-D/{CREMA_D_REVISION}/AudioWAV/{filename}"
        )
        download_with_retry(url, source, timeout_s=timeout_s, retries=retries)
        convert_to_pcm16(source, target)
        source.unlink(missing_ok=True)
    duration, sample_rate, channels, sample_width = wav_metadata(target)
    return {
        "acted_status": "acted",
        "attribution": (
            "CREMA-D by Cao et al. (IEEE Transactions on Affective Computing 2014, "
            "doi:10.1109/TAFFC.2014.2336244); database ODbL 1.0, contents DbCL 1.0."
        ),
        "cache_path": f"crema_d/{filename}",
        "clip_id": f"crema-d-{filename.removesuffix('.wav').lower()}",
        "duration_s": round(duration, 6),
        "fetch": {
            "type": "url",
            "url": (
                "https://media.githubusercontent.com/media/"
                f"CheyneyComputerScience/CREMA-D/{CREMA_D_REVISION}/AudioWAV/{filename}"
            ),
        },
        "intended_attune_labels": {
            "affect": [selected["affect"]],
            "events": [],
            "styles": [],
            "label_status": "weak_source_label",
            "source_emotion": selected["source_label"],
            "source_intensity": "high",
        },
        "licence": {
            "database": "Open Database License 1.0",
            "database_identifier": "ODbL-1.0",
            "contents": "Database Contents License 1.0",
            "url": "https://opendatacommons.org/licenses/odbl/1-0/",
        },
        "notes": (
            f'Acted speech: "{CREMA_SENTENCE}". Source delivery '
            f'{selected["source_label"]}/HI. Intensity is retained only as source metadata '
            "and is not mapped to shouting, whispering, or vocal effort."
        ),
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sha256": digest(target),
        "source_dataset": "CREMA-D",
        "source_filename": filename,
        "speaker_id": f"crema-d:{selected['speaker']}",
        "split": "inspect_affect_expansion",
        "transcript": CREMA_SENTENCE,
    }


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    rows.sort(key=lambda row: row["clip_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise PreparationError(f"cannot read manifest {path}: {error}") from error
    if not rows:
        raise PreparationError(f"manifest is empty: {path}")
    return rows


def verify(rows: list[dict[str, Any]], cache_root: Path) -> None:
    seen: set[str] = set()
    for row in rows:
        if row["clip_id"] in seen:
            raise PreparationError(f"duplicate clip ID: {row['clip_id']}")
        seen.add(row["clip_id"])
        target = (cache_root / row["cache_path"]).resolve()
        if not target.is_relative_to(cache_root.resolve()):
            raise PreparationError(f"cache path escapes root: {row['cache_path']}")
        if not target.is_file():
            raise PreparationError(f"missing audio: {target}")
        if digest(target) != row["sha256"]:
            raise PreparationError(f"checksum mismatch: {target}")
        duration, sample_rate, channels, width = wav_metadata(target)
        if (sample_rate, channels, width) != (16_000, 1, 2):
            raise PreparationError(f"{target} is not 16 kHz mono PCM16")
        if abs(duration - float(row["duration_s"])) > 1e-5:
            raise PreparationError(f"duration mismatch: {target}")
        if row["source_dataset"] == "FSD50K":
            licence = row["licence"]["clip_identifier"]
            if licence not in {"CC0-1.0", "CC-BY-3.0"}:
                raise PreparationError(f"forbidden FSD50K clip licence: {licence}")
    counts = Counter(row["source_dataset"] for row in rows)
    if counts != {"FSD50K": 100, "CREMA-D": 60}:
        raise PreparationError(f"unexpected source counts: {dict(counts)}")


def build(
    cache_root: Path,
    manifest: Path,
    *,
    timeout_s: float,
    retries: int,
    hf_token: str | None,
) -> list[dict[str, Any]]:
    metadata_root = ensure_fsd_metadata(cache_root, timeout_s, retries)
    fsd = select_fsd(metadata_root)
    crema = select_crema()
    rows: list[dict[str, Any]] = []
    for index, selected in enumerate(fsd, 1):
        print(f"FSD50K {index}/{len(fsd)}: {selected['freesound_id']}", flush=True)
        rows.append(
            fetch_fsd_row(selected, cache_root, timeout_s, retries, hf_token)
        )
    for index, selected in enumerate(crema, 1):
        print(f"CREMA-D {index}/{len(crema)}: {selected['filename']}", flush=True)
        rows.append(fetch_crema_row(selected, cache_root, timeout_s, retries))
    write_manifest(rows, manifest)
    verify(rows, cache_root)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--download",
        action="store_true",
        help="select, fetch, convert, and write the deterministic manifest",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    try:
        if arguments.download:
            import os

            rows = build(
                arguments.cache_dir,
                arguments.manifest,
                timeout_s=arguments.timeout,
                retries=arguments.retries,
                hf_token=os.environ.get("HF_TOKEN"),
            )
        else:
            rows = load_manifest(arguments.manifest)
            verify(rows, arguments.cache_dir)
    except PreparationError as error:
        raise SystemExit(f"error: {error}") from error
    print(f"Verified {len(rows)} licence-clean clips in {arguments.cache_dir}")


if __name__ == "__main__":
    main()
