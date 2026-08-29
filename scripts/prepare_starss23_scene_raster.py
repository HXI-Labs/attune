#!/usr/bin/env python3
"""Build a tiled 60-second STARSS23 natural-scene raster (not laughter-centered crops)."""

from __future__ import annotations

import argparse
import audioop
import csv
import hashlib
import json
import re
import subprocess
import wave
import zipfile
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any

from attune.integrity import file_digest

SCENE_MS = 60_000
LABEL_FRAME_MS = 100
LAUGHTER_CLASS = 4
MUSIC_CLASS = 8
SOURCE_CHANNELS = 4
SOURCE_RATE_HZ = 24_000
TARGET_RATE_HZ = 16_000
PROTOCOL = "tiled_60s_mean4_mic"
_ROOM_PATTERN = re.compile(r"_room(?P<room>\d+)_")


def metadata_rows(path: Path) -> list[tuple[int, int, int]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if row:
                frame, class_index, source_index = (int(value) for value in row[:3])
                rows.append((frame, class_index, source_index))
    return rows


def wav_duration_ms(path: Path) -> int:
    """Return WAV duration in whole milliseconds from header nframes/rate."""
    with wave.open(str(path), "rb") as reader:
        rate = reader.getframerate()
        frames = reader.getnframes()
    if rate <= 0:
        raise RuntimeError(f"invalid sample rate in {path}")
    return (frames * 1000) // rate


def window_starts_ms(annotated_extent_ms: int, wav_ms: int) -> list[int]:
    """Non-overlapping full 60 s tiles covered by BOTH wav duration and CSV extent.

    Unlabeled wav tails past the annotation and CSV remainders under 60 s are
    never padded or tiled.
    """
    covered_ms = min(annotated_extent_ms, wav_ms)
    if covered_ms < SCENE_MS:
        return []
    return list(range(0, covered_ms - SCENE_MS + 1, SCENE_MS))


def event_spans_60s_cut(start_ms: int, end_ms: int) -> bool:
    """True when an absolute event crosses a 60 s tile boundary."""
    if end_ms <= start_ms:
        return False
    return start_ms // SCENE_MS != (end_ms - 1) // SCENE_MS


def absolute_laughter_events(rows: list[tuple[int, int, int]]) -> list[dict[str, Any]]:
    """Union source-specific laughter into contiguous events in recording time."""
    sources_by_frame: dict[int, set[int]] = defaultdict(set)
    for frame, class_index, source_index in rows:
        if class_index == LAUGHTER_CLASS:
            sources_by_frame[frame].add(source_index)

    events = []
    start = previous = None
    active_sources: set[int] = set()
    for frame in [*sorted(sources_by_frame), None]:
        if start is None:
            start = previous = frame
            if frame is not None:
                active_sources = set(sources_by_frame[frame])
            continue
        if frame is not None and previous is not None and frame == previous + 1:
            previous = frame
            active_sources.update(sources_by_frame[frame])
            continue
        assert previous is not None
        events.append(
            {
                "label": "laugh",
                "start_ms": start * LABEL_FRAME_MS,
                "end_ms": (previous + 1) * LABEL_FRAME_MS,
                "source_class": LAUGHTER_CLASS,
                "source_indices": sorted(active_sources),
            }
        )
        start = previous = frame
        active_sources = set() if frame is None else set(sources_by_frame[frame])
    return events


def laughter_events(
    rows: list[tuple[int, int, int]], *, window_start_ms: int
) -> list[dict[str, Any]]:
    """Clip absolute laughter events to one 60 s window, truncating at the edges.

    This is the first-60s control behaviour: events that span t=60 s stay as
    truncated gold on window_start_ms == 0.
    """
    window_end_ms = window_start_ms + SCENE_MS
    clipped = []
    for event in absolute_laughter_events(rows):
        if event["end_ms"] <= window_start_ms or event["start_ms"] >= window_end_ms:
            continue
        clipped.append(
            {
                "label": event["label"],
                "start_ms": max(0, event["start_ms"] - window_start_ms),
                "end_ms": min(SCENE_MS, event["end_ms"] - window_start_ms),
                "source_class": event["source_class"],
                "source_indices": event["source_indices"],
            }
        )
    return clipped


def scored_laughter_events(
    rows: list[tuple[int, int, int]], *, window_start_ms: int
) -> tuple[list[dict[str, Any]], int]:
    """Window gold after Kyoto spanning-drop.

    First tile (window_start_ms == 0): keep truncated-at-60s events so the
    48-event first-60s control stays intact. Later tiles: drop fragments that
    span a 60 s cut; they are not train targets and not scored gold.
    """
    window_end_ms = window_start_ms + SCENE_MS
    scored = []
    dropped = 0
    for event in absolute_laughter_events(rows):
        if event["end_ms"] <= window_start_ms or event["start_ms"] >= window_end_ms:
            continue
        spans = event_spans_60s_cut(event["start_ms"], event["end_ms"])
        if spans and window_start_ms > 0:
            dropped += 1
            continue
        scored.append(
            {
                "label": event["label"],
                "start_ms": max(0, event["start_ms"] - window_start_ms),
                "end_ms": min(SCENE_MS, event["end_ms"] - window_start_ms),
                "source_class": event["source_class"],
                "source_indices": event["source_indices"],
            }
        )
    return scored, dropped


def clip_id_for(role: str, source_recording: str, window_start_ms: int) -> str:
    """Stable per-tile ID; sequential role-index names overwrite across tiles."""
    stem = Path(source_recording).stem
    return f"starss23-scene-{role}-{stem}-w{window_start_ms:06d}"


def candidates(
    metadata_root: Path,
    audio_root: Path | None = None,
) -> list[dict[str, Any]]:
    result = []
    for path in sorted(metadata_root.glob("**/*.csv")):
        rows = metadata_rows(path)
        if not rows:
            continue
        partition = path.parent.name
        if not partition.startswith("dev-"):
            continue
        final_frame = max(frame for frame, _class_index, _source_index in rows)
        annotated_extent_ms = (final_frame + 1) * LABEL_FRAME_MS
        room_match = _ROOM_PATTERN.search(path.stem)
        if room_match is None:
            raise ValueError(f"cannot parse STARSS23 room from {path.name}")
        room = f"{partition.rsplit('-', 1)[-1]}-room{room_match.group('room')}"
        source_recording = f"{path.stem}.wav"
        if audio_root is None:
            wav_ms = annotated_extent_ms
        else:
            wav_path = audio_root / partition / source_recording
            if not wav_path.is_file():
                raise RuntimeError(f"missing STARSS23 wav for {source_recording}")
            wav_ms = wav_duration_ms(wav_path)
        for window_start_ms in window_starts_ms(annotated_extent_ms, wav_ms):
            start_frame = window_start_ms // LABEL_FRAME_MS
            end_frame = (window_start_ms + SCENE_MS) // LABEL_FRAME_MS
            if end_frame * LABEL_FRAME_MS - window_start_ms != SCENE_MS:
                raise RuntimeError(f"window crosses the 60 s boundary at {window_start_ms}")
            window_rows = [row for row in rows if start_frame <= row[0] < end_frame]
            if not window_rows or any(row[1] == MUSIC_CLASS for row in window_rows):
                continue
            events, clipped_count = scored_laughter_events(rows, window_start_ms=window_start_ms)
            active_by_frame: dict[int, set[tuple[int, int]]] = defaultdict(set)
            for frame, class_index, source_index in window_rows:
                active_by_frame[frame].add((class_index, source_index))
            overlap_frames = sum(
                len(active_sources) > 1 for active_sources in active_by_frame.values()
            )
            laughter_overlap_frames = sum(
                any(class_index == LAUGHTER_CLASS for class_index, _ in active_sources)
                and len(active_sources) > 1
                for active_sources in active_by_frame.values()
            )
            result.append(
                {
                    "metadata_path": path,
                    "official_partition": partition,
                    "source_recording": source_recording,
                    "room": room,
                    "window_start_ms": window_start_ms,
                    "events": events,
                    "clipped_spanning_event_count": clipped_count,
                    "overlap_frames": overlap_frames,
                    "laughter_overlap_frames": laughter_overlap_frames,
                    "max_polyphony": max(map(len, active_by_frame.values())),
                    "active_classes": sorted({row[1] for row in window_rows}),
                    "annotated_extent_ms": annotated_extent_ms,
                    "wav_duration_ms": wav_ms,
                }
            )
    return sorted(
        result,
        key=lambda row: (
            row["official_partition"],
            row["source_recording"],
            row["window_start_ms"],
        ),
    )


def official_partitions(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep every eligible tiled 60 s scene; split on official room/file partitions."""
    development = [row for row in rows if "dev-train-" in row["official_partition"]]
    inspection = [row for row in rows if "dev-test-" in row["official_partition"]]
    if not development or not inspection:
        raise RuntimeError("STARSS23 60s scene raster produced an empty official partition")
    if {row["source_recording"] for row in development} & {
        row["source_recording"] for row in inspection
    }:
        raise RuntimeError("STARSS23 inspection files overlap development")
    if {row["room"] for row in development} & {row["room"] for row in inspection}:
        raise RuntimeError("STARSS23 inspection rooms overlap development")
    return development, inspection


def downmix_window(source: Path, target: Path, start_ms: int) -> None:
    """Cut one 60 s 4-channel/24 kHz excerpt, mean tetrahedral MIC capsules, write mono/16 kHz.

    Mean-of-4 is the omni MIC mix. Do not pick a single channel, FOA W, or
    switch channels inside the window.
    """
    with wave.open(str(source), "rb") as reader:
        if (
            reader.getframerate() != SOURCE_RATE_HZ
            or reader.getnchannels() != SOURCE_CHANNELS
            or reader.getsampwidth() != 2
        ):
            raise RuntimeError(f"unexpected STARSS23 audio format: {source}")
        start_frame = round(start_ms * reader.getframerate() / 1000)
        frame_count = round(SCENE_MS * reader.getframerate() / 1000)
        if start_frame + frame_count > reader.getnframes():
            raise RuntimeError(f"STARSS23 scene exceeds source duration: {source}")
        reader.setpos(start_frame)
        samples = array("h", reader.readframes(frame_count))
    mono = array(
        "h",
        (
            max(
                -32768,
                min(32767, round(sum(samples[index : index + SOURCE_CHANNELS]) / SOURCE_CHANNELS)),
            )
            for index in range(0, len(samples), SOURCE_CHANNELS)
        ),
    )
    resampled, _state = audioop.ratecv(mono.tobytes(), 2, 1, SOURCE_RATE_HZ, TARGET_RATE_HZ, None)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as writer:
        writer.setparams((1, 2, TARGET_RATE_HZ, 0, "NONE", "not compressed"))
        writer.writeframes(resampled)


def materialize(
    rows: list[dict[str, Any]],
    *,
    audio_root: Path,
    cache_root: Path,
    role: str,
) -> list[dict[str, Any]]:
    manifest = []
    for row in rows:
        source = audio_root / row["official_partition"] / row["source_recording"]
        clip_id = clip_id_for(role, row["source_recording"], row["window_start_ms"])
        cache_path = f"{role}/{clip_id}.wav"
        target = cache_root / cache_path
        downmix_window(source, target, row["window_start_ms"])
        manifest.append(
            {
                "clip_id": clip_id,
                "cache_path": cache_path,
                "sha256": file_digest(target),
                "duration_ms": SCENE_MS,
                "sample_rate_hz": TARGET_RATE_HZ,
                "channels": 1,
                "source_sample_rate_hz": SOURCE_RATE_HZ,
                "source_channels": SOURCE_CHANNELS,
                "downmix": "mean_of_4_tetrahedral_mic",
                "source_recording": row["source_recording"],
                "source_window_start_ms": row["window_start_ms"],
                "official_partition": row["official_partition"],
                "room": row["room"],
                "events": row["events"],
                "clipped_spanning_event_count": row["clipped_spanning_event_count"],
                "natural_overlap": row["overlap_frames"] > 0,
                "laughter_overlap_frames": row["laughter_overlap_frames"],
                "max_polyphony": row["max_polyphony"],
                "source_active_classes_excluding_music": row["active_classes"],
                "language": "unverified",
                "label_status": "STARSS23 human activity label; not reviewed Attune gold",
                "licence": "MIT",
                "attribution": "STARSS23 v1.1, Politis et al. and Shimada et al.",
                "protocol": PROTOCOL,
            }
        )
    return manifest


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def md5(path: Path) -> str:
    value = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download_file(url: str, target: Path) -> None:
    """Fetch one Zenodo asset once. Callers must not retry a failed fetch."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part")
    completed = subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "0",
            "--connect-timeout",
            "30",
            "-o",
            str(temporary),
            url,
        ],
        check=False,
    )
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"download failed for {url} (curl exit {completed.returncode})")
    temporary.replace(target)


def ensure_asset(url: str, target: Path, expected_md5: str) -> None:
    if target.is_file() and md5(target) == expected_md5:
        return
    if target.is_file():
        raise RuntimeError(f"checksum mismatch for existing file: {target}")
    download_file(url, target)
    if md5(target) != expected_md5:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded checksum mismatch: {target}")


def fetch_starss23(raw_root: Path, archive_dir: Path) -> None:
    zenodo_api = "https://zenodo.org/api/records/7880637/files"
    licence_md5 = "1c11108eda7c915172b10c48276cc189"
    metadata_md5 = "e73af95a6d5f3f7e009ac6a70804f44a"
    microphone_md5 = "06967bcc8def1580c2425fabc311dbd2"
    assets = (
        ("LICENSE", licence_md5, raw_root / "LICENSE"),
        ("metadata_dev.zip", metadata_md5, archive_dir / "metadata_dev.zip"),
        ("mic_dev.zip", microphone_md5, archive_dir / "mic_dev.zip"),
    )
    for filename, expected, target in assets:
        ensure_asset(f"{zenodo_api}/{filename}/content", target, expected)
    metadata_root = raw_root / "metadata"
    audio_root = raw_root / "audio"
    if not (metadata_root / "metadata_dev").is_dir():
        metadata_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_dir / "metadata_dev.zip") as bundle:
            bundle.extractall(metadata_root)
    if not (audio_root / "mic_dev").is_dir():
        audio_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_dir / "mic_dev.zip") as bundle:
            bundle.extractall(audio_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path("data/raw/starss23/metadata/metadata_dev"),
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("data/raw/starss23/audio/mic_dev"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/starss23-scene-raster-tiled"),
    )
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=Path("data/manifests/starss23-scene-raster-development.jsonl"),
    )
    parser.add_argument(
        "--inspection-manifest",
        type=Path,
        default=Path("data/manifests/starss23-scene-raster-inspection.jsonl"),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("data/manifests/starss23-scene-raster.provenance.json"),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="fetch STARSS23 development MIC audio and metadata from Zenodo once",
    )
    arguments = parser.parse_args()
    if arguments.download:
        fetch_starss23(
            Path("data/raw/starss23"),
            Path("data/raw/starss23/_archives"),
        )
    if not arguments.metadata_root.is_dir() or not arguments.audio_root.is_dir():
        raise SystemExit(
            "error: STARSS23 development metadata/audio are missing; rerun with --download"
        )

    all_candidates = candidates(arguments.metadata_root, arguments.audio_root)
    development, inspection = official_partitions(all_candidates)
    development_manifest = materialize(
        development,
        audio_root=arguments.audio_root,
        cache_root=arguments.cache_dir,
        role="development",
    )
    inspection_manifest = materialize(
        inspection,
        audio_root=arguments.audio_root,
        cache_root=arguments.cache_dir,
        role="inspection_test",
    )
    write_jsonl(arguments.development_manifest, development_manifest)
    write_jsonl(arguments.inspection_manifest, inspection_manifest)
    development_events = sum(len(row["events"]) for row in development_manifest)
    inspection_events = sum(len(row["events"]) for row in inspection_manifest)
    development_clipped = sum(row["clipped_spanning_event_count"] for row in development_manifest)
    inspection_clipped = sum(row["clipped_spanning_event_count"] for row in inspection_manifest)
    provenance = {
        "schema_version": 1,
        "dataset": "STARSS23 v1.1 development set",
        "record": "https://zenodo.org/records/7880637",
        "doi": "10.5281/zenodo.7880637",
        "licence": "MIT",
        "licence_file_md5": "1c11108eda7c915172b10c48276cc189",
        "metadata_archive_md5": "e73af95a6d5f3f7e009ac6a70804f44a",
        "microphone_development_archive_md5": "06967bcc8def1580c2425fabc311dbd2",
        "protocol": PROTOCOL,
        "selection": (
            "non-overlapping 60-second tiles where BOTH wav duration and CSV annotated "
            "extent cover the full 60 s; skip Music class 8; do not tile unlabeled wav "
            "tails or pad remainders; official dev-train vs dev-test rooms and files "
            "remain disjoint; later-tile laugh fragments that span a 60 s cut are dropped"
        ),
        "downmix": (
            "mean of the four unlabeled tetrahedral MIC capsules over the whole 60 s "
            "window (omni, not FOA W, not max-RMS, no in-window channel switching); "
            "resample 24 kHz to 16 kHz mono"
        ),
        "mapping": {"4_laughter": "laugh"},
        "source_identity_mapping": (
            "simultaneous source-specific laughter frames are unioned because "
            "Attune events do not carry source identity; non-contiguous bursts stay distinct"
        ),
        "unmapped_classes": list(range(13)),
        "music_class_excluded": 8,
        "language": "unverified; STARSS23 metadata has no language field",
        "source_format": {
            "sample_rate_hz": SOURCE_RATE_HZ,
            "channels": SOURCE_CHANNELS,
            "format": "MIC",
        },
        "derived_format": {"sample_rate_hz": TARGET_RATE_HZ, "channels": 1},
        "development_manifest": str(arguments.development_manifest),
        "development_rows": len(development_manifest),
        "development_events": development_events,
        "development_clipped_spanning_events": development_clipped,
        "development_manifest_sha256": file_digest(arguments.development_manifest),
        "inspection_manifest": str(arguments.inspection_manifest),
        "inspection_rows": len(inspection_manifest),
        "inspection_events": inspection_events,
        "inspection_clipped_spanning_events": inspection_clipped,
        "audio_committed": False,
        "privacy_consent": (
            "Natural participant recordings; collection included consent and published "
            "face-blurring procedures, but this operational review is not independent "
            "consent verification. Use only for bounded research."
        ),
        "attribution": (
            "STARSS23 by Politis, Shimada, Sudarsanam, Hakala, Takahashi, Krause, "
            "Adavanne, Koyama, Uchida, Mitsufuji, Virtanen, and collaborators."
        ),
    }
    provenance["unmapped_classes"].remove(LAUGHTER_CLASS)
    arguments.provenance.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(development_manifest)} development ({development_events} events, "
        f"{development_clipped} clipped) and {len(inspection_manifest)} inspection "
        f"({inspection_events} events, {inspection_clipped} clipped) STARSS23 60s tiles"
    )


if __name__ == "__main__":
    main()
