#!/usr/bin/env python3
"""Create deterministic speech-plus-event mixtures with strong event spans."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from attune.training.data import file_sha256
from attune.training.source_adapters import load_jsonl, write_source_rows

SAMPLE_RATE = 16_000
EVENT_LABELS = ("laugh", "sigh", "cough", "throat_clear", "sneeze")
PLACEMENTS = ("before", "overlay", "after")
EVENT_LEVEL_DB = (-6.0, -3.0, 0.0, 3.0, 6.0)


def _read_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as audio:
        if (
            audio.getframerate() != SAMPLE_RATE
            or audio.getnchannels() != 1
            or audio.getsampwidth() != 2
        ):
            raise ValueError(f"expected mono 16 kHz PCM16 audio: {path}")
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    return samples.astype(np.float32) / 32768.0


def _write_pcm16(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.rint(samples * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(pcm.tobytes())


def _active_event(samples: np.ndarray) -> np.ndarray:
    frame_size = round(0.02 * SAMPLE_RATE)
    hop_size = round(0.01 * SAMPLE_RATE)
    if len(samples) < frame_size:
        return samples.copy()
    starts = np.arange(0, len(samples) - frame_size + 1, hop_size)
    rms = np.sqrt(
        np.asarray([np.mean(samples[start : start + frame_size] ** 2) for start in starts])
    )
    threshold = max(float(rms.max()) * 0.10, 1e-4)
    active = np.flatnonzero(rms >= threshold)
    if not len(active):
        raise ValueError("event audio contains no measurable activity")
    context = round(0.05 * SAMPLE_RATE)
    start = max(0, int(starts[active[0]]) - context)
    end = min(len(samples), int(starts[active[-1]]) + frame_size + context)
    if end - start < round(0.12 * SAMPLE_RATE):
        raise ValueError("event activity is shorter than 120 ms")
    return samples[start:end].copy()


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0


def _mix(
    speech: np.ndarray,
    event: np.ndarray,
    *,
    placement: str,
    event_level_db: float,
    seed: int,
) -> tuple[np.ndarray, int, int, dict[str, float | int | str]]:
    if placement not in PLACEMENTS:
        raise ValueError(f"unsupported placement: {placement}")
    generator = np.random.default_rng(seed)
    speech_rms = max(_rms(speech), 1e-4)
    event_rms = max(_rms(event), 1e-4)
    event = event * (speech_rms * 10 ** (event_level_db / 20.0) / event_rms)

    if placement == "before":
        gap = int(generator.uniform(0.12, 0.36) * SAMPLE_RATE)
        start = 0
        mixed = np.concatenate((event, np.zeros(gap, dtype=np.float32), speech))
    elif placement == "after":
        gap = int(generator.uniform(0.12, 0.36) * SAMPLE_RATE)
        start = len(speech) + gap
        mixed = np.concatenate((speech, np.zeros(gap, dtype=np.float32), event))
    else:
        maximum_start = max(0, len(speech) - round(0.2 * SAMPLE_RATE))
        lower = min(maximum_start, round(0.15 * len(speech)))
        upper = max(lower, min(maximum_start, round(0.75 * len(speech))))
        start = int(generator.integers(lower, upper + 1)) if upper > lower else lower
        mixed = np.pad(speech, (0, max(0, start + len(event) - len(speech)))).astype(np.float32)
        mixed[start : start + len(event)] += event

    peak = max(float(np.max(np.abs(mixed))), 1e-6)
    output_peak = float(generator.uniform(0.55, 0.92))
    mixed *= output_peak / peak
    end = start + len(event)
    return (
        mixed,
        start,
        end,
        {
            "placement": placement,
            "event_level_db": event_level_db,
            "output_peak": output_peak,
            "gap_or_start_samples": start,
        },
    )


def _seed(*values: str) -> int:
    digest = hashlib.sha256("\0".join(values).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _synthesis_parameters(selection_seed: int) -> tuple[str, float]:
    placement = PLACEMENTS[(selection_seed >> 16) % len(PLACEMENTS)]
    event_level_db = EVENT_LEVEL_DB[(selection_seed >> 32) % len(EVENT_LEVEL_DB)]
    return placement, event_level_db


def _select_event(
    candidates: list[dict[str, Any]],
    selection_seed: int,
    cache: dict[str, np.ndarray | None],
    excluded: set[str],
) -> tuple[dict[str, Any], np.ndarray]:
    first_index = selection_seed % len(candidates)
    for offset in range(len(candidates)):
        row = candidates[(first_index + offset) % len(candidates)]
        clip_id = str(row["clip_id"])
        if clip_id not in cache:
            event_path = Path(row["audio_path"])
            if file_sha256(event_path) != row["audio_sha256"]:
                raise ValueError(f"VocalSound hash mismatch: {event_path}")
            try:
                cache[clip_id] = _active_event(_read_pcm16(event_path))
            except ValueError:
                cache[clip_id] = None
                excluded.add(clip_id)
        if cache[clip_id] is not None:
            return row, cache[clip_id]
    raise ValueError("no VocalSound candidate contains measurable activity")


def create_mixtures(
    *,
    common_voice_manifest: Path,
    common_voice_root: Path,
    vocalsound_manifest: Path,
    output_root: Path,
    output_manifest: Path,
    audit_path: Path,
) -> list[dict[str, Any]]:
    speech_rows = [
        row
        for row in load_jsonl(common_voice_manifest)
        if row["partition"] in {"train", "development"}
    ]
    event_rows = [
        row for row in load_jsonl(vocalsound_manifest) if row["split"] in {"train", "development"}
    ]
    events_by_split_and_label: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        label = row["event_presence"][0]
        if label in EVENT_LABELS:
            events_by_split_and_label[(row["split"], label)].append(row)
    for candidates in events_by_split_and_label.values():
        candidates.sort(key=lambda row: row["clip_id"])

    rows: list[dict[str, Any]] = []
    event_cache: dict[str, np.ndarray | None] = {}
    excluded_events: set[str] = set()
    for speech_row in sorted(speech_rows, key=lambda row: row["clip_id"]):
        split = speech_row["partition"]
        speech_path = common_voice_root / speech_row["cache_path"]
        if file_sha256(speech_path) != speech_row["sha256"]:
            raise ValueError(f"Common Voice hash mismatch: {speech_path}")
        speech = _read_pcm16(speech_path)
        for label in EVENT_LABELS:
            candidates = events_by_split_and_label[(split, label)]
            if not candidates:
                raise ValueError(f"no {split} VocalSound candidates for {label}")
            selection_seed = _seed(speech_row["clip_id"], label)
            event_row, event = _select_event(
                candidates,
                selection_seed,
                event_cache,
                excluded_events,
            )
            placement, level_db = _synthesis_parameters(selection_seed)
            mixed, start, end, _ = _mix(
                speech,
                event,
                placement=placement,
                event_level_db=level_db,
                seed=selection_seed,
            )
            relative_audio = Path(split) / label / f"{speech_row['clip_id']}.wav"
            output_path = output_root / relative_audio
            _write_pcm16(output_path, mixed)
            clip_id = f"inline-{speech_row['clip_id']}-{label}"
            rows.append(
                {
                    "clip_id": clip_id,
                    "dataset_id": "attune_inline_event_mixtures_v0.1",
                    "split": split,
                    "speaker_id": f"mixture:{speech_row['client_id']}:{event_row['speaker_id']}",
                    "audio_path": os.path.relpath(
                        output_path.resolve(), output_manifest.parent.resolve()
                    ),
                    "audio_sha256": file_sha256(output_path),
                    "duration_ms": round(len(mixed) / SAMPLE_RATE * 1000),
                    "transcript": speech_row["transcript"],
                    "events": [
                        {
                            "label": label,
                            "start_ms": round(start / SAMPLE_RATE * 1000),
                            "end_ms": round(end / SAMPLE_RATE * 1000),
                        }
                    ],
                    "event_presence": [label],
                    "styles": [],
                    "affect_distribution": None,
                    "vad": None,
                    "pair_id": -1,
                    "is_ood": True,
                    "lexical_affect_label": None,
                    "auxiliary_negative_tasks": ["styles"],
                }
            )

    write_source_rows(output_manifest, rows)
    counts = Counter((row["split"], row["events"][0]["label"]) for row in rows)
    placement_counts: Counter[str] = Counter()
    level_counts: Counter[float] = Counter()
    placement_by_label: dict[str, Counter[str]] = {label: Counter() for label in EVENT_LABELS}
    level_by_label: dict[str, Counter[float]] = {label: Counter() for label in EVENT_LABELS}
    for speech_row in speech_rows:
        for label in EVENT_LABELS:
            selection_seed = _seed(speech_row["clip_id"], label)
            placement, level_db = _synthesis_parameters(selection_seed)
            placement_counts[placement] += 1
            level_counts[level_db] += 1
            placement_by_label[label][placement] += 1
            level_by_label[label][level_db] += 1
    audit = {
        "schema_version": "1.0",
        "dataset_id": "attune_inline_event_mixtures_v0.1",
        "algorithm": "deterministic PCM16 concatenation/overlay with active-event trimming",
        "sample_rate_hz": SAMPLE_RATE,
        "source_manifests": {
            "common_voice": {
                "path": str(common_voice_manifest),
                "sha256": file_sha256(common_voice_manifest),
            },
            "vocalsound": {
                "path": str(vocalsound_manifest),
                "sha256": file_sha256(vocalsound_manifest),
            },
        },
        "rows": len(rows),
        "excluded_silent_event_clips": sorted(excluded_events),
        "counts": {f"{split}:{label}": count for (split, label), count in sorted(counts.items())},
        "placement_counts": dict(sorted(placement_counts.items())),
        "event_level_db_counts": {
            f"{level:g}": count for level, count in sorted(level_counts.items())
        },
        "placement_counts_by_label": {
            label: dict(sorted(counts.items())) for label, counts in placement_by_label.items()
        },
        "event_level_db_counts_by_label": {
            label: {f"{level:g}": count for level, count in sorted(counts.items())}
            for label, counts in level_by_label.items()
        },
        "synthesis": {
            "placements": list(PLACEMENTS),
            "event_levels_db": list(EVENT_LEVEL_DB),
            "selection": "SHA-256 of Common Voice clip ID and event label",
            "per_clip_parameters_reproducible_from": [
                "source manifests",
                "output source manifest",
                "generation script",
            ],
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--common-voice-manifest",
        type=Path,
        default=Path("data/manifests/common-voice-replay-v0.1.jsonl"),
    )
    parser.add_argument(
        "--common-voice-root", type=Path, default=Path("data/raw/common-voice-replay-v0.1")
    )
    parser.add_argument(
        "--vocalsound-manifest",
        type=Path,
        default=Path("data/manifests/vocalsound-joint-source-v0.1.jsonl"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/raw/inline-event-mixtures-v0.1")
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/manifests/inline-event-mixtures-source-v0.1.jsonl"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("data/manifests/inline-event-mixtures-v0.1-audit.json"),
    )
    arguments = parser.parse_args()
    rows = create_mixtures(
        common_voice_manifest=arguments.common_voice_manifest,
        common_voice_root=arguments.common_voice_root,
        vocalsound_manifest=arguments.vocalsound_manifest,
        output_root=arguments.output_root,
        output_manifest=arguments.output_manifest,
        audit_path=arguments.audit,
    )
    print(f"Wrote {len(rows)} deterministic inline-event mixtures")


if __name__ == "__main__":
    main()
