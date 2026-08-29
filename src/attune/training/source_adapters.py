"""Adapters from reviewed source manifests to Attune's normalized source rows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from attune.schema.output import AffectCategory

CREMA_SENTENCES = {
    "IEO": "It's eleven o'clock.",
    "IWL": "I would like a new alarm clock.",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _split_group(group: str, *, development_percent: int = 20) -> str:
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return "development" if bucket < development_percent else "train"


def _one_hot_affect(label: str) -> dict[str, float]:
    return {category.value: float(category.value == label) for category in AffectCategory}


def adapt_dcase(
    rows: Iterable[dict[str, Any]], *, cache_root: Path, sealed: bool = False
) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        events = [
            {"label": event["label"], "start_ms": event["start_ms"], "end_ms": event["end_ms"]}
            for event in row["events"]
        ]
        split = "sealed_test" if sealed else _split_group(f"dcase:{row['source_recording']}")
        normalized.append(
            {
                "clip_id": row["clip_id"],
                "dataset_id": "dcase2016_task2",
                "split": split,
                "speaker_id": None,
                "audio_path": str((cache_root / row["cache_path"]).resolve()),
                "audio_sha256": row["sha256"],
                "duration_ms": row["duration_ms"],
                "transcript": None,
                "events": events,
                "event_presence": sorted({event["label"] for event in events}),
                "styles": None,
                "affect_distribution": None,
                "vad": None,
                "pair_id": -1,
                "is_ood": True,
                "lexical_affect_label": None,
            }
        )
    return normalized


def adapt_fsd50k(rows: Iterable[dict[str, Any]], *, cache_root: Path) -> list[dict[str, Any]]:
    materialized = list(rows)
    validation_by_label: dict[str, list[str]] = {}
    for row in materialized:
        if row["partition"] == "validation":
            validation_by_label.setdefault(row["probe_label"], []).append(row["clip_id"])
    sealed_ids = {
        clip_id
        for identifiers in validation_by_label.values()
        for index, clip_id in enumerate(sorted(identifiers))
        if index % 2 == 1
    }
    normalized = []
    for row in materialized:
        label = row["probe_label"]
        styles = {"shout": ["shouting"], "whisper": ["whispering"]}.get(label)
        presence = {"sob": ["sob"], "scream": ["scream"]}.get(label)
        if row["partition"] == "validation":
            split = "sealed_test" if row["clip_id"] in sealed_ids else "development"
        else:
            split = "train"
        normalized.append(
            {
                "clip_id": row["clip_id"],
                "dataset_id": "fsd50k_bounded_v0.1",
                "split": split,
                "speaker_id": None,
                "audio_path": str((cache_root / row["cache_path"]).resolve()),
                "audio_sha256": row["sha256"],
                "duration_ms": round(float(row["duration_s"]) * 1000),
                "transcript": None,
                "events": None,
                "event_presence": presence,
                "styles": styles,
                "affect_distribution": None,
                "vad": None,
                "pair_id": -1,
                "is_ood": True,
                "lexical_affect_label": None,
            }
        )
    return normalized


def adapt_crema(
    rows: Iterable[dict[str, Any]],
    *,
    cache_root: Path,
    auxiliary_negative_controls: bool = False,
) -> list[dict[str, Any]]:
    normalized = []
    pair_ids: dict[str, int] = {}
    for row in rows:
        filename = Path(row["source_filename"]).stem
        parts = filename.split("_")
        if len(parts) != 4 or parts[1] not in CREMA_SENTENCES:
            raise ValueError(f"unsupported CREMA filename: {filename}")
        sentence_code = parts[1]
        pair_id = pair_ids.setdefault(sentence_code, len(pair_ids))
        normalized.append(
            {
                "clip_id": row["clip_id"],
                "dataset_id": "crema_d_paired_v0.1",
                "split": row["partition"],
                "speaker_id": row["speaker_id"],
                "audio_path": str((cache_root / row["cache_path"]).resolve()),
                "audio_sha256": row["sha256"],
                "duration_ms": row["duration_ms"],
                "transcript": CREMA_SENTENCES[sentence_code],
                "events": None,
                "event_presence": [] if auxiliary_negative_controls else None,
                "styles": None,
                "affect_distribution": _one_hot_affect(row["target_affect"]),
                "vad": None,
                "pair_id": pair_id,
                "is_ood": False,
                "lexical_affect_label": "neutral",
                "auxiliary_negative_tasks": (
                    ["event_presence"] if auxiliary_negative_controls else []
                ),
            }
        )
    return normalized


def adapt_common_voice(
    rows: Iterable[dict[str, Any]],
    *,
    cache_root: Path,
    split: str | None,
    auxiliary_negative_controls: bool = False,
) -> list[dict[str, Any]]:
    return [
        {
            "clip_id": row["clip_id"],
            "dataset_id": "common_voice_17_en",
            "split": split or row["partition"],
            "speaker_id": f"cv17:{row['client_id']}",
            "audio_path": str((cache_root / row["cache_path"]).resolve()),
            "audio_sha256": row["sha256"],
            "duration_ms": round(float(row["duration_s"]) * 1000),
            "transcript": row["transcript"],
            "events": None,
            "event_presence": [] if auxiliary_negative_controls else None,
            "styles": [] if auxiliary_negative_controls else None,
            "affect_distribution": None,
            "vad": None,
            "pair_id": -1,
            "is_ood": False,
            "lexical_affect_label": None,
            "auxiliary_negative_tasks": (
                ["event_presence", "styles"] if auxiliary_negative_controls else []
            ),
        }
        for row in rows
    ]


def write_source_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    identifiers = [f"{row['dataset_id']}:{row['clip_id']}" for row in materialized]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("normalized source rows contain duplicate dataset/clip IDs")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in materialized) + "\n")
