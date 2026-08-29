"""Normalize public-data source rows into checksum-verified joint features."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from attune.training.data import JointManifestRow, TemporalTarget, file_sha256, write_manifest


class SourceRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clip_id: str
    dataset_id: str
    split: str
    speaker_id: str | None
    audio_path: Path
    audio_sha256: str
    duration_ms: int = Field(gt=0)
    transcript: str | None = None
    events: list[TemporalTarget] | None = None
    event_presence: list[str] | None = None
    styles: list[str] | None = None
    affect_distribution: dict[str, float] | None = None
    vad: tuple[float, float, float] | None = None
    pair_id: int = -1
    is_ood: bool = False
    lexical_affect_label: str | None = None
    auxiliary_negative_tasks: list[Literal["event_presence", "styles"]] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def audio_hash_shape(self) -> SourceRow:
        if len(self.audio_sha256) != 64:
            raise ValueError("audio_sha256 must be a SHA-256 hex digest")
        if "event_presence" in self.auxiliary_negative_tasks and self.event_presence != []:
            raise ValueError("event-presence controls require an explicit empty target list")
        if "styles" in self.auxiliary_negative_tasks and self.styles != []:
            raise ValueError("style controls require an explicit empty target list")
        if self.auxiliary_negative_tasks and self.transcript is None:
            raise ValueError("auxiliary negative controls must contain verified speech")
        return self


class Frontend(Protocol):
    tokenizer: Any

    def __call__(self, wav: bytes) -> Any: ...


def load_source_rows(path: Path) -> list[SourceRow]:
    base = path.resolve().parent
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = SourceRow.model_validate_json(line)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
        if not row.audio_path.is_absolute():
            row.audio_path = (base / row.audio_path).resolve()
        rows.append(row)
    _validate_partition_leakage(rows)
    return rows


def _validate_partition_leakage(rows: list[SourceRow]) -> None:
    seen_clips: set[str] = set()
    speakers: dict[tuple[str, str], str] = {}
    for row in rows:
        key = f"{row.dataset_id}:{row.clip_id}"
        if key in seen_clips:
            raise ValueError(f"duplicate source clip: {key}")
        seen_clips.add(key)
        if row.speaker_id is None:
            continue
        speaker_key = (row.dataset_id, row.speaker_id)
        previous = speakers.setdefault(speaker_key, row.split)
        if previous != row.split:
            raise ValueError(
                f"speaker {row.speaker_id!r} from {row.dataset_id} crosses {previous}/{row.split}"
            )


def prepare_joint_features(
    rows: list[SourceRow],
    *,
    frontend: Frontend,
    output_dir: Path,
    manifest_path: Path,
    frame_hop_ms: float = 60.0,
) -> list[JointManifestRow]:
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    for row in rows:
        if file_sha256(row.audio_path) != row.audio_sha256:
            raise ValueError(f"audio hash mismatch for {row.clip_id}")
        features = torch.as_tensor(frontend(row.audio_path.read_bytes()), dtype=torch.float32)
        if features.ndim != 2 or features.shape[0] < 1:
            raise ValueError(f"frontend returned invalid features for {row.clip_id}")
        safe_id = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in row.clip_id
        )
        feature_path = feature_dir / f"{row.dataset_id}-{safe_id}.pt"
        torch.save(features, feature_path)
        token_ids = None
        if row.transcript is not None:
            token_ids = list(frontend.tokenizer.encode(row.transcript))
        prepared.append(
            JointManifestRow(
                clip_id=row.clip_id,
                dataset_id=row.dataset_id,
                split=row.split,
                speaker_id=row.speaker_id,
                feature_path=Path(os.path.relpath(feature_path, manifest_path.parent.resolve())),
                feature_sha256=file_sha256(feature_path),
                duration_ms=row.duration_ms,
                frame_hop_ms=frame_hop_ms,
                transcript=row.transcript,
                token_ids=token_ids,
                events=row.events,
                event_presence=row.event_presence,
                styles=row.styles,
                affect_distribution=row.affect_distribution,
                vad=row.vad,
                pair_id=row.pair_id,
                is_ood=row.is_ood,
                lexical_affect_label=row.lexical_affect_label,
                auxiliary_negative_tasks=row.auxiliary_negative_tasks,
            )
        )
    write_manifest(manifest_path, prepared)
    return prepared


def write_source_template(path: Path) -> None:
    example = {
        "clip_id": "source-stable-id",
        "dataset_id": "dataset-version",
        "split": "train",
        "speaker_id": "speaker-id",
        "audio_path": "audio/clip.wav",
        "audio_sha256": "0" * 64,
        "duration_ms": 1200,
        "transcript": "example transcript",
        "events": None,
        "event_presence": None,
        "styles": None,
        "affect_distribution": None,
        "vad": None,
        "pair_id": -1,
        "is_ood": False,
        "lexical_affect_label": None,
        "auxiliary_negative_tasks": [],
    }
    path.write_text(json.dumps(example, sort_keys=True) + "\n")
