"""Versioned manifest and collation contract for joint training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor
from torch.utils.data import Dataset

from attune.models.joint import SUPPORTED_EVENTS, SUPPORTED_STYLES
from attune.schema.output import AffectCategory
from attune.training.losses import JointTargets


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TemporalTarget(StrictModel):
    label: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self) -> TemporalTarget:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must not precede start_ms")
        return self


class JointManifestRow(StrictModel):
    schema_version: str = "1.0"
    clip_id: str
    dataset_id: str
    split: str
    speaker_id: str | None
    feature_path: Path
    feature_sha256: str
    duration_ms: int = Field(gt=0)
    frame_hop_ms: float = Field(gt=0)
    transcript: str | None = None
    token_ids: list[int] | None = None
    events: list[TemporalTarget] | None = None
    event_presence: list[str] | None = None
    styles: list[str] | None = None
    affect_distribution: dict[str, float] | None = None
    vad: tuple[float, float, float] | None = None
    pair_id: int = -1
    is_ood: bool = False
    lexical_affect_label: str | None = None
    split_unit: Literal["speaker", "sentence"] = "speaker"
    auxiliary_negative_tasks: list[Literal["localized_events", "event_presence", "styles"]] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def labels_are_supported(self) -> JointManifestRow:
        event_labels = {label.value for label in SUPPORTED_EVENTS}
        style_labels = {label.value for label in SUPPORTED_STYLES}
        if self.events is not None and any(item.label not in event_labels for item in self.events):
            raise ValueError("manifest contains an unsupported event label")
        if self.event_presence is not None and any(
            label not in event_labels for label in self.event_presence
        ):
            raise ValueError("manifest contains an unsupported event-presence label")
        if self.styles is not None and any(label not in style_labels for label in self.styles):
            raise ValueError("manifest contains an unsupported style label")
        if self.affect_distribution is not None:
            if set(self.affect_distribution) != {label.value for label in AffectCategory}:
                raise ValueError("affect_distribution must contain the complete ontology")
            if abs(sum(self.affect_distribution.values()) - 1.0) > 1e-6:
                raise ValueError("affect_distribution must sum to one")
        if self.vad is not None and any(value < -1 or value > 1 for value in self.vad):
            raise ValueError("V/A/D values must be normalized to [-1, 1]")
        if self.lexical_affect_label is not None and self.lexical_affect_label not in {
            label.value for label in AffectCategory
        }:
            raise ValueError("lexical_affect_label is outside the affect ontology")
        if self.split_unit == "sentence" and self.pair_id < 0:
            raise ValueError("sentence-disjoint rows require a non-negative pair_id")
        if "event_presence" in self.auxiliary_negative_tasks and self.event_presence != []:
            raise ValueError("event-presence controls require an explicit empty target list")
        if "localized_events" in self.auxiliary_negative_tasks and self.events != []:
            raise ValueError("localized-event controls require an explicit empty target list")
        if "styles" in self.auxiliary_negative_tasks and self.styles != []:
            raise ValueError("style controls require an explicit empty target list")
        if self.auxiliary_negative_tasks and self.transcript is None:
            raise ValueError("auxiliary negative controls must contain verified speech")
        return self


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class JointFeatureDataset(Dataset[tuple[JointManifestRow, Tensor]]):
    """Load checksum-verified frontend features without arbitrary pickle objects."""

    _TARGET_FIELDS = {
        "affect": "affect_distribution",
        "ctc": "token_ids",
        "event_presence": "event_presence",
        "events": "events",
        "styles": "styles",
        "vad": "vad",
    }

    def __init__(self, manifest: Path, *, split: str, training_target: str | None = None) -> None:
        if training_target is not None and training_target not in self._TARGET_FIELDS:
            raise ValueError(f"unsupported training target: {training_target}")
        self.manifest = manifest.resolve()
        base = self.manifest.parent
        rows = []
        for line_number, line in enumerate(self.manifest.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = JointManifestRow.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"{manifest}:{line_number}: {error}") from error
            if row.split != split:
                continue
            if training_target is not None:
                target_field = self._TARGET_FIELDS[training_target]
                if getattr(row, target_field) is None:
                    continue
            if not row.feature_path.is_absolute():
                row.feature_path = (base / row.feature_path).resolve()
            rows.append(row)
        if not rows:
            target = f" with {training_target!r} targets" if training_target else ""
            raise ValueError(f"manifest contains no {split!r} rows{target}")
        identifiers = [row.clip_id for row in rows]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"duplicate clip IDs in {split!r} partition")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[JointManifestRow, Tensor]:
        row = self.rows[index]
        if file_sha256(row.feature_path) != row.feature_sha256:
            raise ValueError(f"feature hash mismatch for {row.clip_id}")
        value = torch.load(row.feature_path, map_location="cpu", weights_only=True)
        if not isinstance(value, Tensor) or value.ndim != 2:
            raise ValueError(f"{row.feature_path} must contain one (time, feature) tensor")
        return row, value.float()


@dataclass(frozen=True)
class JointBatch:
    clip_ids: tuple[str, ...]
    speech: Tensor
    speech_lengths: Tensor
    targets: JointTargets

    def to(self, device: torch.device) -> JointBatch:
        values: dict[str, Any] = {}
        for name, value in vars(self.targets).items():
            values[name] = value.to(device) if isinstance(value, Tensor) else value
        return JointBatch(
            clip_ids=self.clip_ids,
            speech=self.speech.to(device),
            speech_lengths=self.speech_lengths.to(device),
            targets=JointTargets(**values),
        )


def collate_joint_examples(examples: list[tuple[JointManifestRow, Tensor]]) -> JointBatch:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    rows, features = zip(*examples, strict=True)
    feature_size = features[0].shape[1]
    if any(item.shape[1] != feature_size for item in features):
        raise ValueError("feature dimensions differ within batch")
    maximum_frames = max(item.shape[0] for item in features)
    speech = torch.zeros(len(features), maximum_frames, feature_size)
    lengths = torch.tensor([item.shape[0] for item in features], dtype=torch.long)
    for index, item in enumerate(features):
        speech[index, : item.shape[0]] = item

    event_count = len(SUPPORTED_EVENTS)
    event_targets = torch.zeros(len(rows), maximum_frames, event_count)
    event_mask = torch.zeros_like(event_targets, dtype=torch.bool)
    starts = torch.zeros_like(event_targets)
    ends = torch.zeros_like(event_targets)
    event_index = {label.value: index for index, label in enumerate(SUPPORTED_EVENTS)}
    for batch_index, row in enumerate(rows):
        if row.events is None:
            continue
        event_mask[batch_index, : lengths[batch_index]] = True
        for event in row.events:
            start = min(int(event.start_ms / row.frame_hop_ms), lengths[batch_index] - 1)
            stop = min(
                max(start + 1, int(event.end_ms / row.frame_hop_ms) + 1), lengths[batch_index]
            )
            label_index = event_index[event.label]
            event_targets[batch_index, start:stop, label_index] = 1.0
            starts[batch_index, start, label_index] = 1.0
            ends[batch_index, stop - 1, label_index] = 1.0

    event_presence_targets = torch.zeros(len(rows), event_count)
    event_presence_mask = torch.zeros_like(event_presence_targets, dtype=torch.bool)
    for batch_index, row in enumerate(rows):
        if row.event_presence is None:
            continue
        event_presence_mask[batch_index] = True
        for label in row.event_presence:
            event_presence_targets[batch_index, event_index[label]] = 1.0

    style_targets = torch.zeros(len(rows), len(SUPPORTED_STYLES))
    style_mask = torch.zeros_like(style_targets, dtype=torch.bool)
    style_index = {label.value: index for index, label in enumerate(SUPPORTED_STYLES)}
    for batch_index, row in enumerate(rows):
        if row.styles is None:
            continue
        style_mask[batch_index] = True
        for label in row.styles:
            style_targets[batch_index, style_index[label]] = 1.0

    category_order = tuple(AffectCategory)
    affect = torch.zeros(len(rows), len(category_order))
    affect_mask = torch.zeros(len(rows), dtype=torch.bool)
    vad = torch.zeros(len(rows), 3)
    vad_mask = torch.zeros_like(vad, dtype=torch.bool)
    for batch_index, row in enumerate(rows):
        if row.affect_distribution is not None:
            affect[batch_index] = torch.tensor(
                [row.affect_distribution[label.value] for label in category_order]
            )
            affect_mask[batch_index] = True
        if row.vad is not None:
            vad[batch_index] = torch.tensor(row.vad)
            vad_mask[batch_index] = True

    token_rows = [row.token_ids for row in rows]
    ctc_targets = None
    ctc_lengths = None
    ctc_mask = torch.tensor([tokens is not None for tokens in token_rows], dtype=torch.bool)
    if ctc_mask.any():
        active_rows = [tokens for tokens in token_rows if tokens is not None]
        ctc_lengths = torch.tensor([len(tokens) for tokens in active_rows], dtype=torch.long)
        ctc_targets = torch.tensor(
            [token for tokens in active_rows for token in tokens], dtype=torch.long
        )

    return JointBatch(
        clip_ids=tuple(row.clip_id for row in rows),
        speech=speech,
        speech_lengths=lengths,
        targets=JointTargets(
            ctc_targets=ctc_targets,
            ctc_target_lengths=ctc_lengths,
            ctc_example_mask=ctc_mask,
            event_targets=event_targets,
            event_target_mask=event_mask,
            event_start_targets=starts,
            event_end_targets=ends,
            event_presence_targets=event_presence_targets,
            event_presence_example_mask=event_presence_mask,
            style_targets=style_targets,
            style_example_mask=style_mask,
            affect_distribution=affect,
            affect_example_mask=affect_mask,
            vad_targets=vad,
            vad_target_mask=vad_mask,
            ood_targets=torch.tensor([row.is_ood for row in rows], dtype=torch.float32),
            pair_ids=torch.tensor([row.pair_id for row in rows]),
        ),
    )


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(path: Path, rows: list[JointManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row.model_dump(mode="json"), sort_keys=True) for row in rows)
    path.write_text(payload + "\n")
