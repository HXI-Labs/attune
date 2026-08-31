"""Versioned, explicit weak-negative controls for auxiliary speech tasks."""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from attune.training.data import JointManifestRow, write_manifest

ControlAction = Literal["explicit_negative", "masked", "preserve_source_annotation"]


class DatasetControlPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_presence: ControlAction
    styles: ControlAction
    localized_events: ControlAction = "preserve_source_annotation"
    split_unit: Literal["speaker", "sentence"] | None = None
    rationale: str | None = None


class AuxiliaryControlPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str
    policy_id: str
    datasets: dict[str, DatasetControlPolicy]
    default: DatasetControlPolicy


def load_control_policy(path: Path) -> AuxiliaryControlPolicy:
    return AuxiliaryControlPolicy.model_validate_json(path.read_text())


def apply_control_policy(row: JointManifestRow, policy: AuxiliaryControlPolicy) -> JointManifestRow:
    selected = policy.datasets.get(row.dataset_id, policy.default)
    updates: dict[str, object] = {}
    if selected.split_unit is not None:
        updates["split_unit"] = selected.split_unit
    negative_tasks = list(row.auxiliary_negative_tasks)
    target_fields = {
        "localized_events": "events",
        "event_presence": "event_presence",
        "styles": "styles",
    }
    for task, field in target_fields.items():
        action = getattr(selected, task)
        if action != "explicit_negative":
            continue
        current = getattr(row, field)
        if current not in (None, []):
            raise ValueError(
                f"{row.dataset_id}:{row.clip_id} already has positive {task} supervision"
            )
        updates[field] = []
        if task not in negative_tasks:
            negative_tasks.append(task)
    if not updates:
        return row
    updates["auxiliary_negative_tasks"] = negative_tasks
    return JointManifestRow.model_validate(row.model_dump() | updates)


def build_control_manifest(
    source: Path,
    destination: Path,
    policy: AuxiliaryControlPolicy,
) -> dict[str, object]:
    rows = [
        JointManifestRow.model_validate_json(line)
        for line in source.read_text().splitlines()
        if line.strip()
    ]
    controlled = []
    for row in rows:
        updated = apply_control_policy(row, policy)
        if not updated.feature_path.is_absolute():
            resolved_feature = (source.resolve().parent / updated.feature_path).resolve()
            updated = updated.model_copy(
                update={
                    "feature_path": Path(
                        os.path.relpath(resolved_feature, destination.resolve().parent)
                    )
                }
            )
        controlled.append(updated)
    counts = Counter(
        (row.dataset_id, task) for row in controlled for task in row.auxiliary_negative_tasks
    )
    write_manifest(destination, controlled)
    return {
        "schema_version": "1.0",
        "policy_id": policy.policy_id,
        "input_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_manifest_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "rows": len(controlled),
        "negative_controls": {
            f"{dataset}:{task}": count for (dataset, task), count in sorted(counts.items())
        },
    }
