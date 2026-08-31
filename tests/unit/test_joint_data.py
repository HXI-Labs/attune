from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from attune.training.data import JointFeatureDataset, JointManifestRow, collate_joint_examples


def _row(path: Path, digest: str, clip_id: str, split: str) -> dict:
    categories = {
        "neutral": 0.7,
        "joy": 0.1,
        "distress": 0.05,
        "anger": 0.05,
        "fear": 0.025,
        "surprise": 0.025,
        "other": 0.025,
        "ambiguous": 0.025,
    }
    return {
        "schema_version": "1.0",
        "clip_id": clip_id,
        "dataset_id": "fixture",
        "split": split,
        "speaker_id": clip_id,
        "feature_path": path.name,
        "feature_sha256": digest,
        "duration_ms": 600,
        "frame_hop_ms": 60,
        "token_ids": [1, 2],
        "events": [{"label": "laugh", "start_ms": 120, "end_ms": 300}],
        "event_presence": ["laugh", "sigh"],
        "styles": ["shouting"],
        "affect_distribution": categories,
        "vad": [-0.2, 0.5, 0.1],
        "pair_id": 1,
    }


def test_manifest_hash_verification_and_collation(tmp_path: Path) -> None:
    feature = tmp_path / "feature.pt"
    torch.save(torch.randn(10, 80), feature)
    digest = hashlib.sha256(feature.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        _row(feature, digest, "train-1", "train"),
        _row(feature, digest, "dev-1", "development"),
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    dataset = JointFeatureDataset(manifest, split="train")
    batch = collate_joint_examples([dataset[0]])

    assert batch.speech.shape == (1, 10, 80)
    assert batch.targets.event_targets[0, 2:6, 0].sum() == 4
    assert batch.targets.event_presence_targets[0].sum() == 2
    assert batch.targets.affect_example_mask.tolist() == [True]


def test_explicit_speech_controls_supervise_all_negative_auxiliary_labels(
    tmp_path: Path,
) -> None:
    feature = tmp_path / "feature.pt"
    torch.save(torch.randn(10, 80), feature)
    digest = hashlib.sha256(feature.read_bytes()).hexdigest()
    row = _row(feature, digest, "control-1", "train")
    row.update(
        {
            "transcript": "ordinary read speech",
            "events": [],
            "event_presence": [],
            "styles": [],
            "auxiliary_negative_tasks": [
                "localized_events",
                "event_presence",
                "styles",
            ],
        }
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n")

    batch = collate_joint_examples([JointFeatureDataset(manifest, split="train")[0]])

    assert batch.targets.event_target_mask.all()
    assert batch.targets.event_targets.sum() == 0
    assert batch.targets.event_presence_example_mask.all()
    assert batch.targets.event_presence_targets.sum() == 0
    assert batch.targets.style_example_mask.all()
    assert batch.targets.style_targets.sum() == 0


def test_pair_ids_are_scoped_to_the_source_dataset(tmp_path: Path) -> None:
    feature = torch.randn(10, 80)
    rows = []
    for dataset_id in ("first", "second"):
        values = _row(tmp_path / "feature.pt", "unused", dataset_id, "train")
        values["dataset_id"] = dataset_id
        rows.append(JointManifestRow.model_validate(values))

    batch = collate_joint_examples([(row, feature) for row in rows])

    assert batch.targets.pair_ids.tolist() == [0, 1]


def test_matching_dataset_pair_ids_remain_linked(tmp_path: Path) -> None:
    feature = torch.randn(10, 80)
    rows = []
    for clip_id in ("first", "second"):
        values = _row(tmp_path / "feature.pt", "unused", clip_id, "train")
        rows.append(JointManifestRow.model_validate(values))

    batch = collate_joint_examples([(row, feature) for row in rows])

    assert batch.targets.pair_ids.tolist() == [0, 0]
