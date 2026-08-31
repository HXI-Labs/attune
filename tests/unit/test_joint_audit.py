from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from attune.training.audit import audit_joint_manifest


def _row(tmp_path: Path, split: str, speaker: str) -> dict[str, object]:
    feature = tmp_path / f"{split}.pt"
    torch.save(torch.ones(3, 8), feature)
    return {
        "clip_id": split,
        "dataset_id": "fixture",
        "split": split,
        "speaker_id": speaker,
        "feature_path": feature.name,
        "feature_sha256": hashlib.sha256(feature.read_bytes()).hexdigest(),
        "duration_ms": 180,
        "frame_hop_ms": 60,
    }


def test_audit_verifies_features_and_speaker_isolation(tmp_path: Path) -> None:
    rows = [
        _row(tmp_path, "train", "speaker-a"),
        _row(tmp_path, "development", "speaker-b"),
        _row(tmp_path, "sealed_test", "speaker-c"),
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    report = audit_joint_manifest(
        manifest, expected_feature_size=8, require_evaluation_supervision=False
    )

    assert report["rows"] == 3
    assert report["feature_sizes"] == {"8": 3}
    assert not any(report["known_speaker_overlap"].values())


def test_audit_rejects_cross_split_speaker(tmp_path: Path) -> None:
    rows = [
        _row(tmp_path, "train", "same"),
        _row(tmp_path, "development", "same"),
        _row(tmp_path, "sealed_test", "other"),
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ValueError, match="known speakers"):
        audit_joint_manifest(
            manifest, expected_feature_size=8, require_evaluation_supervision=False
        )


def test_audit_accepts_sentence_disjoint_single_speaker_corpus(tmp_path: Path) -> None:
    rows = [
        _row(tmp_path, "train", "same"),
        _row(tmp_path, "development", "same"),
        _row(tmp_path, "sealed_test", "same"),
    ]
    for pair_id, row in enumerate(rows):
        row.update({"split_unit": "sentence", "pair_id": pair_id})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    report = audit_joint_manifest(
        manifest, expected_feature_size=8, require_evaluation_supervision=False
    )

    assert report["sentence_disjoint_pairs"] == 3
    assert not any(report["known_speaker_overlap"].values())


def test_audit_rejects_sentence_pair_crossing_partitions(tmp_path: Path) -> None:
    rows = [
        _row(tmp_path, "train", "same"),
        _row(tmp_path, "development", "same"),
        _row(tmp_path, "sealed_test", "same"),
    ]
    for row in rows:
        row.update({"split_unit": "sentence", "pair_id": 7})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ValueError, match="sentence pairs"):
        audit_joint_manifest(
            manifest, expected_feature_size=8, require_evaluation_supervision=False
        )
