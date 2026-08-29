from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[2] / "scripts/merge_joint_manifests.py"
SPEC = importlib.util.spec_from_file_location("merge_joint_manifests", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MERGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGE)


def manifest(path: Path, feature: Path, clip_id: str, dataset: str) -> None:
    digest = hashlib.sha256(feature.read_bytes()).hexdigest()
    path.write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "dataset_id": dataset,
                "split": "train",
                "speaker_id": f"{dataset}-speaker",
                "feature_path": feature.name,
                "feature_sha256": digest,
                "duration_ms": 1000,
                "frame_hop_ms": 60.0,
            }
        )
        + "\n"
    )


def test_merge_preserves_rows_and_rewrites_feature_paths(tmp_path: Path) -> None:
    first_feature = tmp_path / "first.pt"
    second_feature = tmp_path / "second.pt"
    first_feature.write_bytes(b"first")
    second_feature.write_bytes(b"second")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    manifest(first, first_feature, "one", "first")
    manifest(second, second_feature, "two", "second")
    output = tmp_path / "merged" / "joint.jsonl"

    rows = MERGE.merge_manifests([first, second], output)

    assert [row.clip_id for row in rows] == ["one", "two"]
    assert (output.parent / rows[0].feature_path).resolve() == first_feature.resolve()
    assert output.with_suffix(".provenance.json").is_file()
