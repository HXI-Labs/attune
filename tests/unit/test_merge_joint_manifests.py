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


def manifest(
    path: Path,
    feature: Path,
    clip_id: str,
    dataset: str,
    *,
    duration_ms: int = 1000,
) -> None:
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
                "duration_ms": duration_ms,
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


def test_merge_filters_out_of_scope_durations_and_records_provenance(tmp_path: Path) -> None:
    short_feature = tmp_path / "short.pt"
    valid_feature = tmp_path / "valid.pt"
    long_feature = tmp_path / "long.pt"
    for feature in (short_feature, valid_feature, long_feature):
        feature.write_bytes(feature.stem.encode())
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    manifest(first, short_feature, "short", "first", duration_ms=400)
    manifest(second, valid_feature, "valid", "second", duration_ms=1000)
    with second.open("a") as handle:
        digest = hashlib.sha256(long_feature.read_bytes()).hexdigest()
        handle.write(
            json.dumps(
                {
                    "clip_id": "long",
                    "dataset_id": "second",
                    "split": "train",
                    "speaker_id": "second-long-speaker",
                    "feature_path": long_feature.name,
                    "feature_sha256": digest,
                    "duration_ms": 31000,
                    "frame_hop_ms": 60.0,
                }
            )
            + "\n"
        )
    output = tmp_path / "merged" / "joint.jsonl"

    rows = MERGE.merge_manifests(
        [first, second],
        output,
        minimum_duration_ms=500,
        maximum_duration_ms=30000,
    )
    provenance = json.loads(output.with_suffix(".provenance.json").read_text())

    assert [row.clip_id for row in rows] == ["valid"]
    assert provenance["duration_filter"]["excluded_count"] == 2
    assert {row["reason"] for row in provenance["duration_filter"]["excluded_rows"]} == {
        "below_minimum_duration",
        "above_maximum_duration",
    }


def test_merge_can_cap_training_duration_without_removing_long_evaluation_rows(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "merged.jsonl"
    train_feature = tmp_path / "train.pt"
    test_feature = tmp_path / "test.pt"
    train_feature.write_bytes(b"train")
    test_feature.write_bytes(b"test")
    manifest(first, train_feature, "long-train", "first", duration_ms=12000)
    row = json.loads(first.read_text())
    row["clip_id"] = "long-test"
    row["dataset_id"] = "second"
    row["split"] = "sealed_test"
    row["feature_path"] = str(test_feature)
    row["feature_sha256"] = MERGE.file_digest(test_feature)
    second.write_text(json.dumps(row) + "\n")

    rows = MERGE.merge_manifests(
        [first, second],
        output,
        training_maximum_duration_ms=10000,
    )

    assert [row.clip_id for row in rows] == ["long-test"]
    provenance = json.loads(output.with_suffix(".provenance.json").read_text())
    assert provenance["duration_filter"]["training_maximum_duration_ms"] == 10000
    assert provenance["duration_filter"]["excluded_rows"][0]["reason"] == (
        "above_training_maximum_duration"
    )


def test_merge_can_replace_a_dataset_without_rewriting_the_base_manifest(
    tmp_path: Path,
) -> None:
    old_feature = tmp_path / "old.pt"
    retained_feature = tmp_path / "retained.pt"
    replacement_feature = tmp_path / "replacement.pt"
    old_feature.write_bytes(b"old")
    retained_feature.write_bytes(b"retained")
    replacement_feature.write_bytes(b"replacement")
    base = tmp_path / "base.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    manifest(base, old_feature, "old", "replace-me")
    retained = tmp_path / "retained.jsonl"
    manifest(retained, retained_feature, "retained", "keep-me")
    base.write_text(base.read_text() + retained.read_text())
    manifest(replacement, replacement_feature, "new", "replacement")
    output = tmp_path / "merged.jsonl"

    rows = MERGE.merge_manifests(
        [base, replacement],
        output,
        excluded_datasets={"replace-me"},
    )

    assert [(row.dataset_id, row.clip_id) for row in rows] == [
        ("keep-me", "retained"),
        ("replacement", "new"),
    ]
    provenance = json.loads(output.with_suffix(".provenance.json").read_text())
    assert provenance["dataset_filter"]["excluded_datasets"] == ["replace-me"]
    assert provenance["dataset_filter"]["excluded_rows"][0]["reason"] == ("excluded_dataset")


def test_merge_applies_the_declared_split_unit(tmp_path: Path) -> None:
    first_feature = tmp_path / "first.pt"
    second_feature = tmp_path / "second.pt"
    first_feature.write_bytes(b"first")
    second_feature.write_bytes(b"second")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    manifest(first, first_feature, "one", "sentence-disjoint")
    first_row = json.loads(first.read_text())
    first_row.update({"split_unit": "sentence", "pair_id": 1, "speaker_id": "same-speaker"})
    first.write_text(json.dumps(first_row) + "\n")
    manifest(second, second_feature, "two", "sentence-disjoint")
    second_row = json.loads(second.read_text())
    second_row.update(
        {
            "split": "development",
            "split_unit": "sentence",
            "pair_id": 2,
            "speaker_id": "same-speaker",
        }
    )
    second.write_text(json.dumps(second_row) + "\n")

    rows = MERGE.merge_manifests([first, second], tmp_path / "merged.jsonl")

    assert [row.split for row in rows] == ["train", "development"]
