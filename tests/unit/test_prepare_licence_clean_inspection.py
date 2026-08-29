from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[2] / "scripts/prepare_licence_clean_inspection.py"
SPEC = importlib.util.spec_from_file_location("prepare_licence_clean_inspection", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def test_committed_manifest_is_bounded_and_licence_clean() -> None:
    repository = Path(__file__).parents[2]
    rows = PREPARE.load_manifest(repository / "data/manifests/licence-clean-inspection.jsonl")

    assert Counter(row["source_dataset"] for row in rows) == {
        "FSD50K": 100,
        "CREMA-D": 60,
    }
    fsd = [row for row in rows if row["source_dataset"] == "FSD50K"]
    assert Counter(row["intended_attune_labels"]["source_class"] for row in fsd) == {
        "Crying_and_sobbing": 25,
        "Screaming": 25,
        "Shout": 25,
        "Whispering": 25,
    }
    assert {row["licence"]["clip_identifier"] for row in fsd} <= {
        "CC0-1.0",
        "CC-BY-3.0",
    }
    assert all(row["uploader"] in row["attribution"] for row in fsd)
    assert all(row["sample_rate_hz"] == 16_000 for row in rows)
    assert all(row["channels"] == 1 and row["sample_width_bytes"] == 2 for row in rows)
    assert all(len(row["sha256"]) == 64 for row in rows)


def test_weak_label_mappings_do_not_overclaim_speech_or_intensity() -> None:
    repository = Path(__file__).parents[2]
    rows = PREPARE.load_manifest(repository / "data/manifests/licence-clean-inspection.jsonl")
    fsd_by_class = {
        label: [
            row
            for row in rows
            if row["source_dataset"] == "FSD50K"
            and row["intended_attune_labels"]["source_class"] == label
        ]
        for label in PREPARE.FSD_CLASSES
    }

    assert all(
        row["intended_attune_labels"]["events"] == ["sob"]
        and "crying_speech" not in row["intended_attune_labels"]["styles"]
        for row in fsd_by_class["Crying_and_sobbing"]
    )
    assert all(
        row["intended_attune_labels"]["styles"] == ["shouting"] for row in fsd_by_class["Shout"]
    )
    assert all(
        row["intended_attune_labels"]["styles"] == ["whispering"]
        for row in fsd_by_class["Whispering"]
    )
    assert all(
        row["intended_attune_labels"]["events"] == ["scream"]
        and row["intended_attune_labels"]["styles"] == []
        for row in fsd_by_class["Screaming"]
    )
    crema = [row for row in rows if row["source_dataset"] == "CREMA-D"]
    assert all(row["intended_attune_labels"]["styles"] == [] for row in crema)
    assert all("source_intensity" in row["intended_attune_labels"] for row in crema)


def test_crema_expansion_is_disjoint_from_original_inspection_speakers() -> None:
    repository = Path(__file__).parents[2]
    new_rows = PREPARE.load_manifest(repository / "data/manifests/licence-clean-inspection.jsonl")
    old_rows = PREPARE.load_manifest(repository / "data/manifests/inspection-set.jsonl")
    new_speakers = {row["speaker_id"] for row in new_rows if row["source_dataset"] == "CREMA-D"}
    old_speakers = {row["speaker_id"] for row in old_rows if row["source_dataset"] == "CREMA-D"}

    assert len(new_speakers) == 20
    assert new_speakers.isdisjoint(old_speakers)
