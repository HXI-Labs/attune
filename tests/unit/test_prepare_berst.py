from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts/prepare_berst.py"
SPEC = importlib.util.spec_from_file_location("prepare_berst", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BERST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BERST)


def test_berst_affect_mapping_preserves_attune_inventory() -> None:
    assert BERST.affect_distribution("sadness")["distress"] == 1.0
    assert BERST.affect_distribution("disgust")["other"] == 1.0
    assert BERST.affect_distribution("anger")["anger"] == 1.0
    assert sum(BERST.affect_distribution("surprise").values()) == 1.0


def test_berst_shout_labels_do_not_invent_unknown_style_targets() -> None:
    assert BERST.style_target("shout") == (["shouting"], [])
    assert BERST.style_target("no-shout") == ([], ["styles"])
    assert BERST.style_target("n/a") == (None, [])


def test_berst_clip_identity_uses_raw_recording_and_chunk() -> None:
    assert BERST.clip_id("shout_data_1234.wav", "chunk21.wav") == "berst-shout_data_1234-chunk21"


def test_berst_clip_identity_rejects_unsafe_embedded_paths() -> None:
    with pytest.raises(BERST.BerstPreparationError, match="unsafe embedded audio path"):
        BERST.clip_id("shout_data_1234.wav", "../chunk21.wav")


def test_berst_duplicate_audio_is_excluded_without_choosing_a_label() -> None:
    rows = [
        {
            "source_shard": "train.parquet",
            "source_row_index": 0,
            "affect": "anger",
            "shout_level": "shout",
            "script": "baba",
            "user_id": "speaker-1",
            "split": "train",
        },
        {
            "source_shard": "test.parquet",
            "source_row_index": 0,
            "affect": "neutral",
            "shout_level": "no-shout",
            "script": "dada",
            "user_id": "speaker-2",
            "split": "sealed_test",
        },
        {
            "source_shard": "train.parquet",
            "source_row_index": 1,
            "affect": "joy",
            "shout_level": "no-shout",
            "script": "mama",
            "user_id": "speaker-1",
            "split": "train",
        },
    ]
    hashes = {
        ("train.parquet", 0): "duplicate",
        ("test.parquet", 0): "duplicate",
        ("train.parquet", 1): "unique",
    }

    retained, report = BERST.exclude_duplicate_audio(rows, hashes)

    assert [row["source_audio_sha256"] for row in retained] == ["unique"]
    assert report == {
        "source_unique_audio": 2,
        "duplicate_audio_groups": 1,
        "duplicate_audio_rows": 2,
        "conflicting_affect_groups": 1,
        "conflicting_style_groups": 1,
        "conflicting_transcript_groups": 1,
        "cross_speaker_groups": 1,
        "cross_split_groups": 1,
        "retained_unique_audio": 1,
    }
