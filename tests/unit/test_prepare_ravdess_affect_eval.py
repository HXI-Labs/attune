from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts/prepare_ravdess_affect_eval.py"
SPEC = importlib.util.spec_from_file_location("prepare_ravdess_affect_eval", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
RAVDESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RAVDESS)


def test_protocol_is_fixed_actor_disjoint_same_text_affect_evaluation() -> None:
    rows = RAVDESS.protocol_rows()

    assert len(rows) == 480
    assert {row["actor_id"] for row in rows} == set(range(17, 25))
    assert all(sum(row["actor_id"] == actor for row in rows) == 60 for actor in range(17, 25))
    assert {row["transcript"] for row in rows} == set(RAVDESS.STATEMENTS.values())
    assert {row["target_affect"] for row in rows} == {
        "neutral",
        "joy",
        "distress",
        "anger",
        "fear",
        "surprise",
        "other",
    }
    strong_anger = [
        row for row in rows if row["source_emotion"] == "angry" and row["intensity"] == "strong"
    ]
    assert len(strong_anger) == 32


def test_protocol_rejects_invalid_or_duplicate_actor_selection() -> None:
    with pytest.raises(RAVDESS.RavdessPreparationError, match="actors must be unique"):
        RAVDESS.protocol_rows((17, 17))
    with pytest.raises(RAVDESS.RavdessPreparationError, match="actors must be unique"):
        RAVDESS.protocol_rows((25,))


def test_archive_member_validation_rejects_traversal() -> None:
    assert (
        str(RAVDESS._validate_member("Actor_17/03-01-05-02-01-01-17.wav"))
        == "Actor_17/03-01-05-02-01-01-17.wav"
    )
    with pytest.raises(RAVDESS.RavdessPreparationError, match="unsafe or unexpected"):
        RAVDESS._validate_member("../03-01-05-02-01-01-17.wav")
