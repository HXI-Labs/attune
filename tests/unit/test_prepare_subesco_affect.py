from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts/prepare_subesco_affect.py"
SPEC = importlib.util.spec_from_file_location("prepare_subesco_affect", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SUBESCO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBESCO)


def test_parse_member_handles_pinned_source_filename_anomaly() -> None:
    ordinary = SUBESCO.parse_member("SUBESCO/F_01_OISHI_S_10_ANGRY_5.wav")
    anomaly = SUBESCO.parse_member(SUBESCO.MALFORMED_MEMBER)

    assert ordinary["speaker_id"] == "F_01_OISHI"
    assert ordinary["sentence_id"] == 10
    assert ordinary["source_emotion"] == "ANGRY"
    assert anomaly["canonical_filename"] == "F_02_MONIKA_S_2_SURPRISE_3.wav"
    assert anomaly["source_name_anomaly"] is True


def test_parse_member_rejects_unsafe_or_unknown_names() -> None:
    with pytest.raises(SUBESCO.SubescoPreparationError, match="unsafe"):
        SUBESCO.parse_member("../escape.wav")
    with pytest.raises(SUBESCO.SubescoPreparationError, match="unexpected"):
        SUBESCO.parse_member("SUBESCO/F_01_OISHI_S_1_SHOUTING_1.wav")


def test_speaker_partitions_are_disjoint_and_gender_stratified() -> None:
    speakers = [f"{gender}_{index:02d}_NAME" for gender in ("F", "M") for index in range(1, 11)]
    partitions = SUBESCO.speaker_partitions(speakers)

    assert len(partitions) == 20
    assert sum(value == "train" for value in partitions.values()) == 14
    assert sum(value == "development" for value in partitions.values()) == 2
    assert sum(value == "sealed_test" for value in partitions.values()) == 4
    for gender in ("F", "M"):
        values = [split for speaker, split in partitions.items() if speaker.startswith(gender)]
        assert values.count("train") == 7
        assert values.count("development") == 1
        assert values.count("sealed_test") == 2


def test_affect_map_does_not_create_style_targets() -> None:
    assert SUBESCO._distribution("ANGRY")["anger"] == 1.0
    assert SUBESCO._distribution("SAD")["distress"] == 1.0
    assert "shouting" not in SUBESCO.AFFECT_MAP.values()
