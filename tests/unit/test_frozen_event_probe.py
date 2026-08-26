from __future__ import annotations

from pathlib import Path

import pytest

from attune.models.frozen_event_probe import (
    ProbeDataError,
    ProbeExample,
    make_speaker_disjoint_split,
    vocalsound_label,
    vocalsound_speaker_id,
)
from attune.schema.output import EventLabel


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("f0001_0_laughter.wav", EventLabel.LAUGH),
        ("m0042_1_sigh.wav", EventLabel.SIGH),
        ("f0001_2_cough.wav", EventLabel.COUGH),
        ("m0042_3_throatclearing.wav", EventLabel.THROAT_CLEAR),
        ("f0001_4_sneeze.wav", EventLabel.SNEEZE),
    ],
)
def test_vocalsound_label_mapping(filename: str, expected: EventLabel) -> None:
    assert vocalsound_label(filename) == expected


def test_vocalsound_filename_rejects_unmapped_class() -> None:
    with pytest.raises(ProbeDataError, match="unsupported VocalSound label"):
        vocalsound_label("f0001_0_sniff.wav")


def test_vocalsound_speaker_mapping_is_namespaced() -> None:
    assert vocalsound_speaker_id("F0418_0_cough.wav") == "vocalsound:f0418"


def test_split_excludes_all_inspection_speakers_and_is_disjoint() -> None:
    labels = [EventLabel.LAUGH, EventLabel.SIGH, EventLabel.COUGH]
    candidates = [
        ProbeExample(Path(f"{speaker}_{index}_{label.value}.wav"), speaker, label)
        for speaker in (
            "vocalsound:f0001",
            "vocalsound:m0002",
            "vocalsound:f0003",
            "vocalsound:m0004",
        )
        for index, label in enumerate(labels)
    ]
    candidates.append(
        ProbeExample(
            Path("f9999_0_laughter.wav"),
            "vocalsound:f9999",
            EventLabel.LAUGH,
        )
    )
    test = [
        ProbeExample(
            Path("inspection/f9999_0_laughter.wav"),
            "vocalsound:f9999",
            EventLabel.LAUGH,
        )
    ]

    split = make_speaker_disjoint_split(
        candidates,
        test,
        excluded_speakers={"vocalsound:f9999"},
        validation_fraction=0.25,
        seed=7,
    )

    train_speakers = split.speaker_ids("train")
    validation_speakers = split.speaker_ids("validation")
    test_speakers = split.speaker_ids("test")
    assert train_speakers
    assert validation_speakers
    assert train_speakers.isdisjoint(validation_speakers)
    assert (train_speakers | validation_speakers).isdisjoint(test_speakers)
    assert "vocalsound:f9999" not in train_speakers | validation_speakers


def test_split_requires_two_non_inspection_speakers() -> None:
    one = ProbeExample(Path("f0001_0_laughter.wav"), "vocalsound:f0001", EventLabel.LAUGH)
    with pytest.raises(ProbeDataError, match="at least two"):
        make_speaker_disjoint_split(
            [one],
            [],
            excluded_speakers=set(),
        )
