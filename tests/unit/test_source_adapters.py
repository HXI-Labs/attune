from __future__ import annotations

from pathlib import Path

from attune.training.source_adapters import (
    adapt_common_voice,
    adapt_crema,
    adapt_dcase,
    adapt_fsd50k,
)


def test_dcase_preserves_strong_timing_and_derives_presence() -> None:
    row = {
        "clip_id": "d1",
        "cache_path": "development/d1.wav",
        "sha256": "a" * 64,
        "duration_ms": 10000,
        "source_recording": "room-1.wav",
        "events": [{"label": "laugh", "start_ms": 200, "end_ms": 700, "source_label": "laughter"}],
    }
    output = adapt_dcase([row], cache_root=Path("/audio"))[0]
    assert output["events"] == [{"label": "laugh", "start_ms": 200, "end_ms": 700}]
    assert output["event_presence"] == ["laugh"]
    assert output["is_ood"] is True
    assert output["split"] in {"train", "development"}


def test_fsd_weak_labels_do_not_become_fake_spans() -> None:
    rows = [
        {
            "clip_id": "sob-1",
            "cache_path": "sob.wav",
            "sha256": "b" * 64,
            "duration_s": 2.5,
            "partition": "train",
            "probe_label": "sob",
        },
        {
            "clip_id": "shout-1",
            "cache_path": "shout.wav",
            "sha256": "c" * 64,
            "duration_s": 1.0,
            "partition": "validation",
            "probe_label": "shout",
        },
    ]
    sob, shout = adapt_fsd50k(rows, cache_root=Path("/audio"))
    assert sob["events"] is None and sob["event_presence"] == ["sob"]
    assert shout["styles"] == ["shouting"] and shout["split"] == "development"


def test_fsd_validation_is_label_stratified_between_development_and_sealed() -> None:
    rows = [
        {
            "clip_id": f"shout-{index}",
            "cache_path": f"shout-{index}.wav",
            "sha256": "c" * 64,
            "duration_s": 1.0,
            "partition": "validation",
            "probe_label": "shout",
        }
        for index in range(4)
    ]

    output = adapt_fsd50k(rows, cache_root=Path("/audio"))

    assert [row["split"] for row in output] == [
        "development",
        "sealed_test",
        "development",
        "sealed_test",
    ]


def test_crema_pair_keeps_source_label_and_neutral_lexical_control() -> None:
    row = {
        "clip_id": "c1",
        "source_filename": "1081_IEO_ANG_HI.wav",
        "cache_path": "c1.wav",
        "sha256": "d" * 64,
        "duration_ms": 1200,
        "partition": "development",
        "speaker_id": "crema-d:1081",
        "target_affect": "anger",
    }
    output = adapt_crema([row], cache_root=Path("/audio"))[0]
    assert output["affect_distribution"]["anger"] == 1.0
    assert output["lexical_affect_label"] == "neutral"
    assert output["transcript"] == "It's eleven o'clock."


def test_v2_speech_controls_are_explicit_and_opt_in() -> None:
    row = {
        "clip_id": "cv-1",
        "cache_path": "cv-1.wav",
        "sha256": "e" * 64,
        "duration_s": 1.2,
        "partition": "train",
        "client_id": "speaker-1",
        "transcript": "ordinary read speech",
    }

    legacy = adapt_common_voice([row], cache_root=Path("/audio"), split=None)[0]
    controlled = adapt_common_voice(
        [row],
        cache_root=Path("/audio"),
        split=None,
        auxiliary_negative_controls=True,
    )[0]

    assert legacy["event_presence"] is None and legacy["styles"] is None
    assert controlled["event_presence"] == [] and controlled["styles"] == []
    assert controlled["auxiliary_negative_tasks"] == ["event_presence", "styles"]
