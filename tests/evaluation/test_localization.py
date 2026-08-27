import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from attune.evaluation.localization import (
    collar_event_metrics,
    segment_f1,
    whole_clip_predictions,
)


def load_training_script() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "train_temporal_localization.py"
    spec = importlib.util.spec_from_file_location("temporal_localization_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_spans_have_perfect_segment_and_collar_f1() -> None:
    references = [[{"label": "cough", "start_ms": 1000, "end_ms": 2000}]]

    assert segment_f1(references, references, duration_ms=4000)["f1"] == 1.0
    assert collar_event_metrics(references, references)["f1"] == 1.0


def test_whole_clip_baseline_is_explicitly_non_localizing() -> None:
    references = [[{"label": "laugh", "start_ms": 2000, "end_ms": 3000}]]
    baseline = whole_clip_predictions(references, duration_ms=10_000)

    assert baseline == [[{"label": "laugh", "start_ms": 0, "end_ms": 10_000}]]
    assert collar_event_metrics(references, baseline)["f1"] == 0.0
    assert segment_f1(references, baseline, duration_ms=10_000)["f1"] < 1.0


def test_collar_matching_keeps_labels_separate() -> None:
    references = [[{"label": "cough", "start_ms": 1000, "end_ms": 2000}]]
    predictions = [[{"label": "laugh", "start_ms": 1000, "end_ms": 2000}]]

    result = collar_event_metrics(references, predictions)

    assert result["true_positive"] == 0
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1


def test_frame_decoder_uses_approximate_lfr_boundaries() -> None:
    script = load_training_script()

    spans = script.frame_spans(
        [[[0.9, 0.1, 0.1], [0.9, 0.1, 0.1], [0.1, 0.1, 0.1]]],
        0.5,
        first_frame_center_ms=30.0,
        frame_hop_ms=60.0,
        duration_ms=180,
    )

    assert spans == [[{"label": "laugh", "start_ms": 0, "end_ms": 120}]]


def test_frame_targets_use_strong_interval_overlap() -> None:
    torch = pytest.importorskip("torch")
    script = load_training_script()
    rows = [
        {
            "duration_ms": 180,
            "events": [{"label": "cough", "start_ms": 65, "end_ms": 85}],
        }
    ]

    targets = script.frame_targets(
        rows,
        [3],
        first_frame_center_ms=30.0,
        frame_hop_ms=60.0,
        torch=torch,
    )

    assert targets[0].tolist() == [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ]


def test_cascade_wiring_requires_clear_gain_over_whole_clip() -> None:
    script = load_training_script()

    assert script.should_wire_timestamps(0.37, 0.3183) is True
    assert script.should_wire_timestamps(0.36, 0.3183) is False
    assert script.should_wire_timestamps(0.20, 0.3183) is False


def test_development_split_holds_out_complete_scenes() -> None:
    script = load_training_script()
    rows = [
        {"source_recording": "dev_1_ebr_-6_nec_1_poly_0.wav"},
        {"source_recording": "dev_1_ebr_-6_nec_1_poly_1.wav"},
        {"source_recording": "dev_1_ebr_-6_nec_2_poly_0.wav"},
        {"source_recording": "dev_1_ebr_-6_nec_2_poly_1.wav"},
    ]

    train, validation, metadata = script.development_split(rows, validation_scene_count=1)

    assert {row["source_recording"] for row in train} == {
        "dev_1_ebr_-6_nec_1_poly_0.wav",
        "dev_1_ebr_-6_nec_1_poly_1.wav",
    }
    assert {row["source_recording"] for row in validation} == {
        "dev_1_ebr_-6_nec_2_poly_0.wav",
        "dev_1_ebr_-6_nec_2_poly_1.wav",
    }
    assert metadata["validation_scene_disjoint"] is True
