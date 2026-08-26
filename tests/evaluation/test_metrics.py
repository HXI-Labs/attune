from types import SimpleNamespace

import pytest

from attune.evaluation.metrics import (
    acoustic_preference_score,
    character_error_rate,
    expected_calibration_error,
    macro_f1,
    multiclass_brier_score,
    position_aware_event_score,
    soft_cross_entropy,
    span_classification_metrics,
    temporal_iou,
    word_error_rate,
)


def span(label: str, start: int, end: int) -> SimpleNamespace:
    return SimpleNamespace(label=label, start_ms=start, end_ms=end)


def test_asr_error_rates() -> None:
    assert word_error_rate("one two three", "one too three") == pytest.approx(1 / 3)
    assert character_error_rate("cat", "cut") == pytest.approx(1 / 3)
    assert word_error_rate("", "") == 0
    assert word_error_rate("", "unexpected") == 1


def test_span_metrics_and_temporal_iou() -> None:
    references = [span("laugh", 100, 300), span("sigh", 500, 600)]
    predictions = [span("laugh", 150, 300), span("cough", 500, 600)]
    assert temporal_iou(references[0], predictions[0]) == pytest.approx(0.75)
    result = span_classification_metrics(references, predictions, labels=["laugh", "sigh", "cough"])
    assert result["per_class"]["laugh"]["f1"] == 1
    assert result["per_class"]["sigh"]["recall"] == 0
    assert result["mean_temporal_iou"] == pytest.approx(0.75)


def test_position_score_uses_time_not_word_anchors() -> None:
    reference = [span("laugh", 400, 500)]
    prediction = [span("laugh", 420, 520)]
    assert position_aware_event_score(reference, prediction, duration_ms=1000) == pytest.approx(0.8)
    assert position_aware_event_score(
        reference,
        [span("laugh", 400, 500), span("laugh", 700, 800)],
        duration_ms=1000,
    ) == pytest.approx(0.5)


def test_affect_metrics_support_soft_gold() -> None:
    labels = ["joy", "anger"]
    references = [{"joy": 0.75, "anger": 0.25}, {"joy": 0.1, "anger": 0.9}]
    predictions = [{"joy": 0.8, "anger": 0.2}, {"joy": 0.2, "anger": 0.8}]
    assert multiclass_brier_score(references, predictions, labels=labels) == pytest.approx(0.0125)
    assert soft_cross_entropy(references, predictions, labels=labels) > 0
    assert macro_f1(["joy", "anger"], ["joy", "anger"], labels=labels) == 1
    assert expected_calibration_error(
        ["joy", "anger"], predictions, labels=labels, bins=5
    ) == pytest.approx(0.2)


def test_aps_handcrafted_semantic_conflicts() -> None:
    result = acoustic_preference_score(
        predictions=["anger", "joy", "neutral"],
        acoustic_targets=["anger", "anger", "distress"],
        lexical_targets=["joy", "joy", "neutral"],
    )
    assert result == {
        "acoustic_accuracy": pytest.approx(1 / 3),
        "lexical_accuracy": pytest.approx(2 / 3),
        "aps": pytest.approx(-1 / 3),
        "conflict_items": 3,
    }
