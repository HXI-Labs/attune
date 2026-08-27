from attune.evaluation.localization import (
    collar_event_metrics,
    segment_f1,
    whole_clip_predictions,
)


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
