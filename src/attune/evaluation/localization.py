"""Dependency-free metrics for bounded event localization diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def segment_f1(
    references: Sequence[Sequence[dict[str, Any]]],
    predictions: Sequence[Sequence[dict[str, Any]]],
    *,
    duration_ms: int,
    segment_ms: int = 1000,
) -> dict[str, float | int]:
    """Micro F1 over label presence in fixed one-second segments."""
    if len(references) != len(predictions):
        raise ValueError("reference and prediction clip counts differ")
    true_positive = false_positive = false_negative = 0
    for reference, prediction in zip(references, predictions, strict=True):
        for start in range(0, duration_ms, segment_ms):
            end = min(start + segment_ms, duration_ms)
            expected = {
                span["label"]
                for span in reference
                if span["end_ms"] > start and span["start_ms"] < end
            }
            predicted = {
                span["label"]
                for span in prediction
                if span["end_ms"] > start and span["start_ms"] < end
            }
            true_positive += len(expected & predicted)
            false_positive += len(predicted - expected)
            false_negative += len(expected - predicted)
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "f1": 2 * true_positive / denominator if denominator else 0.0,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "segment_ms": segment_ms,
    }


def collar_event_metrics(
    references: Sequence[Sequence[dict[str, Any]]],
    predictions: Sequence[Sequence[dict[str, Any]]],
    *,
    onset_collar_ms: int = 200,
    offset_collar_ms: int = 200,
    offset_duration_ratio: float = 0.2,
) -> dict[str, float | int | None]:
    """Greedy event F1 with DCASE-style onset and duration-aware offset collars."""
    if len(references) != len(predictions):
        raise ValueError("reference and prediction clip counts differ")
    matches = false_positive = false_negative = 0
    onset_errors: list[int] = []
    offset_errors: list[int] = []
    for reference, prediction in zip(references, predictions, strict=True):
        unmatched = set(range(len(reference)))
        for candidate in prediction:
            eligible = []
            for index in unmatched:
                target = reference[index]
                duration = target["end_ms"] - target["start_ms"]
                offset_collar = max(offset_collar_ms, round(duration * offset_duration_ratio))
                onset_error = abs(candidate["start_ms"] - target["start_ms"])
                offset_error = abs(candidate["end_ms"] - target["end_ms"])
                if (
                    candidate["label"] == target["label"]
                    and onset_error <= onset_collar_ms
                    and offset_error <= offset_collar
                ):
                    eligible.append((onset_error + offset_error, index))
            if not eligible:
                false_positive += 1
                continue
            _, matched = min(eligible)
            unmatched.remove(matched)
            matches += 1
            onset_errors.append(abs(candidate["start_ms"] - reference[matched]["start_ms"]))
            offset_errors.append(abs(candidate["end_ms"] - reference[matched]["end_ms"]))
        false_negative += len(unmatched)
    denominator = 2 * matches + false_positive + false_negative
    return {
        "f1": 2 * matches / denominator if denominator else 0.0,
        "true_positive": matches,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "onset_collar_ms": onset_collar_ms,
        "offset_collar_ms": offset_collar_ms,
        "offset_duration_ratio": offset_duration_ratio,
        "matched_onset_mae_ms": (sum(onset_errors) / len(onset_errors) if onset_errors else None),
        "matched_offset_mae_ms": (
            sum(offset_errors) / len(offset_errors) if offset_errors else None
        ),
    }


def whole_clip_predictions(
    references: Sequence[Sequence[dict[str, Any]]],
    *,
    duration_ms: int,
) -> list[list[dict[str, Any]]]:
    """Oracle clip tags materialized as the deliberately non-localizing baseline."""
    return [
        [
            {"label": label, "start_ms": 0, "end_ms": duration_ms}
            for label in sorted({span["label"] for span in clip})
        ]
        for clip in references
    ]
