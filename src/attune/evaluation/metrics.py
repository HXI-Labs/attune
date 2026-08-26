"""Dependency-free metrics for Phase 1 baseline evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol


class LabeledSpan(Protocol):
    """Structural type shared by schema events and styles."""

    label: object
    start_ms: int
    end_ms: int


def _edit_distance(reference: Sequence[object], hypothesis: Sequence[object]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_item in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return word edit distance divided by reference word count."""
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    if not reference_words:
        return float(len(hypothesis_words))
    return _edit_distance(reference_words, hypothesis_words) / len(reference_words)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Return character edit distance divided by reference character count."""
    if not reference:
        return float(len(hypothesis))
    return _edit_distance(reference, hypothesis) / len(reference)


def corpus_word_error_rate(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Return total word errors divided by total reference words."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have equal length")
    errors = 0
    reference_units = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        reference_words = reference.split()
        errors += _edit_distance(reference_words, hypothesis.split())
        reference_units += len(reference_words)
    return errors / max(reference_units, 1)


def corpus_character_error_rate(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Return total character errors divided by total reference characters."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have equal length")
    errors = sum(
        _edit_distance(reference, hypothesis)
        for reference, hypothesis in zip(references, hypotheses, strict=True)
    )
    return errors / max(sum(len(reference) for reference in references), 1)


def temporal_iou(first: LabeledSpan, second: LabeledSpan) -> float:
    """Intersection-over-union for two half-open temporal spans."""
    intersection = max(0, min(first.end_ms, second.end_ms) - max(first.start_ms, second.start_ms))
    union = max(first.end_ms, second.end_ms) - min(first.start_ms, second.start_ms)
    if union == 0:
        return 1.0 if first.start_ms == second.start_ms else 0.0
    return intersection / union


def _label(span: LabeledSpan) -> str:
    value = span.label
    return str(getattr(value, "value", value))


def _match_spans(
    references: Sequence[LabeledSpan],
    predictions: Sequence[LabeledSpan],
    *,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    adjacency: dict[int, list[tuple[int, float]]] = {
        reference_index: sorted(
            [
                (prediction_index, temporal_iou(reference, prediction))
                for prediction_index, prediction in enumerate(predictions)
                if _label(reference) == _label(prediction)
                and temporal_iou(reference, prediction) >= iou_threshold
            ],
            key=lambda candidate: candidate[1],
            reverse=True,
        )
        for reference_index, reference in enumerate(references)
    }
    return _maximum_cardinality_matches(adjacency)


def _maximum_cardinality_matches(
    adjacency: Mapping[int, Sequence[tuple[int, float]]],
) -> list[tuple[int, int, float]]:
    """Return augmenting-path matches, preferring higher-scored edges."""
    prediction_to_reference: dict[int, int] = {}

    def assign(reference_index: int, seen_predictions: set[int]) -> bool:
        for prediction_index, _ in adjacency[reference_index]:
            if prediction_index in seen_predictions:
                continue
            seen_predictions.add(prediction_index)
            previous_reference = prediction_to_reference.get(prediction_index)
            if previous_reference is None or assign(previous_reference, seen_predictions):
                prediction_to_reference[prediction_index] = reference_index
                return True
        return False

    for reference_index in sorted(
        adjacency,
        key=lambda index: max((score for _, score in adjacency[index]), default=0.0),
        reverse=True,
    ):
        assign(reference_index, set())
    return [
        (
            reference_index,
            prediction_index,
            next(
                score
                for candidate_index, score in adjacency[reference_index]
                if candidate_index == prediction_index
            ),
        )
        for prediction_index, reference_index in prediction_to_reference.items()
    ]


def span_classification_metrics(
    references: Sequence[LabeledSpan],
    predictions: Sequence[LabeledSpan],
    *,
    labels: Sequence[str] | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Compute per-class P/R/F1, macro-F1, and matched temporal IoU."""
    matches = _match_spans(references, predictions, iou_threshold=iou_threshold)
    inferred_labels = {_label(span) for span in [*references, *predictions]}
    evaluated_labels = list(labels) if labels is not None else sorted(inferred_labels)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in evaluated_labels:
        reference_count = sum(_label(span) == label for span in references)
        prediction_count = sum(_label(span) == label for span in predictions)
        true_positives = sum(_label(references[ref_index]) == label for ref_index, _, _ in matches)
        precision = true_positives / prediction_count if prediction_count else 0.0
        recall = true_positives / reference_count if reference_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_ious = [
            iou
            for reference_index, _, iou in matches
            if _label(references[reference_index]) == label
        ]
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_temporal_iou": sum(class_ious) / len(class_ious) if class_ious else 0.0,
            "support": reference_count,
        }
        f1_values.append(f1)
    return {
        "per_class": per_class,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "mean_temporal_iou": (sum(iou for _, _, iou in matches) / len(matches) if matches else 0.0),
        "matched": len(matches),
        "reference_count": len(references),
        "prediction_count": len(predictions),
    }


def position_aware_event_score(
    references: Sequence[LabeledSpan],
    predictions: Sequence[LabeledSpan],
    *,
    duration_ms: int,
    tolerance_fraction: float = 0.1,
) -> float:
    """Score label and normalized acoustic position without using transcript anchors.

    Events are paired by label using augmenting-path assignment. A match at the
    same midpoint scores one and decays linearly to
    zero at ``tolerance_fraction`` of clip duration. This deliberately ignores
    ``after_word_id`` so ASR word or alignment errors cannot become event errors.
    """
    if not references:
        return 1.0 if not predictions else 0.0
    if duration_ms <= 0 or not 0 < tolerance_fraction <= 1:
        raise ValueError("duration_ms and tolerance_fraction must be positive")
    tolerance_ms = duration_ms * tolerance_fraction
    adjacency: dict[int, list[tuple[int, float]]] = {}
    for reference_index, reference in enumerate(references):
        reference_midpoint = (reference.start_ms + reference.end_ms) / 2
        candidates = []
        for prediction_index, prediction in enumerate(predictions):
            if _label(prediction) != _label(reference):
                continue
            prediction_midpoint = (prediction.start_ms + prediction.end_ms) / 2
            score = max(0.0, 1.0 - abs(reference_midpoint - prediction_midpoint) / tolerance_ms)
            if score > 0:
                candidates.append((prediction_index, score))
        adjacency[reference_index] = sorted(
            candidates, key=lambda candidate: candidate[1], reverse=True
        )
    matches = _maximum_cardinality_matches(adjacency)
    return sum(score for _, _, score in matches) / max(len(references), len(predictions))


def macro_f1(
    references: Sequence[str], predictions: Sequence[str], *, labels: Sequence[str]
) -> float:
    """Compute unweighted one-vs-rest F1 over a fixed label set."""
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have equal length")
    scores: list[float] = []
    for label in labels:
        true_positives = sum(
            reference == label and prediction == label
            for reference, prediction in zip(references, predictions, strict=True)
        )
        false_positives = sum(
            reference != label and prediction == label
            for reference, prediction in zip(references, predictions, strict=True)
        )
        false_negatives = sum(
            reference == label and prediction != label
            for reference, prediction in zip(references, predictions, strict=True)
        )
        denominator = 2 * true_positives + false_positives + false_negatives
        scores.append(2 * true_positives / denominator if denominator else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def multiclass_brier_score(
    references: Sequence[Mapping[str, float]],
    predictions: Sequence[Mapping[str, float]],
    *,
    labels: Sequence[str],
) -> float:
    """Mean sum of squared errors against hard or soft categorical targets."""
    _validate_distributions(references, predictions, labels)
    if not references:
        return 0.0
    return sum(
        sum((prediction[label] - reference[label]) ** 2 for label in labels)
        for reference, prediction in zip(references, predictions, strict=True)
    ) / len(references)


def soft_cross_entropy(
    references: Sequence[Mapping[str, float]],
    predictions: Sequence[Mapping[str, float]],
    *,
    labels: Sequence[str],
    epsilon: float = 1e-12,
) -> float:
    """Cross-entropy from soft targets, clipping prediction logs for stability."""
    _validate_distributions(references, predictions, labels)
    if not references:
        return 0.0
    return -sum(
        sum(reference[label] * math.log(max(prediction[label], epsilon)) for label in labels)
        for reference, prediction in zip(references, predictions, strict=True)
    ) / len(references)


def expected_calibration_error(
    references: Sequence[str],
    predictions: Sequence[Mapping[str, float]],
    *,
    labels: Sequence[str],
    bins: int = 10,
) -> float:
    """Top-label ECE with equal-width confidence bins."""
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have equal length")
    if bins <= 0:
        raise ValueError("bins must be positive")
    if not references:
        return 0.0
    _validate_distributions(
        [{label: float(label == reference) for label in labels} for reference in references],
        predictions,
        labels,
    )
    total_error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members: list[tuple[float, bool]] = []
        for reference, prediction in zip(references, predictions, strict=True):
            predicted_label = max(labels, key=prediction.__getitem__)
            confidence = prediction[predicted_label]
            if lower <= confidence < upper or (bin_index == bins - 1 and confidence == 1.0):
                members.append((confidence, predicted_label == reference))
        if members:
            confidence = sum(item[0] for item in members) / len(members)
            accuracy = sum(item[1] for item in members) / len(members)
            total_error += len(members) / len(references) * abs(accuracy - confidence)
    return total_error


def acoustic_preference_score(
    predictions: Sequence[str],
    acoustic_targets: Sequence[str],
    lexical_targets: Sequence[str],
) -> dict[str, float | int]:
    """Measure preference for acoustic over lexical labels on conflict items."""
    if not (len(predictions) == len(acoustic_targets) == len(lexical_targets)):
        raise ValueError("predictions and targets must have equal length")
    conflict_indices = [
        index
        for index, (acoustic, lexical) in enumerate(
            zip(acoustic_targets, lexical_targets, strict=True)
        )
        if acoustic != lexical
    ]
    if not conflict_indices:
        return {
            "acoustic_accuracy": 0.0,
            "lexical_accuracy": 0.0,
            "aps": 0.0,
            "conflict_items": 0,
        }
    acoustic_accuracy = sum(
        predictions[index] == acoustic_targets[index] for index in conflict_indices
    ) / len(conflict_indices)
    lexical_accuracy = sum(
        predictions[index] == lexical_targets[index] for index in conflict_indices
    ) / len(conflict_indices)
    return {
        "acoustic_accuracy": acoustic_accuracy,
        "lexical_accuracy": lexical_accuracy,
        "aps": acoustic_accuracy - lexical_accuracy,
        "conflict_items": len(conflict_indices),
    }


def _validate_distributions(
    references: Sequence[Mapping[str, float]],
    predictions: Sequence[Mapping[str, float]],
    labels: Sequence[str],
) -> None:
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have equal length")
    expected = set(labels)
    for distribution in [*references, *predictions]:
        if set(distribution) != expected:
            raise ValueError("each distribution must contain exactly the requested labels")
        if abs(sum(distribution.values()) - 1.0) > 1e-6:
            raise ValueError("each distribution must sum to one")
