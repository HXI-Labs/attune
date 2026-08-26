"""Public Phase 1 evaluation API."""

from attune.evaluation.metrics import (
    acoustic_preference_score,
    character_error_rate,
    corpus_character_error_rate,
    corpus_word_error_rate,
    expected_calibration_error,
    position_aware_event_score,
    span_classification_metrics,
    temporal_iou,
    word_error_rate,
)
from attune.evaluation.report import (
    EvaluationItem,
    EvaluationReport,
    RuntimeMetrics,
    evaluate_items,
)

__all__ = [
    "EvaluationItem",
    "EvaluationReport",
    "RuntimeMetrics",
    "acoustic_preference_score",
    "character_error_rate",
    "corpus_character_error_rate",
    "corpus_word_error_rate",
    "evaluate_items",
    "expected_calibration_error",
    "position_aware_event_score",
    "span_classification_metrics",
    "temporal_iou",
    "word_error_rate",
]
"""Evaluation utilities for Attune."""
