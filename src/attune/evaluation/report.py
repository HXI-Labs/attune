"""Evaluation report assembly for schema-valid Attune outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from attune.evaluation.metrics import (
    acoustic_preference_score,
    corpus_character_error_rate,
    corpus_word_error_rate,
    expected_calibration_error,
    macro_f1,
    multiclass_brier_score,
    position_aware_event_score,
    soft_cross_entropy,
    span_classification_metrics,
)
from attune.schema.output import AffectCategory, AttuneOutput, EventLabel, StyleLabel


@dataclass(frozen=True)
class RuntimeMetrics:
    """Runtime fields populated by a runner when meaningful."""

    audio_seconds: float
    elapsed_seconds: float
    real_time_factor: float | None
    latency_ms: float | None
    first_result_latency_ms: float | None = None

    @classmethod
    def measured(cls, *, audio_seconds: float, elapsed_seconds: float) -> RuntimeMetrics:
        return cls(
            audio_seconds=audio_seconds,
            elapsed_seconds=elapsed_seconds,
            real_time_factor=elapsed_seconds / audio_seconds if audio_seconds > 0 else None,
            latency_ms=elapsed_seconds * 1000,
        )


@dataclass(frozen=True)
class EvaluationItem:
    item_id: str
    reference: AttuneOutput
    prediction: AttuneOutput | Mapping[str, Any]
    acoustic_target: str
    lexical_target: str
    runtime: RuntimeMetrics


@dataclass(frozen=True)
class _ReportSpan:
    label: object
    start_ms: int
    end_ms: int


@dataclass
class EvaluationReport:
    runner: str
    items: list[dict[str, Any]]
    metrics: dict[str, Any]
    schema_validity: dict[str, Any]
    runtime: dict[str, Any]
    skipped_runners: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_items(
    runner: str,
    items: list[EvaluationItem],
    *,
    skipped_runners: list[dict[str, str]] | None = None,
) -> EvaluationReport:
    """Evaluate predictions while preserving per-item schema/runtime evidence."""
    labels = [category.value for category in AffectCategory]
    schema_errors: list[dict[str, str]] = []
    item_rows: list[dict[str, Any]] = []
    valid_items: list[tuple[EvaluationItem, AttuneOutput]] = []
    for item in items:
        try:
            payload = (
                item.prediction.model_dump(mode="json")
                if isinstance(item.prediction, AttuneOutput)
                else item.prediction
            )
            prediction = AttuneOutput.model_validate(payload)
            schema_valid = True
            valid_items.append((item, prediction))
        except ValueError as error:
            schema_valid = False
            schema_errors.append({"item_id": item.item_id, "error": str(error)})
            prediction = None
        item_rows.append(
            {
                "item_id": item.item_id,
                "schema_valid": schema_valid,
                "reference_text": item.reference.transcript.text,
                "predicted_text": prediction.transcript.text if prediction else None,
                "reference_affect": (
                    item.reference.affect.top_label.value
                    if item.reference.affect.top_label
                    else None
                ),
                "predicted_affect": (
                    prediction.affect.top_label.value
                    if prediction and prediction.affect.top_label
                    else None
                ),
                "acoustic_target": item.acoustic_target,
                "lexical_target": item.lexical_target,
                "runtime": asdict(item.runtime),
            }
        )

    references = [item.reference for item, _ in valid_items]
    predictions = [prediction for _, prediction in valid_items]
    reference_labels = [
        output.affect.top_label.value if output.affect.top_label else AffectCategory.AMBIGUOUS.value
        for output in references
    ]
    prediction_labels = [
        output.affect.top_label.value if output.affect.top_label else AffectCategory.AMBIGUOUS.value
        for output in predictions
    ]
    reference_distributions = [
        {category.value: probability for category, probability in output.affect.categories.items()}
        for output in references
    ]
    prediction_distributions = [
        {category.value: probability for category, probability in output.affect.categories.items()}
        for output in predictions
    ]

    all_reference_events, all_prediction_events = _separate_clip_spans(
        references, predictions, attribute="events"
    )
    all_reference_styles, all_prediction_styles = _separate_clip_spans(
        references, predictions, attribute="styles"
    )
    metrics: dict[str, Any] = {
        "asr": {
            "wer": corpus_word_error_rate(
                [reference.transcript.text for reference in references],
                [prediction.transcript.text for prediction in predictions],
            ),
            "cer": corpus_character_error_rate(
                [reference.transcript.text for reference in references],
                [prediction.transcript.text for prediction in predictions],
            ),
        },
        "events": {
            **span_classification_metrics(
                all_reference_events,
                all_prediction_events,
                labels=[label.value for label in EventLabel],
            ),
            "position_aware_score": _mean(
                [
                    position_aware_event_score(
                        reference.events,
                        prediction.events,
                        duration_ms=reference.audio.duration_ms,
                    )
                    for reference, prediction in zip(references, predictions, strict=True)
                ]
            ),
        },
        "styles": span_classification_metrics(
            all_reference_styles,
            all_prediction_styles,
            labels=[label.value for label in StyleLabel],
        ),
        "affect": {
            "macro_f1": macro_f1(reference_labels, prediction_labels, labels=labels),
            "brier_score": multiclass_brier_score(
                reference_distributions, prediction_distributions, labels=labels
            ),
            "soft_cross_entropy": soft_cross_entropy(
                reference_distributions, prediction_distributions, labels=labels
            ),
            "ece": expected_calibration_error(
                reference_labels, prediction_distributions, labels=labels
            ),
        },
        "acoustic_preference": acoustic_preference_score(
            prediction_labels,
            [item.acoustic_target for item, _ in valid_items],
            [item.lexical_target for item, _ in valid_items],
        ),
    }
    elapsed = sum(item.runtime.elapsed_seconds for item in items)
    audio = sum(item.runtime.audio_seconds for item in items)
    return EvaluationReport(
        runner=runner,
        items=item_rows,
        metrics=metrics,
        schema_validity={
            "valid": len(items) - len(schema_errors),
            "invalid": len(schema_errors),
            "errors": schema_errors,
        },
        runtime={
            "audio_seconds": audio,
            "elapsed_seconds": elapsed,
            "real_time_factor": elapsed / audio if audio else None,
            "mean_latency_ms": _mean_optional(
                [item.runtime.latency_ms for item in items if item.runtime.latency_ms is not None]
            ),
            "first_result_latency_ms": _mean_optional(
                [
                    item.runtime.first_result_latency_ms
                    for item in items
                    if item.runtime.first_result_latency_ms is not None
                ]
            ),
        },
        skipped_runners=skipped_runners or [],
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_optional(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _separate_clip_spans(
    references: list[AttuneOutput],
    predictions: list[AttuneOutput],
    *,
    attribute: str,
) -> tuple[list[_ReportSpan], list[_ReportSpan]]:
    """Offset clips so span matching cannot pair annotations across files."""
    reference_spans: list[_ReportSpan] = []
    prediction_spans: list[_ReportSpan] = []
    offset = 0
    for reference, prediction in zip(references, predictions, strict=True):
        for span in getattr(reference, attribute):
            reference_spans.append(
                _ReportSpan(span.label, span.start_ms + offset, span.end_ms + offset)
            )
        for span in getattr(prediction, attribute):
            prediction_spans.append(
                _ReportSpan(span.label, span.start_ms + offset, span.end_ms + offset)
            )
        offset += max(reference.audio.duration_ms, prediction.audio.duration_ms) + 1
    return reference_spans, prediction_spans
