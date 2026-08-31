"""Executable release gates for full-precision and INT8 Attune candidates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReleaseMetrics:
    parameter_count: int
    base_wer: float
    fp_wer: float
    int8_wer: float
    fp_event_macro_f1: float
    int8_event_macro_f1: float
    fp_speech_control_aux_false_positive_rate: float
    int8_speech_control_aux_false_positive_rate: float
    fp_speech_control_false_events_per_minute: float
    int8_speech_control_false_events_per_minute: float
    fp_event_presence_macro_f1: float
    int8_event_presence_macro_f1: float
    fp_style_macro_f1: float
    int8_style_macro_f1: float
    styles_enabled: bool
    fp_affect_macro_f1: float
    int8_affect_macro_f1: float
    fp_ood_f1: float
    int8_ood_f1: float
    acoustic_preference_score: float
    affect_coverage: float
    affect_full_coverage_error: float
    affect_selective_error: float
    json_validity_rate: float
    xml_validity_rate: float
    cpu_real_time_factor: float
    committed_retraction_rate: float
    event_external_validation_passed: bool
    style_external_validation_passed: bool
    affect_external_validation_passed: bool
    int8_event_external_validation_passed: bool
    int8_style_external_validation_passed: bool
    int8_affect_external_validation_passed: bool
    hostile_speech_regression_passed: bool

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ReleaseMetrics:
        missing = sorted(set(cls.__annotations__) - set(value))
        if missing:
            raise ValueError(f"release metrics are missing: {', '.join(missing)}")
        return cls(**{name: value[name] for name in cls.__annotations__})


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    observed: float | int | bool
    requirement: str


def evaluate_release_gates(metrics: ReleaseMetrics) -> list[GateResult]:
    """Evaluate the non-negotiable v0.1 quality and deployment criteria."""

    def maximum(name: str, observed: float | int, limit: float | int) -> GateResult:
        return GateResult(name, observed <= limit, observed, f"<= {limit}")

    def minimum(name: str, observed: float | int, limit: float | int) -> GateResult:
        return GateResult(name, observed >= limit, observed, f">= {limit}")

    return [
        maximum("parameter_count", metrics.parameter_count, 300_000_000),
        maximum("fp_wer_degradation", metrics.fp_wer - metrics.base_wer, 0.01),
        minimum("fp_event_macro_f1", metrics.fp_event_macro_f1, 0.50),
        maximum(
            "fp_speech_control_aux_false_positive_rate",
            metrics.fp_speech_control_aux_false_positive_rate,
            0.01,
        ),
        maximum(
            "fp_speech_control_false_events_per_minute",
            metrics.fp_speech_control_false_events_per_minute,
            0.10,
        ),
        minimum("fp_event_presence_macro_f1", metrics.fp_event_presence_macro_f1, 0.50),
        GateResult(
            "fp_style_macro_f1",
            not metrics.styles_enabled or metrics.fp_style_macro_f1 >= 0.60,
            metrics.fp_style_macro_f1,
            ">= 0.60 when styles are enabled; otherwise output must be disabled",
        ),
        minimum("fp_affect_macro_f1", metrics.fp_affect_macro_f1, 0.40),
        minimum("fp_ood_f1", metrics.fp_ood_f1, 0.75),
        GateResult(
            "positive_acoustic_preference",
            metrics.acoustic_preference_score > 0,
            metrics.acoustic_preference_score,
            "> 0",
        ),
        GateResult(
            "selective_risk_improves",
            metrics.affect_selective_error <= metrics.affect_full_coverage_error,
            metrics.affect_selective_error - metrics.affect_full_coverage_error,
            "<= 0 difference from full-coverage error",
        ),
        minimum("affect_coverage", metrics.affect_coverage, 0.50),
        maximum("int8_wer_degradation", metrics.int8_wer - metrics.fp_wer, 0.005),
        maximum(
            "int8_event_macro_f1_degradation",
            metrics.fp_event_macro_f1 - metrics.int8_event_macro_f1,
            0.02,
        ),
        maximum(
            "int8_speech_control_aux_false_positive_rate",
            metrics.int8_speech_control_aux_false_positive_rate,
            0.01,
        ),
        maximum(
            "int8_speech_control_false_events_per_minute",
            metrics.int8_speech_control_false_events_per_minute,
            0.10,
        ),
        maximum(
            "int8_event_presence_macro_f1_degradation",
            metrics.fp_event_presence_macro_f1 - metrics.int8_event_presence_macro_f1,
            0.02,
        ),
        GateResult(
            "int8_style_macro_f1_degradation",
            not metrics.styles_enabled
            or metrics.fp_style_macro_f1 - metrics.int8_style_macro_f1 <= 0.02,
            metrics.fp_style_macro_f1 - metrics.int8_style_macro_f1,
            "<= 0.02 when styles are enabled; otherwise output must remain disabled",
        ),
        maximum(
            "int8_affect_macro_f1_degradation",
            metrics.fp_affect_macro_f1 - metrics.int8_affect_macro_f1,
            0.02,
        ),
        maximum("int8_ood_f1_degradation", metrics.fp_ood_f1 - metrics.int8_ood_f1, 0.02),
        minimum("json_validity_rate", metrics.json_validity_rate, 1.0),
        minimum("xml_validity_rate", metrics.xml_validity_rate, 1.0),
        maximum("cpu_real_time_factor", metrics.cpu_real_time_factor, 1.0),
        maximum("committed_retraction_rate", metrics.committed_retraction_rate, 0.0),
        GateResult(
            "event_external_validation",
            metrics.event_external_validation_passed,
            metrics.event_external_validation_passed,
            "must pass the predeclared external event gates",
        ),
        GateResult(
            "style_external_validation",
            metrics.style_external_validation_passed,
            metrics.style_external_validation_passed,
            "enabled styles must pass external gates; disabled styles must emit nothing",
        ),
        GateResult(
            "affect_external_validation",
            metrics.affect_external_validation_passed,
            metrics.affect_external_validation_passed,
            "must pass core, conflict, RAVDESS, and sealed BERSt gates",
        ),
        GateResult(
            "int8_event_external_validation",
            metrics.int8_event_external_validation_passed,
            metrics.int8_event_external_validation_passed,
            "INT8 must independently pass the external event gates",
        ),
        GateResult(
            "int8_style_external_validation",
            metrics.int8_style_external_validation_passed,
            metrics.int8_style_external_validation_passed,
            "INT8 enabled styles must pass external gates or remain disabled",
        ),
        GateResult(
            "int8_affect_external_validation",
            metrics.int8_affect_external_validation_passed,
            metrics.int8_affect_external_validation_passed,
            "INT8 must independently pass the external affect gates after recalibration",
        ),
        GateResult(
            "hostile_speech_regression",
            metrics.hostile_speech_regression_passed,
            metrics.hostile_speech_regression_passed,
            "must pass a real-audio retest with no unsupported event or style tags",
        ),
    ]


def write_release_gate_report(
    path: Path, metrics: ReleaseMetrics, results: list[GateResult]
) -> dict[str, Any]:
    payload = {
        "schema_version": "1.2",
        "release_ready": all(result.passed for result in results),
        "metrics": asdict(metrics),
        "gates": [asdict(result) for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
