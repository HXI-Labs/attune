from __future__ import annotations

import pytest

from attune.evaluation.release_gates import ReleaseMetrics, evaluate_release_gates


def passing_metrics() -> ReleaseMetrics:
    return ReleaseMetrics(
        parameter_count=250_000_000,
        base_wer=0.12,
        fp_wer=0.127,
        int8_wer=0.13,
        fp_event_macro_f1=0.71,
        int8_event_macro_f1=0.70,
        fp_speech_control_aux_false_positive_rate=0.0,
        int8_speech_control_aux_false_positive_rate=0.0,
        fp_speech_control_false_events_per_minute=0.0,
        int8_speech_control_false_events_per_minute=0.0,
        fp_event_presence_macro_f1=0.68,
        int8_event_presence_macro_f1=0.67,
        fp_style_macro_f1=0.73,
        int8_style_macro_f1=0.72,
        fp_affect_macro_f1=0.64,
        int8_affect_macro_f1=0.625,
        fp_ood_f1=0.89,
        int8_ood_f1=0.88,
        acoustic_preference_score=0.21,
        affect_coverage=0.80,
        affect_full_coverage_error=0.30,
        affect_selective_error=0.18,
        json_validity_rate=1.0,
        xml_validity_rate=1.0,
        cpu_real_time_factor=0.82,
        committed_retraction_rate=0.0,
        hostile_speech_regression_passed=True,
    )


def test_all_release_gates_pass_for_qualified_candidate() -> None:
    assert all(result.passed for result in evaluate_release_gates(passing_metrics()))


def test_int8_and_acoustic_failures_are_visible() -> None:
    values = passing_metrics().__dict__ | {
        "int8_affect_macro_f1": 0.50,
        "acoustic_preference_score": -0.1,
    }
    failed = {
        result.name
        for result in evaluate_release_gates(ReleaseMetrics(**values))
        if not result.passed
    }
    assert failed == {"int8_affect_macro_f1_degradation", "positive_acoustic_preference"}


def test_metrics_mapping_rejects_incomplete_evidence() -> None:
    with pytest.raises(ValueError, match="missing"):
        ReleaseMetrics.from_mapping({"parameter_count": 10})


def test_release_fails_closed_without_hostile_speech_retest() -> None:
    values = passing_metrics().__dict__ | {"hostile_speech_regression_passed": False}
    failed = {
        result.name
        for result in evaluate_release_gates(ReleaseMetrics(**values))
        if not result.passed
    }
    assert failed == {"hostile_speech_regression"}
