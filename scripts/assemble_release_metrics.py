#!/usr/bin/env python3
"""Assemble release-gate inputs from sealed FP/INT8 and deployment reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _required(value: dict[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"missing metric: {'.'.join(path)}")
        current = current[key]
    return current


def _accepted(value: dict[str, Any], name: str) -> bool:
    accepted = _required(value, "candidate_passes")
    if not isinstance(accepted, bool):
        raise ValueError(f"{name} acceptance candidate_passes must be a boolean")
    return accepted


def _accepted_hostile_regression(value: dict[str, Any] | None) -> bool:
    if value is None:
        return False
    required_confirmations = (
        value.get("passed") is True,
        value.get("human_recording_confirmed") is True,
        value.get("speaker_consent_confirmed") is True,
    )
    evidence = (
        value.get("audio"),
        value.get("inference"),
        value.get("model"),
        value.get("calibration"),
    )
    hashes_are_present = all(
        isinstance(record, dict)
        and isinstance(record.get("sha256"), str)
        and len(record["sha256"]) == 64
        and all(character in "0123456789abcdef" for character in record["sha256"])
        for record in evidence
    )
    return all(required_confirmations) and hashes_are_present


def assemble(
    *,
    base: dict[str, Any],
    full_precision: dict[str, Any],
    int8: dict[str, Any],
    deployment: dict[str, Any],
    event_acceptance: dict[str, Any],
    style_acceptance: dict[str, Any],
    affect_acceptance: dict[str, Any],
    parameter_count: int,
    hostile_speech_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "parameter_count": parameter_count,
        "base_wer": _required(base, "asr_wer"),
        "fp_wer": _required(full_precision, "asr_wer"),
        "int8_wer": _required(int8, "asr_wer"),
        "fp_event_macro_f1": _required(full_precision, "event_segments", "macro_f1"),
        "int8_event_macro_f1": _required(int8, "event_segments", "macro_f1"),
        "fp_speech_control_aux_false_positive_rate": _required(
            full_precision, "speech_controls", "aux_false_positive_rate"
        ),
        "int8_speech_control_aux_false_positive_rate": _required(
            int8, "speech_controls", "aux_false_positive_rate"
        ),
        "fp_speech_control_false_events_per_minute": _required(
            full_precision, "speech_controls", "localized_false_events_per_minute"
        ),
        "int8_speech_control_false_events_per_minute": _required(
            int8, "speech_controls", "localized_false_events_per_minute"
        ),
        "fp_event_presence_macro_f1": _required(full_precision, "event_presence_macro_f1"),
        "int8_event_presence_macro_f1": _required(int8, "event_presence_macro_f1"),
        "fp_style_macro_f1": _required(full_precision, "style_macro_f1"),
        "int8_style_macro_f1": _required(int8, "style_macro_f1"),
        "fp_affect_macro_f1": _required(full_precision, "affect_macro_f1"),
        "int8_affect_macro_f1": _required(int8, "affect_macro_f1"),
        "fp_ood_f1": _required(full_precision, "ood_f1"),
        "int8_ood_f1": _required(int8, "ood_f1"),
        "acoustic_preference_score": _required(full_precision, "acoustic_preference_score"),
        "affect_coverage": _required(full_precision, "affect_coverage"),
        "affect_full_coverage_error": _required(full_precision, "affect_full_coverage_error"),
        "affect_selective_error": _required(full_precision, "affect_selective_error"),
        "json_validity_rate": _required(deployment, "json_validity_rate"),
        "xml_validity_rate": _required(deployment, "xml_validity_rate"),
        "cpu_real_time_factor": _required(deployment, "cpu_real_time_factor"),
        "committed_retraction_rate": _required(deployment, "committed_retraction_rate"),
        "event_external_validation_passed": _accepted(event_acceptance, "event"),
        "style_external_validation_passed": _accepted(style_acceptance, "style"),
        "affect_external_validation_passed": _accepted(affect_acceptance, "affect"),
        "hostile_speech_regression_passed": _accepted_hostile_regression(hostile_speech_report),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--fp", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--event-acceptance", type=Path, required=True)
    parser.add_argument("--style-acceptance", type=Path, required=True)
    parser.add_argument("--affect-acceptance", type=Path, required=True)
    parser.add_argument("--parameter-count", type=int, required=True)
    parser.add_argument("--hostile-speech-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    metrics = assemble(
        base=_load(arguments.base),
        full_precision=_load(arguments.fp),
        int8=_load(arguments.int8),
        deployment=_load(arguments.deployment),
        event_acceptance=_load(arguments.event_acceptance),
        style_acceptance=_load(arguments.style_acceptance),
        affect_acceptance=_load(arguments.affect_acceptance),
        parameter_count=arguments.parameter_count,
        hostile_speech_report=_load(arguments.hostile_speech_report),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
