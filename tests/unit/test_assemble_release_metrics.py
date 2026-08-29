from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "assemble_release_metrics.py"
SPEC = importlib.util.spec_from_file_location("assemble_release_metrics", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sealed(value: float) -> dict[str, object]:
    return {
        "asr_wer": value,
        "event_segments": {"macro_f1": 0.7},
        "event_presence_macro_f1": 0.68,
        "style_macro_f1": 0.72,
        "affect_macro_f1": 0.61,
        "ood_f1": 0.87,
        "acoustic_preference_score": 0.2,
        "affect_coverage": 0.8,
        "affect_full_coverage_error": 0.3,
        "affect_selective_error": 0.2,
    }


def test_assemble_maps_reports_without_manual_copying() -> None:
    metrics = MODULE.assemble(
        base={"asr_wer": 0.1},
        full_precision=_sealed(0.105),
        int8=_sealed(0.108),
        deployment={
            "json_validity_rate": 1.0,
            "xml_validity_rate": 1.0,
            "cpu_real_time_factor": 0.7,
            "committed_retraction_rate": 0.0,
        },
        parameter_count=235_291_018,
    )

    assert metrics["base_wer"] == 0.1
    assert metrics["int8_wer"] == 0.108
    assert metrics["fp_event_macro_f1"] == 0.7


def test_assemble_fails_closed_on_missing_evidence() -> None:
    with pytest.raises(ValueError, match="event_segments.macro_f1"):
        MODULE.assemble(
            base={"asr_wer": 0.1},
            full_precision={"asr_wer": 0.1},
            int8=_sealed(0.1),
            deployment={},
            parameter_count=1,
        )
