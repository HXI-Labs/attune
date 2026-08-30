from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[2] / "scripts/check_berst_style_candidate.py"
SPEC = importlib.util.spec_from_file_location("check_berst_style_candidate", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def report(*, f1: float, precision: float, recall: float, false_positive_rate: float):
    return {
        "style_metrics": {
            "shouting": {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "false_positive_rate": false_positive_rate,
            }
        }
    }


def test_berst_style_candidate_requires_every_development_gate() -> None:
    gates = CHECK.check_candidate(
        report(f1=0.85, precision=0.80, recall=0.80, false_positive_rate=0.05),
        report(f1=0.55, precision=0.50, recall=0.60, false_positive_rate=0.10),
        {"speech_controls": {"style_false_positive_clips": 0}},
        {"passed": True},
    )

    assert all(gate["passed"] for gate in gates)


def test_berst_style_candidate_fails_closed_on_external_recall() -> None:
    gates = CHECK.check_candidate(
        report(f1=0.90, precision=0.90, recall=0.90, false_positive_rate=0.01),
        report(f1=0.70, precision=0.70, recall=0.59, false_positive_rate=0.10),
        {"speech_controls": {"style_false_positive_clips": 0}},
        {"passed": True},
    )

    assert not next(gate for gate in gates if gate["name"] == "wesr_shouting_recall")["passed"]
    assert not all(gate["passed"] for gate in gates)


def test_deployment_style_check_can_omit_checkpoint_scope() -> None:
    gates = CHECK.check_candidate(
        report(f1=0.85, precision=0.80, recall=0.80, false_positive_rate=0.05),
        report(f1=0.55, precision=0.50, recall=0.60, false_positive_rate=0.10),
        {"speech_controls": {"style_false_positive_clips": 0}},
        None,
    )

    assert all(gate["name"] != "checkpoint_scope" for gate in gates)
    assert all(gate["passed"] for gate in gates)
