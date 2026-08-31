from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[2] / "scripts/check_berst_affect_sealed.py"
SPEC = importlib.util.spec_from_file_location("check_berst_affect_sealed", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def report(*, macro_f1: float = 0.30, first_risk: float = 0.1, largest_share: float = 0.45):
    remainder = (1.0 - largest_share) / 2
    return {
        "affect_macro_f1": macro_f1,
        "affect_selective_risk_improves": True,
        "affect_risk_coverage": [
            {"coverage": 0.1, "risk": first_risk},
            {"coverage": 1.0, "risk": 0.4},
        ],
        "affect_prediction_share": {
            "neutral": largest_share,
            "joy": remainder,
            "anger": remainder,
        },
    }


def test_berst_affect_sealed_requires_all_gates() -> None:
    gates = CHECK.check_report(report())

    assert all(gate["passed"] for gate in gates)


def test_berst_affect_sealed_rejects_nonselective_confidence() -> None:
    gates = CHECK.check_report(report(first_risk=0.41))

    assert not next(
        gate for gate in gates if gate["name"] == "berst_sealed_selective_risk_improves"
    )["passed"]


def test_berst_affect_sealed_rejects_class_collapse() -> None:
    gates = CHECK.check_report(report(largest_share=0.451))

    assert not next(
        gate for gate in gates if gate["name"] == "berst_sealed_largest_prediction_share"
    )["passed"]
