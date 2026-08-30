from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[2] / "scripts/check_berst_style_sealed.py"
SPEC = importlib.util.spec_from_file_location("check_berst_style_sealed", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def report(f1: float, false_positive_rate: float):
    return {
        "style_metrics": {
            "shouting": {
                "f1": f1,
                "false_positive_rate": false_positive_rate,
            }
        }
    }


def test_berst_style_sealed_requires_both_quality_gates() -> None:
    gates = CHECK.check_report(report(0.80, 0.08))

    assert all(gate["passed"] for gate in gates)


def test_berst_style_sealed_rejects_no_shout_false_positives() -> None:
    gates = CHECK.check_report(report(0.90, 0.081))

    assert not next(
        gate for gate in gates if gate["name"] == "berst_sealed_no_shout_false_positive_rate"
    )["passed"]
