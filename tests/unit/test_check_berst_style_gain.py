from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[2] / "scripts/check_berst_style_gain.py"
SPEC = importlib.util.spec_from_file_location("check_berst_style_gain", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def baseline(f1: float = 0.85):
    return {"style_metrics": {"shouting": {"f1": f1}}}


def sweep(*, f1: float = 0.80, false_positive_rate: float = 0.10):
    return {
        "by_dataset": {
            dataset_id: {
                "style_metrics": {
                    "shouting": {
                        "f1": f1,
                        "false_positive_rate": false_positive_rate,
                    }
                }
            }
            for dataset_id in CHECK.GAIN_DATASETS
        }
    }


def test_gain_sweep_requires_every_variant_to_hold_quality() -> None:
    gates = CHECK.check_reports(baseline(), sweep())

    assert all(gate["passed"] for gate in gates)


def test_gain_sweep_rejects_one_degraded_variant() -> None:
    gain_report = sweep()
    gain_report["by_dataset"]["berst_v1_gain_p12db"]["style_metrics"]["shouting"]["f1"] = 0.79

    gates = CHECK.check_reports(baseline(), gain_report)

    assert not next(gate for gate in gates if gate["name"] == "berst_v1_gain_p12db_f1")["passed"]


def test_gain_sweep_rejects_false_positive_increase() -> None:
    gain_report = sweep()
    gain_report["by_dataset"]["berst_v1_gain_m12db"]["style_metrics"]["shouting"][
        "false_positive_rate"
    ] = 0.101

    gates = CHECK.check_reports(baseline(), gain_report)

    assert not next(
        gate for gate in gates if gate["name"] == "berst_v1_gain_m12db_false_positive_rate"
    )["passed"]
