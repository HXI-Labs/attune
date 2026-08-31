#!/usr/bin/env python3
"""Record the one-shot BERSt affect result after prior gates pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attune.integrity import file_digest


def _risk_improves(report: dict) -> bool:
    curve = report["affect_risk_coverage"]
    return bool(report["affect_selective_risk_improves"]) and curve[0]["risk"] < curve[-1]["risk"]


def check_report(report: dict) -> list[dict]:
    largest_share = max(report["affect_prediction_share"].values())
    requirements = (
        ("berst_sealed_macro_f1", report["affect_macro_f1"], ">=", 0.30),
        ("berst_sealed_selective_risk_improves", _risk_improves(report), "==", True),
        ("berst_sealed_largest_prediction_share", largest_share, "<=", 0.45),
    )
    gates = []
    for name, value, comparison, threshold in requirements:
        if comparison == ">=":
            passed = value >= threshold
        elif comparison == "<=":
            passed = value <= threshold
        else:
            passed = value == threshold
        gates.append(
            {
                "name": name,
                "value": value,
                "comparison": comparison,
                "threshold": threshold,
                "passed": bool(passed),
            }
        )
    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--prior-acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    prior = json.loads(arguments.prior_acceptance.read_text())
    if not prior.get("candidate_passes", False):
        raise ValueError("sealed BERSt affect report is invalid without passed prior gates")
    gates = check_report(json.loads(arguments.report.read_text()))
    passed = all(gate["passed"] for gate in gates)
    output = {
        "schema_version": "1.0",
        "phase": "sealed_test",
        "inputs": {
            "report": {
                "path": str(arguments.report),
                "sha256": file_digest(arguments.report),
            },
            "prior_acceptance": {
                "path": str(arguments.prior_acceptance),
                "sha256": file_digest(arguments.prior_acceptance),
            },
        },
        "gates": gates,
        "candidate_passes": passed,
        "deployment_permitted": False,
        "remaining_requirements": [
            "compose accepted heads",
            "quantize and recalibrate INT8",
            "run full release regression",
            "record fresh human hostile-delivery confirmation",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"candidate_passes": passed, "output": str(arguments.output)}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
