#!/usr/bin/env python3
"""Record the one-shot BERSt shouting result after development gates pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attune.integrity import file_digest


def check_report(report: dict) -> list[dict]:
    shouting = report["style_metrics"]["shouting"]
    requirements = (
        ("berst_sealed_f1", shouting["f1"], ">=", 0.80),
        ("berst_sealed_no_shout_false_positive_rate", shouting["false_positive_rate"], "<=", 0.08),
    )
    return [
        {
            "name": name,
            "value": value,
            "comparison": comparison,
            "threshold": threshold,
            "passed": value >= threshold if comparison == ">=" else value <= threshold,
        }
        for name, value, comparison, threshold in requirements
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--development-acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    development = json.loads(arguments.development_acceptance.read_text())
    if not development.get("candidate_passes", False):
        raise ValueError("sealed BERSt style report is invalid without passed development gates")
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
            "development_acceptance": {
                "path": str(arguments.development_acceptance),
                "sha256": file_digest(arguments.development_acceptance),
            },
        },
        "gates": gates,
        "candidate_passes": passed,
        "gain_test_required": True,
        "deployment_permitted": False,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"candidate_passes": passed, "output": str(arguments.output)}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
