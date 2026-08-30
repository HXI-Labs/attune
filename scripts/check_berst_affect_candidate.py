#!/usr/bin/env python3
"""Check a BERSt affect candidate before its sealed test is opened."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from attune.integrity import file_digest


def _risk_improves(report: dict[str, Any]) -> bool:
    curve = report["affect_risk_coverage"]
    return bool(report["affect_selective_risk_improves"]) and curve[0]["risk"] < curve[-1]["risk"]


def check_candidate(
    core_report: dict[str, Any],
    berst_report: dict[str, Any],
    ravdess_report: dict[str, Any],
    scope_report: dict[str, Any],
) -> list[dict[str, Any]]:
    largest_ravdess_share = max(ravdess_report["affect_prediction_share"].values())
    requirements = (
        ("core_development_macro_f1", core_report["affect_macro_f1"], ">=", 0.64),
        ("core_acoustic_preference_score", core_report["acoustic_preference_score"], ">", 0.0),
        ("berst_development_macro_f1", berst_report["affect_macro_f1"], ">=", 0.30),
        ("core_selective_risk_improves", _risk_improves(core_report), "==", True),
        ("berst_selective_risk_improves", _risk_improves(berst_report), "==", True),
        ("ravdess_macro_f1", ravdess_report["affect_macro_f1"], ">=", 0.40),
        ("ravdess_largest_prediction_share", largest_ravdess_share, "<=", 0.45),
        ("checkpoint_scope", bool(scope_report.get("passed", False)), "==", True),
    )
    gates = []
    for name, value, comparison, threshold in requirements:
        if comparison == ">=":
            passed = value >= threshold
        elif comparison == ">":
            passed = value > threshold
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
    parser.add_argument("--core-report", type=Path, required=True)
    parser.add_argument("--berst-report", type=Path, required=True)
    parser.add_argument("--ravdess-report", type=Path, required=True)
    parser.add_argument("--scope-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    paths = {
        "core_development": arguments.core_report,
        "berst_development": arguments.berst_report,
        "ravdess": arguments.ravdess_report,
        "checkpoint_scope": arguments.scope_report,
    }
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    gates = check_candidate(
        reports["core_development"],
        reports["berst_development"],
        reports["ravdess"],
        reports["checkpoint_scope"],
    )
    passed = all(gate["passed"] for gate in gates)
    output = {
        "schema_version": "1.0",
        "phase": "development_and_opened_external",
        "inputs": {
            name: {"path": str(path), "sha256": file_digest(path)} for name, path in paths.items()
        },
        "gates": gates,
        "candidate_passes": passed,
        "sealed_test_permitted": passed,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"candidate_passes": passed, "output": str(arguments.output)}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
