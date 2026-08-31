#!/usr/bin/env python3
"""Check a BERSt shouting candidate before its sealed test is opened."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from attune.integrity import file_digest


def check_candidate(
    berst_report: dict[str, Any],
    wesr_report: dict[str, Any],
    british_report: dict[str, Any],
    scope_report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    berst = berst_report["style_metrics"]["shouting"]
    wesr = wesr_report["style_metrics"]["shouting"]
    british_false_positives = british_report["speech_controls"]["style_false_positive_clips"]
    requirements = (
        ("berst_development_f1", berst["f1"], ">=", 0.85),
        ("berst_development_precision", berst["precision"], ">=", 0.80),
        ("berst_development_recall", berst["recall"], ">=", 0.80),
        ("berst_no_shout_false_positive_rate", berst["false_positive_rate"], "<=", 0.05),
        ("wesr_shouting_f1", wesr["f1"], ">=", 0.55),
        ("wesr_shouting_precision", wesr["precision"], ">=", 0.50),
        ("wesr_shouting_recall", wesr["recall"], ">=", 0.60),
        ("british_style_false_positive_clips", british_false_positives, "<=", 0),
    )
    gates = [
        {
            "name": name,
            "value": value,
            "comparison": comparison,
            "threshold": threshold,
            "passed": value >= threshold if comparison == ">=" else value <= threshold,
        }
        for name, value, comparison, threshold in requirements
    ]
    if scope_report is not None:
        gates.append(
            {
                "name": "checkpoint_scope",
                "value": bool(scope_report.get("passed", False)),
                "comparison": "==",
                "threshold": True,
                "passed": bool(scope_report.get("passed", False)),
            }
        )
    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--berst-report", type=Path, required=True)
    parser.add_argument("--wesr-report", type=Path, required=True)
    parser.add_argument("--british-report", type=Path, required=True)
    parser.add_argument("--scope-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    paths = {
        "berst_development": arguments.berst_report,
        "wesr": arguments.wesr_report,
        "british_controls": arguments.british_report,
    }
    if arguments.scope_report is not None:
        paths["checkpoint_scope"] = arguments.scope_report
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    gates = check_candidate(
        reports["berst_development"],
        reports["wesr"],
        reports["british_controls"],
        reports.get("checkpoint_scope"),
    )
    passed = all(gate["passed"] for gate in gates)
    output = {
        "schema_version": "1.0",
        "phase": "development",
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
