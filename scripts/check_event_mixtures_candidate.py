#!/usr/bin/env python3
"""Check an event-mixture candidate against its predeclared acceptance gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from attune.integrity import file_digest


def check_candidate(
    wesr_report: dict[str, Any],
    regression_report: dict[str, Any],
    scope_report: dict[str, Any] | None,
    weak_development_report: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    controls = regression_report["speech_controls"]
    requirements = (
        (
            "wesr_temporal_presence_macro_f1",
            wesr_report["localized_event_presence_macro_f1"],
            ">=",
            0.34,
        ),
        (
            "wesr_temporal_presence_recall",
            wesr_report["localized_event_presence_recall"],
            ">=",
            0.30,
        ),
        (
            "wesr_temporal_presence_false_positive_rate",
            wesr_report["localized_event_presence_false_positive_rate"],
            "<=",
            0.06,
        ),
        (
            "opened_regression_segment_macro_f1",
            regression_report["event_segments"]["macro_f1"],
            ">=",
            0.60,
        ),
        (
            "opened_speech_control_false_positive_clips",
            controls["localized_event_false_positive_clips"],
            "<=",
            2,
        ),
    )
    if weak_development_report is not None:
        requirements += (
            (
                "disfluency_laugh_temporal_presence_f1",
                weak_development_report["localized_event_presence_f1"]["laugh"],
                ">=",
                0.65,
            ),
            (
                "disfluency_temporal_presence_recall",
                weak_development_report["localized_event_presence_recall"],
                ">=",
                0.65,
            ),
            (
                "disfluency_temporal_presence_false_positive_rate",
                weak_development_report["localized_event_presence_false_positive_rate"],
                "<=",
                0.10,
            ),
        )
    if scope_report is not None:
        requirements += (("checkpoint_scope", bool(scope_report.get("passed", False)), "==", True),)
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
    parser.add_argument("--wesr-report", type=Path, required=True)
    parser.add_argument("--regression-report", type=Path, required=True)
    parser.add_argument("--scope-report", type=Path)
    parser.add_argument("--weak-development-report", type=Path)
    parser.add_argument("--candidate", default="event-mixtures-v1.1")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    paths = {
        "wesr": arguments.wesr_report,
        "opened_regression": arguments.regression_report,
    }
    if arguments.scope_report is not None:
        paths["checkpoint_scope"] = arguments.scope_report
    if arguments.weak_development_report is not None:
        paths["weak_development"] = arguments.weak_development_report
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    gates = check_candidate(
        reports["wesr"],
        reports["opened_regression"],
        reports.get("checkpoint_scope"),
        reports.get("weak_development"),
    )
    passed = all(gate["passed"] for gate in gates)
    output = {
        "schema_version": "1.0",
        "candidate": arguments.candidate,
        "inputs": {
            name: {"path": str(path), "sha256": file_digest(path)} for name, path in paths.items()
        },
        "gates": gates,
        "candidate_passes": passed,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"candidate_passes": passed, "output": str(arguments.output)}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
