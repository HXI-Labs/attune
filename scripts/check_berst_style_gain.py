#!/usr/bin/env python3
"""Check BERSt shouting stability across the predeclared gain sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attune.integrity import file_digest

GAIN_DATASETS = (
    "berst_v1_gain_m12db",
    "berst_v1_gain_m6db",
    "berst_v1_gain_z0db",
    "berst_v1_gain_p6db",
    "berst_v1_gain_p12db",
)


def check_reports(baseline: dict, sweep: dict) -> list[dict]:
    baseline_f1 = baseline["style_metrics"]["shouting"]["f1"]
    minimum_f1 = max(0.0, baseline_f1 - 0.05)
    gates = []
    for dataset_id in GAIN_DATASETS:
        shouting = sweep["by_dataset"][dataset_id]["style_metrics"]["shouting"]
        gates.extend(
            (
                {
                    "name": f"{dataset_id}_f1",
                    "value": shouting["f1"],
                    "comparison": ">=",
                    "threshold": minimum_f1,
                    "passed": shouting["f1"] >= minimum_f1,
                },
                {
                    "name": f"{dataset_id}_false_positive_rate",
                    "value": shouting["false_positive_rate"],
                    "comparison": "<=",
                    "threshold": 0.10,
                    "passed": shouting["false_positive_rate"] <= 0.10,
                },
            )
        )
    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--sweep-report", type=Path, required=True)
    parser.add_argument("--sealed-acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    sealed = json.loads(arguments.sealed_acceptance.read_text())
    if not sealed.get("candidate_passes", False):
        raise ValueError("gain evaluation is invalid without a passed sealed style result")
    gates = check_reports(
        json.loads(arguments.baseline_report.read_text()),
        json.loads(arguments.sweep_report.read_text()),
    )
    passed = all(gate["passed"] for gate in gates)
    paths = {
        "baseline_report": arguments.baseline_report,
        "sweep_report": arguments.sweep_report,
        "sealed_acceptance": arguments.sealed_acceptance,
    }
    output = {
        "schema_version": "1.0",
        "phase": "gain_robustness",
        "inputs": {
            name: {"path": str(path), "sha256": file_digest(path)} for name, path in paths.items()
        },
        "gates": gates,
        "candidate_passes": passed,
        "style_head_accepted": passed,
        "deployment_permitted": False,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"candidate_passes": passed, "output": str(arguments.output)}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
