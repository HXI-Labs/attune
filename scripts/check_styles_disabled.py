#!/usr/bin/env python3
"""Create hashed evidence that a release exposes no vocal-style labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from attune.integrity import file_digest


def check_styles_disabled(
    model: Path,
    calibration: Path,
    inferences: list[Path],
) -> dict[str, Any]:
    calibration_payload = json.loads(calibration.read_text())
    if calibration_payload.get("style_enabled_labels") != []:
        raise ValueError("calibration must contain an empty style_enabled_labels list")
    if not inferences:
        raise ValueError("at least one inference result is required")
    records = []
    for inference in inferences:
        output = json.loads(inference.read_text())
        styles = output.get("styles")
        if styles != []:
            raise ValueError(f"style output is not disabled in {inference}: {styles!r}")
        records.append({"path": str(inference), "sha256": file_digest(inference)})
    return {
        "schema_version": "1.0",
        "candidate_passes": True,
        "status": "disabled",
        "deployment_enabled_labels": [],
        "model": {"path": str(model), "sha256": file_digest(model)},
        "calibration": {
            "path": str(calibration),
            "sha256": file_digest(calibration),
        },
        "inferences": records,
        "requirement": (
            "The calibration enables no style labels and every supplied release "
            "regression output contains an empty styles array."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--inference", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    report = check_styles_disabled(
        arguments.model,
        arguments.calibration,
        arguments.inference,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
