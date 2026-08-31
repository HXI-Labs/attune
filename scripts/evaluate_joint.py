#!/usr/bin/env python3
"""Evaluate sealed unified-model scores with a fixed calibration bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attune.evaluation.joint import evaluate_joint_scores, load_scores
from attune.inference.onnx_backend import RuntimeCalibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    calibration = RuntimeCalibration.model_validate_json(arguments.calibration.read_text())
    report = evaluate_joint_scores(load_scores(arguments.scores), calibration)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
