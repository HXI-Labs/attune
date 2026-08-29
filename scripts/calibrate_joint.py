#!/usr/bin/env python3
"""Fit a unified Attune runtime calibration bundle on development scores."""

from __future__ import annotations

import argparse
from pathlib import Path

from attune.calibration_joint import fit_runtime_calibration, load_score_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    calibration = fit_runtime_calibration(load_score_rows(arguments.scores))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(calibration.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
