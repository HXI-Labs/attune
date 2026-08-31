#!/usr/bin/env python3
"""Export and validate the compact v0.13 affect branch as standalone INT8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attune.inference.emotion2vec_export import (
    export_truncated_affect_torchscript,
    validate_truncated_affect_torchscript,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emotion2vec-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-audio", type=Path, action="append", required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    arguments = parser.parse_args()
    if len(arguments.validation_audio) < 2:
        parser.error("--validation-audio must be supplied at least twice")

    artifact = export_truncated_affect_torchscript(
        arguments.emotion2vec_path,
        arguments.checkpoint,
        arguments.output,
    )
    validation = validate_truncated_affect_torchscript(
        arguments.emotion2vec_path,
        arguments.checkpoint,
        arguments.output,
        arguments.validation_audio,
    )
    if validation["maximum_probability_difference"] > 1e-6:
        raise RuntimeError("TorchScript export exceeds the INT8 probability parity tolerance")
    report = {
        "schema_version": "1.0",
        "method": "padding_aware_torchscript_dynamic_int8",
        "artifact": artifact,
        "validation": validation,
    }
    arguments.report_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.report_output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
