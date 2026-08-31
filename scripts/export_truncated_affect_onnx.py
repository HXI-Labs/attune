#!/usr/bin/env python3
"""Export and validate the compact v0.13 affect branch as FP32 and INT8 ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attune.inference.emotion2vec_export import (
    export_truncated_affect_onnx,
    quantize_truncated_affect_onnx,
    validate_truncated_affect_onnx,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emotion2vec-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fp-output", type=Path, required=True)
    parser.add_argument("--int8-output", type=Path, required=True)
    parser.add_argument("--validation-audio", type=Path, action="append", required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    arguments = parser.parse_args()
    if len(arguments.validation_audio) < 2:
        parser.error("--validation-audio must be supplied at least twice")

    fp_artifact = export_truncated_affect_onnx(
        arguments.emotion2vec_path,
        arguments.checkpoint,
        arguments.fp_output,
    )
    fp_validation = validate_truncated_affect_onnx(
        arguments.emotion2vec_path,
        arguments.checkpoint,
        arguments.fp_output,
        arguments.validation_audio,
    )
    if fp_validation["maximum_probability_difference"] > 1e-4:
        raise RuntimeError("FP32 ONNX export exceeds the probability parity tolerance")

    int8_artifact = quantize_truncated_affect_onnx(
        arguments.fp_output,
        arguments.int8_output,
        arguments.checkpoint,
    )
    int8_validation = validate_truncated_affect_onnx(
        arguments.emotion2vec_path,
        arguments.checkpoint,
        arguments.int8_output,
        arguments.validation_audio,
    )
    report = {
        "schema_version": "1.0",
        "method": "padding_aware_legacy_trace_selective_dynamic_int8",
        "fp32": {"artifact": fp_artifact, "validation": fp_validation},
        "int8": {"artifact": int8_artifact, "validation": int8_validation},
    }
    arguments.report_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.report_output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
