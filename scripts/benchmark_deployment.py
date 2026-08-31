#!/usr/bin/env python3
"""Benchmark calibrated ONNX inference including frontend, JSON, and XML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attune.evaluation.deployment import (
    balanced_benchmark_rows,
    benchmark_backend,
    deployment_audio_contract_error,
)
from attune.inference.onnx_backend import OnnxAttuneBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sensevoice-path", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--probe-head", type=Path, action="append", default=[])
    parser.add_argument("--quantization", choices=("fp32", "fp16", "int8"), required=True)
    parser.add_argument("--split", default="sealed_test")
    parser.add_argument("--per-dataset", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    source_rows = [
        json.loads(line)
        for line in arguments.source_manifest.read_text().splitlines()
        if line.strip()
    ]
    candidate_rows = [
        row for row in source_rows if row.get("split") == arguments.split and row.get("audio_path")
    ]
    incompatible_rows = []
    compatible_rows = []
    for row in candidate_rows:
        error = deployment_audio_contract_error(Path(row["audio_path"]))
        if error is None:
            compatible_rows.append(row)
        else:
            incompatible_rows.append(
                {
                    "clip_id": str(row["clip_id"]),
                    "dataset_id": str(row["dataset_id"]),
                    "reason": error,
                }
            )
    rows = balanced_benchmark_rows(
        compatible_rows, split=arguments.split, per_dataset=arguments.per_dataset
    )
    backend = OnnxAttuneBackend.from_local_assets(
        arguments.model,
        arguments.sensevoice_path,
        arguments.calibration,
        quantization=arguments.quantization,
        probe_artifacts=tuple(arguments.probe_head),
    )
    backend.analyse_wav(Path(rows[0]["audio_path"]).read_bytes())
    report = benchmark_backend(backend, rows)
    report["input_contract"] = {
        "required": "RIFF/WAV, PCM16, 16000 Hz, mono",
        "source_rows_considered": len(candidate_rows),
        "compatible_source_rows": len(compatible_rows),
        "excluded_source_rows": len(incompatible_rows),
        "excluded_by_dataset": {
            dataset: sum(row["dataset_id"] == dataset for row in incompatible_rows)
            for dataset in sorted({row["dataset_id"] for row in incompatible_rows})
        },
        "excluded_examples": incompatible_rows[:10],
    }
    report["selected_datasets"] = sorted({str(row["dataset_id"]) for row in rows})
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
