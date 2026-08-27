#!/usr/bin/env python3
"""Fit temperatures on validation records and audit an untouched inspection test."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from attune.calibration import (
    CalibrationError,
    TemperatureCalibration,
    fit_temperature,
    metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="JSONL score records; repeat for component-specific files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/calibration/phase1.json"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("research/calibration-results.json"),
    )
    parser.add_argument("--expected-test-clips", type=int, default=310)
    return parser.parse_args()


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        try:
            records.extend(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except (OSError, json.JSONDecodeError) as error:
            raise CalibrationError(f"cannot read calibration records {path}: {error}") from error
    required = {"component", "split", "clip_id", "labels", "logits", "target"}
    if not records or not all(isinstance(record, dict) for record in records):
        raise CalibrationError("calibration input is empty or invalid")
    seen: set[tuple[str, str, str]] = set()
    for index, record in enumerate(records, 1):
        if missing := required - record.keys():
            raise CalibrationError(f"row {index} lacks {', '.join(sorted(missing))}")
        if record["split"] not in {"validation", "inspection_test"}:
            raise CalibrationError(f"row {index} has forbidden split {record['split']!r}")
        identity = (record["component"], record["split"], record["clip_id"])
        if identity in seen:
            raise CalibrationError(f"duplicate calibration row {identity}")
        seen.add(identity)
    return records


def calibrate(records: list[dict[str, Any]], expected_test_clips: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["component"])].append(record)
    components = {}
    for component, rows in sorted(grouped.items()):
        validation = [row for row in rows if row["split"] == "validation"]
        test = [row for row in rows if row["split"] == "inspection_test"]
        if not validation or not test:
            raise CalibrationError(f"{component} requires validation and inspection_test rows")
        validation_ids = {row["clip_id"] for row in validation}
        if validation_ids & {row["clip_id"] for row in test}:
            raise CalibrationError(f"{component} leaks clip IDs from validation into test")
        labels = tuple(str(label) for label in validation[0]["labels"])
        if any(tuple(row["labels"]) != labels for row in rows):
            raise CalibrationError(f"{component} rows have inconsistent labels")
        label_indices = {label: index for index, label in enumerate(labels)}
        if any(row["target"] not in label_indices for row in rows):
            raise CalibrationError(f"{component} contains a target outside its labels")
        temperature = fit_temperature(
            [row["logits"] for row in validation],
            [label_indices[row["target"]] for row in validation],
        )
        scaling = TemperatureCalibration(labels, temperature)
        components[component] = {
            **scaling.model_dump(),
            "selection": "minimum categorical NLL on validation only",
            "validation": _partition_metrics(validation, scaling),
            "inspection_test": {
                "designation": "test; never used for fitting or selection",
                **_partition_metrics(test, scaling),
            },
        }
    test_ids = {
        record["clip_id"] for record in records if record["split"] == "inspection_test"
    }
    if expected_test_clips and len(test_ids) != expected_test_clips:
        raise CalibrationError(
            f"found {len(test_ids)} unique test clips, expected {expected_test_clips}"
        )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "method": "temperature_scaling",
        "fitted_on": "validation",
        "gate_decision": "closed",
        "label_status": "weak source / acted labels; not reviewed gold",
        "inspection_test": {
            "designation": "test",
            "unique_clips": len(test_ids),
            "used_for_fitting_or_selection": False,
        },
        "components": components,
    }


def _partition_metrics(
    rows: list[dict[str, Any]], scaling: TemperatureCalibration
) -> dict[str, Any]:
    raw = [TemperatureCalibration(scaling.labels, 1.0).probabilities(row["logits"]) for row in rows]
    calibrated = [scaling.probabilities(row["logits"]) for row in rows]
    targets = [row["target"] for row in rows]
    return {
        "clips": len(rows),
        "before": metrics(raw, targets, scaling.labels),
        "after": metrics(calibrated, targets, scaling.labels),
    }


def main() -> None:
    arguments = parse_args()
    try:
        payload = calibrate(load_records(arguments.input), arguments.expected_test_clips)
    except CalibrationError as error:
        raise SystemExit(f"error: {error}") from error
    serialized = json.dumps(payload, indent=2) + "\n"
    for path in (arguments.output, arguments.report_output):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
