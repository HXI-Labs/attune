#!/usr/bin/env python3
"""Build machine-readable confusion, OOD, and runtime slices from cascade output."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def categorical_confusion(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        record
        for record in records
        if record["reference_affect"] and record["emotion2vec_affect"]
    ]
    matrix = Counter(
        (record["reference_affect"][0], record["emotion2vec_affect"])
        for record in rows
    )
    return {
        "evaluated_clips": len(rows),
        "labels": sorted(
            {label for pair in matrix for label in pair}
        ),
        "rows": [
            {"reference": reference, "prediction": prediction, "clips": count}
            for (reference, prediction), count in sorted(matrix.items())
        ],
        "by_source": {
            source: categorical_confusion(source_rows)
            for source in sorted({row["source_dataset"] for row in rows})
            if (
                source_rows := [
                    row for row in rows if row["source_dataset"] == source
                ]
            )
        }
        if len({row["source_dataset"] for row in rows}) > 1
        else {},
    }


def annotation_errors(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted(
        {
            label
            for record in records
            for label in (
                *record["expected_annotations"],
                *record["cascade_annotations"],
            )
        }
    )
    return {
        label: {
            "true_positive": sum(
                label in row["expected_annotations"]
                and label in row["cascade_annotations"]
                for row in records
            ),
            "false_positive": sum(
                label not in row["expected_annotations"]
                and label in row["cascade_annotations"]
                for row in records
            ),
            "false_negative": sum(
                label in row["expected_annotations"]
                and label not in row["cascade_annotations"]
                for row in records
            ),
        }
        for label in labels
    }


def ood_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    probes = (
        ("vocalsound", "VocalSound", "vocalsound_probe_annotations"),
        ("fsd50k", "FSD50K", "fsd50k_probe_annotations"),
    )
    rows = []
    for probe, in_domain, key in probes:
        for source in sorted({record["source_dataset"] for record in records}):
            if source == in_domain:
                continue
            source_rows = [
                record for record in records if record["source_dataset"] == source
            ]
            false_positives = sum(bool(record[key]) for record in source_rows)
            rows.append(
                {
                    "probe": probe,
                    "source_dataset": source,
                    "clips": len(source_rows),
                    "false_positive_clips": false_positives,
                    "false_positive_rate": false_positives / len(source_rows),
                }
            )
    return rows


def runtime_slices(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, rows in {
        "combined": records,
        **{
            source: [
                record for record in records if record["source_dataset"] == source
            ]
            for source in sorted({record["source_dataset"] for record in records})
        },
    }.items():
        elapsed = sum(record["elapsed_seconds"] for record in rows)
        audio = sum(record["audio_seconds"] for record in rows)
        result[name] = {
            "clips": len(rows),
            "audio_seconds": audio,
            "elapsed_seconds": elapsed,
            "real_time_factor": elapsed / audio if audio else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("research/attune-cascade-inspection-results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/error-analysis/cascade-310.json"),
    )
    arguments = parser.parse_args()
    report = json.loads(arguments.input.read_text(encoding="utf-8"))
    records = report["predictions"]
    payload = {
        "report_version": "1",
        "source_report": str(arguments.input),
        "source_report_version": report["report_version"],
        "gate_decision": "closed",
        "label_status": "weak source / acted labels; not reviewed gold",
        "affect_confusion": categorical_confusion(records),
        "annotation_errors": annotation_errors(records),
        "probe_ood_by_source": ood_table(records),
        "runtime": runtime_slices(records),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
