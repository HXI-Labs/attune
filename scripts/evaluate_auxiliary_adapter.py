#!/usr/bin/env python3
"""Audit a frozen auxiliary adapter on raw score rows without enabling deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attune.auxiliary_adapter import (
    AuxiliaryAdapter,
    evaluate_auxiliary_adapter,
    load_score_rows,
    predict_auxiliary_adapter,
)
from attune.integrity import file_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    adapter = AuxiliaryAdapter.model_validate_json(arguments.adapter.read_text())
    rows = load_score_rows(arguments.scores)
    predictions = predict_auxiliary_adapter(adapter, rows)
    flagged = [
        prediction
        for prediction in predictions
        if any(
            value["above_threshold"]
            for task in ("event_presence", "styles")
            for value in prediction[task].values()
        )
    ]
    report = {
        "schema_version": "1.0",
        "adapter_sha256": file_digest(arguments.adapter),
        "scores_sha256": file_digest(arguments.scores),
        "metrics": evaluate_auxiliary_adapter(adapter, rows),
        "flagged_clips": flagged,
        "deployment_unchanged": True,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"Audited {len(rows)} clips; {len(flagged)} crossed at least one candidate threshold",
        flush=True,
    )


if __name__ == "__main__":
    main()
