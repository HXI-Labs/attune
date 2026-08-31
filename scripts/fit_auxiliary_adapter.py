#!/usr/bin/env python3
"""Fit and sealed-evaluate a speech-negative auxiliary correction head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attune.auxiliary_adapter import (
    evaluate_auxiliary_adapter,
    fit_auxiliary_adapter,
    load_score_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--adapter-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    arguments = parser.parse_args()
    train = load_score_rows(arguments.train)
    development = load_score_rows(arguments.development)
    sealed = load_score_rows(arguments.sealed)
    adapter = fit_auxiliary_adapter(train, development)
    report = {
        "schema_version": "1.0",
        "development": evaluate_auxiliary_adapter(adapter, development),
        "sealed": evaluate_auxiliary_adapter(adapter, sealed),
        "deployment_enabled_labels": {
            "event_presence": adapter.event_presence.deployment_enabled_labels,
            "styles": adapter.styles.deployment_enabled_labels,
        },
    }
    arguments.adapter_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.adapter_output.write_text(adapter.model_dump_json(indent=2) + "\n")
    arguments.report_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.report_output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
