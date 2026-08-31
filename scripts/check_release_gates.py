#!/usr/bin/env python3
"""Check a candidate metrics bundle against Attune's v0.1 release gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attune.evaluation.release_gates import (
    ReleaseMetrics,
    evaluate_release_gates,
    write_release_gate_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/release-gates.json"))
    arguments = parser.parse_args()
    metrics = ReleaseMetrics.from_mapping(json.loads(arguments.metrics.read_text()))
    results = evaluate_release_gates(metrics)
    report = write_release_gate_report(arguments.output, metrics, results)
    print(json.dumps(report, indent=2))
    if not report["release_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
