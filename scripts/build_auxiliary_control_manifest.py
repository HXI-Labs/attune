#!/usr/bin/env python3
"""Apply a versioned weak-negative-control policy to an existing feature manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attune.training.controls import build_control_manifest, load_control_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    report = build_control_manifest(
        arguments.source,
        arguments.output,
        load_control_policy(arguments.policy),
    )
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
