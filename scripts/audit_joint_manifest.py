#!/usr/bin/env python3
"""Audit every row and feature tensor in a prepared Attune manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attune.training.audit import audit_joint_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--feature-size", type=int, default=560)
    parser.add_argument("--allow-incomplete-supervision", action="store_true")
    arguments = parser.parse_args()
    report = audit_joint_manifest(
        arguments.manifest,
        expected_feature_size=arguments.feature_size,
        require_evaluation_supervision=not arguments.allow_incomplete_supervision,
    )
    payload = json.dumps(report, indent=2) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
