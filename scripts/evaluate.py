#!/usr/bin/env python3
"""Run Phase 1 baselines against the synthetic fixture set."""

import argparse
import json
from pathlib import Path

from attune.evaluation.harness import run_fixture_harness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("data/fixtures/semantic_conflict"),
        help="fixture directory containing manifest.json and gold.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/baseline-results.json"),
        help="machine-readable report destination",
    )
    arguments = parser.parse_args()
    report = run_fixture_harness(arguments.fixtures)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {arguments.output}")
    for skipped in report["skipped_runners"]:
        print(f"Skipped {skipped['runner']}: {skipped['reason']}")


if __name__ == "__main__":
    main()
