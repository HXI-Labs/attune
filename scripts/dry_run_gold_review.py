#!/usr/bin/env python3
"""Validate the review ledger workflow on generated synthetic fixtures only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from attune.data.gold_review import GoldReviewRecord


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("data/fixtures/semantic_conflict"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/error-analysis/gold-review-fixture-dry-run.jsonl"),
    )
    arguments = parser.parse_args()
    manifest = json.loads(
        (arguments.fixtures / "manifest.json").read_text(encoding="utf-8")
    )
    records = []
    for item in manifest["items"]:
        audio = arguments.fixtures / item["audio"]
        if not audio.is_file():
            raise SystemExit(
                "error: generate synthetic fixtures first with "
                "`python data/fixtures/semantic_conflict/generate_audio.py`"
            )
        records.append(
            GoldReviewRecord.model_validate(
                {
                    "clip_id": item["id"],
                    "audio_sha256": digest(audio),
                    "reviewer": "fixture-dry-run",
                    "reviewed_at_utc": "2026-08-27T00:00:00Z",
                    "source_label_status": "weak_source_or_acted",
                    "transcript": {"decision": "accept"},
                    "affect": {"decision": "accept"},
                    "spans": [],
                    "notes": (
                        "Schema/workflow dry-run only; generated tone/noise is not "
                        "human speech or affect gold."
                    ),
                    "dry_run_fixture": True,
                }
            )
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} fixture dry-run rows to {arguments.output}")


if __name__ == "__main__":
    main()
