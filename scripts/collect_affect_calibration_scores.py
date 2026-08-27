#!/usr/bin/env python3
"""Collect actor-disjoint validation emotion2vec+ scores for calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from attune.baselines.adapters import AFFECT_LABELS, BaselineInput, Emotion2VecPlusAdapter


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_rows(manifest: Path, cache: Path, split: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = []
    for row in rows:
        target = row.get("target_affect")
        if target is None:
            affect = row.get("intended_attune_labels", {}).get("affect", [])
            target = affect[0] if len(affect) == 1 else None
        if target is None:
            continue
        audio = cache / row["cache_path"]
        if not audio.is_file() or digest(audio) != row["sha256"]:
            raise RuntimeError(f"missing or changed calibration audio: {audio}")
        selected.append({**row, "_audio": audio, "_target": target, "_split": split})
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emotion2vec-path", type=Path, required=True)
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=Path("data/manifests/affect-calibration.jsonl"),
    )
    parser.add_argument(
        "--validation-cache",
        type=Path,
        default=Path("data/raw/affect-calibration"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/calibration/affect-scores.jsonl"),
    )
    parser.add_argument(
        "--inspection-manifest",
        type=Path,
        action="append",
        default=[],
        help="untouched inspection manifest; repeat alongside --inspection-cache",
    )
    parser.add_argument(
        "--inspection-cache",
        type=Path,
        action="append",
        default=[],
        help="cache matching an --inspection-manifest",
    )
    arguments = parser.parse_args()
    if len(arguments.inspection_manifest) != len(arguments.inspection_cache):
        raise SystemExit(
            "error: repeat --inspection-manifest and --inspection-cache the same number of times"
        )
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "MODELSCOPE_OFFLINE": "1",
            "FUNASR_DISABLE_UPDATE": "1",
        }
    )
    rows = load_rows(
        arguments.validation_manifest,
        arguments.validation_cache,
        "validation",
    )
    for manifest, cache in zip(
        arguments.inspection_manifest,
        arguments.inspection_cache,
        strict=True,
    ):
        rows.extend(load_rows(manifest, cache, "inspection_test"))
    adapter = Emotion2VecPlusAdapter(checkpoint=arguments.emotion2vec_path)
    records = []
    for index, row in enumerate(rows, 1):
        prediction = adapter.predict(BaselineInput(audio_path=row["_audio"]))
        distribution = {
            label.value: probability
            for label, probability in prediction.output.affect.categories.items()
        }
        records.append(
            {
                "component": "emotion2vec_plus_affect",
                "split": row["_split"],
                "clip_id": row["clip_id"],
                "labels": list(AFFECT_LABELS),
                "logits": [math.log(max(distribution[label], 1e-12)) for label in AFFECT_LABELS],
                "target": row["_target"],
            }
        )
        print(f"emotion2vec+ calibration {index}/{len(rows)} {row['clip_id']}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
