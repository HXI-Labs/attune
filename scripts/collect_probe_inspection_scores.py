#!/usr/bin/env python3
"""Collect frozen-probe logits on the opened 310-clip inspection benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from attune.models.probe_inference import (
    FSD50K_LABEL_MAPPING,
    VOCALSOUND_LABEL_MAPPING,
    FrozenEncoderProvider,
    FrozenLinearProbeHead,
)

FSD_TARGETS = {
    "shouting": "shout",
    "whispering": "whisper",
    "sob": "sob",
    "scream": "scream",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensevoice-path", type=Path, required=True)
    parser.add_argument("--query-language", choices=("auto", "en"), default="auto")
    parser.add_argument("--vocalsound-probe-checkpoint", type=Path, required=True)
    parser.add_argument("--fsd50k-probe-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("artifacts/cascade-sensevoice-embeddings"),
    )
    parser.add_argument(
        "--slice",
        nargs=3,
        action="append",
        metavar=("NAME", "MANIFEST", "AUDIO_ROOT"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def target_for(row: dict[str, Any], component: str) -> str:
    labels = row["intended_attune_labels"]
    if component == "vocalsound_probe" and row["source_dataset"] == "VocalSound":
        events = labels["events"]
        if len(events) != 1 or events[0] not in VOCALSOUND_LABEL_MAPPING:
            raise ValueError(f"invalid VocalSound target for {row['clip_id']}")
        return events[0]
    if component == "fsd50k_probe" and row["source_dataset"] == "FSD50K":
        candidates = [*labels["events"], *labels["styles"]]
        if len(candidates) != 1 or candidates[0] not in FSD_TARGETS:
            raise ValueError(f"invalid FSD50K target for {row['clip_id']}")
        return FSD_TARGETS[candidates[0]]
    return "none"


def main() -> None:
    arguments = parse_args()
    os.environ["ATTUNE_SENSEVOICE_LICENSE_REVIEWED"] = "1"
    rows = []
    for _name, manifest, audio_root in arguments.slice:
        manifest_path = Path(manifest)
        for row in (
            json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()
        ):
            row["_audio_path"] = Path(audio_root) / row["cache_path"]
            rows.append(row)
    if len(rows) != 310 or len({row["clip_id"] for row in rows}) != 310:
        raise SystemExit("error: inspection input must contain 310 unique clips")

    encoder = FrozenEncoderProvider(
        arguments.sensevoice_path,
        arguments.embedding_cache,
        query_language=arguments.query_language,
    )
    heads = (
        (
            "vocalsound_probe",
            FrozenLinearProbeHead(
                name="vocalsound-frozen-linear-probe",
                checkpoint=arguments.vocalsound_probe_checkpoint,
                encoder=encoder,
                label_mapping=VOCALSOUND_LABEL_MAPPING,
            ),
        ),
        (
            "fsd50k_probe",
            FrozenLinearProbeHead(
                name="fsd50k-frozen-linear-probe",
                checkpoint=arguments.fsd50k_probe_checkpoint,
                encoder=encoder,
                label_mapping=FSD50K_LABEL_MAPPING,
            ),
        ),
    )
    scores = []
    for component, head in heads:
        for row in rows:
            prediction = head.predict(row["_audio_path"])
            detail = prediction.diagnostics
            scores.append(
                {
                    "component": component,
                    "split": "inspection_test",
                    "clip_id": row["clip_id"],
                    "labels": detail["calibration_labels"],
                    "logits": detail["uncalibrated_logits"],
                    "target": target_for(row, component),
                }
            )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in scores))
    print(f"Wrote {len(scores)} inspection scores to {arguments.output}")


if __name__ == "__main__":
    main()
