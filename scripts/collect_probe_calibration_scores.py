#!/usr/bin/env python3
"""Recreate PR #18 validation splits and collect frozen-probe logits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from attune.models.frozen_event_probe import (
    discover_vocalsound,
    inspection_examples as vocalsound_inspection_examples,
    load_inspection_rows,
    make_speaker_disjoint_split,
)
from attune.models.fsd50k_probe import (
    inspection_examples as fsd50k_inspection_examples,
)
from attune.models.fsd50k_probe import training_examples as fsd50k_training_examples
from attune.models.probe_inference import (
    FSD50K_LABEL_MAPPING,
    VOCALSOUND_LABEL_MAPPING,
    FrozenEncoderProvider,
    FrozenLinearProbeHead,
)
from attune.models.probe_ood import (
    crema_probe_negatives,
    partition_negatives,
    source_speakers,
)


def score(
    head: FrozenLinearProbeHead,
    examples: Any,
    *,
    component: str,
    targets: list[str],
    ood: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    for example, target in zip(examples, targets, strict=True):
        prediction = head.predict(example.path)
        detail = prediction.diagnostics
        rows.append(
            {
                "component": component,
                "split": "validation",
                "clip_id": f"{'ood-' if ood else ''}{example.path.stem}",
                "labels": detail["calibration_labels"],
                "logits": detail["uncalibrated_logits"],
                "target": target,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensevoice-path", type=Path, required=True)
    parser.add_argument("--vocalsound-probe-checkpoint", type=Path, required=True)
    parser.add_argument("--fsd50k-probe-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("artifacts/cascade-sensevoice-embeddings"),
    )
    parser.add_argument(
        "--vocalsound-dataset",
        type=Path,
        default=Path("data/raw/vocalsound-16k"),
    )
    parser.add_argument(
        "--original-manifest",
        type=Path,
        default=Path("data/manifests/inspection-set.jsonl"),
    )
    parser.add_argument(
        "--original-cache",
        type=Path,
        default=Path("data/raw/inspection-set"),
    )
    parser.add_argument(
        "--fsd50k-manifest",
        type=Path,
        default=Path("data/manifests/fsd50k-frozen-probe.jsonl"),
    )
    parser.add_argument(
        "--fsd50k-cache",
        type=Path,
        default=Path("data/raw/fsd50k-frozen-probe"),
    )
    parser.add_argument(
        "--expansion-manifest",
        type=Path,
        default=Path("data/manifests/licence-clean-inspection.jsonl"),
    )
    parser.add_argument(
        "--expansion-cache",
        type=Path,
        default=Path("data/raw/licence-clean-inspection"),
    )
    parser.add_argument(
        "--crema-manifest",
        type=Path,
        default=Path("data/manifests/crema-probe-ood.jsonl"),
    )
    parser.add_argument(
        "--crema-cache",
        type=Path,
        default=Path("data/raw/crema-probe-ood"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/calibration/probe-validation.jsonl"),
    )
    arguments = parser.parse_args()
    os.environ["ATTUNE_SENSEVOICE_LICENSE_REVIEWED"] = "1"

    vocalsound_rows = load_inspection_rows(arguments.original_manifest)
    vocalsound_split = make_speaker_disjoint_split(
        discover_vocalsound(arguments.vocalsound_dataset),
        vocalsound_inspection_examples(
            arguments.original_manifest,
            arguments.original_cache,
        ),
        excluded_speakers={row["speaker_id"] for row in vocalsound_rows},
        validation_fraction=0.2,
        seed=0,
    )
    fsd50k = fsd50k_training_examples(
        arguments.fsd50k_manifest,
        arguments.fsd50k_cache,
    )
    fsd50k_validation = tuple(
        example for example in fsd50k if example.partition == "validation"
    )
    # Validate that the held-out inspection IDs remain separate from calibration.
    fsd50k_inspection_examples(
        arguments.expansion_manifest,
        arguments.expansion_cache,
    )
    crema = crema_probe_negatives(
        arguments.crema_manifest,
        arguments.crema_cache,
        excluded_speakers=source_speakers(
            (arguments.original_manifest, arguments.expansion_manifest),
            "CREMA-D",
        ),
    )
    crema_validation = partition_negatives(crema, "validation")

    encoder = FrozenEncoderProvider(
        arguments.sensevoice_path,
        arguments.embedding_cache,
    )
    vocalsound_head = FrozenLinearProbeHead(
        name="vocalsound-frozen-linear-probe",
        checkpoint=arguments.vocalsound_probe_checkpoint,
        encoder=encoder,
        label_mapping=VOCALSOUND_LABEL_MAPPING,
    )
    fsd50k_head = FrozenLinearProbeHead(
        name="fsd50k-frozen-linear-probe",
        checkpoint=arguments.fsd50k_probe_checkpoint,
        encoder=encoder,
        label_mapping=FSD50K_LABEL_MAPPING,
    )
    rows = score(
        vocalsound_head,
        vocalsound_split.validation,
        component="vocalsound_probe",
        targets=[example.label.value for example in vocalsound_split.validation],
    )
    vocalsound_ood = (*fsd50k_validation, *crema_validation)
    rows.extend(
        score(
            vocalsound_head,
            vocalsound_ood,
            component="vocalsound_probe",
            targets=["none"] * len(vocalsound_ood),
            ood=True,
        )
    )
    rows.extend(
        score(
            fsd50k_head,
            fsd50k_validation,
            component="fsd50k_probe",
            targets=[example.label for example in fsd50k_validation],
        )
    )
    fsd50k_ood = (*vocalsound_split.validation, *crema_validation)
    rows.extend(
        score(
            fsd50k_head,
            fsd50k_ood,
            component="fsd50k_probe",
            targets=["none"] * len(fsd50k_ood),
            ood=True,
        )
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} validation scores to {arguments.output}")


if __name__ == "__main__":
    main()
