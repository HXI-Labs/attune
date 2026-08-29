#!/usr/bin/env python3
"""Train a four-way linear FSD50K head over frozen SenseVoice embeddings."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from attune.integrity import file_digest
from attune.models.frozen_event_probe import (
    ProbeDataError,
    discover_vocalsound,
    make_speaker_disjoint_split,
)
from attune.models.frozen_event_probe import (
    inspection_examples as vocalsound_inspection_examples,
)
from attune.models.frozen_event_probe import (
    load_inspection_rows as load_vocalsound_inspection_rows,
)
from attune.models.fsd50k_probe import (
    FSD50K_PROBE_LABELS,
    FSD50KProbeExample,
    classification_metrics,
    inspection_examples,
    training_examples,
    validate_clip_disjoint,
)
from attune.models.probe_abstention import calibrate_abstention, fit_none_logit_head
from attune.models.probe_ood import (
    crema_probe_negatives,
    partition_negatives,
    source_speakers,
)
from attune.models.sensevoice_probe import SENSEVOICE_EMBEDDING, FrozenSenseVoiceEncoder

AED_DETECTION_RATE = {
    "shout": 0.0,
    "whisper": 0.0,
    "sob": 0.32,
    "scream": 0.0,
}
AED_ONE_VS_REST_F1 = {
    "shout": 0.0,
    "whisper": 0.0,
    "sob": 0.48484848484848486,
    "scream": 0.0,
}


def extract_partition(
    examples: tuple[FSD50KProbeExample, ...],
    extractor: FrozenSenseVoiceEncoder,
    torch: Any,
) -> tuple[Any, Any]:
    features = extract_features(examples, extractor, torch)
    indices = {label: index for index, label in enumerate(FSD50K_PROBE_LABELS)}
    targets = torch.tensor([indices[example.label] for example in examples])
    return features, targets


def extract_features(examples: Any, extractor: FrozenSenseVoiceEncoder, torch: Any) -> Any:
    """Materialize frozen embeddings for examples that expose an audio path."""
    missing = [str(example.path) for example in examples if not example.path.is_file()]
    if missing:
        raise ProbeDataError(
            f"{len(missing)} probe audio files are missing; first missing file: {missing[0]}"
        )
    with torch.inference_mode():
        features = torch.stack([extractor(example.path) for example in examples])
    return features


def evaluate(head: Any, features: Any, targets: Any, torch: Any) -> dict[str, Any]:
    head.eval()
    with torch.inference_mode():
        predictions = head(features).argmax(dim=1)
    return classification_metrics(targets.tolist(), predictions.tolist())


def train(arguments: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise ProbeDataError(
            "PyTorch is required; install the project torch and model-runners extras"
        ) from error

    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    candidates = training_examples(arguments.probe_manifest, arguments.probe_cache)
    train_examples = tuple(row for row in candidates if row.partition == "train")
    validation_examples = tuple(row for row in candidates if row.partition == "validation")
    test_examples = inspection_examples(arguments.inspection_manifest, arguments.inspection_cache)
    validate_clip_disjoint(train_examples, validation_examples, test_examples)
    vocalsound_rows = load_vocalsound_inspection_rows(arguments.vocalsound_manifest)
    vocalsound_split = make_speaker_disjoint_split(
        discover_vocalsound(arguments.vocalsound_dataset),
        vocalsound_inspection_examples(arguments.vocalsound_manifest, arguments.vocalsound_cache),
        excluded_speakers={row["speaker_id"] for row in vocalsound_rows},
        validation_fraction=arguments.vocalsound_validation_fraction,
        seed=arguments.seed,
    )
    ood_training = vocalsound_split.train
    ood_validation = vocalsound_split.validation
    crema_negatives = crema_probe_negatives(
        arguments.crema_ood_manifest,
        arguments.crema_ood_cache,
        excluded_speakers=source_speakers(
            (arguments.vocalsound_manifest, arguments.inspection_manifest),
            "CREMA-D",
        ),
    )
    crema_ood_training = partition_negatives(crema_negatives, "train")
    crema_ood_validation = partition_negatives(crema_negatives, "validation")

    extractor = FrozenSenseVoiceEncoder(
        arguments.sensevoice_model,
        arguments.embedding_cache,
        torch,
    )
    train_x, train_y = extract_partition(train_examples, extractor, torch)
    validation_x, validation_y = extract_partition(validation_examples, extractor, torch)
    test_x, test_y = extract_partition(test_examples, extractor, torch)
    ood_training_x = torch.cat(
        (
            extract_features(ood_training, extractor, torch),
            extract_features(crema_ood_training, extractor, torch),
        )
    )
    ood_validation_x = torch.cat(
        (
            extract_features(ood_validation, extractor, torch),
            extract_features(crema_ood_validation, extractor, torch),
        )
    )
    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0).clamp_min(1e-5)
    train_x = (train_x - mean) / scale
    validation_x = (validation_x - mean) / scale
    test_x = (test_x - mean) / scale
    ood_training_x = (ood_training_x - mean) / scale
    ood_validation_x = (ood_validation_x - mean) / scale

    torch.manual_seed(arguments.seed)
    head = torch.nn.Linear(train_x.shape[1], len(FSD50K_PROBE_LABELS))
    optimizer = torch.optim.AdamW(head.parameters(), lr=arguments.learning_rate)
    generator = torch.Generator().manual_seed(arguments.seed)
    best_state = None
    best_validation_loss = math.inf
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, arguments.epochs + 1):
        head.train()
        permutation = torch.randperm(len(train_y), generator=generator)
        total_loss = 0.0
        for start in range(0, len(permutation), arguments.batch_size):
            indices = permutation[start : start + arguments.batch_size]
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(head(train_x[indices]), train_y[indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        head.eval()
        with torch.inference_mode():
            validation_loss = torch.nn.functional.cross_entropy(
                head(validation_x), validation_y
            ).item()
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train_y),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = validation_loss
            best_state = {name: value.detach().clone() for name, value in head.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= arguments.patience:
                break
    if best_state is None:
        raise ProbeDataError("linear-head training did not produce a checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    closed_head = head
    try:
        none_head, none_state, none_history = fit_none_logit_head(
            closed_head=closed_head,
            train_features=train_x,
            train_targets=train_y,
            ood_train_features=ood_training_x,
            validation_features=validation_x,
            validation_targets=validation_y,
            ood_validation_features=ood_validation_x,
            label_count=len(FSD50K_PROBE_LABELS),
            learning_rate=arguments.learning_rate,
            batch_size=arguments.batch_size,
            epochs=arguments.epochs,
            patience=arguments.patience,
            seed=arguments.seed,
            torch=torch,
        )
    except RuntimeError as error:
        raise ProbeDataError(str(error)) from error
    with torch.inference_mode():
        abstention = calibrate_abstention(
            id_logits=closed_head(validation_x),
            id_targets=validation_y,
            ood_logits=closed_head(ood_validation_x),
            label_count=len(FSD50K_PROBE_LABELS),
            torch=torch,
            none_id_logits=none_head(validation_x),
            none_ood_logits=none_head(ood_validation_x),
        )
    selected_head = none_head if abstention["method"] == "none_logit" else closed_head
    selected_state = none_state if abstention["method"] == "none_logit" else best_state
    arguments.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": selected_state,
            "feature_mean": mean,
            "feature_scale": scale,
            "labels": list(FSD50K_PROBE_LABELS),
            "embedding": SENSEVOICE_EMBEDDING,
            "abstention": abstention,
        },
        arguments.checkpoint_output,
    )

    validation_metrics = evaluate(closed_head, validation_x, validation_y, torch)
    test_metrics = evaluate(closed_head, test_x, test_y, torch)

    def partition_report(examples: tuple[FSD50KProbeExample, ...]) -> dict[str, Any]:
        return {
            "clips": len(examples),
            "label_counts": dict(sorted(Counter(row.label for row in examples).items())),
        }

    return {
        "report_version": "1",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "task": "FSD50K source-labelled shout/whisper/sob/scream four-way diagnostic",
        "gate_decision": "closed",
        "encoder_frozen": True,
        "fine_tuning_performed": False,
        "head": {
            "type": (
                "linear_with_none_logit" if abstention["method"] == "none_logit" else "linear"
            ),
            "trainable_parameters": sum(
                parameter.numel() for parameter in selected_head.parameters()
            ),
            "checkpoint_committed": False,
            "abstention": abstention,
            "candidate_trainable_parameters": {
                "closed_set": sum(parameter.numel() for parameter in closed_head.parameters()),
                "closed_set_plus_none_checkpoint": sum(
                    parameter.numel() for parameter in none_head.parameters()
                ),
            },
            "fitted_none_logit_parameters": train_x.shape[1] + 1,
            "closed_set_rows_preserved": True,
        },
        "embedding": extractor.metadata(),
        "data_contract": {
            "probe_manifest": str(arguments.probe_manifest),
            "probe_manifest_sha256": file_digest(arguments.probe_manifest),
            "inspection_manifest": str(arguments.inspection_manifest),
            "inspection_manifest_sha256": file_digest(arguments.inspection_manifest),
            "split": (
                "Clip-disjoint train/validation/100-row inspection test. FSD50K does "
                "not provide speaker IDs; uploader metadata is not a speaker identity."
            ),
            "inspection_rows_excluded_from_training": 100,
            "licences": "clip-level CC0 or CC BY only",
            "cross_target_clips": 0,
            "full_audio_archive_downloaded": False,
        },
        "partitions": {
            "train": partition_report(train_examples),
            "validation": partition_report(validation_examples),
            "inspection_test": partition_report(test_examples),
            "ood_training": {
                "clips": len(ood_training) + len(crema_ood_training),
                "speakers": sorted(
                    {example.speaker_id for example in (*ood_training, *crema_ood_training)}
                ),
                "sources": {
                    "VocalSound": len(ood_training),
                    "CREMA-D": len(crema_ood_training),
                },
                "target": "none logit only",
            },
            "ood_validation": {
                "clips": len(ood_validation) + len(crema_ood_validation),
                "speakers": sorted(
                    {example.speaker_id for example in (*ood_validation, *crema_ood_validation)}
                ),
                "sources": {
                    "VocalSound": len(ood_validation),
                    "CREMA-D": len(crema_ood_validation),
                },
                "expected_probe_annotations": "empty for the FSD50K head",
            },
        },
        "seed": arguments.seed,
        "epochs_completed": len(history),
        "early_stopping_patience": arguments.patience,
        "history": history,
        "none_logit_history": none_history,
        "validation_metrics": validation_metrics,
        "inspection_test_metrics": test_metrics,
        "comparison": {
            label: {
                "frozen_encoder_probe_f1": test_metrics["per_class"][label]["f1"],
                "off_the_shelf_aed_weak_label_detection_rate": AED_DETECTION_RATE[label],
                "off_the_shelf_aed_one_vs_rest_f1": AED_ONE_VS_REST_F1[label],
            }
            for label in FSD50K_PROBE_LABELS
        },
        "comparator_note": (
            "The previously reported AED values 0/0/0.32/0 are weak-label detection "
            "rates, not F1. With eight sob true positives, no sob false positives, "
            "and 17 sob false negatives in the committed raw outputs, sob one-vs-rest "
            "F1 is 0.484848; both metrics are named explicitly."
        ),
        "ontology_boundaries": {
            "scream": "Separate source/probe class; never mapped to Attune shouting.",
            "sob": "FSD50K Crying_and_sobbing maps to sob, never crying_speech.",
            "intensity": "CREMA-D intensity is not used by this probe.",
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "attribution": {
            "SenseVoiceSmall": (
                "FunASR/FunAudioLLM; FunASR Model Open Source License Agreement v1.1"
            ),
            "FSD50K": (
                "Fonseca et al.; annotations CC BY 4.0; individual clips retain "
                "their recorded CC0 or CC BY licences."
            ),
        },
        "limitations": [
            "FSD50K source labels are weak labels, not reviewed Attune gold.",
            "Standalone Freesound clips do not establish speech-embedded style coverage.",
            "No localization or broad open-world OOD coverage is established.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-manifest",
        type=Path,
        default=Path("data/manifests/fsd50k-frozen-probe.jsonl"),
    )
    parser.add_argument("--probe-cache", type=Path, default=Path("data/raw/fsd50k-frozen-probe"))
    parser.add_argument(
        "--inspection-manifest",
        type=Path,
        default=Path("data/manifests/licence-clean-inspection.jsonl"),
    )
    parser.add_argument(
        "--inspection-cache",
        type=Path,
        default=Path("data/raw/licence-clean-inspection"),
    )
    parser.add_argument(
        "--vocalsound-dataset",
        type=Path,
        default=Path("data/raw/vocalsound-16k"),
    )
    parser.add_argument(
        "--vocalsound-manifest",
        type=Path,
        default=Path("data/manifests/inspection-set.jsonl"),
    )
    parser.add_argument(
        "--vocalsound-cache",
        type=Path,
        default=Path("data/raw/inspection-set"),
    )
    parser.add_argument("--vocalsound-validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--crema-ood-manifest",
        type=Path,
        default=Path("data/manifests/crema-probe-ood.jsonl"),
    )
    parser.add_argument(
        "--crema-ood-cache",
        type=Path,
        default=Path("data/raw/crema-probe-ood"),
    )
    parser.add_argument("--sensevoice-model", type=Path, required=True)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("artifacts/cascade-sensevoice-embeddings"),
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=Path("artifacts/fsd50k-event-probe/head.pt"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("artifacts/fsd50k-event-probe/metrics.json"),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    try:
        report = train(arguments)
    except (ProbeDataError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    arguments.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.metrics_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote metrics to {arguments.metrics_output}")
    print(f"Wrote local checkpoint to {arguments.checkpoint_output}")


if __name__ == "__main__":
    main()
