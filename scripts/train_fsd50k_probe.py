#!/usr/bin/env python3
"""Train a four-way linear FSD50K head over frozen SenseVoice embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from attune.models.frozen_event_probe import ProbeDataError
from attune.models.fsd50k_probe import (
    FSD50K_PROBE_LABELS,
    FSD50KProbeExample,
    classification_metrics,
    inspection_examples,
    training_examples,
    validate_clip_disjoint,
)
from attune.models.sensevoice_probe import SENSEVOICE_EMBEDDING, FrozenSenseVoiceEncoder

AED_DETECTION_RATE = {
    "shout": 0.0,
    "whisper": 0.0,
    "sob": 0.32,
    "scream": 0.0,
}


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def extract_partition(
    examples: tuple[FSD50KProbeExample, ...],
    extractor: FrozenSenseVoiceEncoder,
    torch: Any,
) -> tuple[Any, Any]:
    missing = [str(example.path) for example in examples if not example.path.is_file()]
    if missing:
        raise ProbeDataError(
            f"{len(missing)} probe audio files are missing; first missing file: {missing[0]}"
        )
    indices = {label: index for index, label in enumerate(FSD50K_PROBE_LABELS)}
    with torch.inference_mode():
        features = torch.stack([extractor(example.path) for example in examples])
    targets = torch.tensor([indices[example.label] for example in examples])
    return features, targets


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
    test_examples = inspection_examples(
        arguments.inspection_manifest, arguments.inspection_cache
    )
    validate_clip_disjoint(train_examples, validation_examples, test_examples)

    extractor = FrozenSenseVoiceEncoder(
        arguments.sensevoice_model,
        arguments.embedding_cache,
        torch,
    )
    train_x, train_y = extract_partition(train_examples, extractor, torch)
    validation_x, validation_y = extract_partition(validation_examples, extractor, torch)
    test_x, test_y = extract_partition(test_examples, extractor, torch)
    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0).clamp_min(1e-5)
    train_x = (train_x - mean) / scale
    validation_x = (validation_x - mean) / scale
    test_x = (test_x - mean) / scale

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
            best_state = {
                name: value.detach().clone() for name, value in head.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= arguments.patience:
                break
    if best_state is None:
        raise ProbeDataError("linear-head training did not produce a checkpoint")
    head.load_state_dict(best_state)
    arguments.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": best_state,
            "feature_mean": mean,
            "feature_scale": scale,
            "labels": list(FSD50K_PROBE_LABELS),
            "embedding": SENSEVOICE_EMBEDDING,
        },
        arguments.checkpoint_output,
    )

    validation_metrics = evaluate(head, validation_x, validation_y, torch)
    test_metrics = evaluate(head, test_x, test_y, torch)

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
            "type": "linear",
            "trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
            "checkpoint_committed": False,
        },
        "embedding": extractor.metadata(),
        "data_contract": {
            "probe_manifest": str(arguments.probe_manifest),
            "probe_manifest_sha256": file_sha256(arguments.probe_manifest),
            "inspection_manifest": str(arguments.inspection_manifest),
            "inspection_manifest_sha256": file_sha256(arguments.inspection_manifest),
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
        },
        "seed": arguments.seed,
        "epochs_completed": len(history),
        "early_stopping_patience": arguments.patience,
        "history": history,
        "validation_metrics": validation_metrics,
        "inspection_test_metrics": test_metrics,
        "comparison": {
            label: {
                "frozen_encoder_probe_f1": test_metrics["per_class"][label]["f1"],
                "off_the_shelf_aed_weak_label_detection_rate": AED_DETECTION_RATE[label],
            }
            for label in FSD50K_PROBE_LABELS
        },
        "comparator_note": (
            "The previously reported AED values 0/0/0.32/0 are weak-label detection "
            "rates, not four-way F1. They are retained under that name rather than "
            "silently relabelled as F1."
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
            "No localization, calibration, abstention, or OOD threshold was evaluated.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-manifest",
        type=Path,
        default=Path("data/manifests/fsd50k-frozen-probe.jsonl"),
    )
    parser.add_argument(
        "--probe-cache", type=Path, default=Path("data/raw/fsd50k-frozen-probe")
    )
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
    parser.add_argument("--sensevoice-model", type=Path, required=True)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("artifacts/fsd50k-sensevoice-embeddings"),
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
    arguments.metrics_output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote metrics to {arguments.metrics_output}")
    print(f"Wrote local checkpoint to {arguments.checkpoint_output}")


if __name__ == "__main__":
    main()
