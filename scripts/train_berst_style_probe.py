#!/usr/bin/env python3
"""Train a conservative shouting head from BERSt speech and speech negatives."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from attune.integrity import file_digest
from attune.models.frozen_event_probe import ProbeDataError
from attune.models.sensevoice_probe import (
    SENSEVOICE_EN_EMBEDDING,
    FrozenSenseVoiceEncoder,
)


@dataclass(frozen=True)
class StyleExample:
    path: Path
    clip_id: str
    split: str
    shouting: bool
    source: str


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise ProbeDataError(f"cannot read manifest {path}: {error}") from error
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ProbeDataError(f"manifest is empty or invalid: {path}")
    return rows


def berst_examples(manifest: Path) -> tuple[StyleExample, ...]:
    examples = []
    for row in load_jsonl(manifest):
        styles = row.get("styles")
        if styles is None:
            continue
        if not isinstance(styles, list) or any(style not in {"shouting"} for style in styles):
            raise ProbeDataError(f"unsupported BERSt style target for {row.get('clip_id')}")
        examples.append(
            StyleExample(
                path=(manifest.parent / row["audio_path"]).resolve(),
                clip_id=str(row["clip_id"]),
                split=str(row["split"]),
                shouting="shouting" in styles,
                source="BERSt",
            )
        )
    return tuple(examples)


def common_voice_examples(manifest: Path, cache_root: Path) -> tuple[StyleExample, ...]:
    split_mapping = {"train": "train", "development": "development"}
    examples = []
    for row in load_jsonl(manifest):
        partition = str(row.get("partition"))
        if partition not in split_mapping:
            continue
        examples.append(
            StyleExample(
                path=(cache_root / row["cache_path"]).resolve(),
                clip_id=str(row["clip_id"]),
                split=split_mapping[partition],
                shouting=False,
                source="Common Voice",
            )
        )
    return tuple(examples)


def bounded_sample(
    examples: tuple[StyleExample, ...], maximum: int, seed: int
) -> tuple[StyleExample, ...]:
    candidates = list(examples)
    random.Random(seed).shuffle(candidates)
    return tuple(candidates[:maximum])


def partition_counts(examples: tuple[StyleExample, ...]) -> dict[str, int]:
    counts = Counter(
        f"{row.source}:{'shouting' if row.shouting else 'ordinary'}" for row in examples
    )
    return dict(sorted(counts.items()))


def extract_features(
    examples: tuple[StyleExample, ...], extractor: FrozenSenseVoiceEncoder, torch: Any
) -> Any:
    missing = [example.path for example in examples if not example.path.is_file()]
    if missing:
        raise ProbeDataError(f"{len(missing)} audio files are missing; first: {missing[0]}")
    features = []
    with torch.inference_mode():
        for index, example in enumerate(examples, start=1):
            features.append(extractor(example.path))
            if index % 100 == 0 or index == len(examples):
                print(f"embedded {index}/{len(examples)}", flush=True)
    return torch.stack(features)


def binary_metrics(targets: Any, scores: Any, threshold: float) -> dict[str, float | int]:
    target_values = [bool(value) for value in targets.tolist()]
    predictions = [float(score) >= threshold for score in scores.tolist()]
    true_positive = sum(
        target and predicted for target, predicted in zip(target_values, predictions, strict=True)
    )
    false_positive = sum(
        not target and predicted
        for target, predicted in zip(target_values, predictions, strict=True)
    )
    false_negative = sum(
        target and not predicted
        for target, predicted in zip(target_values, predictions, strict=True)
    )
    true_negative = sum(
        not target and not predicted
        for target, predicted in zip(target_values, predictions, strict=True)
    )
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive_rate = (
        false_positive / (false_positive + true_negative) if false_positive + true_negative else 0.0
    )
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def select_threshold(
    berst_targets: Any,
    berst_scores: Any,
    external_scores: Any,
) -> dict[str, Any]:
    candidates = {
        -1.0,
        1.0,
        *(float(score) for score in berst_scores.tolist()),
        *(float(score) for score in external_scores.tolist()),
    }
    reports = []
    for threshold in candidates:
        berst = binary_metrics(berst_targets, berst_scores, threshold)
        external_false_positives = sum(
            float(score) >= threshold for score in external_scores.tolist()
        )
        reports.append(
            {
                "threshold": threshold,
                "berst": berst,
                "external_ordinary": {
                    "clips": len(external_scores),
                    "false_positive": external_false_positives,
                    "false_positive_rate": external_false_positives / len(external_scores),
                },
            }
        )
    constrained = [
        report
        for report in reports
        if report["berst"]["false_positive_rate"] <= 0.05
        and report["external_ordinary"]["false_positive"] == 0
    ]
    if not constrained:
        raise ProbeDataError("no validation threshold satisfies the ordinary-speech gates")
    return max(
        constrained,
        key=lambda report: (
            report["berst"]["f1"],
            report["berst"]["recall"],
            report["berst"]["precision"],
            -report["berst"]["false_positive_rate"],
        ),
    )


def train(arguments: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise ProbeDataError("PyTorch is required for probe training") from error

    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    berst = berst_examples(arguments.berst_manifest)
    common_voice = common_voice_examples(
        arguments.common_voice_manifest, arguments.common_voice_cache
    )
    train_shout = bounded_sample(
        tuple(row for row in berst if row.split == "train" and row.shouting),
        arguments.train_per_berst_class,
        arguments.seed,
    )
    train_ordinary = bounded_sample(
        tuple(row for row in berst if row.split == "train" and not row.shouting),
        arguments.train_per_berst_class,
        arguments.seed + 1,
    )
    train_external = bounded_sample(
        tuple(row for row in common_voice if row.split == "train"),
        arguments.train_external_negatives,
        arguments.seed + 2,
    )
    train_examples = (*train_shout, *train_ordinary, *train_external)
    validation_examples = tuple(row for row in berst if row.split == "development") + tuple(
        row for row in common_voice if row.split == "development"
    )
    if not train_examples or not validation_examples:
        raise ProbeDataError("style probe requires non-empty training and development sets")

    extractor = FrozenSenseVoiceEncoder(
        arguments.sensevoice_model,
        arguments.embedding_cache,
        torch,
        query_language="en",
    )
    train_x = extract_features(train_examples, extractor, torch)
    validation_x = extract_features(validation_examples, extractor, torch)
    train_y = torch.tensor([int(row.shouting) for row in train_examples], dtype=torch.long)
    validation_y = torch.tensor(
        [int(row.shouting) for row in validation_examples], dtype=torch.long
    )
    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0).clamp_min(1e-5)
    train_x = (train_x - mean) / scale
    validation_x = (validation_x - mean) / scale

    # Output 0 is the release label and output 1 is the explicit ordinary-speech logit.
    training_targets = 1 - train_y
    validation_targets = 1 - validation_y
    class_counts = torch.bincount(training_targets, minlength=2).float()
    class_weights = len(training_targets) / (2 * class_counts)
    torch.manual_seed(arguments.seed)
    head = torch.nn.Linear(train_x.shape[1], 2)
    optimizer = torch.optim.AdamW(head.parameters(), lr=arguments.learning_rate)
    generator = torch.Generator().manual_seed(arguments.seed)
    best_state = None
    best_loss = math.inf
    stale_epochs = 0
    history = []
    for epoch in range(1, arguments.epochs + 1):
        head.train()
        permutation = torch.randperm(len(training_targets), generator=generator)
        total_loss = 0.0
        for start in range(0, len(permutation), arguments.batch_size):
            indices = permutation[start : start + arguments.batch_size]
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                head(train_x[indices]), training_targets[indices], weight=class_weights
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)
        head.eval()
        with torch.inference_mode():
            validation_loss = float(
                torch.nn.functional.cross_entropy(
                    head(validation_x), validation_targets, weight=class_weights
                ).item()
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(training_targets),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {name: value.detach().clone() for name, value in head.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= arguments.patience:
                break
    if best_state is None:
        raise ProbeDataError("style-head training did not produce a checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(head(validation_x), dim=1)
        scores = probabilities[:, 0] - probabilities[:, 1]
    berst_validation = torch.tensor(
        [row.source == "BERSt" for row in validation_examples], dtype=torch.bool
    )
    selected = select_threshold(
        validation_y[berst_validation],
        scores[berst_validation],
        scores[~berst_validation],
    )
    berst_metrics = selected["berst"]
    gate_passed = (
        berst_metrics["f1"] >= 0.85
        and berst_metrics["precision"] >= 0.80
        and berst_metrics["recall"] >= 0.80
        and berst_metrics["false_positive_rate"] <= 0.05
        and selected["external_ordinary"]["false_positive"] == 0
    )

    checkpoint = {
        "head_state_dict": best_state,
        "feature_mean": mean,
        "feature_scale": scale,
        "labels": ["shout"],
        "embedding": SENSEVOICE_EN_EMBEDDING,
        "abstention": {
            "method": "none_logit",
            "threshold": selected["threshold"],
            "score_rule": "emit when shout probability minus ordinary probability >= threshold",
            "validation": selected,
        },
    }
    arguments.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, arguments.checkpoint_output)
    return {
        "report_version": "1",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "task": "BERSt shouting versus ordinary-speech frozen-encoder probe",
        "gate_passed": gate_passed,
        "encoder_frozen": True,
        "fine_tuning_performed": False,
        "trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "embedding": extractor.metadata(),
        "data": {
            "berst_manifest": str(arguments.berst_manifest),
            "berst_manifest_sha256": file_digest(arguments.berst_manifest),
            "common_voice_manifest": str(arguments.common_voice_manifest),
            "common_voice_manifest_sha256": file_digest(arguments.common_voice_manifest),
            "training_counts": partition_counts(train_examples),
            "validation_counts": partition_counts(validation_examples),
            "sealed_berst_rows_not_opened": sum(row.split == "sealed_test" for row in berst),
        },
        "epochs_completed": len(history),
        "history": history,
        "validation": selected,
        "gate": {
            "f1_minimum": 0.85,
            "precision_minimum": 0.80,
            "recall_minimum": 0.80,
            "false_positive_rate_maximum": 0.05,
        },
        "limitations": [
            "The head detects utterance-level shouting only; it does not localize spans.",
            "Whispering remains disabled.",
            "The BERSt sealed split remains unopened until integration checks pass.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--berst-manifest",
        type=Path,
        default=Path("artifacts/manifests/berst-v0.1-source.jsonl"),
    )
    parser.add_argument(
        "--common-voice-manifest",
        type=Path,
        default=Path("data/manifests/common-voice-replay-v0.1.jsonl"),
    )
    parser.add_argument(
        "--common-voice-cache",
        type=Path,
        default=Path("data/raw/common-voice-replay-v0.1"),
    )
    parser.add_argument("--sensevoice-model", type=Path, required=True)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("artifacts/cascade-sensevoice-en-embeddings"),
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=Path("artifacts/release-candidate/berst-shouting-en-head.pt"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("artifacts/release-candidate/berst-shouting-en-metrics.json"),
    )
    parser.add_argument("--train-per-berst-class", type=int, default=500)
    parser.add_argument("--train-external-negatives", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    report = train(arguments)
    arguments.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.metrics_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["validation"], indent=2, sort_keys=True))
    print(f"gate_passed={report['gate_passed']}")


if __name__ == "__main__":
    main()
