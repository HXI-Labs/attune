#!/usr/bin/env python3
"""Train the Stage 2 frozen-embedding VocalSound event probe.

The parameter-free log-mel representation is intentionally fixed. Only one
linear five-class classification head receives gradients.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import random
import sys
import wave
from collections import Counter
from pathlib import Path
from typing import Any

from attune.models.frozen_event_probe import (
    EVENT_LABELS,
    ProbeDataError,
    ProbeExample,
    discover_vocalsound,
    inspection_examples,
    load_inspection_rows,
    make_speaker_disjoint_split,
)

DEFAULT_MANIFEST = Path("data/manifests/inspection-set.jsonl")
DEFAULT_INSPECTION_CACHE = Path("data/raw/inspection-set")
DEFAULT_DATASET = Path("data/raw/vocalsound-16k")


def require_torch() -> Any:
    """Import the optional training dependency with an actionable error."""
    try:
        import torch
    except ImportError as error:
        raise ProbeDataError(
            "PyTorch is required for probe training; run `uv sync --extra torch`"
        ) from error
    return torch


def read_pcm_wav(path: Path, torch: Any) -> tuple[Any, int]:
    """Read an uncompressed PCM WAV without adding an audio-library dependency."""
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            sample_width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
    except (OSError, wave.Error) as error:
        raise ProbeDataError(f"cannot read PCM WAV {path}: {error}") from error

    typecodes = {1: "B", 2: "h", 4: "i"}
    if sample_width not in typecodes:
        raise ProbeDataError(f"{path}: unsupported {sample_width * 8}-bit PCM")
    samples = array.array(typecodes[sample_width])
    samples.frombytes(frames)
    if sys.byteorder != "little" and sample_width > 1:
        samples.byteswap()
    waveform = torch.tensor(samples, dtype=torch.float32)
    if sample_width == 1:
        waveform = (waveform - 128.0) / 128.0
    else:
        waveform /= float(2 ** (sample_width * 8 - 1))
    if channels > 1:
        waveform = waveform.reshape(-1, channels).mean(dim=1)
    return waveform, sample_rate


def resample(waveform: Any, source_rate: int, target_rate: int, torch: Any) -> Any:
    """Linearly resample a mono waveform when a source is not already 16 kHz."""
    if source_rate == target_rate:
        return waveform
    output_length = max(1, round(waveform.numel() * target_rate / source_rate))
    return torch.nn.functional.interpolate(
        waveform.reshape(1, 1, -1),
        size=output_length,
        mode="linear",
        align_corners=False,
    ).reshape(-1)


def mel_filterbank(
    torch: Any,
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    device: str,
) -> Any:
    """Construct a fixed triangular mel filter bank."""

    def hz_to_mel(value: Any) -> Any:
        return 2595.0 * torch.log10(1.0 + value / 700.0)

    def mel_to_hz(value: Any) -> Any:
        return 700.0 * (torch.pow(10.0, value / 2595.0) - 1.0)

    frequencies = torch.linspace(0.0, sample_rate / 2, n_fft // 2 + 1, device=device)
    low = hz_to_mel(torch.tensor(20.0, device=device))
    high = hz_to_mel(torch.tensor(sample_rate / 2, device=device))
    edges = mel_to_hz(torch.linspace(low, high, n_mels + 2, device=device))
    filters = torch.zeros((n_mels, frequencies.numel()), device=device)
    for index in range(n_mels):
        rising = (frequencies - edges[index]) / (edges[index + 1] - edges[index])
        falling = (edges[index + 2] - frequencies) / (edges[index + 2] - edges[index + 1])
        filters[index] = torch.clamp(torch.minimum(rising, falling), min=0.0)
    return filters


def frozen_logmel_embedding(
    path: Path,
    torch: Any,
    *,
    sample_rate: int = 16_000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 40,
    temporal_bins: int = 8,
    max_duration_s: float = 10.0,
) -> Any:
    """Return a deterministic fixed-dimensional embedding with no parameters."""
    waveform, source_rate = read_pcm_wav(path, torch)
    waveform = resample(waveform, source_rate, sample_rate, torch)
    waveform = waveform[: round(sample_rate * max_duration_s)]
    if waveform.numel() < n_fft:
        waveform = torch.nn.functional.pad(waveform, (0, n_fft - waveform.numel()))
    window = torch.hann_window(n_fft)
    spectrum = (
        torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )
        .abs()
        .square()
    )
    filters = mel_filterbank(
        torch,
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        device="cpu",
    )
    logmel = torch.log(filters @ spectrum + 1e-6)
    pooled = torch.nn.functional.adaptive_avg_pool1d(logmel, temporal_bins).flatten()
    return torch.cat((pooled, logmel.mean(dim=1), logmel.std(dim=1)))


def extract_partition(
    examples: tuple[ProbeExample, ...],
    torch: Any,
) -> tuple[Any, Any]:
    """Materialize frozen embeddings and integer labels for one partition."""
    missing = [str(example.path) for example in examples if not example.path.is_file()]
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:10])
        raise ProbeDataError(f"{len(missing)} audio files are missing:\n{preview}")
    label_indices = {label: index for index, label in enumerate(EVENT_LABELS)}
    with torch.inference_mode():
        features = torch.stack(
            [frozen_logmel_embedding(example.path, torch) for example in examples]
        )
    labels = torch.tensor([label_indices[example.label] for example in examples])
    return features, labels


def classification_metrics(targets: list[int], predictions: list[int]) -> dict[str, Any]:
    """Compute dependency-free per-class and macro classification metrics."""
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for index, label in enumerate(EVENT_LABELS):
        pairs = zip(targets, predictions, strict=True)
        true_positive = sum(t == index and p == index for t, p in pairs)
        pairs = zip(targets, predictions, strict=True)
        false_positive = sum(t != index and p == index for t, p in pairs)
        pairs = zip(targets, predictions, strict=True)
        false_negative = sum(t == index and p != index for t, p in pairs)
        support = sum(t == index for t in targets)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        f1_values.append(f1)
        per_class[label.value] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    accuracy = sum(t == p for t, p in zip(targets, predictions, strict=True)) / len(targets)
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
    }


def evaluate(head: Any, features: Any, labels: Any, torch: Any) -> dict[str, Any]:
    """Evaluate a classification head over frozen features."""
    head.eval()
    with torch.inference_mode():
        predictions = head(features).argmax(dim=1)
    return classification_metrics(labels.tolist(), predictions.tolist())


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Run speaker-disjoint head training and return a serializable report."""
    torch = require_torch()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    inspection_rows = load_inspection_rows(args.inspection_manifest)
    excluded_speakers = {row["speaker_id"] for row in inspection_rows}
    test_split = None if args.test_set == "inspection" else "held_out_speakers"
    test_examples = inspection_examples(
        args.inspection_manifest,
        args.inspection_cache,
        split=test_split,
    )
    split = make_speaker_disjoint_split(
        discover_vocalsound(args.dataset_dir),
        test_examples,
        excluded_speakers=excluded_speakers,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    if len(split.train) < args.min_train_clips:
        raise ProbeDataError(
            f"training pool has {len(split.train)} clips after inspection-speaker exclusion; "
            f"at least {args.min_train_clips} are required"
        )
    for partition in ("train", "validation", "test"):
        counts = Counter(example.label for example in getattr(split, partition))
        missing_labels = [label.value for label in EVENT_LABELS if not counts[label]]
        if missing_labels:
            raise ProbeDataError(f"{partition} partition lacks labels: {', '.join(missing_labels)}")

    train_x, train_y = extract_partition(split.train, torch)
    validation_x, validation_y = extract_partition(split.validation, torch)
    test_x, test_y = extract_partition(split.test, torch)
    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0).clamp_min(1e-5)
    train_x = (train_x - mean) / scale
    validation_x = (validation_x - mean) / scale
    test_x = (test_x - mean) / scale

    head = torch.nn.Linear(train_x.shape[1], len(EVENT_LABELS))
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    best_state = None
    best_validation_loss = math.inf
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        head.train()
        permutation = torch.randperm(len(train_y), generator=generator)
        total_loss = 0.0
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
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
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise ProbeDataError("training did not produce a checkpoint")
    head.load_state_dict(best_state)
    args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": best_state,
            "feature_mean": mean,
            "feature_scale": scale,
            "labels": [label.value for label in EVENT_LABELS],
            "embedding": "fixed-logmel-v1",
        },
        args.checkpoint_output,
    )

    def partition_report(examples: tuple[ProbeExample, ...]) -> dict[str, Any]:
        return {
            "clips": len(examples),
            "speakers": sorted({example.speaker_id for example in examples}),
            "label_counts": dict(
                sorted(Counter(example.label.value for example in examples).items())
            ),
        }

    return {
        "stage": 2,
        "task": "VocalSound utterance-level 5-way event classification",
        "gate_passed": False,
        "encoder_frozen": True,
        "embedding": {
            "name": "fixed-logmel-v1",
            "trainable_parameters": 0,
            "description": "40-bin log-mel temporal pooling plus per-bin mean/std",
        },
        "head": {
            "type": "linear",
            "trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
        },
        "seed": args.seed,
        "epochs_completed": len(history),
        "early_stopping_patience": args.patience,
        "history": history,
        "partitions": {
            "train": partition_report(split.train),
            "validation": partition_report(split.validation),
            "test": partition_report(split.test),
            "excluded_inspection_speakers": sorted(excluded_speakers),
        },
        "validation_metrics": evaluate(head, validation_x, validation_y, torch),
        "test_metrics": evaluate(head, test_x, test_y, torch),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a linear event head over a frozen log-mel embedding."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--inspection-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--inspection-cache", type=Path, default=DEFAULT_INSPECTION_CACHE)
    parser.add_argument(
        "--test-set",
        choices=("inspection", "held_out_speakers"),
        default="inspection",
        help="use all 80 inspection VocalSound clips or only its held-out-speaker slice",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--min-train-clips", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=Path("artifacts/event-probe/head.pt"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("artifacts/event-probe/metrics.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = train(args)
    except (ProbeDataError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote metrics to {args.metrics_output}")
    print(f"Wrote local checkpoint to {args.checkpoint_output}")


if __name__ == "__main__":
    main()
