#!/usr/bin/env python3
"""Train a small temporal head on frozen DCASE/SenseVoice bin features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from attune.evaluation.localization import (
    collar_event_metrics,
    segment_f1,
    whole_clip_predictions,
)
from attune.models.sensevoice_probe import (
    TEMPORAL_BINS,
    FrozenSenseVoiceEncoder,
)

LABELS = ("laugh", "cough", "throat_clear")
DURATION_MS = 10_000


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_rows(manifest: Path, cache: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        audio = cache / row["cache_path"]
        if not audio.is_file() or digest(audio) != row["sha256"]:
            raise RuntimeError(f"missing or changed DCASE clip: {audio}")
        row["_audio"] = audio
    return rows


def targets(rows: list[dict[str, Any]], torch: Any) -> Any:
    values = torch.zeros((len(rows), TEMPORAL_BINS, len(LABELS)))
    bin_ms = DURATION_MS / TEMPORAL_BINS
    for clip_index, row in enumerate(rows):
        for event in row["events"]:
            label_index = LABELS.index(event["label"])
            for bin_index in range(TEMPORAL_BINS):
                start = bin_index * bin_ms
                end = (bin_index + 1) * bin_ms
                if event["end_ms"] > start and event["start_ms"] < end:
                    values[clip_index, bin_index, label_index] = 1.0
    return values


def spans(probabilities: Any, threshold: float) -> list[list[dict[str, Any]]]:
    bin_ms = DURATION_MS / TEMPORAL_BINS
    result = []
    for clip in probabilities.tolist():
        clip_spans = []
        for label_index, label in enumerate(LABELS):
            active = [row[label_index] >= threshold for row in clip]
            start = None
            for index, enabled in enumerate([*active, False]):
                if enabled and start is None:
                    start = index
                elif not enabled and start is not None:
                    clip_spans.append(
                        {
                            "label": label,
                            "start_ms": round(start * bin_ms),
                            "end_ms": round(index * bin_ms),
                        }
                    )
                    start = None
        result.append(clip_spans)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensevoice-path", type=Path, required=True)
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=Path("data/manifests/dcase2016-localization-training.jsonl"),
    )
    parser.add_argument(
        "--inspection-manifest",
        type=Path,
        default=Path("data/manifests/dcase2016-localization-inspection.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/dcase2016-localization"),
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("artifacts/dcase-localization/embeddings"),
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=Path("artifacts/dcase-localization/temporal-head.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/dcase-localization-results.json"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    import torch

    os.environ["ATTUNE_SENSEVOICE_LICENSE_REVIEWED"] = "1"
    torch.manual_seed(arguments.seed)
    training_rows = load_rows(arguments.training_manifest, arguments.cache_dir)
    inspection_rows = load_rows(arguments.inspection_manifest, arguments.cache_dir)
    recordings = sorted({row["source_recording"] for row in training_rows})
    validation_recordings = set(recordings[-4:])
    train_rows = [
        row for row in training_rows if row["source_recording"] not in validation_recordings
    ]
    validation_rows = [
        row for row in training_rows if row["source_recording"] in validation_recordings
    ]
    encoder = FrozenSenseVoiceEncoder(
        arguments.sensevoice_path,
        arguments.embedding_cache,
        torch,
    )

    def features(rows: list[dict[str, Any]]) -> Any:
        pooled = torch.stack([encoder(row["_audio"]) for row in rows])
        return pooled[:, : TEMPORAL_BINS * 512].reshape(-1, TEMPORAL_BINS, 512)

    train_x = features(train_rows)
    validation_x = features(validation_rows)
    inspection_x = features(inspection_rows)
    train_y = targets(train_rows, torch)
    validation_y = targets(validation_rows, torch)
    mean = train_x.reshape(-1, 512).mean(dim=0)
    scale = train_x.reshape(-1, 512).std(dim=0).clamp_min(1e-5)
    train_x = (train_x - mean) / scale
    validation_x = (validation_x - mean) / scale
    inspection_x = (inspection_x - mean) / scale
    positives = train_y.reshape(-1, len(LABELS)).sum(dim=0)
    negatives = len(train_rows) * TEMPORAL_BINS - positives
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=negatives / positives.clamp_min(1))
    head = torch.nn.Sequential(
        torch.nn.Linear(512, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, len(LABELS)),
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    best_state = None
    best_loss = math.inf
    stale = 0
    history = []
    for epoch in range(1, arguments.epochs + 1):
        head.train()
        optimizer.zero_grad()
        train_loss = loss_function(head(train_x), train_y)
        train_loss.backward()
        optimizer.step()
        head.eval()
        with torch.inference_mode():
            validation_loss = loss_function(head(validation_x), validation_y).item()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss.item(),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {
                name: value.detach().clone() for name, value in head.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= arguments.patience:
                break
    if best_state is None:
        raise RuntimeError("temporal head training produced no checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        validation_probabilities = torch.sigmoid(head(validation_x))
        inspection_probabilities = torch.sigmoid(head(inspection_x))
    validation_references = [row["events"] for row in validation_rows]
    thresholds = [index / 20 for index in range(1, 20)]
    threshold = max(
        thresholds,
        key=lambda value: segment_f1(
            validation_references,
            spans(validation_probabilities, value),
            duration_ms=DURATION_MS,
        )["f1"],
    )
    references = [row["events"] for row in inspection_rows]
    predictions = spans(inspection_probabilities, threshold)
    baseline = whole_clip_predictions(references, duration_ms=DURATION_MS)
    temporal_segment = segment_f1(references, predictions, duration_ms=DURATION_MS)
    baseline_segment = segment_f1(references, baseline, duration_ms=DURATION_MS)
    temporal_collar = collar_event_metrics(references, predictions)
    baseline_collar = collar_event_metrics(references, baseline)
    arguments.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": best_state,
            "feature_mean": mean,
            "feature_scale": scale,
            "labels": LABELS,
            "threshold": threshold,
            "temporal_bins": TEMPORAL_BINS,
            "encoder_frozen": True,
        },
        arguments.checkpoint_output,
    )
    payload = {
        "report_version": "1",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "gate_decision": "closed",
        "task": "DCASE 2016 Task 2 synthetic strong-label event localization",
        "labels": LABELS,
        "label_status": "synthetic strong onset/offset; not reviewed Attune gold",
        "encoder": encoder.metadata(),
        "encoder_frozen": True,
        "head": {
            "type": "two-layer temporal MLP over 8 frozen encoder bins",
            "trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
            "threshold": threshold,
            "threshold_selected_on": "development validation recordings",
            "checkpoint_committed": False,
        },
        "partitions": {
            "train_clips": len(train_rows),
            "validation_clips": len(validation_rows),
            "inspection_test_clips": len(inspection_rows),
            "validation_recordings": sorted(validation_recordings),
            "source_disjoint_test": True,
        },
        "training": {
            "seed": arguments.seed,
            "epochs_completed": len(history),
            "history": history,
        },
        "inspection_test": {
            "designation": "test; never used for fitting or threshold selection",
            "event_counts": dict(
                sorted(
                    Counter(
                        event["label"] for row in inspection_rows for event in row["events"]
                    ).items()
                )
            ),
            "temporal_head": {
                "segment_f1": temporal_segment,
                "collar_event_metrics": temporal_collar,
            },
            "whole_clip_oracle_tag_baseline": {
                "definition": (
                    "uses each clip's known label set but assigns every event 0..10000 ms; "
                    "this is deliberately not localization"
                ),
                "segment_f1": baseline_segment,
                "collar_event_metrics": baseline_collar,
            },
            "beats_whole_clip_on_segment_f1": (
                temporal_segment["f1"] > baseline_segment["f1"]
            ),
            "beats_whole_clip_on_collar_event_f1": (
                temporal_collar["f1"] > baseline_collar["f1"]
            ),
        },
        "runtime": {
            "device": "cpu",
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "limitations": [
            "The scenes are synthetic office mixtures, not in-the-wild speech.",
            "Eight encoder bins impose coarse 1.25-second boundaries.",
            "Only laugh, cough, and throat-clear overlap the Attune ontology.",
            "The result does not localize the existing VocalSound/FSD50K clip labels.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
