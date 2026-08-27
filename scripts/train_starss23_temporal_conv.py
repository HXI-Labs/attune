#!/usr/bin/env python3
"""Train one tiny temporal Conv1d head on cached frozen STARSS23 frames."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from train_starss23_localization import (
    DURATION_MS,
    LABELS,
    VALIDATION_ROOMS,
    digest,
    frame_targets,
    hysteresis_spans,
    load_rows,
    select_hysteresis_decoder,
)

from attune.evaluation.localization import (
    collar_event_metrics,
    segment_f1,
    whole_clip_predictions,
)
from attune.models.sensevoice_probe import (
    SENSEVOICE_FRAME_EMBEDDING,
    FrozenSenseVoiceFrameEncoder,
)

SEGMENT_MARGIN_REQUIRED = 0.05
COLLAR_F1_REQUIRED = 0.25
CONV_CHANNELS = 128
KERNEL_SIZE = 7


def build_head(torch: Any) -> Any:
    """Return the single predeclared sub-1M temporal architecture."""
    return torch.nn.Sequential(
        torch.nn.Conv1d(
            512,
            CONV_CHANNELS,
            kernel_size=KERNEL_SIZE,
            padding=KERNEL_SIZE // 2,
        ),
        torch.nn.ReLU(),
        torch.nn.Conv1d(CONV_CHANNELS, 1, kernel_size=1),
    )


def logits(head: Any, features: Any) -> Any:
    """Map ``(N,T,512)`` frames to ``(N,T,1)`` logits."""
    return head(features.transpose(1, 2)).transpose(1, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensevoice-path", type=Path, required=True)
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=Path("data/manifests/starss23-localization-development.jsonl"),
    )
    parser.add_argument(
        "--inspection-manifest",
        type=Path,
        default=Path("data/manifests/starss23-localization-inspection.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/starss23-localization"),
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("artifacts/starss23-localization/embeddings"),
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=Path("artifacts/starss23-temporal-conv-v1/frame-head.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/starss23-temporal-conv-results.json"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    import torch

    os.environ["ATTUNE_SENSEVOICE_LICENSE_REVIEWED"] = "1"
    torch.manual_seed(arguments.seed)
    development = load_rows(arguments.development_manifest, arguments.cache_dir)
    inspection = load_rows(arguments.inspection_manifest, arguments.cache_dir)
    train_rows = [row for row in development if row["room"] not in VALIDATION_ROOMS]
    validation_rows = [row for row in development if row["room"] in VALIDATION_ROOMS]
    if not train_rows or not validation_rows:
        raise RuntimeError("STARSS23 scene split produced an empty partition")
    if {row["room"] for row in train_rows} & {row["room"] for row in validation_rows}:
        raise RuntimeError("STARSS23 validation rooms overlap training")
    if {row["source_recording"] for row in development} & {
        row["source_recording"] for row in inspection
    }:
        raise RuntimeError("STARSS23 inspection files overlap development")

    encoder = FrozenSenseVoiceFrameEncoder(
        arguments.sensevoice_path,
        arguments.embedding_cache,
        torch,
    )

    def features(rows: list[dict[str, Any]]) -> list[Any]:
        return [encoder(row["_audio"]) for row in rows]

    train_frames = features(train_rows)
    validation_frames = features(validation_rows)
    inspection_frames = features(inspection)
    frame_counts = {
        len(frames) for frames in [*train_frames, *validation_frames, *inspection_frames]
    }
    if len(frame_counts) != 1:
        raise RuntimeError("temporal Conv1d requires equal frame counts for fixed windows")
    target_arguments = {
        "first_frame_center_ms": encoder.first_frame_center_ms,
        "frame_hop_ms": encoder.frame_hop_ms,
        "torch": torch,
    }
    train_targets = torch.stack(
        frame_targets(
            train_rows,
            [len(clip) for clip in train_frames],
            **target_arguments,
        )
    )
    validation_targets = torch.stack(
        frame_targets(
            validation_rows,
            [len(clip) for clip in validation_frames],
            **target_arguments,
        )
    )
    train_matrix = torch.stack(train_frames)
    validation_matrix = torch.stack(validation_frames)
    inspection_matrix = torch.stack(inspection_frames)
    mean = train_matrix.reshape(-1, 512).mean(dim=0)
    scale = train_matrix.reshape(-1, 512).std(dim=0).clamp_min(1e-5)
    train_matrix = (train_matrix - mean) / scale
    validation_matrix = (validation_matrix - mean) / scale
    inspection_matrix = (inspection_matrix - mean) / scale
    positives = train_targets.sum()
    negatives = train_targets.numel() - positives
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=(negatives / positives.clamp_min(1)).reshape(1)
    )
    head = build_head(torch)
    trainable_parameters = sum(parameter.numel() for parameter in head.parameters())
    if trainable_parameters >= 1_000_000:
        raise RuntimeError("temporal head exceeds the predeclared parameter budget")
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    best_state = None
    best_loss = math.inf
    stale = 0
    history = []
    for epoch in range(1, arguments.epochs + 1):
        head.train()
        optimizer.zero_grad()
        train_loss = loss_function(logits(head, train_matrix), train_targets)
        train_loss.backward()
        optimizer.step()
        head.eval()
        with torch.inference_mode():
            validation_loss = loss_function(
                logits(head, validation_matrix),
                validation_targets,
            ).item()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss.item(),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {name: value.detach().clone() for name, value in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= arguments.patience:
                break
    if best_state is None:
        raise RuntimeError("STARSS23 temporal Conv1d training produced no checkpoint")
    head.load_state_dict(best_state)
    head.eval()

    # Select the one decoding recipe on validation before touching inspection logits.
    with torch.inference_mode():
        validation_probabilities = list(torch.sigmoid(logits(head, validation_matrix)).unbind(0))
    validation_references = [row["events"] for row in validation_rows]
    span_arguments = {
        "first_frame_center_ms": encoder.first_frame_center_ms,
        "frame_hop_ms": encoder.frame_hop_ms,
    }
    decoder = select_hysteresis_decoder(
        validation_probabilities,
        validation_references,
        **span_arguments,
    )

    # The held-out inspection set is evaluated exactly once below.
    with torch.inference_mode():
        inspection_probabilities = list(torch.sigmoid(logits(head, inspection_matrix)).unbind(0))
    references = [row["events"] for row in inspection]
    predictions = hysteresis_spans(
        inspection_probabilities,
        **decoder,
        **span_arguments,
    )
    baseline = whole_clip_predictions(references, duration_ms=DURATION_MS)
    temporal_segment = segment_f1(references, predictions, duration_ms=DURATION_MS)
    baseline_segment = segment_f1(references, baseline, duration_ms=DURATION_MS)
    temporal_collar = collar_event_metrics(references, predictions)
    baseline_collar = collar_event_metrics(references, baseline)
    segment_margin = float(temporal_segment["f1"]) - float(baseline_segment["f1"])
    collar_f1 = float(temporal_collar["f1"])
    gate_passed = segment_margin >= SEGMENT_MARGIN_REQUIRED and collar_f1 >= COLLAR_F1_REQUIRED
    gate = {
        "passed": gate_passed,
        "segment_margin_required": SEGMENT_MARGIN_REQUIRED,
        "segment_margin_observed": segment_margin,
        "collar_f1_required": COLLAR_F1_REQUIRED,
        "collar_f1_observed": collar_f1,
        "temporal_segment_f1": float(temporal_segment["f1"]),
        "whole_clip_segment_f1": float(baseline_segment["f1"]),
    }
    arguments.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "temporal_conv1d",
            "conv_channels": CONV_CHANNELS,
            "kernel_size": KERNEL_SIZE,
            "head_state_dict": best_state,
            "feature_mean": mean,
            "feature_scale": scale,
            "labels": LABELS,
            "threshold": decoder["high_threshold"],
            "decoder": {"type": "hysteresis", **decoder},
            "dataset": "starss23",
            "embedding": SENSEVOICE_FRAME_EMBEDDING,
            "frame_hop_ms_approx": encoder.frame_hop_ms,
            "first_frame_center_ms_approx": encoder.first_frame_center_ms,
            "encoder_frozen": True,
            "gate": gate,
        },
        arguments.checkpoint_output,
    )
    payload = {
        "report_version": "1",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "gate_decision": "closed",
        "task": "single-pass STARSS23 natural-scene laughter temporal Conv1d",
        "inspection_evaluations": 1,
        "label_mapping": {"STARSS23 class 4 laughter": "Attune laugh"},
        "language": "unverified; STARSS23 metadata has no language field",
        "label_status": "human 100 ms activity label; not reviewed Attune gold",
        "manifests": {
            "development": {
                "path": str(arguments.development_manifest),
                "sha256": digest(arguments.development_manifest),
            },
            "inspection_test": {
                "path": str(arguments.inspection_manifest),
                "sha256": digest(arguments.inspection_manifest),
            },
        },
        "encoder": encoder.metadata(),
        "encoder_frozen": True,
        "head": {
            "architecture": "Conv1d(512,128,kernel_size=7)+ReLU+Conv1d(128,1,kernel_size=1)",
            "trainable_parameters": trainable_parameters,
            "parameter_budget": 1_000_000,
            "checkpoint_namespace": "starss23-temporal-conv-v1",
            "checkpoint_committed": False,
        },
        "partitions": {
            "train_clips": len(train_rows),
            "validation_clips": len(validation_rows),
            "inspection_test_clips": len(inspection),
            "validation_rooms": sorted(VALIDATION_ROOMS),
            "scene_disjoint_validation": True,
            "file_and_room_disjoint_inspection": True,
        },
        "training": {
            "seed": arguments.seed,
            "epochs_completed": len(history),
            "early_stopping": "minimum validation BCE; patience 6",
            "history": history,
        },
        "decoder": {
            **decoder,
            "selected_on": "development validation only",
        },
        "inspection_test": {
            "designation": "evaluated once after model and decoder selection",
            "event_count": sum(len(row["events"]) for row in inspection),
            "temporal_head": {
                "segment_f1": temporal_segment,
                "collar_event_metrics": temporal_collar,
            },
            "whole_clip_oracle_tag_baseline": {
                "segment_f1": baseline_segment,
                "collar_event_metrics": baseline_collar,
                "definition": "oracle laughter presence assigned 0..10000 ms; not localization",
            },
        },
        "cascade_wiring": {
            **gate,
            "timestamps_wired": gate_passed,
            "decision": (
                "retain STARSS23 laugh timing"
                if gate_passed
                else "unwire STARSS23 laugh timing; natural-scene boundaries remain unsolved"
            ),
        },
        "runtime": {
            "device": "cpu",
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "limitations": [
            "This is one architecture and one seed on a bounded natural-scene slice.",
            "Speech language is unverified and speech classes are not targets.",
            "The scientific gold gate remains closed.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
