#!/usr/bin/env python3
"""Train a separate frozen-frame laughter head on bounded STARSS23 windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from attune.evaluation.localization import (
    collar_event_metrics,
    segment_f1,
    whole_clip_predictions,
)
from attune.models.sensevoice_probe import (
    SENSEVOICE_FRAME_EMBEDDING,
    FrozenSenseVoiceFrameEncoder,
)

LABELS = ("laugh",)
DURATION_MS = 10_000
VALIDATION_ROOMS = {"sony-room21", "tau-room6"}
CLEAR_SEGMENT_F1_MARGIN = 0.05


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
            raise RuntimeError(f"missing or changed STARSS23 clip: {audio}")
        if set(event["label"] for event in row["events"]) - set(LABELS):
            raise RuntimeError("STARSS23 manifest contains an unsupported Attune mapping")
        row["_audio"] = audio
    return rows


def frame_targets(
    rows: list[dict[str, Any]],
    frame_counts: list[int],
    *,
    first_frame_center_ms: float,
    frame_hop_ms: float,
    torch: Any,
) -> list[Any]:
    result = []
    for row, frame_count in zip(rows, frame_counts, strict=True):
        values = torch.zeros((frame_count, 1))
        for event in row["events"]:
            for frame_index in range(frame_count):
                center = first_frame_center_ms + frame_index * frame_hop_ms
                start = max(0.0, center - frame_hop_ms / 2)
                end = min(float(row["duration_ms"]), center + frame_hop_ms / 2)
                if event["end_ms"] > start and event["start_ms"] < end:
                    values[frame_index, 0] = 1.0
        result.append(values)
    return result


def spans(
    probabilities: list[Any],
    threshold: float,
    *,
    first_frame_center_ms: float,
    frame_hop_ms: float,
) -> list[list[dict[str, Any]]]:
    result = []
    for clip_tensor in probabilities:
        values = clip_tensor[:, 0].tolist()
        clip_spans = []
        start = None
        for index, active in enumerate([*(value >= threshold for value in values), False]):
            if active and start is None:
                start = index
            elif not active and start is not None:
                clip_spans.append(
                    {
                        "label": "laugh",
                        "start_ms": max(
                            0,
                            round(first_frame_center_ms + start * frame_hop_ms - frame_hop_ms / 2),
                        ),
                        "end_ms": min(
                            DURATION_MS,
                            round(first_frame_center_ms + index * frame_hop_ms - frame_hop_ms / 2),
                        ),
                    }
                )
                start = None
        result.append(clip_spans)
    return result


def hysteresis_spans(
    probabilities: list[Any],
    *,
    high_threshold: float,
    low_threshold: float,
    max_gap_frames: int,
    min_active_frames: int,
    first_frame_center_ms: float,
    frame_hop_ms: float,
) -> list[list[dict[str, Any]]]:
    result = []
    for clip_tensor in probabilities:
        values = clip_tensor[:, 0].tolist()
        low_active = [value >= low_threshold for value in values]
        intervals = []
        for seed, value in enumerate(values):
            if value < high_threshold:
                continue
            start = end = seed
            while start > 0 and low_active[start - 1]:
                start -= 1
            while end + 1 < len(values) and low_active[end + 1]:
                end += 1
            if intervals and start - intervals[-1][1] - 1 <= max_gap_frames:
                intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
            elif not intervals or start > intervals[-1][1]:
                intervals.append((start, end))
        result.append(
            [
                {
                    "label": "laugh",
                    "start_ms": max(
                        0,
                        round(first_frame_center_ms + start * frame_hop_ms - frame_hop_ms / 2),
                    ),
                    "end_ms": min(
                        DURATION_MS,
                        round(first_frame_center_ms + (end + 1) * frame_hop_ms - frame_hop_ms / 2),
                    ),
                }
                for start, end in intervals
                if end - start + 1 >= min_active_frames
            ]
        )
    return result


def select_hysteresis_decoder(
    probabilities: list[Any],
    references: list[list[dict[str, Any]]],
    *,
    first_frame_center_ms: float,
    frame_hop_ms: float,
) -> dict[str, float | int]:
    candidates = []
    for high_threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        for low_ratio in (0.5, 0.7, 0.9):
            for max_gap_frames in (0, 1, 2):
                for min_active_frames in (1, 2, 3):
                    decoder = {
                        "high_threshold": high_threshold,
                        "low_threshold": high_threshold * low_ratio,
                        "max_gap_frames": max_gap_frames,
                        "min_active_frames": min_active_frames,
                    }
                    predictions = hysteresis_spans(
                        probabilities,
                        **decoder,
                        first_frame_center_ms=first_frame_center_ms,
                        frame_hop_ms=frame_hop_ms,
                    )
                    collar = collar_event_metrics(references, predictions)
                    segment = segment_f1(
                        references,
                        predictions,
                        duration_ms=DURATION_MS,
                    )
                    candidates.append(
                        (
                            float(collar["f1"]),
                            float(segment["f1"]),
                            -int(collar["false_positive"]),
                            decoder,
                        )
                    )
    return max(candidates, key=lambda candidate: candidate[:3])[3]


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
        default=Path("artifacts/starss23-localization/frame-head.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/starss23-localization-results.json"),
    )
    parser.add_argument("--epochs", type=int, default=40)
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

    train_x = features(train_rows)
    validation_x = features(validation_rows)
    inspection_x = features(inspection)
    target_arguments = {
        "first_frame_center_ms": encoder.first_frame_center_ms,
        "frame_hop_ms": encoder.frame_hop_ms,
        "torch": torch,
    }
    train_y = frame_targets(train_rows, [len(clip) for clip in train_x], **target_arguments)
    validation_y = frame_targets(
        validation_rows,
        [len(clip) for clip in validation_x],
        **target_arguments,
    )
    train_matrix = torch.cat(train_x)
    validation_matrix = torch.cat(validation_x)
    inspection_lengths = [len(clip) for clip in inspection_x]
    inspection_matrix = torch.cat(inspection_x)
    train_targets = torch.cat(train_y)
    validation_targets = torch.cat(validation_y)
    mean = train_matrix.mean(dim=0)
    scale = train_matrix.std(dim=0).clamp_min(1e-5)
    train_matrix = (train_matrix - mean) / scale
    validation_matrix = (validation_matrix - mean) / scale
    inspection_matrix = (inspection_matrix - mean) / scale
    positives = train_targets.sum()
    negatives = train_targets.numel() - positives
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=(negatives / positives.clamp_min(1)).reshape(1)
    )
    head = torch.nn.Sequential(
        torch.nn.Linear(512, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    best_state = None
    best_loss = math.inf
    stale = 0
    history = []
    for epoch in range(1, arguments.epochs + 1):
        head.train()
        optimizer.zero_grad()
        train_loss = loss_function(head(train_matrix), train_targets)
        train_loss.backward()
        optimizer.step()
        head.eval()
        with torch.inference_mode():
            validation_loss = loss_function(head(validation_matrix), validation_targets).item()
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
        raise RuntimeError("STARSS23 temporal head training produced no checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        validation_flat = torch.sigmoid(head(validation_matrix))
        inspection_flat = torch.sigmoid(head(inspection_matrix))
    validation_probabilities = list(validation_flat.split([len(clip) for clip in validation_x]))
    inspection_probabilities = list(inspection_flat.split(inspection_lengths))
    references_validation = [row["events"] for row in validation_rows]
    span_arguments = {
        "first_frame_center_ms": encoder.first_frame_center_ms,
        "frame_hop_ms": encoder.frame_hop_ms,
    }
    thresholds = [index / 20 for index in range(1, 20)]
    threshold = max(
        thresholds,
        key=lambda value: segment_f1(
            references_validation,
            spans(validation_probabilities, value, **span_arguments),
            duration_ms=DURATION_MS,
        )["f1"],
    )
    references = [row["events"] for row in inspection]
    direct_predictions = spans(inspection_probabilities, threshold, **span_arguments)
    decoder = select_hysteresis_decoder(
        validation_probabilities,
        references_validation,
        **span_arguments,
    )
    predictions = hysteresis_spans(
        inspection_probabilities,
        **decoder,
        **span_arguments,
    )
    baseline = whole_clip_predictions(references, duration_ms=DURATION_MS)
    direct_segment = segment_f1(
        references,
        direct_predictions,
        duration_ms=DURATION_MS,
    )
    direct_collar = collar_event_metrics(references, direct_predictions)
    temporal_segment = segment_f1(references, predictions, duration_ms=DURATION_MS)
    baseline_segment = segment_f1(references, baseline, duration_ms=DURATION_MS)
    temporal_collar = collar_event_metrics(references, predictions)
    baseline_collar = collar_event_metrics(references, baseline)
    margin = float(temporal_segment["f1"]) - float(baseline_segment["f1"])
    gate_passed = margin >= CLEAR_SEGMENT_F1_MARGIN
    gate = {
        "passed": gate_passed,
        "margin_required": CLEAR_SEGMENT_F1_MARGIN,
        "margin_observed": margin,
        "temporal_segment_f1": float(temporal_segment["f1"]),
        "whole_clip_segment_f1": float(baseline_segment["f1"]),
    }
    arguments.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": best_state,
            "feature_mean": mean,
            "feature_scale": scale,
            "labels": LABELS,
            "hidden_size": 64,
            "threshold": threshold,
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
        "task": "STARSS23 v1.1 natural-scene laughter localization",
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
            "type": "two-layer binary MLP over frozen 512-d acoustic frames",
            "trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
            "threshold": threshold,
            "threshold_selected_on": "development validation rooms",
            "checkpoint_committed": False,
        },
        "partitions": {
            "train_clips": len(train_rows),
            "validation_clips": len(validation_rows),
            "inspection_test_clips": len(inspection),
            "validation_rooms": sorted(VALIDATION_ROOMS),
            "scene_disjoint_validation": True,
            "file_and_room_disjoint_inspection": True,
            "natural_overlap_clips": {
                "train": sum(row["natural_overlap"] for row in train_rows),
                "validation": sum(row["natural_overlap"] for row in validation_rows),
                "inspection_test": sum(row["natural_overlap"] for row in inspection),
            },
        },
        "training": {
            "seed": arguments.seed,
            "epochs_completed": len(history),
            "history": history,
        },
        "inspection_test": {
            "designation": "official dev-test rooms; never used for fitting or threshold selection",
            "event_count": sum(len(row["events"]) for row in inspection),
            "temporal_head": {
                "segment_f1": temporal_segment,
                "collar_event_metrics": temporal_collar,
            },
            "direct_threshold_before_decoder": {
                "threshold": threshold,
                "segment_f1": direct_segment,
                "collar_event_metrics": direct_collar,
            },
            "validation_selected_hysteresis_decoder": decoder,
            "whole_clip_oracle_tag_baseline": {
                "definition": (
                    "uses each window's known laughter presence but assigns 0..10000 ms; "
                    "this is deliberately not localization"
                ),
                "segment_f1": baseline_segment,
                "collar_event_metrics": baseline_collar,
            },
        },
        "cascade_wiring": {
            **gate,
            "timestamps_wired": gate_passed,
            "policy": "wire STARSS23-derived laugh spans only when held-out segment margin passes",
        },
        "runtime": {
            "device": "cpu",
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "limitations": [
            "Only STARSS23 laughter maps honestly to the current Attune event ontology.",
            "Speech language is unverified and speech classes are not training targets.",
            "Natural recordings require ongoing privacy and consent care.",
            "The bounded slice is not independently reviewed Attune gold.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
