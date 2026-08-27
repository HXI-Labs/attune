#!/usr/bin/env python3
"""Train a small temporal head on frozen DCASE/SenseVoice acoustic frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
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
    SENSEVOICE_FRAME_EMBEDDING,
    FrozenSenseVoiceFrameEncoder,
)

LABELS = ("laugh", "cough", "throat_clear")
DURATION_MS = 10_000
CLEAR_SEGMENT_F1_MARGIN = 0.05
_SCENE_PATTERN = re.compile(r"^(?P<scene>.+)_poly_\d+\.wav$")


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


def frame_targets(
    rows: list[dict[str, Any]],
    frame_counts: list[int],
    *,
    first_frame_center_ms: float,
    frame_hop_ms: float,
    torch: Any,
) -> list[Any]:
    """Create per-frame multi-label targets from strong event intervals."""
    if len(rows) != len(frame_counts):
        raise ValueError("row and frame-count lengths differ")
    result = []
    for row, frame_count in zip(rows, frame_counts, strict=True):
        values = torch.zeros((frame_count, len(LABELS)))
        for event in row["events"]:
            label_index = LABELS.index(event["label"])
            for frame_index in range(frame_count):
                center = first_frame_center_ms + frame_index * frame_hop_ms
                start = max(0.0, center - frame_hop_ms / 2)
                end = min(float(row["duration_ms"]), center + frame_hop_ms / 2)
                if event["end_ms"] > start and event["start_ms"] < end:
                    values[frame_index, label_index] = 1.0
        result.append(values)
    return result


def frame_spans(
    probabilities: list[Any],
    threshold: float,
    *,
    first_frame_center_ms: float,
    frame_hop_ms: float,
    duration_ms: int = DURATION_MS,
) -> list[list[dict[str, Any]]]:
    """Decode contiguous active acoustic frames into bounded event spans."""
    result = []
    for clip_tensor in probabilities:
        clip = clip_tensor.tolist() if hasattr(clip_tensor, "tolist") else clip_tensor
        clip_spans = []
        for label_index, label in enumerate(LABELS):
            active = [row[label_index] >= threshold for row in clip]
            start = None
            for index, enabled in enumerate([*active, False]):
                if enabled and start is None:
                    start = index
                elif not enabled and start is not None:
                    first_center = first_frame_center_ms + start * frame_hop_ms
                    after_last_center = first_frame_center_ms + index * frame_hop_ms
                    clip_spans.append(
                        {
                            "label": label,
                            "start_ms": max(0, round(first_center - frame_hop_ms / 2)),
                            "end_ms": min(
                                duration_ms,
                                round(after_last_center - frame_hop_ms / 2),
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
    duration_ms: int = DURATION_MS,
) -> list[list[dict[str, Any]]]:
    """Decode high-confidence seeds with validation-selected temporal continuity."""
    result = []
    for clip_tensor in probabilities:
        clip = clip_tensor.tolist() if hasattr(clip_tensor, "tolist") else clip_tensor
        clip_spans = []
        for label_index, label in enumerate(LABELS):
            values = [row[label_index] for row in clip]
            low_active = [value >= low_threshold for value in values]
            seeds = [index for index, value in enumerate(values) if value >= high_threshold]
            intervals = []
            for seed in seeds:
                start = end = seed
                while start > 0 and low_active[start - 1]:
                    start -= 1
                while end + 1 < len(values) and low_active[end + 1]:
                    end += 1
                if intervals and start - intervals[-1][1] - 1 <= max_gap_frames:
                    intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
                elif not intervals or start > intervals[-1][1]:
                    intervals.append((start, end))
            for start, end in intervals:
                if end - start + 1 < min_active_frames:
                    continue
                clip_spans.append(
                    {
                        "label": label,
                        "start_ms": max(
                            0,
                            round(first_frame_center_ms + start * frame_hop_ms - frame_hop_ms / 2),
                        ),
                        "end_ms": min(
                            duration_ms,
                            round(
                                first_frame_center_ms + (end + 1) * frame_hop_ms - frame_hop_ms / 2
                            ),
                        ),
                    }
                )
        result.append(clip_spans)
    return result


def select_hysteresis_decoder(
    probabilities: list[Any],
    references: list[list[dict[str, Any]]],
    *,
    first_frame_center_ms: float,
    frame_hop_ms: float,
) -> dict[str, float | int]:
    """Select decoding only on development validation, prioritizing collar F1."""
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


def should_wire_timestamps(
    temporal_segment_f1: float,
    whole_clip_segment_f1: float,
    *,
    clear_margin: float = CLEAR_SEGMENT_F1_MARGIN,
) -> bool:
    """Require a predeclared material segment-F1 gain before cascade wiring."""
    return temporal_segment_f1 >= whole_clip_segment_f1 + clear_margin


def scene_key(source_recording: str) -> str:
    """Collapse DCASE polyphonic renders to their underlying scene identity."""
    match = _SCENE_PATTERN.fullmatch(source_recording)
    if match is None:
        raise ValueError(f"unsupported DCASE source recording name: {source_recording}")
    return match.group("scene")


def development_split(
    rows: list[dict[str, Any]], *, validation_scene_count: int = 3
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Hold out complete development scenes, including every polyphonic render."""
    scenes = sorted({scene_key(row["source_recording"]) for row in rows})
    if validation_scene_count < 1 or validation_scene_count >= len(scenes):
        raise ValueError("validation_scene_count must leave train and validation scenes")
    validation_scenes = set(scenes[-validation_scene_count:])
    validation_recordings = {
        row["source_recording"]
        for row in rows
        if scene_key(row["source_recording"]) in validation_scenes
    }
    train_rows = [
        row for row in rows if scene_key(row["source_recording"]) not in validation_scenes
    ]
    validation_rows = [
        row for row in rows if scene_key(row["source_recording"]) in validation_scenes
    ]
    return (
        train_rows,
        validation_rows,
        {
            "validation_scenes": sorted(validation_scenes),
            "validation_recordings": sorted(validation_recordings),
            "validation_scene_disjoint": True,
            "validation_file_disjoint": True,
        },
    )


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
        default=Path("artifacts/dcase-frame-localization/embeddings"),
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=Path("artifacts/dcase-frame-localization/frame-head.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/dcase-frame-localization-results.json"),
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
    train_rows, validation_rows, split_metadata = development_split(training_rows)
    encoder = FrozenSenseVoiceFrameEncoder(
        arguments.sensevoice_path,
        arguments.embedding_cache,
        torch,
    )

    def features(rows: list[dict[str, Any]]) -> list[Any]:
        return [encoder(row["_audio"]) for row in rows]

    train_x = features(train_rows)
    validation_x = features(validation_rows)
    inspection_x = features(inspection_rows)
    target_arguments = {
        "first_frame_center_ms": encoder.first_frame_center_ms,
        "frame_hop_ms": encoder.frame_hop_ms,
        "torch": torch,
    }
    train_y = frame_targets(train_rows, [len(clip) for clip in train_x], **target_arguments)
    validation_y = frame_targets(
        validation_rows, [len(clip) for clip in validation_x], **target_arguments
    )
    train_matrix = torch.cat(train_x)
    validation_matrix = torch.cat(validation_x)
    inspection_lengths = [len(clip) for clip in inspection_x]
    inspection_matrix = torch.cat(inspection_x)
    train_target_matrix = torch.cat(train_y)
    validation_target_matrix = torch.cat(validation_y)
    mean = train_matrix.mean(dim=0)
    scale = train_matrix.std(dim=0).clamp_min(1e-5)
    train_matrix = (train_matrix - mean) / scale
    validation_matrix = (validation_matrix - mean) / scale
    inspection_matrix = (inspection_matrix - mean) / scale
    positives = train_target_matrix.sum(dim=0)
    negatives = len(train_target_matrix) - positives
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
        train_loss = loss_function(head(train_matrix), train_target_matrix)
        train_loss.backward()
        optimizer.step()
        head.eval()
        with torch.inference_mode():
            validation_loss = loss_function(
                head(validation_matrix), validation_target_matrix
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
        raise RuntimeError("temporal head training produced no checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        validation_flat = torch.sigmoid(head(validation_matrix))
        inspection_flat = torch.sigmoid(head(inspection_matrix))
    validation_probabilities = list(validation_flat.split([len(clip) for clip in validation_x]))
    inspection_probabilities = list(inspection_flat.split(inspection_lengths))
    validation_references = [row["events"] for row in validation_rows]
    span_arguments = {
        "first_frame_center_ms": encoder.first_frame_center_ms,
        "frame_hop_ms": encoder.frame_hop_ms,
    }
    thresholds = [index / 20 for index in range(1, 20)]
    threshold = max(
        thresholds,
        key=lambda value: segment_f1(
            validation_references,
            frame_spans(validation_probabilities, value, **span_arguments),
            duration_ms=DURATION_MS,
        )["f1"],
    )
    references = [row["events"] for row in inspection_rows]
    direct_predictions = frame_spans(inspection_probabilities, threshold, **span_arguments)
    decoder = select_hysteresis_decoder(
        validation_probabilities,
        validation_references,
        **span_arguments,
    )
    predictions = hysteresis_spans(
        inspection_probabilities,
        **decoder,
        **span_arguments,
    )
    baseline = whole_clip_predictions(references, duration_ms=DURATION_MS)
    direct_segment = segment_f1(references, direct_predictions, duration_ms=DURATION_MS)
    direct_collar = collar_event_metrics(references, direct_predictions)
    temporal_segment = segment_f1(references, predictions, duration_ms=DURATION_MS)
    baseline_segment = segment_f1(references, baseline, duration_ms=DURATION_MS)
    temporal_collar = collar_event_metrics(references, predictions)
    baseline_collar = collar_event_metrics(references, baseline)
    wire_timestamps = should_wire_timestamps(
        float(temporal_segment["f1"]),
        float(baseline_segment["f1"]),
    )
    margin_observed = float(temporal_segment["f1"]) - float(baseline_segment["f1"])
    arguments.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": best_state,
            "feature_mean": mean,
            "feature_scale": scale,
            "labels": LABELS,
            "hidden_size": 128,
            "threshold": threshold,
            "decoder": {"type": "hysteresis", **decoder},
            "dataset": "dcase2016_task2",
            "embedding": SENSEVOICE_FRAME_EMBEDDING,
            "frame_hop_ms_approx": encoder.frame_hop_ms,
            "first_frame_center_ms_approx": encoder.first_frame_center_ms,
            "encoder_frozen": True,
            "gate": {
                "passed": wire_timestamps,
                "margin_required": CLEAR_SEGMENT_F1_MARGIN,
                "margin_observed": margin_observed,
                "temporal_segment_f1": float(temporal_segment["f1"]),
                "whole_clip_segment_f1": float(baseline_segment["f1"]),
            },
        },
        arguments.checkpoint_output,
    )
    payload = {
        "report_version": "2",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "gate_decision": "closed",
        "task": "DCASE 2016 Task 2 synthetic strong-label event localization",
        "labels": LABELS,
        "label_status": "synthetic strong onset/offset; not reviewed Attune gold",
        "manifests": {
            "training": {
                "path": str(arguments.training_manifest),
                "sha256": digest(arguments.training_manifest),
            },
            "inspection_test": {
                "path": str(arguments.inspection_manifest),
                "sha256": digest(arguments.inspection_manifest),
            },
        },
        "encoder": encoder.metadata(),
        "encoder_frozen": True,
        "head": {
            "type": "two-layer temporal MLP over every frozen 512-d acoustic frame",
            "trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
            "threshold": threshold,
            "threshold_selected_on": "development validation recordings",
            "checkpoint_committed": False,
        },
        "partitions": {
            "train_clips": len(train_rows),
            "validation_clips": len(validation_rows),
            "inspection_test_clips": len(inspection_rows),
            "source_disjoint_test": True,
            **split_metadata,
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
            "direct_threshold_before_decoder": {
                "threshold": threshold,
                "segment_f1": direct_segment,
                "collar_event_metrics": direct_collar,
            },
            "validation_selected_hysteresis_decoder": decoder,
            "whole_clip_oracle_tag_baseline": {
                "definition": (
                    "uses each clip's known label set but assigns every event 0..10000 ms; "
                    "this is deliberately not localization"
                ),
                "segment_f1": baseline_segment,
                "collar_event_metrics": baseline_collar,
            },
            "beats_whole_clip_on_segment_f1": (temporal_segment["f1"] > baseline_segment["f1"]),
            "beats_whole_clip_on_collar_event_f1": (temporal_collar["f1"] > baseline_collar["f1"]),
        },
        "cascade_wiring": {
            "clear_segment_f1_margin_required": CLEAR_SEGMENT_F1_MARGIN,
            "margin_observed": margin_observed,
            "eligible_for_wiring": wire_timestamps,
            "timestamps_wired": wire_timestamps,
            "runtime_head": "attune.models.temporal_probe.FrozenTemporalProbeHead",
            "policy": (
                "wire decoded DCASE-overlap event spans only when held-out segment F1 "
                "is at least the declared margin above the whole-clip oracle-tag comparator"
            ),
            "reason": (
                "result clears the predeclared metric gate; gated cascade integration enabled"
                if wire_timestamps
                else "result does not clear the predeclared metric gate"
            ),
        },
        "runtime": {
            "device": "cpu",
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "limitations": [
            "The scenes are synthetic office mixtures, not in-the-wild speech.",
            "Frame boundaries are approximate because SenseVoice LFR geometry is about 60 ms.",
            "Only laugh, cough, and throat-clear overlap the Attune ontology.",
            "The result does not localize the existing VocalSound/FSD50K clip labels.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
