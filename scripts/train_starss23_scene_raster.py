#!/usr/bin/env python3
"""Train one frozen-frame laughter MLP on STARSS23 60-second scene rasters."""

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
DURATION_MS = 60_000
VALIDATION_ROOMS = {"sony-room21", "tau-room6"}
SEGMENT_MARGIN_REQUIRED = 0.05
COLLAR_F1_REQUIRED = 0.25
PRIOR_40_EPOCH_PASS = {
    "epochs_completed": 40,
    "segment_f1": 0.4794007490636704,
    "whole_clip_segment_f1": 0.17212490479817213,
    "collar_event_f1": 0.10738255033557047,
    "collar_true_positive": 8,
    "collar_false_positive": 93,
    "collar_false_negative": 40,
    "reference_event_count": 48,
    "decoder": {
        "high_threshold": 0.95,
        "low_threshold": 0.855,
        "max_gap_frames": 2,
        "min_active_frames": 3,
    },
}
GOLD_DURATION_PERCENTILES = (10, 25, 50)
DECODER_HIGH_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
DECODER_LOW_RATIOS = (0.5, 0.7, 0.9)
DECODER_GAP_FRAMES = (0, 2, 4, 8, 12)
DECODER_MEDIAN_WINDOWS = (1, 3, 5)


def should_wire_starss23_timestamps(
    *,
    segment_f1: float,
    whole_clip_segment_f1: float,
    collar_f1: float,
) -> bool:
    """Require collar F1 and a clear segment margin; do not lower the bar after seeing scores."""
    margin = segment_f1 - whole_clip_segment_f1
    return margin >= SEGMENT_MARGIN_REQUIRED and collar_f1 >= COLLAR_F1_REQUIRED


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


def positive_class_weight_from_train_frames(train_targets: Any) -> Any:
    """Return BCE `pos_weight` from TRAIN frames only.

    Inspection tensors are not an argument and must never enter this prior.
    """
    if train_targets.ndim != 2 or train_targets.shape[1] != 1:
        raise ValueError("train_targets must have shape (frames, 1)")
    positives = train_targets.sum()
    negatives = train_targets.numel() - positives
    return (negatives / positives.clamp_min(1)).reshape(1)


def gold_event_durations_ms(rows: list[dict[str, Any]]) -> list[int]:
    return [int(event["end_ms"] - event["start_ms"]) for row in rows for event in row["events"]]


def duration_percentile_ms(durations_ms: list[int], percentile: float) -> int:
    if not durations_ms:
        raise ValueError("cannot compute a duration percentile over no events")
    ordered = sorted(durations_ms)
    index = min(len(ordered) - 1, max(0, round((percentile / 100) * (len(ordered) - 1))))
    return ordered[index]


def min_active_frames_from_gold(
    durations_ms: list[int],
    *,
    frame_hop_ms: float,
    percentiles: tuple[int, ...] = GOLD_DURATION_PERCENTILES,
) -> tuple[int, ...]:
    """Map TRAIN gold length percentiles to decoder minimum-duration candidates."""
    frames = [
        max(1, round(duration_percentile_ms(durations_ms, percentile) / frame_hop_ms))
        for percentile in percentiles
    ]
    return tuple(sorted(set(frames)))


def duration_summary(durations_ms: list[int]) -> dict[str, float | int | None]:
    if not durations_ms:
        return {
            "count": 0,
            "min_ms": None,
            "p25_ms": None,
            "median_ms": None,
            "p75_ms": None,
            "max_ms": None,
            "mean_ms": None,
            "shorter_than_300ms": 0,
            "shorter_than_600ms": 0,
        }
    ordered = sorted(durations_ms)
    return {
        "count": len(ordered),
        "min_ms": ordered[0],
        "p25_ms": duration_percentile_ms(ordered, 25),
        "median_ms": duration_percentile_ms(ordered, 50),
        "p75_ms": duration_percentile_ms(ordered, 75),
        "max_ms": ordered[-1],
        "mean_ms": sum(ordered) / len(ordered),
        "shorter_than_300ms": sum(duration < 300 for duration in ordered),
        "shorter_than_600ms": sum(duration < 600 for duration in ordered),
    }


def duration_error_table(
    references: list[list[dict[str, Any]]],
    predictions: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Collar-match predicted vs gold events and summarize durations. No audio."""
    true_positive_gold: list[int] = []
    true_positive_predicted: list[int] = []
    false_positive: list[int] = []
    false_negative: list[int] = []
    gold = gold_event_durations_ms([{"events": clip} for clip in references])
    predicted = gold_event_durations_ms([{"events": clip} for clip in predictions])
    for reference, prediction in zip(references, predictions, strict=True):
        unmatched = set(range(len(reference)))
        for candidate in prediction:
            eligible = []
            for index in unmatched:
                target = reference[index]
                duration = target["end_ms"] - target["start_ms"]
                offset_collar = max(200, round(duration * 0.2))
                onset_error = abs(candidate["start_ms"] - target["start_ms"])
                offset_error = abs(candidate["end_ms"] - target["end_ms"])
                if (
                    candidate["label"] == target["label"]
                    and onset_error <= 200
                    and offset_error <= offset_collar
                ):
                    eligible.append((onset_error + offset_error, index))
            if not eligible:
                false_positive.append(int(candidate["end_ms"] - candidate["start_ms"]))
                continue
            _, matched = min(eligible)
            unmatched.remove(matched)
            true_positive_gold.append(
                int(reference[matched]["end_ms"] - reference[matched]["start_ms"])
            )
            true_positive_predicted.append(int(candidate["end_ms"] - candidate["start_ms"]))
        for index in unmatched:
            event = reference[index]
            false_negative.append(int(event["end_ms"] - event["start_ms"]))
    gold_summary = duration_summary(gold)
    false_positive_summary = duration_summary(false_positive)
    gold_p25 = gold_summary["p25_ms"]
    shorter_than_gold_p25 = (
        None if gold_p25 is None else sum(duration < gold_p25 for duration in false_positive)
    )
    return {
        "matching": "DCASE-style 200 ms onset and duration-aware offset collar",
        "gold": gold_summary,
        "predicted": duration_summary(predicted),
        "true_positive": {
            **duration_summary(true_positive_predicted),
            "gold_durations_ms": true_positive_gold,
            "predicted_durations_ms": true_positive_predicted,
        },
        "false_positive": {
            **false_positive_summary,
            "durations_ms": false_positive,
            "shorter_than_gold_p25": shorter_than_gold_p25,
            "gold_p25_ms": gold_p25,
        },
        "false_negative": {
            **duration_summary(false_negative),
            "durations_ms": false_negative,
        },
        "false_positives_are_short_fragments": bool(
            false_positive
            and gold_p25 is not None
            and false_positive_summary["median_ms"] is not None
            and false_positive_summary["median_ms"] < gold_p25
        ),
    }


def median_filter_values(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)
    if window % 2 == 0:
        raise ValueError("median filter window must be odd")
    radius = window // 2
    filtered = []
    count = len(values)
    for index in range(count):
        start = max(0, index - radius)
        end = min(count, index + radius + 1)
        ordered = sorted(values[start:end])
        filtered.append(ordered[len(ordered) // 2])
    return filtered


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
    median_filter_frames: int = 1,
) -> list[list[dict[str, Any]]]:
    result = []
    for clip_tensor in probabilities:
        values = median_filter_values(clip_tensor[:, 0].tolist(), median_filter_frames)
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


def decoder_span_kwargs(decoder: dict[str, float | int]) -> dict[str, float | int]:
    return {
        "high_threshold": float(decoder["high_threshold"]),
        "low_threshold": float(decoder["low_threshold"]),
        "max_gap_frames": int(decoder["max_gap_frames"]),
        "min_active_frames": int(decoder["min_active_frames"]),
        "median_filter_frames": int(decoder.get("median_filter_frames", 1)),
    }


def select_hysteresis_decoder(
    probabilities: list[Any],
    references: list[list[dict[str, Any]]],
    *,
    min_active_frames: tuple[int, ...],
    first_frame_center_ms: float,
    frame_hop_ms: float,
) -> dict[str, float | int]:
    """Select decoding on validation rooms only. Min-duration candidates come from train gold."""
    candidates = []
    for high_threshold in DECODER_HIGH_THRESHOLDS:
        for low_ratio in DECODER_LOW_RATIOS:
            for max_gap_frames in DECODER_GAP_FRAMES:
                for minimum in min_active_frames:
                    for median_filter_frames in DECODER_MEDIAN_WINDOWS:
                        decoder = {
                            "high_threshold": high_threshold,
                            "low_threshold": high_threshold * low_ratio,
                            "max_gap_frames": max_gap_frames,
                            "min_active_frames": minimum,
                            "median_filter_frames": median_filter_frames,
                        }
                        predictions = hysteresis_spans(
                            probabilities,
                            **decoder_span_kwargs(decoder),
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
    selected = max(candidates, key=lambda candidate: candidate[:3])[3]
    selected["min_duration_ms"] = round(int(selected["min_active_frames"]) * frame_hop_ms)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensevoice-path", type=Path, required=True)
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=Path("data/manifests/starss23-scene-raster-development.jsonl"),
    )
    parser.add_argument(
        "--inspection-manifest",
        type=Path,
        default=Path("data/manifests/starss23-scene-raster-inspection.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/starss23-scene-raster"),
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("artifacts/starss23-scene-raster/embeddings"),
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=Path("artifacts/starss23-scene-raster/frame-head.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/starss23-scene-raster-results.json"),
    )
    parser.add_argument(
        "--duration-error-output",
        type=Path,
        default=Path("research/error-analysis/starss23-scene-raster-durations.json"),
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=25)
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
    pos_weight = positive_class_weight_from_train_frames(train_targets)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
    train_gold_durations = gold_event_durations_ms(train_rows)
    min_active_frames = min_active_frames_from_gold(
        train_gold_durations,
        frame_hop_ms=encoder.frame_hop_ms,
    )
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
        min_active_frames=min_active_frames,
        **span_arguments,
    )
    predictions = hysteresis_spans(
        inspection_probabilities,
        **decoder_span_kwargs(decoder),
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
    collar_f1 = float(temporal_collar["f1"])
    gate_passed = should_wire_starss23_timestamps(
        segment_f1=float(temporal_segment["f1"]),
        whole_clip_segment_f1=float(baseline_segment["f1"]),
        collar_f1=collar_f1,
    )
    durations = duration_error_table(references, predictions)
    gate = {
        "passed": gate_passed,
        "segment_margin_required": SEGMENT_MARGIN_REQUIRED,
        "segment_margin_observed": margin,
        "collar_f1_required": COLLAR_F1_REQUIRED,
        "collar_f1_observed": collar_f1,
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
    train_positive_frames = int(train_targets.sum().item())
    train_total_frames = int(train_targets.numel())
    payload = {
        "report_version": "1",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "gate_decision": "closed",
        "inspection_evaluations": 1,
        "protocol": "first_60s_scene_raster",
        "task": "STARSS23 v1.1 60-second scene-raster laughter localization",
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
            "patience": arguments.patience,
            "loss": "BCEWithLogitsLoss",
            "positive_class_weight_source": "train_frames_only",
            "train_positive_frames": train_positive_frames,
            "train_total_frames": train_total_frames,
            "train_positive_frame_prior": train_positive_frames / train_total_frames,
            "positive_class_weight": float(pos_weight.reshape(())),
            "history": history,
        },
        "decoder_search": {
            "selected_on": "development validation rooms",
            "min_duration_source": "train gold length percentiles; inspection unused",
            "gold_duration_percentiles": list(GOLD_DURATION_PERCENTILES),
            "train_gold_duration_ms": duration_summary(train_gold_durations),
            "min_active_frames_candidates": list(min_active_frames),
            "max_gap_frames_candidates": list(DECODER_GAP_FRAMES),
            "median_filter_frames_candidates": list(DECODER_MEDIAN_WINDOWS),
            "inspection_used": False,
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
            "duration_error_table": durations,
            "whole_clip_oracle_tag_baseline": {
                "definition": (
                    "uses each scene's known laughter presence but assigns 0..60000 ms; "
                    "this is deliberately not localization"
                ),
                "segment_f1": baseline_segment,
                "collar_event_metrics": baseline_collar,
            },
        },
        "prior_40_epoch_pass": PRIOR_40_EPOCH_PASS,
        "cascade_wiring": {
            **gate,
            "timestamps_wired": gate_passed,
            "policy": (
                "wire STARSS23 laugh timestamps only if collar F1 >= 0.25 "
                "and segment margin >= 0.05"
            ),
            "decision": (
                "wire STARSS23 laugh timing from the 60s scene raster"
                if gate_passed
                else "leave STARSS23 unwired; 60s scene raster did not clear the collar gate"
            ),
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
    error_payload = {
        "report_version": "1",
        "generated_at_utc": payload["generated_at_utc"],
        "gate_decision": "closed",
        "label_status": "human 100 ms activity label; not reviewed Attune gold",
        "language": "unverified",
        "protocol": "first_60s_scene_raster",
        "partition": "official_dev_test_inspection",
        "audio_committed": False,
        "this_pass": durations,
        "this_pass_collar": {
            "true_positive": int(temporal_collar["true_positive"]),
            "false_positive": int(temporal_collar["false_positive"]),
            "false_negative": int(temporal_collar["false_negative"]),
            "f1": collar_f1,
        },
    }
    error_payload["previous_40_epoch_pass"] = {
        "collar_true_positive": PRIOR_40_EPOCH_PASS["collar_true_positive"],
        "collar_false_positive": PRIOR_40_EPOCH_PASS["collar_false_positive"],
        "collar_false_negative": PRIOR_40_EPOCH_PASS["collar_false_negative"],
        "collar_event_f1": PRIOR_40_EPOCH_PASS["collar_event_f1"],
        "segment_f1": PRIOR_40_EPOCH_PASS["segment_f1"],
        "whole_clip_segment_f1": PRIOR_40_EPOCH_PASS["whole_clip_segment_f1"],
        "note": "40-epoch pass did not write a duration table; collar counts only",
    }
    arguments.duration_error_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.duration_error_output.write_text(
        json.dumps(error_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {arguments.output}")
    print(f"Wrote {arguments.duration_error_output}")


if __name__ == "__main__":
    main()
