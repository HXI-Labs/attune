#!/usr/bin/env python3
"""Train one frozen-frame laughter MLP on tiled STARSS23 60-second scene rasters."""

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
PROTOCOL = "tiled_60s_mean4_mic"
HEADLINE = "mean-4 tiled MLP negative; do not replace reported best 0.1395"
SPLIT_PROTOCOL = "kyoto_val_first60s"
FIRST_60S_INSPECTION_CLIPS = 49
FIRST_60S_INSPECTION_EVENTS = 48
FIRST_60S_VAL_CLIPS = 19
FROZEN_40EPOCH_CHECKPOINT = Path("artifacts/starss23-scene-raster/frame-head-40epoch.pt")
NEW_CHECKPOINT = Path("artifacts/starss23-scene-raster/frame-head-tiled-valfirst60s.pt")
BANNED_OVERWRITE_PATHS = (
    FROZEN_40EPOCH_CHECKPOINT,
    Path("research/error-analysis/starss23-tiled-mean4-negative-results.json"),
    Path("research/error-analysis/starss23-tiled-mean4-negative-durations.json"),
    Path("research/error-analysis/starss23-tiled-mean4-negative-index.json"),
    Path("research/error-analysis/starss23-40epoch-on-first60s-diagnostic.json"),
)
DECODER_LOCK_NOTE = (
    "Decoder locked to 0a27733 (high=0.95, low=0.855, gap=0, min_active=1, "
    "median=3, shift=0). A miss cannot be blamed on tiling vs decoder mismatch."
)
AUDIO_CACHE_SCORED = "data/raw/starss23-scene-raster-tiled"
AUDIO_CACHE_V2_MAX_RMS = "data/raw/starss23-scene-raster-v2"
AUDIO_CACHES = {
    "scored_this_eval": {
        "path": AUDIO_CACHE_SCORED,
        "downmix": "mean_of_4_tetrahedral_mic",
        "note": "hash-verified mean-of-4; this inspection",
    },
    "v2_max_rms_audio_only": {
        "path": AUDIO_CACHE_V2_MAX_RMS,
        "downmix": "max_rms_channel",
        "wav_count": 259,
        "note": "max-RMS audio only; NOT this eval; do not mix into embeddings",
    },
}
MLP_HIDDEN_SIZE = 64
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
GOLD_DURATION_PERCENTILES = (10,)
SHORT_FLOOR_MIN_ACTIVE_FRAMES = (1, 2, 3)
DECODER_HIGH_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
DECODER_LOW_RATIOS = (0.2, 0.3, 0.5, 0.7)
DECODER_GAP_FRAMES = (0, 2, 4, 8)
DECODER_MEDIAN_WINDOWS = (1,)
DECODER_ONSET_SHIFTS_MS = (-180, -120, -60, 0)
OLD_DECODER = {
    "high_threshold": 0.95,
    "low_threshold": 0.855,
    "max_gap_frames": 2,
    "min_active_frames": 3,
    "median_filter_frames": 1,
}
CHECKPOINT_UNWEIGHTED = "40_epoch_unweighted"
CHECKPOINT_POSWEIGHT = "posweight_73_epoch"
PREDECLARED_DECODER = {
    "high_threshold": 0.95,
    "low_threshold": 0.855,
    "max_gap_frames": 0,
    "min_active_frames": 1,
    "median_filter_frames": 3,
    "onset_shift_ms": 0,
}
PRIOR_BEST_COLLAR_F1 = 0.13953488372093023
PRIOR_BEST_PASS = {
    "id": "decoder_validity_0a27733",
    "epochs_completed": 40,
    "segment_f1": 0.512396694214876,
    "whole_clip_segment_f1": 0.17212490479817213,
    "collar_event_f1": PRIOR_BEST_COLLAR_F1,
    "collar_true_positive": 9,
    "collar_false_positive": 72,
    "collar_false_negative": 39,
    "reference_event_count": 48,
    "checkpoint": "artifacts/starss23-scene-raster/frame-head-40epoch.pt",
    "median_onset_error_ms_on_overlapping_misses": 200,
    "decoder": dict(PREDECLARED_DECODER),
}
PRIOR_ONSET_SHIFT_PASS = {
    "segment_f1": 0.37163814180929094,
    "whole_clip_segment_f1": 0.17212490479817213,
    "collar_event_f1": 0.05853658536585366,
    "collar_true_positive": 6,
    "collar_false_positive": 151,
    "collar_false_negative": 42,
    "reference_event_count": 48,
    "median_onset_error_ms_on_overlapping_misses": 360,
    "decoder": {
        "high_threshold": 0.9,
        "low_threshold": 0.63,
        "max_gap_frames": 8,
        "min_active_frames": 1,
        "median_filter_frames": 1,
        "onset_shift_ms": -120,
    },
}
PRIOR_BOUNDARY_WEIGHTED_PASS = {
    "epochs_completed": 214,
    "segment_f1": 0.3673469387755102,
    "whole_clip_segment_f1": 0.17212490479817213,
    "collar_event_f1": 0.058823529411764705,
    "collar_true_positive": 2,
    "collar_false_positive": 18,
    "collar_false_negative": 46,
    "reference_event_count": 48,
    "median_onset_error_ms_on_overlapping_misses": 300,
    "decoder": dict(PREDECLARED_DECODER),
}


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


def keep_prior_best_checkpoint(this_collar_f1: float) -> bool:
    """Keep the 0.1395 40-epoch report unless this inspection strictly beats it."""
    return float(this_collar_f1) <= PRIOR_BEST_COLLAR_F1


def bce_with_logits(logits: Any, targets: Any, torch: Any, *, weight: Any | None = None) -> Any:
    """Mean BCE with logits; optional per-frame weights, never class pos_weight."""
    per_frame = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    if weight is None:
        return per_frame.mean()
    return (per_frame * weight).mean()


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
    """Short-floor min-duration plus train-gold p10, capped at train-gold p25.

    Train-gold p50 is never a minimum-duration candidate. Inspection unused.
    The percentiles argument is accepted for call-site compatibility; only p10
    is gold-derived, and p25 is a cap, not a searched floor.
    """
    del percentiles
    p10_frames = max(1, round(duration_percentile_ms(durations_ms, 10) / frame_hop_ms))
    p25_cap = max(1, round(duration_percentile_ms(durations_ms, 25) / frame_hop_ms))
    candidates = [
        value for value in (*SHORT_FLOOR_MIN_ACTIVE_FRAMES, p10_frames) if 1 <= value <= p25_cap
    ]
    unique = tuple(sorted(set(candidates)))
    if not unique:
        unique = (min(p25_cap, min(SHORT_FLOOR_MIN_ACTIVE_FRAMES)),)
    if max(unique) > p25_cap:
        raise RuntimeError("min_active candidates exceeded train-gold p25")
    return unique


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


def apply_onset_shift(
    predictions: list[list[dict[str, Any]]],
    shift_ms: int,
    *,
    duration_ms: int = DURATION_MS,
) -> list[list[dict[str, Any]]]:
    """Move predicted start_ms; negative values pull onsets earlier. End is unchanged."""
    result = []
    for clip in predictions:
        shifted = []
        for event in clip:
            start = max(0, int(event["start_ms"]) + int(shift_ms))
            end = min(duration_ms, int(event["end_ms"]))
            if start < end:
                shifted.append({**event, "start_ms": start, "end_ms": end})
        result.append(shifted)
    return result


def decode_spans(
    probabilities: list[Any],
    decoder: dict[str, float | int],
    *,
    first_frame_center_ms: float,
    frame_hop_ms: float,
) -> list[list[dict[str, Any]]]:
    predictions = hysteresis_spans(
        probabilities,
        **decoder_span_kwargs(decoder),
        first_frame_center_ms=first_frame_center_ms,
        frame_hop_ms=frame_hop_ms,
    )
    return apply_onset_shift(predictions, int(decoder.get("onset_shift_ms", 0)))


def collar_recall(collar: dict[str, Any]) -> float:
    true_positive = int(collar["true_positive"])
    false_negative = int(collar["false_negative"])
    denominator = true_positive + false_negative
    return true_positive / denominator if denominator else 0.0


def decoder_selection_key(
    collar: dict[str, Any],
    segment: dict[str, Any],
) -> tuple[float, float, float]:
    """Prefer collar F1, then recall, then segment F1. Never minimize FP."""
    return (float(collar["f1"]), collar_recall(collar), float(segment["f1"]))


def clip_probabilities(
    head: Any,
    clips: list[Any],
    mean: Any,
    scale: Any,
    torch: Any,
) -> list[Any]:
    lengths = [len(clip) for clip in clips]
    matrix = (torch.cat(clips) - mean) / scale
    with torch.inference_mode():
        flat = torch.sigmoid(head(matrix))
    return list(flat.split(lengths))


def metrics_from_predictions(
    references: list[list[dict[str, Any]]],
    predictions: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    collar = collar_event_metrics(references, predictions)
    segment = segment_f1(references, predictions, duration_ms=DURATION_MS)
    return {
        "collar_f1": float(collar["f1"]),
        "recall": collar_recall(collar),
        "segment_f1": float(segment["f1"]),
        "true_positive": int(collar["true_positive"]),
        "false_positive": int(collar["false_positive"]),
        "false_negative": int(collar["false_negative"]),
        "collar_event_metrics": collar,
        "segment": segment,
    }


def evaluate_decoder(
    probabilities: list[Any],
    references: list[list[dict[str, Any]]],
    decoder: dict[str, float | int],
    *,
    first_frame_center_ms: float,
    frame_hop_ms: float,
) -> dict[str, Any]:
    predictions = decode_spans(
        probabilities,
        decoder,
        first_frame_center_ms=first_frame_center_ms,
        frame_hop_ms=frame_hop_ms,
    )
    metrics = metrics_from_predictions(references, predictions)
    return {"decoder": dict(decoder), "predictions": predictions, **metrics}


def json_safe_decoder(decoder: dict[str, float | int]) -> dict[str, float | int]:
    skip = {"predictions"}
    return {key: value for key, value in decoder.items() if key not in skip}


def miss_mechanism(
    references: list[list[dict[str, Any]]],
    predictions: list[list[dict[str, Any]]],
    probabilities: list[Any],
    *,
    high_threshold: float,
    first_frame_center_ms: float,
    frame_hop_ms: float,
) -> dict[str, Any]:
    """Separate inspection misses the head never fires from collar/onset failures."""
    never_fired = 0
    decoder_suppressed = 0
    predicted_but_collar_failed = 0
    onset_errors_ms: list[int] = []
    for reference, prediction, clip_tensor in zip(
        references, predictions, probabilities, strict=True
    ):
        values = clip_tensor[:, 0].tolist()
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
            if eligible:
                unmatched.remove(min(eligible)[1])
        for index in unmatched:
            event = reference[index]
            overlapping = [
                pred
                for pred in prediction
                if pred["label"] == event["label"]
                and pred["end_ms"] > event["start_ms"]
                and pred["start_ms"] < event["end_ms"]
            ]
            max_prob = 0.0
            for frame_index, value in enumerate(values):
                center = first_frame_center_ms + frame_index * frame_hop_ms
                start = max(0.0, center - frame_hop_ms / 2)
                end = min(float(DURATION_MS), center + frame_hop_ms / 2)
                if event["end_ms"] > start and event["start_ms"] < end:
                    max_prob = max(max_prob, float(value))
            if overlapping:
                predicted_but_collar_failed += 1
                onset_errors_ms.append(
                    min(abs(pred["start_ms"] - event["start_ms"]) for pred in overlapping)
                )
            elif max_prob >= high_threshold:
                decoder_suppressed += 1
            else:
                never_fired += 1
    false_negative = never_fired + decoder_suppressed + predicted_but_collar_failed
    ordered = sorted(onset_errors_ms)
    median_onset = None if not ordered else ordered[len(ordered) // 2]
    mean_onset = None if not ordered else sum(onset_errors_ms) / len(onset_errors_ms)
    counts = {
        "head_never_fires": never_fired,
        "decoder_suppressed": decoder_suppressed,
        "predicted_but_collar_failed": predicted_but_collar_failed,
    }
    dominant = max(counts, key=lambda name: counts[name]) if false_negative else None
    return {
        "false_negative": false_negative,
        **counts,
        "median_onset_error_ms_on_overlapping_misses": median_onset,
        "mean_onset_error_ms_on_overlapping_misses": mean_onset,
        "dominant_miss": dominant,
        "interpretation": (
            "most missed gold events have no head fire inside the gold span"
            if dominant == "head_never_fires"
            else "most missed gold events were predicted but failed the 200 ms collar"
            if dominant == "predicted_but_collar_failed"
            else "most missed gold events were suppressed by the decoder after the head fired"
            if dominant == "decoder_suppressed"
            else "no false negatives"
        ),
    }


def frame_targets(
    rows: list[dict[str, Any]],
    frame_counts: list[int],
    *,
    first_frame_center_ms: float,
    frame_hop_ms: float,
    torch: Any,
    min_event_duration_ms: int = 0,
) -> list[Any]:
    result = []
    for row, frame_count in zip(rows, frame_counts, strict=True):
        values = torch.zeros((frame_count, 1))
        for event in row["events"]:
            if int(event["end_ms"]) - int(event["start_ms"]) < min_event_duration_ms:
                continue
            for frame_index in range(frame_count):
                center = first_frame_center_ms + frame_index * frame_hop_ms
                start = max(0.0, center - frame_hop_ms / 2)
                end = min(float(row["duration_ms"]), center + frame_hop_ms / 2)
                if event["end_ms"] > start and event["start_ms"] < end:
                    values[frame_index, 0] = 1.0
        result.append(values)
    return result


def build_mlp_head(torch: Any, *, hidden_size: int = MLP_HIDDEN_SIZE) -> Any:
    """Return the frozen-encoder 512→64→1 laugh MLP. No Conv/GRU/attention/CRF."""
    return torch.nn.Sequential(
        torch.nn.Linear(512, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, 1),
    )


def assert_head_is_mlp_not_conv_or_gru(head: Any, torch: Any) -> None:
    """Refuse Conv1d and recurrent layers; this pass trains only the two-layer MLP."""
    if any(isinstance(module, torch.nn.Conv1d) for module in head.modules()):
        raise RuntimeError("STARSS23 MLP head must not contain Conv1d")
    recurrent = (torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN)
    if any(isinstance(module, recurrent) for module in head.modules()):
        raise RuntimeError("STARSS23 MLP head must not contain recurrent layers")
    linears = [module for module in head.modules() if isinstance(module, torch.nn.Linear)]
    if len(linears) != 2 or tuple(linears[0].weight.shape) != (MLP_HIDDEN_SIZE, 512):
        raise RuntimeError("STARSS23 head is not the predeclared 512→64→1 MLP")


def iter_hysteresis_decoder_candidates(
    probabilities: list[Any],
    references: list[list[dict[str, Any]]],
    *,
    min_active_frames: tuple[int, ...],
    first_frame_center_ms: float,
    frame_hop_ms: float,
) -> list[
    tuple[tuple[float, float, float], dict[str, float | int], dict[str, Any], dict[str, Any]]
]:
    """Score the repaired 0a27733 validation-only decoder grid. No onset-shift cells."""
    if not min_active_frames:
        raise ValueError("min_active_frames must not be empty")
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
                            "onset_shift_ms": 0,
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
                                decoder_selection_key(collar, segment),
                                decoder,
                                collar,
                                segment,
                            )
                        )
    return candidates


def select_hysteresis_decoder(
    probabilities: list[Any],
    references: list[list[dict[str, Any]]],
    *,
    min_active_frames: tuple[int, ...],
    first_frame_center_ms: float,
    frame_hop_ms: float,
) -> dict[str, float | int]:
    """Select the 0a27733 decoder grid on validation rooms only.

    Min-duration candidates come from train gold. Inspection unused.
    """
    candidates = iter_hysteresis_decoder_candidates(
        probabilities,
        references,
        min_active_frames=min_active_frames,
        first_frame_center_ms=first_frame_center_ms,
        frame_hop_ms=frame_hop_ms,
    )
    selected = max(candidates, key=lambda candidate: candidate[0])
    decoder = selected[1]
    decoder["min_duration_ms"] = round(int(decoder["min_active_frames"]) * frame_hop_ms)
    decoder["validation_collar_f1"] = float(selected[2]["f1"])
    decoder["validation_recall"] = selected[0][1]
    decoder["validation_segment_f1"] = float(selected[3]["f1"])
    return decoder


def event_count(rows: list[dict[str, Any]]) -> int:
    return sum(len(row["events"]) for row in rows)


def clipped_event_count(rows: list[dict[str, Any]]) -> int:
    return sum(int(row.get("clipped_spanning_event_count", 0)) for row in rows)


def first_60s_subset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """window_start_ms == 0 tiles; truncated spanning events stay in gold."""
    return [row for row in rows if int(row["source_window_start_ms"]) == 0]


def kyoto_train_val_split(
    development: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Train all non-val-room tiles; early-stop on first-60s val rooms only.

    Later tiles from sony-room21 / tau-room6 stay unused. Putting them in train
    would leak the same recording; they are not extra train N.
    """
    train_rows = [row for row in development if row["room"] not in VALIDATION_ROOMS]
    val_room_rows = [row for row in development if row["room"] in VALIDATION_ROOMS]
    early_stop_rows = first_60s_subset(val_room_rows)
    unused_later_rows = [
        row for row in val_room_rows if int(row["source_window_start_ms"]) != 0
    ]
    return train_rows, early_stop_rows, unused_later_rows


def assert_no_val_room_later_tiles_in_train(
    train_rows: list[dict[str, Any]],
    unused_later_rows: list[dict[str, Any]],
) -> None:
    if any(row["room"] in VALIDATION_ROOMS for row in train_rows):
        raise RuntimeError("STARSS23 val-room windows leaked into train")
    train_ids = {row["clip_id"] for row in train_rows}
    unused_ids = {row["clip_id"] for row in unused_later_rows}
    overlap = train_ids & unused_ids
    if overlap:
        raise RuntimeError("STARSS23 val-room later tiles leaked into train")
    train_files = {row["source_recording"] for row in train_rows}
    unused_files = {row["source_recording"] for row in unused_later_rows}
    if train_files & unused_files:
        raise RuntimeError("STARSS23 val-room recordings leaked into train")


def assert_early_stop_is_first_60s_val(
    validation_rows: list[dict[str, Any]],
    *,
    expected_clips: int = FIRST_60S_VAL_CLIPS,
) -> None:
    if not validation_rows:
        raise RuntimeError("STARSS23 early-stop validation is empty")
    if any(int(row["source_window_start_ms"]) != 0 for row in validation_rows):
        raise RuntimeError("STARSS23 early-stop val includes later tiles")
    if any(row["room"] not in VALIDATION_ROOMS for row in validation_rows):
        raise RuntimeError("STARSS23 early-stop val left the locked rooms")
    if {row["room"] for row in validation_rows} != VALIDATION_ROOMS:
        raise RuntimeError("STARSS23 early-stop val drifted from sony-room21/tau-room6")
    if len(validation_rows) != expected_clips:
        raise RuntimeError(
            "STARSS23 early-stop val must be "
            f"{expected_clips} first-60s clips, got {len(validation_rows)}"
        )


def assert_first_60s_inspection_gold(
    inspection: list[dict[str, Any]],
    *,
    clips: int = FIRST_60S_INSPECTION_CLIPS,
    events: int = FIRST_60S_INSPECTION_EVENTS,
) -> None:
    subset = first_60s_subset(inspection)
    if len(subset) != clips:
        raise RuntimeError(f"first-60s inspection clips {len(subset)} != {clips}")
    counted = event_count(subset)
    if counted != events:
        raise RuntimeError(f"first-60s inspection gold {counted} != {events}")


def assert_checkpoint_does_not_overwrite_40epoch(
    checkpoint_output: Path,
    frozen_40epoch: Path,
) -> None:
    if checkpoint_output.resolve() == frozen_40epoch.resolve():
        raise RuntimeError("must not overwrite frame-head-40epoch.pt")
    if checkpoint_output.name == "frame-head-40epoch.pt":
        raise RuntimeError("must not write a file named frame-head-40epoch.pt")


def assert_output_paths_are_not_banned(paths: list[Path]) -> None:
    banned = {path.resolve() for path in BANNED_OVERWRITE_PATHS}
    for path in paths:
        if path.resolve() in banned:
            raise RuntimeError(f"must not overwrite frozen snapshot {path}")


def keep_reported_best_0_1395(
    *,
    first60s_collar_f1: float,
    tiled_gate_passed: bool,
) -> bool:
    """Keep 0.1395 unless first-60s control beats it AND tiled gate passes."""
    return not (bool(tiled_gate_passed) and float(first60s_collar_f1) > PRIOR_BEST_COLLAR_F1)


def result_headline(
    *,
    first60s_collar_f1: float,
    tiled_collar_f1: float,
    tiled_gate_passed: bool,
) -> str:
    if keep_reported_best_0_1395(
        first60s_collar_f1=first60s_collar_f1,
        tiled_gate_passed=tiled_gate_passed,
    ):
        return (
            "Kyoto first-60s-val tiled mean-4 MLP; "
            f"tiled collar {tiled_collar_f1:.4f}; "
            "do not replace reported best 0.1395"
        )
    return (
        "Kyoto first-60s-val tiled mean-4 MLP beat 0.1395 on first-60s control "
        "and passed the tiled inspection gate"
    )


def assert_whole_files_stay_in_one_split(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> None:
    train_files = {row["source_recording"] for row in train_rows}
    validation_files = {row["source_recording"] for row in validation_rows}
    if train_files & validation_files:
        raise RuntimeError("STARSS23 validation files overlap training")


def split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "clips": len(rows),
        "events": event_count(rows),
        "clipped_spanning_events": clipped_event_count(rows),
        "first_60s_clips": len(first_60s_subset(rows)),
        "first_60s_events": event_count(first_60s_subset(rows)),
    }


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
        default=Path("data/raw/starss23-scene-raster-tiled"),
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("artifacts/starss23-scene-raster/embeddings-tiled"),
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=Path("artifacts/starss23-scene-raster/frame-head-tiled-valfirst60s.pt"),
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
    parser.add_argument(
        "--ablation-output",
        type=Path,
        default=Path("research/error-analysis/starss23-decoder-ablation.json"),
    )
    parser.add_argument(
        "--checkpoint-unweighted",
        type=Path,
        default=Path("artifacts/starss23-scene-raster/frame-head-40epoch.pt"),
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    import torch

    os.environ["ATTUNE_SENSEVOICE_LICENSE_REVIEWED"] = "1"
    torch.manual_seed(arguments.seed)
    assert_checkpoint_does_not_overwrite_40epoch(
        arguments.checkpoint_output,
        arguments.checkpoint_unweighted,
    )
    assert_output_paths_are_not_banned(
        [
            arguments.checkpoint_output,
            arguments.output,
            arguments.duration_error_output,
        ]
    )
    development = load_rows(arguments.development_manifest, arguments.cache_dir)
    inspection = load_rows(arguments.inspection_manifest, arguments.cache_dir)
    train_rows, validation_rows, unused_val_later_rows = kyoto_train_val_split(development)
    if not train_rows or not validation_rows:
        raise RuntimeError("STARSS23 scene split produced an empty partition")
    if {row["room"] for row in train_rows} & {row["room"] for row in validation_rows}:
        raise RuntimeError("STARSS23 validation rooms overlap training")
    if {row["source_recording"] for row in development} & {
        row["source_recording"] for row in inspection
    }:
        raise RuntimeError("STARSS23 inspection files overlap development")
    if {row["room"] for row in development} & {row["room"] for row in inspection}:
        raise RuntimeError("STARSS23 inspection rooms overlap development")
    assert_early_stop_is_first_60s_val(validation_rows)
    assert_no_val_room_later_tiles_in_train(train_rows, unused_val_later_rows)
    assert_whole_files_stay_in_one_split(train_rows, validation_rows)
    assert_whole_files_stay_in_one_split(train_rows, unused_val_later_rows)
    assert_first_60s_inspection_gold(inspection)

    encoder = FrozenSenseVoiceFrameEncoder(
        arguments.sensevoice_path,
        arguments.embedding_cache,
        torch,
    )
    if any(parameter.requires_grad for parameter in encoder.model.parameters()):
        raise RuntimeError("SenseVoice encoder must stay frozen while training the MLP head")

    def features(rows: list[dict[str, Any]], name: str) -> list[Any]:
        result = []
        for index, row in enumerate(rows, start=1):
            result.append(encoder(row["_audio"]))
            if index == 1 or index % 10 == 0 or index == len(rows):
                print(
                    f"{name} embeddings {index}/{len(rows)} "
                    f"hits={encoder.cache_hits} misses={encoder.cache_misses}",
                    flush=True,
                )
        return result

    train_x = features(train_rows, "train")
    validation_x = features(validation_rows, "validation")
    target_arguments = {
        "first_frame_center_ms": encoder.first_frame_center_ms,
        "frame_hop_ms": encoder.frame_hop_ms,
        "torch": torch,
    }
    train_y = frame_targets(
        train_rows,
        [len(clip) for clip in train_x],
        **target_arguments,
    )
    validation_y = frame_targets(
        validation_rows,
        [len(clip) for clip in validation_x],
        **target_arguments,
    )
    train_matrix = torch.cat(train_x)
    validation_matrix = torch.cat(validation_x)
    train_targets = torch.cat(train_y)
    validation_targets = torch.cat(validation_y)
    mean = train_matrix.mean(dim=0)
    scale = train_matrix.std(dim=0).clamp_min(1e-5)
    train_matrix = (train_matrix - mean) / scale
    validation_matrix = (validation_matrix - mean) / scale
    head = build_mlp_head(torch)
    assert_head_is_mlp_not_conv_or_gru(head, torch)
    trainable_parameters = sum(parameter.numel() for parameter in head.parameters())
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    best_state = None
    best_loss = math.inf
    stale = 0
    history = []
    for epoch in range(1, arguments.epochs + 1):
        head.train()
        optimizer.zero_grad()
        train_loss = bce_with_logits(head(train_matrix), train_targets, torch)
        train_loss.backward()
        optimizer.step()
        head.eval()
        with torch.inference_mode():
            validation_loss = float(
                bce_with_logits(head(validation_matrix), validation_targets, torch).item()
            )
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
        if epoch == 1 or epoch % 10 == 0 or stale == 0:
            print(
                f"epoch {epoch} train={train_loss.item():.6f} "
                f"val={validation_loss:.6f} best={best_loss:.6f} stale={stale}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("STARSS23 temporal head training produced no checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    span_arguments = {
        "first_frame_center_ms": encoder.first_frame_center_ms,
        "frame_hop_ms": encoder.frame_hop_ms,
    }
    validation_probabilities = clip_probabilities(head, validation_x, mean, scale, torch)
    references_validation = [row["events"] for row in validation_rows]
    train_gold_durations = gold_event_durations_ms(train_rows)
    decoder = {
        **PREDECLARED_DECODER,
        "min_duration_ms": round(
            int(PREDECLARED_DECODER["min_active_frames"]) * encoder.frame_hop_ms
        ),
    }
    print(
        "locked decoder "
        f"high={decoder['high_threshold']} low={decoder['low_threshold']} "
        f"gap={decoder['max_gap_frames']} min_active={decoder['min_active_frames']} "
        f"median={decoder['median_filter_frames']} shift={decoder['onset_shift_ms']}",
        flush=True,
    )
    validation_eval = evaluate_decoder(
        validation_probabilities,
        references_validation,
        decoder,
        **span_arguments,
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
    inspection_x = features(inspection, "inspection")
    inspection_probabilities = clip_probabilities(head, inspection_x, mean, scale, torch)
    references = [row["events"] for row in inspection]
    direct_predictions = spans(inspection_probabilities, threshold, **span_arguments)
    predictions = decode_spans(
        inspection_probabilities,
        decoder,
        **span_arguments,
    )
    baseline = whole_clip_predictions(references, duration_ms=DURATION_MS)
    direct_segment = segment_f1(references, direct_predictions, duration_ms=DURATION_MS)
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
    misses = miss_mechanism(
        references,
        predictions,
        inspection_probabilities,
        high_threshold=float(decoder["high_threshold"]),
        **span_arguments,
    )
    first60_index = [
        index for index, row in enumerate(inspection) if int(row["source_window_start_ms"]) == 0
    ]
    first60_rows = [inspection[index] for index in first60_index]
    first60_probabilities = [inspection_probabilities[index] for index in first60_index]
    first60_references = [row["events"] for row in first60_rows]
    first60_predictions = decode_spans(
        first60_probabilities,
        decoder,
        **span_arguments,
    )
    first60_baseline = whole_clip_predictions(first60_references, duration_ms=DURATION_MS)
    first60_segment = segment_f1(first60_references, first60_predictions, duration_ms=DURATION_MS)
    first60_baseline_segment = segment_f1(
        first60_references, first60_baseline, duration_ms=DURATION_MS
    )
    first60_collar = collar_event_metrics(first60_references, first60_predictions)
    first60_baseline_collar = collar_event_metrics(first60_references, first60_baseline)
    first60_margin = float(first60_segment["f1"]) - float(first60_baseline_segment["f1"])
    first60_collar_f1 = float(first60_collar["f1"])
    first60_misses = miss_mechanism(
        first60_references,
        first60_predictions,
        first60_probabilities,
        high_threshold=float(decoder["high_threshold"]),
        **span_arguments,
    )
    keep_best = keep_reported_best_0_1395(
        first60s_collar_f1=first60_collar_f1,
        tiled_gate_passed=gate_passed,
    )
    headline = result_headline(
        first60s_collar_f1=first60_collar_f1,
        tiled_collar_f1=collar_f1,
        tiled_gate_passed=gate_passed,
    )
    result_status = "positive" if gate_passed else "negative"
    gate = {
        "passed": gate_passed,
        "segment_margin_required": SEGMENT_MARGIN_REQUIRED,
        "segment_margin_observed": margin,
        "collar_f1_required": COLLAR_F1_REQUIRED,
        "collar_f1_observed": collar_f1,
        "temporal_segment_f1": float(temporal_segment["f1"]),
        "whole_clip_segment_f1": float(baseline_segment["f1"]),
        "eval": "tiled_inspection",
        "note": (
            "wiring gate is tiled inspection only; 0.1395 is not the same comparator; "
            + DECODER_LOCK_NOTE
        ),
    }
    arguments.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": best_state,
            "feature_mean": mean,
            "feature_scale": scale,
            "labels": LABELS,
            "hidden_size": MLP_HIDDEN_SIZE,
            "architecture": "mlp",
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
    this_pass = {
        "id": "tiled_mean4_mlp_kyoto_valfirst60s",
        "split_protocol": SPLIT_PROTOCOL,
        "epochs_completed": len(history),
        "segment_f1": float(temporal_segment["f1"]),
        "whole_clip_segment_f1": float(baseline_segment["f1"]),
        "collar_event_f1": collar_f1,
        "collar_true_positive": int(temporal_collar["true_positive"]),
        "collar_false_positive": int(temporal_collar["false_positive"]),
        "collar_false_negative": int(temporal_collar["false_negative"]),
        "reference_event_count": event_count(inspection),
        "checkpoint": str(arguments.checkpoint_output),
        "median_onset_error_ms_on_overlapping_misses": misses[
            "median_onset_error_ms_on_overlapping_misses"
        ],
        "decoder": json_safe_decoder(decoder),
        "comparator_note": (
            "tiled inspection gold is a new event set; do not treat 0.1395 as the same comparator"
        ),
        "do_not_replace_reported_best": keep_best,
        "result_status": result_status,
        "first_60s_collar_f1": first60_collar_f1,
        "decoder_lock_note": DECODER_LOCK_NOTE,
    }
    wiring_decision = (
        "wire STARSS23 laugh timing from the tiled 60s scene raster"
        if gate_passed
        else (
            "leave STARSS23 unwired; tiled mean-4 MLP did not clear the collar/segment gate; "
            "do not replace reported best 0.1395"
        )
    )
    overlapping_onset = {
        "prior_best_median_ms": PRIOR_BEST_PASS["median_onset_error_ms_on_overlapping_misses"],
        "this_pass_median_ms": misses["median_onset_error_ms_on_overlapping_misses"],
        "this_pass_mae_ms": misses["mean_onset_error_ms_on_overlapping_misses"],
        "improved_vs_200ms": (
            misses["median_onset_error_ms_on_overlapping_misses"] is not None
            and misses["median_onset_error_ms_on_overlapping_misses"] < 200
        ),
    }
    prior_clip_event_counts = {
        "train_clips": 43,
        "validation_clips": 19,
        "inspection_test_clips": 49,
        "train_laugh_events": 51,
        "validation_laugh_events": 14,
        "inspection_laugh_events": 48,
        "protocol": "first_60s_scene_raster",
    }
    payload = {
        "report_version": "1",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "gate_decision": "closed" if not gate_passed else "open",
        "inspection_evaluations": 1,
        "protocol": PROTOCOL,
        "split_protocol": SPLIT_PROTOCOL,
        "task": "STARSS23 v1.1 tiled 60-second mean-of-4 MIC laughter localization",
        "headline": headline,
        "result_status": result_status,
        "do_not_replace_reported_best_0.1395": keep_best,
        "decoder_lock_note": DECODER_LOCK_NOTE,
        "audio_caches": AUDIO_CACHES,
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
            "architecture": "mlp",
            "hidden_size": MLP_HIDDEN_SIZE,
            "trainable_parameters": trainable_parameters,
            "threshold": threshold,
            "threshold_selected_on": "first-60s development validation rooms only",
            "checkpoint_committed": False,
            "source_checkpoint": str(arguments.checkpoint_output),
            "control_mlp_checkpoint": str(arguments.checkpoint_unweighted),
            "retrained": True,
        },
        "partitions": {
            "train": split_counts(train_rows),
            "validation": split_counts(validation_rows),
            "inspection_test": split_counts(inspection),
            "validation_rooms": sorted(VALIDATION_ROOMS),
            "scene_disjoint_validation": True,
            "file_and_room_disjoint_inspection": True,
            "whole_file_stays_in_one_split": True,
            "tiles_are_correlated_not_iid": True,
            "natural_overlap_clips": {
                "train": sum(row["natural_overlap"] for row in train_rows),
                "validation": sum(row["natural_overlap"] for row in validation_rows),
                "inspection_test": sum(row["natural_overlap"] for row in inspection),
            },
            "prior_first_60s_counts": prior_clip_event_counts,
            "unused_val_room_later_tiles": split_counts(unused_val_later_rows),
            "kyoto_split": {
                "train_excludes_all_val_room_windows": True,
                "early_stop_first_60s_val_only": True,
                "val_room_later_tiles_unused": True,
                "val_room_later_tiles_in_train": 0,
                "train_clip_count": len(train_rows),
                "early_stop_clip_count": len(validation_rows),
                "unused_later_clip_count": len(unused_val_later_rows),
                "note": (
                    "later tiles of sony-room21/tau-room6 stay unused; "
                    "same recording would leak if they entered train; "
                    "they are not extra train N"
                ),
            },
        },
        "training": {
            "seed": arguments.seed,
            "epochs_completed": len(history),
            "patience": arguments.patience,
            "loss": "unweighted_BCEWithLogitsLoss",
            "positive_class_weight": None,
            "positive_class_weight_source": "unused; 44x pos_weight banned",
            "boundary_weight": None,
            "train_positive_frames": train_positive_frames,
            "train_total_frames": train_total_frames,
            "train_positive_frame_prior": train_positive_frames / train_total_frames,
            "early_stop_on": "unweighted_bce_on_first60s_val_rooms_only",
            "early_stop_clip_count": len(validation_rows),
            "val_room_later_tiles_in_early_stop": 0,
            "val_room_later_tiles_in_train": 0,
            "history": history,
            "retrained": True,
        },
        "decoder_search": {
            "performed": False,
            "reason": "decoder-grid path is exhausted; predeclared 0a27733 decoder",
            "decoder_lock_note": DECODER_LOCK_NOTE,
            "selected_on": "not searched; frozen from decoder-validity inspection winner",
            "decoder": json_safe_decoder(decoder),
            "inspection_used": False,
            "train_gold_duration_ms": duration_summary(train_gold_durations),
        },
        "validation_sanity": {
            "decoder": json_safe_decoder(decoder),
            "collar_f1": validation_eval["collar_f1"],
            "recall": validation_eval["recall"],
            "segment_f1": validation_eval["segment_f1"],
            "true_positive": validation_eval["true_positive"],
            "false_positive": validation_eval["false_positive"],
            "false_negative": validation_eval["false_negative"],
            "note": (
                "fixed 0a27733 decoder scored on first-60s val rooms only; "
                "not used to pick a decoder; later val-room tiles unused"
            ),
            "clip_count": len(validation_rows),
        },
        "inspection_test": {
            "designation": "official dev-test rooms; tiled windows; never used for fitting",
            "eval": "tiled_inspection",
            "wiring_gate": True,
            "decoder_lock_note": DECODER_LOCK_NOTE,
            "event_count": event_count(inspection),
            "clipped_spanning_event_count": clipped_event_count(inspection),
            "clip_count": len(inspection),
            "temporal_head": {
                "segment_f1": temporal_segment,
                "collar_event_metrics": temporal_collar,
            },
            "direct_threshold_before_decoder": {
                "threshold": threshold,
                "segment_f1": direct_segment,
                "collar_event_metrics": direct_collar,
            },
            "predeclared_hysteresis_decoder": json_safe_decoder(decoder),
            "duration_error_table": durations,
            "miss_mechanism": misses,
            "overlapping_onset_error_vs_prior_best": overlapping_onset,
            "whole_clip_oracle_tag_baseline": {
                "definition": (
                    "uses each scene's known laughter presence but assigns 0..60000 ms; "
                    "this is deliberately not localization"
                ),
                "segment_f1": baseline_segment,
                "collar_event_metrics": baseline_collar,
            },
        },
        "first_60s_subset": {
            "designation": (
                "window_start_ms == 0 tiles under the new prepare; truncated "
                "t=60s spanning events are kept so gold stays the 48-event control"
            ),
            "clip_count": len(first60_rows),
            "event_count": event_count(first60_rows),
            "clipped_spanning_event_count": clipped_event_count(first60_rows),
            "segment_f1": first60_segment,
            "collar_event_metrics": first60_collar,
            "whole_clip_segment_f1": first60_baseline_segment,
            "whole_clip_collar_event_metrics": first60_baseline_collar,
            "segment_margin": first60_margin,
            "collar_f1": first60_collar_f1,
            "true_positive": int(first60_collar["true_positive"]),
            "false_positive": int(first60_collar["false_positive"]),
            "false_negative": int(first60_collar["false_negative"]),
            "decoder": json_safe_decoder(decoder),
            "miss_mechanism": first60_misses,
            "prior_first_60s_collar_f1": PRIOR_BEST_COLLAR_F1,
            "note": (
                "same locked 0a27733 decoder as tiled inspection; compare vs the old "
                "48-event 0.1395 control. "
                + DECODER_LOCK_NOTE
            ),
        },
        "prior_40_epoch_pass": PRIOR_40_EPOCH_PASS,
        "prior_decoder_validity_pass": {
            "chosen_checkpoint": CHECKPOINT_UNWEIGHTED,
            "segment_f1": 0.512396694214876,
            "whole_clip_segment_f1": 0.17212490479817213,
            "collar_event_f1": PRIOR_BEST_COLLAR_F1,
            "collar_true_positive": 9,
            "collar_false_positive": 72,
            "collar_false_negative": 39,
            "reference_event_count": 48,
            "decoder": dict(PREDECLARED_DECODER),
        },
        "prior_onset_shift_pass": PRIOR_ONSET_SHIFT_PASS,
        "prior_boundary_weighted_pass": PRIOR_BOUNDARY_WEIGHTED_PASS,
        "prior_best": PRIOR_BEST_PASS,
        "this_pass": this_pass,
        "cascade_wiring": {
            **gate,
            "timestamps_wired": gate_passed,
            "policy": (
                "wire STARSS23 laugh timestamps only if tiled inspection collar F1 >= 0.25 "
                "and segment margin >= 0.05"
            ),
            "dcase_unchanged": True,
            "decision": wiring_decision,
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
            "Tiled windows from one file are correlated and are not extra i.i.d. samples.",
            DECODER_LOCK_NOTE,
            "Val-room later tiles were unused for train and early-stop.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    error_payload = {
        "report_version": "1",
        "generated_at_utc": payload["generated_at_utc"],
        "gate_decision": payload["gate_decision"],
        "label_status": "human 100 ms activity label; not reviewed Attune gold",
        "language": "unverified",
        "protocol": PROTOCOL,
        "split_protocol": SPLIT_PROTOCOL,
        "partition": "official_dev_test_inspection",
        "audio_committed": False,
        "loss": "unweighted_BCEWithLogitsLoss",
        "architecture": "mlp",
        "epochs_completed": len(history),
        "predeclared_decoder": json_safe_decoder(decoder),
        "this_pass": durations,
        "this_pass_collar": {
            "true_positive": int(temporal_collar["true_positive"]),
            "false_positive": int(temporal_collar["false_positive"]),
            "false_negative": int(temporal_collar["false_negative"]),
            "f1": collar_f1,
            "segment_f1": float(temporal_segment["f1"]),
            "whole_clip_segment_f1": float(baseline_segment["f1"]),
        },
        "first_60s_subset": payload["first_60s_subset"],
        "miss_mechanism": misses,
        "overlapping_onset_error_vs_prior_best": overlapping_onset,
        "clip_event_counts": payload["partitions"],
        "prior_first_60s_counts": prior_clip_event_counts,
        "this_pass_summary": this_pass,
        "previous_decoder_validity_pass": {
            "collar_true_positive": 9,
            "collar_false_positive": 72,
            "collar_false_negative": 39,
            "collar_event_f1": PRIOR_BEST_COLLAR_F1,
            "segment_f1": 0.512396694214876,
            "whole_clip_segment_f1": 0.17212490479817213,
            "median_onset_error_ms_on_overlapping_misses": 200,
            "note": "best first-60s inspection so far; not the tiled comparator",
        },
    }
    arguments.duration_error_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.duration_error_output.write_text(
        json.dumps(error_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {arguments.output}")
    print(f"Wrote {arguments.duration_error_output}")
    print(
        "MLP train epochs "
        f"{len(history)}; params {trainable_parameters}; "
        f"clips train/val/insp {len(train_rows)}/{len(validation_rows)}/{len(inspection)}; "
        f"unused-val-later {len(unused_val_later_rows)}; "
        f"laughs {event_count(train_rows)}/"
        f"{event_count(validation_rows)}/{event_count(inspection)}; "
        f"clipped {clipped_event_count(train_rows)}/"
        f"{clipped_event_count(validation_rows)}/{clipped_event_count(inspection)}; "
        f"tiled inspection collar {collar_f1:.4f} TP/FP/FN "
        f"{temporal_collar['true_positive']}/"
        f"{temporal_collar['false_positive']}/"
        f"{temporal_collar['false_negative']}; "
        f"first60 collar {first60_collar_f1:.4f} "
        f"events={event_count(first60_rows)}; "
        f"wired={gate_passed}"
    )


if __name__ == "__main__":
    main()
