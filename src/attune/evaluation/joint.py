"""Unified-candidate score collection and release metrics."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from attune.evaluation.metrics import corpus_word_error_rate
from attune.inference.export import OUTPUT_NAMES
from attune.inference.onnx_backend import (
    RuntimeCalibration,
    _bridge_short_gaps,
    _runs,
    _sigmoid,
    _softmax,
)
from attune.models.joint import SUPPORTED_EVENTS, SUPPORTED_STYLES
from attune.schema.output import AffectCategory
from attune.training.data import JointFeatureDataset, collate_joint_examples


def collect_onnx_scores(
    *,
    manifest: Path,
    split: str,
    model_path: Path,
    output_path: Path,
    session: Any | None = None,
    transcript_decoder: Any | None = None,
) -> list[dict[str, Any]]:
    if session is None:
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    dataset = JointFeatureDataset(manifest, split=split)
    rows = []
    for index in range(len(dataset)):
        manifest_row, features = dataset[index]
        batch = collate_joint_examples([(manifest_row, features)])
        values = session.run(
            list(OUTPUT_NAMES),
            {
                "speech": features.unsqueeze(0).numpy(),
                "speech_lengths": np.asarray([features.shape[0]], dtype=np.int64),
            },
        )
        outputs = dict(zip(OUTPUT_NAMES, values, strict=True))
        length = int(outputs["acoustic_lengths"][0])
        row: dict[str, Any] = {
            "clip_id": manifest_row.clip_id,
            "dataset_id": manifest_row.dataset_id,
            "split": split,
            "event_logits": outputs["event_logits"][0, :length].tolist(),
            "event_presence_logits": outputs["event_presence_logits"][0].tolist(),
            "style_logits": outputs["style_logits"][0].tolist(),
            "affect_logits": outputs["affect_logits"][0].tolist(),
            "vad_prediction": outputs["vad"][0].tolist(),
            "ood_embedding": outputs["ood_embedding"][0].tolist(),
            "ood_logit": float(outputs["ood_logit"][0]),
            "is_ood": manifest_row.is_ood,
            "pair_id": manifest_row.pair_id,
            "lexical_affect_label": manifest_row.lexical_affect_label,
            "frame_hop_ms": manifest_row.frame_hop_ms,
        }
        if manifest_row.transcript is not None and transcript_decoder is not None:
            predicted, _confidence = transcript_decoder(outputs["ctc_logits"][0], length)
            row["reference_transcript"] = manifest_row.transcript
            row["predicted_transcript"] = predicted
        if batch.targets.event_target_mask is not None and batch.targets.event_target_mask.any():
            row["event_targets"] = batch.targets.event_targets[0, :length].tolist()
        if (
            batch.targets.event_presence_example_mask is not None
            and batch.targets.event_presence_example_mask.any()
        ):
            row["event_presence_targets"] = batch.targets.event_presence_targets[0].tolist()
        if batch.targets.style_example_mask is not None and batch.targets.style_example_mask.any():
            row["style_targets"] = batch.targets.style_targets[0].tolist()
        if (
            batch.targets.affect_example_mask is not None
            and batch.targets.affect_example_mask.any()
        ):
            row["affect_distribution"] = batch.targets.affect_distribution[0].tolist()
        if batch.targets.vad_target_mask is not None and batch.targets.vad_target_mask.any():
            row["vad_target"] = batch.targets.vad_targets[0].tolist()
        rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    return rows


def _binary_f1(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    true_positive = int((prediction & target).sum())
    false_positive = int((prediction & ~target).sum())
    false_negative = int((~prediction & target).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def _average_precision(probability: np.ndarray, target: np.ndarray) -> float:
    truth = target.astype(bool).reshape(-1)
    if not truth.any():
        return 0.0
    order = np.argsort(-probability.reshape(-1))
    ranked = truth[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].mean())


def _multiclass_macro_f1(prediction: np.ndarray, target: np.ndarray, classes: int) -> float:
    return float(
        np.mean(
            [
                _binary_f1(prediction == class_index, target == class_index)
                for class_index in range(classes)
            ]
        )
    )


def _present_class_macro_f1(prediction: np.ndarray, target: np.ndarray) -> float:
    present = np.unique(target)
    return float(np.mean([_binary_f1(prediction == index, target == index) for index in present]))


def _ccc(prediction: np.ndarray, target: np.ndarray) -> float:
    covariance = np.mean((prediction - prediction.mean()) * (target - target.mean()))
    denominator = prediction.var() + target.var() + (prediction.mean() - target.mean()) ** 2
    return float(2 * covariance / denominator) if denominator > 0 else 0.0


def _ece(probabilities: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=-1)
    correct = probabilities.argmax(axis=-1) == target
    error = 0.0
    for lower in np.linspace(0, 1, bins, endpoint=False):
        upper = lower + 1 / bins
        selected = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= 1)
        if selected.any():
            error += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return error


def _mce(probabilities: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=-1)
    correct = probabilities.argmax(axis=-1) == target
    errors = []
    for lower in np.linspace(0, 1, bins, endpoint=False):
        upper = lower + 1 / bins
        selected = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= 1)
        if selected.any():
            errors.append(abs(float(correct[selected].mean()) - float(confidence[selected].mean())))
    return max(errors, default=0.0)


def _risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> list[dict[str, float]]:
    order = np.argsort(-confidence)
    points = []
    for requested in np.linspace(0.1, 1.0, 10):
        count = max(1, int(np.ceil(len(order) * requested)))
        retained = order[:count]
        points.append(
            {
                "coverage": count / len(order),
                "risk": 1.0 - float(correct[retained].mean()),
            }
        )
    return points


def _jensen_shannon(probabilities: np.ndarray, targets: np.ndarray) -> float:
    midpoint = (probabilities + targets) / 2.0
    first = np.sum(
        probabilities
        * (np.log(np.clip(probabilities, 1e-7, 1.0)) - np.log(np.clip(midpoint, 1e-7, 1.0))),
        axis=-1,
    )
    second = np.sum(
        targets * (np.log(np.clip(targets, 1e-7, 1.0)) - np.log(np.clip(midpoint, 1e-7, 1.0))),
        axis=-1,
    )
    return float(np.mean((first + second) / 2.0))


def _interval_iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union else 1.0


def _event_segment_metrics(
    rows: list[dict[str, Any]], calibration: RuntimeCalibration
) -> dict[str, Any]:
    per_class = {}
    for label_index, label in enumerate(SUPPORTED_EVENTS):
        true_positive = false_positive = false_negative = 0
        overlaps: list[float] = []
        boundary_errors_ms: list[float] = []
        for row in rows:
            probabilities = _sigmoid(
                np.asarray(row["event_logits"])[:, label_index] / calibration.event_temperature
            )
            active = probabilities >= calibration.event_thresholds[label.value]
            active = _bridge_short_gaps(active, calibration.bridge_event_frames)
            predictions = _runs(active, calibration.minimum_event_frames)
            targets = _runs(
                np.asarray(row["event_targets"])[:, label_index] >= 0.5,
                1,
            )
            unmatched = set(range(len(targets)))
            for prediction in predictions:
                candidates = [
                    (_interval_iou(prediction, targets[index]), index) for index in unmatched
                ]
                overlap, matched = max(candidates, default=(0.0, -1))
                if overlap <= 0:
                    false_positive += 1
                    continue
                unmatched.remove(matched)
                true_positive += 1
                overlaps.append(overlap)
                frame_hop = float(row["frame_hop_ms"])
                boundary_errors_ms.extend(
                    (
                        abs(prediction[0] - targets[matched][0]) * frame_hop,
                        abs(prediction[1] - targets[matched][1]) * frame_hop,
                    )
                )
            false_negative += len(unmatched)
        denominator = 2 * true_positive + false_positive + false_negative
        per_class[label.value] = {
            "segment_f1": 2 * true_positive / denominator if denominator else 0.0,
            "matched_temporal_iou": float(np.mean(overlaps)) if overlaps else 0.0,
            "boundary_mae_ms": (float(np.mean(boundary_errors_ms)) if boundary_errors_ms else None),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }
    return {
        "macro_f1": float(
            np.mean(
                [per_class[label]["segment_f1"] for label in calibration.localized_event_labels]
            )
        ),
        "macro_f1_ontology": float(np.mean([value["segment_f1"] for value in per_class.values()])),
        "localized_labels": calibration.localized_event_labels,
        "per_class": per_class,
    }


def evaluate_joint_scores(
    rows: list[dict[str, Any]],
    calibration: RuntimeCalibration,
    *,
    include_dataset_slices: bool = True,
) -> dict[str, Any]:
    report: dict[str, Any] = {"schema_version": "1.0", "clips": len(rows)}
    asr_rows = [
        row for row in rows if "reference_transcript" in row and "predicted_transcript" in row
    ]
    if asr_rows:
        normalize = lambda value: " ".join(  # noqa: E731
            re.findall(r"[a-z0-9']+", value.lower())
        )
        report["asr_wer"] = corpus_word_error_rate(
            [normalize(row["reference_transcript"]) for row in asr_rows],
            [normalize(row["predicted_transcript"]) for row in asr_rows],
        )
        report["asr_clips"] = len(asr_rows)
    ood_rows = [row for row in rows if "ood_logit" in row]
    if ood_rows and {bool(row.get("is_ood", False)) for row in ood_rows} == {
        False,
        True,
    }:
        logits = np.asarray([row["ood_logit"] for row in ood_rows])
        targets = np.asarray([bool(row.get("is_ood", False)) for row in ood_rows])
        probabilities = _sigmoid(logits / calibration.ood_temperature)
        predictions = probabilities >= calibration.ood_threshold
        report["ood_f1"] = _binary_f1(predictions, targets)
        report["ood_false_positive_rate"] = float(predictions[~targets].mean())
        report["ood_true_positive_rate"] = float(predictions[targets].mean())
    event_rows = [row for row in rows if "event_targets" in row]
    if event_rows:
        logits = np.concatenate([np.asarray(row["event_logits"]) for row in event_rows])
        targets = np.concatenate([np.asarray(row["event_targets"]) for row in event_rows])
        probabilities = _sigmoid(logits / calibration.event_temperature)
        scores = []
        per_class = {}
        average_precision = {}
        for index, label in enumerate(SUPPORTED_EVENTS):
            prediction = probabilities[:, index] >= calibration.event_thresholds[label.value]
            score = _binary_f1(prediction, targets[:, index] >= 0.5)
            per_class[label.value] = score
            average_precision[label.value] = _average_precision(
                probabilities[:, index], targets[:, index]
            )
            scores.append(score)
        report["event_frame_macro_f1_ontology"] = float(np.mean(scores))
        report["event_frame_macro_f1"] = float(
            np.mean([per_class[label] for label in calibration.localized_event_labels])
        )
        report["event_localized_labels"] = calibration.localized_event_labels
        report["event_frame_f1"] = per_class
        report["event_frame_average_precision"] = average_precision
        report["event_frame_map"] = float(
            np.mean([average_precision[label] for label in calibration.localized_event_labels])
        )
        report["event_segments"] = _event_segment_metrics(event_rows, calibration)
    presence_rows = [row for row in rows if "event_presence_targets" in row]
    if presence_rows:
        logits = np.asarray([row["event_presence_logits"] for row in presence_rows])
        targets = np.asarray([row["event_presence_targets"] for row in presence_rows])
        probabilities = _sigmoid(logits / calibration.event_presence_temperature)
        scores = []
        per_class = {}
        average_precision = {}
        for index, label in enumerate(SUPPORTED_EVENTS):
            prediction = (
                probabilities[:, index] >= calibration.event_presence_thresholds[label.value]
            )
            score = _binary_f1(prediction, targets[:, index] >= 0.5)
            per_class[label.value] = score
            average_precision[label.value] = _average_precision(
                probabilities[:, index], targets[:, index]
            )
            scores.append(score)
        report["event_presence_macro_f1"] = float(np.mean(scores))
        report["event_presence_f1"] = per_class
        report["event_presence_average_precision"] = average_precision
        report["event_presence_map"] = float(np.mean(list(average_precision.values())))
    style_rows = [row for row in rows if "style_targets" in row]
    if style_rows:
        logits = np.asarray([row["style_logits"] for row in style_rows])
        targets = np.asarray([row["style_targets"] for row in style_rows])
        probabilities = _sigmoid(logits / calibration.style_temperature)
        scores = []
        per_class = {}
        average_precision = {}
        for index, label in enumerate(SUPPORTED_STYLES):
            prediction = probabilities[:, index] >= calibration.style_thresholds[label.value]
            score = _binary_f1(prediction, targets[:, index] >= 0.5)
            per_class[label.value] = score
            average_precision[label.value] = _average_precision(
                probabilities[:, index], targets[:, index]
            )
            scores.append(score)
        report["style_macro_f1"] = float(np.mean(scores))
        report["style_f1"] = per_class
        report["style_average_precision"] = average_precision
        report["style_map"] = float(np.mean(list(average_precision.values())))
    affect_rows = [row for row in rows if "affect_distribution" in row]
    if affect_rows:
        logits = np.asarray([row["affect_logits"] for row in affect_rows])
        targets = np.asarray([row["affect_distribution"] for row in affect_rows])
        probabilities = _softmax(logits / calibration.affect_temperature)
        hard_targets = targets.argmax(axis=-1)
        predictions = probabilities.argmax(axis=-1)
        report["affect_macro_f1_ontology"] = _multiclass_macro_f1(
            predictions, hard_targets, len(AffectCategory)
        )
        report["affect_macro_f1"] = _present_class_macro_f1(predictions, hard_targets)
        report["affect_supported_classes"] = [
            tuple(AffectCategory)[index].value for index in np.unique(hard_targets)
        ]
        report["affect_brier"] = float(np.mean(np.sum((probabilities - targets) ** 2, axis=-1)))
        report["affect_ece"] = _ece(probabilities, hard_targets)
        report["affect_mce"] = _mce(probabilities, hard_targets)
        report["affect_cross_entropy"] = float(
            -np.mean(np.sum(targets * np.log(np.clip(probabilities, 1e-7, 1.0)), axis=-1))
        )
        report["affect_jensen_shannon"] = _jensen_shannon(probabilities, targets)
        confidence = probabilities.max(axis=-1)
        correct = predictions == hard_targets
        full_error = 1.0 - float(correct.mean())
        retained = confidence >= calibration.affect_threshold
        selective_error = 1.0 - float(correct[retained].mean()) if retained.any() else 1.0
        report["affect_coverage"] = float(retained.mean())
        report["affect_full_coverage_error"] = full_error
        report["affect_selective_error"] = selective_error
        report["affect_selective_risk_improves"] = selective_error <= full_error
        report["affect_risk_coverage"] = _risk_coverage(confidence, correct)
        conflict = [
            index
            for index, row in enumerate(affect_rows)
            if row.get("lexical_affect_label") is not None
            and tuple(AffectCategory)[hard_targets[index]].value != row["lexical_affect_label"]
        ]
        if conflict:
            lexical_indices = np.asarray(
                [
                    tuple(AffectCategory).index(
                        AffectCategory(affect_rows[index]["lexical_affect_label"])
                    )
                    for index in conflict
                ]
            )
            acoustic_accuracy = float((predictions[conflict] == hard_targets[conflict]).mean())
            lexical_accuracy = float((predictions[conflict] == lexical_indices).mean())
            report["acoustic_preference_score"] = acoustic_accuracy - lexical_accuracy
            report["conflict_clips"] = len(conflict)
    vad_rows = [row for row in rows if "vad_target" in row]
    if vad_rows:
        prediction = np.asarray([row["vad_prediction"] for row in vad_rows])
        target = np.asarray([row["vad_target"] for row in vad_rows])
        report["vad_ccc"] = {
            name: _ccc(prediction[:, index], target[:, index])
            for index, name in enumerate(("valence", "arousal", "dominance"))
        }
    if include_dataset_slices:
        datasets = sorted({str(row.get("dataset_id", "unknown")) for row in rows})
        report["by_dataset"] = {
            dataset: evaluate_joint_scores(
                [row for row in rows if str(row.get("dataset_id", "unknown")) == dataset],
                calibration,
                include_dataset_slices=False,
            )
            for dataset in datasets
        }
    return report


def load_scores(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
