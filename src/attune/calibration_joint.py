"""Validation-only calibration for unified event, style, affect, and OOD heads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from attune.inference.onnx_backend import (
    RuntimeCalibration,
    _sigmoid,
    _softmax,
)
from attune.models.joint import SUPPORTED_EVENTS, SUPPORTED_STYLES
from attune.schema.output import AffectCategory

AFFECT_BIAS_REGULARIZATION = 1e-3


def _binary_nll(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    probabilities = _sigmoid(logits / temperature)
    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return float(
        -(targets * np.log(probabilities) + (1 - targets) * np.log(1 - probabilities)).mean()
    )


def _binary_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    candidates = np.geomspace(0.25, 10.0, 160)
    return float(
        min(candidates, key=lambda value: _binary_nll(logits, targets, float(value)))
    )


def _affect_nll(
    logits: np.ndarray,
    targets: np.ndarray,
    temperature: float,
    bias: np.ndarray,
) -> float:
    probabilities = np.clip(_softmax(logits / temperature + bias), 1e-7, 1.0)
    cross_entropy = -(targets * np.log(probabilities)).sum(axis=-1).mean()
    penalty = AFFECT_BIAS_REGULARIZATION * np.mean(bias**2)
    return float(cross_entropy + penalty)


def _affect_bias(
    logits: np.ndarray,
    targets: np.ndarray,
    temperature: float,
    initial: np.ndarray,
) -> np.ndarray:
    bias = initial.copy()
    class_count = logits.shape[-1]
    identity = np.eye(class_count)
    regularization = 2 * AFFECT_BIAS_REGULARIZATION / class_count
    for _ in range(100):
        probabilities = _softmax(logits / temperature + bias)
        gradient = (probabilities - targets).mean(axis=0) + regularization * bias
        hessian = np.mean(
            identity[None, :, :] * probabilities[:, :, None]
            - probabilities[:, :, None] * probabilities[:, None, :],
            axis=0,
        )
        hessian += regularization * identity
        step = np.linalg.solve(hessian, gradient)
        previous_loss = _affect_nll(logits, targets, temperature, bias)
        step_size = 1.0
        while step_size > 1e-6:
            candidate = bias - step_size * step
            candidate -= candidate.mean()
            if _affect_nll(logits, targets, temperature, candidate) < previous_loss:
                break
            step_size /= 2
        if step_size <= 1e-6:
            break
        bias = candidate
        if np.max(np.abs(step_size * step)) < 1e-8:
            break
    return bias


def _affect_calibration(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    temperature = 1.0
    bias = np.zeros(len(AffectCategory), dtype=np.float64)
    for _ in range(8):
        bias = _affect_bias(logits, targets, temperature, bias)
        candidates = np.geomspace(0.25, 10.0, 400)
        temperature = float(
            min(
                candidates,
                key=lambda value: _affect_nll(logits, targets, float(value), bias),
            )
        )
    return temperature, bias


def _f1_threshold(probabilities: np.ndarray, targets: np.ndarray) -> float:
    best = (0.0, 0.5)
    for threshold in np.linspace(0.05, 0.95, 91):
        prediction = probabilities >= threshold
        truth = targets >= 0.5
        true_positive = int((prediction & truth).sum())
        false_positive = int((prediction & ~truth).sum())
        false_negative = int((~prediction & truth).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        score = 2 * true_positive / denominator if denominator else 0.0
        candidate = (score, float(threshold))
        if candidate > best:
            best = candidate
    return best[1]


def _affect_threshold(probabilities: np.ndarray, targets: np.ndarray) -> float:
    confidence = probabilities.max(axis=-1)
    prediction = probabilities.argmax(axis=-1)
    truth = targets.argmax(axis=-1)
    best = (-1.0, 0.55)
    for threshold in sorted(set(confidence.tolist())):
        retained = confidence >= threshold
        coverage = float(retained.mean())
        if coverage < 0.8:
            continue
        accuracy = float((prediction[retained] == truth[retained]).mean())
        candidate = (accuracy, float(threshold))
        if candidate > best:
            best = candidate
    return best[1]


def _speech_controls(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if bool(row.get("auxiliary_negative_tasks"))
        or (
            "reference_transcript" in row
            and "event_targets" not in row
            and "event_presence_targets" not in row
            and "style_targets" not in row
        )
    ]


def _event_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    control_rows: list[dict[str, Any]],
    temperature: float,
) -> dict[str, float]:
    thresholds = {}
    for index, label in enumerate(SUPPORTED_EVENTS):
        threshold = _f1_threshold(probabilities[..., index], targets[..., index])
        if control_rows:
            highest_control_probability = max(
                float(
                    _sigmoid(
                        np.asarray(row["event_logits"], dtype=np.float32)[..., index] / temperature
                    ).max()
                )
                for row in control_rows
            )
            threshold = max(threshold, min(1.0, highest_control_probability + 1e-4))
        thresholds[label.value] = threshold
    return thresholds


def fit_runtime_calibration(rows: list[dict[str, Any]]) -> RuntimeCalibration:
    if not rows:
        raise ValueError("calibration requires validation rows")
    event_rows = [row for row in rows if "event_targets" in row]
    presence_rows = [row for row in rows if "event_presence_targets" in row]
    style_rows = [row for row in rows if "style_targets" in row]
    affect_rows = [row for row in rows if "affect_distribution" in row]
    if not event_rows or not presence_rows or not style_rows or not affect_rows:
        raise ValueError(
            "calibration requires localized event, event presence, style, and affect targets"
        )
    event_logits = np.concatenate(
        [np.asarray(row["event_logits"], dtype=np.float32) for row in event_rows], axis=0
    )
    event_targets = np.concatenate(
        [np.asarray(row["event_targets"], dtype=np.float32) for row in event_rows], axis=0
    )
    presence_logits = np.asarray(
        [row["event_presence_logits"] for row in presence_rows], dtype=np.float32
    )
    presence_targets = np.asarray(
        [row["event_presence_targets"] for row in presence_rows], dtype=np.float32
    )
    style_logits = np.asarray([row["style_logits"] for row in style_rows], dtype=np.float32)
    style_targets = np.asarray([row["style_targets"] for row in style_rows], dtype=np.float32)
    affect_logits = np.asarray([row["affect_logits"] for row in affect_rows], dtype=np.float32)
    affect_targets = np.asarray(
        [row["affect_distribution"] for row in affect_rows], dtype=np.float32
    )
    ood_rows = [
        row for row in rows if "affect_distribution" in row or bool(row.get("is_ood", False))
    ]
    embeddings = np.asarray([row["ood_embedding"] for row in ood_rows], dtype=np.float32)
    ood_logits = np.asarray([row["ood_logit"] for row in ood_rows], dtype=np.float32)
    is_ood = np.asarray([bool(row.get("is_ood", False)) for row in ood_rows])
    if not (~is_ood).any() or not is_ood.any():
        raise ValueError("calibration requires affect in-distribution and known OOD rows")

    event_temperature = _binary_temperature(event_logits, event_targets)
    presence_temperature = _binary_temperature(presence_logits, presence_targets)
    style_temperature = _binary_temperature(style_logits, style_targets)
    affect_temperature, affect_bias = _affect_calibration(affect_logits, affect_targets)
    event_probabilities = _sigmoid(event_logits / event_temperature)
    presence_probabilities = _sigmoid(presence_logits / presence_temperature)
    style_probabilities = _sigmoid(style_logits / style_temperature)
    affect_probabilities = _softmax(affect_logits / affect_temperature + affect_bias)
    in_distribution = embeddings[~is_ood]
    centroid = in_distribution.mean(axis=0)
    distances = np.linalg.norm(in_distribution - centroid, axis=-1)
    distance_scale = max(float(np.quantile(distances, 0.95)), 1e-6)
    ood_temperature = _binary_temperature(ood_logits, is_ood.astype(np.float32))
    ood_probabilities = _sigmoid(ood_logits / ood_temperature)
    return RuntimeCalibration(
        event_temperature=event_temperature,
        event_thresholds=_event_thresholds(
            event_probabilities,
            event_targets,
            _speech_controls(rows),
            event_temperature,
        ),
        event_presence_temperature=presence_temperature,
        event_presence_thresholds={
            label.value: _f1_threshold(presence_probabilities[:, index], presence_targets[:, index])
            for index, label in enumerate(SUPPORTED_EVENTS)
        },
        localized_event_labels=[
            label.value
            for index, label in enumerate(SUPPORTED_EVENTS)
            if event_targets[..., index].sum() > 0
        ],
        localized_event_min_confidence=0.98,
        event_presence_enabled_labels=[],
        style_temperature=style_temperature,
        style_thresholds={
            label.value: _f1_threshold(style_probabilities[:, index], style_targets[:, index])
            for index, label in enumerate(SUPPORTED_STYLES)
        },
        style_enabled_labels=[],
        affect_temperature=affect_temperature,
        affect_bias=affect_bias.tolist(),
        affect_threshold=_affect_threshold(affect_probabilities, affect_targets),
        vad_available=any("vad_target" in row for row in rows),
        ood_centroid=centroid.tolist(),
        ood_distance_scale=distance_scale,
        ood_available=True,
        ood_temperature=ood_temperature,
        ood_threshold=_f1_threshold(ood_probabilities, is_ood.astype(np.float32)),
    )


def load_score_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
