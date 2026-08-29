"""Lightweight auxiliary correction trained on explicit speech-negative controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from attune.models.joint import SUPPORTED_EVENTS, SUPPORTED_STYLES


class TaskAdapter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    labels: list[str]
    weights: dict[str, list[float]]
    biases: dict[str, float]
    thresholds: dict[str, float]
    selected_l2: dict[str, float]
    eligible_labels: list[str]
    deployment_enabled_labels: list[str] = Field(default_factory=list)


class AuxiliaryAdapter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0"
    feature_mean: list[float]
    feature_scale: list[float]
    event_presence: TaskAdapter
    styles: TaskAdapter


def load_score_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score_features(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [row["ood_embedding"] + row["event_presence_logits"] + row["style_logits"] for row in rows],
        dtype=np.float64,
    )


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _balanced_logistic_fit(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    l2: float,
    maximum_iterations: int = 40,
) -> tuple[np.ndarray, float]:
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if not positives or not negatives:
        raise ValueError("auxiliary correction requires positive and negative training rows")
    sample_weight = np.where(targets, 0.5 / positives, 0.5 / negatives)
    design = np.column_stack((features, np.ones(len(features))))
    parameters = np.zeros(design.shape[1], dtype=np.float64)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * l2
    regularizer[-1, -1] = 0.0
    for _ in range(maximum_iterations):
        probability = _sigmoid(design @ parameters)
        gradient = design.T @ ((probability - targets) * sample_weight)
        gradient += regularizer @ parameters
        curvature = probability * (1.0 - probability) * sample_weight
        hessian = design.T @ (design * curvature[:, None]) + regularizer
        hessian += np.eye(hessian.shape[0]) * 1e-7
        update = np.linalg.solve(hessian, gradient)
        parameters -= update
        if float(np.linalg.norm(update)) < 1e-7:
            break
    return parameters[:-1], float(parameters[-1])


def _binary_counts(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int, int]:
    truth = target.astype(bool)
    predicted = prediction.astype(bool)
    return (
        int((predicted & truth).sum()),
        int((predicted & ~truth).sum()),
        int((~predicted & truth).sum()),
    )


def _f1(prediction: np.ndarray, target: np.ndarray) -> float:
    true_positive, false_positive, false_negative = _binary_counts(prediction, target)
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def _safe_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    controls: np.ndarray,
) -> tuple[float, float]:
    candidates = np.unique(np.concatenate((np.linspace(0.05, 0.995, 190), probabilities)))
    best = (-1.0, 0.0)
    for threshold in candidates:
        if controls.any() and (probabilities[controls] >= threshold).any():
            continue
        score = _f1(probabilities >= threshold, targets)
        candidate = (score, float(threshold))
        if candidate > best:
            best = candidate
    return best[1], best[0]


def _task_targets(
    rows: list[dict[str, Any]], target_key: str, label_index: int
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray([target_key in row for row in rows], dtype=bool)
    targets = np.asarray(
        [row[target_key][label_index] if target_key in row else 0.0 for row in rows],
        dtype=np.float64,
    )
    return selected, targets


def _fit_task(
    train_rows: list[dict[str, Any]],
    development_rows: list[dict[str, Any]],
    train_features: np.ndarray,
    development_features: np.ndarray,
    *,
    labels: list[str],
    target_key: str,
    control_task: str,
) -> TaskAdapter:
    weights: dict[str, list[float]] = {}
    biases: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    selected_l2: dict[str, float] = {}
    eligible = []
    for index, label in enumerate(labels):
        train_selected, train_targets = _task_targets(train_rows, target_key, index)
        development_selected, development_targets = _task_targets(
            development_rows, target_key, index
        )
        controls = np.asarray(
            [control_task in row.get("auxiliary_negative_tasks", []) for row in development_rows],
            dtype=bool,
        )[development_selected]
        best: tuple[float, float, float, np.ndarray, float] | None = None
        for l2 in (0.0001, 0.001, 0.01, 0.1):
            weight, bias = _balanced_logistic_fit(
                train_features[train_selected],
                train_targets[train_selected],
                l2=l2,
            )
            probabilities = _sigmoid(development_features[development_selected] @ weight + bias)
            threshold, score = _safe_threshold(
                probabilities,
                development_targets[development_selected],
                controls,
            )
            candidate = (score, threshold, -l2, weight, bias)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        assert best is not None
        score, threshold, negative_l2, weight, bias = best
        weights[label] = weight.tolist()
        biases[label] = bias
        thresholds[label] = threshold
        selected_l2[label] = -negative_l2
        if score >= 0.5:
            eligible.append(label)
    return TaskAdapter(
        labels=labels,
        weights=weights,
        biases=biases,
        thresholds=thresholds,
        selected_l2=selected_l2,
        eligible_labels=eligible,
        deployment_enabled_labels=[],
    )


def fit_auxiliary_adapter(
    train_rows: list[dict[str, Any]], development_rows: list[dict[str, Any]]
) -> AuxiliaryAdapter:
    train_features = score_features(train_rows)
    development_features = score_features(development_rows)
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_features = (train_features - mean) / scale
    development_features = (development_features - mean) / scale
    event_labels = [label.value for label in SUPPORTED_EVENTS]
    style_labels = [label.value for label in SUPPORTED_STYLES]
    return AuxiliaryAdapter(
        feature_mean=mean.tolist(),
        feature_scale=scale.tolist(),
        event_presence=_fit_task(
            train_rows,
            development_rows,
            train_features,
            development_features,
            labels=event_labels,
            target_key="event_presence_targets",
            control_task="event_presence",
        ),
        styles=_fit_task(
            train_rows,
            development_rows,
            train_features,
            development_features,
            labels=style_labels,
            target_key="style_targets",
            control_task="styles",
        ),
    )


def evaluate_auxiliary_adapter(
    adapter: AuxiliaryAdapter, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    features = (score_features(rows) - np.asarray(adapter.feature_mean)) / np.asarray(
        adapter.feature_scale
    )
    report: dict[str, Any] = {"clips": len(rows)}
    for name, task, target_key, control_task in (
        ("event_presence", adapter.event_presence, "event_presence_targets", "event_presence"),
        ("styles", adapter.styles, "style_targets", "styles"),
    ):
        per_label = {}
        control_any = np.zeros(len(rows), dtype=bool)
        for index, label in enumerate(task.labels):
            selected, targets = _task_targets(rows, target_key, index)
            probability = _sigmoid(features @ np.asarray(task.weights[label]) + task.biases[label])
            prediction = probability >= task.thresholds[label]
            true_positive, false_positive, false_negative = _binary_counts(
                prediction[selected], targets[selected]
            )
            controls = np.asarray(
                [control_task in row.get("auxiliary_negative_tasks", []) for row in rows],
                dtype=bool,
            )
            control_any |= prediction & controls
            per_label[label] = {
                "f1": _f1(prediction[selected], targets[selected]),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "control_false_positive_clips": int((prediction & controls).sum()),
                "threshold": task.thresholds[label],
                "eligible_from_development": label in task.eligible_labels,
            }
        report[name] = {
            "macro_f1": float(np.mean([value["f1"] for value in per_label.values()])),
            "eligible_macro_f1": (
                float(np.mean([per_label[label]["f1"] for label in task.eligible_labels]))
                if task.eligible_labels
                else 0.0
            ),
            "control_false_positive_clips": int(control_any.sum()),
            "control_clips": int(
                sum(control_task in row.get("auxiliary_negative_tasks", []) for row in rows)
            ),
            "per_label": per_label,
        }
    return report


def predict_auxiliary_adapter(
    adapter: AuxiliaryAdapter, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return per-clip audit probabilities without enabling runtime deployment."""
    features = (score_features(rows) - np.asarray(adapter.feature_mean)) / np.asarray(
        adapter.feature_scale
    )
    predictions: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        result: dict[str, Any] = {"clip_id": row["clip_id"]}
        for task_name, task in (
            ("event_presence", adapter.event_presence),
            ("styles", adapter.styles),
        ):
            labels: dict[str, Any] = {}
            for label in task.labels:
                probability = float(
                    _sigmoid(
                        np.asarray(
                            [
                                features[row_index] @ np.asarray(task.weights[label])
                                + task.biases[label]
                            ]
                        )
                    )[0]
                )
                labels[label] = {
                    "probability": probability,
                    "threshold": task.thresholds[label],
                    "above_threshold": probability >= task.thresholds[label],
                    "deployment_enabled": label in task.deployment_enabled_labels,
                }
            result[task_name] = labels
        predictions.append(result)
    return predictions
