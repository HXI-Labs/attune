"""Whisper-style correction over frozen Attune acoustic outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict


class WhisperStyleAdapter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0"
    feature_mean: list[float]
    feature_scale: list[float]
    weights: list[float]
    bias: float
    selected_c: float
    temperature: float
    threshold: float
    deployment_enabled: bool = False


def style_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if "style_targets" in row]
    if not selected:
        raise ValueError("style adapter requires rows with style targets")
    return selected


def score_features(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [row["ood_embedding"] + row["style_logits"] for row in rows], dtype=np.float64
    )


def _targets(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([bool(row["style_targets"][1]) for row in rows], dtype=np.int64)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


def _binary_metrics(
    probability: np.ndarray, target: np.ndarray, threshold: float
) -> dict[str, float]:
    prediction = probability >= threshold
    truth = target.astype(bool)
    true_positive = int((prediction & truth).sum())
    false_positive = int((prediction & ~truth).sum())
    false_negative = int((~prediction & truth).sum())
    true_negative = int((~prediction & ~truth).sum())
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    false_positive_denominator = false_positive + true_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive_rate = (
        false_positive / false_positive_denominator if false_positive_denominator else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "accuracy": float((prediction == truth).mean()),
    }


def _binary_cross_entropy(probability: np.ndarray, target: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-9, 1.0 - 1e-9)
    return float(-(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)).mean())


def fit_whisper_style_adapter(
    train_rows: list[dict[str, Any]],
    development_rows: list[dict[str, Any]],
    *,
    candidate_c: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
) -> WhisperStyleAdapter:
    from sklearn.linear_model import LogisticRegression

    train = style_rows(train_rows)
    development = style_rows(development_rows)
    train_features = score_features(train)
    development_features = score_features(development)
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_features = (train_features - mean) / scale
    development_features = (development_features - mean) / scale
    train_targets = _targets(train)
    development_targets = _targets(development)
    if set(np.unique(train_targets)) != {0, 1} or set(np.unique(development_targets)) != {0, 1}:
        raise ValueError("whisper adapter requires positive and negative examples in both splits")

    best: tuple[float, float, float, Any] | None = None
    for regularization in candidate_c:
        classifier = LogisticRegression(
            C=regularization,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=0,
            tol=1e-6,
        )
        classifier.fit(train_features, train_targets)
        probability = classifier.predict_proba(development_features)[:, 1]
        metrics = _binary_metrics(probability, development_targets, 0.5)
        loss = _binary_cross_entropy(probability, development_targets)
        candidate = (metrics["f1"], -loss, -regularization, classifier)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    assert best is not None
    classifier = best[3]
    logits = classifier.decision_function(development_features)
    temperatures = np.geomspace(0.25, 10.0, 160)
    temperature = float(
        min(
            temperatures,
            key=lambda value: _binary_cross_entropy(_sigmoid(logits / value), development_targets),
        )
    )
    probability = _sigmoid(logits / temperature)
    thresholds = sorted(set(probability.tolist()))
    threshold = max(
        thresholds,
        key=lambda value: (
            _binary_metrics(probability, development_targets, value)["f1"],
            -_binary_metrics(probability, development_targets, value)["false_positive_rate"],
            value,
        ),
    )
    return WhisperStyleAdapter(
        feature_mean=mean.tolist(),
        feature_scale=scale.tolist(),
        weights=classifier.coef_[0].tolist(),
        bias=float(classifier.intercept_[0]),
        selected_c=-best[2],
        temperature=temperature,
        threshold=float(threshold),
        deployment_enabled=False,
    )


def predict_whisper_probability(
    adapter: WhisperStyleAdapter, rows: list[dict[str, Any]]
) -> np.ndarray:
    features = (score_features(rows) - np.asarray(adapter.feature_mean)) / np.asarray(
        adapter.feature_scale
    )
    logits = features @ np.asarray(adapter.weights) + adapter.bias
    return _sigmoid(logits / adapter.temperature)


def evaluate_whisper_style_adapter(
    adapter: WhisperStyleAdapter, rows: list[dict[str, Any]]
) -> dict[str, float | int]:
    selected = style_rows(rows)
    probability = predict_whisper_probability(adapter, selected)
    target = _targets(selected)
    return {
        "clips": len(selected),
        "positive_clips": int(target.sum()),
        **_binary_metrics(probability, target, adapter.threshold),
    }
