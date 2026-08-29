"""Affect-only correction over frozen Attune acoustic outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from attune.schema.output import AffectCategory


class AffectAdapter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0"
    feature_mean: list[float]
    feature_scale: list[float]
    supported_labels: list[str]
    weights: dict[str, list[float]]
    biases: dict[str, float]
    selected_c: float
    temperature: float
    abstention_threshold: float
    deployment_enabled: bool = False


def load_score_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def affect_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if "affect_distribution" in row]
    if not selected:
        raise ValueError("affect adapter requires rows with affect targets")
    return selected


def score_features(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [row["ood_embedding"] + row["affect_logits"] + row["vad_prediction"] for row in rows],
        dtype=np.float64,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponent = np.exp(np.clip(shifted, -80.0, 0.0))
    return exponent / exponent.sum(axis=-1, keepdims=True)


def _hard_targets(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [int(np.argmax(row["affect_distribution"])) for row in rows], dtype=np.int64
    )


def _f1(prediction: np.ndarray, target: np.ndarray, class_index: int) -> float:
    predicted = prediction == class_index
    truth = target == class_index
    true_positive = int((predicted & truth).sum())
    false_positive = int((predicted & ~truth).sum())
    false_negative = int((~predicted & truth).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def _macro_f1(prediction: np.ndarray, target: np.ndarray) -> float:
    classes = np.unique(target)
    return float(np.mean([_f1(prediction, target, int(index)) for index in classes]))


def _cross_entropy(probabilities: np.ndarray, target: np.ndarray) -> float:
    selected = probabilities[np.arange(len(target)), target]
    return float(-np.log(np.clip(selected, 1e-9, 1.0)).mean())


def _temperature(logits: np.ndarray, target_columns: np.ndarray) -> float:
    candidates = np.geomspace(0.25, 10.0, 160)
    return float(
        min(
            candidates,
            key=lambda value: _cross_entropy(_softmax(logits / value), target_columns),
        )
    )


def _abstention_threshold(probabilities: np.ndarray, targets: np.ndarray) -> float:
    confidence = probabilities.max(axis=-1)
    prediction = probabilities.argmax(axis=-1)
    candidates = sorted(set(confidence.tolist()))
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        retained = confidence >= threshold
        coverage = float(retained.mean())
        if coverage < 0.5:
            continue
        accuracy = float((prediction[retained] == targets[retained]).mean())
        candidate = (accuracy, coverage, -float(threshold))
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise ValueError("development data cannot satisfy minimum affect coverage")
    return -best[2]


def fit_affect_adapter(
    train_rows: list[dict[str, Any]],
    development_rows: list[dict[str, Any]],
    *,
    candidate_c: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
) -> AffectAdapter:
    """Fit and select one deterministic balanced linear correction on development only."""
    from sklearn.linear_model import LogisticRegression

    train = affect_rows(train_rows)
    development = affect_rows(development_rows)
    train_features = score_features(train)
    development_features = score_features(development)
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_features = (train_features - mean) / scale
    development_features = (development_features - mean) / scale
    train_targets = _hard_targets(train)
    development_targets = _hard_targets(development)
    if not set(np.unique(development_targets)).issubset(set(np.unique(train_targets))):
        raise ValueError("development contains an affect class absent from training")

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
        logits = classifier.decision_function(development_features)
        probabilities = _softmax(logits)
        prediction_columns = probabilities.argmax(axis=-1)
        predictions = classifier.classes_[prediction_columns]
        score = _macro_f1(predictions, development_targets)
        target_columns = np.searchsorted(classifier.classes_, development_targets)
        cross_entropy = _cross_entropy(probabilities, target_columns)
        candidate = (score, -cross_entropy, -regularization, classifier)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    assert best is not None
    classifier = best[3]
    development_logits = classifier.decision_function(development_features)
    target_columns = np.searchsorted(classifier.classes_, development_targets)
    temperature = _temperature(development_logits, target_columns)
    development_probabilities = _softmax(development_logits / temperature)
    threshold = _abstention_threshold(development_probabilities, target_columns)
    labels = tuple(AffectCategory)
    supported = [labels[int(index)].value for index in classifier.classes_]
    return AffectAdapter(
        feature_mean=mean.tolist(),
        feature_scale=scale.tolist(),
        supported_labels=supported,
        weights={
            label: classifier.coef_[index].tolist() for index, label in enumerate(supported)
        },
        biases={
            label: float(classifier.intercept_[index]) for index, label in enumerate(supported)
        },
        selected_c=-best[2],
        temperature=temperature,
        abstention_threshold=threshold,
        deployment_enabled=False,
    )


def predict_affect_adapter(
    adapter: AffectAdapter, rows: list[dict[str, Any]]
) -> np.ndarray:
    features = (score_features(rows) - np.asarray(adapter.feature_mean)) / np.asarray(
        adapter.feature_scale
    )
    logits = np.column_stack(
        [
            features @ np.asarray(adapter.weights[label]) + adapter.biases[label]
            for label in adapter.supported_labels
        ]
    )
    supported_probabilities = _softmax(logits / adapter.temperature)
    categories = [category.value for category in AffectCategory]
    probabilities = np.zeros((len(rows), len(categories)), dtype=np.float64)
    for source_index, label in enumerate(adapter.supported_labels):
        probabilities[:, categories.index(label)] = supported_probabilities[:, source_index]
    return probabilities


def evaluate_affect_adapter(
    adapter: AffectAdapter, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    selected = affect_rows(rows)
    probabilities = predict_affect_adapter(adapter, selected)
    targets = _hard_targets(selected)
    predictions = probabilities.argmax(axis=-1)
    confidence = probabilities.max(axis=-1)
    retained = confidence >= adapter.abstention_threshold
    correct = predictions == targets
    categories = [category.value for category in AffectCategory]
    per_class: dict[str, Any] = {}
    for index, label in enumerate(categories):
        support = int((targets == index).sum())
        predicted = int((predictions == index).sum())
        true_positive = int(((targets == index) & (predictions == index)).sum())
        per_class[label] = {
            "support": support,
            "predicted": predicted,
            "recall": true_positive / support if support else None,
            "f1": _f1(predictions, targets, index) if support else None,
        }
    full_error = 1.0 - float(correct.mean())
    selective_error = 1.0 - float(correct[retained].mean()) if retained.any() else 1.0
    one_hot = np.eye(len(categories), dtype=np.float64)[targets]
    lexical_conflict = [
        index
        for index, row in enumerate(selected)
        if row.get("lexical_affect_label") is not None
        and categories[targets[index]] != row["lexical_affect_label"]
    ]
    report: dict[str, Any] = {
        "clips": len(selected),
        "supported_labels": adapter.supported_labels,
        "macro_f1": _macro_f1(predictions, targets),
        "accuracy": float(correct.mean()),
        "cross_entropy": _cross_entropy(probabilities, targets),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=-1))),
        "coverage": float(retained.mean()),
        "full_coverage_error": full_error,
        "selective_error": selective_error,
        "selective_risk_improves": selective_error < full_error,
        "maximum_predicted_class_share": max(
            value["predicted"] for value in per_class.values()
        )
        / len(selected),
        "per_class": per_class,
    }
    if lexical_conflict:
        lexical_indices = np.asarray(
            [
                categories.index(selected[index]["lexical_affect_label"])
                for index in lexical_conflict
            ]
        )
        acoustic_accuracy = float(
            (predictions[lexical_conflict] == targets[lexical_conflict]).mean()
        )
        lexical_accuracy = float((predictions[lexical_conflict] == lexical_indices).mean())
        report["acoustic_preference_score"] = acoustic_accuracy - lexical_accuracy
        report["conflict_clips"] = len(lexical_conflict)
    return report
