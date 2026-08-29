"""Small nonlinear affect adapter over frozen Attune acoustic outputs."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

from attune.affect_adapter import (
    _abstention_threshold,
    _cross_entropy,
    _f1,
    _hard_targets,
    _macro_f1,
    _softmax,
    _temperature,
    affect_rows,
    score_features,
)
from attune.schema.output import AffectCategory


class AffectMLPAdapter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0"
    feature_mean: list[float]
    feature_scale: list[float]
    supported_labels: list[str]
    hidden_sizes: tuple[int, int]
    weights_1: list[list[float]]
    biases_1: list[float]
    weights_2: list[list[float]]
    biases_2: list[float]
    weights_out: list[list[float]]
    biases_out: list[float]
    temperature: float
    abstention_threshold: float
    best_epoch: int
    deployment_enabled: bool = False


class _Network(nn.Module):
    def __init__(self, inputs: int, hidden: tuple[int, int], outputs: int, dropout: float):
        super().__init__()
        self.first = nn.Linear(inputs, hidden[0])
        self.second = nn.Linear(hidden[0], hidden[1])
        self.output = nn.Linear(hidden[1], outputs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.dropout(torch.nn.functional.gelu(self.first(features)))
        value = self.dropout(torch.nn.functional.gelu(self.second(value)))
        return self.output(value)


def _group_weights(rows: list[dict[str, Any]], targets: np.ndarray) -> np.ndarray:
    groups = [
        (str(row["dataset_id"]), int(target)) for row, target in zip(rows, targets, strict=True)
    ]
    counts = Counter(groups)
    weights = np.asarray([1.0 / counts[group] for group in groups], dtype=np.float64)
    return weights / weights.mean()


def _domain_macro_f1(
    rows: list[dict[str, Any]], prediction: np.ndarray, targets: np.ndarray
) -> float:
    scores = []
    for dataset_id in sorted({str(row["dataset_id"]) for row in rows}):
        selected = np.asarray([str(row["dataset_id"]) == dataset_id for row in rows])
        scores.append(_macro_f1(prediction[selected], targets[selected]))
    return float(np.mean(scores))


def fit_affect_mlp_adapter(
    train_rows: list[dict[str, Any]],
    development_rows: list[dict[str, Any]],
    *,
    hidden_sizes: tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    maximum_epochs: int = 300,
    patience: int = 35,
    seed: int = 0,
) -> AffectMLPAdapter:
    train = affect_rows(train_rows)
    development = affect_rows(development_rows)
    train_features = score_features(train)
    development_features = score_features(development)
    train_targets = _hard_targets(train)
    development_targets = _hard_targets(development)
    classes = np.unique(train_targets)
    if not set(np.unique(development_targets)).issubset(set(classes)):
        raise ValueError("development contains an affect class absent from training")
    train_columns = np.searchsorted(classes, train_targets)
    development_columns = np.searchsorted(classes, development_targets)
    weights = _group_weights(train, train_targets)
    mean = np.average(train_features, axis=0, weights=weights)
    variance = np.average((train_features - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1.0
    train_features = (train_features - mean) / scale
    development_features = (development_features - mean) / scale

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    network = _Network(train_features.shape[1], hidden_sizes, len(classes), dropout)
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    features_tensor = torch.tensor(train_features, dtype=torch.float32)
    targets_tensor = torch.tensor(train_columns, dtype=torch.long)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    development_tensor = torch.tensor(development_features, dtype=torch.float32)
    best: tuple[float, float, int, dict[str, torch.Tensor]] | None = None
    stale = 0
    for epoch in range(1, maximum_epochs + 1):
        network.train()
        optimizer.zero_grad(set_to_none=True)
        logits = network(features_tensor)
        losses = torch.nn.functional.cross_entropy(logits, targets_tensor, reduction="none")
        loss = (losses * weights_tensor).mean()
        loss.backward()
        optimizer.step()
        network.eval()
        with torch.inference_mode():
            validation_logits = network(development_tensor).numpy()
        probabilities = _softmax(validation_logits)
        predicted_classes = classes[probabilities.argmax(axis=-1)]
        domain_score = _domain_macro_f1(development, predicted_classes, development_targets)
        cross_entropy = _cross_entropy(probabilities, development_columns)
        candidate = (domain_score, -cross_entropy, epoch)
        if best is None or candidate[:2] > best[:2]:
            best = (
                *candidate,
                {name: value.detach().clone() for name, value in network.state_dict().items()},
            )
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    assert best is not None
    network.load_state_dict(best[3])
    network.eval()
    with torch.inference_mode():
        development_logits = network(development_tensor).numpy()
    temperature = _temperature(development_logits, development_columns)
    development_probabilities = _softmax(development_logits / temperature)
    threshold = _abstention_threshold(development_probabilities, development_columns)
    labels = tuple(AffectCategory)
    supported = [labels[int(index)].value for index in classes]
    return AffectMLPAdapter(
        feature_mean=mean.tolist(),
        feature_scale=scale.tolist(),
        supported_labels=supported,
        hidden_sizes=hidden_sizes,
        weights_1=network.first.weight.detach().tolist(),
        biases_1=network.first.bias.detach().tolist(),
        weights_2=network.second.weight.detach().tolist(),
        biases_2=network.second.bias.detach().tolist(),
        weights_out=network.output.weight.detach().tolist(),
        biases_out=network.output.bias.detach().tolist(),
        temperature=temperature,
        abstention_threshold=threshold,
        best_epoch=best[2],
        deployment_enabled=False,
    )


def predict_affect_mlp(adapter: AffectMLPAdapter, rows: list[dict[str, Any]]) -> np.ndarray:
    features = (score_features(rows) - np.asarray(adapter.feature_mean)) / np.asarray(
        adapter.feature_scale
    )
    network = _Network(features.shape[1], adapter.hidden_sizes, len(adapter.supported_labels), 0.0)
    state = {
        "first.weight": torch.tensor(adapter.weights_1),
        "first.bias": torch.tensor(adapter.biases_1),
        "second.weight": torch.tensor(adapter.weights_2),
        "second.bias": torch.tensor(adapter.biases_2),
        "output.weight": torch.tensor(adapter.weights_out),
        "output.bias": torch.tensor(adapter.biases_out),
    }
    network.load_state_dict(state)
    network.eval()
    with torch.inference_mode():
        logits = network(torch.tensor(features, dtype=torch.float32)).numpy()
    supported = _softmax(logits / adapter.temperature)
    categories = [category.value for category in AffectCategory]
    probabilities = np.zeros((len(rows), len(categories)), dtype=np.float64)
    for source_index, label in enumerate(adapter.supported_labels):
        probabilities[:, categories.index(label)] = supported[:, source_index]
    return probabilities


def evaluate_affect_mlp(adapter: AffectMLPAdapter, rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = affect_rows(rows)
    probabilities = predict_affect_mlp(adapter, selected)
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
    return {
        "clips": len(selected),
        "macro_f1": _macro_f1(predictions, targets),
        "accuracy": float(correct.mean()),
        "coverage": float(retained.mean()),
        "full_coverage_error": full_error,
        "selective_error": selective_error,
        "selective_risk_improves": selective_error < full_error,
        "maximum_predicted_class_share": max(value["predicted"] for value in per_class.values())
        / len(selected),
        "per_class": per_class,
    }
