"""Validation-only temperature scaling for categorical model outputs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from attune.evaluation.metrics import expected_calibration_error, multiclass_brier_score


class CalibrationError(ValueError):
    """Raised when calibration data or an artifact violates the split contract."""


@dataclass(frozen=True)
class TemperatureCalibration:
    """One scalar categorical temperature fitted on validation data."""

    labels: tuple[str, ...]
    temperature: float
    fitted_on: str = "validation"

    def __post_init__(self) -> None:
        if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels):
            raise CalibrationError("calibration labels must be unique")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise CalibrationError("temperature must be finite and positive")
        if self.fitted_on != "validation":
            raise CalibrationError("temperature must be fitted on validation")

    def probabilities(self, logits: Sequence[float]) -> dict[str, float]:
        if len(logits) != len(self.labels):
            raise CalibrationError("logit and label counts differ")
        scaled = [float(value) / self.temperature for value in logits]
        maximum = max(scaled)
        exponentials = [math.exp(value - maximum) for value in scaled]
        total = sum(exponentials)
        return {
            label: value / total for label, value in zip(self.labels, exponentials, strict=True)
        }

    def scale_distribution(self, distribution: Mapping[str, float]) -> dict[str, float]:
        if set(distribution) != set(self.labels):
            raise CalibrationError("distribution labels differ from calibration labels")
        values = [float(distribution[label]) for label in self.labels]
        if any(not math.isfinite(value) or value < 0 for value in values) or sum(values) <= 0:
            raise CalibrationError("distribution must have finite non-negative mass")
        total = sum(values)
        logits = [math.log(max(value / total, 1e-12)) for value in values]
        return self.probabilities(logits)

    def model_dump(self) -> dict[str, Any]:
        return {
            "method": "temperature_scaling",
            "labels": list(self.labels),
            "temperature": self.temperature,
            "fitted_on": self.fitted_on,
        }


@dataclass(frozen=True)
class ConfidenceAbstention:
    """Validation-selected rule that withholds low-confidence top labels."""

    threshold: float
    fitted_on: str = "validation"
    score: str = "calibrated_max_probability"

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise CalibrationError("abstention threshold must be between zero and one")
        if self.fitted_on != "validation":
            raise CalibrationError("abstention must be fitted on validation")
        if self.score != "calibrated_max_probability":
            raise CalibrationError("unsupported affect abstention score")

    def abstains(self, distribution: Mapping[str, float]) -> bool:
        if not distribution:
            raise CalibrationError("cannot apply abstention to an empty distribution")
        return max(float(value) for value in distribution.values()) < self.threshold

    def model_dump(self) -> dict[str, Any]:
        return {
            "method": "confidence_threshold",
            "score": self.score,
            "threshold": self.threshold,
            "fitted_on": self.fitted_on,
        }


def fit_temperature(logits: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
    """Minimize validation NLL over one bounded positive scalar temperature."""
    if not logits or len(logits) != len(targets):
        raise CalibrationError("validation logits and targets must be aligned and non-empty")
    width = len(logits[0])
    if width < 2 or any(len(row) != width for row in logits):
        raise CalibrationError("validation logits must have one fixed width")
    if any(target < 0 or target >= width for target in targets):
        raise CalibrationError("validation target is outside the logit width")

    def nll(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        loss = 0.0
        for row, target in zip(logits, targets, strict=True):
            scaled = [float(value) / temperature for value in row]
            maximum = max(scaled)
            loss += (
                maximum
                + math.log(sum(math.exp(value - maximum) for value in scaled))
                - scaled[target]
            )
        return loss / len(targets)

    lower, upper = math.log(0.05), math.log(20.0)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_loss, right_loss = nll(left), nll(right)
    for _ in range(96):
        if left_loss <= right_loss:
            upper, right, right_loss = right, left, left_loss
            left = upper - ratio * (upper - lower)
            left_loss = nll(left)
        else:
            lower, left, left_loss = left, right, right_loss
            right = lower + ratio * (upper - lower)
            right_loss = nll(right)
    candidate = math.exp((lower + upper) / 2)
    return candidate if nll(math.log(candidate)) < nll(0.0) else 1.0


def metrics(
    probabilities: Sequence[Mapping[str, float]],
    targets: Sequence[str],
    labels: Sequence[str],
) -> dict[str, float]:
    """Compute top-label ECE and multiclass Brier score."""
    references = [{label: float(label == target) for label in labels} for target in targets]
    return {
        "ece": expected_calibration_error(targets, probabilities, labels=labels),
        "brier": multiclass_brier_score(references, probabilities, labels=labels),
    }


def selective_metrics(
    probabilities: Sequence[Mapping[str, float]],
    targets: Sequence[str],
    labels: Sequence[str],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate a confidence rule without treating retained-only scores as population scores."""
    if not probabilities or len(probabilities) != len(targets):
        raise CalibrationError("probabilities and targets must be aligned and non-empty")
    predictions = [max(row, key=row.__getitem__) for row in probabilities]
    retained = [max(row.values()) >= threshold for row in probabilities]
    supported_labels = [label for label in labels if label in targets]

    def macro_f1(indices: Sequence[int]) -> float:
        scores = []
        for label in supported_labels:
            true_positive = sum(targets[index] == predictions[index] == label for index in indices)
            false_positive = sum(
                targets[index] != label and predictions[index] == label for index in indices
            )
            false_negative = sum(
                targets[index] == label and predictions[index] != label for index in indices
            )
            denominator = 2 * true_positive + false_positive + false_negative
            scores.append(2 * true_positive / denominator if denominator else 0.0)
        return sum(scores) / len(scores)

    all_indices = list(range(len(targets)))
    retained_indices = [index for index, keep in enumerate(retained) if keep]
    # For full-population F1, an abstention is no predicted class: it can remove
    # a false positive but remains a false negative for the reference class.
    full_scores = []
    for label in supported_labels:
        true_positive = sum(
            retained[index] and targets[index] == predictions[index] == label
            for index in all_indices
        )
        false_positive = sum(
            retained[index] and targets[index] != label and predictions[index] == label
            for index in all_indices
        )
        false_negative = sum(
            targets[index] == label and (not retained[index] or predictions[index] != label)
            for index in all_indices
        )
        denominator = 2 * true_positive + false_positive + false_negative
        full_scores.append(2 * true_positive / denominator if denominator else 0.0)
    retained_probabilities = [probabilities[index] for index in retained_indices]
    retained_targets = [targets[index] for index in retained_indices]
    retained_calibration = (
        metrics(retained_probabilities, retained_targets, labels)
        if retained_indices
        else {"ece": 0.0, "brier": 0.0}
    )
    return {
        "clips": len(targets),
        "emitted": len(retained_indices),
        "abstained": len(targets) - len(retained_indices),
        "coverage": len(retained_indices) / len(targets),
        "macro_f1_full_population": sum(full_scores) / len(full_scores),
        "retained_only": {
            "macro_f1": macro_f1(retained_indices) if retained_indices else 0.0,
            **retained_calibration,
        },
    }


def fit_confidence_threshold(
    probabilities: Sequence[Mapping[str, float]],
    targets: Sequence[str],
    labels: Sequence[str],
    *,
    minimum_coverage: float = 0.8,
) -> float:
    """Select a validation threshold for retained-set quality at bounded coverage."""
    if not probabilities or len(probabilities) != len(targets):
        raise CalibrationError("validation probabilities and targets must be aligned")
    if not 0.0 < minimum_coverage <= 1.0:
        raise CalibrationError("minimum coverage must be in (0, 1]")
    confidences = sorted({max(row.values()) for row in probabilities})
    candidates = [0.0, *(math.nextafter(value, 1.0) for value in confidences)]
    scored = []
    for threshold in candidates:
        result = selective_metrics(
            probabilities,
            targets,
            labels,
            threshold=threshold,
        )
        if result["coverage"] >= minimum_coverage:
            scored.append(
                (
                    result["retained_only"]["macro_f1"],
                    result["coverage"],
                    -threshold,
                    threshold,
                )
            )
    return max(scored)[3]


def calibration_from_payload(
    payload: Mapping[str, Any],
    *,
    expected_labels: Sequence[str],
) -> TemperatureCalibration:
    """Validate a checkpoint or JSON calibration object."""
    if payload.get("method") != "temperature_scaling":
        raise CalibrationError("unsupported calibration method")
    calibration = TemperatureCalibration(
        labels=tuple(str(label) for label in payload.get("labels", [])),
        temperature=float(payload.get("temperature", 0.0)),
        fitted_on=str(payload.get("fitted_on", "")),
    )
    if calibration.labels != tuple(expected_labels):
        raise CalibrationError("calibration labels do not match model outputs")
    return calibration


def load_calibration(path: Path, *, component: str) -> TemperatureCalibration:
    """Load one component from a committed validation calibration bundle."""
    payload = _load_calibration_bundle(path)
    components = payload.get("components")
    if not isinstance(components, dict) or not isinstance(components.get(component), dict):
        raise CalibrationError(f"calibration bundle lacks {component}")
    labels = components[component].get("labels", [])
    return calibration_from_payload(components[component], expected_labels=labels)


def load_affect_abstention(path: Path, *, component: str) -> ConfidenceAbstention:
    """Load a validation-only affect abstention rule from a calibration bundle."""
    payload = _load_calibration_bundle(path)
    components = payload.get("components")
    if not isinstance(components, dict) or not isinstance(components.get(component), dict):
        raise CalibrationError(f"calibration bundle lacks {component}")
    abstention = components[component].get("abstention")
    if not isinstance(abstention, dict) or abstention.get("method") != "confidence_threshold":
        raise CalibrationError(f"calibration bundle lacks affect abstention for {component}")
    return ConfidenceAbstention(
        threshold=float(abstention.get("threshold", -1.0)),
        fitted_on=str(abstention.get("fitted_on", "")),
        score=str(abstention.get("score", "")),
    )


def _load_calibration_bundle(
    path: Path,
    seen: frozenset[Path] = frozenset(),
) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise CalibrationError("calibration bundle inheritance contains a cycle")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError(f"cannot read calibration bundle {path}: {error}") from error
    if payload.get("schema_version") != 1 or payload.get("fitted_on") != "validation":
        raise CalibrationError("unsupported or non-validation calibration bundle")
    components = payload.get("components")
    if not isinstance(components, dict):
        raise CalibrationError("calibration bundle components must be an object")
    base_name = payload.get("base_bundle")
    if base_name is None:
        return payload
    if not isinstance(base_name, str) or Path(base_name).name != base_name:
        raise CalibrationError("base_bundle must name a sibling JSON file")
    base_path = path.parent / base_name
    if base_path == path:
        raise CalibrationError("calibration bundle cannot extend itself")
    base = _load_calibration_bundle(base_path, seen | {resolved})
    merged_components = {name: dict(value) for name, value in base["components"].items()}
    for name, value in components.items():
        if not isinstance(value, dict):
            raise CalibrationError("calibration component must be an object")
        merged_components[name] = {
            **merged_components.get(name, {}),
            **value,
        }
    return {**base, **payload, "components": merged_components}
