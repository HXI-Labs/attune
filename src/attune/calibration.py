"""Validation-only temperature scaling for categorical model outputs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
            label: value / total
            for label, value in zip(self.labels, exponentials, strict=True)
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
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError(f"cannot read calibration bundle {path}: {error}") from error
    if payload.get("schema_version") != 1 or payload.get("fitted_on") != "validation":
        raise CalibrationError("unsupported or non-validation calibration bundle")
    components = payload.get("components")
    if not isinstance(components, dict) or not isinstance(components.get(component), dict):
        raise CalibrationError(f"calibration bundle lacks {component}")
    labels = components[component].get("labels", [])
    return calibration_from_payload(components[component], expected_labels=labels)
