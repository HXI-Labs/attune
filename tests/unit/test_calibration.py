from __future__ import annotations

import json
from pathlib import Path

import pytest

from attune.calibration import (
    CalibrationError,
    TemperatureCalibration,
    fit_temperature,
    load_calibration,
    metrics,
)


def test_temperature_scaling_reduces_overconfidence() -> None:
    logits = [[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 8.0]]
    targets = [0, 1, 1, 0]

    temperature = fit_temperature(logits, targets)
    scaling = TemperatureCalibration(("a", "b"), temperature)
    raw = [TemperatureCalibration(("a", "b"), 1.0).probabilities(row) for row in logits]
    calibrated = [scaling.probabilities(row) for row in logits]

    assert temperature > 1.0
    assert metrics(calibrated, ["a", "b", "b", "a"], ("a", "b"))["brier"] < (
        metrics(raw, ["a", "b", "b", "a"], ("a", "b"))["brier"]
    )


def test_load_calibration_rejects_test_fitting(tmp_path: Path) -> None:
    artifact = tmp_path / "calibration.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fitted_on": "inspection_test",
                "components": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CalibrationError, match="non-validation"):
        load_calibration(artifact, component="affect")


def test_scale_distribution_preserves_label_order_and_argmax() -> None:
    scaling = TemperatureCalibration(("neutral", "joy", "other"), 2.0)

    result = scaling.scale_distribution(
        {"other": 0.1, "neutral": 0.7, "joy": 0.2}
    )

    assert list(result) == ["neutral", "joy", "other"]
    assert max(result, key=result.__getitem__) == "neutral"
    assert sum(result.values()) == pytest.approx(1.0)
