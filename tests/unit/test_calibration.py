from __future__ import annotations

import json
from pathlib import Path

import pytest

from attune.calibration import (
    CalibrationError,
    ConfidenceAbstention,
    TemperatureCalibration,
    fit_confidence_threshold,
    fit_temperature,
    load_affect_abstention,
    load_calibration,
    metrics,
    selective_metrics,
)


def test_temperature_scaling_reduces_overconfidence() -> None:
    logits = [[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 8.0]]
    targets = [0, 1, 1, 0]

    temperature = fit_temperature(logits, targets)
    scaling = TemperatureCalibration(("a", "b"), temperature)
    raw = [TemperatureCalibration(("a", "b"), 1.0).probabilities(row) for row in logits]
    calibrated = [scaling.probabilities(row) for row in logits]

    assert temperature > 1.0
    assert (
        metrics(calibrated, ["a", "b", "b", "a"], ("a", "b"))["brier"]
        < (metrics(raw, ["a", "b", "b", "a"], ("a", "b"))["brier"])
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

    result = scaling.scale_distribution({"other": 0.1, "neutral": 0.7, "joy": 0.2})

    assert list(result) == ["neutral", "joy", "other"]
    assert max(result, key=result.__getitem__) == "neutral"
    assert sum(result.values()) == pytest.approx(1.0)


def test_confidence_abstention_selects_validation_threshold() -> None:
    probabilities = [
        {"a": 0.9, "b": 0.1},
        {"a": 0.55, "b": 0.45},
        {"a": 0.1, "b": 0.9},
        {"a": 0.6, "b": 0.4},
    ]
    targets = ["a", "b", "b", "b"]

    threshold = fit_confidence_threshold(
        probabilities,
        targets,
        ("a", "b"),
        minimum_coverage=0.5,
    )
    result = selective_metrics(
        probabilities,
        targets,
        ("a", "b"),
        threshold=threshold,
    )

    assert threshold > 0.6
    assert result["coverage"] == 0.5
    assert (
        result["macro_f1_full_population"]
        > selective_metrics(
            probabilities,
            targets,
            ("a", "b"),
            threshold=0.0,
        )["macro_f1_full_population"]
    )


def test_load_affect_abstention_rejects_test_fitting(tmp_path: Path) -> None:
    artifact = tmp_path / "calibration.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fitted_on": "validation",
                "components": {
                    "affect": {
                        "abstention": {
                            "method": "confidence_threshold",
                            "score": "calibrated_max_probability",
                            "threshold": 0.6,
                            "fitted_on": "inspection_test",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CalibrationError, match="fitted on validation"):
        load_affect_abstention(artifact, component="affect")


def test_confidence_abstention_applies_strict_threshold() -> None:
    policy = ConfidenceAbstention(0.6)

    assert policy.abstains({"a": 0.59, "b": 0.41})
    assert not policy.abstains({"a": 0.6, "b": 0.4})


def test_phase2_bundle_extends_phase1_without_rewriting_it(tmp_path: Path) -> None:
    (tmp_path / "phase1.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fitted_on": "validation",
                "components": {
                    "affect": {
                        "method": "temperature_scaling",
                        "labels": ["a", "b"],
                        "temperature": 2.0,
                        "fitted_on": "validation",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    phase2 = tmp_path / "phase2.json"
    phase2.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fitted_on": "validation",
                "base_bundle": "phase1.json",
                "components": {
                    "affect": {
                        "abstention": {
                            "method": "confidence_threshold",
                            "score": "calibrated_max_probability",
                            "threshold": 0.8,
                            "fitted_on": "validation",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert load_calibration(phase2, component="affect").temperature == 2.0
    assert load_affect_abstention(phase2, component="affect").threshold == 0.8
