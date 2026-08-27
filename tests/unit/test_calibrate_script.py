from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from attune.calibration import CalibrationError


def load_script() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "calibrate.py"
    spec = importlib.util.spec_from_file_location("calibrate_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(split: str, clip_id: str, target: str) -> dict:
    return {
        "component": "probe",
        "split": split,
        "clip_id": clip_id,
        "labels": ["class", "none"],
        "logits": [3.0, 0.0],
        "target": target,
    }


def test_script_marks_inspection_metrics_as_test() -> None:
    report = load_script().calibrate(
        [
            record("validation", "validation-1", "class"),
            record("validation", "validation-2", "none"),
            record("inspection_test", "test-1", "class"),
        ],
        expected_test_clips=1,
    )

    test = report["components"]["probe"]["inspection_test"]
    assert test["designation"].startswith("test")
    assert report["inspection_test"]["used_for_fitting_or_selection"] is False
    assert set(test["before"]) == {"ece", "brier"}


def test_script_rejects_validation_test_leakage() -> None:
    with pytest.raises(CalibrationError, match="leaks clip IDs"):
        load_script().calibrate(
            [
                record("validation", "shared", "class"),
                record("inspection_test", "shared", "class"),
            ],
            expected_test_clips=1,
        )
