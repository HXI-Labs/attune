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


def test_affect_abstention_fit_never_uses_inspection_test_ids() -> None:
    script = load_script()
    validation = [
        {
            **record("validation", "actor-1051-a", "class"),
            "component": "emotion2vec_plus_affect",
        },
        {
            **record("validation", "actor-1052-b", "none"),
            "component": "emotion2vec_plus_affect",
            "logits": [2.0, 1.9],
        },
    ]
    first_test = [
        {
            **record("inspection_test", f"test-{index}", "class"),
            "component": "emotion2vec_plus_affect",
            "logits": [10.0, -10.0],
        }
        for index in range(2)
    ]
    changed_test = [{**row, "target": "none", "logits": [-10.0, 10.0]} for row in first_test]

    first = script.calibrate(validation + first_test, expected_test_clips=2)
    changed = script.calibrate(validation + changed_test, expected_test_clips=2)

    first_policy = first["components"]["emotion2vec_plus_affect"]["abstention"]
    changed_policy = changed["components"]["emotion2vec_plus_affect"]["abstention"]
    assert first_policy["threshold"] == changed_policy["threshold"]
    assert first_policy["fitted_on"] == "validation"
    assert first_policy["inspection_test"]["designation"].startswith("test")
