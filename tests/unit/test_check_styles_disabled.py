from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "check_styles_disabled.py"
SPEC = importlib.util.spec_from_file_location("check_styles_disabled", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_check_styles_disabled_hashes_empty_runtime_output(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"style_enabled_labels": []}))
    inference = tmp_path / "output.json"
    inference.write_text(json.dumps({"styles": []}))

    report = MODULE.check_styles_disabled(model, calibration, [inference])

    assert report["candidate_passes"] is True
    assert report["deployment_enabled_labels"] == []
    assert len(report["model"]["sha256"]) == 64


def test_check_styles_disabled_rejects_style_output(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"style_enabled_labels": []}))
    inference = tmp_path / "output.json"
    inference.write_text(json.dumps({"styles": [{"label": "whispering"}]}))

    with pytest.raises(ValueError, match="not disabled"):
        MODULE.check_styles_disabled(model, calibration, [inference])
