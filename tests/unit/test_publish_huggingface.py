from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "publish_huggingface.py"
SPEC = importlib.util.spec_from_file_location("publish_huggingface", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_release_files_default_to_int8_bundle(tmp_path: Path) -> None:
    required = (
        "huggingface/README.md",
        "huggingface/THIRD_PARTY_NOTICE.md",
        "artifacts/models/attune-split-tail-v0.1-int8.onnx",
        "artifacts/models/attune-split-tail-v0.1-int8.quantization.json",
        "artifacts/evaluation/split-tail-v0.1/int8-calibration.json",
        "data/schemas/attune-output-v2.0.schema.json",
        "artifacts/release/v0.1/release-gates.json",
        "artifacts/release/v0.1/artifact-manifest.json",
    )
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    files = MODULE.release_files(tmp_path, include_fp=False)

    assert {destination for _, destination in files} == {
        "README.md",
        "THIRD_PARTY_NOTICE.md",
        "attune-cadence-241m-int8.onnx",
        "quantization.json",
        "int8-calibration.json",
        "attune-output-v2.0.schema.json",
        "release-gates.json",
        "artifact-manifest.json",
    }


def test_release_files_fail_closed_when_an_artifact_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing release files"):
        MODULE.release_files(tmp_path, include_fp=False)
