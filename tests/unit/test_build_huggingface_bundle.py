from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "build_huggingface_bundle.py"
SPEC = importlib.util.spec_from_file_location("build_huggingface_bundle", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_bundle_manifest_hashes_explicit_file_mappings(tmp_path: Path) -> None:
    model = tmp_path / "release/model.onnx"
    gate = tmp_path / "release/release-gates.json"
    model.parent.mkdir()
    model.write_bytes(b"model")
    gate.write_text("{}")

    manifest = MODULE.build_bundle_manifest(
        root=tmp_path,
        release_name="Attune Cadence v0.1",
        gate_report="release/release-gates.json",
        mappings=[
            ("release/model.onnx", "cadence.onnx"),
            ("release/release-gates.json", "release-gates.json"),
        ],
    )

    assert manifest["gate_report"] == "release/release-gates.json"
    assert manifest["files"][0]["bytes"] == 5
    assert len(manifest["files"][0]["sha256"]) == 64


def test_bundle_manifest_requires_gate_report_mapping(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")

    with pytest.raises(ValueError, match="gate report must be included"):
        MODULE.build_bundle_manifest(
            root=tmp_path,
            release_name="Attune Cadence v0.1",
            gate_report="release-gates.json",
            mappings=[("model.onnx", "cadence.onnx")],
        )


def test_bundle_manifest_rejects_duplicate_destinations(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_text("first")
    second.write_text("second")

    with pytest.raises(ValueError, match="duplicate.*destination"):
        MODULE.build_bundle_manifest(
            root=tmp_path,
            release_name="Attune Cadence v0.1",
            gate_report="first",
            mappings=[("first", "same"), ("second", "same")],
        )


def test_mapping_file_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    mapping_file = tmp_path / "files.txt"
    mapping_file.write_text(
        "# Model\n\nrelease/model.onnx=cadence.onnx\nrelease/gates.json=release-gates.json\n"
    )

    assert MODULE.load_mapping_file(mapping_file) == [
        ("release/model.onnx", "cadence.onnx"),
        ("release/gates.json", "release-gates.json"),
    ]
