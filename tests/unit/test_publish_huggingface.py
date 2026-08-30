from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from attune.integrity import file_digest

SCRIPT = Path(__file__).parents[2] / "scripts" / "publish_huggingface.py"
SPEC = importlib.util.spec_from_file_location("publish_huggingface", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def bundle_manifest(*files: tuple[str, str, str]) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "release_name": "Attune Cadence v0.1",
        "gate_report": "artifacts/release/v0.1/release-gates.json",
        "files": [
            {"source": source, "destination": destination, "sha256": sha256}
            for source, destination, sha256 in files
        ],
    }


def complete_bundle(tmp_path: Path) -> dict[str, object]:
    destinations = MODULE.REQUIRED_BUNDLE_DESTINATIONS | {"cadence-int8.onnx"}
    records = []
    for destination in sorted(destinations):
        relative = (
            "artifacts/release/v0.1/release-gates.json"
            if destination == "release-gates.json"
            else f"release/{destination}"
        )
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        records.append((relative, destination, file_digest(path)))
    return bundle_manifest(*records)


def test_release_files_follow_explicit_bundle_manifest(tmp_path: Path) -> None:
    manifest = complete_bundle(tmp_path)

    files = MODULE.release_files(tmp_path, manifest)

    assert {destination for _, destination in files} == (
        MODULE.REQUIRED_BUNDLE_DESTINATIONS | {"cadence-int8.onnx"}
    )


def test_release_files_fail_closed_when_an_artifact_is_missing(tmp_path: Path) -> None:
    manifest = complete_bundle(tmp_path)
    manifest["files"][0]["source"] = "missing.onnx"

    with pytest.raises(FileNotFoundError, match="missing release files"):
        MODULE.release_files(tmp_path, manifest)


def test_release_files_reject_duplicate_destinations(tmp_path: Path) -> None:
    manifest = complete_bundle(tmp_path)
    manifest["files"][1]["destination"] = manifest["files"][0]["destination"]

    with pytest.raises(RuntimeError, match="duplicate Hugging Face destination"):
        MODULE.release_files(tmp_path, manifest)


def test_release_files_reject_checksum_mismatch(tmp_path: Path) -> None:
    manifest = complete_bundle(tmp_path)
    manifest["files"][0]["sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        MODULE.release_files(tmp_path, manifest)


def test_release_files_require_gate_report_in_upload(tmp_path: Path) -> None:
    manifest = complete_bundle(tmp_path)
    gate_record = next(
        record for record in manifest["files"] if record["destination"] == "release-gates.json"
    )
    gate_record["source"] = "release/release-gates.json"
    alternate_gate = tmp_path / gate_record["source"]
    alternate_gate.parent.mkdir(parents=True, exist_ok=True)
    alternate_gate.touch()
    gate_record["sha256"] = file_digest(alternate_gate)

    with pytest.raises(RuntimeError, match="gate_report must be included"):
        MODULE.release_files(tmp_path, manifest)


def test_load_bundle_manifest_rejects_stale_schema(tmp_path: Path) -> None:
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps({"schema_version": "0.1"}))

    with pytest.raises(RuntimeError, match="unsupported.*schema"):
        MODULE.load_bundle_manifest(path)


def test_publication_is_blocked_when_release_gate_is_false(tmp_path: Path) -> None:
    gate = tmp_path / "artifacts/release/v0.1/release-gates.json"
    gate.parent.mkdir(parents=True)
    gate.write_text(
        json.dumps(
            {
                "schema_version": "1.2",
                "release_ready": False,
                "gates": [
                    {
                        "name": name,
                        "passed": name != "hostile_speech_regression",
                    }
                    for name in MODULE.REQUIRED_PUBLICATION_GATES
                ],
            }
        )
    )

    with pytest.raises(RuntimeError, match="hostile_speech_regression"):
        MODULE.require_release_ready(gate)


def test_publication_gate_accepts_ready_report(tmp_path: Path) -> None:
    gate = tmp_path / "artifacts/release/v0.1/release-gates.json"
    gate.parent.mkdir(parents=True)
    gate.write_text(
        json.dumps(
            {
                "schema_version": "1.2",
                "release_ready": True,
                "gates": [
                    {"name": name, "passed": True} for name in MODULE.REQUIRED_PUBLICATION_GATES
                ],
            }
        )
    )

    MODULE.require_release_ready(gate)


def test_publication_rejects_stale_gate_schema(tmp_path: Path) -> None:
    gate = tmp_path / "artifacts/release/v0.1/release-gates.json"
    gate.parent.mkdir(parents=True)
    gate.write_text(json.dumps({"schema_version": "1.0", "release_ready": True, "gates": []}))

    with pytest.raises(RuntimeError, match="stale release-gate schema"):
        MODULE.require_release_ready(gate)


def test_publication_rejects_missing_external_gate(tmp_path: Path) -> None:
    gate = tmp_path / "artifacts/release/v0.1/release-gates.json"
    gate.parent.mkdir(parents=True)
    gate.write_text(
        json.dumps(
            {
                "schema_version": "1.2",
                "release_ready": True,
                "gates": [
                    {"name": "hostile_speech_regression", "passed": True},
                ],
            }
        )
    )

    with pytest.raises(RuntimeError, match="missing required gates"):
        MODULE.require_release_ready(gate)
