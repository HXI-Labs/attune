#!/usr/bin/env python3
"""Publish a gated Attune Cadence bundle to a Hugging Face model repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from attune.integrity import file_digest

REQUIRED_PUBLICATION_GATES = {
    "event_external_validation",
    "style_external_validation",
    "affect_external_validation",
    "int8_event_external_validation",
    "int8_style_external_validation",
    "int8_affect_external_validation",
    "hostile_speech_regression",
}
REQUIRED_BUNDLE_DESTINATIONS = {
    "README.md",
    "THIRD_PARTY_NOTICE.md",
    "calibration.json",
    "attune-output-v2.0.schema.json",
    "quantization.json",
    "release-gates.json",
    "artifact-manifest.json",
    "hostile-speech-regression.json",
}
SENSEVOICE_REVISION = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"
REQUIRED_TRAINING_SOURCES = {
    "attune_inline_event_mixtures_v0.1",
    "berst_v1",
    "common_voice_17_en",
    "crema_d_paired_v0.1",
    "crema_d_perceptual_v1",
    "dcase2016_task2",
    "disfluency_speech_v0.1",
    "fsd50k_bounded_v0.1",
    "subesco_v1_1",
    "thorsten_voice_2021_06_emotional",
    "vocalsound_v0.1",
}


def load_bundle_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read release bundle manifest {path}: {error}") from error
    if manifest.get("schema_version") != "1.0":
        raise RuntimeError("unsupported Hugging Face bundle-manifest schema")
    if not isinstance(manifest.get("release_name"), str) or not manifest["release_name"].strip():
        raise RuntimeError("bundle manifest requires release_name")
    if not isinstance(manifest.get("gate_report"), str) or not manifest["gate_report"]:
        raise RuntimeError("bundle manifest requires gate_report")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("bundle manifest requires a non-empty files list")
    return manifest


def release_files(root: Path, manifest: dict[str, Any]) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    destinations: set[str] = set()
    for record in manifest["files"]:
        if not isinstance(record, dict):
            raise RuntimeError("bundle file entries must be objects")
        source_name = record.get("source")
        destination = record.get("destination")
        expected_sha256 = record.get("sha256")
        if not isinstance(source_name, str) or not source_name:
            raise RuntimeError("bundle file entries require source")
        if not isinstance(destination, str) or not destination:
            raise RuntimeError("bundle file entries require destination")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise RuntimeError("bundle file entries require a SHA-256 digest")
        if destination.startswith("/") or ".." in Path(destination).parts:
            raise RuntimeError(f"unsafe Hugging Face destination: {destination}")
        if destination in destinations:
            raise RuntimeError(f"duplicate Hugging Face destination: {destination}")
        destinations.add(destination)
        source = (root / source_name).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"bundle source is outside repository: {source_name}") from error
        files.append((source, destination))
    missing = [str(source) for source, _ in files if not source.is_file()]
    if missing:
        raise FileNotFoundError("missing release files: " + ", ".join(missing))
    mismatched = [
        str(source)
        for (source, _), record in zip(files, manifest["files"], strict=True)
        if file_digest(source) != record["sha256"]
    ]
    if mismatched:
        raise RuntimeError("release file checksum mismatch: " + ", ".join(mismatched))
    missing_destinations = sorted(REQUIRED_BUNDLE_DESTINATIONS - destinations)
    if missing_destinations:
        raise RuntimeError(
            "bundle is missing required destinations: " + ", ".join(missing_destinations)
        )
    if not any(destination.endswith(".onnx") for destination in destinations):
        raise RuntimeError("bundle requires an ONNX model")
    gate_source = (root / manifest["gate_report"]).resolve()
    if gate_source not in {source for source, _ in files}:
        raise RuntimeError("bundle gate_report must be included in files")
    return files


def require_release_ready(gate_path: Path) -> None:
    try:
        report = json.loads(gate_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot verify release gates at {gate_path}: {error}") from error
    if report.get("schema_version") != "1.2":
        raise RuntimeError("Hugging Face publication is blocked: stale release-gate schema")

    gate_status = {
        str(gate.get("name")): gate.get("passed") is True
        for gate in report.get("gates", [])
        if isinstance(gate, dict)
    }
    missing = sorted(REQUIRED_PUBLICATION_GATES - gate_status.keys())
    if missing:
        raise RuntimeError(
            "Hugging Face publication is blocked: missing required gates: " + ", ".join(missing)
        )

    if report.get("release_ready") is not True:
        failed = [name for name, passed in gate_status.items() if not passed]
        detail = ", ".join(failed) if failed else "release_ready is not true"
        raise RuntimeError(f"Hugging Face publication is blocked: {detail}")


def require_public_redistribution_review(path: Path, bundle_manifest: dict[str, Any]) -> None:
    try:
        review = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot verify public redistribution review at {path}: {error}"
        ) from error
    if review.get("schema_version") != "1.1":
        raise RuntimeError("public publication is blocked: stale redistribution-review schema")
    if review.get("public_weight_redistribution_approved") is not True:
        raise RuntimeError("public publication is blocked: weight redistribution is not approved")
    if review.get("sensevoice_revision") != SENSEVOICE_REVISION:
        raise RuntimeError("public publication is blocked: review covers a different base revision")
    for field in ("reviewed_by", "reviewed_at", "scope"):
        if not isinstance(review.get(field), str) or not review[field].strip():
            raise RuntimeError(
                f"public publication is blocked: redistribution review lacks {field}"
            )
    reviewed_sources = review.get("training_sources")
    if not isinstance(reviewed_sources, list) or set(reviewed_sources) != REQUIRED_TRAINING_SOURCES:
        raise RuntimeError(
            "public publication is blocked: review does not cover the exact training-source set"
        )
    approved_artifacts = review.get("approved_artifacts")
    if not isinstance(approved_artifacts, list):
        raise RuntimeError("public publication is blocked: review lacks approved_artifacts")
    approved_by_destination = {
        record.get("destination"): record.get("sha256")
        for record in approved_artifacts
        if isinstance(record, dict)
    }
    bundled_models = {
        record["destination"]: record["sha256"]
        for record in bundle_manifest["files"]
        if record["destination"].endswith(".onnx")
    }
    if approved_by_destination != bundled_models:
        raise RuntimeError(
            "public publication is blocked: review does not cover the exact ONNX artifacts"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repo, e.g. org/name")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a public repo. Omit until the public redistribution review is complete.",
    )
    parser.add_argument(
        "--redistribution-review",
        type=Path,
        help="Required approval record when --public is used.",
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    root = arguments.root.resolve()
    manifest_path = arguments.bundle_manifest
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = load_bundle_manifest(manifest_path)
    files = release_files(root, manifest)
    gate_path = (root / manifest["gate_report"]).resolve()
    try:
        gate_path.relative_to(root)
    except ValueError as error:
        raise RuntimeError("gate report is outside repository") from error
    require_release_ready(gate_path)
    if arguments.public:
        if arguments.redistribution_review is None:
            raise RuntimeError("public publication requires --redistribution-review")
        review_path = arguments.redistribution_review
        if not review_path.is_absolute():
            review_path = root / review_path
        require_public_redistribution_review(review_path, manifest)
    plan = {
        "release_name": manifest["release_name"],
        "repo_id": arguments.repo_id,
        "private": not arguments.public,
        "files": [
            {"source": str(source.relative_to(root)), "destination": destination}
            for source, destination in files
        ],
    }
    if arguments.dry_run:
        print(json.dumps(plan, indent=2))
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError("install huggingface_hub before publishing") from error

    api = HfApi()
    api.create_repo(
        repo_id=arguments.repo_id,
        repo_type="model",
        private=not arguments.public,
        exist_ok=True,
    )
    for source, destination in files:
        print(f"Uploading {destination}...")
        api.upload_file(
            path_or_fileobj=source,
            path_in_repo=destination,
            repo_id=arguments.repo_id,
            repo_type="model",
            commit_message=f"Upload {manifest['release_name']}: {destination}",
        )
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
