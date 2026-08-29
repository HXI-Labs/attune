#!/usr/bin/env python3
"""Publish the Attune Cadence release bundle to a private Hugging Face model repo."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReleaseFile:
    source: str
    destination: str
    optional: bool = False


RELEASE_FILES = (
    ReleaseFile("huggingface/README.md", "README.md"),
    ReleaseFile("huggingface/THIRD_PARTY_NOTICE.md", "THIRD_PARTY_NOTICE.md"),
    ReleaseFile(
        "artifacts/models/attune-split-tail-v0.1-int8.onnx",
        "attune-cadence-241m-int8.onnx",
    ),
    ReleaseFile(
        "artifacts/models/attune-split-tail-v0.1-int8.quantization.json",
        "quantization.json",
    ),
    ReleaseFile(
        "artifacts/evaluation/split-tail-v0.1/int8-calibration.json",
        "int8-calibration.json",
    ),
    ReleaseFile(
        "data/schemas/attune-output-v2.0.schema.json",
        "attune-output-v2.0.schema.json",
    ),
    ReleaseFile("artifacts/release/v0.1/release-gates.json", "release-gates.json"),
    ReleaseFile(
        "artifacts/release/v0.1/artifact-manifest.json",
        "artifact-manifest.json",
    ),
    ReleaseFile(
        "artifacts/models/attune-split-tail-v0.1-fp.onnx",
        "attune-cadence-241m-fp.onnx",
        optional=True,
    ),
    ReleaseFile(
        "artifacts/models/attune-split-tail-v0.1-fp.export.json",
        "fp-export.json",
        optional=True,
    ),
)


def release_files(root: Path, *, include_fp: bool) -> list[tuple[Path, str]]:
    selected = [item for item in RELEASE_FILES if include_fp or not item.optional]
    files = [(root / item.source, item.destination) for item in selected]
    missing = [str(source) for source, _ in files if not source.is_file()]
    if missing:
        raise FileNotFoundError("missing release files: " + ", ".join(missing))
    return files


def require_release_ready(root: Path) -> None:
    gate_path = root / "artifacts/release/v0.1/release-gates.json"
    try:
        report = json.loads(gate_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot verify release gates at {gate_path}: {error}") from error
    if report.get("release_ready") is not True:
        failed = [
            str(gate.get("name", "unknown"))
            for gate in report.get("gates", [])
            if gate.get("passed") is not True
        ]
        detail = ", ".join(failed) if failed else "release_ready is not true"
        raise RuntimeError(f"Hugging Face publication is blocked: {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repo, e.g. org/name")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--include-fp", action="store_true")
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a public repo. Omit until the public redistribution review is complete.",
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    root = arguments.root.resolve()
    files = release_files(root, include_fp=arguments.include_fp)
    require_release_ready(root)
    plan = {
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
            commit_message=f"Upload Attune Cadence v0.1: {destination}",
        )
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
