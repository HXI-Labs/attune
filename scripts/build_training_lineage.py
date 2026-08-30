#!/usr/bin/env python3
"""Bind a training checkpoint to its code, inputs, command, and hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import torch

from attune.integrity import file_digest
from attune.training import trainer as training_module


def training_source_digest(read_source: Callable[[str], bytes]) -> str:
    digest = hashlib.sha256()
    for relative in training_module._TRAINING_SOURCE_FILES:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(read_source(relative))
        digest.update(b"\0")
    return digest.hexdigest()


def git_source_digest(root: Path, revision: str) -> str:
    def read_source(relative: str) -> bytes:
        completed = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:{relative}"],
            check=True,
            capture_output=True,
        )
        return completed.stdout

    return training_source_digest(read_source)


def resolve_revision(root: Path, revision: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", f"{revision}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def host_environment(requested_device: str) -> dict[str, object]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "requested_device": requested_device,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }


def build_lineage(
    *,
    root: Path,
    report_path: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    config_path: Path,
    loss_weights_path: Path,
    training_revision: str,
    command: str,
    concurrent_change_reason: str | None,
) -> dict[str, object]:
    report = json.loads(report_path.read_text())
    manifest_digest = file_digest(manifest_path)
    if report.get("manifest_sha256") != manifest_digest:
        raise ValueError("training report does not match the supplied manifest")
    resolved_revision = resolve_revision(root, training_revision)
    source_digest = git_source_digest(root, resolved_revision)
    report_source_digest = report.get("training_source_sha256")
    source_changed = report_source_digest != source_digest
    if source_changed and not concurrent_change_reason:
        raise ValueError(
            "training report source hash differs from the start revision; "
            "provide --concurrent-change-reason"
        )
    if not command.strip():
        raise ValueError("training command must not be empty")
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "training_revision": resolved_revision,
        "training_source_sha256_at_start": source_digest,
        "training_report_source_sha256": report_source_digest,
        "concurrent_source_change": source_changed,
        "concurrent_change_reason": concurrent_change_reason if source_changed else None,
        "command": command,
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": manifest_digest},
            "config": {"path": str(config_path), "sha256": file_digest(config_path)},
            "loss_weights": {
                "path": str(loss_weights_path),
                "sha256": file_digest(loss_weights_path),
            },
            "training_report": {
                "path": str(report_path),
                "sha256": file_digest(report_path),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": file_digest(checkpoint_path),
            },
        },
        "trainer": report.get("trainer"),
        "parameter_summary": report.get("parameter_summary"),
        "elapsed_seconds": report.get("elapsed_seconds"),
        "environment": host_environment(str(report.get("trainer", {}).get("device", "unknown"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--loss-weights", type=Path, required=True)
    parser.add_argument("--training-revision", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--concurrent-change-reason")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    lineage = build_lineage(
        root=root,
        report_path=arguments.training_report,
        checkpoint_path=arguments.checkpoint,
        manifest_path=arguments.manifest,
        config_path=arguments.config,
        loss_weights_path=arguments.loss_weights,
        training_revision=arguments.training_revision,
        command=arguments.command,
        concurrent_change_reason=arguments.concurrent_change_reason,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(lineage, indent=2) + "\n")
    print(json.dumps(lineage, indent=2))


if __name__ == "__main__":
    main()
