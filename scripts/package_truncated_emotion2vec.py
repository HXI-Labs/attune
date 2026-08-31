#!/usr/bin/env python3
"""Package only the emotion2vec+ weights reachable by the compact student."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import torch

BLOCK_PATTERN = re.compile(r"^d2v_model\.blocks\.(\d+)\.")
FRONTEND_PREFIX = "d2v_model.modality_encoders.AUDIO."
DECODER_PREFIX = "d2v_model.modality_encoders.AUDIO.decoder."
PACKAGE_FILES = ("config.yaml", "configuration.json", "tokens.txt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reachable_state(state: dict[str, Any], depth: int) -> dict[str, Any]:
    selected = {}
    for name, value in state.items():
        if name.startswith(FRONTEND_PREFIX) and not name.startswith(DECODER_PREFIX):
            selected[name] = value
            continue
        match = BLOCK_PATTERN.match(name)
        if match and int(match.group(1)) < depth:
            selected[name] = value
    if not selected:
        raise ValueError("teacher checkpoint contains no reachable emotion2vec weights")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-path", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    arguments = parser.parse_args()

    student = torch.load(arguments.student_checkpoint, map_location="cpu", weights_only=True)
    depth = int(student["depth"])
    source_path = arguments.teacher_path / "model.pt"
    source_sha256 = sha256(source_path)
    if source_sha256 != student["teacher_checkpoint_sha256"]:
        raise ValueError("student and teacher checkpoint hashes differ")
    teacher = torch.load(source_path, map_location="cpu", weights_only=True)
    state = reachable_state(teacher["model"], depth)
    parameter_count = sum(value.numel() for value in state.values())
    expected_parameters = int(student["truncated_emotion2vec_parameters"]) - int(
        student["head_parameters"]
    )
    if parameter_count != expected_parameters:
        raise ValueError(
            f"reachable parameter count is {parameter_count}, expected {expected_parameters}"
        )

    package = {
        key: teacher[key]
        for key in ("args", "cfg", "criterion", "optimizer_history", "task_state", "extra_state")
    }
    package["model"] = state
    package["last_optimizer_state"] = None
    package["attune_source_checkpoint_sha256"] = source_sha256
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = arguments.output_dir / "model.pt"
    temporary = model_path.with_suffix(".pt.tmp")
    torch.save(package, temporary)
    temporary.replace(model_path)
    for name in PACKAGE_FILES:
        shutil.copyfile(arguments.teacher_path / name, arguments.output_dir / name)

    report = {
        "schema_version": "1.0",
        "method": "reachable_emotion2vec_weights_only",
        "depth": depth,
        "source_teacher_sha256": source_sha256,
        "student_checkpoint": str(arguments.student_checkpoint),
        "student_checkpoint_sha256": sha256(arguments.student_checkpoint),
        "parameter_count": parameter_count,
        "tensor_count": len(state),
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "model_bytes": model_path.stat().st_size,
    }
    arguments.provenance.parent.mkdir(parents=True, exist_ok=True)
    arguments.provenance.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
