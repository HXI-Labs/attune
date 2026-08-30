#!/usr/bin/env python3
"""Verify that a candidate checkpoint changed only approved tensor prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict(checkpoint: dict[str, Any], path: Path) -> dict[str, torch.Tensor]:
    if checkpoint.get("format") != "attune_delta_v1":
        raise ValueError(f"{path} is not an attune_delta_v1 checkpoint")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{path} contains no state_dict")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"{path} state_dict contains a non-tensor value")
    return state


def audit_scope(
    initial_path: Path,
    candidate_path: Path,
    *,
    allowed_prefixes: tuple[str, ...],
) -> dict[str, Any]:
    if not allowed_prefixes or any(not prefix for prefix in allowed_prefixes):
        raise ValueError("at least one non-empty allowed prefix is required")
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    if not isinstance(initial, dict) or not isinstance(candidate, dict):
        raise ValueError("checkpoint root must be a mapping")
    if initial.get("adaptation_policy") != candidate.get("adaptation_policy"):
        raise ValueError("candidate adaptation policy differs from the initial checkpoint")
    initial_state = _state_dict(initial, initial_path)
    candidate_state = _state_dict(candidate, candidate_path)
    missing = sorted(set(initial_state) - set(candidate_state))
    unexpected = sorted(set(candidate_state) - set(initial_state))
    if missing or unexpected:
        raise ValueError(
            f"checkpoint tensor keys differ; missing={missing}, unexpected={unexpected}"
        )

    changed = []
    shape_mismatches = []
    for name in sorted(initial_state):
        before = initial_state[name]
        after = candidate_state[name]
        if before.shape != after.shape or before.dtype != after.dtype:
            shape_mismatches.append(name)
            continue
        if torch.equal(before, after):
            continue
        difference = (after.float() - before.float()).abs()
        changed.append(
            {
                "name": name,
                "shape": list(after.shape),
                "dtype": str(after.dtype),
                "maximum_absolute_delta": float(difference.max()) if difference.numel() else 0.0,
            }
        )
    violations = [
        tensor["name"] for tensor in changed if not tensor["name"].startswith(allowed_prefixes)
    ]
    report = {
        "schema_version": "1.0",
        "initial": {"path": str(initial_path), "sha256": _sha256(initial_path)},
        "candidate": {"path": str(candidate_path), "sha256": _sha256(candidate_path)},
        "adaptation_policy": candidate["adaptation_policy"],
        "allowed_prefixes": list(allowed_prefixes),
        "tensor_count": len(candidate_state),
        "changed_tensor_count": len(changed),
        "unchanged_tensor_count": len(candidate_state) - len(changed),
        "changed_tensors": changed,
        "shape_or_dtype_mismatches": shape_mismatches,
        "scope_violations": violations,
        "passed": bool(changed) and not shape_mismatches and not violations,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--allowed-prefix", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = audit_scope(
        arguments.initial,
        arguments.candidate,
        allowed_prefixes=tuple(arguments.allowed_prefix),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(arguments.output), "passed": report["passed"]}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
