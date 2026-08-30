#!/usr/bin/env python3
"""Compose independently accepted Attune heads into one delta checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

OVERLAY_PREFIXES = {
    "event": ("event_head.",),
    "style": ("style_projection.", "style_head."),
    "affect": ("affect_projection.", "affect_head."),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_delta(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "attune_delta_v1":
        raise ValueError(f"{path}: expected an attune_delta_v1 checkpoint")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{path}: checkpoint contains no state_dict")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"{path}: state_dict contains a non-tensor value")
    return checkpoint


def load_acceptance(path: Path, branch: str) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{branch} acceptance report cannot be read: {error}") from error
    if report.get("candidate_passes") is not True:
        raise ValueError(f"{branch} acceptance report does not pass")
    return report


def compose(
    base: dict[str, Any],
    overlays: list[tuple[str, dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_state = base["state_dict"]
    composed_state = {name: tensor.clone() for name, tensor in base_state.items()}
    overlay_reports = []
    for branch, overlay in overlays:
        if branch not in OVERLAY_PREFIXES:
            raise ValueError(f"unsupported overlay branch: {branch}")
        if overlay.get("adaptation_policy") != base.get("adaptation_policy"):
            raise ValueError(f"{branch} overlay uses a different adaptation policy")
        overlay_state = overlay["state_dict"]
        if set(overlay_state) != set(base_state):
            raise ValueError(f"{branch} overlay tensor keys differ from the base")
        prefixes = OVERLAY_PREFIXES[branch]
        selected = sorted(name for name in overlay_state if name.startswith(prefixes))
        if not selected:
            raise ValueError(f"{branch} overlay contains no matching tensors")
        changed = []
        for name in selected:
            source = overlay_state[name]
            target = composed_state[name]
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(f"{branch} overlay tensor shape or dtype differs: {name}")
            if not torch.equal(source, target):
                changed.append(name)
            composed_state[name] = source.clone()
        if not changed:
            raise ValueError(f"{branch} overlay changes no tensors")
        overlay_reports.append(
            {
                "branch": branch,
                "prefixes": list(prefixes),
                "selected_tensor_count": len(selected),
                "changed_tensor_count": len(changed),
                "changed_tensors": changed,
            }
        )
    return (
        {
            "format": "attune_delta_v1",
            "adaptation_policy": base["adaptation_policy"],
            "state_dict": composed_state,
        },
        overlay_reports,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--event", type=Path)
    parser.add_argument("--event-acceptance", type=Path)
    parser.add_argument("--style", type=Path)
    parser.add_argument("--style-acceptance", type=Path)
    parser.add_argument("--affect", type=Path)
    parser.add_argument("--affect-acceptance", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    arguments = parser.parse_args()
    overlay_specs = [
        (branch, checkpoint_path, acceptance_path)
        for branch, checkpoint_path, acceptance_path in (
            ("event", arguments.event, arguments.event_acceptance),
            ("style", arguments.style, arguments.style_acceptance),
            ("affect", arguments.affect, arguments.affect_acceptance),
        )
        if checkpoint_path is not None or acceptance_path is not None
    ]
    if not overlay_specs:
        parser.error("at least one overlay checkpoint is required")
    for branch, checkpoint_path, acceptance_path in overlay_specs:
        if checkpoint_path is None or acceptance_path is None:
            parser.error(f"{branch} overlay requires both checkpoint and acceptance report")

    base = load_delta(arguments.base)
    loaded_overlays = [
        (branch, load_delta(checkpoint_path))
        for branch, checkpoint_path, _acceptance_path in overlay_specs
    ]
    for branch, _checkpoint_path, acceptance_path in overlay_specs:
        load_acceptance(acceptance_path, branch)
    checkpoint, overlay_reports = compose(base, loaded_overlays)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(arguments.output)
    provenance = {
        "schema_version": "1.0",
        "base": {"path": str(arguments.base), "sha256": digest(arguments.base)},
        "overlays": [
            {
                **report,
                "path": str(path),
                "sha256": digest(path),
                "acceptance": {
                    "path": str(acceptance_path),
                    "sha256": digest(acceptance_path),
                },
            }
            for report, (_branch, path, acceptance_path) in zip(
                overlay_reports, overlay_specs, strict=True
            )
        ],
        "output": {"path": str(arguments.output), "sha256": digest(arguments.output)},
    }
    arguments.provenance.parent.mkdir(parents=True, exist_ok=True)
    arguments.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance["output"]))


if __name__ == "__main__":
    main()
