#!/usr/bin/env python3
"""Compose two affect branches behind one shared Attune speech encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

AFFECT_PREFIXES = ("affect_projection.", "affect_head.")


def _digest(path: Path) -> str:
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
    return checkpoint


def compose(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    *,
    alpha: float,
) -> dict[str, Any]:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if primary.get("adaptation_policy") != secondary.get("adaptation_policy"):
        raise ValueError("affect checkpoints use different adaptation policies")
    primary_state = primary["state_dict"]
    secondary_state = secondary["state_dict"]
    if set(primary_state) != set(secondary_state):
        raise ValueError("affect checkpoint tensor keys differ")

    state = {name: tensor.clone() for name, tensor in primary_state.items()}
    selected = [name for name in state if name.startswith(AFFECT_PREFIXES)]
    if not selected:
        raise ValueError("secondary checkpoint contains no affect tensors")
    for name in selected:
        destination = name.replace(
            "affect_projection.",
            "affect_projection_secondary.",
            1,
        ).replace("affect_head.", "affect_head_secondary.", 1)
        source = secondary_state[name]
        if source.shape != state[name].shape or source.dtype != state[name].dtype:
            raise ValueError(f"affect tensor shape or dtype differs: {name}")
        state[destination] = source.clone()
    return {
        "format": "attune_delta_v1",
        "adaptation_policy": primary["adaptation_policy"],
        "affect_ensemble": {"alpha": alpha},
        "state_dict": state,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    arguments = parser.parse_args()

    acceptance = json.loads(arguments.acceptance.read_text())
    if acceptance.get("candidate_passes") is not True:
        raise ValueError("affect interpolation acceptance report does not pass")
    alpha = acceptance.get("selected_alpha")
    if not isinstance(alpha, float):
        raise ValueError("affect interpolation report has no selected alpha")
    checkpoint = compose(
        load_delta(arguments.primary),
        load_delta(arguments.secondary),
        alpha=alpha,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(arguments.output)
    provenance = {
        "schema_version": "1.0",
        "method": "shared_encoder_dual_affect_branch_logit_interpolation",
        "alpha": alpha,
        "primary": {"path": str(arguments.primary), "sha256": _digest(arguments.primary)},
        "secondary": {
            "path": str(arguments.secondary),
            "sha256": _digest(arguments.secondary),
        },
        "acceptance": {
            "path": str(arguments.acceptance),
            "sha256": _digest(arguments.acceptance),
        },
        "output": {"path": str(arguments.output), "sha256": _digest(arguments.output)},
    }
    arguments.provenance.parent.mkdir(parents=True, exist_ok=True)
    arguments.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance["output"]))


if __name__ == "__main__":
    main()
