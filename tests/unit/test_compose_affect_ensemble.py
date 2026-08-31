from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT_PATH = Path(__file__).parents[2] / "scripts/compose_affect_ensemble.py"
SPEC = importlib.util.spec_from_file_location("compose_affect_ensemble", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
COMPOSE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPOSE)


def checkpoint(value: float, *, policy: str = "upper_two") -> dict:
    return {
        "format": "attune_delta_v1",
        "adaptation_policy": policy,
        "state_dict": {
            "event_head.weight": torch.tensor([0.0]),
            "affect_projection.0.weight": torch.tensor([value]),
            "affect_head.weight": torch.tensor([value]),
        },
    }


def test_compose_adds_only_a_secondary_affect_branch() -> None:
    composed = COMPOSE.compose(checkpoint(1.0), checkpoint(2.0), alpha=0.5)

    assert composed["affect_ensemble"] == {"alpha": 0.5}
    assert composed["state_dict"]["affect_projection.0.weight"].item() == 1.0
    assert composed["state_dict"]["affect_projection_secondary.0.weight"].item() == 2.0
    assert composed["state_dict"]["affect_head_secondary.weight"].item() == 2.0
    assert "event_head_secondary.weight" not in composed["state_dict"]


def test_compose_rejects_incompatible_checkpoints() -> None:
    with pytest.raises(ValueError, match="different adaptation policies"):
        COMPOSE.compose(checkpoint(1.0), checkpoint(2.0, policy="frozen"), alpha=0.5)
