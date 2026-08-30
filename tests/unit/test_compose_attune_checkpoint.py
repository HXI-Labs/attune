from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

SCRIPT_PATH = Path(__file__).parents[2] / "scripts/compose_attune_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("compose_attune_checkpoint", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
COMPOSE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPOSE)


def checkpoint(value: float = 0.0, *, policy: str = "upper_two"):
    return {
        "format": "attune_delta_v1",
        "adaptation_policy": policy,
        "state_dict": {
            "sensevoice.encoder.block.weight": torch.tensor([value]),
            "event_head.weight": torch.tensor([value]),
            "style_projection.0.weight": torch.tensor([value]),
            "style_head.weight": torch.tensor([value]),
            "affect_projection.0.weight": torch.tensor([value]),
            "affect_head.weight": torch.tensor([value]),
        },
    }


def test_composition_changes_only_selected_branch_tensors() -> None:
    base = checkpoint()
    event = checkpoint(1.0)
    affect = checkpoint(2.0)

    composed, reports = COMPOSE.compose(base, [("event", event), ("affect", affect)])

    state = composed["state_dict"]
    assert state["event_head.weight"].item() == 1.0
    assert state["affect_projection.0.weight"].item() == 2.0
    assert state["affect_head.weight"].item() == 2.0
    assert state["style_head.weight"].item() == 0.0
    assert state["sensevoice.encoder.block.weight"].item() == 0.0
    assert [report["branch"] for report in reports] == ["event", "affect"]


def test_composition_rejects_policy_mismatch() -> None:
    with pytest.raises(ValueError, match="different adaptation policy"):
        COMPOSE.compose(checkpoint(), [("style", checkpoint(1.0, policy="frozen"))])


def test_composition_rejects_overlay_with_different_keys() -> None:
    overlay = checkpoint(1.0)
    del overlay["state_dict"]["style_head.weight"]

    with pytest.raises(ValueError, match="tensor keys differ"):
        COMPOSE.compose(checkpoint(), [("style", overlay)])


def test_composition_rejects_unchanged_branch() -> None:
    with pytest.raises(ValueError, match="changes no tensors"):
        COMPOSE.compose(checkpoint(), [("affect", checkpoint())])


def test_acceptance_report_must_explicitly_pass(tmp_path: Path) -> None:
    report = tmp_path / "acceptance.json"
    report.write_text(json.dumps({"candidate_passes": False}))

    with pytest.raises(ValueError, match="does not pass"):
        COMPOSE.load_acceptance(report, "event")


def test_acceptance_report_returns_passing_evidence(tmp_path: Path) -> None:
    report = tmp_path / "acceptance.json"
    report.write_text(json.dumps({"candidate_passes": True, "candidate": "style-v1"}))

    loaded = COMPOSE.load_acceptance(report, "style")

    assert loaded["candidate"] == "style-v1"
