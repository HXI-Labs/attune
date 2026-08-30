from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


def _module():
    path = Path("scripts/audit_checkpoint_scope.py")
    spec = importlib.util.spec_from_file_location("audit_checkpoint_scope", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint(path: Path, *, affect: float, event: float = 0.0) -> None:
    torch.save(
        {
            "format": "attune_delta_v1",
            "adaptation_policy": "upper_two",
            "state_dict": {
                "affect_head.weight": torch.tensor([affect]),
                "event_head.weight": torch.tensor([event]),
                "style_head.weight": torch.tensor([0.0]),
            },
        },
        path,
    )


def test_audit_accepts_only_allowed_changes(tmp_path: Path) -> None:
    module = _module()
    initial = tmp_path / "initial.pt"
    candidate = tmp_path / "candidate.pt"
    _checkpoint(initial, affect=0.0)
    _checkpoint(candidate, affect=1.0)
    report = module.audit_scope(initial, candidate, allowed_prefixes=("affect_head.",))
    assert report["passed"] is True
    assert report["changed_tensor_count"] == 1
    assert report["changed_tensors"][0]["name"] == "affect_head.weight"
    assert report["scope_violations"] == []


def test_audit_rejects_unexpected_change(tmp_path: Path) -> None:
    module = _module()
    initial = tmp_path / "initial.pt"
    candidate = tmp_path / "candidate.pt"
    _checkpoint(initial, affect=0.0)
    _checkpoint(candidate, affect=1.0, event=1.0)
    report = module.audit_scope(initial, candidate, allowed_prefixes=("affect_head.",))
    assert report["passed"] is False
    assert report["scope_violations"] == ["event_head.weight"]


def test_audit_requires_matching_keys(tmp_path: Path) -> None:
    module = _module()
    initial = tmp_path / "initial.pt"
    candidate = tmp_path / "candidate.pt"
    _checkpoint(initial, affect=0.0)
    _checkpoint(candidate, affect=1.0)
    checkpoint = torch.load(candidate, weights_only=True)
    del checkpoint["state_dict"]["style_head.weight"]
    torch.save(checkpoint, candidate)
    with pytest.raises(ValueError, match="tensor keys differ"):
        module.audit_scope(initial, candidate, allowed_prefixes=("affect_head.",))
