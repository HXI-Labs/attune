from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from attune.models.temporal_probe import FrozenTemporalProbeHead


def load_script() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "train_starss23_scene_raster.py"
    spec = importlib.util.spec_from_file_location("train_starss23_scene_raster", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wiring_gate_requires_collar_and_segment_margin() -> None:
    script = load_script()

    assert (
        script.should_wire_starss23_timestamps(
            segment_f1=0.4794,
            whole_clip_segment_f1=0.1721,
            collar_f1=0.1074,
        )
        is False
    )
    assert (
        script.should_wire_starss23_timestamps(
            segment_f1=0.4794,
            whole_clip_segment_f1=0.1721,
            collar_f1=0.25,
        )
        is True
    )
    assert (
        script.should_wire_starss23_timestamps(
            segment_f1=0.20,
            whole_clip_segment_f1=0.1721,
            collar_f1=0.40,
        )
        is False
    )


def test_starss23_loader_rejects_weak_collar_even_if_checkpoint_claims_pass(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    head = torch.nn.Sequential(
        torch.nn.Linear(512, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )
    payload = {
        "head_state_dict": head.state_dict(),
        "feature_mean": torch.zeros(512),
        "feature_scale": torch.ones(512),
        "labels": ("laugh",),
        "hidden_size": 64,
        "threshold": 0.95,
        "dataset": "starss23",
        "embedding": "sensevoice-small-encoder-frames-v1",
        "encoder_frozen": True,
        "gate": {
            "passed": True,
            "segment_margin_required": 0.05,
            "segment_margin_observed": 0.3073,
            "collar_f1_required": 0.25,
            "collar_f1_observed": 0.1074,
        },
    }
    checkpoint = tmp_path / "starss23-scene-head.pt"
    torch.save(payload, checkpoint)
    probe = FrozenTemporalProbeHead(
        checkpoint=checkpoint,
        sensevoice_checkpoint=tmp_path,
        frame_cache=tmp_path / "cache",
    )

    with pytest.raises(RuntimeError, match="boundary-alignment"):
        probe._load()
