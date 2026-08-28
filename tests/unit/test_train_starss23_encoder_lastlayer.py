from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from types import ModuleType

import pytest

from attune.models.temporal_probe import (
    dcase_checkpoint_acceptance,
    starss23_checkpoint_acceptance,
)


def load_script() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "train_starss23_encoder_lastlayer.py"
    spec = importlib.util.spec_from_file_location("train_starss23_encoder_lastlayer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clip(
    *,
    clip_id: str,
    room: str,
    recording: str,
    window_start_ms: int = 0,
    events: list[dict] | None = None,
) -> dict:
    return {
        "clip_id": clip_id,
        "room": room,
        "source_recording": recording,
        "source_window_start_ms": window_start_ms,
        "events": events or [],
    }


def test_last_block_trainable_params_are_only_tp_encoders_19(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    from tests.unit.test_sensevoice_last_block import last_block_encoder_twenty

    encoder = last_block_encoder_twenty(tmp_path, monkeypatch)
    names = encoder.trainable_parameter_names()
    assert encoder.last_block_name == "encoder.tp_encoders.19"
    assert encoder.unfrozen_module_names() == ("encoder.tp_encoders.19",)
    assert names
    assert all(name.startswith("encoder.tp_encoders.19.") for name in names)
    frozen = [
        name for name, parameter in encoder.model.named_parameters() if not parameter.requires_grad
    ]
    assert any(name.startswith("encoder.tp_encoders.18.") for name in frozen)
    assert any(name.startswith("encoder.encoders.") for name in frozen)
    assert any(name.startswith("encoder.tp_norm.") for name in frozen)


def test_inspection_manifest_clips_are_not_in_the_train_list() -> None:
    script = load_script()
    root = Path(__file__).parents[2]
    development = script.load_manifest_rows(
        root / "data/manifests/starss23-scene-raster-development.jsonl"
    )
    inspection = script.load_manifest_rows(
        root / "data/manifests/starss23-scene-raster-inspection.jsonl"
    )
    train, validation, inspection_first = script.first_60s_disjoint_split(development, inspection)
    train_ids = {row["clip_id"] for row in train}
    inspection_ids = {row["clip_id"] for row in inspection_first}
    assert train_ids.isdisjoint(inspection_ids)
    assert {row["source_recording"] for row in train}.isdisjoint(
        {row["source_recording"] for row in inspection_first}
    )
    assert {row["room"] for row in train}.isdisjoint({row["room"] for row in inspection_first})
    assert len(train) == script.FIRST_60S_TRAIN_CLIPS == 43
    assert len(validation) == script.FIRST_60S_VAL_CLIPS == 19
    assert len(inspection_first) == 49
    assert script.starss23.event_count(inspection_first) == 48
    assert all(int(row["source_window_start_ms"]) == 0 for row in train)
    assert all(int(row["source_window_start_ms"]) == 0 for row in validation)
    gold = script.load_manifest_rows(root / "data/manifests/starss23-gold-review-pack.jsonl")
    assert {row["clip_id"] for row in gold} == inspection_ids
    jerry = json.loads(
        (root / "research/error-analysis/starss23-jerry-firstpass-ledger.json").read_text(
            encoding="utf-8"
        )
    )
    jerry_ids = {row["clip_id"] for row in jerry["clips"]}
    assert jerry_ids.isdisjoint(train_ids)
    assert jerry_ids <= inspection_ids
    assert len(train) > 6


def test_split_rejects_inspection_clip_leak() -> None:
    script = load_script()
    inspection = [
        _clip(clip_id="ins-0", room="sony-room23", recording="ins.wav", events=[{"label": "laugh"}])
    ]
    train = [
        _clip(clip_id="ins-0", room="tau-room4", recording="train.wav"),
        *[_clip(clip_id=f"t{i}", room="tau-room4", recording=f"t{i}.wav") for i in range(42)],
    ]
    with pytest.raises(RuntimeError, match="inspection clips leaked into train"):
        script.assert_inspection_clips_not_in_train(train, inspection)


def test_gate_and_decoder_are_not_lowered() -> None:
    script = load_script()
    assert script.COLLAR_F1_REQUIRED == 0.25
    assert script.SEGMENT_MARGIN_REQUIRED == 0.05
    assert script.PREDECLARED_DECODER == {
        "high_threshold": 0.95,
        "low_threshold": 0.855,
        "max_gap_frames": 0,
        "min_active_frames": 1,
        "median_filter_frames": 3,
        "onset_shift_ms": 0,
    }
    assert (
        script.should_wire(segment_f1_value=0.5124, whole_clip_segment_f1=0.1721, collar_f1=0.1395)
        is False
    )
    assert (
        script.should_wire(segment_f1_value=0.40, whole_clip_segment_f1=0.17, collar_f1=0.249)
        is False
    )
    assert (
        script.should_wire(segment_f1_value=0.20, whole_clip_segment_f1=0.1721, collar_f1=0.40)
        is False
    )
    assert (
        script.should_wire(segment_f1_value=0.40, whole_clip_segment_f1=0.17, collar_f1=0.25)
        is True
    )
    source = inspect.getsource(script.main)
    assert "select_hysteresis_decoder(" not in source
    assert "iter_hysteresis_decoder_candidates" not in source
    assert "BCEWithLogitsLoss(pos_weight" not in source
    assert "LastBlockSenseVoiceFrameEncoder" in source
    assert "encoder.tp_encoders.19" in source
    assert "require_protocol" in source
    assert "first_60s_disjoint_split" in source


def test_protocol_is_locked_before_train() -> None:
    script = load_script()
    protocol = Path(__file__).parents[2] / "research/starss23-lastblock-highlights.md"
    script.require_protocol(protocol)
    text = protocol.read_text(encoding="utf-8")
    assert "Not second-listener gold" in text
    assert "tp_encoders.19" in text
    assert "No tiling-in-train" in text


def test_dcase_checkpoint_acceptance_still_rejects_starss23() -> None:
    torch = pytest.importorskip("torch")
    payload = {
        "head_state_dict": torch.nn.Linear(512, 1).state_dict(),
        "feature_mean": torch.zeros(512),
        "feature_scale": torch.ones(512),
        "labels": ("laugh",),
        "threshold": 0.95,
        "dataset": "starss23",
        "embedding": "sensevoice-small-encoder-lastblock-frames-v1",
        "encoder_frozen": False,
        "gate": {
            "passed": True,
            "collar_f1_required": 0.25,
            "collar_f1_observed": 0.40,
            "segment_margin_required": 0.05,
            "segment_margin_observed": 0.20,
        },
    }
    accepted, reason = dcase_checkpoint_acceptance(payload)
    assert accepted is False
    assert reason is not None and "STARSS23" in reason


def test_starss23_checkpoint_acceptance_requires_collar_0_25() -> None:
    torch = pytest.importorskip("torch")
    payload = {
        "labels": ("laugh",),
        "dataset": "starss23",
        "gate": {
            "passed": True,
            "collar_f1_required": 0.25,
            "collar_f1_observed": 0.1395,
            "segment_margin_required": 0.05,
            "segment_margin_observed": 0.34,
        },
    }
    accepted, reason = starss23_checkpoint_acceptance(payload)
    assert accepted is False
    assert reason is not None and "0.25" in reason
    payload["gate"]["collar_f1_observed"] = 0.25
    payload["gate"]["segment_margin_observed"] = 0.04
    accepted, reason = starss23_checkpoint_acceptance(payload)
    assert accepted is False
    payload["gate"]["segment_margin_observed"] = 0.05
    accepted, reason = starss23_checkpoint_acceptance(payload)
    assert accepted is True
    assert reason is None
    payload["gate"]["collar_f1_required"] = 0.20
    accepted, reason = starss23_checkpoint_acceptance(payload)
    assert accepted is False
    payload["dataset"] = "dcase2016_task2"
    payload["gate"]["collar_f1_required"] = 0.25
    accepted, reason = starss23_checkpoint_acceptance(payload)
    assert accepted is False
    del torch
