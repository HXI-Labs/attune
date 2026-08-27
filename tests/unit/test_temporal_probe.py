from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from attune.models.temporal_probe import (  # noqa: E402
    FrozenTemporalProbeHead,
    _decode_annotations,
)


def checkpoint_payload(*, gate_passed: bool) -> dict:
    head = torch.nn.Sequential(
        torch.nn.Linear(512, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 3),
    )
    return {
        "head_state_dict": head.state_dict(),
        "feature_mean": torch.zeros(512),
        "feature_scale": torch.ones(512),
        "labels": ("laugh", "cough", "throat_clear"),
        "hidden_size": 128,
        "threshold": 0.5,
        "dataset": "dcase2016_task2",
        "embedding": "sensevoice-small-encoder-frames-v1",
        "encoder_frozen": True,
        "gate": {
            "passed": gate_passed,
            "margin_required": 0.05,
            "margin_observed": 0.41 if gate_passed else -0.1,
        },
    }


def test_temporal_checkpoint_is_rejected_when_held_out_gate_failed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "head.pt"
    torch.save(checkpoint_payload(gate_passed=False), checkpoint)
    head = FrozenTemporalProbeHead(
        checkpoint=checkpoint,
        sensevoice_checkpoint=tmp_path,
        frame_cache=tmp_path / "cache",
    )

    with pytest.raises(RuntimeError, match="did not pass"):
        head._load()


def test_starss23_checkpoint_requires_collar_and_segment_gate(tmp_path: Path) -> None:
    payload = checkpoint_payload(gate_passed=True)
    payload["dataset"] = "starss23"
    payload["labels"] = ("laugh",)
    payload["hidden_size"] = 64
    payload["gate"] = {
        "passed": True,
        "margin_required": 0.05,
        "margin_observed": 0.25,
    }
    checkpoint = tmp_path / "starss23-head.pt"
    torch.save(payload, checkpoint)
    head = FrozenTemporalProbeHead(
        checkpoint=checkpoint,
        sensevoice_checkpoint=tmp_path,
        frame_cache=tmp_path / "cache",
    )

    with pytest.raises(RuntimeError, match="boundary-alignment"):
        head._load()


def test_frame_probabilities_decode_multiple_bounded_event_spans() -> None:
    probabilities = torch.tensor(
        [
            [0.9, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.7, 0.1, 0.1],
        ]
    )

    annotations = _decode_annotations(
        probabilities,
        labels=("laugh", "cough", "throat_clear"),
        threshold=0.5,
        first_frame_center_ms=30.0,
        frame_hop_ms=60.0,
        duration_ms=240,
    )

    assert [
        (annotation.label.value, annotation.start_ms, annotation.end_ms)
        for annotation in annotations
    ] == [("laugh", 0, 120), ("laugh", 180, 240)]


def test_hysteresis_decoder_applies_median_filter_and_min_duration() -> None:
    probabilities = torch.tensor(
        [
            [0.1],
            [0.1],
            [0.9],
            [0.1],
            [0.9],
            [0.9],
            [0.9],
            [0.1],
            [0.1],
        ]
    )
    decoder = {
        "type": "hysteresis",
        "high_threshold": 0.8,
        "low_threshold": 0.5,
        "max_gap_frames": 2,
        "min_active_frames": 3,
        "median_filter_frames": 3,
    }
    annotations = _decode_annotations(
        probabilities,
        labels=("laugh",),
        threshold=0.5,
        decoder=decoder,
        first_frame_center_ms=30.0,
        frame_hop_ms=60.0,
        duration_ms=540,
    )
    assert [(annotation.start_ms, annotation.end_ms) for annotation in annotations] == [(180, 420)]


def test_bigru_factory_is_not_conv1d() -> None:
    from attune.models.temporal_probe import BIGRU_ARCHITECTURE, build_bigru_head

    head = build_bigru_head(torch)
    frames = torch.randn(6, 512)
    logits = head(frames)
    assert head.architecture == BIGRU_ARCHITECTURE
    assert logits.shape == (6, 1)
    assert not any(isinstance(module, torch.nn.Conv1d) for module in head.modules())
    restored = build_bigru_head(torch)
    restored.load_state_dict(head.state_dict())
    restored.eval()
    with torch.inference_mode():
        assert torch.allclose(restored(frames), logits)
