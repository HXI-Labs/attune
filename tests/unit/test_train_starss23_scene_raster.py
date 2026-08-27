from __future__ import annotations

import importlib.util
import inspect
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


def test_positive_class_weight_uses_train_frames_only() -> None:
    torch = pytest.importorskip("torch")
    script = load_script()
    train = torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0]])
    inspection = torch.ones((200, 1))
    weight = script.positive_class_weight_from_train_frames(train)
    assert float(weight.reshape(())) == pytest.approx(4.0)
    leaked = script.positive_class_weight_from_train_frames(inspection)
    assert float(leaked.reshape(())) != pytest.approx(float(weight.reshape(())))
    with pytest.raises(TypeError):
        script.positive_class_weight_from_train_frames(train, inspection)


def test_min_active_frames_come_from_train_gold_not_inspection() -> None:
    script = load_script()
    train_rows = [
        {"events": [{"start_ms": 0, "end_ms": duration}]} for duration in (600, 600, 900, 900, 1200)
    ]
    inspection_rows = [{"events": [{"start_ms": 0, "end_ms": 100}]}]
    train_frames = script.min_active_frames_from_gold(
        script.gold_event_durations_ms(train_rows),
        frame_hop_ms=60.0,
        percentiles=(10, 25, 50),
    )
    leaked_frames = script.min_active_frames_from_gold(
        script.gold_event_durations_ms(train_rows + inspection_rows),
        frame_hop_ms=60.0,
        percentiles=(10, 25, 50),
    )
    assert 1 in train_frames and 2 in train_frames and 3 in train_frames
    assert leaked_frames != train_frames
    assert 10 in train_frames
    assert 10 not in leaked_frames


def test_duration_error_table_flags_short_false_positives() -> None:
    script = load_script()
    references = [[{"label": "laugh", "start_ms": 1000, "end_ms": 2200}]]
    predictions = [
        [
            {"label": "laugh", "start_ms": 1000, "end_ms": 2200},
            {"label": "laugh", "start_ms": 4000, "end_ms": 4120},
            {"label": "laugh", "start_ms": 5000, "end_ms": 5060},
            {"label": "laugh", "start_ms": 6000, "end_ms": 6180},
        ]
    ]
    table = script.duration_error_table(references, predictions)
    assert table["true_positive"]["count"] == 1
    assert table["false_positive"]["count"] == 3
    assert table["false_positive"]["median_ms"] == 120
    assert table["false_positives_are_short_fragments"] is True
    assert table["false_positive"]["shorter_than_gold_p25"] == 3


def test_wiring_gate_constants_are_not_lowered() -> None:
    script = load_script()
    assert script.COLLAR_F1_REQUIRED == 0.25
    assert script.SEGMENT_MARGIN_REQUIRED == 0.05
    assert (
        script.should_wire_starss23_timestamps(
            segment_f1=0.40,
            whole_clip_segment_f1=0.17,
            collar_f1=0.249,
        )
        is False
    )


def test_decoder_span_kwargs_drops_search_metadata() -> None:
    script = load_script()
    decoder = {
        "high_threshold": 0.95,
        "low_threshold": 0.855,
        "max_gap_frames": 8,
        "min_active_frames": 5,
        "median_filter_frames": 3,
        "min_duration_ms": 300,
        "type": "hysteresis",
    }
    kwargs = script.decoder_span_kwargs(decoder)
    assert kwargs == {
        "high_threshold": 0.95,
        "low_threshold": 0.855,
        "max_gap_frames": 8,
        "min_active_frames": 5,
        "median_filter_frames": 3,
    }
    assert "min_duration_ms" not in kwargs
    assert "type" not in kwargs


def test_hysteresis_spans_rejects_min_duration_ms_kwarg() -> None:
    torch = pytest.importorskip("torch")
    script = load_script()
    probabilities = [torch.tensor([[0.9], [0.9], [0.1]])]
    with pytest.raises(TypeError):
        script.hysteresis_spans(
            probabilities,
            high_threshold=0.5,
            low_threshold=0.4,
            max_gap_frames=0,
            min_active_frames=1,
            first_frame_center_ms=30.0,
            frame_hop_ms=60.0,
            min_duration_ms=180,
        )


def test_main_keeps_inspection_out_of_pos_weight_and_decoder_search() -> None:
    script = load_script()
    source = inspect.getsource(script.main)
    assert "positive_class_weight_from_train_frames(train_targets)" in source
    assert "gold_event_durations_ms(train_rows)" in source
    assert "select_hysteresis_decoder(" in source
    assert "validation_probabilities" in source
    assert "references_validation" in source
    assert "**decoder_span_kwargs(decoder)" in source
    assert "frame_targets(inspection" not in source
    assert "gold_event_durations_ms(inspection" not in source
    assert "positive_class_weight_from_train_frames(inspection" not in source
    assert "select_hysteresis_decoder(\n        inspection" not in source
    assert "hysteresis_spans(\n        inspection_probabilities,\n        **decoder," not in source
    assert script.COLLAR_F1_REQUIRED == 0.25
    assert script.SEGMENT_MARGIN_REQUIRED == 0.05
    assert script.PRIOR_40_EPOCH_PASS["collar_true_positive"] == 8
    assert script.PRIOR_40_EPOCH_PASS["collar_false_positive"] == 93
    assert script.PRIOR_40_EPOCH_PASS["collar_false_negative"] == 40


def test_min_active_is_capped_below_train_gold_p50() -> None:
    script = load_script()
    durations_ms = [500] * 13 + [900] * 26 + [1800] * 12
    frames = script.min_active_frames_from_gold(durations_ms, frame_hop_ms=60.0)
    p25_frames = max(1, round(script.duration_percentile_ms(durations_ms, 25) / 60.0))
    p50_frames = max(1, round(script.duration_percentile_ms(durations_ms, 50) / 60.0))
    assert p50_frames > p25_frames
    assert p50_frames == 15
    assert p25_frames == 8
    assert max(frames) <= p25_frames
    assert p50_frames not in frames
    assert 1 in frames and 2 in frames and 3 in frames
    assert 5 not in script.DECODER_MEDIAN_WINDOWS
    assert {2, 4, 8}.issubset(script.DECODER_GAP_FRAMES)
    assert script.DECODER_GAP_FRAMES != (0,)
    assert 50 not in script.GOLD_DURATION_PERCENTILES


def test_decoder_selection_prefers_recall_over_fewer_false_positives() -> None:
    script = load_script()
    high_fp_high_recall = {
        "f1": 0.5,
        "true_positive": 2,
        "false_positive": 2,
        "false_negative": 2,
    }
    low_fp_low_recall = {
        "f1": 0.5,
        "true_positive": 2,
        "false_positive": 0,
        "false_negative": 4,
    }
    segment = {"f1": 0.4}
    assert script.decoder_selection_key(
        high_fp_high_recall, segment
    ) > script.decoder_selection_key(
        low_fp_low_recall,
        segment,
    )
    source = inspect.getsource(script.select_hysteresis_decoder)
    assert "decoder_selection_key" in source
    assert '-int(collar["false_positive"])' not in source
    assert script.choose_checkpoint_from_ablation(0.12, 0.12) == script.CHECKPOINT_UNWEIGHTED
    assert script.choose_checkpoint_from_ablation(0.12, 0.13) == script.CHECKPOINT_POSWEIGHT
    assert script.choose_checkpoint_from_ablation(0.14, 0.13) == script.CHECKPOINT_UNWEIGHTED
