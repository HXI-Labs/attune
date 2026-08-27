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


def test_main_trains_mlp_with_unweighted_bce_and_locked_decoder() -> None:
    script = load_script()
    source = inspect.getsource(script.main)
    assert "build_mlp_head" in source
    assert "build_bigru_head" not in source
    assert "AdamW(head.parameters()" in source
    assert "Conv1d" not in source
    assert "boundary_weights_from_train_clips" not in source
    assert "gold_event_durations_ms(train_rows)" in source
    assert "PREDECLARED_DECODER" in source
    assert "select_hysteresis_decoder(" not in source
    assert "iter_hysteresis_decoder_candidates" not in source
    assert "BCEWithLogitsLoss(pos_weight" not in source
    assert "requires_grad_(True)" not in source
    assert "bce_with_logits" in source
    assert "decode_spans(" in source
    assert "first_60s_subset" in source or "first60_index" in source
    assert "frame_targets(inspection" not in source
    assert "gold_event_durations_ms(inspection" not in source
    assert 'default=Path("artifacts/starss23-scene-raster/frame-head-tiled.pt")' in source
    assert 'default=Path("artifacts/starss23-scene-raster/frame-head-40epoch.pt")' in source
    assert 'default=Path("artifacts/starss23-scene-raster/embeddings-tiled")' in source
    assert 'default=Path("data/raw/starss23-scene-raster-tiled")' in source
    assert script.COLLAR_F1_REQUIRED == 0.25
    assert script.SEGMENT_MARGIN_REQUIRED == 0.05
    assert script.PROTOCOL == "tiled_60s_mean4_mic"
    assert {"sony-room21", "tau-room6"} == script.VALIDATION_ROOMS
    assert "choose_max_rms_channel" not in source
    assert "chosen_channel" not in source
    assert "train_only_events_for_frame_targets" not in source
    assert 'default=Path("data/raw/starss23-scene-raster-v2")' not in source
    assert script.HEADLINE.startswith("mean-4 tiled MLP negative")
    assert "do not replace reported best 0.1395" in script.HEADLINE
    assert script.AUDIO_CACHE_SCORED == "data/raw/starss23-scene-raster-tiled"
    assert script.AUDIO_CACHE_V2_MAX_RMS == "data/raw/starss23-scene-raster-v2"
    assert script.AUDIO_CACHES["v2_max_rms_audio_only"]["downmix"] == "max_rms_channel"


def test_predeclared_decoder_is_locked_in_main() -> None:
    script = load_script()
    assert script.PREDECLARED_DECODER == {
        "high_threshold": 0.95,
        "low_threshold": 0.855,
        "max_gap_frames": 0,
        "min_active_frames": 1,
        "median_filter_frames": 3,
        "onset_shift_ms": 0,
    }
    source = inspect.getsource(script.main)
    assert "PREDECLARED_DECODER" in source
    assert "select_hysteresis_decoder(" not in source
    assert script.COLLAR_F1_REQUIRED == 0.25
    assert script.SEGMENT_MARGIN_REQUIRED == 0.05


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


def test_decoder_span_kwargs_drops_search_metadata() -> None:
    script = load_script()
    decoder = {
        "high_threshold": 0.95,
        "low_threshold": 0.855,
        "max_gap_frames": 8,
        "min_active_frames": 5,
        "median_filter_frames": 1,
        "min_duration_ms": 300,
        "onset_shift_ms": -120,
        "type": "hysteresis",
    }
    kwargs = script.decoder_span_kwargs(decoder)
    assert kwargs == {
        "high_threshold": 0.95,
        "low_threshold": 0.855,
        "max_gap_frames": 8,
        "min_active_frames": 5,
        "median_filter_frames": 1,
    }
    assert "min_duration_ms" not in kwargs
    assert "type" not in kwargs
    assert "onset_shift_ms" not in kwargs


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
    search_source = inspect.getsource(script.iter_hysteresis_decoder_candidates)
    assert "decoder_selection_key" in search_source
    assert "iter_hysteresis_decoder_candidates" in source
    assert '-int(collar["false_positive"])' not in source
    assert '-int(collar["false_positive"])' not in search_source


def test_mlp_head_is_512_64_1_not_conv_or_gru() -> None:
    torch = pytest.importorskip("torch")
    script = load_script()
    head = script.build_mlp_head(torch)
    script.assert_head_is_mlp_not_conv_or_gru(head, torch)
    assert not any(isinstance(module, torch.nn.Conv1d) for module in head.modules())
    assert not any(isinstance(module, torch.nn.GRU) for module in head.modules())
    trainable = sum(parameter.numel() for parameter in head.parameters())
    assert trainable == 32_897
    conv = torch.nn.Conv1d(512, 128, kernel_size=7, padding=3)
    with pytest.raises(RuntimeError, match="must not contain Conv1d"):
        script.assert_head_is_mlp_not_conv_or_gru(conv, torch)


def test_val_rooms_locked_and_whole_file_stays_in_one_split() -> None:
    script = load_script()
    assert {"sony-room21", "tau-room6"} == script.VALIDATION_ROOMS
    source = inspect.getsource(script.main)
    assert "assert_whole_files_stay_in_one_split" in source
    train = [{"source_recording": "a.wav", "room": "sony-room1"}]
    validation = [{"source_recording": "b.wav", "room": "sony-room21"}]
    script.assert_whole_files_stay_in_one_split(train, validation)
    with pytest.raises(RuntimeError, match="validation files overlap training"):
        script.assert_whole_files_stay_in_one_split(
            [{"source_recording": "a.wav"}],
            [{"source_recording": "a.wav"}],
        )


def test_train_val_inspection_rooms_are_disjoint_in_main() -> None:
    script = load_script()
    source = inspect.getsource(script.main)
    assert "inspection rooms overlap development" in source
    assert "validation rooms drifted" in source
    assert "VALIDATION_ROOMS" in source


def test_first_60s_subset_keeps_window_zero_only() -> None:
    script = load_script()
    rows = [
        {"source_window_start_ms": 0, "events": [1]},
        {"source_window_start_ms": 60_000, "events": [2]},
        {"source_window_start_ms": 0, "events": [3]},
    ]
    subset = script.first_60s_subset(rows)
    assert [row["events"][0] for row in subset] == [1, 3]
    assert (
        script.clipped_event_count(
            [{"clipped_spanning_event_count": 2}, {"clipped_spanning_event_count": 1}]
        )
        == 3
    )


def test_banned_loss_tricks_are_absent() -> None:
    script = load_script()
    source = inspect.getsource(script.main)
    text = inspect.getsource(script)
    assert "BCEWithLogitsLoss(pos_weight" not in source
    assert "boundary_weights_from_train_clips" not in source
    assert "positive_class_weight=" not in source
    assert "build_bigru_head" not in text
    assert getattr(script, "positive_class_weight_from_train_frames", None) is None
    assert getattr(script, "boundary_weights_from_train_clips", None) is None


def test_wiring_gate_and_decoder_constants_stay_locked() -> None:
    script = load_script()
    assert script.COLLAR_F1_REQUIRED == 0.25
    assert script.SEGMENT_MARGIN_REQUIRED == 0.05
    assert script.PREDECLARED_DECODER["high_threshold"] == 0.95
    assert script.PREDECLARED_DECODER["low_threshold"] == 0.855
    assert script.PREDECLARED_DECODER["max_gap_frames"] == 0
    assert script.PREDECLARED_DECODER["min_active_frames"] == 1
    assert script.PREDECLARED_DECODER["median_filter_frames"] == 3
    assert script.PREDECLARED_DECODER["onset_shift_ms"] == 0
    source = inspect.getsource(script.main)
    assert "encoder.model.parameters()" in source
    assert "AdamW(head.parameters()" in source
    assert "PREDECLARED_DECODER" in source
    assert "select_hysteresis_decoder(" not in source
    assert script.PROTOCOL == "tiled_60s_mean4_mic"
