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


def test_main_trains_bigru_with_masked_loss_and_predeclared_decoder() -> None:
    script = load_script()
    source = inspect.getsource(script.main)
    assert "build_bigru_head" in source
    assert "masked_bce_with_logits" in source
    assert "pad_clip_batch" in source
    assert "AdamW(head.parameters()" in source
    assert "Conv1d" not in source
    assert "boundary_weights_from_train_clips(train_y, torch=torch)" not in source
    assert "gold_event_durations_ms(train_rows)" in source
    assert "PREDECLARED_DECODER" in source
    assert "select_hysteresis_decoder(" not in source
    assert "iter_hysteresis_decoder_candidates" not in source
    assert "BCEWithLogitsLoss(pos_weight" not in source
    assert "requires_grad_(True)" not in source
    assert "validation_probabilities" in source
    assert "references_validation" in source
    assert "decode_spans(" in source
    assert "frame_targets(inspection" not in source
    assert "gold_event_durations_ms(inspection" not in source
    assert "boundary_weights_from_train_clips(inspection" not in source
    assert "positive_class_weight_from_train_frames(inspection" not in source
    assert "select_hysteresis_decoder(\n        inspection" not in source
    assert "hysteresis_spans(\n        inspection_probabilities,\n        **decoder," not in source
    assert 'default=Path("artifacts/starss23-scene-raster/bigru-head.pt")' in source
    assert 'default=Path("artifacts/starss23-scene-raster/frame-head-40epoch.pt")' in source
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
    assert script.DECODER_MEDIAN_WINDOWS == (1,)
    assert 3 not in script.DECODER_MEDIAN_WINDOWS
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
    search_source = inspect.getsource(script.iter_hysteresis_decoder_candidates)
    assert "decoder_selection_key" in search_source
    assert "iter_hysteresis_decoder_candidates" in source
    assert "DECODER_ONSET_SHIFTS_MS" in search_source
    assert "apply_onset_shift" in search_source
    assert '-int(collar["false_positive"])' not in source
    assert '-int(collar["false_positive"])' not in search_source
    assert script.choose_checkpoint_from_ablation(0.12, 0.12) == script.CHECKPOINT_UNWEIGHTED
    assert script.choose_checkpoint_from_ablation(0.12, 0.13) == script.CHECKPOINT_POSWEIGHT
    assert script.choose_checkpoint_from_ablation(0.14, 0.13) == script.CHECKPOINT_UNWEIGHTED


def test_onset_shift_pass_locks_median_and_bans_low_ratio_09() -> None:
    script = load_script()
    inspect_source = inspect.getsource(script.run_inspection_eval)
    search_source = inspect.getsource(script.iter_hysteresis_decoder_candidates)
    ablation_source = inspect.getsource(script.run_onset_shift_ablation)
    assert script.DECODER_MEDIAN_WINDOWS == (1,)
    assert 0.9 not in script.DECODER_LOW_RATIOS
    assert script.DECODER_LOW_RATIOS == (0.2, 0.3, 0.5, 0.7)
    assert script.DECODER_ONSET_SHIFTS_MS == (-180, -120, -60, 0)
    assert all(shift <= 0 for shift in script.DECODER_ONSET_SHIFTS_MS)
    assert "DECODER_ONSET_SHIFTS_MS" in search_source
    assert "select_hysteresis_decoder" not in inspect_source
    assert "iter_hysteresis_decoder_candidates" not in inspect_source
    assert "decode_spans(" in inspect_source
    assert "validation_rows" in ablation_source
    assert "inspection_rows" not in ablation_source
    assert "inspection_used" in ablation_source
    assert script.COLLAR_F1_REQUIRED == 0.25
    assert script.SEGMENT_MARGIN_REQUIRED == 0.05


def test_apply_onset_shift_moves_predicted_start_earlier() -> None:
    script = load_script()
    predictions = [[{"label": "laugh", "start_ms": 1000, "end_ms": 2000}]]
    shifted = script.apply_onset_shift(predictions, -180)
    assert shifted == [[{"label": "laugh", "start_ms": 820, "end_ms": 2000}]]
    clamped = script.apply_onset_shift(
        [[{"label": "laugh", "start_ms": 50, "end_ms": 400}]],
        -180,
    )
    assert clamped[0][0]["start_ms"] == 0
    assert clamped[0][0]["end_ms"] == 400
    dropped = script.apply_onset_shift(
        [[{"label": "laugh", "start_ms": 100, "end_ms": 150}]],
        0,
    )
    assert dropped[0][0]["start_ms"] == 100


def test_boundary_weights_ignore_inspection() -> None:
    torch = pytest.importorskip("torch")
    script = load_script()
    train = torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0]])
    inspection = torch.ones((200, 1))
    train_weights = script.boundary_weights_from_train_clips([train], torch=torch)
    leaked = script.boundary_weights_from_train_clips([inspection], torch=torch)
    assert train_weights.shape == train.shape
    assert leaked.shape != train_weights.shape
    assert float(train_weights[0, 0]) == script.BOUNDARY_WEIGHT
    assert float(train_weights[1, 0]) == script.BOUNDARY_NEIGHBOR_WEIGHT
    with pytest.raises(TypeError):
        script.boundary_weights_from_train_clips([train], inspection, torch=torch)


def test_boundary_weights_mark_first_and_last_active_frames() -> None:
    torch = pytest.importorskip("torch")
    script = load_script()
    clip = torch.tensor([[0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [0.0]])
    weights = script.boundary_weights_from_train_clips([clip], torch=torch).reshape(-1).tolist()
    assert weights[1] == script.BOUNDARY_WEIGHT
    assert weights[3] == script.BOUNDARY_WEIGHT
    assert weights[6] == script.BOUNDARY_WEIGHT
    assert weights[0] == script.BOUNDARY_NEIGHBOR_WEIGHT
    assert weights[2] == script.BOUNDARY_NEIGHBOR_WEIGHT
    assert weights[4] == script.BOUNDARY_NEIGHBOR_WEIGHT
    assert weights[5] == script.BOUNDARY_NEIGHBOR_WEIGHT
    assert weights[7] == script.BOUNDARY_NEIGHBOR_WEIGHT


def test_predeclared_decoder_matches_0a27733() -> None:
    script = load_script()
    assert script.PREDECLARED_DECODER == {
        "high_threshold": 0.95,
        "low_threshold": 0.855,
        "max_gap_frames": 0,
        "min_active_frames": 1,
        "median_filter_frames": 3,
        "onset_shift_ms": 0,
    }
    assert script.PRIOR_BEST_PASS["decoder"] == script.PREDECLARED_DECODER
    assert pytest.approx(0.13953488372093023) == script.PRIOR_BEST_COLLAR_F1
    assert script.keep_prior_best_checkpoint(0.1395) is True
    assert script.keep_prior_best_checkpoint(0.13953488372093023) is True
    assert script.keep_prior_best_checkpoint(0.1396) is False
    source = inspect.getsource(script.main)
    assert "PREDECLARED_DECODER" in source
    assert "select_hysteresis_decoder(" not in source


def test_masked_bce_ignores_padded_frames() -> None:
    torch = pytest.importorskip("torch")
    script = load_script()
    logits = torch.tensor([[[4.0], [-4.0], [4.0]]])
    targets = torch.tensor([[[1.0], [0.0], [1.0]]])
    mask = torch.tensor([[True, True, True]])
    unpadded = script.masked_bce_with_logits(logits, targets, mask, torch)
    padded_logits = torch.cat([logits, torch.tensor([[[80.0]]])], dim=1)
    padded_targets = torch.cat([targets, torch.tensor([[[0.0]]])], dim=1)
    padded_mask = torch.tensor([[True, True, True, False]])
    padded = script.masked_bce_with_logits(padded_logits, padded_targets, padded_mask, torch)
    leaked = script.bce_with_logits(padded_logits, padded_targets, torch)
    assert float(padded) == pytest.approx(float(unpadded))
    assert float(leaked) > float(padded) + 1.0


def test_pad_clip_batch_masks_only_real_frames() -> None:
    torch = pytest.importorskip("torch")
    script = load_script()
    short = torch.ones((2, 3))
    long = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    batch, mask, lengths = script.pad_clip_batch([short, long], torch)
    assert tuple(batch.shape) == (2, 4, 3)
    assert mask.tolist() == [[True, True, False, False], [True, True, True, True]]
    assert lengths.tolist() == [2, 4]
    assert torch.equal(batch[0, :2], short)
    assert torch.equal(batch[0, 2:], torch.zeros((2, 3)))


def test_bigru_head_is_not_conv1d_and_stays_tiny() -> None:
    torch = pytest.importorskip("torch")
    script = load_script()
    head = script.build_bigru_head(torch)
    script.assert_head_is_bigru_not_conv(head, torch)
    assert not any(isinstance(module, torch.nn.Conv1d) for module in head.modules())
    assert isinstance(head.gru, torch.nn.GRU)
    assert head.gru.bidirectional is True
    assert head.gru.num_layers == 1
    assert head.gru.hidden_size == 64
    trainable = sum(parameter.numel() for parameter in head.parameters())
    assert trainable == 222081
    assert trainable < 1_000_000
    conv = torch.nn.Conv1d(512, 128, kernel_size=7, padding=3)
    with pytest.raises(RuntimeError, match="must not contain Conv1d"):
        script.assert_head_is_bigru_not_conv(conv, torch)


def test_bigru_packing_ignores_padded_suffix() -> None:
    torch = pytest.importorskip("torch")
    script = load_script()
    torch.manual_seed(0)
    head = script.build_bigru_head(torch)
    head.eval()
    real = torch.randn(5, 512)
    padded = torch.cat([real, torch.randn(4, 512)])
    with torch.inference_mode():
        from_real = head(real.unsqueeze(0), lengths=torch.tensor([5]))[0]
        from_padded = head(padded.unsqueeze(0), lengths=torch.tensor([5]))[0, :5]
    assert torch.allclose(from_real, from_padded, atol=1e-5)


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
