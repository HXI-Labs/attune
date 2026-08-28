from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]


def load_script() -> ModuleType:
    path = REPO / "scripts" / "train_starss23_frozen_mlp_listenpack.py"
    spec = importlib.util.spec_from_file_location("train_starss23_frozen_mlp_listenpack", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_note_exists_and_locks_gates() -> None:
    note = (REPO / "research" / "starss23-frozen-mlp-listenpack.md").read_text(encoding="utf-8")
    assert "100 ms" in note
    assert "not second-listener" in note.lower() or "not second-listener" in note
    assert "0.25" in note and "0.05" in note
    assert "high 0.95" in note
    assert "low 0.855" in note
    assert "inspection never" in note.lower() or "never used for fit" in note.lower() or "Inspection never" in note
    assert "Do not search" in note or "not searched" in note.lower() or "not searched" in note
    assert "frame-head-40epoch.pt" in note


def test_wiring_gates_are_not_lowered() -> None:
    script = load_script()
    assert script.COLLAR_F1_REQUIRED == 0.25
    assert script.SEGMENT_MARGIN_REQUIRED == 0.05
    assert (
        script.should_wire_starss23_timestamps(
            segment_f1=0.5124,
            whole_clip_segment_f1=0.1721,
            collar_f1=0.1395,
        )
        is False
    )
    assert (
        script.should_wire_starss23_timestamps(
            segment_f1=0.40,
            whole_clip_segment_f1=0.17,
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
    assert (
        script.should_wire_starss23_timestamps(
            segment_f1=0.40,
            whole_clip_segment_f1=0.17,
            collar_f1=0.249,
        )
        is False
    )


def test_decoder_is_locked_to_0a27733_and_not_searched() -> None:
    script = load_script()
    assert script.PREDECLARED_DECODER == {
        "high_threshold": 0.95,
        "low_threshold": 0.855,
        "max_gap_frames": 0,
        "min_active_frames": 1,
        "median_filter_frames": 3,
        "onset_shift_ms": 0,
    }
    raster = script.load_scene_raster_module()
    assert script.PREDECLARED_DECODER == raster.PREDECLARED_DECODER
    source = inspect.getsource(script.main)
    assert "PREDECLARED_DECODER" in source
    assert "select_hysteresis_decoder(" not in source
    assert "iter_hysteresis_decoder_candidates" not in source
    assert "DECODER_HIGH_THRESHOLDS" not in source
    assert "decoder = dict(PREDECLARED_DECODER)" in source
    assert "decoder_searched" in inspect.getsource(script)


def test_main_keeps_encoder_frozen_and_trains_mlp_only() -> None:
    script = load_script()
    source = inspect.getsource(script.main)
    text = inspect.getsource(script)
    assert "FrozenSenseVoiceFrameEncoder" in source
    assert "assert_encoder_frozen" in source
    assert "AdamW(head.parameters()" in source
    assert "requires_grad_(True)" not in source
    assert "encoder.model.parameters(), lr" not in source
    assert "build_mlp_head" in source
    assert "build_bigru_head" not in source
    assert "last-block" not in text.lower()
    assert "last_two" not in text
    assert script.PROTOCOL == "first_60s_scene_raster"
    assert "tiling_in_train" in source
    assert "assert_no_later_tiles_in_fit" in source


def test_assert_encoder_frozen_rejects_trainable_weights() -> None:
    script = load_script()

    class FakeParam:
        def __init__(self, n: int, requires_grad: bool) -> None:
            self._n = n
            self.requires_grad = requires_grad

        def numel(self) -> int:
            return self._n

    class FakeEncoder:
        def __init__(self, trainable: int) -> None:
            self.model = type("M", (), {"parameters": lambda self: [FakeParam(trainable, trainable > 0)]})()

        def metadata(self) -> dict[str, int]:
            return {"trainable_parameters": 0}

    script.assert_encoder_frozen(FakeEncoder(0))
    with pytest.raises(RuntimeError, match="not frozen"):
        script.assert_encoder_frozen(FakeEncoder(4))


def test_inspection_pack_is_held_out_of_train_and_val() -> None:
    script = load_script()
    development = [
        json.loads(line)
        for line in (REPO / "data/manifests/starss23-scene-raster-development.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    inspection = [
        json.loads(line)
        for line in (REPO / "data/manifests/starss23-gold-review-pack.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    first60 = script.first_60s_subset(development)
    train_rows, validation_rows = script.split_development(development)
    script.assert_no_later_tiles_in_fit(train_rows, split="train")
    script.assert_no_later_tiles_in_fit(validation_rows, split="val")
    script.assert_inspection_held_out(train_rows, validation_rows, inspection)
    script.assert_first_60s_counts(train_rows, validation_rows, inspection)
    train_ids = {row["clip_id"] for row in train_rows}
    val_ids = {row["clip_id"] for row in validation_rows}
    insp_ids = {row["clip_id"] for row in inspection}
    assert train_ids.isdisjoint(insp_ids)
    assert val_ids.isdisjoint(insp_ids)
    assert {row["source_recording"] for row in first60}.isdisjoint(
        {row["source_recording"] for row in inspection}
    )
    assert {row["room"] for row in first60}.isdisjoint({row["room"] for row in inspection})
    assert len(train_rows) == 43
    assert len(validation_rows) == 19
    assert len(inspection) == 49
    assert all(int(row["source_window_start_ms"]) == 0 for row in train_rows + validation_rows)
    source = inspect.getsource(script.main)
    assert "starss23-gold-review-pack.jsonl" in source
    assert "frame-targets(inspection" not in source.replace(" ", "")
    assert "gold_event_durations_ms(inspection" not in source


def test_checkpoint_path_does_not_clobber_40epoch() -> None:
    script = load_script()
    with pytest.raises(RuntimeError, match="must not overwrite frame-head-40epoch.pt"):
        script.assert_checkpoint_does_not_overwrite_40epoch(
            Path("artifacts/starss23-scene-raster/frame-head-40epoch.pt")
        )
    script.assert_checkpoint_does_not_overwrite_40epoch(script.NEW_CHECKPOINT)
    assert script.NEW_CHECKPOINT == Path("artifacts/starss23-frozen-mlp-listenpack/frame-head.pt")
    source = inspect.getsource(script.main)
    assert "assert_checkpoint_does_not_overwrite_40epoch" in source


def test_listenpack_selects_mixed_inspection_clips() -> None:
    script = load_script()
    inspection = [
        json.loads(line)
        for line in (REPO / "data/manifests/starss23-gold-review-pack.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    selected = script.select_listenpack_clips(inspection)
    assert 8 <= len(selected) <= 12
    laughs = [row for row in selected if row["events"]]
    negatives = [row for row in selected if not row["events"]]
    assert laughs and negatives
    selected_ids = {row["clip_id"] for row in selected}
    train = script.split_development(
        [
            json.loads(line)
            for line in (REPO / "data/manifests/starss23-scene-raster-development.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
    )[0]
    assert selected_ids.isdisjoint({row["clip_id"] for row in train})
    assert script.PREFERRED_LAUGH_CLIP_IDS[0] in selected_ids
    assert script.PREFERRED_TRUE_NEGATIVE_CLIP_IDS[0] in selected_ids


def test_listenpack_fixtures_caption_unwired_and_orange_overlay(tmp_path: Path) -> None:
    script = load_script()
    script.write_fixture_htmls(tmp_path)
    laugh_html = (tmp_path / "fixtures" / "true-laugh.attune.html").read_text(encoding="utf-8")
    negative_html = (tmp_path / "fixtures" / "true-negative.attune.html").read_text(encoding="utf-8")
    assert "unwired in product infer" in laugh_html
    assert "research listen test" in laugh_html
    assert "100 ms overlay" in laugh_html
    assert "orange" in laugh_html.lower()
    assert "predicted" in laugh_html.lower()
    assert "True negative" in negative_html or "true_negative=True" in negative_html
    laugh_json = json.loads((tmp_path / "fixtures" / "true-laugh.attune.json").read_text())
    assert laugh_json["wired"] is False
    assert laugh_json["overlays_100ms"]
    assert laugh_json["predicted_spans"]


def test_dcase_loader_still_refuses_starss23_when_gate_closed() -> None:
    from attune.models.temporal_probe import dcase_checkpoint_acceptance

    accepted, reason = dcase_checkpoint_acceptance({"dataset": "starss23"})
    assert accepted is False
    assert reason is not None
    assert "unwired" in reason.lower()
