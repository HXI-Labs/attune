from __future__ import annotations

import importlib.util
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from attune.training.prepare import load_source_rows


def _module():
    path = Path("scripts/prepare_inline_event_mixtures.py")
    spec = importlib.util.spec_from_file_location("prepare_inline_event_mixtures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_active_event_removes_long_silence() -> None:
    module = _module()
    samples = np.zeros(module.SAMPLE_RATE, dtype=np.float32)
    samples[6_000:8_000] = 0.5
    active = module._active_event(samples)
    assert 2_000 <= len(active) <= 4_000
    assert np.max(active) == pytest.approx(0.5)


@pytest.mark.parametrize("placement", ("before", "overlay", "after"))
def test_mixture_span_matches_inserted_event(placement: str) -> None:
    module = _module()
    speech = np.full(module.SAMPLE_RATE, 0.05, dtype=np.float32)
    event = np.full(module.SAMPLE_RATE // 4, 0.25, dtype=np.float32)
    mixed, start, end, synthesis = module._mix(
        speech,
        event,
        placement=placement,
        event_level_db=0.0,
        seed=7,
    )
    assert end - start == len(event)
    assert len(mixed) >= end
    assert synthesis["placement"] == placement
    assert 0.54 < np.max(np.abs(mixed)) < 0.93


def test_synthesis_level_and_placement_are_not_tied_to_event_class() -> None:
    module = _module()
    levels_by_label = {label: set() for label in module.EVENT_LABELS}
    placements_by_label = {label: set() for label in module.EVENT_LABELS}
    for index in range(100):
        for label in module.EVENT_LABELS:
            placement, level = module._synthesis_parameters(module._seed(f"speech-{index}", label))
            levels_by_label[label].add(level)
            placements_by_label[label].add(placement)
    assert all(values == set(module.EVENT_LEVEL_DB) for values in levels_by_label.values())
    assert all(values == set(module.PLACEMENTS) for values in placements_by_label.values())


def test_pcm16_round_trip(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "audio.wav"
    expected = np.linspace(-0.8, 0.8, 1_000, dtype=np.float32)
    module._write_pcm16(path, expected)
    actual = module._read_pcm16(path)
    assert np.max(np.abs(actual - expected)) < 1e-4
    with wave.open(str(path), "rb") as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16_000, 1, 2)


def test_event_selection_skips_silent_source(tmp_path: Path) -> None:
    module = _module()
    silent = tmp_path / "silent.wav"
    audible = tmp_path / "audible.wav"
    module._write_pcm16(silent, np.zeros(module.SAMPLE_RATE, dtype=np.float32))
    samples = np.zeros(module.SAMPLE_RATE, dtype=np.float32)
    samples[4_000:8_000] = 0.4
    module._write_pcm16(audible, samples)
    candidates = [
        {
            "clip_id": "silent",
            "audio_path": str(silent),
            "audio_sha256": module.file_sha256(silent),
        },
        {
            "clip_id": "audible",
            "audio_path": str(audible),
            "audio_sha256": module.file_sha256(audible),
        },
    ]
    excluded: set[str] = set()
    selected, event = module._select_event(candidates, 0, {}, excluded)
    assert selected["clip_id"] == "audible"
    assert len(event) > 0
    assert excluded == {"silent"}


def test_create_mixtures_excludes_sealed_sources_and_writes_valid_rows(tmp_path: Path) -> None:
    module = _module()
    speech_root = tmp_path / "speech"
    speech_path = speech_root / "speech.wav"
    event_path = tmp_path / "event.wav"
    module._write_pcm16(speech_path, np.full(module.SAMPLE_RATE, 0.05, dtype=np.float32))
    event = np.zeros(module.SAMPLE_RATE, dtype=np.float32)
    event[4_000:8_000] = 0.4
    module._write_pcm16(event_path, event)

    common_voice_manifest = tmp_path / "common-voice.jsonl"
    common_voice_rows = [
        {
            "clip_id": "speech-train",
            "partition": "train",
            "client_id": "speaker-train",
            "cache_path": "speech.wav",
            "sha256": module.file_sha256(speech_path),
            "transcript": "The parcel arrives on Thursday.",
        },
        {
            "clip_id": "speech-sealed",
            "partition": "sealed_test",
            "client_id": "speaker-sealed",
            "cache_path": "speech.wav",
            "sha256": module.file_sha256(speech_path),
            "transcript": "This row must remain sealed.",
        },
    ]
    common_voice_manifest.write_text("".join(json.dumps(row) + "\n" for row in common_voice_rows))
    vocalsound_manifest = tmp_path / "vocalsound.jsonl"
    vocalsound_rows = [
        {
            "clip_id": f"event-{label}",
            "split": "train",
            "speaker_id": f"event-speaker-{label}",
            "audio_path": str(event_path),
            "audio_sha256": module.file_sha256(event_path),
            "event_presence": [label],
        }
        for label in module.EVENT_LABELS
    ]
    vocalsound_manifest.write_text("".join(json.dumps(row) + "\n" for row in vocalsound_rows))

    output_manifest = tmp_path / "mixtures.jsonl"
    rows = module.create_mixtures(
        common_voice_manifest=common_voice_manifest,
        common_voice_root=speech_root,
        vocalsound_manifest=vocalsound_manifest,
        output_root=tmp_path / "mixtures",
        output_manifest=output_manifest,
        audit_path=tmp_path / "audit.json",
    )
    assert len(rows) == len(module.EVENT_LABELS)
    assert {row["split"] for row in rows} == {"train"}
    assert {row["events"][0]["label"] for row in rows} == set(module.EVENT_LABELS)
    assert all(row["events"][0]["end_ms"] <= row["duration_ms"] for row in rows)
    assert len(load_source_rows(output_manifest)) == len(module.EVENT_LABELS)
