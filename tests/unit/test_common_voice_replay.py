from __future__ import annotations

import importlib.util
import sys
import wave
from pathlib import Path


def test_replay_partition_is_stable_and_has_both_sides() -> None:
    path = Path("scripts/prepare_common_voice_replay.py")
    sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location("prepare_common_voice_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = [module._partition(f"speaker-{index}") for index in range(100)]
    assert set(values) == {"train", "development", "sealed_test"}
    assert module._partition("stable-speaker") == module._partition("stable-speaker")


def test_partial_manifest_is_atomic_and_resumable(tmp_path: Path) -> None:
    path = Path("scripts/prepare_common_voice_replay.py")
    sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location("prepare_common_voice_replay_partial", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    audio = tmp_path / "clip.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 16_000)
    row = {
        "cache_path": audio.name,
        "sha256": module.file_digest(audio),
        "duration_s": 1.0,
        "client_id": "speaker",
        "source_row_index": 1,
    }
    manifest = tmp_path / "partial.jsonl"
    module._write_partial(manifest, [row])
    loaded = module._load_partial(manifest, tmp_path)
    assert {key: value for key, value in loaded[0].items() if key != "partition"} == row
    assert loaded[0]["partition"] == module._partition("speaker")
