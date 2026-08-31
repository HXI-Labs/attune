from __future__ import annotations

import hashlib
import importlib.util
import wave
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).parents[2] / "scripts/prepare_berst_gain_sweep.py"
SPEC = importlib.util.spec_from_file_location("prepare_berst_gain_sweep", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
GAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GAIN)


def write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(samples.astype("<i2").tobytes())


def test_gain_labels_are_stable() -> None:
    assert GAIN.gain_label(-12) == "m12db"
    assert GAIN.gain_label(0) == "z0db"
    assert GAIN.gain_label(6.5) == "p6p5db"


def test_apply_gain_scales_and_reports_pcm_clipping(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    write_wav(source, np.asarray([-20_000, -1_000, 1_000, 20_000]))

    report = GAIN.apply_gain(source, target, 6.0)

    with wave.open(str(target), "rb") as audio:
        output = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    assert output.tolist() == [-32768, -1995, 1995, 32767]
    assert report["clipped_samples"] == 2
    assert report["clipping_ratio"] == 0.5


def test_gain_sweep_preserves_targets_and_separates_datasets(tmp_path: Path) -> None:
    audio = tmp_path / "source.wav"
    write_wav(audio, np.asarray([100, -100] * 8000))
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    source = tmp_path / "source.jsonl"
    source.write_text(
        "{"
        f'"clip_id":"clip","dataset_id":"berst_v1","split":"sealed_test",'
        f'"speaker_id":"speaker","audio_path":"{audio.name}","audio_sha256":"{digest}",'
        '"duration_ms":1000,"styles":["shouting"]}'
        "\n"
    )
    output = tmp_path / "prepared" / "gain.jsonl"

    rows, report = GAIN.prepare_gain_sweep(
        source,
        output,
        tmp_path / "audio",
        gains_db=(-6.0, 0.0, 6.0),
    )

    assert [row.dataset_id for row in rows] == [
        "berst_v1_gain_m6db",
        "berst_v1_gain_z0db",
        "berst_v1_gain_p6db",
    ]
    assert all(row.styles == ["shouting"] for row in rows)
    assert report["source_clips"] == 1
    assert report["prepared_clips"] == 3
