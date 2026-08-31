from __future__ import annotations

import wave

from attune.evaluation.deployment import (
    balanced_benchmark_rows,
    deployment_audio_contract_error,
)


def test_balanced_benchmark_rows_caps_each_source() -> None:
    rows = [
        {
            "clip_id": f"{dataset}-{index}",
            "dataset_id": dataset,
            "split": "sealed_test",
            "audio_path": "/fixture.wav",
        }
        for dataset in ("a", "b")
        for index in range(4)
    ]

    selected = balanced_benchmark_rows(rows, split="sealed_test", per_dataset=2)

    assert [row["clip_id"] for row in selected] == ["a-0", "a-1", "b-0", "b-1"]


def test_deployment_audio_contract_audits_wav_header(tmp_path) -> None:
    valid = tmp_path / "valid.wav"
    with wave.open(str(valid), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 160)
    assert deployment_audio_contract_error(valid) is None

    stereo = tmp_path / "stereo.wav"
    with wave.open(str(stereo), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 320)
    assert deployment_audio_contract_error(stereo) == "channels=2 (required 1)"
