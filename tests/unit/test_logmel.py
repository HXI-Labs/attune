from __future__ import annotations

import math
import wave
from pathlib import Path

import torch

from attune.audio.logmel import (
    AudioFeatureError,
    speech_normalized_logmel_embedding,
    speech_normalized_logmel_sequence,
)


def _write_wave(path: Path, samples: torch.Tensor, sample_rate: int = 16_000) -> None:
    pcm = (samples.clamp(-1.0, 1.0) * 32_767).round().to(torch.int16).numpy().tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def test_speech_normalized_embedding_is_gain_invariant(tmp_path: Path) -> None:
    time = torch.arange(16_000) / 16_000
    speech = torch.sin(2 * math.pi * 220 * time) * (0.6 + 0.4 * torch.sin(2 * math.pi * 3 * time))
    quiet_path = tmp_path / "quiet.wav"
    loud_path = tmp_path / "loud.wav"
    _write_wave(quiet_path, speech * 0.08)
    _write_wave(loud_path, speech * 0.7)

    quiet = speech_normalized_logmel_embedding(quiet_path, torch)
    loud = speech_normalized_logmel_embedding(loud_path, torch)

    assert quiet.shape == (400,)
    assert torch.corrcoef(torch.stack((quiet, loud)))[0, 1] > 0.999
    assert (quiet - loud).abs().mean() < 0.3

    quiet_sequence = speech_normalized_logmel_sequence(quiet_path, torch)
    loud_sequence = speech_normalized_logmel_sequence(loud_path, torch)
    assert quiet_sequence.shape == (64, 256)
    assert (
        torch.corrcoef(torch.stack((quiet_sequence.flatten(), loud_sequence.flatten())))[0, 1]
        > 0.999
    )


def test_speech_normalized_embedding_rejects_silence(tmp_path: Path) -> None:
    path = tmp_path / "silence.wav"
    _write_wave(path, torch.zeros(8_000))

    try:
        speech_normalized_logmel_embedding(path, torch)
    except AudioFeatureError as error:
        assert "no usable speech energy" in str(error)
    else:
        raise AssertionError("silent audio should not produce an affect embedding")
