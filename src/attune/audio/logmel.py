"""Deterministic log-mel features for compact acoustic probes."""

from __future__ import annotations

import array
import sys
import wave
from pathlib import Path
from typing import Any


class AudioFeatureError(ValueError):
    """Raised when audio cannot satisfy the fixed feature contract."""


SPEECH_NORMALIZED_LOGMEL_VERSION = "speech-normalized-v1"
TEMPORAL_LOGMEL_VERSION = "speech-normalized-64x256-v1"


def read_pcm_wav(path: Path, torch: Any) -> tuple[Any, int]:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            sample_width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
    except (OSError, wave.Error) as error:
        raise AudioFeatureError(f"cannot read PCM WAV {path}: {error}") from error

    typecodes = {1: "B", 2: "h", 4: "i"}
    if sample_width not in typecodes:
        raise AudioFeatureError(f"{path}: unsupported {sample_width * 8}-bit PCM")
    samples = array.array(typecodes[sample_width])
    samples.frombytes(frames)
    if sys.byteorder != "little" and sample_width > 1:
        samples.byteswap()
    waveform = torch.tensor(samples, dtype=torch.float32)
    if sample_width == 1:
        waveform = (waveform - 128.0) / 128.0
    else:
        waveform /= float(2 ** (sample_width * 8 - 1))
    if channels > 1:
        waveform = waveform.reshape(-1, channels).mean(dim=1)
    return waveform, sample_rate


def resample(waveform: Any, source_rate: int, target_rate: int, torch: Any) -> Any:
    if source_rate == target_rate:
        return waveform
    output_length = max(1, round(waveform.numel() * target_rate / source_rate))
    return torch.nn.functional.interpolate(
        waveform.reshape(1, 1, -1),
        size=output_length,
        mode="linear",
        align_corners=False,
    ).reshape(-1)


def mel_filterbank(
    torch: Any,
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    device: str,
) -> Any:
    def hz_to_mel(value: Any) -> Any:
        return 2595.0 * torch.log10(1.0 + value / 700.0)

    def mel_to_hz(value: Any) -> Any:
        return 700.0 * (torch.pow(10.0, value / 2595.0) - 1.0)

    frequencies = torch.linspace(0.0, sample_rate / 2, n_fft // 2 + 1, device=device)
    low = hz_to_mel(torch.tensor(20.0, device=device))
    high = hz_to_mel(torch.tensor(sample_rate / 2, device=device))
    edges = mel_to_hz(torch.linspace(low, high, n_mels + 2, device=device))
    filters = torch.zeros((n_mels, frequencies.numel()), device=device)
    for index in range(n_mels):
        rising = (frequencies - edges[index]) / (edges[index + 1] - edges[index])
        falling = (edges[index + 2] - frequencies) / (edges[index + 2] - edges[index + 1])
        filters[index] = torch.clamp(torch.minimum(rising, falling), min=0.0)
    return filters


def _logmel_spectrogram(
    waveform: Any,
    torch: Any,
    *,
    sample_rate: int = 16_000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 40,
    max_duration_s: float = 10.0,
) -> Any:
    waveform = waveform[: round(sample_rate * max_duration_s)]
    if waveform.numel() < n_fft:
        waveform = torch.nn.functional.pad(waveform, (0, n_fft - waveform.numel()))
    window = torch.hann_window(n_fft)
    spectrum = (
        torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )
        .abs()
        .square()
    )
    filters = mel_filterbank(
        torch,
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        device="cpu",
    )
    return torch.log(filters @ spectrum + 1e-6)


def _logmel_embedding(
    waveform: Any,
    torch: Any,
    *,
    sample_rate: int = 16_000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 40,
    temporal_bins: int = 8,
    max_duration_s: float = 10.0,
) -> Any:
    logmel = _logmel_spectrogram(
        waveform,
        torch,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        max_duration_s=max_duration_s,
    )
    pooled = torch.nn.functional.adaptive_avg_pool1d(logmel, temporal_bins).flatten()
    return torch.cat((pooled, logmel.mean(dim=1), logmel.std(dim=1)))


def fixed_logmel_embedding(
    path: Path,
    torch: Any,
    *,
    sample_rate: int = 16_000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 40,
    temporal_bins: int = 8,
    max_duration_s: float = 10.0,
) -> Any:
    """Return temporal log-mel pooling with no learned parameters."""
    waveform, source_rate = read_pcm_wav(path, torch)
    waveform = resample(waveform, source_rate, sample_rate, torch)
    return _logmel_embedding(
        waveform,
        torch,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        temporal_bins=temporal_bins,
        max_duration_s=max_duration_s,
    )


def speech_normalized_logmel_embedding(
    path: Path,
    torch: Any,
    *,
    sample_rate: int = 16_000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 40,
    temporal_bins: int = 8,
    max_duration_s: float = 10.0,
) -> Any:
    """Return gain-normalized log-mel pooling over the active speech region."""
    waveform, source_rate = read_pcm_wav(path, torch)
    waveform = resample(waveform, source_rate, sample_rate, torch)
    waveform = waveform - waveform.mean()
    if waveform.numel() >= n_fft:
        frames = waveform.unfold(0, n_fft, hop_length)
        frame_rms = frames.square().mean(dim=-1).sqrt()
        peak_rms = frame_rms.max()
        active = torch.nonzero(frame_rms >= peak_rms * 0.03).flatten()
        if active.numel():
            margin = round(0.1 * sample_rate)
            start = max(0, int(active[0]) * hop_length - margin)
            end = min(
                waveform.numel(),
                int(active[-1]) * hop_length + n_fft + margin,
            )
            waveform = waveform[start:end]
    rms = waveform.square().mean().sqrt()
    if not torch.isfinite(rms) or rms <= 1e-7:
        raise AudioFeatureError(f"{path}: audio contains no usable speech energy")
    waveform = (waveform * (0.1 / rms)).clamp(-1.0, 1.0)
    return _logmel_embedding(
        waveform,
        torch,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        temporal_bins=temporal_bins,
        max_duration_s=max_duration_s,
    )


def speech_normalized_logmel_sequence(
    path: Path,
    torch: Any,
    *,
    sample_rate: int = 16_000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 64,
    temporal_bins: int = 256,
    max_duration_s: float = 10.0,
) -> Any:
    """Return a fixed-size, gain-normalized log-mel sequence."""
    waveform, source_rate = read_pcm_wav(path, torch)
    waveform = resample(waveform, source_rate, sample_rate, torch)
    waveform = waveform - waveform.mean()
    if waveform.numel() >= n_fft:
        frames = waveform.unfold(0, n_fft, hop_length)
        frame_rms = frames.square().mean(dim=-1).sqrt()
        active = torch.nonzero(frame_rms >= frame_rms.max() * 0.03).flatten()
        if active.numel():
            margin = round(0.1 * sample_rate)
            start = max(0, int(active[0]) * hop_length - margin)
            end = min(
                waveform.numel(),
                int(active[-1]) * hop_length + n_fft + margin,
            )
            waveform = waveform[start:end]
    rms = waveform.square().mean().sqrt()
    if not torch.isfinite(rms) or rms <= 1e-7:
        raise AudioFeatureError(f"{path}: audio contains no usable speech energy")
    waveform = (waveform * (0.1 / rms)).clamp(-1.0, 1.0)
    logmel = _logmel_spectrogram(
        waveform,
        torch,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        max_duration_s=max_duration_s,
    )
    return torch.nn.functional.adaptive_avg_pool1d(logmel, temporal_bins)
