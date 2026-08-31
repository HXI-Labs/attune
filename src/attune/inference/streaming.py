"""Bounded pseudo-streaming coordinator with provisional and committed state."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StreamingConfig:
    sample_rate_hz: int = 16_000
    channels: int = 1
    sample_width_bytes: int = 2
    window_ms: int = 3_000
    overlap_ms: int = 750
    maximum_duration_ms: int = 30_000

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0 or self.channels <= 0 or self.sample_width_bytes <= 0:
            raise ValueError("stream audio format values must be positive")
        if self.window_ms <= 0 or self.maximum_duration_ms <= 0:
            raise ValueError("stream durations must be positive")
        if not 0 <= self.overlap_ms < self.window_ms:
            raise ValueError("stream overlap must be shorter than the window")
        if self.window_ms > self.maximum_duration_ms:
            raise ValueError("stream window cannot exceed the maximum duration")

    @property
    def bytes_per_ms(self) -> float:
        return self.sample_rate_hz * self.channels * self.sample_width_bytes / 1000


@dataclass
class StreamingBuffer:
    config: StreamingConfig
    pcm: bytearray = field(default_factory=bytearray)
    last_emitted_bytes: int = 0
    revision: int = 0

    def append_base64(self, encoded: str) -> bool:
        maximum = int(self.config.maximum_duration_ms * self.config.bytes_per_ms)
        remaining = maximum - len(self.pcm)
        maximum_encoded_length = 4 * ((remaining + 2) // 3)
        if len(encoded) > maximum_encoded_length:
            raise ValueError("stream exceeds the configured maximum duration")
        try:
            chunk = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise ValueError("audio.chunk data must be valid base64") from error
        if len(self.pcm) + len(chunk) > maximum:
            raise ValueError("stream exceeds the configured maximum duration")
        self.pcm.extend(chunk)
        threshold = int(self.config.window_ms * self.config.bytes_per_ms)
        stride = int((self.config.window_ms - self.config.overlap_ms) * self.config.bytes_per_ms)
        ready = len(self.pcm) >= threshold and len(self.pcm) - self.last_emitted_bytes >= stride
        if ready:
            self.last_emitted_bytes = len(self.pcm)
            self.revision += 1
        return ready

    def wav_bytes(self) -> bytes:
        """Wrap accumulated PCM16 bytes in a deterministic mono WAV container."""
        return self._wav_bytes(bytes(self.pcm))

    def window_wav_bytes(self) -> bytes:
        """Return only the configured rolling window for provisional inference."""
        window_bytes = int(self.config.window_ms * self.config.bytes_per_ms)
        return self._wav_bytes(bytes(self.pcm[-window_bytes:]))

    @property
    def window_start_ms(self) -> int:
        duration_ms = round(len(self.pcm) / self.config.bytes_per_ms)
        return max(0, duration_ms - self.config.window_ms)

    def _wav_bytes(self, pcm: bytes) -> bytes:
        import io
        import wave

        target = io.BytesIO()
        with wave.open(target, "wb") as handle:
            handle.setnchannels(self.config.channels)
            handle.setsampwidth(self.config.sample_width_bytes)
            handle.setframerate(self.config.sample_rate_hz)
            handle.writeframes(pcm)
        return target.getvalue()
