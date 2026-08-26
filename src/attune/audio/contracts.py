"""Lightweight Phase 0 audio input contracts."""

from __future__ import annotations

from dataclasses import dataclass

TARGET_SAMPLE_RATE_HZ = 16_000
TARGET_CHANNELS = 1
MIN_DURATION_SECONDS = 0.5
MAX_DURATION_SECONDS = 30.0


@dataclass(frozen=True)
class AudioContractResult:
    valid: bool
    issues: tuple[str, ...]


def validate_audio_contract(
    *, sample_rate_hz: int, channels: int, duration_seconds: float
) -> AudioContractResult:
    """Validate input after conversion; conversion itself belongs to Phase 1."""
    issues: list[str] = []
    if sample_rate_hz != TARGET_SAMPLE_RATE_HZ:
        issues.append("audio must be converted to 16 kHz")
    if channels != TARGET_CHANNELS:
        issues.append("audio must be converted to mono")
    if not MIN_DURATION_SECONDS <= duration_seconds <= MAX_DURATION_SECONDS:
        issues.append("audio duration must be between 0.5 and 30 seconds")
    return AudioContractResult(valid=not issues, issues=tuple(issues))


def placeholder_quality_probabilities() -> dict[str, float]:
    """Return explicit unknown-like placeholders until quality models exist."""
    return {"clipping": 0.0, "low_snr": 0.0, "far_field": 0.0}
