from __future__ import annotations

from attune.audio.contracts import validate_audio_contract


def test_audio_contract_reports_each_invalid_boundary() -> None:
    valid = validate_audio_contract(
        sample_rate_hz=16_000,
        channels=1,
        duration_seconds=3.0,
    )
    invalid = validate_audio_contract(
        sample_rate_hz=48_000,
        channels=2,
        duration_seconds=0.1,
    )

    assert valid.valid
    assert invalid.issues == (
        "audio must be converted to 16 kHz",
        "audio must be converted to mono",
        "audio duration must be between 0.5 and 30 seconds",
    )
