from __future__ import annotations

from attune.audio.contracts import validate_audio_contract
from attune.inference.packaging import package_for_trusted_channel


def test_audio_contract() -> None:
    assert validate_audio_contract(sample_rate_hz=16_000, channels=1, duration_seconds=3.0).valid
    invalid = validate_audio_contract(sample_rate_hz=48_000, channels=2, duration_seconds=0.1)
    assert not invalid.valid
    assert len(invalid.issues) == 3


def test_trusted_channel_never_concatenates_metadata(example_output) -> None:
    package = package_for_trusted_channel(example_output)
    assert set(package) == {"spoken_transcript", "paralinguistic_metadata"}
    assert package["spoken_transcript"]["text"] == "I said leave me alone"
    assert package["spoken_transcript"]["words"][0]["start_ms"] == 0
    assert "transcript" not in package["paralinguistic_metadata"]
    assert "interpretation_warning" not in package["spoken_transcript"]["text"]
