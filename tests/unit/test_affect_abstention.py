from __future__ import annotations

import wave
from pathlib import Path

from attune.baselines.adapters import BaselineInput, build_partial_output
from attune.schema.output import AffectCategory, AttuneOutput


def _write_silence(path: Path) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0" * 1_600)


def test_abstaining_affect_has_null_top_label_and_valid_schema(tmp_path: Path) -> None:
    audio = tmp_path / "fixture.wav"
    _write_silence(audio)
    distribution = {label: 0.01 for label in AffectCategory}
    distribution[AffectCategory.NEUTRAL] = 0.93

    output = build_partial_output(
        BaselineInput(audio_path=audio),
        model_name="fixture",
        transcript="",
        category=AffectCategory.NEUTRAL,
        distribution=distribution,
        abstain=True,
    )

    assert output.affect.abstain is True
    assert output.affect.top_label is None
    AttuneOutput.model_validate_json(output.model_dump_json())
