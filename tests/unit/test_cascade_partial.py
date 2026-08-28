from __future__ import annotations

import wave
from pathlib import Path

from attune.baselines.adapters import BaselineInput
from attune.baselines.cascade import AttuneCascade, MissingAffectAdapter
from attune.schema.output import AffectCategory


def write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0" * 1_600)


def test_missing_affect_adapter_abstains(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    write_wav(audio)

    prediction = MissingAffectAdapter().predict(BaselineInput(audio_path=audio))

    assert prediction.output.affect.abstain is True
    assert prediction.output.affect.top_label is None
    assert set(prediction.output.affect.categories) == set(AffectCategory)
    assert prediction.diagnostics is not None
    assert prediction.diagnostics["affect_source"] == "missing_emotion2vec_abstain"


def test_attune_cascade_name_omits_missing_optional_heads(tmp_path: Path) -> None:
    cascade = AttuneCascade(
        sensevoice_checkpoint=tmp_path,
        embedding_cache=tmp_path / "cache",
    )

    assert cascade.name == "attune-cascade:sensevoice+affect-abstain+aed"
    assert isinstance(cascade.affect, MissingAffectAdapter)
    assert cascade.event_heads == ()


def test_attune_cascade_name_keeps_full_package_when_all_heads_present(tmp_path: Path) -> None:
    cascade = AttuneCascade(
        sensevoice_checkpoint=tmp_path,
        emotion2vec_checkpoint=tmp_path / "emotion2vec",
        vocalsound_probe_checkpoint=tmp_path / "vocalsound.pt",
        fsd50k_probe_checkpoint=tmp_path / "fsd50k.pt",
        embedding_cache=tmp_path / "cache",
        temporal_head_checkpoints=(tmp_path / "dcase.pt",),
    )

    assert cascade.name == (
        "attune-cascade:sensevoice+emotion2vec+aed+vocalsound-probe+fsd50k-probe+1-gated-frame-heads"
    )
