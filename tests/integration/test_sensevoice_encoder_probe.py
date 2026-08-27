from __future__ import annotations

import math
import os
import wave
from array import array
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from attune.models.sensevoice_probe import FrozenSenseVoiceEncoder  # noqa: E402


class FakeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(560, 512)

    def forward(self, features: object, lengths: object) -> tuple[object, object]:
        return self.projection(features), lengths


class FakeSenseVoiceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = FakeEncoder()
        self.embed = torch.nn.Embedding(4, 560)
        self.lid_dict = {"auto": 0}
        self.textnorm_dict = {"woitn": 3}


class FakeAutoModel:
    def __init__(self, **_kwargs: object) -> None:
        self.model = FakeSenseVoiceModel()
        self.kwargs = {"frontend": torch.nn.Identity()}

    def generate(self, **_kwargs: object) -> list[dict[str, str]]:
        raise AssertionError("embedding extraction must not call generate()")


def write_tone(path: Path) -> None:
    samples = array(
        "h",
        (round(8_000 * math.sin(2 * math.pi * 440 * index / 16_000)) for index in range(1_600)),
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(samples.tobytes())


def fake_feature_loader(
    _audio_path: Path, _frontend: object, _torch: object
) -> tuple[object, object]:
    return torch.ones(1, 8, 560), torch.tensor([8], dtype=torch.int32)


@pytest.mark.integration
def test_fake_encoder_is_frozen_and_embeddings_are_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"fake model")
    audio = tmp_path / "clip.wav"
    write_tone(audio)
    monkeypatch.setenv("ATTUNE_SENSEVOICE_LICENSE_REVIEWED", "1")

    extractor = FrozenSenseVoiceEncoder(
        checkpoint,
        tmp_path / "cache",
        torch,
        model_factory=FakeAutoModel,
        feature_loader=fake_feature_loader,
    )
    first = extractor(audio)
    second = extractor(audio)

    assert first.shape == (extractor.output_size,)
    assert torch.equal(first, second)
    assert not first.requires_grad
    assert all(not parameter.requires_grad for parameter in extractor.model.parameters())
    assert extractor.metadata()["frontend_dither"] == 0.0
    assert extractor.metadata()["cache_misses"] == 1
    assert extractor.metadata()["cache_hits"] == 1


@pytest.mark.integration
def test_optional_official_sensevoice_encoder_extracts_without_gradients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = os.environ.get("ATTUNE_SENSEVOICE_SMALL_PATH")
    if not model_path or not (Path(model_path) / "model.pt").is_file():
        pytest.skip("official local SenseVoiceSmall weights are unavailable")
    pytest.importorskip("funasr")
    monkeypatch.setenv("ATTUNE_SENSEVOICE_LICENSE_REVIEWED", "1")
    audio = tmp_path / "clip.wav"
    write_tone(audio)

    extractor = FrozenSenseVoiceEncoder(Path(model_path), tmp_path / "cache", torch)
    embedding = extractor(audio)

    assert embedding.shape == (extractor.output_size,)
    assert embedding.grad_fn is None
    assert all(not parameter.requires_grad for parameter in extractor.model.parameters())
