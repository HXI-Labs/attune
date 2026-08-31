from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from attune.models.sensevoice_probe import (  # noqa: E402
    SENSEVOICE_EN_EMBEDDING,
    SENSEVOICE_FRAME_EMBEDDING,
    FrozenSenseVoiceEncoder,
    FrozenSenseVoiceFrameEncoder,
    pool_sensevoice_frames,
)


class FakeFrontend(torch.nn.Module):
    frame_shift = 10
    frame_length = 25
    lfr_n = 6


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
        self.lid_dict = {"auto": 0, "en": 1}
        self.textnorm_dict = {"woitn": 3}


class FakeAutoModel:
    def __init__(self, **_kwargs: object) -> None:
        self.model = FakeSenseVoiceModel()
        self.kwargs = {"frontend": FakeFrontend()}

    def generate(self, **_kwargs: object) -> object:
        raise AssertionError("feature extraction must not call generate()")


def fake_feature_loader(
    _audio_path: Path, _frontend: object, _torch: object
) -> tuple[object, object]:
    return torch.arange(8 * 560, dtype=torch.float32).reshape(1, 8, 560), torch.tensor([8])


def extractor(
    extractor_type: type[FrozenSenseVoiceEncoder],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> FrozenSenseVoiceEncoder:
    checkpoint = tmp_path / "model"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "model.pt").write_bytes(b"fake model")
    monkeypatch.setenv("ATTUNE_SENSEVOICE_LICENSE_REVIEWED", "1")
    return extractor_type(
        checkpoint,
        tmp_path / "cache",
        torch,
        model_factory=FakeAutoModel,
        feature_loader=fake_feature_loader,
    )


def test_frame_encoder_is_frozen_deterministic_and_uses_separate_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fixture")
    frames_extractor = extractor(FrozenSenseVoiceFrameEncoder, tmp_path, monkeypatch)

    first = frames_extractor(audio)
    second = frames_extractor(audio)

    assert first.shape == (8, 512)
    assert torch.equal(first, second)
    assert first.grad_fn is None and not first.requires_grad
    assert all(not parameter.requires_grad for parameter in frames_extractor.model.parameters())
    assert frames_extractor.frontend.dither == 0.0
    assert SENSEVOICE_FRAME_EMBEDDING in str(frames_extractor._cache_path(audio))
    assert frames_extractor.cache_misses == 1
    assert frames_extractor.cache_hits == 1


def test_pooling_frames_exactly_reconstructs_existing_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fixture")
    torch.manual_seed(7)
    pooled_extractor = extractor(FrozenSenseVoiceEncoder, tmp_path, monkeypatch)
    torch.manual_seed(7)
    frame_extractor = extractor(FrozenSenseVoiceFrameEncoder, tmp_path, monkeypatch)

    pooled = pooled_extractor(audio)
    reconstructed = pool_sensevoice_frames(frame_extractor(audio), torch)

    assert torch.equal(pooled, reconstructed)


def test_frame_time_geometry_excludes_query_prefix_without_audio_shift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame_extractor = extractor(FrozenSenseVoiceFrameEncoder, tmp_path, monkeypatch)

    assert frame_extractor.frame_hop_ms == 60.0
    assert frame_extractor.frame_centers_ms(3) == (30.0, 90.0, 150.0)
    assert frame_extractor.metadata()["query_frames_excluded"] == 4
    assert frame_extractor.metadata()["trainable_parameters"] == 0


def test_english_query_uses_a_separate_embedding_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fixture")
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"fake model")
    monkeypatch.setenv("ATTUNE_SENSEVOICE_LICENSE_REVIEWED", "1")
    english = FrozenSenseVoiceEncoder(
        checkpoint,
        tmp_path / "cache",
        torch,
        model_factory=FakeAutoModel,
        feature_loader=fake_feature_loader,
        query_language="en",
    )
    automatic = FrozenSenseVoiceEncoder(
        checkpoint,
        tmp_path / "cache",
        torch,
        model_factory=FakeAutoModel,
        feature_loader=fake_feature_loader,
    )

    assert english.metadata()["name"] == SENSEVOICE_EN_EMBEDDING
    assert english.metadata()["query_language"] == "en"
    assert english._cache_path(audio) != automatic._cache_path(audio)
