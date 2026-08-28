from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from attune.models.sensevoice_last_block import (  # noqa: E402
    LastBlockSenseVoiceFrameEncoder,
    LastTwoBlockSenseVoiceFrameEncoder,
)
from attune.models.sensevoice_probe import (  # noqa: E402
    FrozenSenseVoiceEncoder,
    FrozenSenseVoiceFrameEncoder,
)


class FakeFrontend(torch.nn.Module):
    frame_shift = 10
    frame_length = 25
    lfr_n = 6
    dither = 1.0


class FakeSANM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(512, 512)

    def forward(self, features: object, mask: object, *args: object, **kwargs: object) -> tuple:
        return self.proj(features), mask


class FakeSenseVoiceEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoders0 = torch.nn.ModuleList([FakeSANM()])
        self.encoders = torch.nn.ModuleList([FakeSANM(), FakeSANM()])
        self.tp_encoders = torch.nn.ModuleList([FakeSANM(), FakeSANM()])
        self.after_norm = torch.nn.LayerNorm(512)
        self.tp_norm = torch.nn.LayerNorm(512)

    def forward(self, features: object, lengths: object) -> tuple[object, object]:
        encoded = features
        batch, time, _ = encoded.shape
        mask = torch.ones(batch, 1, time)
        for layer in self.encoders0:
            encoded, mask = layer(encoded, mask)[:2]
        for layer in self.encoders:
            encoded, mask = layer(encoded, mask)[:2]
        encoded = self.after_norm(encoded)
        for layer in self.tp_encoders:
            encoded, mask = layer(encoded, mask)[:2]
        encoded = self.tp_norm(encoded)
        return encoded, lengths


class FakeSenseVoiceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = FakeSenseVoiceEncoder()
        self.embed = torch.nn.Embedding(4, 512)
        self.lid_dict = {"auto": 0}
        self.textnorm_dict = {"woitn": 3}


class FakeAutoModel:
    def __init__(self, **_kwargs: object) -> None:
        self.model = FakeSenseVoiceModel()
        self.kwargs = {"frontend": FakeFrontend()}

    def generate(self, **_kwargs: object) -> object:
        raise AssertionError("feature extraction must not call generate()")


class FrozenFakeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(560, 512)

    def forward(self, features: object, lengths: object) -> tuple[object, object]:
        return self.projection(features), lengths


class FrozenFakeSenseVoiceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = FrozenFakeEncoder()
        self.embed = torch.nn.Embedding(4, 560)
        self.lid_dict = {"auto": 0}
        self.textnorm_dict = {"woitn": 3}


class FrozenFakeAutoModel:
    def __init__(self, **_kwargs: object) -> None:
        self.model = FrozenFakeSenseVoiceModel()
        self.kwargs = {"frontend": FakeFrontend()}


def fake_feature_loader(
    _audio_path: Path, _frontend: object, _torch: object
) -> tuple[object, object]:
    return torch.arange(8 * 512, dtype=torch.float32).reshape(1, 8, 512), torch.tensor([8])


def frozen_feature_loader(
    _audio_path: Path, _frontend: object, _torch: object
) -> tuple[object, object]:
    return torch.arange(8 * 560, dtype=torch.float32).reshape(1, 8, 560), torch.tensor([8])


def last_block_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> LastBlockSenseVoiceFrameEncoder:
    checkpoint = tmp_path / "model"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "model.pt").write_bytes(b"fake model")
    monkeypatch.setenv("ATTUNE_SENSEVOICE_LICENSE_REVIEWED", "1")
    return LastBlockSenseVoiceFrameEncoder(
        checkpoint,
        tmp_path / "cache",
        torch,
        model_factory=FakeAutoModel,
        feature_loader=fake_feature_loader,
        prefix_cache=True,
    )


def test_frozen_encoder_still_has_zero_trainable_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "model.pt").write_bytes(b"fake model")
    monkeypatch.setenv("ATTUNE_SENSEVOICE_LICENSE_REVIEWED", "1")
    encoder = FrozenSenseVoiceEncoder(
        checkpoint,
        tmp_path / "cache",
        torch,
        model_factory=FrozenFakeAutoModel,
        feature_loader=frozen_feature_loader,
    )
    frames = FrozenSenseVoiceFrameEncoder(
        checkpoint,
        tmp_path / "frames",
        torch,
        model_factory=FrozenFakeAutoModel,
        feature_loader=frozen_feature_loader,
    )
    assert (
        sum(
            parameter.numel() for parameter in encoder.model.parameters() if parameter.requires_grad
        )
        == 0
    )
    assert (
        sum(parameter.numel() for parameter in frames.model.parameters() if parameter.requires_grad)
        == 0
    )
    assert encoder.metadata()["trainable_parameters"] == 0
    assert frames.metadata()["trainable_parameters"] == 0


def test_last_block_trainable_params_are_only_the_final_sanm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder = last_block_encoder(tmp_path, monkeypatch)
    names = encoder.trainable_parameter_names()
    assert encoder.last_block_name == "encoder.tp_encoders.1"
    assert names
    assert all(name.startswith("encoder.tp_encoders.1.") for name in names)
    frozen = [
        name for name, parameter in encoder.model.named_parameters() if not parameter.requires_grad
    ]
    assert any(name.startswith("encoder.tp_encoders.0.") for name in frozen)
    assert any(name.startswith("encoder.encoders.") for name in frozen)
    assert any(name.startswith("encoder.tp_norm.") for name in frozen)
    assert encoder.frontend.dither == 0.0
    assert encoder.trainable_parameter_count() == sum(
        parameter.numel() for parameter in encoder.last_block.parameters()
    )


def test_last_block_frames_keep_grad_only_when_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder = last_block_encoder(tmp_path, monkeypatch)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fixture")
    eval_frames = encoder(audio, train=False)
    train_frames = encoder(audio, train=True)
    assert eval_frames.shape[1] == 512
    assert eval_frames.grad_fn is None
    assert train_frames.requires_grad
    loss = train_frames.sum()
    loss.backward()
    assert any(parameter.grad is not None for parameter in encoder.last_block.parameters())
    leaked = [
        name
        for name, parameter in encoder.model.named_parameters()
        if parameter.grad is not None and not name.startswith(encoder.last_block_name + ".")
    ]
    assert leaked == []
    assert encoder.cache_misses == 1
    assert encoder.cache_hits == 1


class FakeAutoModelTwenty:
    def __init__(self, **_kwargs: object) -> None:
        model = FakeSenseVoiceModel()
        model.encoder.tp_encoders = torch.nn.ModuleList([FakeSANM() for _ in range(20)])
        self.model = model
        self.kwargs = {"frontend": FakeFrontend()}

    def generate(self, **_kwargs: object) -> object:
        raise AssertionError("feature extraction must not call generate()")


def last_two_block_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> LastTwoBlockSenseVoiceFrameEncoder:
    checkpoint = tmp_path / "model"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "model.pt").write_bytes(b"fake model")
    monkeypatch.setenv("ATTUNE_SENSEVOICE_LICENSE_REVIEWED", "1")
    return LastTwoBlockSenseVoiceFrameEncoder(
        checkpoint,
        tmp_path / "cache",
        torch,
        model_factory=FakeAutoModelTwenty,
        feature_loader=fake_feature_loader,
        prefix_cache=True,
    )


def last_block_encoder_twenty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> LastBlockSenseVoiceFrameEncoder:
    checkpoint = tmp_path / "model"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "model.pt").write_bytes(b"fake model")
    monkeypatch.setenv("ATTUNE_SENSEVOICE_LICENSE_REVIEWED", "1")
    return LastBlockSenseVoiceFrameEncoder(
        checkpoint,
        tmp_path / "cache",
        torch,
        model_factory=FakeAutoModelTwenty,
        feature_loader=fake_feature_loader,
        prefix_cache=True,
    )


def test_last_two_block_trainable_names_are_only_tp_encoders_18_and_19(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder = last_two_block_encoder(tmp_path, monkeypatch)
    names = encoder.trainable_parameter_names()
    assert encoder.unfrozen_module_names() == (
        "encoder.tp_encoders.18",
        "encoder.tp_encoders.19",
    )
    assert names
    assert all(
        name.startswith("encoder.tp_encoders.18.") or name.startswith("encoder.tp_encoders.19.")
        for name in names
    )
    assert any(name.startswith("encoder.tp_encoders.18.") for name in names)
    assert any(name.startswith("encoder.tp_encoders.19.") for name in names)
    frozen = [
        name for name, parameter in encoder.model.named_parameters() if not parameter.requires_grad
    ]
    assert any(name.startswith("encoder.tp_encoders.17.") for name in frozen)
    assert any(name.startswith("encoder.encoders.") for name in frozen)
    assert any(name.startswith("encoder.tp_norm.") for name in frozen)
    assert encoder.frontend.dither == 0.0
    assert encoder.trainable_parameter_count() == sum(
        parameter.numel() for parameter in encoder.unfrozen_parameters()
    )


def test_last_block_class_still_unfreezes_only_final_sanm_on_twenty_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder = last_block_encoder_twenty(tmp_path, monkeypatch)
    names = encoder.trainable_parameter_names()
    assert encoder.last_block_name == "encoder.tp_encoders.19"
    assert encoder.unfrozen_module_names() == ("encoder.tp_encoders.19",)
    assert names
    assert all(name.startswith("encoder.tp_encoders.19.") for name in names)
    frozen = [
        name for name, parameter in encoder.model.named_parameters() if not parameter.requires_grad
    ]
    assert any(name.startswith("encoder.tp_encoders.18.") for name in frozen)


def test_last_two_block_frames_keep_grad_only_when_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder = last_two_block_encoder(tmp_path, monkeypatch)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fixture")
    eval_frames = encoder(audio, train=False)
    train_frames = encoder(audio, train=True)
    assert eval_frames.shape[1] == 512
    assert eval_frames.grad_fn is None
    assert train_frames.requires_grad
    loss = train_frames.sum()
    loss.backward()
    allowed = tuple(name + "." for name in encoder.unfrozen_module_names())
    assert any(
        parameter.grad is not None
        for name, parameter in encoder.model.named_parameters()
        if name.startswith("encoder.tp_encoders.18.")
    )
    assert any(
        parameter.grad is not None
        for name, parameter in encoder.model.named_parameters()
        if name.startswith("encoder.tp_encoders.19.")
    )
    leaked = [
        name
        for name, parameter in encoder.model.named_parameters()
        if parameter.grad is not None and not name.startswith(allowed)
    ]
    assert leaked == []
    assert encoder.cache_misses == 1
    assert encoder.cache_hits == 1
