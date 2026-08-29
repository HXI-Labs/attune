from __future__ import annotations

import pytest
import torch
from torch import nn

from attune.models.joint import (
    SUPPORTED_EVENTS,
    AdaptationPolicy,
    AttuneJointModel,
    attune_delta_checkpoint,
    load_attune_checkpoint,
    warm_start_attune_heads,
)


class FakeCTC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ctc_lo = nn.Linear(512, 20)


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoders = nn.ModuleList([nn.Linear(512, 512) for _ in range(4)])
        self.after_norm = nn.LayerNorm(512)
        self.tp_norm = nn.LayerNorm(512)


class FakeSenseVoice(nn.Module):
    encoder_output_size = 512

    def __init__(self) -> None:
        super().__init__()
        self.encoder = FakeEncoder()
        self.ctc = FakeCTC()
        self.input = nn.Linear(80, 512)

    def encode(self, speech, lengths, rich_tokens):
        del rich_tokens
        acoustic = self.input(speech)
        queries = torch.zeros(speech.shape[0], 4, 512)
        return torch.cat((queries, acoustic), dim=1), lengths + 4


class RichFakeEncoder(FakeEncoder):
    def forward(self, speech, lengths):
        return speech, lengths


class RichFakeSenseVoice(nn.Module):
    encoder_output_size = 512

    def __init__(self) -> None:
        super().__init__()
        self.encoder = RichFakeEncoder()
        self.ctc = FakeCTC()
        self.embed = nn.Embedding(10, 512)
        self.lid_int_dict = {24885: 3}
        self.textnorm_int_dict = {25017: 4}
        self.textnorm_dict = {"woitn": 4}
        self.specaug = None
        self.normalize = None


class SplitBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(512, 512)

    def forward(self, frames, mask):
        return frames + self.projection(frames), mask


class SplitEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Identity()
        self.encoders0 = nn.ModuleList([SplitBlock()])
        self.encoders = nn.ModuleList([SplitBlock() for _ in range(4)])
        self.after_norm = nn.LayerNorm(512)
        self.tp_encoders = nn.ModuleList([SplitBlock()])
        self.tp_norm = nn.LayerNorm(512)

    def output_size(self) -> int:
        return 512


class SplitFakeSenseVoice(nn.Module):
    encoder_output_size = 512

    def __init__(self) -> None:
        super().__init__()
        self.encoder = SplitEncoder()
        self.ctc = FakeCTC()
        self.embed = nn.Embedding(10, 512)
        self.lid_int_dict = {24885: 3}
        self.textnorm_int_dict = {25017: 4}
        self.textnorm_dict = {"woitn": 4}
        self.specaug = None
        self.normalize = None


def test_joint_model_shapes_and_frozen_policy() -> None:
    model = AttuneJointModel(FakeSenseVoice())
    output = model(torch.randn(2, 12, 80), torch.tensor([12, 9]))

    assert output.ctc_logits.shape == (2, 12, 20)
    assert output.event_logits.shape == (2, 12, len(SUPPORTED_EVENTS))
    assert output.event_presence_logits.shape == (2, len(SUPPORTED_EVENTS))
    assert output.style_logits.shape == (2, 2)
    assert output.affect_logits.shape == (2, 8)
    assert output.vad.shape == (2, 3)
    assert output.ood_logit.shape == (2,)
    assert output.frame_mask[1].sum() == 9
    assert model.trainable_parameter_summary()["encoder_trainable"] == 0


def test_upper_two_policy_unfreezes_only_two_encoder_blocks() -> None:
    model = AttuneJointModel(FakeSenseVoice(), adaptation_policy=AdaptationPolicy.UPPER_TWO)
    assert all(
        not parameter.requires_grad
        for parameter in model.sensevoice.encoder.encoders[0].parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.sensevoice.encoder.encoders[-1].parameters()
    )
    assert model.trainable_parameter_summary()["encoder_trainable"] > 0


def test_delta_checkpoint_contains_only_trainable_parameters() -> None:
    source = AttuneJointModel(FakeSenseVoice())
    checkpoint = attune_delta_checkpoint(source)
    restored = AttuneJointModel(FakeSenseVoice())
    load_attune_checkpoint(restored, checkpoint)

    assert checkpoint["format"] == "attune_delta_v1"
    assert checkpoint["state_dict"]
    assert all(not name.startswith("sensevoice.") for name in checkpoint["state_dict"])
    assert torch.equal(restored.affect_head.weight, source.affect_head.weight)


def test_frozen_heads_warm_start_upper_two_without_overwriting_encoder() -> None:
    source = AttuneJointModel(FakeSenseVoice())
    target = AttuneJointModel(FakeSenseVoice(), adaptation_policy=AdaptationPolicy.UPPER_TWO)
    encoder_before = target.sensevoice.encoder.encoders[-1].weight.detach().clone()

    warm_start_attune_heads(target, attune_delta_checkpoint(source))

    assert torch.equal(target.affect_head.weight, source.affect_head.weight)
    assert torch.equal(target.sensevoice.encoder.encoders[-1].weight, encoder_before)


def test_warm_start_rejects_upper_two_source() -> None:
    source = AttuneJointModel(FakeSenseVoice(), adaptation_policy=AdaptationPolicy.UPPER_TWO)
    target = AttuneJointModel(FakeSenseVoice(), adaptation_policy=AdaptationPolicy.UPPER_TWO)

    with pytest.raises(ValueError, match="source must use the frozen"):
        warm_start_attune_heads(target, attune_delta_checkpoint(source))


def test_rich_query_lookup_supports_variable_batch_sizes() -> None:
    model = AttuneJointModel(RichFakeSenseVoice())

    first = model(torch.randn(1, 8, 512), torch.tensor([8]))
    second = model(torch.randn(3, 8, 512), torch.tensor([8, 7, 6]))

    assert first.ctc_logits.shape[0] == 1
    assert second.ctc_logits.shape[0] == 3


def test_split_tail_preserves_base_asr_while_perception_tail_changes() -> None:
    model = AttuneJointModel(
        SplitFakeSenseVoice(),
        adaptation_policy=AdaptationPolicy.UPPER_TWO,
        preserve_base_asr=True,
    ).eval()
    speech = torch.randn(1, 8, 512)
    lengths = torch.tensor([8])
    with torch.inference_mode():
        before = model(speech, lengths)
        model.sensevoice.encoder.encoders[-1].projection.weight.add_(0.1)
        after = model(speech, lengths)

    assert torch.equal(before.ctc_logits, after.ctc_logits)
    assert not torch.equal(before.event_logits, after.event_logits)
    assert all(not parameter.requires_grad for parameter in model.base_asr_tail.parameters())
