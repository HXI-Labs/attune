"""Unified SenseVoice ASR, event/style, and affect model."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from torch import Tensor, nn

from attune.schema.output import AffectCategory, EventLabel, StyleLabel

SUPPORTED_EVENTS = (
    EventLabel.LAUGH,
    EventLabel.SOB,
    EventLabel.SCREAM,
    EventLabel.SIGH,
    EventLabel.COUGH,
    EventLabel.THROAT_CLEAR,
    EventLabel.SNEEZE,
)
SUPPORTED_STYLES = (StyleLabel.SHOUTING, StyleLabel.WHISPERING)
DEFAULT_RICH_TOKENS = (24885, 25009, 25010, 25017)


class AdaptationPolicy(StrEnum):
    FROZEN = "frozen"
    UPPER_TWO = "upper_two"


@dataclass(frozen=True)
class JointOutput:
    ctc_logits: Tensor
    acoustic_lengths: Tensor
    frame_mask: Tensor
    event_logits: Tensor
    event_start_logits: Tensor
    event_end_logits: Tensor
    event_presence_logits: Tensor
    style_logits: Tensor
    affect_logits: Tensor
    vad: Tensor
    ood_logit: Tensor
    affect_embedding: Tensor
    ood_embedding: Tensor


class AttentiveStatisticsPooling(nn.Module):
    """Masked attentive mean and standard deviation pooling."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Tanh(),
            nn.Linear(input_size // 2, 1),
        )

    def forward(self, frames: Tensor, mask: Tensor) -> Tensor:
        scores = self.score(frames).squeeze(-1).masked_fill(~mask, -torch.inf)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        mean = (weights * frames).sum(dim=1)
        variance = (weights * (frames - mean.unsqueeze(1)).square()).sum(dim=1)
        return torch.cat((mean, variance.clamp_min(1e-8).sqrt()), dim=-1)


class TemporalEventHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, event_count: int) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.presence = nn.Conv1d(hidden_size, event_count, kernel_size=1)
        self.start = nn.Conv1d(hidden_size, event_count, kernel_size=1)
        self.end = nn.Conv1d(hidden_size, event_count, kernel_size=1)

    def forward(self, frames: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.temporal(frames.transpose(1, 2))
        return tuple(
            projection(hidden).transpose(1, 2)
            for projection in (self.presence, self.start, self.end)
        )


class AttuneJointModel(nn.Module):
    """Add compact multi-task heads to an official SenseVoiceSmall model.

    The wrapped model is expected to expose ``encode``, ``ctc.ctc_lo``, and an
    encoder with 512-dimensional output.  This is the interface exposed by the
    pinned FunASR SenseVoiceSmall implementation.
    """

    def __init__(
        self,
        sensevoice: nn.Module,
        *,
        adaptation_policy: AdaptationPolicy = AdaptationPolicy.FROZEN,
        hidden_size: int = 256,
        affect_embedding_size: int = 128,
        preserve_base_asr: bool = False,
    ) -> None:
        super().__init__()
        self.sensevoice = sensevoice
        encoder_size = int(getattr(sensevoice, "encoder_output_size", 512))
        self.event_head = TemporalEventHead(encoder_size, hidden_size, len(SUPPORTED_EVENTS))
        self.pooling = AttentiveStatisticsPooling(encoder_size)
        pooled_size = encoder_size * 2
        self.affect_projection = nn.Sequential(
            nn.Linear(pooled_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, affect_embedding_size),
            nn.LayerNorm(affect_embedding_size),
        )
        self.style_head = nn.Linear(affect_embedding_size, len(SUPPORTED_STYLES))
        self.event_presence_head = nn.Linear(affect_embedding_size, len(SUPPORTED_EVENTS))
        self.affect_head = nn.Linear(affect_embedding_size, len(AffectCategory))
        self.vad_head = nn.Sequential(nn.Linear(affect_embedding_size, 3), nn.Tanh())
        self.ood_projection = nn.Linear(affect_embedding_size, 32)
        self.ood_head = nn.Linear(affect_embedding_size, 1)
        self.preserve_base_asr = preserve_base_asr
        if preserve_base_asr:
            if adaptation_policy != AdaptationPolicy.UPPER_TWO:
                raise ValueError("base-ASR preservation requires the upper_two adaptation policy")
            encoder = sensevoice.encoder
            required = (
                "encoders0",
                "encoders",
                "after_norm",
                "tp_encoders",
                "tp_norm",
                "embed",
                "output_size",
            )
            if not all(hasattr(encoder, name) for name in required):
                raise ValueError("SenseVoice encoder does not support a split-tail ASR route")
            if len(encoder.encoders) < 2:
                raise ValueError("SenseVoice encoder does not expose two upper encoder blocks")
            self.base_asr_tail = nn.ModuleList(copy.deepcopy(list(encoder.encoders[-2:])))
            self.base_asr_after_norm = copy.deepcopy(encoder.after_norm)
            self.base_asr_tp_norm = copy.deepcopy(encoder.tp_norm)
            for module in (
                self.base_asr_tail,
                self.base_asr_after_norm,
                self.base_asr_tp_norm,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        self.register_buffer("_lid_lookup", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_textnorm_lookup", torch.empty(0, dtype=torch.long), persistent=False)
        if all(
            hasattr(sensevoice, name)
            for name in ("lid_int_dict", "textnorm_int_dict", "textnorm_dict")
        ):
            lid_size = max(max(sensevoice.lid_int_dict, default=0), DEFAULT_RICH_TOKENS[0]) + 1
            lid_lookup = torch.zeros(lid_size, dtype=torch.long)
            for token, value in sensevoice.lid_int_dict.items():
                lid_lookup[int(token)] = int(value)
            textnorm_default = int(sensevoice.textnorm_dict["woitn"])
            textnorm_size = (
                max(max(sensevoice.textnorm_int_dict, default=0), DEFAULT_RICH_TOKENS[3]) + 1
            )
            textnorm_lookup = torch.full((textnorm_size,), textnorm_default, dtype=torch.long)
            for token, value in sensevoice.textnorm_int_dict.items():
                textnorm_lookup[int(token)] = int(value)
            self._lid_lookup = lid_lookup
            self._textnorm_lookup = textnorm_lookup
        self.set_adaptation_policy(adaptation_policy)

    def set_adaptation_policy(self, policy: AdaptationPolicy) -> None:
        for parameter in self.sensevoice.parameters():
            parameter.requires_grad_(False)
        if policy == AdaptationPolicy.UPPER_TWO:
            layers = getattr(self.sensevoice.encoder, "encoders", None)
            if layers is None or len(layers) < 2:
                raise ValueError("SenseVoice encoder does not expose two upper encoder blocks")
            for layer in layers[-2:]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
            for name in ("after_norm", "tp_norm"):
                module = getattr(self.sensevoice.encoder, name, None)
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
        elif policy != AdaptationPolicy.FROZEN:
            raise ValueError(f"unsupported adaptation policy: {policy}")
        self.adaptation_policy = policy

    def trainable_parameter_summary(self) -> dict[str, int]:
        encoder = sum(
            parameter.numel()
            for parameter in self.sensevoice.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {"total": total, "trainable": trainable, "encoder_trainable": encoder}

    def forward(
        self,
        speech: Tensor,
        speech_lengths: Tensor,
        rich_tokens: Tensor | None = None,
    ) -> JointOutput:
        batch_size = speech.shape[0]
        if rich_tokens is None:
            rich_tokens = torch.tensor(
                DEFAULT_RICH_TOKENS, device=speech.device, dtype=torch.long
            ).repeat(batch_size, 1)
        if rich_tokens.shape != (batch_size, 4):
            raise ValueError("rich_tokens must have shape (batch, 4)")
        encoded, asr_encoded, encoded_lengths = self._encode_views(
            speech, speech_lengths.clone(), rich_tokens
        )
        if encoded.shape[1] <= 4:
            raise ValueError("SenseVoice returned no acoustic encoder frames")
        acoustic = encoded[:, 4:, :]
        acoustic_lengths = encoded_lengths - 4
        positions = torch.arange(acoustic.shape[1], device=acoustic.device)
        frame_mask = positions.unsqueeze(0) < acoustic_lengths.unsqueeze(1)
        asr_acoustic = asr_encoded[:, 4:, :]
        ctc_logits = self.sensevoice.ctc.ctc_lo(asr_acoustic)
        event_logits, start_logits, end_logits = self.event_head(acoustic)
        pooled = self.pooling(acoustic, frame_mask)
        affect_embedding = self.affect_projection(pooled)
        return JointOutput(
            ctc_logits=ctc_logits,
            acoustic_lengths=acoustic_lengths,
            frame_mask=frame_mask,
            event_logits=event_logits,
            event_start_logits=start_logits,
            event_end_logits=end_logits,
            event_presence_logits=self.event_presence_head(affect_embedding),
            style_logits=self.style_head(affect_embedding),
            affect_logits=self.affect_head(affect_embedding),
            vad=self.vad_head(affect_embedding),
            ood_logit=self.ood_head(affect_embedding).squeeze(-1),
            affect_embedding=affect_embedding,
            ood_embedding=nn.functional.normalize(self.ood_projection(affect_embedding), dim=-1),
        )

    def _encode(
        self, speech: Tensor, speech_lengths: Tensor, rich_tokens: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Use deterministic rich queries for export and calibrated inference."""
        required = ("embed", "lid_int_dict", "textnorm_int_dict", "encoder")
        if not all(hasattr(self.sensevoice, name) for name in required):
            return self.sensevoice.encode(speech, speech_lengths, rich_tokens)
        if self.sensevoice.specaug is not None and self.training:
            speech, speech_lengths = self.sensevoice.specaug(speech, speech_lengths)
        if self.sensevoice.normalize is not None:
            speech, speech_lengths = self.sensevoice.normalize(speech, speech_lengths)
        lid_ids = self._lid_lookup[rich_tokens[:, 0]]
        style_ids = self._textnorm_lookup[rich_tokens[:, 3]]
        language_query = self.sensevoice.embed(lid_ids.unsqueeze(1))
        style_query = self.sensevoice.embed(style_ids.unsqueeze(1))
        event_emo_query = self.sensevoice.embed(
            torch.tensor([[1, 2]], device=speech.device, dtype=torch.long)
        ).repeat(speech.size(0), 1, 1)
        speech = torch.cat((language_query, event_emo_query, style_query, speech), dim=1)
        return self.sensevoice.encoder(speech, speech_lengths + 4)

    def _encode_views(
        self, speech: Tensor, speech_lengths: Tensor, rich_tokens: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return adapted perception frames and a frozen-tail ASR view."""
        if not self.preserve_base_asr:
            encoded, encoded_lengths = self._encode(speech, speech_lengths, rich_tokens)
            return encoded, encoded, encoded_lengths

        if self.sensevoice.specaug is not None and self.training:
            speech, speech_lengths = self.sensevoice.specaug(speech, speech_lengths)
        if self.sensevoice.normalize is not None:
            speech, speech_lengths = self.sensevoice.normalize(speech, speech_lengths)
        lid_ids = self._lid_lookup[rich_tokens[:, 0]]
        style_ids = self._textnorm_lookup[rich_tokens[:, 3]]
        language_query = self.sensevoice.embed(lid_ids.unsqueeze(1))
        style_query = self.sensevoice.embed(style_ids.unsqueeze(1))
        event_emo_query = self.sensevoice.embed(
            torch.tensor([[1, 2]], device=speech.device, dtype=torch.long)
        ).repeat(speech.size(0), 1, 1)
        speech = torch.cat((language_query, event_emo_query, style_query, speech), dim=1)
        lengths = speech_lengths + 4

        encoder = self.sensevoice.encoder
        positions = torch.arange(speech.shape[1], device=lengths.device)
        mask = (positions.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1)
        shared = encoder.embed(speech * (float(encoder.output_size()) ** 0.5))
        for layer in encoder.encoders0:
            shared, mask = layer(shared, mask)[:2]
        for layer in encoder.encoders[:-2]:
            shared, mask = layer(shared, mask)[:2]

        adapted, adapted_mask = shared, mask
        for layer in encoder.encoders[-2:]:
            adapted, adapted_mask = layer(adapted, adapted_mask)[:2]
        adapted = encoder.after_norm(adapted)

        base_asr, base_mask = shared, mask
        for layer in self.base_asr_tail:
            base_asr, base_mask = layer(base_asr, base_mask)[:2]
        base_asr = self.base_asr_after_norm(base_asr)
        encoded_lengths = base_mask.squeeze(1).sum(1).int()

        for layer in encoder.tp_encoders:
            adapted, adapted_mask = layer(adapted, adapted_mask)[:2]
            base_asr, base_mask = layer(base_asr, base_mask)[:2]
        return (
            encoder.tp_norm(adapted),
            self.base_asr_tp_norm(base_asr),
            encoded_lengths,
        )


def attune_delta_checkpoint(model: AttuneJointModel) -> dict[str, Any]:
    """Serialize only trainable Attune/encoder deltas, never base-model copies."""
    trainable = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return {
        "format": "attune_delta_v1",
        "adaptation_policy": model.adaptation_policy.value,
        "state_dict": trainable,
    }


def load_attune_checkpoint(model: AttuneJointModel, checkpoint: dict[str, Any]) -> None:
    """Load a versioned Attune delta, with support for legacy full state dicts."""
    if checkpoint.get("format") == "attune_delta_v1":
        policy = checkpoint.get("adaptation_policy")
        if policy != model.adaptation_policy.value:
            raise ValueError(
                f"checkpoint policy {policy!r} does not match model "
                f"{model.adaptation_policy.value!r}"
            )
        state = checkpoint.get("state_dict")
        if not isinstance(state, dict) or not state:
            raise ValueError("Attune delta checkpoint contains no state_dict")
        expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        if set(state) != expected:
            missing = sorted(expected - set(state))
            unexpected = sorted(set(state) - expected)
            raise ValueError(
                f"Attune delta keys differ; missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        model.load_state_dict(state, strict=False)
        return
    model.load_state_dict(checkpoint)


def warm_start_attune_heads(model: AttuneJointModel, checkpoint: dict[str, Any]) -> None:
    """Load only Attune heads from a frozen delta into an adapted candidate.

    This deliberately excludes every ``sensevoice.*`` tensor.  An upper-layer
    candidate must always start its encoder from the reviewed base checkpoint,
    while retaining the already-trained task heads from the frozen probe.
    """
    if checkpoint.get("format") != "attune_delta_v1":
        raise ValueError("warm-start requires an attune_delta_v1 checkpoint")
    if checkpoint.get("adaptation_policy") != AdaptationPolicy.FROZEN.value:
        raise ValueError("warm-start source must use the frozen adaptation policy")
    if model.adaptation_policy == AdaptationPolicy.FROZEN:
        raise ValueError("warm-start target must adapt at least one encoder layer")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Attune delta checkpoint contains no state_dict")
    if any(name.startswith("sensevoice.") for name in state):
        raise ValueError("frozen warm-start checkpoint unexpectedly contains encoder tensors")
    expected = {
        name for name, _parameter in model.named_parameters() if not name.startswith("sensevoice.")
    }
    if set(state) != expected:
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        raise ValueError(
            f"warm-start head keys differ; missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    model.load_state_dict(state, strict=False)


def load_local_sensevoice(checkpoint: str, *, device: str = "cpu") -> Any:
    """Load only a local, explicitly reviewed SenseVoice checkpoint."""
    import os
    from pathlib import Path

    if os.environ.get("ATTUNE_SENSEVOICE_LICENSE_REVIEWED") != "1":
        raise RuntimeError(
            "set ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1 only after reviewing the model licence"
        )
    path = Path(checkpoint).expanduser().resolve()
    if not (path / "model.pt").is_file():
        raise FileNotFoundError(f"SenseVoice model.pt is missing below {path}")
    from funasr import AutoModel

    previous = {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "MODELSCOPE_OFFLINE")}
    os.environ.update({"HF_HUB_OFFLINE": "1", "MODELSCOPE_OFFLINE": "1"})
    try:
        wrapper = AutoModel(model=str(path), disable_update=True, device=device)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return wrapper.model
