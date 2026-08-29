"""ONNX export and parity validation for unified Attune models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from attune.models.joint import AttuneJointModel

OUTPUT_NAMES = (
    "ctc_logits",
    "acoustic_lengths",
    "event_logits",
    "event_start_logits",
    "event_end_logits",
    "event_presence_logits",
    "style_logits",
    "affect_logits",
    "vad",
    "ood_logit",
    "ood_embedding",
)


def sensevoice_feature_size(model: AttuneJointModel) -> int:
    """Return the post-frontend feature width expected by SenseVoice.

    Official SenseVoice uses LFR-stacked filterbanks, so the encoder input is
    commonly wider than the underlying 80-bin mel representation.  Export must
    match the cached features instead of silently assuming 80 dimensions.
    """
    sensevoice = model.sensevoice
    for owner in (sensevoice, getattr(sensevoice, "encoder", None)):
        if owner is None:
            continue
        for name in ("input_size", "input_dim", "idim"):
            value = getattr(owner, name, None)
            if isinstance(value, int) and value > 0:
                return value

    encoder = getattr(sensevoice, "encoder", None)
    initial_blocks = getattr(encoder, "encoders0", None)
    if initial_blocks is not None and len(initial_blocks):
        attention = getattr(initial_blocks[0], "self_attn", None)
        query_key_value = getattr(attention, "linear_q_k_v", None)
        width = getattr(query_key_value, "in_features", None)
        if isinstance(width, int) and width > 0:
            return width

    raise ValueError(
        "could not infer SenseVoice post-frontend feature size; pass feature_size explicitly"
    )


class ExportableAttune(nn.Module):
    def __init__(self, model: AttuneJointModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, speech: Tensor, speech_lengths: Tensor) -> tuple[Tensor, ...]:
        output = self.model(speech, speech_lengths)
        return (
            output.ctc_logits,
            output.acoustic_lengths,
            output.event_logits,
            output.event_start_logits,
            output.event_end_logits,
            output.event_presence_logits,
            output.style_logits,
            output.affect_logits,
            output.vad,
            output.ood_logit,
            output.ood_embedding,
        )


@dataclass(frozen=True)
class ParityResult:
    maximum_absolute_error: float
    mean_absolute_error: float
    outputs_checked: int


def export_onnx(
    model: AttuneJointModel,
    output_path: Path,
    *,
    feature_size: int | None = None,
    sample_frames: int = 160,
    opset: int = 18,
) -> None:
    model.eval().cpu()
    wrapper = ExportableAttune(model).eval()
    feature_size = feature_size or sensevoice_feature_size(model)
    speech = torch.randn(1, sample_frames, feature_size)
    lengths = torch.tensor([sample_frames], dtype=torch.long)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (speech, lengths),
        output_path,
        input_names=("speech", "speech_lengths"),
        output_names=OUTPUT_NAMES,
        dynamic_axes={
            "speech": {0: "batch", 1: "input_frames"},
            "speech_lengths": {0: "batch"},
            "ctc_logits": {0: "batch", 1: "encoder_frames"},
            "acoustic_lengths": {0: "batch"},
            "event_logits": {0: "batch", 1: "encoder_frames"},
            "event_start_logits": {0: "batch", 1: "encoder_frames"},
            "event_end_logits": {0: "batch", 1: "encoder_frames"},
            "event_presence_logits": {0: "batch"},
            "style_logits": {0: "batch"},
            "affect_logits": {0: "batch"},
            "vad": {0: "batch"},
            "ood_logit": {0: "batch"},
            "ood_embedding": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )


def validate_onnx_parity(
    model: AttuneJointModel,
    onnx_path: Path,
    *,
    sample: Tensor | None = None,
    absolute_tolerance: float = 1e-3,
) -> ParityResult:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("install the deployment extra for ONNX validation") from error
    model.eval().cpu()
    sample = sample if sample is not None else torch.randn(2, 121, sensevoice_feature_size(model))
    lengths = torch.tensor([sample.shape[1], sample.shape[1] - 17], dtype=torch.long)
    with torch.inference_mode():
        expected = ExportableAttune(model)(sample, lengths)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        list(OUTPUT_NAMES),
        {"speech": sample.numpy(), "speech_lengths": lengths.numpy()},
    )
    errors = [
        np.abs(reference.detach().numpy() - candidate)
        for reference, candidate in zip(expected, actual, strict=True)
        if np.issubdtype(candidate.dtype, np.floating)
    ]
    maximum = max(float(error.max()) for error in errors)
    mean = sum(float(error.mean()) for error in errors) / len(errors)
    if maximum > absolute_tolerance:
        raise ValueError(
            f"ONNX parity failed: maximum absolute error {maximum:.6g} exceeds "
            f"{absolute_tolerance:.6g}"
        )
    return ParityResult(maximum, mean, len(errors))


def write_export_report(
    path: Path,
    *,
    model_path: Path,
    parity: ParityResult,
    metadata: dict[str, Any],
) -> None:
    payload = {
        "schema_version": "1.0",
        "model_path": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "parity": {
            "maximum_absolute_error": parity.maximum_absolute_error,
            "mean_absolute_error": parity.mean_absolute_error,
            "outputs_checked": parity.outputs_checked,
        },
        **metadata,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
