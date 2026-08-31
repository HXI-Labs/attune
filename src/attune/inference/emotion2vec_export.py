"""Portable export for the truncated emotion2vec+ affect branch."""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch import nn

from attune.inference.emotion2vec_student import (
    OnnxTruncatedEmotion2VecPredictor,
    TorchScriptTruncatedEmotion2VecPredictor,
    TruncatedEmotion2VecPredictor,
)
from attune.models.truncated_emotion2vec import pool_layer
from attune.training.data import file_sha256


def _convert_padding_mask(frontend, features: torch.Tensor, padding_mask: torch.Tensor):
    input_lengths = (~padding_mask).sum(-1)
    for _, kernel_size, stride in frontend.feature_enc_layers:
        input_lengths = torch.div(input_lengths - kernel_size, stride, rounding_mode="floor") + 1
    positions = torch.arange(features.shape[1], device=features.device).unsqueeze(0)
    return positions >= input_lengths.unsqueeze(1)


class _ExportModel(nn.Module):
    def __init__(self, predictor: TruncatedEmotion2VecPredictor) -> None:
        super().__init__()
        self.model = predictor.model
        self.head = predictor.head
        self.register_buffer("feature_mean", predictor.feature_mean.clone())
        self.register_buffer("feature_scale", predictor.feature_scale.clone())
        self.depth = predictor.depth
        self.temperature = predictor.temperature

    def forward(self, samples: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        extracted = self.model.extract_features(samples, padding_mask=padding_mask)
        features = pool_layer(
            extracted["layer_results"][self.depth - 1],
            extracted["padding_mask"],
        )
        normalized = (features - self.feature_mean) / self.feature_scale
        return (self.head(normalized) / self.temperature).softmax(dim=-1)


def _write_metadata(
    model_path: Path,
    labels: tuple[str, ...],
    parameter_count: int,
    checkpoint_path: Path,
    *,
    quantization: str,
) -> None:
    import onnx

    model = onnx.load(str(model_path))
    onnx.helper.set_model_props(
        model,
        {
            "attune.schema_version": "1.0",
            "attune.model": "cadence-affect-student-v0.13",
            "attune.labels": json.dumps(labels),
            "attune.parameter_count": str(parameter_count),
            "attune.quantization": quantization,
            "attune.sample_rate_hz": "16000",
            "attune.source_checkpoint_sha256": file_sha256(checkpoint_path),
            "attune.export_method": "padding-aware-legacy-trace-v1",
        },
    )
    onnx.save(model, str(model_path))


def export_truncated_affect_onnx(
    teacher_path: Path,
    checkpoint_path: Path,
    destination: Path,
) -> dict[str, object]:
    predictor = TruncatedEmotion2VecPredictor(
        teacher_path,
        checkpoint_path,
        device="cpu",
        batch_size=2,
    )
    frontend = predictor.model.modality_encoders["AUDIO"]
    frontend.convert_padding_mask = MethodType(_convert_padding_mask, frontend)
    export_model = _ExportModel(predictor).eval()

    generator = torch.Generator().manual_seed(42)
    samples = torch.randn((2, 48_000), generator=generator)
    samples[1, 40_000:] = 0.0
    padding_mask = torch.zeros_like(samples, dtype=torch.bool)
    padding_mask[1, 40_000:] = True
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        (samples, padding_mask),
        destination,
        input_names=["samples", "padding_mask"],
        output_names=["affect_probabilities"],
        opset_version=18,
        dynamo=False,
        dynamic_axes={
            "samples": {0: "batch", 1: "samples"},
            "padding_mask": {0: "batch", 1: "samples"},
            "affect_probabilities": {0: "batch"},
        },
        external_data=False,
        do_constant_folding=True,
    )
    _write_metadata(
        destination,
        predictor.labels,
        predictor.parameter_count,
        checkpoint_path,
        quantization="fp32",
    )
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
        "parameter_count": predictor.parameter_count,
    }


def quantize_truncated_affect_onnx(
    source: Path,
    destination: Path,
    checkpoint_path: Path,
) -> dict[str, object]:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(source),
        str(destination),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    source_predictor = OnnxTruncatedEmotion2VecPredictor(source)
    _write_metadata(
        destination,
        source_predictor.labels,
        source_predictor.parameter_count,
        checkpoint_path,
        quantization="int8",
    )
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
        "quantized_operator_types": ["MatMul", "Gemm"],
    }


def validate_truncated_affect_onnx(
    teacher_path: Path,
    checkpoint_path: Path,
    model_path: Path,
    audio_paths: list[Path],
) -> dict[str, float | int]:
    if len(audio_paths) < 2:
        raise ValueError("at least two validation clips are required")
    payloads = [path.read_bytes() for path in audio_paths]
    reference = TruncatedEmotion2VecPredictor(
        teacher_path,
        checkpoint_path,
        device="cpu",
        batch_size=1,
    ).predict_probabilities(payloads)
    candidate = OnnxTruncatedEmotion2VecPredictor(
        model_path,
        batch_size=1,
    ).predict_probabilities(payloads)
    difference = np.abs(candidate - reference)
    return {
        "clips": len(payloads),
        "top_label_agreement": float(
            (candidate.argmax(axis=-1) == reference.argmax(axis=-1)).mean()
        ),
        "mean_l1_probability_difference": float(difference.sum(axis=-1).mean()),
        "maximum_probability_difference": float(difference.max()),
    }


def export_truncated_affect_torchscript(
    teacher_path: Path,
    checkpoint_path: Path,
    destination: Path,
) -> dict[str, object]:
    predictor = TruncatedEmotion2VecPredictor(
        teacher_path,
        checkpoint_path,
        device="cpu",
        batch_size=2,
        quantization="int8",
    )
    frontend = predictor.model.modality_encoders["AUDIO"]
    frontend.convert_padding_mask = MethodType(_convert_padding_mask, frontend)
    export_model = _ExportModel(predictor).eval()

    generator = torch.Generator().manual_seed(42)
    samples = torch.randn((2, 48_000), generator=generator)
    samples[1, 40_000:] = 0.0
    padding_mask = torch.zeros_like(samples, dtype=torch.bool)
    padding_mask[1, 40_000:] = True
    with torch.no_grad():
        traced = torch.jit.trace(
            export_model,
            (samples, padding_mask),
            strict=False,
            check_trace=False,
        )
    metadata = {
        "schema_version": "1.0",
        "model": "cadence-affect-student-v0.13",
        "labels": predictor.labels,
        "parameter_count": predictor.parameter_count,
        "quantization": "int8",
        "quantization_engine": torch.backends.quantized.engine,
        "sample_rate_hz": 16_000,
        "source_checkpoint_sha256": file_sha256(checkpoint_path),
        "export_method": "padding-aware-torchscript-trace-v1",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(
        traced,
        str(destination),
        _extra_files={"attune.json": json.dumps(metadata)},
    )
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
        "parameter_count": predictor.parameter_count,
        "quantization_engine": torch.backends.quantized.engine,
    }


def validate_truncated_affect_torchscript(
    teacher_path: Path,
    checkpoint_path: Path,
    model_path: Path,
    audio_paths: list[Path],
) -> dict[str, float | int]:
    if len(audio_paths) < 2:
        raise ValueError("at least two validation clips are required")
    payloads = [path.read_bytes() for path in audio_paths]
    reference = TruncatedEmotion2VecPredictor(
        teacher_path,
        checkpoint_path,
        device="cpu",
        batch_size=1,
        quantization="int8",
    ).predict_probabilities(payloads)
    candidate = TorchScriptTruncatedEmotion2VecPredictor(
        model_path,
        batch_size=1,
    ).predict_probabilities(payloads)
    difference = np.abs(candidate - reference)
    return {
        "clips": len(payloads),
        "top_label_agreement": float(
            (candidate.argmax(axis=-1) == reference.argmax(axis=-1)).mean()
        ),
        "mean_l1_probability_difference": float(difference.sum(axis=-1).mean()),
        "maximum_probability_difference": float(difference.max()),
    }
