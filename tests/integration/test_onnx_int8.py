from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from attune.inference.export import export_onnx, validate_onnx_parity
from attune.inference.quantization import quantize_dynamic_int8, validate_quantized_session
from attune.models.joint import AttuneJointModel


class TinyCTC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ctc_lo = nn.Linear(16, 12)


class TinySenseVoice(nn.Module):
    encoder_output_size = 16

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.encoders = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])
        self.encoder.after_norm = nn.LayerNorm(16)
        self.ctc = TinyCTC()
        self.input = nn.Linear(8, 16)

    def encode(self, speech, lengths, rich_tokens):
        del rich_tokens
        acoustic = self.input(speech)
        queries = torch.zeros(speech.shape[0], 4, 16, device=speech.device)
        return torch.cat((queries, acoustic), dim=1), lengths + 4


@pytest.mark.integration
def test_onnx_export_parity_and_int8_structure(tmp_path: Path) -> None:
    model = AttuneJointModel(TinySenseVoice(), hidden_size=8, affect_embedding_size=4)
    fp_model = tmp_path / "attune.onnx"
    int8_model = tmp_path / "attune-int8.onnx"

    export_onnx(
        model,
        fp_model,
        feature_size=8,
        sample_frames=20,
        include_probe_embedding=True,
    )
    parity = validate_onnx_parity(
        model,
        fp_model,
        sample=torch.randn(2, 21, 8),
        absolute_tolerance=2e-4,
        include_probe_embedding=True,
    )
    report = quantize_dynamic_int8(
        fp_model,
        int8_model,
        nodes_to_exclude=("/model/style_head/Gemm",),
    )
    validate_quantized_session(int8_model)

    assert parity.outputs_checked == 11
    assert report["method"] == "onnxruntime_dynamic_qint8_per_channel"
    assert report["materialized_aliased_gemm_weights"] == [
        "/model/style_projection/style_projection.0/Gemm",
        "/model/style_projection/style_projection.3/Gemm",
    ]
    assert report["nodes_excluded_from_int8"] == ["/model/style_head/Gemm"]
    assert int8_model.is_file()
