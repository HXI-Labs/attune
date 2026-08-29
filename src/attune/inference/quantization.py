"""INT8 quantization with structural checks."""

from __future__ import annotations

import json
from pathlib import Path


def quantize_dynamic_int8(
    source: Path,
    destination: Path,
    *,
    nodes_to_exclude: tuple[str, ...] = (),
) -> dict[str, object]:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as error:
        raise RuntimeError("install the deployment extra for INT8 quantization") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(source),
        model_output=str(destination),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        nodes_to_exclude=list(nodes_to_exclude),
    )
    source_size = source.stat().st_size
    destination_size = destination.stat().st_size
    return {
        "method": "onnxruntime_dynamic_qint8_per_channel",
        "source_bytes": source_size,
        "quantized_bytes": destination_size,
        "size_ratio": destination_size / source_size,
        "nodes_excluded_from_int8": list(nodes_to_exclude),
    }


def validate_quantized_session(path: Path) -> None:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("install the deployment extra for INT8 validation") from error
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if {item.name for item in session.get_inputs()} != {"speech", "speech_lengths"}:
        raise ValueError("quantized model input contract changed")


def write_quantization_report(path: Path, report: dict[str, object]) -> None:
    path.write_text(json.dumps({"schema_version": "1.0", **report}, indent=2) + "\n")
