"""INT8 quantization with structural checks."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


def _materialize_aliased_gemm_weights(source: Path, destination: Path) -> tuple[str, ...]:
    """Replace Identity-backed Gemm weights with distinct initializers.

    PyTorch may deduplicate equal projection weights during export and connect
    one of the Gemm nodes through ``Identity``. ONNX Runtime 1.29 corrupts the
    matrix shape when optimizing that pattern for dynamic quantization.
    """
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("install the deployment extra for INT8 quantization") from error

    model = onnx.load(str(source), load_external_data=True)
    graph = model.graph
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    producers = {output: node for node in graph.node for output in node.output}
    materialized_nodes: list[str] = []
    for node in graph.node:
        if node.op_type != "Gemm" or len(node.input) < 2 or node.input[1] in initializers:
            continue
        alias = producers.get(node.input[1])
        if alias is None or alias.op_type != "Identity" or alias.input[0] not in initializers:
            continue
        initializer = onnx.TensorProto()
        initializer.CopyFrom(initializers[alias.input[0]])
        initializer.name = f"{node.input[1]}__materialized_for_quantization"
        graph.initializer.append(initializer)
        node.input[1] = initializer.name
        materialized_nodes.append(node.name)
    if materialized_nodes:
        onnx.save(model, str(destination))
    return tuple(materialized_nodes)


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
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary_directory:
        prepared_source = Path(temporary_directory) / "quantization-source.onnx"
        materialized_nodes = _materialize_aliased_gemm_weights(source, prepared_source)
        quantize_dynamic(
            model_input=str(prepared_source if materialized_nodes else source),
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
        "materialized_aliased_gemm_weights": list(materialized_nodes),
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
