#!/usr/bin/env python3
"""Quantize an exported Attune ONNX model to per-channel INT8."""

import argparse
import json
from pathlib import Path

from attune.inference.quantization import (
    quantize_dynamic_int8,
    validate_quantized_session,
    write_quantization_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-node",
        action="append",
        default=[],
        help="Exact ONNX node name to retain in floating point; repeat as needed",
    )
    parser.add_argument(
        "--exclude-node-prefix",
        action="append",
        default=[],
        help=(
            "ONNX node-name prefix to retain in floating point; repeat as needed. "
            "The resolved exact names are recorded in the quantization report."
        ),
    )
    arguments = parser.parse_args()
    excluded_nodes = list(arguments.exclude_node)
    if arguments.exclude_node_prefix:
        try:
            import onnx
        except ImportError as error:
            raise RuntimeError(
                "install the deployment extra to resolve ONNX node prefixes"
            ) from error
        graph = onnx.load(str(arguments.input), load_external_data=False).graph
        matched_prefixes: set[str] = set()
        for node in graph.node:
            for prefix in arguments.exclude_node_prefix:
                if node.name.startswith(prefix):
                    excluded_nodes.append(node.name)
                    matched_prefixes.add(prefix)
        unmatched = sorted(set(arguments.exclude_node_prefix) - matched_prefixes)
        if unmatched:
            parser.error(f"no ONNX nodes matched prefix(es): {', '.join(unmatched)}")
    excluded_nodes = list(dict.fromkeys(excluded_nodes))
    report = quantize_dynamic_int8(
        arguments.input,
        arguments.output,
        nodes_to_exclude=tuple(excluded_nodes),
    )
    validate_quantized_session(arguments.output)
    report_path = arguments.output.with_suffix(".quantization.json")
    write_quantization_report(report_path, report)
    print(json.dumps({"output": str(arguments.output), "report": str(report_path), **report}))


if __name__ == "__main__":
    main()
