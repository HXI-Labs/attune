#!/usr/bin/env python3
"""Run a local Attune ONNX/INT8 model and emit schema-v2 JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from attune.client import AttuneClient
from attune.inference.onnx_backend import OnnxAttuneBackend
from attune.schema.xml_v2 import render_xml_v2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sensevoice-path", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--quantization", choices=("fp32", "fp16", "int8"), default="int8")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--xml", action="store_true")
    arguments = parser.parse_args()
    backend = OnnxAttuneBackend.from_local_assets(
        arguments.model,
        arguments.sensevoice_path,
        arguments.calibration,
        quantization=arguments.quantization,
    )
    client = AttuneClient(backend)
    if len(arguments.audio) > 1 and arguments.output_dir is None:
        parser.error("--output-dir is required for multiple audio files")
    if arguments.output_dir:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for audio_path in arguments.audio:
        result = client.analyse_file(audio_path)
        if arguments.output_dir:
            target = arguments.output_dir / f"{audio_path.stem}.attune.json"
            target.write_text(result.model_dump_json(indent=2) + "\n")
            if arguments.xml:
                target.with_suffix(".xml").write_text(render_xml_v2(result) + "\n")
        else:
            print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
