#!/usr/bin/env python3
"""Create joint-training fbank features from normalized public-data rows."""

from __future__ import annotations

import argparse
from pathlib import Path

from attune.inference.onnx_backend import LocalSenseVoiceFrontend
from attune.training.prepare import load_source_rows, prepare_joint_features, write_source_template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--sensevoice-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived/joint-v0.1"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/joint-v0.1.jsonl"))
    parser.add_argument("--write-template", type=Path)
    arguments = parser.parse_args()
    if arguments.write_template:
        write_source_template(arguments.write_template)
        return
    if arguments.source is None or arguments.sensevoice_path is None:
        parser.error("--source and --sensevoice-path are required unless --write-template is used")
    frontend = LocalSenseVoiceFrontend(arguments.sensevoice_path)
    rows = load_source_rows(arguments.source)
    prepare_joint_features(
        rows,
        frontend=frontend,
        output_dir=arguments.output_dir.resolve(),
        manifest_path=arguments.manifest.resolve(),
    )


if __name__ == "__main__":
    main()
