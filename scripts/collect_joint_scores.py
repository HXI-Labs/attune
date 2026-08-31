#!/usr/bin/env python3
"""Collect raw unified-model scores without selecting thresholds on test data."""

from __future__ import annotations

import argparse
from pathlib import Path

from attune.evaluation.joint import collect_onnx_scores
from attune.inference.onnx_backend import LocalSenseVoiceFrontend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "development", "sealed_test"), required=True)
    parser.add_argument(
        "--sensevoice-path",
        type=Path,
        help="Reviewed local checkpoint assets; required to decode ASR WER rows",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    decoder = (
        LocalSenseVoiceFrontend(arguments.sensevoice_path).decoder
        if arguments.sensevoice_path
        else None
    )
    collect_onnx_scores(
        manifest=arguments.manifest,
        split=arguments.split,
        model_path=arguments.model,
        output_path=arguments.output,
        transcript_decoder=decoder,
    )


if __name__ == "__main__":
    main()
