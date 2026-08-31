#!/usr/bin/env python3
"""Run a local Attune ONNX/INT8 model and emit schema-v2 JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from attune.client import AttuneClient
from attune.inference.affect_fusion import (
    DEFAULT_ACOUSTIC_WEIGHT,
    DEFAULT_AFFECT_CONFIDENCE_THRESHOLD,
    AffectProbabilityCalibration,
    FusedAffectBackend,
)
from attune.inference.emotion2vec_student import (
    OnnxTruncatedEmotion2VecPredictor,
    TorchScriptTruncatedEmotion2VecPredictor,
    TruncatedEmotion2VecPredictor,
)
from attune.inference.onnx_backend import OnnxAttuneBackend
from attune.schema.xml_v2 import render_xml_v2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sensevoice-path", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--probe-head",
        type=Path,
        action="append",
        default=[],
        help="Calibrated NumPy probe artifact; may be supplied more than once",
    )
    parser.add_argument("--quantization", choices=("fp32", "fp16", "int8"), default="int8")
    parser.add_argument("--emotion2vec-path", type=Path)
    parser.add_argument("--affect-student", type=Path)
    parser.add_argument("--affect-onnx", type=Path)
    parser.add_argument("--affect-torchscript", type=Path)
    parser.add_argument(
        "--affect-calibration",
        type=Path,
        help="Post-quantization calibration for a portable affect artifact",
    )
    parser.add_argument("--affect-weight", type=float, default=DEFAULT_ACOUSTIC_WEIGHT)
    parser.add_argument(
        "--affect-confidence-threshold",
        type=float,
        default=DEFAULT_AFFECT_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument("--affect-device", default="auto")
    parser.add_argument("--affect-quantization", choices=("fp32", "int8"), default="fp32")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--xml", action="store_true")
    arguments = parser.parse_args()
    if bool(arguments.emotion2vec_path) != bool(arguments.affect_student):
        parser.error("--emotion2vec-path and --affect-student must be supplied together")
    affect_artifacts = [
        arguments.affect_student,
        arguments.affect_onnx,
        arguments.affect_torchscript,
    ]
    if sum(artifact is not None for artifact in affect_artifacts) > 1:
        parser.error("affect student, ONNX, and TorchScript artifacts are mutually exclusive")
    if arguments.affect_calibration is not None and not (
        arguments.affect_onnx or arguments.affect_torchscript
    ):
        parser.error("--affect-calibration requires --affect-onnx or --affect-torchscript")
    backend = OnnxAttuneBackend.from_local_assets(
        arguments.model,
        arguments.sensevoice_path,
        arguments.calibration,
        quantization=arguments.quantization,
        probe_artifacts=tuple(arguments.probe_head),
    )
    if arguments.affect_torchscript is not None:
        predictor = TorchScriptTruncatedEmotion2VecPredictor(arguments.affect_torchscript)
        probability_calibration = (
            AffectProbabilityCalibration.from_file(
                arguments.affect_calibration,
                model_sha256=predictor.artifact_sha256,
            )
            if arguments.affect_calibration is not None
            else None
        )
        backend = FusedAffectBackend(
            backend,
            predictor,
            acoustic_weight=arguments.affect_weight,
            confidence_threshold=arguments.affect_confidence_threshold,
            probability_calibration=probability_calibration,
        )
    elif arguments.affect_onnx is not None:
        predictor = OnnxTruncatedEmotion2VecPredictor(arguments.affect_onnx)
        probability_calibration = (
            AffectProbabilityCalibration.from_file(
                arguments.affect_calibration,
                model_sha256=predictor.artifact_sha256,
            )
            if arguments.affect_calibration is not None
            else None
        )
        backend = FusedAffectBackend(
            backend,
            predictor,
            acoustic_weight=arguments.affect_weight,
            confidence_threshold=arguments.affect_confidence_threshold,
            probability_calibration=probability_calibration,
        )
    elif arguments.affect_student is not None:
        predictor = TruncatedEmotion2VecPredictor(
            arguments.emotion2vec_path,
            arguments.affect_student,
            device=arguments.affect_device,
            quantization=arguments.affect_quantization,
        )
        backend = FusedAffectBackend(
            backend,
            predictor,
            acoustic_weight=arguments.affect_weight,
            confidence_threshold=arguments.affect_confidence_threshold,
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
