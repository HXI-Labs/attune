#!/usr/bin/env python3
"""Serve a local Attune ONNX model with batch and pseudo-streaming APIs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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
from attune.service.app import create_app
from attune.service.auth import BasicAuthCredentials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--basic-auth-username",
        default=os.getenv("ATTUNE_BASIC_AUTH_USERNAME"),
        help="Protect every route with Basic Authentication.",
    )
    parser.add_argument(
        "--basic-auth-password",
        default=os.getenv("ATTUNE_BASIC_AUTH_PASSWORD"),
        help="Basic Authentication password. Never commit credentials.",
    )
    arguments = parser.parse_args()
    if bool(arguments.basic_auth_username) != bool(arguments.basic_auth_password):
        parser.error("Basic Authentication username and password must be configured together")
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
    credentials = (
        BasicAuthCredentials(arguments.basic_auth_username, arguments.basic_auth_password)
        if arguments.basic_auth_username
        else None
    )
    app = create_app(backend, credentials=credentials)
    import uvicorn

    uvicorn.run(app, host=arguments.host, port=arguments.port)


if __name__ == "__main__":
    main()
