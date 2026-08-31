#!/usr/bin/env python3
"""Serve a local Attune ONNX model with batch and pseudo-streaming APIs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from attune.inference.affect_fusion import (
    DEFAULT_ACOUSTIC_WEIGHT,
    DEFAULT_AFFECT_CONFIDENCE_THRESHOLD,
    FusedAffectBackend,
)
from attune.inference.emotion2vec_student import TruncatedEmotion2VecPredictor
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
    parser.add_argument("--affect-weight", type=float, default=DEFAULT_ACOUSTIC_WEIGHT)
    parser.add_argument(
        "--affect-confidence-threshold",
        type=float,
        default=DEFAULT_AFFECT_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument("--affect-device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--demo-username",
        default=os.getenv("ATTUNE_DEMO_USERNAME"),
        help="Protect every route with Basic Auth; can use ATTUNE_DEMO_USERNAME.",
    )
    parser.add_argument(
        "--demo-password",
        default=os.getenv("ATTUNE_DEMO_PASSWORD"),
        help="Can use ATTUNE_DEMO_PASSWORD. Never commit demo credentials.",
    )
    arguments = parser.parse_args()
    if bool(arguments.demo_username) != bool(arguments.demo_password):
        parser.error("demo username and password must be configured together")
    if bool(arguments.emotion2vec_path) != bool(arguments.affect_student):
        parser.error("--emotion2vec-path and --affect-student must be supplied together")
    backend = OnnxAttuneBackend.from_local_assets(
        arguments.model,
        arguments.sensevoice_path,
        arguments.calibration,
        quantization=arguments.quantization,
        probe_artifacts=tuple(arguments.probe_head),
    )
    if arguments.affect_student is not None:
        predictor = TruncatedEmotion2VecPredictor(
            arguments.emotion2vec_path,
            arguments.affect_student,
            device=arguments.affect_device,
        )
        backend = FusedAffectBackend(
            backend,
            predictor,
            acoustic_weight=arguments.affect_weight,
            confidence_threshold=arguments.affect_confidence_threshold,
        )
    credentials = (
        BasicAuthCredentials(arguments.demo_username, arguments.demo_password)
        if arguments.demo_username
        else None
    )
    app = create_app(backend, credentials=credentials)
    import uvicorn

    uvicorn.run(app, host=arguments.host, port=arguments.port)


if __name__ == "__main__":
    main()
