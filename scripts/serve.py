#!/usr/bin/env python3
"""Serve a local Attune ONNX model with batch and pseudo-streaming APIs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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
    backend = OnnxAttuneBackend.from_local_assets(
        arguments.model,
        arguments.sensevoice_path,
        arguments.calibration,
        quantization=arguments.quantization,
        probe_artifacts=tuple(arguments.probe_head),
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
