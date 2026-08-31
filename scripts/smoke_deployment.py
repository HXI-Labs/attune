#!/usr/bin/env python3
"""Exercise the real calibrated backend through batch and streaming APIs."""

from __future__ import annotations

import argparse
import base64
import json
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from attune.inference.onnx_backend import OnnxAttuneBackend
from attune.inference.streaming import StreamingConfig
from attune.integrity import file_digest
from attune.schema.v2 import AttuneOutputV2
from attune.service.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sensevoice-path", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--probe-head", type=Path, action="append", default=[])
    parser.add_argument("--quantization", choices=("fp32", "fp16", "int8"), required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    backend = OnnxAttuneBackend.from_local_assets(
        arguments.model,
        arguments.sensevoice_path,
        arguments.calibration,
        quantization=arguments.quantization,
        probe_artifacts=tuple(arguments.probe_head),
    )
    streaming = StreamingConfig(window_ms=1_000, overlap_ms=250)
    client = TestClient(create_app(backend, streaming=streaming))
    audio_bytes = arguments.audio.read_bytes()

    health = client.get("/healthz")
    health.raise_for_status()
    batch = client.post(
        "/v1/analyse",
        files={"audio": (arguments.audio.name, audio_bytes, "audio/wav")},
    )
    batch.raise_for_status()
    batch_payload = batch.json()
    batch_result = AttuneOutputV2.model_validate(batch_payload["result"])

    with wave.open(str(arguments.audio), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16_000, 1, 2):
            raise ValueError("smoke audio must be 16 kHz mono PCM16 WAV")
        pcm = audio.readframes(audio.getnframes())

    provisional_results = []
    with client.websocket_connect("/v1/stream") as socket:
        socket.send_json({"type": "session.start", "sample_rate_hz": 16_000, "channels": 1})
        started = socket.receive_json()
        if started.get("type") != "session.started":
            raise RuntimeError(f"unexpected streaming start response: {started}")
        stride_bytes = int((streaming.window_ms - streaming.overlap_ms) * streaming.bytes_per_ms)
        threshold_bytes = int(streaming.window_ms * streaming.bytes_per_ms)
        last_emitted_bytes = 0
        for start in range(0, len(pcm), stride_bytes):
            socket.send_json(
                {
                    "type": "audio.chunk",
                    "data": base64.b64encode(pcm[start : start + stride_bytes]).decode(),
                }
            )
            accumulated = min(start + stride_bytes, len(pcm))
            if accumulated >= threshold_bytes and accumulated - last_emitted_bytes >= stride_bytes:
                message = socket.receive_json()
                if message.get("type") != "result.provisional":
                    raise RuntimeError(f"unexpected provisional response: {message}")
                AttuneOutputV2.model_validate(message["result"])
                provisional_results.append(message)
                last_emitted_bytes = accumulated
        socket.send_json({"type": "audio.end"})
        committed = socket.receive_json()
        if committed.get("type") != "result.committed":
            raise RuntimeError(f"unexpected committed response: {committed}")
        committed_result = AttuneOutputV2.model_validate(committed["result"])
        socket.send_json({"type": "session.close"})

    if not provisional_results:
        raise RuntimeError("stream produced no provisional result")
    if any(
        item.status != "committed" for item in committed_result.events + committed_result.styles
    ):
        raise RuntimeError("committed output contains a non-committed temporal item")

    metrics = client.get("/metrics")
    metrics.raise_for_status()
    report = {
        "schema_version": "1.0",
        "model_sha256": file_digest(arguments.model),
        "calibration_sha256": file_digest(arguments.calibration),
        "audio_sha256": file_digest(arguments.audio),
        "health": health.json(),
        "batch": {
            "status_code": batch.status_code,
            "output_schema_version": batch_result.schema_version,
            "transcript": batch_result.transcript.text,
            "processing_ms": batch_payload["timing"]["processing_ms"],
            "real_time_factor": batch_payload["timing"]["real_time_factor"],
        },
        "streaming": {
            "provisional_results": len(provisional_results),
            "last_revision": provisional_results[-1]["revision"],
            "committed_schema_version": committed_result.schema_version,
            "committed_transcript": committed_result.transcript.text,
            "committed_temporal_items": len(committed_result.events) + len(committed_result.styles),
        },
        "metrics": metrics.text,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
