"""FastAPI batch and pseudo-streaming service for Attune v2."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from attune.inference.backend import InferenceBackend, set_output_status
from attune.inference.streaming import StreamingBuffer, StreamingConfig
from attune.service.auth import BasicAuthCredentials, BasicAuthMiddleware
from attune.service.ui import INDEX_HTML

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, PlainTextResponse
except ImportError:  # pragma: no cover - exercised by the dependency error below
    FastAPI = None  # type: ignore[assignment,misc]

MAX_UPLOAD_BYTES = 2 * 1024 * 1024


@dataclass
class ServiceMetrics:
    counters: Counter[str] = field(default_factory=Counter)
    processing_seconds: float = 0.0
    audio_seconds: float = 0.0

    def prometheus(self) -> str:
        lines = [
            f"attune_requests_total {self.counters['requests']}",
            f"attune_errors_total {self.counters['errors']}",
            f"attune_stream_revisions_total {self.counters['stream_revisions']}",
            f"attune_processing_seconds_total {self.processing_seconds:.9f}",
            f"attune_audio_seconds_total {self.audio_seconds:.9f}",
        ]
        return "\n".join(lines) + "\n"


def create_app(
    backend: InferenceBackend,
    *,
    streaming: StreamingConfig | None = None,
    credentials: BasicAuthCredentials | None = None,
) -> Any:
    if FastAPI is None:
        raise RuntimeError("install the serving extra to run the Attune API")

    app = FastAPI(title="Attune Cadence", version="0.1.0")
    if credentials is not None:
        app.add_middleware(BasicAuthMiddleware, credentials=credentials)
    metrics = ServiceMetrics()
    stream_config = streaming or StreamingConfig()

    @app.get("/", response_class=HTMLResponse)
    def browser_interface() -> str:
        return INDEX_HTML

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok", "backend": backend.name, "schema_version": "2.0"}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics_endpoint() -> str:
        return metrics.prometheus()

    @app.post("/v1/analyse")
    async def analyse(audio: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
        started = time.perf_counter()
        audio_bytes = await audio.read(MAX_UPLOAD_BYTES + 1)
        if not audio_bytes:
            metrics.counters["errors"] += 1
            raise HTTPException(status_code=400, detail="audio upload is empty")
        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            metrics.counters["errors"] += 1
            raise HTTPException(status_code=413, detail="audio upload exceeds 2 MiB")
        try:
            analysis = await asyncio.to_thread(backend.analyse_wav, audio_bytes)
        except (ValueError, OSError) as error:
            metrics.counters["errors"] += 1
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            metrics.counters["errors"] += 1
            raise HTTPException(status_code=500, detail="model inference failed") from error
        elapsed = time.perf_counter() - started
        duration = analysis.audio.duration_ms / 1000
        metrics.counters["requests"] += 1
        metrics.processing_seconds += elapsed
        metrics.audio_seconds += duration
        return {
            "request_id": f"req_{uuid.uuid4().hex}",
            "result": analysis.model_dump(mode="json"),
            "timing": {
                "audio_duration_ms": analysis.audio.duration_ms,
                "processing_ms": round(elapsed * 1000),
                "real_time_factor": elapsed / duration,
            },
        }

    @app.websocket("/v1/stream")
    async def stream_audio(websocket: WebSocket) -> None:
        await websocket.accept()
        buffer: StreamingBuffer | None = None
        committed = False
        session_id = f"session_{uuid.uuid4().hex}"
        try:
            while True:
                message = json.loads(await websocket.receive_text())
                event_type = message.get("type")
                if event_type == "session.start":
                    if buffer is not None:
                        raise ValueError("session has already started")
                    requested_rate = int(message.get("sample_rate_hz", 16_000))
                    requested_channels = int(message.get("channels", 1))
                    if requested_rate != 16_000 or requested_channels != 1:
                        raise ValueError("streaming requires 16 kHz mono PCM16")
                    buffer = StreamingBuffer(stream_config)
                    await websocket.send_json({"type": "session.started", "session_id": session_id})
                elif event_type == "audio.chunk":
                    if buffer is None:
                        raise ValueError("session.start is required before audio.chunk")
                    if committed:
                        raise ValueError("committed streaming results are immutable")
                    if buffer.append_base64(str(message.get("data", ""))):
                        started = time.perf_counter()
                        analysis = await asyncio.to_thread(
                            backend.analyse_wav, buffer.window_wav_bytes()
                        )
                        metrics.processing_seconds += time.perf_counter() - started
                        metrics.audio_seconds += analysis.audio.duration_ms / 1000
                        analysis = set_output_status(analysis, "provisional")
                        metrics.counters["stream_revisions"] += 1
                        await websocket.send_json(
                            {
                                "type": "result.provisional",
                                "revision": buffer.revision,
                                "window_start_ms": buffer.window_start_ms,
                                "result": analysis.model_dump(mode="json"),
                            }
                        )
                elif event_type == "audio.end":
                    if buffer is None or not buffer.pcm:
                        raise ValueError("audio.end requires non-empty audio")
                    if committed:
                        raise ValueError("committed streaming results are immutable")
                    started = time.perf_counter()
                    analysis = await asyncio.to_thread(backend.analyse_wav, buffer.wav_bytes())
                    metrics.processing_seconds += time.perf_counter() - started
                    metrics.audio_seconds += analysis.audio.duration_ms / 1000
                    analysis = set_output_status(analysis, "committed")
                    metrics.counters["requests"] += 1
                    committed = True
                    await websocket.send_json(
                        {"type": "result.committed", "result": analysis.model_dump(mode="json")}
                    )
                elif event_type == "session.close":
                    await websocket.close(code=1000)
                    return
                else:
                    raise ValueError(f"unknown client event: {event_type}")
        except WebSocketDisconnect:
            return
        except (ValueError, json.JSONDecodeError) as error:
            metrics.counters["errors"] += 1
            await websocket.send_json({"type": "error", "message": str(error)})
            await websocket.close(code=1003)
        except RuntimeError:
            metrics.counters["errors"] += 1
            await websocket.send_json({"type": "error", "message": "model inference failed"})
            await websocket.close(code=1011)

    app.state.attune_metrics = metrics
    return app
