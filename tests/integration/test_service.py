from __future__ import annotations

import base64
import io
import wave
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from attune.inference.onnx_backend import inspect_wav
from attune.inference.streaming import StreamingBuffer, StreamingConfig
from attune.schema.migration import migrate_v1_to_v2
from attune.schema.output import AttuneOutput
from attune.service.app import create_app
from attune.service.auth import BasicAuthCredentials


class FakeBackend:
    name = "fixture"

    def __init__(self, payload: dict) -> None:
        self.output = migrate_v1_to_v2(AttuneOutput.model_validate(payload))

    def analyse_wav(self, audio: bytes):
        assert audio[:4] == b"RIFF"
        return self.output


class FailingBackend:
    name = "failing-fixture"

    def analyse_wav(self, audio: bytes):
        raise RuntimeError("private backend detail")


def _wav() -> bytes:
    target = io.BytesIO()
    with wave.open(target, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 400)
    return target.getvalue()


def test_batch_api_returns_schema_v2_and_metrics(example_payload: dict) -> None:
    client = TestClient(create_app(FakeBackend(deepcopy(example_payload))))
    response = client.post("/v1/analyse", files={"audio": ("clip.wav", _wav(), "audio/wav")})

    assert response.status_code == 200
    assert response.json()["result"]["schema_version"] == "2.0"
    assert response.json()["request_id"].startswith("req_")
    assert "attune_requests_total 1" in client.get("/metrics").text


def test_batch_api_counts_inference_failures_without_exposing_details() -> None:
    client = TestClient(create_app(FailingBackend()))

    response = client.post("/v1/analyse", files={"audio": ("clip.wav", _wav(), "audio/wav")})

    assert response.status_code == 500
    assert response.json() == {"detail": "model inference failed"}
    assert "attune_errors_total 1" in client.get("/metrics").text


def test_batch_api_rejects_empty_audio_and_counts_the_error(example_payload: dict) -> None:
    client = TestClient(create_app(FakeBackend(deepcopy(example_payload))))

    response = client.post("/v1/analyse", files={"audio": ("clip.wav", b"", "audio/wav")})

    assert response.status_code == 400
    assert response.json() == {"detail": "audio upload is empty"}
    assert "attune_errors_total 1" in client.get("/metrics").text


def test_batch_api_rejects_oversized_upload(example_payload: dict) -> None:
    client = TestClient(create_app(FakeBackend(deepcopy(example_payload))))

    response = client.post(
        "/v1/analyse",
        files={"audio": ("clip.wav", b"0" * (2 * 1024 * 1024 + 1), "audio/wav")},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "audio upload exceeds 2 MiB"}


def test_browser_interface_is_served(example_payload: dict) -> None:
    client = TestClient(create_app(FakeBackend(deepcopy(example_payload))))

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Attune Cadence" in response.text
    assert "Things to try" in response.text
    assert "[laughing speech; perceived joy]" in response.text
    assert "[crying speech; perceived distress]" in response.text
    assert "illustrative target outputs" in response.text
    assert "/v1/analyse" in response.text


def test_demo_credentials_protect_every_http_route(example_payload: dict) -> None:
    app = create_app(
        FakeBackend(deepcopy(example_payload)),
        credentials=BasicAuthCredentials("cadence", "test-secret"),
    )
    client = TestClient(app)

    denied = client.get("/healthz")
    allowed = client.get("/healthz", auth=("cadence", "test-secret"))

    assert denied.status_code == 401
    assert denied.headers["www-authenticate"].startswith("Basic")
    assert allowed.status_code == 200


def test_demo_credentials_reject_unauthenticated_websocket(example_payload: dict) -> None:
    app = create_app(
        FakeBackend(deepcopy(example_payload)),
        credentials=BasicAuthCredentials("cadence", "test-secret"),
    )

    with (
        pytest.raises(WebSocketDisconnect) as denied,
        TestClient(app).websocket_connect("/v1/stream"),
    ):
        pass

    assert denied.value.code == 1008


def test_streaming_protocol_emits_provisional_then_committed(example_payload: dict) -> None:
    config = StreamingConfig(window_ms=10, overlap_ms=0)
    client = TestClient(create_app(FakeBackend(deepcopy(example_payload)), streaming=config))
    pcm = base64.b64encode(b"\x00\x00" * 200).decode()

    with client.websocket_connect("/v1/stream") as socket:
        socket.send_json({"type": "session.start", "sample_rate_hz": 16_000, "channels": 1})
        assert socket.receive_json()["type"] == "session.started"
        socket.send_json({"type": "audio.chunk", "data": pcm})
        assert socket.receive_json()["type"] == "result.provisional"
        socket.send_json({"type": "audio.end"})
        committed = socket.receive_json()
        assert committed["type"] == "result.committed"
        assert committed["result"]["events"][0]["status"] == "committed"


def test_streaming_rejects_audio_after_commit(example_payload: dict) -> None:
    client = TestClient(create_app(FakeBackend(deepcopy(example_payload))))
    pcm = base64.b64encode(b"\x00\x00" * 200).decode()

    with client.websocket_connect("/v1/stream") as socket:
        socket.send_json({"type": "session.start"})
        assert socket.receive_json()["type"] == "session.started"
        socket.send_json({"type": "audio.chunk", "data": pcm})
        socket.send_json({"type": "audio.end"})
        assert socket.receive_json()["type"] == "result.committed"
        socket.send_json({"type": "audio.chunk", "data": pcm})
        error = socket.receive_json()

    assert error == {
        "type": "error",
        "message": "committed streaming results are immutable",
    }


def test_streaming_reports_inference_failure_without_exposing_details() -> None:
    client = TestClient(
        create_app(FailingBackend(), streaming=StreamingConfig(window_ms=10, overlap_ms=0))
    )
    pcm = base64.b64encode(b"\x00\x00" * 200).decode()

    with client.websocket_connect("/v1/stream") as socket:
        socket.send_json({"type": "session.start"})
        assert socket.receive_json()["type"] == "session.started"
        socket.send_json({"type": "audio.chunk", "data": pcm})
        assert socket.receive_json() == {"type": "error", "message": "model inference failed"}
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()

    assert closed.value.code == 1011


def test_streaming_buffer_returns_bounded_rolling_window() -> None:
    config = StreamingConfig(window_ms=1000, overlap_ms=250)
    buffer = StreamingBuffer(config)
    one_second = b"\x00\x00" * 16_000
    buffer.pcm.extend(one_second * 2)

    metadata = inspect_wav(buffer.window_wav_bytes())

    assert metadata.duration_ms == 1000
    assert buffer.window_start_ms == 1000


def test_streaming_configuration_rejects_invalid_windows() -> None:
    with pytest.raises(ValueError, match="overlap must be shorter"):
        StreamingConfig(window_ms=1000, overlap_ms=1000)
    with pytest.raises(ValueError, match="window cannot exceed"):
        StreamingConfig(window_ms=2000, maximum_duration_ms=1000)


def test_streaming_buffer_rejects_chunk_larger_than_remaining_capacity() -> None:
    buffer = StreamingBuffer(StreamingConfig(window_ms=10, overlap_ms=0, maximum_duration_ms=10))
    oversized = base64.b64encode(b"\x00" * 321).decode()

    with pytest.raises(ValueError, match="maximum duration"):
        buffer.append_base64(oversized)
