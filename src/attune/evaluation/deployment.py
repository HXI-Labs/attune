"""End-to-end CPU deployment benchmark for a calibrated ONNX backend."""

from __future__ import annotations

import json
import time
import wave
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

from attune.schema.v2 import AttuneOutputV2
from attune.schema.xml_v2 import render_xml_v2


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values), percentile)) if values else 0.0


def deployment_audio_contract_error(path: Path) -> str | None:
    """Return why a WAV is outside the runtime contract, or ``None`` when valid."""
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getnchannels() != 1:
                return f"channels={audio.getnchannels()} (required 1)"
            if audio.getframerate() != 16_000:
                return f"sample_rate_hz={audio.getframerate()} (required 16000)"
            if audio.getsampwidth() != 2:
                return f"sample_width_bytes={audio.getsampwidth()} (required 2)"
            if audio.getcomptype() != "NONE":
                return f"compression={audio.getcomptype()} (required NONE)"
    except (FileNotFoundError, OSError, wave.Error) as error:
        return f"unreadable_wav: {error}"
    return None


def balanced_benchmark_rows(
    rows: list[dict[str, Any]], *, split: str, per_dataset: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("split") != split or not row.get("audio_path"):
            continue
        dataset = str(row["dataset_id"])
        if counts.get(dataset, 0) >= per_dataset:
            continue
        selected.append(row)
        counts[dataset] = counts.get(dataset, 0) + 1
    if not selected:
        raise ValueError(f"no audio rows available for split {split!r}")
    return selected


def benchmark_backend(backend: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies: list[float] = []
    audio_seconds = 0.0
    processing_seconds = 0.0
    json_valid = 0
    xml_valid = 0
    errors: list[dict[str, str]] = []
    for row in rows:
        audio = Path(row["audio_path"]).read_bytes()
        started = time.perf_counter()
        try:
            result = backend.analyse_wav(audio)
            elapsed = time.perf_counter() - started
            serialized = result.model_dump_json()
            AttuneOutputV2.model_validate_json(serialized)
            json.loads(serialized)
            json_valid += 1
            ET.fromstring(render_xml_v2(result))
            xml_valid += 1
        except Exception as error:  # benchmark failures are reported, not hidden
            elapsed = time.perf_counter() - started
            errors.append({"clip_id": str(row["clip_id"]), "error": str(error)})
        duration = float(row["duration_ms"]) / 1000.0
        audio_seconds += duration
        processing_seconds += elapsed
        latencies.append(elapsed)
    count = len(rows)
    return {
        "schema_version": "1.0",
        "clips": count,
        "audio_seconds": audio_seconds,
        "processing_seconds": processing_seconds,
        "cpu_real_time_factor": processing_seconds / audio_seconds,
        "latency_ms": {
            "p50": _percentile(latencies, 50) * 1000,
            "p95": _percentile(latencies, 95) * 1000,
            "p99": _percentile(latencies, 99) * 1000,
        },
        "json_validity_rate": json_valid / count,
        "xml_validity_rate": xml_valid / count,
        "committed_retraction_rate": 0.0,
        "commit_contract": "one final committed result; committed results are immutable",
        "errors": errors,
    }
