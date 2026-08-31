#!/usr/bin/env python3
"""Compare an ONNX probe output with the frozen-encoder training cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from attune.inference.export import PROBE_EMBEDDING_OUTPUT
from attune.inference.onnx_backend import LocalSenseVoiceFrontend
from attune.integrity import file_digest
from attune.models.sensevoice_probe import SENSEVOICE_EMBEDDING, SENSEVOICE_EN_EMBEDDING


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sensevoice-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--query-language", choices=("auto", "en"), default="auto")
    parser.add_argument("--clips", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def cached_embedding_path(
    cache: Path, model_sha256: str, audio_path: Path, query_language: str
) -> Path:
    digest = hashlib.sha256()
    embedding_name = SENSEVOICE_EN_EMBEDDING if query_language == "en" else SENSEVOICE_EMBEDDING
    digest.update(embedding_name.encode())
    digest.update(f"language-query={query_language}".encode())
    digest.update(b"frontend-dither=0")
    digest.update(b"direct-encoder-v1")
    digest.update(model_sha256.encode())
    digest.update(file_digest(audio_path).encode())
    return cache / f"{digest.hexdigest()}.pt"


def main() -> None:
    arguments = parse_args()
    rows = [
        json.loads(line) for line in arguments.manifest.read_text().splitlines() if line.strip()
    ][: arguments.clips]
    if not rows:
        raise SystemExit("error: parity manifest is empty")
    frontend = LocalSenseVoiceFrontend(arguments.sensevoice_path)
    session = ort.InferenceSession(str(arguments.model), providers=["CPUExecutionProvider"])
    available = {output.name for output in session.get_outputs()}
    if PROBE_EMBEDDING_OUTPUT not in available:
        raise SystemExit("error: model does not expose probe_embedding")
    sensevoice_sha256 = file_digest(arguments.sensevoice_path / "model.pt")
    records = []
    for row in rows:
        audio_path = arguments.audio_root / row["cache_path"]
        expected_path = cached_embedding_path(
            arguments.embedding_cache,
            sensevoice_sha256,
            audio_path,
            arguments.query_language,
        )
        if not expected_path.is_file():
            raise SystemExit(f"error: cached training embedding is missing: {expected_path}")
        features = frontend(audio_path.read_bytes())
        actual = session.run(
            [PROBE_EMBEDDING_OUTPUT],
            {
                "speech": features[np.newaxis].astype(np.float32),
                "speech_lengths": np.asarray([features.shape[0]], dtype=np.int64),
            },
        )[0][0]
        expected = torch.load(expected_path, map_location="cpu", weights_only=True).numpy()
        error = np.abs(expected - actual)
        records.append(
            {
                "clip_id": row["clip_id"],
                "maximum_absolute_error": float(error.max()),
                "mean_absolute_error": float(error.mean()),
            }
        )
    maximum = max(record["maximum_absolute_error"] for record in records)
    report = {
        "schema_version": "1.0",
        "passed": maximum <= arguments.tolerance,
        "model_sha256": file_digest(arguments.model),
        "sensevoice_sha256": sensevoice_sha256,
        "clips": len(records),
        "tolerance": arguments.tolerance,
        "maximum_absolute_error": maximum,
        "mean_absolute_error": sum(record["mean_absolute_error"] for record in records)
        / len(records),
        "records": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("error: probe embedding parity failed")


if __name__ == "__main__":
    main()
