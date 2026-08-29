#!/usr/bin/env python3
"""Run the unified heads once against the pinned real SenseVoice checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import wave
from pathlib import Path

import torch

from attune.inference.export import sensevoice_feature_size
from attune.inference.onnx_backend import LocalSenseVoiceFrontend
from attune.models.joint import AdaptationPolicy, AttuneJointModel, load_local_sensevoice


def _silence() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 16_000)
    return buffer.getvalue()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensevoice-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("research/real-checkpoint-smoke.json"))
    arguments = parser.parse_args()
    started = time.monotonic()
    frontend = LocalSenseVoiceFrontend(arguments.sensevoice_path)
    features = torch.from_numpy(frontend(_silence())).unsqueeze(0)
    del frontend
    backbone = load_local_sensevoice(str(arguments.sensevoice_path))
    model = AttuneJointModel(backbone).eval()
    expected_width = sensevoice_feature_size(model)
    if features.shape[-1] != expected_width:
        raise ValueError(
            f"frontend width {features.shape[-1]} does not match encoder {expected_width}"
        )
    with torch.inference_mode():
        output = model(features, torch.tensor([features.shape[1]]))
    summary = model.trainable_parameter_summary()
    if summary["total"] > 300_000_000:
        raise ValueError("real checkpoint and Attune heads exceed the parameter budget")
    model.set_adaptation_policy(AdaptationPolicy.UPPER_TWO)
    upper_two_summary = model.trainable_parameter_summary()
    model.set_adaptation_policy(AdaptationPolicy.FROZEN)
    payload = {
        "schema_version": "1.0",
        "checkpoint_sha256": _sha256(arguments.sensevoice_path / "model.pt"),
        "feature_shape": list(features.shape),
        "encoder_feature_size": expected_width,
        "acoustic_frames": int(output.acoustic_lengths[0]),
        "ctc_vocabulary_size": output.ctc_logits.shape[-1],
        "event_classes": output.event_logits.shape[-1],
        "style_classes": output.style_logits.shape[-1],
        "affect_classes": output.affect_logits.shape[-1],
        "parameter_summary": summary,
        "upper_two_parameter_summary": upper_two_summary,
        "elapsed_seconds": time.monotonic() - started,
        "passed": True,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
