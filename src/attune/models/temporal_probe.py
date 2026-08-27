"""Gated frame-level inference for the DCASE-overlap event labels."""

from __future__ import annotations

import importlib.util
import os
import time
import wave
from pathlib import Path
from typing import Any

from attune.models.probe_inference import ProbeAnnotation, ProbePrediction
from attune.models.sensevoice_probe import (
    SENSEVOICE_FRAME_EMBEDDING,
    FrozenSenseVoiceFrameEncoder,
)
from attune.schema.output import EventLabel

TEMPORAL_LABELS = {
    "laugh": EventLabel.LAUGH,
    "cough": EventLabel.COUGH,
    "throat_clear": EventLabel.THROAT_CLEAR,
}


class FrozenTemporalProbeHead:
    """Decode event spans from frozen SenseVoice frames after a held-out gate."""

    name = "dcase-frozen-frame-temporal-head"

    def __init__(
        self,
        *,
        checkpoint: Path,
        sensevoice_checkpoint: Path,
        frame_cache: Path,
    ) -> None:
        self.checkpoint = checkpoint
        self.sensevoice_checkpoint = sensevoice_checkpoint
        self.frame_cache = frame_cache
        self._extractor: FrozenSenseVoiceFrameEncoder | None = None
        self._head: Any | None = None
        self._payload: dict[str, Any] | None = None
        self._torch: Any | None = None

    def availability(self) -> tuple[bool, str | None]:
        if not self.checkpoint.is_file():
            return False, f"gitignored temporal checkpoint is missing: {self.checkpoint}"
        if os.environ.get("ATTUNE_SENSEVOICE_LICENSE_REVIEWED") != "1":
            return False, "SenseVoice licence review is required for temporal inference."
        if not (self.sensevoice_checkpoint / "model.pt").is_file():
            return False, "SenseVoiceSmall model.pt is missing for temporal inference."
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("funasr") is None:
            return False, "Temporal inference requires torch and funasr."
        return True, None

    def predict(self, audio_path: Path) -> ProbePrediction:
        started = time.perf_counter()
        extractor, head, payload, torch = self._load()
        frames = extractor(audio_path)
        normalized = (frames - payload["feature_mean"]) / payload["feature_scale"]
        with torch.inference_mode():
            probabilities = torch.sigmoid(head(normalized))
        duration_ms = _duration_ms(audio_path)
        annotations = _decode_annotations(
            probabilities,
            labels=payload["labels"],
            threshold=float(payload["threshold"]),
            first_frame_center_ms=extractor.first_frame_center_ms,
            frame_hop_ms=extractor.frame_hop_ms,
            duration_ms=duration_ms,
        )
        return ProbePrediction(
            annotations=tuple(annotations),
            elapsed_seconds=time.perf_counter() - started,
            diagnostics={
                "name": self.name,
                "embedding": SENSEVOICE_FRAME_EMBEDDING,
                "encoder_frozen": True,
                "span_scope": "frame",
                "frame_hop_ms_approx": extractor.frame_hop_ms,
                "threshold": float(payload["threshold"]),
                "gate": payload["gate"],
            },
        )

    def _load(self) -> tuple[FrozenSenseVoiceFrameEncoder, Any, dict[str, Any], Any]:
        if self._extractor is not None and self._head is not None and self._payload is not None:
            return self._extractor, self._head, self._payload, self._torch
        import torch

        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        required = {
            "head_state_dict",
            "feature_mean",
            "feature_scale",
            "labels",
            "threshold",
            "embedding",
            "encoder_frozen",
            "gate",
        }
        if not isinstance(payload, dict) or not required <= payload.keys():
            raise RuntimeError("temporal checkpoint has an unsupported payload")
        if payload["embedding"] != SENSEVOICE_FRAME_EMBEDDING:
            raise RuntimeError("temporal checkpoint uses the wrong frame embedding")
        if payload["encoder_frozen"] is not True:
            raise RuntimeError("temporal checkpoint does not attest a frozen encoder")
        labels = payload["labels"]
        if tuple(labels) != tuple(TEMPORAL_LABELS):
            raise RuntimeError("temporal checkpoint labels do not match the DCASE mapping")
        gate = payload["gate"]
        if not isinstance(gate, dict) or gate.get("passed") is not True:
            raise RuntimeError("temporal checkpoint did not pass the held-out wiring gate")
        if float(gate["margin_observed"]) < float(gate["margin_required"]):
            raise RuntimeError("temporal checkpoint margin is below its wiring requirement")

        head = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, len(labels)),
        )
        head.load_state_dict(payload["head_state_dict"])
        head.eval()
        extractor = FrozenSenseVoiceFrameEncoder(
            self.sensevoice_checkpoint,
            self.frame_cache,
            torch,
        )
        self._torch = torch
        self._payload = payload
        self._head = head
        self._extractor = extractor
        return extractor, head, payload, torch


def _decode_annotations(
    probabilities: Any,
    *,
    labels: list[str] | tuple[str, ...],
    threshold: float,
    first_frame_center_ms: float,
    frame_hop_ms: float,
    duration_ms: int,
) -> list[ProbeAnnotation]:
    annotations = []
    for label_index, source_label in enumerate(labels):
        active = probabilities[:, label_index] >= threshold
        start = None
        for index, enabled in enumerate([*active.tolist(), False]):
            if enabled and start is None:
                start = index
            elif not enabled and start is not None:
                start_ms = max(
                    0,
                    round(first_frame_center_ms + start * frame_hop_ms - frame_hop_ms / 2),
                )
                end_ms = min(
                    duration_ms,
                    round(first_frame_center_ms + index * frame_hop_ms - frame_hop_ms / 2),
                )
                confidence = float(probabilities[start:index, label_index].max())
                annotations.append(
                    ProbeAnnotation(
                        channel="event",
                        label=TEMPORAL_LABELS[source_label],
                        confidence=confidence,
                        start_ms=start_ms,
                        end_ms=end_ms,
                    )
                )
                start = None
    return annotations


def _duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as audio:
        return round(audio.getnframes() / audio.getframerate() * 1000)
