"""Inference wrappers for gitignored frozen-encoder linear probe heads."""

from __future__ import annotations

import importlib.util
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from attune.models.probe_abstention import (
    accepts,
    checkpoint_abstention,
    confidence_scores,
)
from attune.models.sensevoice_probe import SENSEVOICE_EMBEDDING, FrozenSenseVoiceEncoder
from attune.schema.output import EventLabel, StyleLabel

AnnotationChannel = Literal["event", "style"]

VOCALSOUND_LABEL_MAPPING: dict[str, tuple[AnnotationChannel, EventLabel | StyleLabel]] = {
    "laugh": ("event", EventLabel.LAUGH),
    "sigh": ("event", EventLabel.SIGH),
    "cough": ("event", EventLabel.COUGH),
    "throat_clear": ("event", EventLabel.THROAT_CLEAR),
    "sneeze": ("event", EventLabel.SNEEZE),
}

FSD50K_LABEL_MAPPING: dict[str, tuple[AnnotationChannel, EventLabel | StyleLabel]] = {
    "shout": ("style", StyleLabel.SHOUTING),
    "whisper": ("style", StyleLabel.WHISPERING),
    "sob": ("event", EventLabel.SOB),
    "scream": ("event", EventLabel.SCREAM),
}


@dataclass(frozen=True)
class ProbeAnnotation:
    """One utterance-level annotation emitted by a closed-set probe."""

    channel: AnnotationChannel
    label: EventLabel | StyleLabel
    confidence: float


@dataclass(frozen=True)
class ProbePrediction:
    """Probe annotations and bounded inference diagnostics."""

    annotations: tuple[ProbeAnnotation, ...]
    elapsed_seconds: float
    diagnostics: dict[str, Any]
    abstained: bool = False


class FrozenEncoderProvider:
    """Lazily share one immutable SenseVoice encoder across multiple heads."""

    def __init__(self, checkpoint: Path, cache_dir: Path) -> None:
        self.checkpoint = checkpoint
        self.cache_dir = cache_dir
        self._extractor: FrozenSenseVoiceEncoder | None = None
        self._torch: Any | None = None

    def availability(self) -> tuple[bool, str | None]:
        if os.environ.get("ATTUNE_SENSEVOICE_LICENSE_REVIEWED") != "1":
            return False, "SenseVoice licence review is required for probe inference."
        if not (self.checkpoint / "model.pt").is_file():
            return False, f"SenseVoiceSmall model.pt is missing below {self.checkpoint}"
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("funasr") is None:
            return False, "Probe inference requires the torch and funasr packages."
        return True, None

    def get(self) -> tuple[FrozenSenseVoiceEncoder, Any]:
        available, reason = self.availability()
        if not available:
            raise RuntimeError(reason)
        if self._extractor is None:
            import torch

            self._torch = torch
            self._extractor = FrozenSenseVoiceEncoder(
                self.checkpoint,
                self.cache_dir,
                torch,
            )
        return self._extractor, self._torch


class FrozenLinearProbeHead:
    """Load and run one linear head over a shared frozen SenseVoice encoder."""

    def __init__(
        self,
        *,
        name: str,
        checkpoint: Path,
        encoder: FrozenEncoderProvider,
        label_mapping: dict[str, tuple[AnnotationChannel, EventLabel | StyleLabel]],
    ) -> None:
        self.name = name
        self.checkpoint = checkpoint
        self.encoder = encoder
        self.label_mapping = label_mapping
        self._head: Any | None = None
        self._payload: dict[str, Any] | None = None

    def availability(self) -> tuple[bool, str | None]:
        if not self.checkpoint.is_file():
            return False, f"gitignored linear checkpoint is missing: {self.checkpoint}"
        return self.encoder.availability()

    def predict(self, audio_path: Path) -> ProbePrediction:
        started = time.perf_counter()
        extractor, torch = self.encoder.get()
        head, payload = self._load(torch)
        with torch.inference_mode():
            features = extractor(audio_path)
            normalized = (features - payload["feature_mean"]) / payload["feature_scale"]
            logits = head(normalized.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)[0]
        method, threshold = checkpoint_abstention(payload)
        if method == "none_logit":
            none_index = len(payload["labels"])
            index = int(probabilities[:none_index].argmax())
            score = float(
                probabilities[:none_index].max() - probabilities[none_index]
            )
            abstained = not accepts(score, threshold)
        else:
            score = float(confidence_scores(logits, method, torch)[0])
            abstained = not accepts(score, threshold)
            index = int(probabilities.argmax())
        source_label = payload["labels"][index]
        channel, label = self.label_mapping[source_label]
        confidence = float(probabilities[index])
        return ProbePrediction(
            annotations=(
                () if abstained else (ProbeAnnotation(channel, label, confidence),)
            ),
            elapsed_seconds=time.perf_counter() - started,
            diagnostics={
                "name": self.name,
                "source_label": source_label,
                "mapped_channel": None if abstained else channel,
                "mapped_label": None if abstained else label.value,
                "confidence": confidence,
                "confidence_role": (
                    "diagnostic closed-set softmax; not reviewed gold"
                ),
                "abstained": abstained,
                "abstention_method": method,
                "abstention_score": score,
                "abstention_threshold": threshold,
                "abstention_rule": "emit when score >= threshold",
                "energy": -score if method == "energy" else None,
                "class_probabilities": {
                    source: float(probability)
                    for source, probability in zip(
                        payload["labels"],
                        probabilities[: len(payload["labels"])].tolist(),
                        strict=True,
                    )
                },
                "none_probability": (
                    float(probabilities[-1]) if method == "none_logit" else None
                ),
                "encoder_frozen": True,
                "embedding": SENSEVOICE_EMBEDDING,
                "span_scope": "utterance",
            },
            abstained=abstained,
        )

    def _load(self, torch: Any) -> tuple[Any, dict[str, Any]]:
        if self._head is not None and self._payload is not None:
            return self._head, self._payload
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        required = {
            "head_state_dict",
            "feature_mean",
            "feature_scale",
            "labels",
            "embedding",
            "abstention",
        }
        if not isinstance(payload, dict) or not required <= payload.keys():
            raise RuntimeError(f"{self.name} checkpoint has an unsupported payload")
        if payload["embedding"] != SENSEVOICE_EMBEDDING:
            raise RuntimeError(
                f"{self.name} checkpoint uses {payload['embedding']!r}, "
                f"expected {SENSEVOICE_EMBEDDING!r}"
            )
        method, _threshold = checkpoint_abstention(payload)
        labels = payload["labels"]
        if not isinstance(labels, list) or set(labels) != set(self.label_mapping):
            raise RuntimeError(f"{self.name} checkpoint labels do not match its ontology mapping")
        feature_count = int(payload["feature_mean"].numel())
        expected_features = FrozenSenseVoiceEncoder.output_size
        if feature_count != expected_features:
            raise RuntimeError(
                f"{self.name} checkpoint has {feature_count} features, "
                f"expected {expected_features}"
            )
        output_count = len(labels) + (method == "none_logit")
        head = torch.nn.Linear(feature_count, output_count)
        head.load_state_dict(payload["head_state_dict"])
        head.eval()
        self._head = head
        self._payload = payload
        return head, payload
