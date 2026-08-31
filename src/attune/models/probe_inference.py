"""Inference wrappers for gitignored frozen-encoder linear probe heads."""

from __future__ import annotations

import importlib.util
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from attune.calibration import TemperatureCalibration, calibration_from_payload
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

BERST_STYLE_LABEL_MAPPING: dict[str, tuple[AnnotationChannel, EventLabel | StyleLabel]] = {
    "shout": ("style", StyleLabel.SHOUTING),
}


@dataclass(frozen=True)
class ProbeAnnotation:
    """One utterance-level annotation emitted by a closed-set probe."""

    channel: AnnotationChannel
    label: EventLabel | StyleLabel
    confidence: float
    start_ms: int | None = None
    end_ms: int | None = None


@dataclass(frozen=True)
class ProbePrediction:
    """Probe annotations and bounded inference diagnostics."""

    annotations: tuple[ProbeAnnotation, ...]
    elapsed_seconds: float
    diagnostics: dict[str, Any]
    abstained: bool = False


class FrozenEncoderProvider:
    """Lazily share one immutable SenseVoice encoder across multiple heads."""

    def __init__(self, checkpoint: Path, cache_dir: Path, *, query_language: str = "auto") -> None:
        self.checkpoint = checkpoint
        self.cache_dir = cache_dir
        self.query_language = query_language
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
                query_language=self.query_language,
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
        calibration: TemperatureCalibration | None = None,
    ) -> None:
        self.name = name
        self.checkpoint = checkpoint
        self.encoder = encoder
        self.label_mapping = label_mapping
        self.calibration = calibration
        self._head: Any | None = None
        self._payload: dict[str, Any] | None = None

    def availability(self) -> tuple[bool, str | None]:
        if not self.checkpoint.is_file():
            return False, f"gitignored linear checkpoint is missing: {self.checkpoint}"
        return self.encoder.availability()

    def predict(self, audio_path: Path) -> ProbePrediction:
        started = time.perf_counter()
        extractor, torch = self.encoder.get()
        embedding_name = getattr(extractor, "embedding_name", SENSEVOICE_EMBEDDING)
        head, payload = self._load(torch, embedding_name)
        with torch.inference_mode():
            features = extractor(audio_path)
            normalized = (features - payload["feature_mean"]) / payload["feature_scale"]
            logits = head(normalized.unsqueeze(0))
            raw_probabilities = torch.softmax(logits, dim=1)[0]
        method, threshold = checkpoint_abstention(payload)
        if method == "none_logit":
            none_index = len(payload["labels"])
            index = int(raw_probabilities[:none_index].argmax())
            score = float(raw_probabilities[:none_index].max() - raw_probabilities[none_index])
            abstained = not accepts(score, threshold)
        else:
            score = float(confidence_scores(logits, method, torch)[0])
            abstained = not accepts(score, threshold)
            index = int(raw_probabilities.argmax())
        source_label = payload["labels"][index]
        channel, label = self.label_mapping[source_label]
        probability_labels = [
            *payload["labels"],
            *(["none"] if method == "none_logit" else []),
        ]
        raw_distribution = dict(zip(probability_labels, raw_probabilities.tolist(), strict=True))
        calibration = self.calibration
        if calibration is not None and calibration.labels != tuple(probability_labels):
            raise RuntimeError(f"{self.name} calibration labels do not match checkpoint")
        calibration_payload = payload.get("calibration")
        if calibration is None and isinstance(calibration_payload, dict):
            calibration = calibration_from_payload(
                calibration_payload,
                expected_labels=probability_labels,
            )
        calibrated_distribution = (
            calibration.probabilities(logits[0].tolist())
            if calibration is not None
            else raw_distribution
        )
        confidence = calibrated_distribution[source_label]
        return ProbePrediction(
            annotations=(() if abstained else (ProbeAnnotation(channel, label, confidence),)),
            elapsed_seconds=time.perf_counter() - started,
            diagnostics={
                "name": self.name,
                "source_label": source_label,
                "mapped_channel": None if abstained else channel,
                "mapped_label": None if abstained else label.value,
                "confidence": confidence,
                "confidence_role": (
                    "validation-temperature-scaled probability; not reviewed gold"
                    if calibration is not None
                    else "diagnostic uncalibrated softmax; not reviewed gold"
                ),
                "calibration_method": ("temperature_scaling" if calibration is not None else None),
                "calibration_temperature": (
                    calibration.temperature if calibration is not None else None
                ),
                "calibration_fitted_on": (
                    calibration.fitted_on if calibration is not None else None
                ),
                "abstained": abstained,
                "abstention_method": method,
                "abstention_score": score,
                "abstention_threshold": threshold,
                "abstention_rule": "emit when score >= threshold",
                "abstention_score_uses_uncalibrated_margin": True,
                "energy": -score if method == "energy" else None,
                "class_probabilities": {
                    source: calibrated_distribution[source] for source in payload["labels"]
                },
                "uncalibrated_class_probabilities": {
                    source: raw_distribution[source] for source in payload["labels"]
                },
                "none_probability": (
                    calibrated_distribution["none"] if method == "none_logit" else None
                ),
                "uncalibrated_none_probability": (
                    raw_distribution["none"] if method == "none_logit" else None
                ),
                "calibration_labels": probability_labels,
                "uncalibrated_logits": [float(value) for value in logits[0].tolist()],
                "encoder_frozen": True,
                "embedding": embedding_name,
                "span_scope": "utterance",
            },
            abstained=abstained,
        )

    def _load(self, torch: Any, embedding_name: str) -> tuple[Any, dict[str, Any]]:
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
        if payload["embedding"] != embedding_name:
            raise RuntimeError(
                f"{self.name} checkpoint uses {payload['embedding']!r}, expected {embedding_name!r}"
            )
        method, _threshold = checkpoint_abstention(payload)
        labels = payload["labels"]
        if not isinstance(labels, list) or set(labels) != set(self.label_mapping):
            raise RuntimeError(f"{self.name} checkpoint labels do not match its ontology mapping")
        feature_count = int(payload["feature_mean"].numel())
        expected_features = FrozenSenseVoiceEncoder.output_size
        if feature_count != expected_features:
            raise RuntimeError(
                f"{self.name} checkpoint has {feature_count} features, expected {expected_features}"
            )
        output_count = len(labels) + (method == "none_logit")
        head = torch.nn.Linear(feature_count, output_count)
        head.load_state_dict(payload["head_state_dict"])
        head.eval()
        self._head = head
        self._payload = payload
        return head, payload
