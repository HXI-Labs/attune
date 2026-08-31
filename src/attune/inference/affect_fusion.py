"""Fuse Cadence affect scores with the compact emotion2vec student."""

from __future__ import annotations

import io
import json
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from attune.inference.backend import InferenceBackend
from attune.schema.output import AffectCategory
from attune.schema.v2 import AttuneOutputV2

CADENCE_PARAMETER_COUNT = 241_904_650
PARAMETER_LIMIT = 300_000_000
DEFAULT_ACOUSTIC_WEIGHT = 0.86
DEFAULT_AFFECT_CONFIDENCE_THRESHOLD = 0.40


class AffectPredictor(Protocol):
    labels: tuple[str, ...]
    parameter_count: int

    def predict_probabilities(self, payloads: Sequence[bytes]) -> np.ndarray: ...


@dataclass(frozen=True)
class AffectProbabilityCalibration:
    """Post-quantization calibration bound to one portable affect artifact."""

    labels: tuple[str, ...]
    model_sha256: str
    temperature: float
    bias: tuple[float, ...]

    def __post_init__(self) -> None:
        expected_labels = tuple(category.value for category in AffectCategory)
        if self.labels != expected_labels:
            raise ValueError("affect calibration does not match the Attune ontology")
        if (
            not np.isfinite(self.temperature)
            or self.temperature <= 0.0
            or len(self.bias) != len(self.labels)
            or not np.isfinite(self.bias).all()
        ):
            raise ValueError("affect calibration has invalid parameters")

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        model_sha256: str,
    ) -> AffectProbabilityCalibration:
        try:
            serialized = json.loads(path.read_text())
            calibration = cls(
                labels=tuple(serialized["labels"]),
                model_sha256=str(serialized["model_sha256"]),
                temperature=float(serialized["temperature"]),
                bias=tuple(float(value) for value in serialized["bias"]),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid affect calibration: {path}") from error
        if serialized.get("schema_version") != "1.0":
            raise ValueError("affect calibration does not match the Attune ontology")
        if calibration.model_sha256 != model_sha256:
            raise ValueError("affect calibration model hash mismatch")
        return calibration

    def apply(self, probabilities: np.ndarray) -> np.ndarray:
        if probabilities.shape != (len(self.labels),) or not np.isfinite(probabilities).all():
            raise ValueError("affect probabilities have an invalid shape or value")
        logits = np.log(np.clip(probabilities, 1e-8, 1.0)) / self.temperature
        logits += np.asarray(self.bias, dtype=np.float64)
        logits -= logits.max()
        exponent = np.exp(logits)
        return exponent / exponent.sum()


def fuse_probabilities(
    cadence: np.ndarray,
    acoustic_student: np.ndarray,
    *,
    acoustic_weight: float,
) -> np.ndarray:
    if cadence.shape != acoustic_student.shape:
        raise ValueError("affect probability vectors must have the same shape")
    if not 0.0 <= acoustic_weight <= 1.0:
        raise ValueError("acoustic fusion weight must be between zero and one")
    probabilities = (1.0 - acoustic_weight) * cadence + acoustic_weight * acoustic_student
    return probabilities / probabilities.sum()


def _slice_wav(payload: bytes, start_ms: int, end_ms: int) -> bytes:
    with wave.open(io.BytesIO(payload), "rb") as source:
        sample_rate = source.getframerate()
        start_frame = round(start_ms * sample_rate / 1000)
        frame_count = round((end_ms - start_ms) * sample_rate / 1000)
        source.setpos(start_frame)
        frames = source.readframes(frame_count)
        output = io.BytesIO()
        with wave.open(output, "wb") as destination:
            destination.setnchannels(source.getnchannels())
            destination.setsampwidth(source.getsampwidth())
            destination.setframerate(sample_rate)
            destination.writeframes(frames)
    return output.getvalue()


class FusedAffectBackend:
    """Preserve Cadence ASR/events while replacing its affect decision with a fixed fusion."""

    def __init__(
        self,
        cadence: InferenceBackend,
        acoustic_student: AffectPredictor,
        *,
        acoustic_weight: float = DEFAULT_ACOUSTIC_WEIGHT,
        confidence_threshold: float = DEFAULT_AFFECT_CONFIDENCE_THRESHOLD,
        probability_calibration: AffectProbabilityCalibration | None = None,
    ) -> None:
        expected_labels = tuple(category.value for category in AffectCategory)
        if acoustic_student.labels != expected_labels:
            raise ValueError("acoustic student labels do not match the Attune affect ontology")
        if CADENCE_PARAMETER_COUNT + acoustic_student.parameter_count > PARAMETER_LIMIT:
            raise ValueError("fused model exceeds the 300M parameter limit")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("affect confidence threshold must be between zero and one")
        self.cadence = cadence
        self.acoustic_student = acoustic_student
        self.acoustic_weight = acoustic_weight
        self.confidence_threshold = confidence_threshold
        self.probability_calibration = probability_calibration
        self.name = "attune-cadence-300m-affect-fusion"

    def analyse_wav(self, audio: bytes) -> AttuneOutputV2:
        cadence_output = self.cadence.analyse_wav(audio)
        serialized = cadence_output.model_dump(mode="json")
        spans = serialized["affect_spans"] or [serialized["affect"]]
        utterance_probabilities = self.acoustic_student.predict_probabilities([audio])[0]
        if (
            len(spans) == 1
            and spans[0]["start_ms"] == 0
            and spans[0]["end_ms"] == int(serialized["audio"]["duration_ms"])
        ):
            span_probabilities = utterance_probabilities[np.newaxis]
        else:
            span_audio = [
                _slice_wav(audio, int(span["start_ms"]), int(span["end_ms"])) for span in spans
            ]
            span_probabilities = self.acoustic_student.predict_probabilities(span_audio)
        fused_spans = [
            self._fused_affect(span, probabilities)
            for span, probabilities in zip(spans, span_probabilities, strict=True)
        ]
        utterance_affect = self._fused_affect(serialized["affect"], utterance_probabilities)
        confident_span_labels = {
            span["top_label"]
            for span in fused_spans
            if not span["abstain"] and span["top_label"] is not None
        }
        if len(confident_span_labels) > 1:
            utterance_affect.update(
                top_label=None,
                top_label_confidence=0.0,
                abstain=True,
                abstention_reason="mixed_affect_spans",
            )
        serialized["affect"] = utterance_affect
        serialized["affect_spans"] = fused_spans
        serialized["model"] = {
            "name": self.name,
            "version": "0.2.0-dev",
            "quantization": (
                f"{cadence_output.model.quantization}"
                f"+{getattr(self.acoustic_student, 'quantization', 'fp32')}-affect"
            ),
        }
        serialized["uncertainty"]["out_of_distribution_probability"] = None
        serialized["uncertainty"]["out_of_distribution_available"] = False
        return AttuneOutputV2.model_validate(serialized)

    def _fused_affect(
        self,
        cadence_affect: dict,
        student_probabilities: np.ndarray,
    ) -> dict:
        cadence_probabilities = np.asarray(
            [cadence_affect["categories"][label] for label in self.acoustic_student.labels],
            dtype=np.float32,
        )
        probabilities = fuse_probabilities(
            cadence_probabilities,
            student_probabilities,
            acoustic_weight=self.acoustic_weight,
        )
        if self.probability_calibration is not None:
            probabilities = self.probability_calibration.apply(probabilities)
        top_index = int(probabilities.argmax())
        confidence = float(probabilities[top_index])
        abstain = confidence < self.confidence_threshold
        return {
            **cadence_affect,
            "categories": dict(
                zip(self.acoustic_student.labels, probabilities.tolist(), strict=True)
            ),
            "top_label": None if abstain else self.acoustic_student.labels[top_index],
            "top_label_confidence": 0.0 if abstain else confidence,
            "abstain": abstain,
            "abstention_reason": "category_confidence_below_threshold" if abstain else None,
        }
