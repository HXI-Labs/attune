"""Schema-v2 inference over an exported FP or INT8 Attune ONNX graph."""

from __future__ import annotations

import array
import gc
import io
import math
import os
import re
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from attune.inference.export import OUTPUT_NAMES, PROBE_EMBEDDING_OUTPUT
from attune.inference.probe_head import LinearProbeHead, ProbeDecision
from attune.models.joint import SUPPORTED_EVENTS, SUPPORTED_STYLES
from attune.schema.output import AffectCategory
from attune.schema.v2 import AttuneOutputV2

RICH_TAG = re.compile(r"<\|[^|]+\|>")


class FeatureExtractor(Protocol):
    def __call__(self, wav: bytes) -> np.ndarray: ...


class TranscriptDecoder(Protocol):
    def __call__(self, logits: np.ndarray, length: int) -> tuple[str, float]: ...


class RuntimeCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0"
    event_temperature: float = Field(default=1.0, gt=0)
    event_thresholds: dict[str, float]
    event_presence_temperature: float = Field(default=1.0, gt=0)
    event_presence_thresholds: dict[str, float]
    localized_event_labels: list[str] = Field(
        default_factory=lambda: ["laugh", "cough", "throat_clear"]
    )
    localized_event_min_confidence: float = Field(default=0.98, ge=0, le=1)
    event_presence_enabled_labels: list[str] = Field(default_factory=list)
    style_temperature: float = Field(default=1.0, gt=0)
    style_thresholds: dict[str, float]
    style_enabled_labels: list[str] = Field(default_factory=list)
    affect_temperature: float = Field(default=1.0, gt=0)
    affect_bias: list[float] = Field(default_factory=lambda: [0.0] * len(AffectCategory))
    affect_bias_mode: Literal["fitted", "none", "aps_constrained"] = "fitted"
    affect_bias_scale: float = Field(default=1.0, ge=0, le=1)
    affect_threshold: float = Field(default=0.55, ge=0, le=1)
    vad_available: bool = False
    ood_centroid: list[float] | None = None
    ood_distance_scale: float | None = Field(default=None, gt=0)
    ood_available: bool = False
    ood_temperature: float = Field(default=1.0, gt=0)
    ood_threshold: float = Field(default=0.5, ge=0, le=1)
    minimum_event_frames: int = Field(default=2, ge=1)
    bridge_event_frames: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def complete_label_thresholds(self) -> RuntimeCalibration:
        expected_events = {label.value for label in SUPPORTED_EVENTS}
        expected_styles = {label.value for label in SUPPORTED_STYLES}
        if set(self.event_thresholds) != expected_events:
            raise ValueError("event_thresholds must contain the supported event inventory")
        if set(self.event_presence_thresholds) != expected_events:
            raise ValueError("event_presence_thresholds must contain the supported event inventory")
        if not set(self.localized_event_labels) <= expected_events:
            raise ValueError("localized_event_labels contains an unsupported event")
        if not set(self.event_presence_enabled_labels) <= expected_events:
            raise ValueError("event_presence_enabled_labels contains an unsupported event")
        if set(self.style_thresholds) != expected_styles:
            raise ValueError("style_thresholds must contain the supported style inventory")
        if not set(self.style_enabled_labels) <= expected_styles:
            raise ValueError("style_enabled_labels contains an unsupported style")
        if len(self.affect_bias) != len(AffectCategory):
            raise ValueError("affect_bias must contain the complete affect inventory")
        if any(value < 0 or value > 1 for value in self.event_thresholds.values()):
            raise ValueError("event thresholds must be probabilities")
        if any(value < 0 or value > 1 for value in self.event_presence_thresholds.values()):
            raise ValueError("event-presence thresholds must be probabilities")
        if any(value < 0 or value > 1 for value in self.style_thresholds.values()):
            raise ValueError("style thresholds must be probabilities")
        if (self.ood_centroid is None) != (self.ood_distance_scale is None):
            raise ValueError("OOD centroid and scale must be supplied together")
        return self

    def localized_event_threshold(self, label: str) -> float:
        return max(self.event_thresholds[label], self.localized_event_min_confidence)


def _affect_probabilities(logits: np.ndarray, calibration: RuntimeCalibration) -> np.ndarray:
    bias = np.asarray(calibration.affect_bias, dtype=np.float32)
    return _softmax(logits / calibration.affect_temperature + bias)


@dataclass(frozen=True)
class WavMetadata:
    duration_ms: int
    sample_rate_hz: int
    channels: int
    clipping_ratio: float


def inspect_wav(payload: bytes) -> WavMetadata:
    try:
        with wave.open(io.BytesIO(payload), "rb") as handle:
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            sample_width = handle.getsampwidth()
            frame_count = handle.getnframes()
            frames = handle.readframes(frame_count)
    except (wave.Error, EOFError) as error:
        raise ValueError(f"invalid WAV payload: {error}") from error
    if channels != 1 or sample_rate != 16_000 or sample_width != 2:
        raise ValueError("Attune requires 16 kHz mono PCM16 WAV audio")
    duration_ms = round(frame_count / sample_rate * 1000)
    if duration_ms < 500 or duration_ms > 30_000:
        raise ValueError("audio duration must be between 0.5 and 30 seconds")
    samples = array.array("h")
    samples.frombytes(frames)
    clipping = sum(abs(sample) >= 32760 for sample in samples) / max(1, len(samples))
    return WavMetadata(duration_ms, sample_rate, channels, min(1.0, clipping))


class GreedySenseVoiceDecoder:
    def __init__(self, tokenizer: Any, *, blank_id: int = 0) -> None:
        self.tokenizer = tokenizer
        self.blank_id = blank_id

    def __call__(self, logits: np.ndarray, length: int) -> tuple[str, float]:
        text, confidence, _words = self.decode_with_words(logits, length, duration_ms=None)
        return text, confidence

    def decode_with_words(
        self, logits: np.ndarray, length: int, duration_ms: int | None
    ) -> tuple[str, float, list[dict[str, Any]]]:
        active = logits[:length]
        probabilities = _softmax(active, axis=-1)
        frame_ids = active.argmax(axis=-1)
        selected: list[int] = []
        selected_confidence: list[float] = []
        token_spans: list[tuple[int, int, int, float]] = []
        previous = None
        for frame_index, token_id in enumerate(frame_ids.tolist()):
            if token_id != previous and token_id != self.blank_id:
                selected.append(token_id)
                selected_confidence.append(float(probabilities[frame_index, token_id]))
                token_spans.append(
                    (
                        token_id,
                        frame_index,
                        frame_index + 1,
                        float(probabilities[frame_index, token_id]),
                    )
                )
            elif token_id == previous and token_id != self.blank_id:
                identifier, start, _stop, confidence = token_spans[-1]
                token_spans[-1] = (
                    identifier,
                    start,
                    frame_index + 1,
                    max(confidence, float(probabilities[frame_index, token_id])),
                )
            previous = token_id
        text = self.tokenizer.decode(selected) if selected else ""
        text = re.sub(r"\s+", " ", RICH_TAG.sub("", text)).strip()
        confidence = sum(selected_confidence) / len(selected_confidence) if selected else 0.0
        words = self._word_spans(token_spans, length, duration_ms) if duration_ms else []
        return text, confidence, words

    def _word_spans(
        self,
        token_spans: list[tuple[int, int, int, float]],
        length: int,
        duration_ms: int,
    ) -> list[dict[str, Any]]:
        encoding = getattr(self.tokenizer, "encoding", None)
        sentencepiece = getattr(self.tokenizer, "sp", None)
        if encoding is None and sentencepiece is None:
            return []
        groups: list[dict[str, Any]] = []
        pending_word_boundary = False
        for token_id, start, stop, confidence in token_spans:
            if encoding is not None:
                piece = encoding.decode([token_id])
                begins_word = bool(piece[:1].isspace())
                clean = piece.strip() if begins_word else piece
            else:
                piece = sentencepiece.IdToPiece(token_id)
                begins_word = piece.startswith("▁")
                clean = piece.removeprefix("▁")
            if not piece or RICH_TAG.fullmatch(piece):
                continue
            if clean in {"<unk>", "<s>", "</s>"}:
                continue
            if not clean:
                pending_word_boundary = pending_word_boundary or begins_word
                continue
            begins_word = begins_word or pending_word_boundary
            pending_word_boundary = False
            if begins_word or not groups:
                groups.append(
                    {"text": clean, "start": start, "stop": stop, "confidence": [confidence]}
                )
            else:
                groups[-1]["text"] += clean
                groups[-1]["stop"] = stop
                groups[-1]["confidence"].append(confidence)
        return [
            {
                "id": f"w{index}",
                "text": group["text"],
                "start_ms": min(duration_ms, round(group["start"] / length * duration_ms)),
                "end_ms": min(duration_ms, round(group["stop"] / length * duration_ms)),
                "confidence": sum(group["confidence"]) / len(group["confidence"]),
            }
            for index, group in enumerate(groups, start=1)
        ]


class LocalSenseVoiceFrontend:
    """Load the official frontend/tokenizer, then release full model weights."""

    def __init__(self, checkpoint: Path) -> None:
        if os.environ.get("ATTUNE_SENSEVOICE_LICENSE_REVIEWED") != "1":
            raise RuntimeError("review the SenseVoice model licence before loading its assets")
        if not (checkpoint / "model.pt").is_file():
            raise FileNotFoundError(f"SenseVoice model.pt is missing below {checkpoint}")
        from funasr import AutoModel

        previous = {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "MODELSCOPE_OFFLINE")}
        os.environ.update({"HF_HUB_OFFLINE": "1", "MODELSCOPE_OFFLINE": "1"})
        try:
            wrapper = AutoModel(
                model=str(checkpoint),
                disable_update=True,
                device="cpu",
                frontend_conf={"dither": 0.0},
            )
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        self.frontend = wrapper.kwargs["frontend"]
        self.frontend.dither = 0.0
        self.tokenizer = wrapper.kwargs["tokenizer"]
        self.decoder = GreedySenseVoiceDecoder(self.tokenizer)
        del wrapper
        gc.collect()

    def __call__(self, wav: bytes) -> np.ndarray:
        from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video

        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            handle.write(wav)
            handle.flush()
            audio = load_audio_text_image_video(
                handle.name, fs=self.frontend.fs, audio_fs=16_000, data_type="sound"
            )
            speech, _lengths = extract_fbank(audio, data_type="sound", frontend=self.frontend)
        return speech[0].detach().float().cpu().numpy()


class OnnxAttuneBackend:
    def __init__(
        self,
        model_path: Path,
        *,
        feature_extractor: FeatureExtractor,
        transcript_decoder: TranscriptDecoder,
        calibration: RuntimeCalibration,
        session: Any | None = None,
        quantization: str = "int8",
        probe_heads: tuple[LinearProbeHead, ...] = (),
    ) -> None:
        if session is None:
            try:
                import onnxruntime as ort
            except ImportError as error:
                raise RuntimeError("install the deployment extra to run ONNX inference") from error
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.session = session
        self.model_path = model_path
        self.feature_extractor = feature_extractor
        self.transcript_decoder = transcript_decoder
        self.calibration = calibration
        self.quantization = quantization
        self.probe_heads = probe_heads
        if probe_heads:
            available_outputs = {output.name for output in session.get_outputs()}
            if PROBE_EMBEDDING_OUTPUT not in available_outputs:
                raise ValueError("configured probe heads require a probe_embedding model output")
        probe_suffix = "+calibrated-probes" if probe_heads else ""
        self.name = f"attune-cadence-242m-{quantization}{probe_suffix}"

    @classmethod
    def from_local_assets(
        cls,
        model_path: Path,
        sensevoice_checkpoint: Path,
        calibration_path: Path,
        *,
        quantization: str = "int8",
        probe_artifacts: tuple[Path, ...] = (),
    ) -> OnnxAttuneBackend:
        frontend = LocalSenseVoiceFrontend(sensevoice_checkpoint)
        calibration = RuntimeCalibration.model_validate_json(calibration_path.read_text())
        return cls(
            model_path,
            feature_extractor=frontend,
            transcript_decoder=frontend.decoder,
            calibration=calibration,
            quantization=quantization,
            probe_heads=tuple(LinearProbeHead(path) for path in probe_artifacts),
        )

    def analyse_wav(self, audio: bytes) -> AttuneOutputV2:
        metadata = inspect_wav(audio)
        features = self.feature_extractor(audio)
        if features.ndim != 2:
            raise ValueError("feature extractor must return (time, feature) values")
        requested_outputs = [
            *OUTPUT_NAMES,
            *([PROBE_EMBEDDING_OUTPUT] if self.probe_heads else []),
        ]
        values = self.session.run(
            requested_outputs,
            {
                "speech": features[np.newaxis].astype(np.float32),
                "speech_lengths": np.asarray([features.shape[0]], dtype=np.int64),
            },
        )
        outputs = dict(zip(requested_outputs, values, strict=True))
        acoustic_length = int(outputs["acoustic_lengths"][0])
        decode_with_words = getattr(self.transcript_decoder, "decode_with_words", None)
        if decode_with_words is None:
            transcript, transcript_confidence = self.transcript_decoder(
                outputs["ctc_logits"][0], acoustic_length
            )
            words: list[dict[str, Any]] = []
        else:
            transcript, transcript_confidence, words = decode_with_words(
                outputs["ctc_logits"][0], acoustic_length, metadata.duration_ms
            )
        probe_decisions = self._probe_decisions(outputs)
        events = self._events(outputs, acoustic_length, metadata.duration_ms, probe_decisions)
        styles = self._styles(outputs, probe_decisions)
        affect, ood_probability = self._affect(outputs, metadata.duration_ms)
        payload = {
            "schema_version": "2.0",
            "model": {
                "name": self.name,
                "version": "0.1.0",
                "quantization": self.quantization,
            },
            "audio": {
                "duration_ms": metadata.duration_ms,
                "sample_rate_hz": metadata.sample_rate_hz,
                "channels": metadata.channels,
                "quality": {
                    "clipping": {"value": metadata.clipping_ratio, "method": "heuristic"},
                    "low_snr": {"value": None, "method": "unavailable"},
                    "far_field": {"value": None, "method": "unavailable"},
                },
            },
            "language": {"label": "en", "confidence": 1.0},
            "transcript": {
                "text": transcript,
                "confidence": transcript_confidence,
                "words": words,
            },
            "events": events,
            "styles": styles,
            "affect": affect,
            "uncertainty": {
                "out_of_distribution_probability": ood_probability,
                "interpretation_warning": (
                    "Vocal affect is a probabilistic perception, not a verified internal state."
                ),
            },
        }
        return AttuneOutputV2.model_validate(payload)

    def _events(
        self,
        outputs: dict[str, np.ndarray],
        length: int,
        duration_ms: int,
        probe_decisions: tuple[ProbeDecision, ...] = (),
    ) -> list[dict[str, Any]]:
        probabilities = _sigmoid(
            outputs["event_logits"][0, :length] / self.calibration.event_temperature
        )
        items = []
        frame_ms = duration_ms / max(1, length)
        for label_index, label in enumerate(SUPPORTED_EVENTS):
            if label.value not in self.calibration.localized_event_labels:
                continue
            active = probabilities[:, label_index] >= self.calibration.localized_event_threshold(
                label.value
            )
            active = _bridge_short_gaps(active, self.calibration.bridge_event_frames)
            for start, stop in _runs(active, self.calibration.minimum_event_frames):
                confidence = float(probabilities[start:stop, label_index].max())
                items.append(
                    {
                        "id": f"event-{len(items) + 1}",
                        "label": label.value,
                        "temporal_scope": "localized",
                        "start_ms": min(duration_ms, round(start * frame_ms)),
                        "end_ms": min(duration_ms, round(stop * frame_ms)),
                        "after_word_id": None,
                        "confidence": confidence,
                        "status": "committed",
                    }
                )
        presence = _sigmoid(
            outputs["event_presence_logits"][0] / self.calibration.event_presence_temperature
        )
        already_present = {item["label"] for item in items}
        for label_index, label in enumerate(SUPPORTED_EVENTS):
            confidence = float(presence[label_index])
            if (
                label.value in self.calibration.event_presence_enabled_labels
                and label.value not in already_present
                and confidence >= self.calibration.event_presence_thresholds[label.value]
            ):
                items.append(
                    {
                        "id": f"event-{len(items) + 1}",
                        "label": label.value,
                        "temporal_scope": "utterance",
                        "start_ms": None,
                        "end_ms": None,
                        "after_word_id": None,
                        "confidence": confidence,
                        "status": "committed",
                    }
                )
        already_present = {item["label"] for item in items}
        for decision in probe_decisions:
            if decision.abstained or decision.channel != "event":
                continue
            if decision.label in already_present:
                continue
            items.append(
                {
                    "id": f"event-{len(items) + 1}",
                    "label": decision.label,
                    "temporal_scope": "utterance",
                    "start_ms": None,
                    "end_ms": None,
                    "after_word_id": None,
                    "confidence": decision.confidence,
                    "status": "committed",
                }
            )
            already_present.add(decision.label)
        return items

    def _styles(
        self,
        outputs: dict[str, np.ndarray],
        probe_decisions: tuple[ProbeDecision, ...] = (),
    ) -> list[dict[str, Any]]:
        probabilities = _sigmoid(outputs["style_logits"][0] / self.calibration.style_temperature)
        items = []
        for index, label in enumerate(SUPPORTED_STYLES):
            if label.value not in self.calibration.style_enabled_labels:
                continue
            confidence = float(probabilities[index])
            if confidence >= self.calibration.style_thresholds[label.value]:
                items.append(
                    {
                        "id": f"style-{len(items) + 1}",
                        "label": label.value,
                        "temporal_scope": "utterance",
                        "start_ms": None,
                        "end_ms": None,
                        "start_word_id": None,
                        "end_word_id": None,
                        "confidence": confidence,
                        "status": "committed",
                    }
                )
        already_present = {item["label"] for item in items}
        for decision in probe_decisions:
            if decision.abstained or decision.channel != "style":
                continue
            if decision.label in already_present:
                continue
            items.append(
                {
                    "id": f"style-{len(items) + 1}",
                    "label": decision.label,
                    "temporal_scope": "utterance",
                    "start_ms": None,
                    "end_ms": None,
                    "start_word_id": None,
                    "end_word_id": None,
                    "confidence": decision.confidence,
                    "status": "committed",
                }
            )
            already_present.add(decision.label)
        return items

    def _probe_decisions(self, outputs: dict[str, np.ndarray]) -> tuple[ProbeDecision, ...]:
        if not self.probe_heads:
            return ()
        embedding = outputs[PROBE_EMBEDDING_OUTPUT][0]
        return tuple(head.predict(embedding) for head in self.probe_heads)

    def _affect(
        self, outputs: dict[str, np.ndarray], duration_ms: int
    ) -> tuple[dict[str, Any], float]:
        probabilities = _affect_probabilities(outputs["affect_logits"][0], self.calibration)
        labels = tuple(AffectCategory)
        top_index = int(probabilities.argmax())
        top_confidence = float(probabilities[top_index])
        ood_probability = self._ood(outputs)
        abstain = (
            top_confidence < self.calibration.affect_threshold
            or ood_probability >= self.calibration.ood_threshold
        )
        if ood_probability >= self.calibration.ood_threshold:
            reason = "out_of_distribution"
        elif top_confidence < self.calibration.affect_threshold:
            reason = "category_confidence_below_threshold"
        else:
            reason = None
        dimension_confidence = max(0.0, min(1.0, top_confidence * (1.0 - ood_probability)))
        vad = outputs["vad"][0]
        if self.calibration.vad_available:
            dimensions = {
                name: {
                    "value": float(vad[index]),
                    "confidence": dimension_confidence,
                    "available": True,
                }
                for index, name in enumerate(("valence", "arousal", "dominance"))
            }
        else:
            dimensions = {
                name: {"value": None, "confidence": 0.0, "available": False}
                for name in ("valence", "arousal", "dominance")
            }
        return (
            {
                "start_ms": 0,
                "end_ms": duration_ms,
                **dimensions,
                "categories": {
                    label.value: float(probabilities[index]) for index, label in enumerate(labels)
                },
                "top_label": None if abstain else labels[top_index].value,
                "top_label_confidence": 0.0 if abstain else top_confidence,
                "abstain": abstain,
                "abstention_reason": reason,
            },
            ood_probability,
        )

    def _ood(self, outputs: dict[str, np.ndarray]) -> float:
        if not self.calibration.ood_available:
            return 1.0
        if "ood_logit" in outputs:
            value = outputs["ood_logit"][0] / self.calibration.ood_temperature
            return float(_sigmoid(np.asarray(value)))
        embedding = outputs["ood_embedding"][0]
        if self.calibration.ood_centroid is None:
            return 1.0
        centroid = np.asarray(self.calibration.ood_centroid, dtype=np.float32)
        if centroid.shape != embedding.shape:
            raise ValueError("OOD centroid does not match exported embedding size")
        distance = float(np.linalg.norm(embedding - centroid))
        scale = float(self.calibration.ood_distance_scale)
        return 1.0 / (1.0 + math.exp(-(distance / scale - 1.0)))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - values.max(axis=axis, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=axis, keepdims=True)


def _bridge_short_gaps(active: np.ndarray, maximum_gap: int) -> np.ndarray:
    result = active.copy()
    if maximum_gap <= 0:
        return result
    false_runs = _runs(~result, 1)
    for start, stop in false_runs:
        if start > 0 and stop < len(result) and stop - start <= maximum_gap:
            result[start:stop] = True
    return result


def _runs(active: np.ndarray, minimum: int) -> list[tuple[int, int]]:
    padded = np.pad(active.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return [
        (int(start), int(stop))
        for start, stop in zip(starts, stops, strict=True)
        if stop - start >= minimum
    ]
