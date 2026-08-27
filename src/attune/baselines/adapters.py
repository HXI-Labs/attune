"""Offline-safe baseline adapters that preserve the Attune schema contract."""

from __future__ import annotations

import os
import re
import time
import wave
from abc import ABC, abstractmethod
from array import array
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from attune.audio.contracts import placeholder_quality_probabilities
from attune.baselines.sensevoice import (
    build_utterance_spans,
    parse_sensevoice_output,
    sensevoice_affect_trace,
)
from attune.calibration import ConfidenceAbstention, TemperatureCalibration
from attune.evaluation.report import RuntimeMetrics
from attune.schema.output import AffectCategory, AttuneOutput, Word

AFFECT_LABELS = tuple(category.value for category in AffectCategory)
NEUTRAL_DISTRIBUTION = {
    AffectCategory.NEUTRAL: 0.65,
    AffectCategory.JOY: 0.05,
    AffectCategory.DISTRESS: 0.05,
    AffectCategory.ANGER: 0.05,
    AffectCategory.FEAR: 0.05,
    AffectCategory.SURPRISE: 0.05,
    AffectCategory.OTHER: 0.05,
    AffectCategory.AMBIGUOUS: 0.05,
}


class BaselineUnavailableError(RuntimeError):
    """An optional adapter cannot run from local resources."""


@dataclass(frozen=True)
class BaselineInput:
    audio_path: Path
    transcript_hint: str | None = None
    language_hint: str = "en"


@dataclass(frozen=True)
class BaselinePrediction:
    output: AttuneOutput
    runtime: RuntimeMetrics
    diagnostics: dict[str, Any] | None = None


class BaselineAdapter(ABC):
    """Common interface for all Phase 1 full-output runners."""

    name: str

    @abstractmethod
    def availability(self) -> tuple[bool, str | None]:
        """Return whether the adapter can run without downloading weights."""

    @abstractmethod
    def predict(self, item: BaselineInput) -> BaselinePrediction:
        """Emit a schema-valid Attune output."""


class TranscriptSentimentAdapter(BaselineAdapter):
    """Deterministic CPU lexicon baseline; audio channels are explicit placeholders."""

    name = "transcript-only-lexicon"
    _lexicons = {
        AffectCategory.JOY: {"happy", "great", "love", "wonderful", "glad", "excellent"},
        AffectCategory.DISTRESS: {"sad", "upset", "hurt", "crying", "miserable"},
        AffectCategory.ANGER: {"angry", "furious", "hate", "rage", "annoyed"},
        AffectCategory.FEAR: {"afraid", "scared", "terrified", "fear", "worried"},
        AffectCategory.SURPRISE: {"surprised", "wow", "unexpected", "amazing"},
    }

    def availability(self) -> tuple[bool, str | None]:
        return True, None

    def predict(self, item: BaselineInput) -> BaselinePrediction:
        if item.transcript_hint is None:
            raise ValueError("transcript-only baseline requires transcript_hint")
        started = time.perf_counter()
        category, distribution = self.classify(item.transcript_hint)
        output = build_partial_output(
            item,
            model_name=self.name,
            transcript=item.transcript_hint,
            category=category,
            distribution=distribution,
        )
        elapsed = time.perf_counter() - started
        return BaselinePrediction(
            output=output,
            runtime=RuntimeMetrics.measured(
                audio_seconds=output.audio.duration_ms / 1000,
                elapsed_seconds=elapsed,
            ),
            diagnostics={
                "affect_source": "transcript_lexicon",
                "raw_affect_label": None,
                "schema_affect_label": category.value,
                "note": "No acoustic affect model ran; this is a lexical fallback.",
            },
        )

    def classify(self, text: str) -> tuple[AffectCategory, dict[AffectCategory, float]]:
        tokens = set(re.findall(r"[a-z']+", text.lower()))
        counts = {category: len(tokens & words) for category, words in self._lexicons.items()}
        highest = max(counts.values(), default=0)
        if highest == 0:
            return AffectCategory.NEUTRAL, dict(NEUTRAL_DISTRIBUTION)
        winners = [category for category, count in counts.items() if count == highest]
        category = AffectCategory.AMBIGUOUS if len(winners) > 1 else winners[0]
        distribution = {label: 0.03 for label in AffectCategory}
        distribution[category] = 0.79
        return category, distribution


class WhisperSmallAdapter(BaselineAdapter):
    """Whisper-Small ASR loaded only from an explicit path or local HF cache."""

    name = "whisper-small"

    def __init__(self, checkpoint: Path | None = None) -> None:
        self.checkpoint = checkpoint or _local_checkpoint(
            "ATTUNE_WHISPER_SMALL_PATH", "models--openai--whisper-small"
        )
        self._processor: Any | None = None
        self._model: Any | None = None
        self._pipeline: Any | None = None

    def availability(self) -> tuple[bool, str | None]:
        if self.checkpoint is None:
            return (
                False,
                "Whisper-Small weights are not local; set ATTUNE_WHISPER_SMALL_PATH "
                "or populate the Hugging Face cache. Downloads are disabled.",
            )
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False, "Whisper-Small requires the optional torch and transformers packages."
        return True, None

    def predict(self, item: BaselineInput) -> BaselinePrediction:
        available, reason = self.availability()
        if not available:
            raise BaselineUnavailableError(reason)
        assert self.checkpoint is not None
        started = time.perf_counter()
        samples, sample_rate = _read_pcm16(item.audio_path)
        if sample_rate != 16_000:
            raise ValueError("Whisper adapter currently requires 16 kHz fixture audio")
        import numpy
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        if self._processor is None or self._model is None:
            self._processor = AutoProcessor.from_pretrained(self.checkpoint, local_files_only=True)
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.checkpoint, local_files_only=True
            )
        if self._pipeline is None:
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self._model,
                tokenizer=self._processor.tokenizer,
                feature_extractor=self._processor.feature_extractor,
                device=-1,
            )
        transcription = self._pipeline(
            {"raw": numpy.asarray(samples, dtype=numpy.float32), "sampling_rate": sample_rate},
            return_timestamps="word",
            generate_kwargs={"language": item.language_hint, "task": "transcribe"},
        )
        transcript = str(transcription.get("text", "")).strip()
        duration_ms, _, _ = _wave_info(item.audio_path)
        words = parse_whisper_word_timestamps(transcription.get("chunks"), duration_ms=duration_ms)
        category, distribution = TranscriptSentimentAdapter().classify(transcript)
        output = build_partial_output(
            item,
            model_name=self.name,
            transcript=transcript,
            category=category,
            distribution=distribution,
            words=words,
        )
        elapsed = time.perf_counter() - started
        return BaselinePrediction(
            output,
            RuntimeMetrics.measured(
                audio_seconds=output.audio.duration_ms / 1000, elapsed_seconds=elapsed
            ),
            diagnostics={
                "affect_source": "transcript_lexicon",
                "raw_model_output": {
                    "transcript": transcript,
                    "word_timestamps_returned": bool(words),
                },
                "raw_affect_label": None,
                "schema_affect_label": category.value,
                "note": "Whisper has no affect head; affect is a transcript-lexicon fallback.",
            },
        )


class SenseVoiceSmallAdapter(BaselineAdapter):
    """SenseVoice-Small adapter with an explicit licence and local-weight gate."""

    name = "sensevoice-small"

    def __init__(self, checkpoint: Path | None = None) -> None:
        self.checkpoint = checkpoint or _local_checkpoint(
            "ATTUNE_SENSEVOICE_SMALL_PATH",
            "models--FunAudioLLM--SenseVoiceSmall",
        )
        self._model: Any | None = None

    def availability(self) -> tuple[bool, str | None]:
        if os.environ.get("ATTUNE_SENSEVOICE_LICENSE_REVIEWED") != "1":
            return (
                False,
                "SenseVoice weights have a separate licence. Complete review, then set "
                "ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1; never download them in CI.",
            )
        if self.checkpoint is None:
            return (
                False,
                "SenseVoice-Small weights are not local; set ATTUNE_SENSEVOICE_SMALL_PATH. "
                "Downloads are disabled.",
            )
        try:
            import funasr  # noqa: F401
        except ImportError:
            return False, "SenseVoice-Small requires the optional funasr package."
        return True, None

    def predict(self, item: BaselineInput) -> BaselinePrediction:
        available, reason = self.availability()
        if not available:
            raise BaselineUnavailableError(reason)
        assert self.checkpoint is not None
        started = time.perf_counter()
        from funasr import AutoModel

        with _offline_model_environment():
            if self._model is None:
                self._model = AutoModel(
                    model=str(self.checkpoint),
                    disable_update=True,
                    frontend_conf={"dither": 0.0},
                )
            result = self._model.generate(
                input=str(item.audio_path),
                cache={},
                language="auto",
                output_timestamp=True,
                sentence_timestamp=True,
            )
        parsed = parse_sensevoice_output(result)
        affect_trace = sensevoice_affect_trace(result)
        mapped_affect_trace = [row for row in affect_trace if row["schema_label"] is not None]
        if parsed.affect:
            category = parsed.affect[0].label
            if not isinstance(category, AffectCategory):
                raise RuntimeError("SenseVoice affect tag mapped outside the affect ontology")
            distribution = {label: 0.01 for label in AffectCategory}
            distribution[category] = 0.93
        else:
            category, distribution = TranscriptSentimentAdapter().classify(parsed.transcript)
        duration_ms, _, _ = _wave_info(item.audio_path)
        words = parse_sensevoice_word_timestamps(result, duration_ms=duration_ms)
        output = build_partial_output(
            item,
            model_name=self.name,
            transcript=parsed.transcript,
            category=category,
            distribution=distribution,
            words=words,
        )
        events, styles = build_utterance_spans(
            parsed,
            duration_ms=output.audio.duration_ms,
            word_ids=[word.id for word in output.transcript.words],
        )
        payload = output.model_dump(mode="json")
        payload["events"] = [event.model_dump(mode="json") for event in events]
        payload["styles"] = [style.model_dump(mode="json") for style in styles]
        output = AttuneOutput.model_validate(payload)
        elapsed = time.perf_counter() - started
        return BaselinePrediction(
            output,
            RuntimeMetrics.measured(
                audio_seconds=output.audio.duration_ms / 1000, elapsed_seconds=elapsed
            ),
            diagnostics={
                "affect_source": (
                    "sensevoice_ser"
                    if mapped_affect_trace
                    else (
                        "sensevoice_unmapped_ser_with_transcript_lexicon_fallback"
                        if affect_trace
                        else "transcript_lexicon_fallback"
                    )
                ),
                "raw_model_output": _diagnostic_value(result),
                "raw_affect_label": (str(affect_trace[0]["raw_label"]) if affect_trace else None),
                "schema_affect_label": category.value,
                "affect_mapping": affect_trace,
                "word_alignment": "model_returned" if words else "unavailable",
                "note": (
                    "SenseVoice SER/rich-transcription affect was mapped directly."
                    if mapped_affect_trace
                    else (
                        "SenseVoice emitted an unmapped SER label; transcript lexicon was used."
                        if affect_trace
                        else "SenseVoice emitted no SER label; transcript lexicon was used."
                    )
                ),
            },
        )


class Emotion2VecPlusAdapter(BaselineAdapter):
    """emotion2vec+ affect adapter, retaining a transcript hint as partial ASR."""

    name = "emotion2vec-plus"

    def __init__(
        self,
        checkpoint: Path | None = None,
        calibration: TemperatureCalibration | None = None,
        abstention: ConfidenceAbstention | None = None,
    ) -> None:
        self.checkpoint = checkpoint or _local_checkpoint(
            "ATTUNE_EMOTION2VEC_PLUS_PATH",
            "models--emotion2vec--emotion2vec_plus_base",
        )
        self.calibration = calibration
        self.abstention = abstention
        self._model: Any | None = None

    def availability(self) -> tuple[bool, str | None]:
        if self.checkpoint is None:
            return (
                False,
                "emotion2vec+ weights are not local; set ATTUNE_EMOTION2VEC_PLUS_PATH. "
                "Downloads are disabled.",
            )
        try:
            import funasr  # noqa: F401
        except ImportError:
            return False, "emotion2vec+ requires the optional funasr package."
        return True, None

    def predict(self, item: BaselineInput) -> BaselinePrediction:
        available, reason = self.availability()
        if not available:
            raise BaselineUnavailableError(reason)
        assert self.checkpoint is not None
        started = time.perf_counter()
        from funasr import AutoModel

        with _offline_model_environment():
            if self._model is None:
                self._model = AutoModel(model=str(self.checkpoint), disable_update=True)
            result = self._model.generate(input=str(item.audio_path), granularity="utterance")
        category, raw_distribution = _map_emotion2vec_result(result)
        distribution = raw_distribution
        if self.calibration is not None:
            calibrated = self.calibration.scale_distribution(
                {label.value: value for label, value in raw_distribution.items()}
            )
            distribution = {AffectCategory(label): value for label, value in calibrated.items()}
            category = max(distribution, key=distribution.__getitem__)
        abstain = self.abstention is not None and self.abstention.abstains(
            {label.value: value for label, value in distribution.items()}
        )
        emotion_diagnostics = _emotion2vec_diagnostics(
            result,
            category,
            raw_distribution,
            distribution,
            self.calibration,
            self.abstention,
            abstain,
        )
        output = build_partial_output(
            item,
            model_name=self.name,
            transcript=item.transcript_hint or "",
            category=category,
            distribution=distribution,
            abstain=abstain,
        )
        elapsed = time.perf_counter() - started
        return BaselinePrediction(
            output,
            RuntimeMetrics.measured(
                audio_seconds=output.audio.duration_ms / 1000, elapsed_seconds=elapsed
            ),
            diagnostics=emotion_diagnostics,
        )


def parse_sensevoice_word_timestamps(result: Any, *, duration_ms: int) -> list[dict[str, Any]]:
    """Parse explicit SenseVoice token/word alignment without inferring boundaries."""
    row = result[0] if isinstance(result, list) and result else result
    if not isinstance(row, dict):
        return []
    candidates = row.get("words", row.get("word_timestamps"))
    if candidates is None:
        token_timestamps = row.get("timestamp")
        if not isinstance(token_timestamps, list):
            return []
        normalized_tokens = []
        for token_timestamp in token_timestamps:
            if (
                not isinstance(token_timestamp, list | tuple)
                or len(token_timestamp) != 3
                or not isinstance(token_timestamp[0], str)
                or not all(isinstance(value, int | float) for value in token_timestamp[1:])
            ):
                return []
            normalized_tokens.append(
                {
                    "text": token_timestamp[0],
                    # Official SenseVoice model.py emits token times in seconds.
                    "start_ms": round(float(token_timestamp[1]) * 1000),
                    "end_ms": round(float(token_timestamp[2]) * 1000),
                    "confidence": 0.0,
                }
            )
        return _validated_word_spans(normalized_tokens, duration_ms=duration_ms)
    if not isinstance(candidates, list):
        return []
    normalized = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            return []
        text = candidate.get("word", candidate.get("text"))
        timestamp = candidate.get("timestamp")
        start = candidate.get("start_ms")
        end = candidate.get("end_ms")
        if start is None and end is None and isinstance(timestamp, list | tuple):
            if len(timestamp) != 2:
                return []
            start, end = timestamp
        if (
            not isinstance(text, str)
            or not isinstance(start, int | float)
            or not isinstance(end, int | float)
        ):
            return []
        normalized.append(
            {
                "text": text.strip(),
                "start_ms": round(float(start)),
                "end_ms": round(float(end)),
                "confidence": _bounded_confidence(
                    candidate.get("confidence", candidate.get("score", 0.0))
                ),
            }
        )
    return _validated_word_spans(normalized, duration_ms=duration_ms)


def parse_whisper_word_timestamps(chunks: Any, *, duration_ms: int) -> list[dict[str, Any]]:
    """Parse official Transformers Whisper word chunks, whose times are seconds."""
    if not isinstance(chunks, list):
        return []
    normalized = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            return []
        text = chunk.get("text")
        timestamp = chunk.get("timestamp")
        if (
            not isinstance(text, str)
            or not isinstance(timestamp, list | tuple)
            or len(timestamp) != 2
            or not all(isinstance(value, int | float) for value in timestamp)
        ):
            return []
        normalized.append(
            {
                "text": text.strip(),
                "start_ms": round(float(timestamp[0]) * 1000),
                "end_ms": round(float(timestamp[1]) * 1000),
                "confidence": _bounded_confidence(chunk.get("confidence", 0.0)),
            }
        )
    return _validated_word_spans(normalized, duration_ms=duration_ms)


def _bounded_confidence(value: Any) -> float:
    if not isinstance(value, int | float):
        return 0.0
    return min(1.0, max(0.0, float(value)))


def _validated_word_spans(
    candidates: list[dict[str, Any]], *, duration_ms: int
) -> list[dict[str, Any]]:
    """Accept a complete genuine alignment only when every span is valid."""
    if not candidates:
        return []
    words = []
    previous_start = -1
    for index, candidate in enumerate(candidates, start=1):
        start = candidate["start_ms"]
        end = candidate["end_ms"]
        if (
            not candidate["text"]
            or start < 0
            or end < start
            or end > duration_ms
            or start < previous_start
        ):
            return []
        word = Word(
            id=f"w{index}",
            text=candidate["text"],
            start_ms=start,
            end_ms=end,
            confidence=candidate["confidence"],
        )
        words.append(word.model_dump(mode="json"))
        previous_start = start
    return words


def build_partial_output(
    item: BaselineInput,
    *,
    model_name: str,
    transcript: str,
    category: AffectCategory,
    distribution: dict[AffectCategory, float],
    abstain: bool = False,
    words: list[dict[str, Any]] | None = None,
) -> AttuneOutput:
    """Build a valid contract with documented placeholders for unsupported heads."""
    duration_ms, sample_rate, channels = _wave_info(item.audio_path)
    top_probability = distribution[category]
    return AttuneOutput.model_validate(
        {
            "schema_version": "1.0",
            "model": {"name": model_name, "version": "phase-1", "quantization": None},
            "audio": {
                "duration_ms": duration_ms,
                "sample_rate_hz": sample_rate,
                "channels": channels,
                "quality": placeholder_quality_probabilities(),
            },
            "language": {"label": item.language_hint, "confidence": 0.5},
            # Empty remains authoritative unless a runner returned genuine alignment.
            "transcript": {"text": transcript, "confidence": 0.5, "words": words or []},
            "styles": [],
            "events": [],
            "affect": {
                "start_ms": 0,
                "end_ms": duration_ms,
                "valence": {"value": 0.0, "confidence": 0.0},
                "arousal": {"value": 0.0, "confidence": 0.0},
                "dominance": {"value": 0.0, "confidence": 0.0},
                "categories": distribution,
                "top_label": None if abstain else category,
                "top_label_confidence": top_probability,
                "abstain": abstain,
            },
            "uncertainty": {"out_of_distribution_probability": 0.5},
        }
    )


def _wave_info(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as audio:
        duration_ms = round(audio.getnframes() / audio.getframerate() * 1000)
        return duration_ms, audio.getframerate(), audio.getnchannels()


def _read_pcm16(path: Path) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as audio:
        if audio.getsampwidth() != 2 or audio.getnchannels() != 1:
            raise ValueError("optional adapters require mono PCM16 WAV input")
        values = array("h", audio.readframes(audio.getnframes()))
        return [value / 32768 for value in values], audio.getframerate()


def _local_checkpoint(environment_variable: str, cache_name: str) -> Path | None:
    explicit = os.environ.get(environment_variable)
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.exists() else None
    snapshots = Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots"
    if snapshots.is_dir():
        return next((path for path in snapshots.iterdir() if path.is_dir()), None)
    return None


@contextmanager
def _offline_model_environment() -> Iterator[None]:
    """Prevent supported model hubs and FunASR update checks from using the network."""
    values = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MODELSCOPE_OFFLINE": "1",
        "FUNASR_DISABLE_UPDATE": "1",
    }
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _map_emotion2vec_result(
    result: Any,
) -> tuple[AffectCategory, dict[AffectCategory, float]]:
    if not isinstance(result, list) or not result or not isinstance(result[0], dict):
        raise RuntimeError("emotion2vec+ returned an unsupported result shape")
    row = result[0]
    labels = row.get("labels")
    scores = row.get("scores")
    if not isinstance(labels, list) or not isinstance(scores, list) or len(labels) != len(scores):
        raise RuntimeError("emotion2vec+ result does not contain aligned labels and scores")
    aliases = {
        "neutral": AffectCategory.NEUTRAL,
        "happy": AffectCategory.JOY,
        "sad": AffectCategory.DISTRESS,
        "angry": AffectCategory.ANGER,
        "fearful": AffectCategory.FEAR,
        "disgusted": AffectCategory.OTHER,
        "other": AffectCategory.OTHER,
        "surprised": AffectCategory.SURPRISE,
        "unknown": AffectCategory.AMBIGUOUS,
    }
    distribution = {category: 1e-6 for category in AffectCategory}
    for label, score in zip(labels, scores, strict=True):
        normalized_labels = re.split(r"[/|]", str(label).lower())
        category = next(
            (aliases[normalized] for normalized in normalized_labels if normalized in aliases),
            AffectCategory.OTHER,
        )
        distribution[category] += max(0.0, float(score))
    total = sum(distribution.values())
    distribution = {category: score / total for category, score in distribution.items()}
    category = max(distribution, key=distribution.__getitem__)
    return category, distribution


def _emotion2vec_diagnostics(
    result: Any,
    category: AffectCategory,
    raw_distribution: dict[AffectCategory, float],
    distribution: dict[AffectCategory, float],
    calibration: TemperatureCalibration | None,
    abstention: ConfidenceAbstention | None,
    abstained: bool,
) -> dict[str, Any]:
    row = result[0]
    labels = row["labels"]
    scores = row["scores"]
    raw_index = max(range(len(scores)), key=lambda index: float(scores[index]))
    return {
        "affect_source": "emotion2vec_plus_ser",
        "raw_model_output": {
            "labels": _diagnostic_value(labels),
            "scores": _diagnostic_value(scores),
        },
        "raw_affect_label": str(labels[raw_index]),
        "raw_affect_score": float(scores[raw_index]),
        "schema_affect_label": category.value,
        "mapping_rule": "published emotion2vec+ label aliases; disgust maps to Attune other",
        "uncalibrated_affect_probabilities": {
            label.value: value for label, value in raw_distribution.items()
        },
        "affect_probabilities": {label.value: value for label, value in distribution.items()},
        "calibration_method": ("temperature_scaling" if calibration is not None else None),
        "calibration_temperature": (calibration.temperature if calibration is not None else None),
        "calibration_fitted_on": calibration.fitted_on if calibration is not None else None,
        "affect_abstained": abstained,
        "affect_abstention_method": ("confidence_threshold" if abstention is not None else None),
        "affect_abstention_score": (abstention.score if abstention is not None else None),
        "affect_abstention_threshold": (abstention.threshold if abstention is not None else None),
        "affect_abstention_fitted_on": (abstention.fitted_on if abstention is not None else None),
    }


def _diagnostic_value(value: Any) -> Any:
    """Convert model output containers to bounded JSON-compatible values."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _diagnostic_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_diagnostic_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _diagnostic_value(value.tolist())
    return repr(value)
