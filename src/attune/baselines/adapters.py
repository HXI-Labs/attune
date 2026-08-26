"""Offline-safe baseline adapters that preserve the Attune schema contract."""

from __future__ import annotations

import os
import re
import time
import wave
from abc import ABC, abstractmethod
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from attune.audio.contracts import placeholder_quality_probabilities
from attune.evaluation.report import RuntimeMetrics
from attune.schema.output import AffectCategory, AttuneOutput

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
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        processor = AutoProcessor.from_pretrained(self.checkpoint, local_files_only=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(self.checkpoint, local_files_only=True)
        inputs = processor(samples, sampling_rate=sample_rate, return_tensors="pt")
        generated_ids = model.generate(inputs.input_features)
        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        category, distribution = TranscriptSentimentAdapter().classify(transcript)
        output = build_partial_output(
            item,
            model_name=self.name,
            transcript=transcript,
            category=category,
            distribution=distribution,
        )
        elapsed = time.perf_counter() - started
        return BaselinePrediction(
            output,
            RuntimeMetrics.measured(
                audio_seconds=output.audio.duration_ms / 1000, elapsed_seconds=elapsed
            ),
        )


class SenseVoiceSmallAdapter(BaselineAdapter):
    """SenseVoice-Small adapter with an explicit licence and local-weight gate."""

    name = "sensevoice-small"

    def __init__(self, checkpoint: Path | None = None) -> None:
        self.checkpoint = checkpoint or _local_checkpoint(
            "ATTUNE_SENSEVOICE_SMALL_PATH",
            "models--FunAudioLLM--SenseVoiceSmall",
        )

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

        model = AutoModel(model=str(self.checkpoint), disable_update=True)
        result = model.generate(input=str(item.audio_path), cache={}, language="auto")
        transcript = _extract_funasr_text(result)
        category, distribution = TranscriptSentimentAdapter().classify(transcript)
        output = build_partial_output(
            item,
            model_name=self.name,
            transcript=transcript,
            category=category,
            distribution=distribution,
        )
        elapsed = time.perf_counter() - started
        return BaselinePrediction(
            output,
            RuntimeMetrics.measured(
                audio_seconds=output.audio.duration_ms / 1000, elapsed_seconds=elapsed
            ),
        )


class Emotion2VecPlusAdapter(BaselineAdapter):
    """emotion2vec+ affect adapter, retaining a transcript hint as partial ASR."""

    name = "emotion2vec-plus"

    def __init__(self, checkpoint: Path | None = None) -> None:
        self.checkpoint = checkpoint or _local_checkpoint(
            "ATTUNE_EMOTION2VEC_PLUS_PATH",
            "models--iic--emotion2vec_plus_large",
        )

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

        model = AutoModel(model=str(self.checkpoint), disable_update=True)
        result = model.generate(input=str(item.audio_path), granularity="utterance")
        category, distribution = _map_emotion2vec_result(result)
        output = build_partial_output(
            item,
            model_name=self.name,
            transcript=item.transcript_hint or "",
            category=category,
            distribution=distribution,
        )
        elapsed = time.perf_counter() - started
        return BaselinePrediction(
            output,
            RuntimeMetrics.measured(
                audio_seconds=output.audio.duration_ms / 1000, elapsed_seconds=elapsed
            ),
        )


def build_partial_output(
    item: BaselineInput,
    *,
    model_name: str,
    transcript: str,
    category: AffectCategory,
    distribution: dict[AffectCategory, float],
) -> AttuneOutput:
    """Build a valid contract with documented placeholders for unsupported heads."""
    duration_ms, sample_rate, channels = _wave_info(item.audio_path)
    words = transcript.split()
    word_rows = []
    for index, word in enumerate(words):
        start_ms = round(index * duration_ms / max(len(words), 1))
        end_ms = round((index + 1) * duration_ms / max(len(words), 1))
        word_rows.append(
            {
                "id": f"w{index + 1}",
                "text": word,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "confidence": 0.5,
            }
        )
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
            "transcript": {"text": transcript, "confidence": 0.5, "words": word_rows},
            "styles": [],
            "events": [],
            "affect": {
                "start_ms": 0,
                "end_ms": duration_ms,
                "valence": {"value": 0.0, "confidence": 0.0},
                "arousal": {"value": 0.0, "confidence": 0.0},
                "dominance": {"value": 0.0, "confidence": 0.0},
                "categories": distribution,
                "top_label": category,
                "top_label_confidence": top_probability,
                "abstain": False,
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


def _extract_funasr_text(result: Any) -> str:
    if isinstance(result, list) and result and isinstance(result[0], dict):
        text = str(result[0].get("text", ""))
        return re.sub(r"<\|.*?\|>", "", text).strip()
    raise RuntimeError("SenseVoice returned an unsupported result shape")


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
        "surprised": AffectCategory.SURPRISE,
    }
    distribution = {category: 1e-6 for category in AffectCategory}
    for label, score in zip(labels, scores, strict=True):
        category = aliases.get(str(label).lower(), AffectCategory.OTHER)
        distribution[category] += max(0.0, float(score))
    total = sum(distribution.values())
    distribution = {category: score / total for category, score in distribution.items()}
    category = max(distribution, key=distribution.__getitem__)
    return category, distribution
