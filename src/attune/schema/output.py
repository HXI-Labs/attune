"""Authoritative schema_version 1.0 JSON contract."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
Timestamp = Annotated[int, Field(ge=0)]
AffectValue = Annotated[float, Field(ge=-1.0, le=1.0)]

DEFAULT_INTERPRETATION_WARNING = (
    "Vocal affect is a probabilistic perception, not a verified internal state."
)


class StrictModel(BaseModel):
    """Base for a versioned contract that rejects accidental fields."""

    model_config = ConfigDict(extra="forbid")


class Status(StrEnum):
    PROVISIONAL = "provisional"
    REVISED = "revised"
    COMMITTED = "committed"
    RETRACTED = "retracted"


class StyleLabel(StrEnum):
    SHOUTING = "shouting"
    WHISPERING = "whispering"
    CRYING_SPEECH = "crying_speech"
    LAUGHING_SPEECH = "laughing_speech"
    SINGING = "singing"
    STRAINED_SPEECH = "strained_speech"


class EventLabel(StrEnum):
    LAUGH = "laugh"
    SOB = "sob"
    SCREAM = "scream"
    SIGH = "sigh"
    COUGH = "cough"
    THROAT_CLEAR = "throat_clear"
    SNEEZE = "sneeze"
    BREATH = "breath"


class AffectCategory(StrEnum):
    NEUTRAL = "neutral"
    JOY = "joy"
    DISTRESS = "distress"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    OTHER = "other"
    AMBIGUOUS = "ambiguous"


class TimedSpan(StrictModel):
    start_ms: Timestamp
    end_ms: Timestamp

    @model_validator(mode="after")
    def end_not_before_start(self) -> TimedSpan:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        return self


class ModelInfo(StrictModel):
    name: str
    version: str
    quantization: str | None = None


class QualityProbabilities(StrictModel):
    clipping: Probability
    low_snr: Probability
    far_field: Probability


class AudioInfo(StrictModel):
    duration_ms: Annotated[int, Field(gt=0)]
    sample_rate_hz: Annotated[int, Field(gt=0)]
    channels: Annotated[int, Field(gt=0)]
    quality: QualityProbabilities


class LanguageInfo(StrictModel):
    label: str
    confidence: Probability


class Word(TimedSpan):
    id: str
    text: str
    confidence: Probability


class Transcript(StrictModel):
    text: str
    confidence: Probability
    words: list[Word]

    @model_validator(mode="after")
    def words_are_ordered(self) -> Transcript:
        for previous, current in zip(self.words, self.words[1:], strict=False):
            if current.start_ms < previous.start_ms:
                raise ValueError("words must be ordered by start_ms")
        ids = [word.id for word in self.words]
        if len(ids) != len(set(ids)):
            raise ValueError("word ids must be unique")
        return self


class VocalStyle(TimedSpan):
    id: str
    label: StyleLabel
    start_word_id: str | None = None
    end_word_id: str | None = None
    confidence: Probability
    status: Status


class VocalEvent(TimedSpan):
    id: str
    label: EventLabel
    after_word_id: str | None = None
    confidence: Probability
    status: Status


class AffectDimension(StrictModel):
    value: AffectValue
    confidence: Probability


class Affect(StrictModel):
    start_ms: Timestamp
    end_ms: Timestamp
    valence: AffectDimension
    arousal: AffectDimension
    dominance: AffectDimension
    categories: dict[AffectCategory, Probability]
    top_label: AffectCategory | None
    top_label_confidence: Probability
    abstain: bool

    @model_validator(mode="after")
    def validate_affect(self) -> Affect:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        expected = set(AffectCategory)
        if set(self.categories) != expected:
            raise ValueError("categories must contain every affect category exactly once")
        if abs(sum(self.categories.values()) - 1.0) > 1e-6:
            raise ValueError("affect category probabilities must sum to 1")
        if self.abstain and self.top_label is not None:
            raise ValueError("top_label must be null when abstain is true")
        if not self.abstain and self.top_label is None:
            raise ValueError("top_label is required when abstain is false")
        if not self.abstain and self.top_label is not None:
            maximum = max(self.categories.values())
            if self.categories[self.top_label] != maximum:
                raise ValueError("top_label must identify a maximum-probability category")
            if abs(self.top_label_confidence - self.categories[self.top_label]) > 1e-6:
                raise ValueError("top_label_confidence must equal the top category probability")
        return self


class Uncertainty(StrictModel):
    out_of_distribution_probability: Probability
    interpretation_warning: str = DEFAULT_INTERPRETATION_WARNING


class AttuneOutput(StrictModel):
    schema_version: Literal["1.0"]
    model: ModelInfo
    audio: AudioInfo
    language: LanguageInfo
    transcript: Transcript
    styles: list[VocalStyle] = Field(default_factory=list)
    events: list[VocalEvent] = Field(default_factory=list)
    affect: Affect
    uncertainty: Uncertainty

    @model_validator(mode="after")
    def validate_references_and_bounds(self) -> AttuneOutput:
        words_by_id = {word.id: word for word in self.transcript.words}
        if any(word.end_ms > self.audio.duration_ms for word in self.transcript.words):
            raise ValueError("word timestamps must be within audio duration")
        if any(style.end_ms > self.audio.duration_ms for style in self.styles):
            raise ValueError("style timestamps must be within audio duration")
        if any(event.end_ms > self.audio.duration_ms for event in self.events):
            raise ValueError("event timestamps must be within audio duration")
        if self.affect.end_ms > self.audio.duration_ms:
            raise ValueError("affect timestamps must be within audio duration")

        style_ids = [style.id for style in self.styles]
        event_ids = [event.id for event in self.events]
        if len(style_ids) != len(set(style_ids)):
            raise ValueError("style ids must be unique")
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event ids must be unique")

        for style in self.styles:
            if style.start_word_id is not None and style.start_word_id not in words_by_id:
                raise ValueError(f"unknown start_word_id: {style.start_word_id}")
            if style.end_word_id is not None and style.end_word_id not in words_by_id:
                raise ValueError(f"unknown end_word_id: {style.end_word_id}")
            if (
                style.start_word_id is not None
                and style.end_word_id is not None
                and words_by_id[style.end_word_id].start_ms
                < words_by_id[style.start_word_id].start_ms
            ):
                raise ValueError("style word references must be ordered")
        for event in self.events:
            if event.after_word_id is not None and event.after_word_id not in words_by_id:
                raise ValueError(f"unknown after_word_id: {event.after_word_id}")
        return self
