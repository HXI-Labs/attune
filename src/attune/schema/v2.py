"""Authoritative Attune schema v2.0.

Version 2 makes evidence scope explicit.  A whole-utterance classification is
not represented as a fake ``0..duration`` localization, and unavailable affect
dimensions or audio-quality estimates are carried as unavailable rather than
invented numeric probabilities.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from attune.schema.output import (
    DEFAULT_INTERPRETATION_WARNING,
    AffectCategory,
    EventLabel,
    Status,
    StyleLabel,
)

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
Timestamp = Annotated[int, Field(ge=0)]
AffectValue = Annotated[float, Field(ge=-1.0, le=1.0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TemporalScope(StrEnum):
    LOCALIZED = "localized"
    UTTERANCE = "utterance"


class EstimateMethod(StrEnum):
    CALIBRATED = "calibrated"
    HEURISTIC = "heuristic"
    UNAVAILABLE = "unavailable"


class ModelInfo(StrictModel):
    name: str
    version: str
    quantization: str | None = None


class QualityEstimate(StrictModel):
    value: Probability | None
    method: EstimateMethod

    @model_validator(mode="after")
    def availability_matches_value(self) -> QualityEstimate:
        if self.method == EstimateMethod.UNAVAILABLE and self.value is not None:
            raise ValueError("unavailable quality estimates must have a null value")
        if self.method != EstimateMethod.UNAVAILABLE and self.value is None:
            raise ValueError("available quality estimates require a value")
        return self


class AudioQuality(StrictModel):
    clipping: QualityEstimate
    low_snr: QualityEstimate
    far_field: QualityEstimate


class AudioInfo(StrictModel):
    duration_ms: Annotated[int, Field(gt=0)]
    sample_rate_hz: Annotated[int, Field(gt=0)]
    channels: Annotated[int, Field(gt=0)]
    quality: AudioQuality


class LanguageInfo(StrictModel):
    label: str
    confidence: Probability


class Word(StrictModel):
    id: str
    text: str
    start_ms: Timestamp
    end_ms: Timestamp
    confidence: Probability

    @model_validator(mode="after")
    def end_not_before_start(self) -> Word:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        return self


class Transcript(StrictModel):
    text: str
    confidence: Probability
    words: list[Word]

    @model_validator(mode="after")
    def words_are_ordered(self) -> Transcript:
        if any(
            current.start_ms < previous.start_ms
            for previous, current in zip(self.words, self.words[1:], strict=False)
        ):
            raise ValueError("words must be ordered by start_ms")
        identifiers = [word.id for word in self.words]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("word ids must be unique")
        return self


class ScopedAnnotation(StrictModel):
    id: str
    temporal_scope: TemporalScope
    start_ms: Timestamp | None = None
    end_ms: Timestamp | None = None
    confidence: Probability
    status: Status

    @model_validator(mode="after")
    def validate_scope(self) -> ScopedAnnotation:
        if self.temporal_scope == TemporalScope.LOCALIZED:
            if self.start_ms is None or self.end_ms is None:
                raise ValueError("localized annotations require start_ms and end_ms")
            if self.end_ms < self.start_ms:
                raise ValueError("end_ms must be greater than or equal to start_ms")
        elif self.start_ms is not None or self.end_ms is not None:
            raise ValueError("utterance-scope annotations must not carry timestamps")
        return self


class VocalStyle(ScopedAnnotation):
    label: StyleLabel
    start_word_id: str | None = None
    end_word_id: str | None = None

    @model_validator(mode="after")
    def word_references_require_localization(self) -> VocalStyle:
        if self.temporal_scope == TemporalScope.UTTERANCE and (
            self.start_word_id is not None or self.end_word_id is not None
        ):
            raise ValueError("utterance-scope styles must not carry word boundaries")
        return self


class VocalEvent(ScopedAnnotation):
    label: EventLabel
    after_word_id: str | None = None


class AffectDimension(StrictModel):
    value: AffectValue | None
    confidence: Probability
    available: bool

    @model_validator(mode="after")
    def validate_availability(self) -> AffectDimension:
        if self.available and self.value is None:
            raise ValueError("available affect dimensions require a value")
        if not self.available and (self.value is not None or self.confidence != 0.0):
            raise ValueError("unavailable affect dimensions require null value and zero confidence")
        return self


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
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def validate_affect(self) -> Affect:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        if set(self.categories) != set(AffectCategory):
            raise ValueError("categories must contain every affect category exactly once")
        if abs(sum(self.categories.values()) - 1.0) > 1e-6:
            raise ValueError("affect category probabilities must sum to 1")
        if self.abstain:
            if self.top_label is not None:
                raise ValueError("top_label must be null when abstain is true")
            if not self.abstention_reason:
                raise ValueError("abstaining outputs require abstention_reason")
        else:
            if self.top_label is None:
                raise ValueError("top_label is required when abstain is false")
            if self.categories[self.top_label] != max(self.categories.values()):
                raise ValueError("top_label must identify a maximum-probability category")
            if abs(self.top_label_confidence - self.categories[self.top_label]) > 1e-6:
                raise ValueError("top_label_confidence must equal the top category probability")
            if self.abstention_reason is not None:
                raise ValueError("non-abstaining outputs must not carry abstention_reason")
        return self


class Uncertainty(StrictModel):
    out_of_distribution_probability: Probability
    interpretation_warning: str = DEFAULT_INTERPRETATION_WARNING


class AttuneOutputV2(StrictModel):
    schema_version: Literal["2.0"]
    model: ModelInfo
    audio: AudioInfo
    language: LanguageInfo
    transcript: Transcript
    styles: list[VocalStyle] = Field(default_factory=list)
    events: list[VocalEvent] = Field(default_factory=list)
    affect: Affect
    affect_spans: list[Affect] = Field(default_factory=list)
    uncertainty: Uncertainty

    @model_validator(mode="after")
    def validate_references_and_bounds(self) -> AttuneOutputV2:
        words_by_id = {word.id: word for word in self.transcript.words}
        timed = [*self.transcript.words]
        timed.extend(item for item in [*self.styles, *self.events] if item.end_ms is not None)
        if any(item.end_ms is not None and item.end_ms > self.audio.duration_ms for item in timed):
            raise ValueError("timestamps must be within audio duration")
        if self.affect.end_ms > self.audio.duration_ms:
            raise ValueError("affect timestamps must be within audio duration")
        if any(span.end_ms > self.audio.duration_ms for span in self.affect_spans):
            raise ValueError("affect span timestamps must be within audio duration")
        if any(
            current.start_ms < previous.end_ms
            for previous, current in zip(self.affect_spans, self.affect_spans[1:], strict=False)
        ):
            raise ValueError("affect spans must be ordered and non-overlapping")
        for collection, name in ((self.styles, "style"), (self.events, "event")):
            identifiers = [item.id for item in collection]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"{name} ids must be unique")
        for style in self.styles:
            for field_name in ("start_word_id", "end_word_id"):
                word_id = getattr(style, field_name)
                if word_id is not None and word_id not in words_by_id:
                    raise ValueError(f"unknown {field_name}: {word_id}")
            if (
                style.start_word_id
                and style.end_word_id
                and words_by_id[style.end_word_id].start_ms
                < words_by_id[style.start_word_id].start_ms
            ):
                raise ValueError("style word references must be ordered")
        for event in self.events:
            if event.after_word_id is not None and event.after_word_id not in words_by_id:
                raise ValueError(f"unknown after_word_id: {event.after_word_id}")
        return self
