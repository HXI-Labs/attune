"""Human review ledger contract for weak-label inspection rows."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from attune.schema.output import AffectCategory, EventLabel, StyleLabel


class ReviewModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TranscriptReview(ReviewModel):
    decision: Literal["accept", "reject", "not_reviewable"]
    corrected_text: str | None = None

    @model_validator(mode="after")
    def correction_matches_decision(self) -> TranscriptReview:
        if self.decision == "reject" and not self.corrected_text:
            raise ValueError("rejected transcript requires corrected_text")
        if self.decision != "reject" and self.corrected_text is not None:
            raise ValueError("corrected_text is allowed only for reject")
        return self


class AffectReview(ReviewModel):
    decision: Literal["accept", "reject", "ambiguous", "not_reviewable"]
    reviewed_label: AffectCategory | None = None

    @model_validator(mode="after")
    def label_matches_decision(self) -> AffectReview:
        if self.decision == "reject" and self.reviewed_label is None:
            raise ValueError("rejected affect requires reviewed_label")
        if self.decision == "ambiguous" and self.reviewed_label != AffectCategory.AMBIGUOUS:
            raise ValueError("ambiguous affect requires the ambiguous label")
        if self.decision in {"accept", "not_reviewable"} and self.reviewed_label is not None:
            raise ValueError("reviewed_label is not allowed for this affect decision")
        return self


class SpanReview(ReviewModel):
    channel: Literal["event", "style"]
    source_label: str | None = None
    decision: Literal["accept", "reject", "retime", "add"]
    reviewed_label: EventLabel | StyleLabel | None = None
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def span_matches_decision(self) -> SpanReview:
        if self.channel == "event" and isinstance(self.reviewed_label, StyleLabel):
            raise ValueError("event review cannot use a style label")
        if self.channel == "style" and isinstance(self.reviewed_label, EventLabel):
            raise ValueError("style review cannot use an event label")
        requires_span = self.decision in {"retime", "add"}
        if requires_span and (self.start_ms is None or self.end_ms is None):
            raise ValueError("retime/add requires start_ms and end_ms")
        if not requires_span and (self.start_ms is not None or self.end_ms is not None):
            raise ValueError("accept/reject must not replace span boundaries")
        if self.start_ms is not None and self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError("end_ms must not precede start_ms")
        if self.decision == "add" and self.reviewed_label is None:
            raise ValueError("added span requires reviewed_label")
        if self.decision == "retime" and self.source_label is None:
            raise ValueError("retimed span requires source_label")
        return self


class GoldReviewRecord(ReviewModel):
    schema_version: Literal["1.0"] = "1.0"
    clip_id: str
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer: str
    reviewed_at_utc: str
    source_label_status: Literal["weak_source_or_acted"]
    transcript: TranscriptReview
    affect: AffectReview
    spans: list[SpanReview]
    notes: str = ""
    dry_run_fixture: bool = False

    @model_validator(mode="after")
    def unique_span_reviews(self) -> GoldReviewRecord:
        identities = [
            (span.channel, span.source_label, span.decision)
            for span in self.spans
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate span review decisions")
        return self
