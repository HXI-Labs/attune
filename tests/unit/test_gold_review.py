import pytest
from pydantic import ValidationError

from attune.data.gold_review import GoldReviewRecord, SpanReview


def test_retimed_span_requires_bounded_offsets() -> None:
    review = SpanReview(
        channel="event",
        source_label="laugh",
        decision="retime",
        reviewed_label="laugh",
        start_ms=100,
        end_ms=250,
    )

    assert review.end_ms == 250


def test_retimed_span_rejects_missing_or_reversed_offsets() -> None:
    with pytest.raises(ValidationError, match="requires start_ms"):
        SpanReview(
            channel="event",
            source_label="laugh",
            decision="retime",
            reviewed_label="laugh",
        )
    with pytest.raises(ValidationError, match="must not precede"):
        SpanReview(
            channel="event",
            source_label="laugh",
            decision="retime",
            reviewed_label="laugh",
            start_ms=250,
            end_ms=100,
        )


def test_real_review_cannot_use_an_invalid_audio_hash() -> None:
    with pytest.raises(ValidationError, match="audio_sha256"):
        GoldReviewRecord.model_validate(
            {
                "clip_id": "clip",
                "audio_sha256": "not-a-hash",
                "reviewer": "reviewer",
                "reviewed_at_utc": "2026-08-27T00:00:00Z",
                "source_label_status": "weak_source_or_acted",
                "transcript": {"decision": "accept"},
                "affect": {"decision": "accept"},
                "spans": [],
            }
        )
