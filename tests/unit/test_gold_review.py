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
                "audio_duration_ms": 500,
                "reviewer": "reviewer",
                "reviewed_at_utc": "2026-08-27T00:00:00Z",
                "source_label_status": "weak_source_or_acted",
                "transcript": {"decision": "accept"},
                "affect": {"decision": "accept"},
                "spans": [],
            }
        )


def test_human_100ms_activity_status_is_valid_and_not_gold() -> None:
    review = GoldReviewRecord.model_validate(
        {
            "clip_id": "clip",
            "audio_sha256": "a" * 64,
            "audio_duration_ms": 500,
            "reviewer": "reviewer",
            "reviewed_at_utc": "2026-08-27T00:00:00Z",
            "source_label_status": "human_100ms_activity_not_attune_gold",
            "transcript": {"decision": "not_reviewable"},
            "affect": {"decision": "not_reviewable"},
            "spans": [],
        }
    )
    assert review.source_label_status == "human_100ms_activity_not_attune_gold"
    from attune.data.gold_review import evaluate_gold_promotion

    result = evaluate_gold_promotion([review])
    assert result.gold is False
    assert result.status == "reviewer_labelled"


def test_weak_source_or_acted_still_validates() -> None:
    review = GoldReviewRecord.model_validate(
        {
            "clip_id": "clip",
            "audio_sha256": "a" * 64,
            "audio_duration_ms": 500,
            "reviewer": "reviewer",
            "reviewed_at_utc": "2026-08-27T00:00:00Z",
            "source_label_status": "weak_source_or_acted",
            "transcript": {"decision": "accept"},
            "affect": {"decision": "accept"},
            "spans": [],
            "dry_run_fixture": True,
        }
    )
    assert review.source_label_status == "weak_source_or_acted"
    assert review.dry_run_fixture is True


def test_committed_dry_run_fixture_still_uses_weak_source_or_acted() -> None:
    import json
    from pathlib import Path as _Path

    fixture = (
        _Path(__file__).resolve().parents[2]
        / "research/error-analysis/gold-review-fixture-dry-run.jsonl"
    )
    rows = [
        json.loads(line)
        for line in fixture.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    for row in rows:
        review = GoldReviewRecord.model_validate(row)
        assert review.source_label_status == "weak_source_or_acted"
        assert review.dry_run_fixture is True
        assert review.transcript.decision == "accept"
