from __future__ import annotations

import hashlib
import math
import wave
from array import array
from pathlib import Path

import pytest

from attune.data.gold_review import (
    HUMAN_100MS_ACTIVITY_NOT_ATTUNE_GOLD,
    WEAK_SOURCE_OR_ACTED,
    GoldReviewRecord,
    evaluate_gold_promotion,
)
from attune.data.starss23_gold_pack import (
    EXPECTED_FIRST_60S_CLIPS,
    EXPECTED_FIRST_60S_LAUGH_EVENTS,
    EXPECTED_FIRST_60S_TRUE_NEGATIVES,
    EXPECTED_WINDOW_END_TRUNCATED_LAUGHS,
    PACK_RELATIVE_PATH,
    WINDOW_END_TRUNCATED_LAUGHS,
    AudioHashMismatchError,
    Starss23ReviewStatusError,
    annotate_pack_row,
    append_ledger,
    assert_expected_pack_counts,
    build_pack_rows,
    first_60s_rows,
    is_true_negative,
    laugh_event_count,
    load_jsonl,
    resolve_pack_audio,
    source_span_id,
    true_negative_count,
    validate_starss23_review,
    verify_audio_sha256,
    write_html_pack,
    write_jsonl,
)

REPO = Path(__file__).resolve().parents[2]


def write_tone(path: Path, duration_ms: int = 1000, frequency: float = 440.0) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16_000
    frame_count = round(sample_rate * duration_ms / 1000)
    samples = array(
        "h",
        (
            round(12000 * math.sin(2 * math.pi * frequency * index / sample_rate))
            for index in range(frame_count)
        ),
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return _sha256(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspection_row(
    *,
    clip_id: str,
    sha256: str,
    window_start_ms: int = 0,
    events: list[dict] | None = None,
    duration_ms: int = 1000,
    cache_path: str | None = None,
) -> dict:
    return {
        "attribution": "synthetic",
        "cache_path": cache_path or f"inspection_test/{clip_id}.wav",
        "channels": 1,
        "clip_id": clip_id,
        "downmix": "mean_of_4_tetrahedral_mic",
        "duration_ms": duration_ms,
        "events": events or [],
        "label_status": "STARSS23 human activity label; not reviewed Attune gold",
        "language": "unverified",
        "licence": "MIT",
        "sample_rate_hz": 16000,
        "sha256": sha256,
        "source_recording": f"{clip_id}.wav",
        "source_window_start_ms": window_start_ms,
        "clipped_spanning_event_count": 0,
    }


def review_payload(**overrides: object) -> dict:
    payload = {
        "clip_id": "clip-a",
        "audio_sha256": "a" * 64,
        "audio_duration_ms": 1000,
        "reviewer": "reviewer-1",
        "reviewed_at_utc": "2026-08-27T13:00:00Z",
        "source_label_status": HUMAN_100MS_ACTIVITY_NOT_ATTUNE_GOLD,
        "transcript": {"decision": "not_reviewable"},
        "affect": {"decision": "not_reviewable"},
        "spans": [],
        "notes": "synthetic tone; not gold",
        "dry_run_fixture": False,
    }
    payload.update(overrides)
    return payload


def test_committed_pack_is_exactly_49_clips_and_48_laughs() -> None:
    rows = load_jsonl(REPO / PACK_RELATIVE_PATH)
    assert_expected_pack_counts(rows)
    assert len(rows) == EXPECTED_FIRST_60S_CLIPS
    assert laugh_event_count(rows) == EXPECTED_FIRST_60S_LAUGH_EVENTS
    assert all(row["source_window_start_ms"] == 0 for row in rows)
    assert all(row["source_label_status"] == HUMAN_100MS_ACTIVITY_NOT_ATTUNE_GOLD for row in rows)
    assert {row["sha256"] for row in rows} == {row["sha256"] for row in rows}
    assert len({row["sha256"] for row in rows}) == EXPECTED_FIRST_60S_CLIPS
    assert len({row["clip_id"] for row in rows}) == EXPECTED_FIRST_60S_CLIPS


def test_later_tiles_are_excluded_from_the_pack() -> None:
    rows = [
        inspection_row(
            clip_id="first",
            sha256="a" * 64,
            window_start_ms=0,
            events=[{"label": "laugh", "start_ms": 100, "end_ms": 400, "source_class": 4}],
        ),
        inspection_row(
            clip_id="tile",
            sha256="b" * 64,
            window_start_ms=60_000,
            events=[{"label": "laugh", "start_ms": 0, "end_ms": 300, "source_class": 4}],
        ),
    ]
    packed = first_60s_rows(rows)
    assert [row["clip_id"] for row in packed] == ["first"]
    with pytest.raises(ValueError, match="exactly 49"):
        build_pack_rows(rows)


def test_hash_check_fails_closed(tmp_path: Path) -> None:
    good = tmp_path / "good.wav"
    bad = tmp_path / "bad.wav"
    digest = write_tone(good, frequency=440.0)
    write_tone(bad, frequency=220.0)
    assert _sha256(bad) != digest
    with pytest.raises(AudioHashMismatchError, match="sha256 mismatch"):
        verify_audio_sha256(bad, digest)
    row = inspection_row(
        clip_id="clip-a",
        sha256=digest,
        cache_path="good.wav",
        events=[{"label": "laugh", "start_ms": 100, "end_ms": 400, "source_class": 4}],
    )
    # Explicit path exists with the wrong bytes.
    mismatch_dir = tmp_path / "mismatch"
    mismatch_dir.mkdir()
    (mismatch_dir / "good.wav").write_bytes(bad.read_bytes())
    with pytest.raises(AudioHashMismatchError, match="refusing to invent"):
        resolve_pack_audio([row], [mismatch_dir])


def test_retime_writes_valid_starss23_gold_review_record(tmp_path: Path) -> None:
    wav = tmp_path / "tone.wav"
    digest = write_tone(wav)
    event = {"label": "laugh", "start_ms": 100, "end_ms": 400, "source_class": 4}
    pack_row = inspection_row(clip_id="clip-a", sha256=digest, events=[event])
    pack_row["source_label_status"] = HUMAN_100MS_ACTIVITY_NOT_ATTUNE_GOLD
    record = GoldReviewRecord.model_validate(
        review_payload(
            audio_sha256=digest,
            spans=[
                {
                    "channel": "event",
                    "source_label": source_span_id(event),
                    "decision": "retime",
                    "reviewed_label": "laugh",
                    "start_ms": 80,
                    "end_ms": 510,
                }
            ],
        )
    )
    validate_starss23_review(record, pack_row)
    ledger = tmp_path / "ledger.jsonl"
    append_ledger(ledger, record, pack_row)
    loaded = GoldReviewRecord.model_validate_json(ledger.read_text().splitlines()[0])
    assert loaded.source_label_status == HUMAN_100MS_ACTIVITY_NOT_ATTUNE_GOLD
    assert loaded.spans[0].decision == "retime"
    assert loaded.spans[0].start_ms == 80
    assert loaded.dry_run_fixture is False
    promotion = evaluate_gold_promotion([loaded])
    assert promotion.gold is False
    assert promotion.status == "reviewer_labelled"


def test_starss23_ledger_rejects_weak_source_or_acted() -> None:
    pack_row = inspection_row(clip_id="clip-a", sha256="a" * 64)
    record = GoldReviewRecord.model_validate(
        review_payload(source_label_status=WEAK_SOURCE_OR_ACTED)
    )
    with pytest.raises(Starss23ReviewStatusError, match="human_100ms_activity_not_attune_gold"):
        validate_starss23_review(record, pack_row)


def test_promotion_helper_does_not_mark_gold_from_a_single_pass() -> None:
    record = GoldReviewRecord.model_validate(review_payload())
    result = evaluate_gold_promotion([record], adjudication_resolved=True)
    assert result.gold is False
    assert result.status == "reviewer_labelled"
    second = GoldReviewRecord.model_validate(review_payload(reviewer="reviewer-2"))
    waiting = evaluate_gold_promotion([record, second], adjudication_resolved=False)
    assert waiting.gold is False
    assert waiting.status == "needs_adjudication"


def test_html_pack_from_synthetic_tones_does_not_embed_model_scores(
    tmp_path: Path,
) -> None:
    wav = tmp_path / "cache" / "clip-a.wav"
    digest = write_tone(wav)
    event = {"label": "laugh", "start_ms": 120, "end_ms": 480, "source_class": 4}
    row = inspection_row(
        clip_id="clip-a",
        sha256=digest,
        cache_path="clip-a.wav",
        events=[event],
    )
    output = tmp_path / "html"
    resolved, missing, _ = resolve_pack_audio([row], [tmp_path / "cache"])
    assert missing == []
    summary = write_html_pack([row], output, resolved, missing)
    index = (output / "index.html").read_text(encoding="utf-8")
    page = (output / "clips" / "clip-a.html").read_text(encoding="utf-8")
    assert "Gold gate CLOSED" in index
    assert "Not gold" in page
    assert "model scores are not provided" in page.lower()
    assert "must not be treated as answers" in page.lower()
    assert '"confidence"' not in page
    assert source_span_id(event) in page
    assert summary["copied_wavs"] == 1
    assert (output / "audio" / "clip-a.wav").is_file()
    verify_audio_sha256(output / "audio" / "clip-a.wav", digest)


def test_write_jsonl_roundtrip_keeps_hashes(tmp_path: Path) -> None:
    rows = [
        inspection_row(
            clip_id="clip-a",
            sha256="ab" * 32,
            events=[{"label": "laugh", "start_ms": 0, "end_ms": 300, "source_class": 4}],
        )
    ]
    path = tmp_path / "pack.jsonl"
    write_jsonl(path, rows)
    loaded = load_jsonl(path)
    assert loaded[0]["sha256"] == "ab" * 32


def test_committed_pack_keeps_true_negatives_and_flags_window_end_truncation() -> None:
    rows = load_jsonl(REPO / PACK_RELATIVE_PATH)
    assert true_negative_count(rows) == EXPECTED_FIRST_60S_TRUE_NEGATIVES
    assert sum(1 for row in rows if is_true_negative(row)) == 29
    found: list[tuple[str, int, int]] = []
    for row in rows:
        for event in row.get("events", []):
            if event.get("label") != "laugh":
                continue
            start, end = event["start_ms"], event["end_ms"]
            if event.get("truncated_at_window_end") or end == row["duration_ms"]:
                found.append((row["clip_id"], start, end))
                assert event["start_ms"] == start
                assert event["end_ms"] == end
                assert event.get("incomplete") is True
                assert event.get("collar_eligible") is False
                assert event.get("truncated_at_window_end") is True
            else:
                assert event.get("collar_eligible") is True
                assert event.get("incomplete") is False
        assert row.get("clipped_spanning_event_count") == 0
    assert len(found) == EXPECTED_WINDOW_END_TRUNCATED_LAUGHS
    assert set(found) == set(WINDOW_END_TRUNCATED_LAUGHS)


def test_window_end_event_is_truncated_even_if_clipped_spanning_count_is_zero() -> None:
    event = {"label": "laugh", "start_ms": 59400, "end_ms": 60000, "source_class": 4}
    row = inspection_row(
        clip_id="starss23-scene-inspection_test-fold4_room24_mix006-w000000",
        sha256="a" * 64,
        duration_ms=60_000,
        events=[dict(event)],
    )
    row["clipped_spanning_event_count"] = 0
    packed = annotate_pack_row(row)
    annotated = packed["events"][0]
    assert packed["clipped_spanning_event_count"] == 0
    assert annotated["start_ms"] == 59400
    assert annotated["end_ms"] == 60000
    assert annotated["truncated_at_window_end"] is True
    assert annotated["incomplete"] is True
    assert annotated["collar_eligible"] is False
    assert packed["window_end_truncated_event_count"] == 1


def test_starss23_review_rejects_accept_and_requires_not_reviewable_fields() -> None:
    pack_row = inspection_row(clip_id="clip-a", sha256="a" * 64)
    accepted = GoldReviewRecord.model_validate(
        review_payload(
            spans=[
                {
                    "channel": "event",
                    "source_label": "laugh@100-400",
                    "decision": "accept",
                    "reviewed_label": "laugh",
                }
            ]
        )
    )
    with pytest.raises(Starss23ReviewStatusError, match="cannot be accepted as onset gold"):
        validate_starss23_review(accepted, pack_row)
    transcribed = GoldReviewRecord.model_validate(review_payload(transcript={"decision": "accept"}))
    with pytest.raises(Starss23ReviewStatusError, match="not_reviewable"):
        validate_starss23_review(transcribed, pack_row)


def test_html_pack_protocol_holes(tmp_path: Path) -> None:
    wav_pos = tmp_path / "cache" / "clip-pos.wav"
    wav_neg = tmp_path / "cache" / "clip-neg.wav"
    digest_pos = write_tone(wav_pos, frequency=440.0)
    digest_neg = write_tone(wav_neg, frequency=330.0)
    truncated = {"label": "laugh", "start_ms": 900, "end_ms": 1000, "source_class": 4}
    positive = inspection_row(
        clip_id="clip-pos",
        sha256=digest_pos,
        cache_path="clip-pos.wav",
        events=[truncated],
        duration_ms=1000,
    )
    negative = inspection_row(
        clip_id="clip-neg",
        sha256=digest_neg,
        cache_path="clip-neg.wav",
        events=[],
        duration_ms=1000,
    )
    positive = annotate_pack_row(positive)
    negative = annotate_pack_row(negative)
    output = tmp_path / "html"
    resolved, missing, _ = resolve_pack_audio([positive, negative], [tmp_path / "cache"])
    write_html_pack([positive, negative], output, resolved, missing)
    index = (output / "index.html").read_text(encoding="utf-8")
    pos_page = (output / "clips" / "clip-pos.html").read_text(encoding="utf-8")
    neg_page = (output / "clips" / "clip-neg.html").read_text(encoding="utf-8")
    assert "true negative" in index
    assert "truncated window-end" in index
    assert "Label definitions" in index
    assert "do not compare gold collar to 0.1395" in index.lower()
    assert ">= 300 ms" in index
    assert "first voiced burst" in index
    assert "last voiced frame" in index
    assert "downmix-audible" in index.lower()
    assert ">= 2 independent reviewers" in index
    assert 'value="accept"' not in pos_page
    assert 'value="retime" selected' in pos_page
    assert 'step="1"' in pos_page
    assert "never snap" in pos_page.lower()
    assert "Collar-ineligible" in pos_page
    assert "Do not gold clip-end as offset" in pos_page
    assert "transcript_decision" not in pos_page
    assert "affect_decision" not in pos_page
    assert "not_reviewable" in pos_page
    assert "consent is not independently verified" in pos_page.lower()
    assert "must not be treated as answers" in pos_page.lower()
    assert source_span_id(truncated) in pos_page
    assert "Add missing audible laugh" in neg_page
