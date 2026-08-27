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
    PACK_RELATIVE_PATH,
    AudioHashMismatchError,
    Starss23ReviewStatusError,
    append_ledger,
    assert_expected_pack_counts,
    build_pack_rows,
    first_60s_rows,
    laugh_event_count,
    load_jsonl,
    resolve_pack_audio,
    source_span_id,
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
