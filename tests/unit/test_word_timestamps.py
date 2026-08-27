from attune.baselines.adapters import (
    parse_sensevoice_word_timestamps,
    parse_whisper_word_timestamps,
)


def test_sensevoice_explicit_word_timestamps_are_preserved_in_milliseconds() -> None:
    result = [
        {
            "text": "hello there",
            "words": [
                {"word": "hello", "start_ms": 120, "end_ms": 410, "confidence": 0.9},
                {"word": "there", "timestamp": [430, 780], "score": 0.8},
            ],
        }
    ]

    assert parse_sensevoice_word_timestamps(result, duration_ms=1000) == [
        {"id": "w1", "text": "hello", "start_ms": 120, "end_ms": 410, "confidence": 0.9},
        {"id": "w2", "text": "there", "start_ms": 430, "end_ms": 780, "confidence": 0.8},
    ]


def test_sensevoice_official_token_timestamp_shape_converts_seconds() -> None:
    result = {
        "text": "<|en|><|NEUTRAL|><|Speech|><|woitn|> hello",
        "timestamp": [["hello", 0.12, 0.41]],
    }

    assert parse_sensevoice_word_timestamps(result, duration_ms=1000) == [
        {"id": "w1", "text": "hello", "start_ms": 120, "end_ms": 410, "confidence": 0.0}
    ]


def test_sensevoice_parallel_word_and_millisecond_arrays_are_zipped() -> None:
    result = {
        "text": "hello there",
        "words": ["hello", "there"],
        "timestamp": [[120, 410], [430, 780]],
    }

    assert parse_sensevoice_word_timestamps(result, duration_ms=1000) == [
        {"id": "w1", "text": "hello", "start_ms": 120, "end_ms": 410, "confidence": 0.0},
        {"id": "w2", "text": "there", "start_ms": 430, "end_ms": 780, "confidence": 0.0},
    ]


def test_whisper_official_word_chunks_convert_seconds_to_milliseconds() -> None:
    chunks = [
        {"text": " hello", "timestamp": (0.12, 0.41)},
        {"text": " there", "timestamp": (0.43, 0.78)},
    ]

    words = parse_whisper_word_timestamps(chunks, duration_ms=1000)

    assert [(word["text"], word["start_ms"], word["end_ms"]) for word in words] == [
        ("hello", 120, 410),
        ("there", 430, 780),
    ]


def test_incomplete_or_out_of_bounds_alignment_is_not_partially_fabricated() -> None:
    assert (
        parse_sensevoice_word_timestamps(
            {"words": [{"word": "hello", "start_ms": 0, "end_ms": 1100}]},
            duration_ms=1000,
        )
        == []
    )
    assert (
        parse_whisper_word_timestamps(
            [{"text": "hello", "timestamp": (0.0, None)}],
            duration_ms=1000,
        )
        == []
    )
