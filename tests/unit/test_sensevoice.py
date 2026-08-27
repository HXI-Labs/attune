from attune.baselines.sensevoice import (
    build_utterance_spans,
    parse_sensevoice_output,
    sensevoice_affect_trace,
)


def test_rich_transcript_tags_map_to_events_and_speech_styles() -> None:
    parsed = parse_sensevoice_output(
        "<|en|><|NEUTRAL|><|Speech|><|Laughter|><|Cry|><|BGM|> hello there "
    )

    assert parsed.transcript == "hello there"
    assert [(row.label.value, row.confidence) for row in parsed.events] == [
        ("laugh", 0.0),
        ("sob", 0.0),
    ]
    assert [(row.label.value, row.confidence) for row in parsed.styles] == [
        ("laughing_speech", 0.0),
        ("crying_speech", 0.0),
    ]
    assert [(row.label.value, row.confidence) for row in parsed.affect] == [
        ("neutral", 0.0)
    ]


def test_structured_events_use_scores_and_drop_unmapped_labels() -> None:
    parsed = parse_sensevoice_output(
        {
            "text": "<|en|><|Cough|> excuse me",
            "events": [
                {"label": "Cough", "score": 0.83},
                {"name": "Throat Clearing"},
                {"label": "Applause", "score": 0.99},
                {"Sneeze": 1.4, "Noise": 0.7},
            ],
        }
    )

    assert parsed.transcript == "excuse me"
    assert [(row.label.value, row.confidence) for row in parsed.events] == [
        ("cough", 0.83),
        ("throat_clear", 0.0),
        ("sneeze", 1.0),
    ]
    assert parsed.styles == ()
    assert parsed.affect == ()


def test_structured_ser_is_mapped_and_retained_in_raw_trace() -> None:
    result = {
        "text": "please leave",
        "emotion": {"label": "Disgusted", "score": 0.72},
    }

    parsed = parse_sensevoice_output(result)

    assert [(row.label.value, row.confidence) for row in parsed.affect] == [
        ("other", 0.72)
    ]
    assert sensevoice_affect_trace(result) == [
        {
            "raw_label": "Disgusted",
            "normalized_label": "disgusted",
            "schema_label": "other",
            "confidence": 0.72,
            "source": "structured_output",
        }
    ]


def test_utterance_tags_become_provisional_whole_clip_spans() -> None:
    parsed = parse_sensevoice_output(
        [{"text": "<|Breath|><|Singing|> la la"}]
    )

    events, styles = build_utterance_spans(
        parsed,
        duration_ms=1250,
        word_ids=["w1", "w2"],
    )

    assert events[0].model_dump(mode="json") == {
        "start_ms": 0,
        "end_ms": 1250,
        "id": "e1",
        "label": "breath",
        "after_word_id": None,
        "confidence": 0.0,
        "status": "provisional",
    }
    assert styles[0].label == "singing"
    assert (styles[0].start_word_id, styles[0].end_word_id) == ("w1", "w2")
