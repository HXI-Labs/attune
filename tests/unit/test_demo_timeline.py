from __future__ import annotations

from copy import deepcopy

from attune.inference.timeline import (
    FRAME_LOCAL,
    UTTERANCE_SCOPE,
    classify_span,
    render_demo_html,
)
from attune.schema.output import AttuneOutput


def test_classify_span_treats_whole_clip_as_utterance_scope() -> None:
    assert (
        classify_span(
            channel="event",
            label="laugh",
            start_ms=0,
            end_ms=4000,
            duration_ms=4000,
        )
        == UTTERANCE_SCOPE
    )
    assert (
        classify_span(
            channel="style",
            label="shouting",
            start_ms=0,
            end_ms=4000,
            duration_ms=4000,
        )
        == UTTERANCE_SCOPE
    )


def test_classify_span_marks_bounded_dcase_overlap_as_frame_local() -> None:
    assert (
        classify_span(
            channel="event",
            label="cough",
            start_ms=420,
            end_ms=980,
            duration_ms=4000,
        )
        == FRAME_LOCAL
    )


def test_html_distinguishes_frame_local_from_utterance_scope(example_payload: dict) -> None:
    payload = deepcopy(example_payload)
    payload["audio"]["duration_ms"] = 4000
    payload["affect"]["end_ms"] = 4000
    payload["events"] = [
        {
            "id": "e1",
            "label": "laugh",
            "start_ms": 800,
            "end_ms": 1400,
            "after_word_id": None,
            "confidence": 0.91,
            "status": "provisional",
        },
        {
            "id": "e2",
            "label": "sigh",
            "start_ms": 0,
            "end_ms": 4000,
            "after_word_id": None,
            "confidence": 0.4,
            "status": "provisional",
        },
    ]
    payload["styles"] = [
        {
            "id": "s1",
            "label": "shouting",
            "start_ms": 0,
            "end_ms": 4000,
            "start_word_id": None,
            "end_word_id": None,
            "confidence": 0.2,
            "status": "provisional",
        }
    ]
    rendered = render_demo_html(
        AttuneOutput.model_validate(payload),
        audio_src="isolated-cough.wav",
        dcase_head_configured=True,
    )

    assert 'class="span frame-local"' in rendered
    assert "laugh 800–1400ms" in rendered
    assert "Frame-local events" in rendered
    assert "Utterance-scope events (not localization)" in rendered
    assert "sigh 0–4000ms" in rendered
    assert "not localization" in rendered
    assert "isolated-cough.wav" in rendered
    assert "schema/CLI fixture" not in rendered.lower()


def test_html_fixture_banner_omits_dcase_when_unconfigured(example_output: AttuneOutput) -> None:
    rendered = render_demo_html(example_output, fixture=True, dcase_head_configured=False)
    assert "Not a model prediction" in rendered
    assert "DCASE frame timestamps omitted" in rendered
    assert "does not interpolate words" not in rendered
