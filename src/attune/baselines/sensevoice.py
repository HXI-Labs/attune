"""Parse SenseVoice rich-transcription tags into the Attune ontology."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from attune.schema.output import EventLabel, Status, StyleLabel, VocalEvent, VocalStyle

UNKNOWN_CONFIDENCE = 0.0

_TAG_PATTERN = re.compile(r"<\|([^|]+)\|>")
_EVENT_LABELS = {
    "laughter": EventLabel.LAUGH,
    "laugh": EventLabel.LAUGH,
    "cry": EventLabel.SOB,
    "crying": EventLabel.SOB,
    "sob": EventLabel.SOB,
    "sigh": EventLabel.SIGH,
    "cough": EventLabel.COUGH,
    "throat_clear": EventLabel.THROAT_CLEAR,
    "throat_clearing": EventLabel.THROAT_CLEAR,
    "sneeze": EventLabel.SNEEZE,
    "breath": EventLabel.BREATH,
    "breathing": EventLabel.BREATH,
}
_SPEECH_STYLE_LABELS = {
    "laughter": StyleLabel.LAUGHING_SPEECH,
    "laugh": StyleLabel.LAUGHING_SPEECH,
    "cry": StyleLabel.CRYING_SPEECH,
    "crying": StyleLabel.CRYING_SPEECH,
}
_STYLE_LABELS = {
    "singing": StyleLabel.SINGING,
    "whispering": StyleLabel.WHISPERING,
    "whisper": StyleLabel.WHISPERING,
    "shouting": StyleLabel.SHOUTING,
    "shout": StyleLabel.SHOUTING,
}
_STRUCTURED_EVENT_KEYS = ("events", "event", "aed", "audio_events", "audio_event")


@dataclass(frozen=True)
class SenseVoiceAnnotation:
    """One ontology label extracted from SenseVoice output."""

    label: EventLabel | StyleLabel
    confidence: float = UNKNOWN_CONFIDENCE


@dataclass(frozen=True)
class SenseVoiceOutput:
    """Clean transcript plus conservatively mapped utterance-level annotations."""

    transcript: str
    events: tuple[SenseVoiceAnnotation, ...]
    styles: tuple[SenseVoiceAnnotation, ...]


def parse_sensevoice_output(result: Any) -> SenseVoiceOutput:
    """Parse the first FunASR result row or a raw rich-transcription string.

    SenseVoice normally places AED labels in ``<|...|>`` tags and does not
    expose their scores. Structured event dictionaries are also accepted for
    forward compatibility when a runtime does expose a score.
    """

    row = _result_row(result)
    if isinstance(row, str):
        rich_text = row
        structured_values: list[Any] = []
    elif isinstance(row, dict):
        rich_text = str(row.get("text", ""))
        structured_values = [
            row[key] for key in _STRUCTURED_EVENT_KEYS if key in row and row[key] is not None
        ]
    else:
        raise RuntimeError("SenseVoice returned an unsupported result shape")

    transcript = _TAG_PATTERN.sub("", rich_text)
    transcript = re.sub(r"\s+", " ", transcript).strip()
    candidates = [
        (tag, UNKNOWN_CONFIDENCE) for tag in _TAG_PATTERN.findall(rich_text)
    ]
    for value in structured_values:
        candidates.extend(_structured_annotations(value))

    events: dict[EventLabel, float] = {}
    styles: dict[StyleLabel, float] = {}
    for raw_label, confidence in candidates:
        normalized = _normalize_label(raw_label)
        if event_label := _EVENT_LABELS.get(normalized):
            events[event_label] = max(events.get(event_label, 0.0), confidence)
        if style_label := _STYLE_LABELS.get(normalized):
            styles[style_label] = max(styles.get(style_label, 0.0), confidence)
        if transcript and (speech_style := _SPEECH_STYLE_LABELS.get(normalized)):
            styles[speech_style] = max(styles.get(speech_style, 0.0), confidence)

    return SenseVoiceOutput(
        transcript=transcript,
        events=tuple(
            SenseVoiceAnnotation(label=label, confidence=confidence)
            for label, confidence in events.items()
        ),
        styles=tuple(
            SenseVoiceAnnotation(label=label, confidence=confidence)
            for label, confidence in styles.items()
        ),
    )


def build_utterance_spans(
    parsed: SenseVoiceOutput,
    *,
    duration_ms: int,
    word_ids: list[str],
) -> tuple[list[VocalEvent], list[VocalStyle]]:
    """Materialize utterance-level tags as provisional whole-clip spans."""

    events = [
        VocalEvent(
            id=f"e{index}",
            label=annotation.label,
            start_ms=0,
            end_ms=duration_ms,
            after_word_id=None,
            confidence=annotation.confidence,
            status=Status.PROVISIONAL,
        )
        for index, annotation in enumerate(parsed.events, start=1)
        if isinstance(annotation.label, EventLabel)
    ]
    styles = [
        VocalStyle(
            id=f"s{index}",
            label=annotation.label,
            start_ms=0,
            end_ms=duration_ms,
            start_word_id=word_ids[0] if word_ids else None,
            end_word_id=word_ids[-1] if word_ids else None,
            confidence=annotation.confidence,
            status=Status.PROVISIONAL,
        )
        for index, annotation in enumerate(parsed.styles, start=1)
        if isinstance(annotation.label, StyleLabel)
    ]
    return events, styles


def _result_row(result: Any) -> Any:
    if isinstance(result, list):
        if not result:
            raise RuntimeError("SenseVoice returned an empty result")
        return result[0]
    return result


def _structured_annotations(value: Any) -> list[tuple[str, float]]:
    if isinstance(value, str):
        return [(value, UNKNOWN_CONFIDENCE)]
    if isinstance(value, list):
        annotations: list[tuple[str, float]] = []
        for item in value:
            annotations.extend(_structured_annotations(item))
        return annotations
    if not isinstance(value, dict):
        return []

    label = value.get("label", value.get("name", value.get("event")))
    if label is not None:
        return [(str(label), _confidence(value.get("score", value.get("confidence"))))]

    annotations = []
    for key, score in value.items():
        if isinstance(score, int | float):
            annotations.append((str(key), _confidence(score)))
    return annotations


def _confidence(value: Any) -> float:
    if not isinstance(value, int | float):
        return UNKNOWN_CONFIDENCE
    return min(1.0, max(0.0, float(value)))


def _normalize_label(label: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower())
    return normalized.strip("_")
