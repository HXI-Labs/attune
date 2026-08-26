from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from attune.schema.output import AttuneOutput, DEFAULT_INTERPRETATION_WARNING


def test_example_round_trip_and_schema_version(example_payload: dict) -> None:
    output = AttuneOutput.model_validate(example_payload)
    restored = AttuneOutput.model_validate_json(output.model_dump_json())

    assert restored == output
    assert restored.schema_version == "1.0"
    assert restored.transcript.text == "I said leave me alone"
    assert restored.styles[0].label == "shouting"
    assert restored.events[0].label == "sob"
    assert restored.uncertainty.interpretation_warning == DEFAULT_INTERPRETATION_WARNING


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("transcript", "words", 0, "start_ms"), -1),
        (("styles", 0, "start_ms"), -1),
        (("events", 0, "end_ms"), -1),
    ],
)
def test_negative_timestamps_rejected(
    example_payload: dict, path: tuple[str | int, ...], value: int
) -> None:
    payload = deepcopy(example_payload)
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        AttuneOutput.model_validate(payload)


def test_end_before_start_rejected(example_payload: dict) -> None:
    payload = deepcopy(example_payload)
    payload["events"][0]["end_ms"] = payload["events"][0]["start_ms"] - 1
    with pytest.raises(ValidationError, match="end_ms"):
        AttuneOutput.model_validate(payload)


def test_unordered_words_rejected(example_payload: dict) -> None:
    payload = deepcopy(example_payload)
    payload["transcript"]["words"][1]["start_ms"] = 0
    payload["transcript"]["words"][0]["start_ms"] = 10
    with pytest.raises(ValidationError, match="ordered"):
        AttuneOutput.model_validate(payload)


def test_affect_probabilities_must_sum_to_one(example_payload: dict) -> None:
    payload = deepcopy(example_payload)
    payload["affect"]["categories"]["neutral"] = 0.5
    with pytest.raises(ValidationError, match="sum to 1"):
        AttuneOutput.model_validate(payload)


def test_abstention_requires_null_top_label(example_payload: dict) -> None:
    payload = deepcopy(example_payload)
    payload["affect"]["abstain"] = True
    with pytest.raises(ValidationError, match="top_label must be null"):
        AttuneOutput.model_validate(payload)

    payload["affect"]["top_label"] = None
    payload["affect"]["top_label_confidence"] = 0.0
    assert AttuneOutput.model_validate(payload).affect.abstain is True
