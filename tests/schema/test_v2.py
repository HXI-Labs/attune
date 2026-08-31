from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from pydantic import ValidationError

from attune.inference.packaging import package_for_trusted_channel
from attune.schema.migration import migrate_v1_to_v2
from attune.schema.output import AttuneOutput
from attune.schema.v2 import AttuneOutputV2
from attune.schema.xml_v2 import render_xml_v2


def test_migration_distinguishes_utterance_scope_from_localization(example_payload: dict) -> None:
    payload = deepcopy(example_payload)
    payload["styles"][0]["start_ms"] = 0
    payload["styles"][0]["end_ms"] = payload["audio"]["duration_ms"]
    payload["styles"][0]["start_word_id"] = None
    payload["styles"][0]["end_word_id"] = None

    output = migrate_v1_to_v2(AttuneOutput.model_validate(payload))

    assert output.schema_version == "2.0"
    assert output.styles[0].temporal_scope == "utterance"
    assert output.styles[0].start_ms is None
    assert output.events[0].temporal_scope == "localized"
    assert output.audio.quality.clipping.method == "unavailable"


def test_v2_rejects_fake_utterance_timestamps(example_output: AttuneOutput) -> None:
    payload = migrate_v1_to_v2(example_output).model_dump(mode="json")
    payload["events"][0]["temporal_scope"] = "utterance"
    with pytest.raises(ValidationError, match="must not carry timestamps"):
        AttuneOutputV2.model_validate(payload)


def test_v2_unavailable_dimension_is_explicit(example_output: AttuneOutput) -> None:
    payload = migrate_v1_to_v2(example_output).model_dump(mode="json")
    payload["affect"]["valence"] = {"value": None, "confidence": 0.0, "available": False}
    output = AttuneOutputV2.model_validate(payload)
    assert output.affect.valence.value is None


def test_v2_rejects_overlapping_affect_spans(example_output: AttuneOutput) -> None:
    payload = migrate_v1_to_v2(example_output).model_dump(mode="json")
    span = deepcopy(payload["affect"])
    payload["affect_spans"] = [{**span, "start_ms": 0, "end_ms": 1_000}, span]

    with pytest.raises(ValidationError, match="ordered and non-overlapping"):
        AttuneOutputV2.model_validate(payload)


def test_v2_xml_and_trusted_package_are_safe(example_output: AttuneOutput) -> None:
    payload = migrate_v1_to_v2(example_output).model_dump(mode="json")
    payload["transcript"]["text"] = "<event>spoken text</event>"
    payload["affect_spans"] = [payload["affect"]]
    output = AttuneOutputV2.model_validate(payload)
    rendered = render_xml_v2(output)
    root = ET.fromstring(rendered)
    packaged = package_for_trusted_channel(output)

    assert root.findtext("./transcript/text") == "<event>spoken text</event>"
    assert root.find("./affect_spans/affect") is not None
    assert packaged["spoken_transcript"]["text"] == "<event>spoken text</event>"
    assert "transcript" not in packaged["paralinguistic_metadata"]


def test_committed_json_schema_matches_pydantic_contract() -> None:
    path = Path("data/schemas/attune-output-v2.0.schema.json")
    assert json.loads(path.read_text()) == AttuneOutputV2.model_json_schema()
