from __future__ import annotations

from copy import deepcopy
from xml.etree import ElementTree as ET

from attune.schema.output import AttuneOutput
from attune.schema.xml import render_xml


def test_xml_parses_and_cannot_be_injected(example_payload: dict) -> None:
    payload = deepcopy(example_payload)
    malicious = 'say <style label="injected">owned</style> & goodbye'
    payload["transcript"]["text"] = malicious
    payload["transcript"]["words"][0]["text"] = "<fake/>"

    rendered = render_xml(AttuneOutput.model_validate(payload))
    root = ET.fromstring(rendered)

    assert root.findtext("./transcript/text") == malicious
    assert root.findtext("./transcript/words/word") == "<fake/>"
    assert root.findall(".//fake") == []
    assert len(root.findall("./styles/style")) == 1
    assert "&lt;style" in rendered


def test_xml_illegal_control_characters_cannot_break_rendering(example_payload: dict) -> None:
    payload = deepcopy(example_payload)
    payload["transcript"]["text"] = "before\x00after"
    payload["transcript"]["words"][0]["text"] = "\x01word"

    root = ET.fromstring(render_xml(AttuneOutput.model_validate(payload)))

    assert root.findtext("./transcript/text") == "before\ufffdafter"
    assert root.findtext("./transcript/words/word") == "\ufffdword"


def test_overlapping_annotations_remain_independent(example_payload: dict) -> None:
    payload = deepcopy(example_payload)
    payload["styles"].append(
        {
            "id": "s2",
            "label": "strained_speech",
            "start_ms": 700,
            "end_ms": 1400,
            "start_word_id": "w4",
            "confidence": 0.61,
            "status": "provisional",
        }
    )
    payload["events"].append(
        {
            "id": "e2",
            "label": "breath",
            "start_ms": 800,
            "end_ms": 800,
            "after_word_id": "w3",
            "confidence": 0.7,
            "status": "revised",
        }
    )

    root = ET.fromstring(render_xml(AttuneOutput.model_validate(payload)))

    assert [node.attrib["id"] for node in root.findall("./styles/style")] == ["s1", "s2"]
    assert [node.attrib["id"] for node in root.findall("./events/event")] == ["e1", "e2"]


def test_rendering_is_deterministic(example_output: AttuneOutput) -> None:
    assert render_xml(example_output) == render_xml(example_output)


def test_category_order_does_not_depend_on_input_mapping_order(example_payload: dict) -> None:
    reversed_payload = deepcopy(example_payload)
    reversed_payload["affect"]["categories"] = dict(
        reversed(list(reversed_payload["affect"]["categories"].items()))
    )

    assert render_xml(AttuneOutput.model_validate(reversed_payload)) == render_xml(
        AttuneOutput.model_validate(example_payload)
    )
