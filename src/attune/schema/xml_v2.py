"""Deterministic XML projection for Attune schema v2."""

from __future__ import annotations

from xml.etree import ElementTree as ET

from attune.schema.output import AffectCategory
from attune.schema.v2 import Affect, AttuneOutputV2
from attune.schema.xml import _probability, _xml_safe


def _append_affect(parent: ET.Element, name: str, value: Affect) -> None:
    affect = ET.SubElement(
        parent,
        name,
        {
            "start_ms": str(value.start_ms),
            "end_ms": str(value.end_ms),
            "abstain": str(value.abstain).lower(),
        },
    )
    if value.abstention_reason:
        affect.set("abstention_reason", _xml_safe(value.abstention_reason))
    for name in ("valence", "arousal", "dominance"):
        dimension = getattr(value, name)
        attributes = {
            "name": name,
            "available": str(dimension.available).lower(),
            "confidence": _probability(dimension.confidence),
        }
        if dimension.value is not None:
            attributes["value"] = _probability(dimension.value)
        ET.SubElement(affect, "dimension", attributes)
    categories = ET.SubElement(affect, "categories")
    for label in AffectCategory:
        ET.SubElement(
            categories,
            "category",
            {"label": label.value, "probability": _probability(value.categories[label])},
        )


def render_xml_v2(output: AttuneOutputV2) -> str:
    root = ET.Element("attune", {"schema_version": output.schema_version})
    transcript = ET.SubElement(
        root, "transcript", {"confidence": _probability(output.transcript.confidence)}
    )
    text = ET.SubElement(transcript, "text")
    text.text = _xml_safe(output.transcript.text)
    words = ET.SubElement(transcript, "words")
    for word in output.transcript.words:
        node = ET.SubElement(
            words,
            "word",
            {
                "id": _xml_safe(word.id),
                "start_ms": str(word.start_ms),
                "end_ms": str(word.end_ms),
                "confidence": _probability(word.confidence),
            },
        )
        node.text = _xml_safe(word.text)
    for channel, channel_items in (("styles", output.styles), ("events", output.events)):
        parent = ET.SubElement(root, channel)
        element_name = channel[:-1]
        for annotation in channel_items:
            attributes = {
                "id": _xml_safe(annotation.id),
                "label": annotation.label.value,
                "temporal_scope": annotation.temporal_scope.value,
                "confidence": _probability(annotation.confidence),
                "status": annotation.status.value,
            }
            if annotation.start_ms is not None:
                attributes["start_ms"] = str(annotation.start_ms)
                attributes["end_ms"] = str(annotation.end_ms)
            for field in ("start_word_id", "end_word_id", "after_word_id"):
                field_value = getattr(annotation, field, None)
                if field_value is not None:
                    attributes[field] = _xml_safe(field_value)
            ET.SubElement(parent, element_name, attributes)
    _append_affect(root, "affect", output.affect)
    affect_spans = ET.SubElement(root, "affect_spans")
    for span in output.affect_spans:
        _append_affect(affect_spans, "affect", span)
    uncertainty_attributes = {
        "out_of_distribution_available": str(
            output.uncertainty.out_of_distribution_available
        ).lower()
    }
    if output.uncertainty.out_of_distribution_probability is not None:
        uncertainty_attributes["out_of_distribution_probability"] = _probability(
            output.uncertainty.out_of_distribution_probability
        )
    uncertainty = ET.SubElement(root, "uncertainty", uncertainty_attributes)
    warning = ET.SubElement(uncertainty, "interpretation_warning")
    warning.text = _xml_safe(output.uncertainty.interpretation_warning)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=False)
