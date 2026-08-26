"""Deterministic XML projection of trusted Attune JSON.

The renderer accepts only validated schema objects. Models never generate markup.
Overlapping annotations remain independent time spans instead of nested tags.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

from attune.schema.output import AffectCategory, AttuneOutput


def _probability(value: float) -> str:
    return format(value, ".6g")


def _xml_safe(value: str) -> str:
    """Replace characters forbidden by XML 1.0 while preserving valid text."""
    return "".join(
        character
        if character in "\t\n\r"
        or "\u0020" <= character <= "\ud7ff"
        or "\ue000" <= character <= "\ufffd"
        or "\U00010000" <= character <= "\U0010ffff"
        else "\ufffd"
        for character in value
    )


def render_xml(output: AttuneOutput) -> str:
    """Render a validated output as well-formed, injection-safe XML."""
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

    styles = ET.SubElement(root, "styles")
    for style in output.styles:
        attributes = {
            "id": _xml_safe(style.id),
            "label": style.label.value,
            "start_ms": str(style.start_ms),
            "end_ms": str(style.end_ms),
            "confidence": _probability(style.confidence),
            "status": style.status.value,
        }
        if style.start_word_id is not None:
            attributes["start_word_id"] = _xml_safe(style.start_word_id)
        if style.end_word_id is not None:
            attributes["end_word_id"] = _xml_safe(style.end_word_id)
        ET.SubElement(styles, "style", attributes)

    events = ET.SubElement(root, "events")
    for event in output.events:
        attributes = {
            "id": _xml_safe(event.id),
            "label": event.label.value,
            "start_ms": str(event.start_ms),
            "end_ms": str(event.end_ms),
            "confidence": _probability(event.confidence),
            "status": event.status.value,
        }
        if event.after_word_id is not None:
            attributes["after_word_id"] = _xml_safe(event.after_word_id)
        ET.SubElement(events, "event", attributes)

    affect = ET.SubElement(
        root,
        "affect",
        {
            "start_ms": str(output.affect.start_ms),
            "end_ms": str(output.affect.end_ms),
            "abstain": str(output.affect.abstain).lower(),
        },
    )
    for name in ("valence", "arousal", "dominance"):
        dimension = getattr(output.affect, name)
        ET.SubElement(
            affect,
            "dimension",
            {
                "name": name,
                "value": _probability(dimension.value),
                "confidence": _probability(dimension.confidence),
            },
        )
    categories = ET.SubElement(affect, "categories")
    for label in AffectCategory:
        ET.SubElement(
            categories,
            "category",
            {
                "label": label.value,
                "probability": _probability(output.affect.categories[label]),
            },
        )
    ET.SubElement(
        affect,
        "top",
        {
            "label": output.affect.top_label.value if output.affect.top_label else "",
            "confidence": _probability(output.affect.top_label_confidence),
        },
    )

    uncertainty = ET.SubElement(
        root,
        "uncertainty",
        {
            "out_of_distribution_probability": _probability(
                output.uncertainty.out_of_distribution_probability
            )
        },
    )
    warning = ET.SubElement(uncertainty, "interpretation_warning")
    warning.text = _xml_safe(output.uncertainty.interpretation_warning)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=False)
