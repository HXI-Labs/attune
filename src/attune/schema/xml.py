"""Deterministic XML projection of trusted Attune JSON.

The renderer accepts only validated schema objects. Models never generate markup.
Overlapping annotations remain independent time spans instead of nested tags.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

from attune.schema.output import AttuneOutput


def _probability(value: float) -> str:
    return format(value, ".6g")


def render_xml(output: AttuneOutput) -> str:
    """Render a validated output as well-formed, injection-safe XML."""
    root = ET.Element("attune", {"schema_version": output.schema_version})

    transcript = ET.SubElement(
        root, "transcript", {"confidence": _probability(output.transcript.confidence)}
    )
    text = ET.SubElement(transcript, "text")
    text.text = output.transcript.text
    words = ET.SubElement(transcript, "words")
    for word in output.transcript.words:
        node = ET.SubElement(
            words,
            "word",
            {
                "id": word.id,
                "start_ms": str(word.start_ms),
                "end_ms": str(word.end_ms),
                "confidence": _probability(word.confidence),
            },
        )
        node.text = word.text

    styles = ET.SubElement(root, "styles")
    for style in output.styles:
        attributes = {
            "id": style.id,
            "label": style.label.value,
            "start_ms": str(style.start_ms),
            "end_ms": str(style.end_ms),
            "confidence": _probability(style.confidence),
            "status": style.status.value,
        }
        if style.start_word_id is not None:
            attributes["start_word_id"] = style.start_word_id
        if style.end_word_id is not None:
            attributes["end_word_id"] = style.end_word_id
        ET.SubElement(styles, "style", attributes)

    events = ET.SubElement(root, "events")
    for event in output.events:
        attributes = {
            "id": event.id,
            "label": event.label.value,
            "start_ms": str(event.start_ms),
            "end_ms": str(event.end_ms),
            "confidence": _probability(event.confidence),
            "status": event.status.value,
        }
        if event.after_word_id is not None:
            attributes["after_word_id"] = event.after_word_id
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
    for label, probability in output.affect.categories.items():
        ET.SubElement(
            categories,
            "category",
            {"label": label.value, "probability": _probability(probability)},
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
    warning.text = output.uncertainty.interpretation_warning

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=False)
