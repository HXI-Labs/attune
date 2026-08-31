"""Explicit migrations between published Attune schemas."""

from __future__ import annotations

from attune.schema.output import AttuneOutput
from attune.schema.v2 import AttuneOutputV2


def migrate_v1_to_v2(output: AttuneOutput) -> AttuneOutputV2:
    """Convert v1 without promoting whole-utterance spans to localization."""
    payload = output.model_dump(mode="json")
    payload["schema_version"] = "2.0"
    duration_ms = output.audio.duration_ms
    payload["audio"]["quality"] = {
        name: {"value": None, "method": "unavailable"}
        for name in ("clipping", "low_snr", "far_field")
    }
    for channel in ("styles", "events"):
        for annotation in payload[channel]:
            whole_utterance = annotation["start_ms"] == 0 and annotation["end_ms"] == duration_ms
            annotation["temporal_scope"] = "utterance" if whole_utterance else "localized"
            if whole_utterance:
                annotation["start_ms"] = None
                annotation["end_ms"] = None
                if channel == "styles":
                    annotation["start_word_id"] = None
                    annotation["end_word_id"] = None
    for name in ("valence", "arousal", "dominance"):
        dimension = payload["affect"][name]
        dimension["available"] = dimension["confidence"] > 0.0
        if not dimension["available"]:
            dimension["value"] = None
    payload["affect"]["abstention_reason"] = (
        "legacy_v1_abstention" if payload["affect"]["abstain"] else None
    )
    return AttuneOutputV2.model_validate(payload)
