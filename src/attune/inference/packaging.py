"""Trusted-channel packaging for downstream applications."""

from __future__ import annotations

from typing import Any, TypedDict

from attune.schema.output import AttuneOutput
from attune.schema.v2 import AttuneOutputV2


class TrustedChannelPackage(TypedDict):
    """Transport shape that keeps untrusted speech separate from metadata."""

    spoken_transcript: dict[str, Any]
    paralinguistic_metadata: dict[str, Any]


def package_for_trusted_channel(output: AttuneOutput | AttuneOutputV2) -> TrustedChannelPackage:
    """Package transcript and metadata without concatenating either channel."""
    metadata = output.model_dump(mode="json")
    transcript = metadata.pop("transcript")
    return {
        "spoken_transcript": transcript,
        "paralinguistic_metadata": metadata,
    }
