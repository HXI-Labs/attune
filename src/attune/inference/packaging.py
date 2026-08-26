"""Trusted-channel packaging for downstream applications."""

from __future__ import annotations

from typing import Any, TypedDict

from attune.schema.output import AttuneOutput


class TrustedChannelPackage(TypedDict):
    """Transport shape that keeps untrusted speech separate from metadata."""

    user_text: str
    paralinguistic_metadata: dict[str, Any]


def package_for_trusted_channel(output: AttuneOutput) -> TrustedChannelPackage:
    """Package transcript and metadata without concatenating either channel."""
    metadata = output.model_dump(mode="json")
    metadata.pop("transcript")
    return {
        "user_text": output.transcript.text,
        "paralinguistic_metadata": metadata,
    }
