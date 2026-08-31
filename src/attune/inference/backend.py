"""Inference backend interfaces shared by local runners and the API."""

from __future__ import annotations

from typing import Protocol

from attune.schema.v2 import AttuneOutputV2


class InferenceBackend(Protocol):
    name: str

    def analyse_wav(self, audio: bytes) -> AttuneOutputV2:
        """Analyse one complete WAV payload."""


def set_output_status(output: AttuneOutputV2, status: str) -> AttuneOutputV2:
    serialized = output.model_dump(mode="json")
    for channel in ("styles", "events"):
        for annotation in serialized[channel]:
            annotation["status"] = status
    return AttuneOutputV2.model_validate(serialized)
