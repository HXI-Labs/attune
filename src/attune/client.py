"""Small local Python SDK for schema-v2 Attune inference."""

from __future__ import annotations

from pathlib import Path

from attune.inference.backend import InferenceBackend
from attune.inference.packaging import TrustedChannelPackage, package_for_trusted_channel
from attune.schema.v2 import AttuneOutputV2


class AttuneClient:
    def __init__(self, backend: InferenceBackend) -> None:
        self.backend = backend

    def analyse(self, audio: bytes) -> AttuneOutputV2:
        return self.backend.analyse_wav(audio)

    def analyse_file(self, path: Path | str) -> AttuneOutputV2:
        return self.analyse(Path(path).expanduser().read_bytes())

    def trusted_context(self, audio: bytes) -> TrustedChannelPackage:
        return package_for_trusted_channel(self.analyse(audio))
