from __future__ import annotations

from copy import deepcopy

from attune.client import AttuneClient
from attune.schema.migration import migrate_v1_to_v2
from attune.schema.output import AttuneOutput


class Backend:
    name = "fixture"

    def __init__(self, payload: dict) -> None:
        self.output = migrate_v1_to_v2(AttuneOutput.model_validate(payload))

    def analyse_wav(self, audio: bytes):
        assert audio == b"wav"
        return self.output


def test_local_client_keeps_trusted_channels_separate(example_payload: dict) -> None:
    client = AttuneClient(Backend(deepcopy(example_payload)))
    package = client.trusted_context(b"wav")
    assert package["spoken_transcript"]["text"] == "I said leave me alone"
    assert "transcript" not in package["paralinguistic_metadata"]
