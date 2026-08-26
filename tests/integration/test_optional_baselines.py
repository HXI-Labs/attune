from pathlib import Path

import pytest

from attune.baselines.adapters import (
    BaselineInput,
    Emotion2VecPlusAdapter,
    SenseVoiceSmallAdapter,
    WhisperSmallAdapter,
)
from attune.schema.output import AttuneOutput

FIXTURE = BaselineInput(
    Path("data/fixtures/semantic_conflict/explicit_match_joy.wav"),
    transcript_hint="I am happy about this",
)


@pytest.mark.integration
@pytest.mark.parametrize(
    "adapter",
    [WhisperSmallAdapter(), SenseVoiceSmallAdapter(), Emotion2VecPlusAdapter()],
    ids=lambda adapter: adapter.name,
)
def test_optional_local_weight_adapter(adapter) -> None:
    available, reason = adapter.availability()
    if not available:
        pytest.skip(reason)
    prediction = adapter.predict(FIXTURE)
    AttuneOutput.model_validate(prediction.output.model_dump(mode="json"))
