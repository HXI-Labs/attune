import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_whisper_uses_known_language_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    generate_arguments = {}

    class Processor:
        def __call__(self, samples, *, sampling_rate, return_tensors):
            return SimpleNamespace(input_features="features")

        def batch_decode(self, generated_ids, *, skip_special_tokens):
            return ["test transcript"]

    class Model:
        def generate(self, input_features, **kwargs):
            generate_arguments.update(kwargs)
            return ["tokens"]

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForSpeechSeq2Seq=object, AutoProcessor=object),
    )
    adapter = WhisperSmallAdapter(checkpoint=Path("."))
    adapter._processor = Processor()
    adapter._model = Model()

    adapter.predict(BaselineInput(FIXTURE.audio_path, language_hint="en"))

    assert generate_arguments == {"language": "en", "task": "transcribe"}
