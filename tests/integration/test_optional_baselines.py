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
    pipeline_arguments = {}

    class Processor:
        tokenizer = "tokenizer"
        feature_extractor = "feature-extractor"

    def pipeline(*args, **kwargs):
        pipeline_arguments["construction"] = (args, kwargs)

        def transcribe(audio, **options):
            pipeline_arguments["audio"] = audio
            pipeline_arguments["options"] = options
            return {
                "text": "test transcript",
                "chunks": [{"text": "test", "timestamp": (0.0, 0.1)}],
            }

        return transcribe

    monkeypatch.setitem(
        sys.modules,
        "numpy",
        SimpleNamespace(asarray=lambda values, dtype: values, float32="float32"),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForSpeechSeq2Seq=object,
            AutoProcessor=object,
            pipeline=pipeline,
        ),
    )
    adapter = WhisperSmallAdapter(checkpoint=Path("."))
    adapter._processor = Processor()
    adapter._model = object()

    prediction = adapter.predict(BaselineInput(FIXTURE.audio_path, language_hint="en"))

    assert pipeline_arguments["options"] == {
        "return_timestamps": "word",
        "generate_kwargs": {"language": "en", "task": "transcribe"},
    }
    assert prediction.output.transcript.words[0].text == "test"
