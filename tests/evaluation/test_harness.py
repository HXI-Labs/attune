from pathlib import Path

from attune.baselines.adapters import (
    BaselineInput,
    TranscriptSentimentAdapter,
    _map_emotion2vec_result,
)
from attune.baselines.cascade import ModularCascade
from attune.evaluation.harness import run_fixture_harness
from attune.schema.output import AttuneOutput

FIXTURES = Path("data/fixtures/semantic_conflict")


def test_transcript_baseline_is_cpu_and_schema_valid() -> None:
    prediction = TranscriptSentimentAdapter().predict(
        BaselineInput(
            FIXTURES / "explicit_match_joy.wav",
            transcript_hint="I am happy about this",
        )
    )
    assert prediction.output.affect.top_label == "joy"
    assert prediction.output.events == []
    assert prediction.output.styles == []
    AttuneOutput.model_validate(prediction.output.model_dump(mode="json"))


def test_fixture_harness_runs_without_optional_weights() -> None:
    report = run_fixture_harness(FIXTURES)
    transcript = next(
        row for row in report["reports"] if row["runner"] == "transcript-only-lexicon"
    )
    assert transcript["schema_validity"]["invalid"] == 0
    assert transcript["metrics"]["asr"]["wer"] == 0
    assert transcript["metrics"]["acoustic_preference"]["conflict_items"] == 5
    assert report["fixture_notice"].startswith("Synthetic")


def test_modular_cascade_composes_schema_valid_channels() -> None:
    baseline = TranscriptSentimentAdapter()
    prediction = ModularCascade(asr=baseline, affect=baseline).predict(
        BaselineInput(
            FIXTURES / "explicit_match_joy.wav",
            transcript_hint="I am happy about this",
        )
    )
    assert prediction.output.transcript.text == "I am happy about this"
    assert prediction.output.affect.top_label == "joy"
    assert prediction.output.events == []
    AttuneOutput.model_validate(prediction.output.model_dump(mode="json"))


def test_emotion2vec_bilingual_labels_are_mapped() -> None:
    category, distribution = _map_emotion2vec_result(
        [{"labels": ["生气/angry", "开心/happy", "未知/unknown"], "scores": [0.7, 0.2, 0.1]}]
    )
    assert category == "anger"
    assert distribution[category] > 0.69
