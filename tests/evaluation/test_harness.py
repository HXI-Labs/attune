from pathlib import Path

from attune.baselines.adapters import BaselineInput, TranscriptSentimentAdapter
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
