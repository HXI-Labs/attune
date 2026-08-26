import json
from pathlib import Path

from attune.baselines.adapters import (
    BaselineInput,
    TranscriptSentimentAdapter,
    _map_emotion2vec_result,
)
from attune.baselines.cascade import ModularCascade
from attune.evaluation.harness import run_fixture_harness, run_inspection_harness
from attune.evaluation.report import EvaluationItem, RuntimeMetrics, evaluate_items
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


def test_inspection_harness_reads_jsonl_and_local_wavs(tmp_path: Path) -> None:
    rows = [
        {
            "clip_id": "crema-example",
            "source_dataset": "CREMA-D",
            "source_filename": "explicit_match_joy.wav",
            "cache_path": "explicit_match_joy.wav",
            "source_metadata": {"transcript": "I am happy about this"},
            "intended_attune_labels": {"affect": ["joy"], "events": []},
        },
        {
            "clip_id": "vocal-example",
            "source_dataset": "VocalSound",
            "source_filename": "semantic_conflict_joy.wav",
            "cache_path": "semantic_conflict_joy.wav",
            "source_metadata": {},
            "intended_attune_labels": {"affect": [], "events": ["laugh"]},
        },
    ]
    manifest = tmp_path / "inspection.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = run_inspection_harness(manifest, FIXTURES)

    transcript = next(
        row
        for row in report["runner_results"]
        if row["runner"] == "transcript-only-lexicon"
    )
    assert report["scope"]["sealed_gold_baseline"] is False
    assert transcript["schema_validity"] == {"valid": 1, "invalid": 0, "errors": []}
    assert transcript["not_applicable_clips"] == 1
    assert transcript["metrics"]["asr"]["wer"] == 0
    assert transcript["metrics"]["affect"]["macro_f1"] == 1 / 3


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


def test_modular_cascade_retains_asr_events() -> None:
    class EventASR(TranscriptSentimentAdapter):
        name = "event-asr"

        def predict(self, item):
            prediction = super().predict(item)
            payload = prediction.output.model_dump(mode="json")
            payload["events"] = [
                {
                    "id": "e1",
                    "label": "cough",
                    "start_ms": 0,
                    "end_ms": prediction.output.audio.duration_ms,
                    "after_word_id": None,
                    "confidence": 0.0,
                    "status": "provisional",
                }
            ]
            return prediction.__class__(
                output=AttuneOutput.model_validate(payload),
                runtime=prediction.runtime,
            )

    prediction = ModularCascade(
        asr=EventASR(),
        affect=TranscriptSentimentAdapter(),
    ).predict(
        BaselineInput(
            FIXTURES / "explicit_match_joy.wav",
            transcript_hint="I am happy about this",
        )
    )

    assert [event.label for event in prediction.output.events] == ["cough"]
    assert prediction.output.model.name.endswith("+asr-event-output")


def test_emotion2vec_bilingual_labels_are_mapped() -> None:
    category, distribution = _map_emotion2vec_result(
        [{"labels": ["生气/angry", "开心/happy", "未知/unknown"], "scores": [0.7, 0.2, 0.1]}]
    )
    assert category == "anger"
    assert distribution[category] > 0.69


def test_report_records_invalid_raw_prediction_and_optional_runtime() -> None:
    valid = (
        TranscriptSentimentAdapter()
        .predict(
            BaselineInput(
                FIXTURES / "explicit_match_joy.wav",
                transcript_hint="I am happy about this",
            )
        )
        .output
    )
    invalid = valid.model_dump(mode="json")
    invalid["schema_version"] = "broken"
    report = evaluate_items(
        "invalid-test",
        [
            EvaluationItem(
                item_id="invalid",
                reference=valid,
                prediction=invalid,
                acoustic_target="joy",
                lexical_target="joy",
                runtime=RuntimeMetrics(
                    audio_seconds=0.5,
                    elapsed_seconds=0.1,
                    real_time_factor=0.2,
                    latency_ms=None,
                    first_result_latency_ms=25.0,
                ),
            )
        ],
    )
    assert report.schema_validity["invalid"] == 1
    assert report.items[0]["predicted_text"] is None
    assert report.runtime["mean_latency_ms"] is None
    assert report.runtime["first_result_latency_ms"] == 25.0
