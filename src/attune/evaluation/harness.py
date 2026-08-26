"""Fixture loading and offline-safe baseline harness."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from attune.baselines.adapters import (
    BaselineAdapter,
    BaselineInput,
    Emotion2VecPlusAdapter,
    SenseVoiceSmallAdapter,
    TranscriptSentimentAdapter,
    WhisperSmallAdapter,
)
from attune.baselines.cascade import ModularCascade
from attune.evaluation.metrics import (
    _edit_distance,
    acoustic_preference_score,
    corpus_character_error_rate,
    corpus_word_error_rate,
    macro_f1,
    position_aware_event_score,
    span_classification_metrics,
)
from attune.evaluation.report import EvaluationItem, evaluate_items
from attune.schema.output import AffectCategory, AttuneOutput, EventLabel


@dataclass(frozen=True)
class _InspectionSpan:
    label: str
    start_ms: int
    end_ms: int


def _baseline_candidates() -> list[BaselineAdapter]:
    transcript = TranscriptSentimentAdapter()
    whisper = WhisperSmallAdapter()
    sensevoice = SenseVoiceSmallAdapter()
    emotion2vec = Emotion2VecPlusAdapter()
    return [
        transcript,
        whisper,
        sensevoice,
        emotion2vec,
        ModularCascade(asr=whisper, affect=emotion2vec),
        ModularCascade(asr=sensevoice, affect=emotion2vec),
    ]


def run_fixture_harness(fixtures: Path) -> dict[str, Any]:
    """Run the always-available baseline and every locally available optional runner."""
    manifest = json.loads((fixtures / "manifest.json").read_text())
    gold_payloads = json.loads((fixtures / "gold.json").read_text())
    inputs = [
        (
            row,
            BaselineInput(
                audio_path=fixtures / row["audio"],
                transcript_hint=row["transcript"],
            ),
            AttuneOutput.model_validate(gold_payloads[row["id"]]),
        )
        for row in manifest["items"]
    ]

    candidates = _baseline_candidates()

    available: list[BaselineAdapter] = []
    skipped: list[dict[str, str]] = []
    for runner in candidates:
        can_run, reason = runner.availability()
        if can_run:
            available.append(runner)
        else:
            skipped.append({"runner": runner.name, "reason": reason or "unavailable"})

    reports = []
    for runner in available:
        evaluated_items = []
        for row, baseline_input, reference in inputs:
            prediction = runner.predict(baseline_input)
            evaluated_items.append(
                EvaluationItem(
                    item_id=row["id"],
                    reference=reference,
                    prediction=prediction.output,
                    acoustic_target=row["acoustic_target"],
                    lexical_target=row["lexical_target"],
                    runtime=prediction.runtime,
                )
            )
        reports.append(
            evaluate_items(
                runner.name,
                evaluated_items,
                skipped_runners=(
                    skipped if runner.name == TranscriptSentimentAdapter.name else None
                ),
            ).to_dict()
        )
    return {
        "report_version": "1",
        "fixture_version": manifest["fixture_version"],
        "fixture_notice": (
            "Synthetic wiring fixtures only; these metrics are not scientific model results."
        ),
        "reports": reports,
        "skipped_runners": skipped,
    }


def run_inspection_harness(manifest: Path, cache_root: Path) -> dict[str, Any]:
    """Run available baselines on local inspection WAVs without downloading weights."""
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidates = _baseline_candidates()
    runner_results: list[dict[str, Any]] = []
    for runner in candidates:
        try:
            available, reason = runner.availability()
        except Exception as error:  # optional runtimes can fail while being imported
            available, reason = False, f"{type(error).__name__}: {error}"
        if not available:
            runner_results.append(
                {
                    "runner": runner.name,
                    "status": "skipped",
                    "skip_reason": reason or "unavailable",
                    "schema_validity": {"valid": 0, "invalid": 0, "errors": []},
                    "metrics": _empty_inspection_metrics(),
                    "runtime": _empty_runtime(),
                    "failures": [],
                    "qualitative_errors": [],
                }
            )
            continue
        runner_results.append(_run_inspection_runner(runner, rows, cache_root))

    return {
        "report_version": "1",
        "title": "Inspection-set smoke (weak/acted labels, not gold)",
        "scope": {
            "manifest": str(manifest),
            "cache_root": str(cache_root),
            "clip_count": len(rows),
            "source_counts": {
                source: sum(row["source_dataset"] == source for row in rows)
                for source in ("VocalSound", "CREMA-D")
            },
            "label_status": "weak_source_labels",
            "acted_status": "acted",
            "sealed_gold_baseline": False,
            "gate_decision": "not_evaluated",
            "fine_tuning_performed": False,
        },
        "metric_notes": {
            "asr": (
                "WER/CER use lowercase alphanumeric normalization and only "
                "CREMA-D clips with source transcripts."
            ),
            "events": (
                "VocalSound source labels are weak whole-clip event spans; "
                "SenseVoice AED tags are provisional whole-clip predictions, "
                "not frame-level localization. Other runners may emit no events."
            ),
            "affect": (
                "CREMA-D HAP/SAD/NEU source labels map to "
                "joy/distress/neutral and are weak acted labels."
            ),
            "aps": (
                "APS compares CREMA-D acoustic source labels with "
                "transcript-lexicon labels; it is descriptive, not gold."
            ),
        },
        "runner_results": runner_results,
    }


def _run_inspection_runner(
    runner: BaselineAdapter,
    rows: list[dict[str, Any]],
    cache_root: Path,
) -> dict[str, Any]:
    records: list[tuple[dict[str, Any], AttuneOutput, Any]] = []
    failures: list[dict[str, str]] = []
    schema_errors: list[dict[str, str]] = []
    transcript_only_skips = 0
    for row in rows:
        transcript_hint = row.get("source_metadata", {}).get("transcript")
        if runner.name == TranscriptSentimentAdapter.name and transcript_hint is None:
            transcript_only_skips += 1
            continue
        try:
            prediction = runner.predict(
                BaselineInput(
                    audio_path=cache_root / row["cache_path"],
                    transcript_hint=transcript_hint,
                )
            )
            output = AttuneOutput.model_validate(prediction.output.model_dump(mode="json"))
            records.append((row, output, prediction.runtime))
        except Exception as error:
            failures.append(
                {
                    "clip_id": row["clip_id"],
                    "source_filename": row["source_filename"],
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
            )
            if "out of memory" in str(error).lower():
                break

    schema_valid = len(records)
    completed = schema_valid + len(failures)
    status = "completed"
    if failures:
        status = "partial_failure" if records else "failed"
    runtime_audio = sum(record[2].audio_seconds for record in records)
    runtime_elapsed = sum(record[2].elapsed_seconds for record in records)
    metrics, qualitative_errors = _inspection_metrics(runner, records)
    return {
        "runner": runner.name,
        "status": status,
        "attempted_clips": completed,
        "not_applicable_clips": transcript_only_skips,
        "not_applicable_reason": (
            "VocalSound has no reference transcript for transcript-only inference."
            if transcript_only_skips
            else None
        ),
        "schema_validity": {
            "valid": schema_valid,
            "invalid": len(schema_errors),
            "errors": schema_errors,
        },
        "metrics": metrics,
        "runtime": {
            "audio_seconds": runtime_audio,
            "elapsed_seconds": runtime_elapsed,
            "real_time_factor": (
                runtime_elapsed / runtime_audio if runtime_audio else None
            ),
            "mean_latency_ms": (
                sum(record[2].latency_ms or 0.0 for record in records) / len(records)
                if records
                else None
            ),
            "first_result_latency_ms": None,
        },
        "failures": failures,
        "qualitative_errors": qualitative_errors,
    }


def _inspection_metrics(
    runner: BaselineAdapter,
    records: list[tuple[dict[str, Any], AttuneOutput, Any]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    crema = [record for record in records if record[0]["source_dataset"] == "CREMA-D"]
    vocalsound = [
        record for record in records if record[0]["source_dataset"] == "VocalSound"
    ]
    asr_capable = runner.name != Emotion2VecPlusAdapter.name
    references = [
        _normalize_asr(row["source_metadata"]["transcript"]) for row, _, _ in crema
    ]
    hypotheses = [_normalize_asr(output.transcript.text) for _, output, _ in crema]
    asr = {
        "evaluated_clips": len(crema) if asr_capable else 0,
        "wer": (
            corpus_word_error_rate(references, hypotheses)
            if crema and asr_capable
            else None
        ),
        "cer": (
            corpus_character_error_rate(references, hypotheses)
            if crema and asr_capable
            else None
        ),
        "note": (
            "Transcript-only receives the supplied source transcript."
            if runner.name == TranscriptSentimentAdapter.name
            else (
                "Not applicable: emotion2vec+ is affect-only and merely retains transcript hints."
                if not asr_capable
                else None
            )
        ),
    }
    event_references: list[_InspectionSpan] = []
    event_predictions: list[_InspectionSpan] = []
    event_position_scores: list[float] = []
    offset = 0
    for row, output, _ in vocalsound:
        duration_ms = output.audio.duration_ms
        reference = [
            _InspectionSpan(
                label=row["intended_attune_labels"]["events"][0],
                start_ms=0,
                end_ms=duration_ms,
            )
        ]
        predictions = [
            _InspectionSpan(
                label=event.label.value,
                start_ms=event.start_ms,
                end_ms=event.end_ms,
            )
            for event in output.events
        ]
        event_position_scores.append(
            position_aware_event_score(reference, predictions, duration_ms=duration_ms)
        )
        event_references.extend(
            _InspectionSpan(span.label, span.start_ms + offset, span.end_ms + offset)
            for span in reference
        )
        event_predictions.extend(
            _InspectionSpan(span.label, span.start_ms + offset, span.end_ms + offset)
            for span in predictions
        )
        offset += duration_ms + 1
    event_scores = span_classification_metrics(
        event_references,
        event_predictions,
        labels=[label.value for label in EventLabel],
    )
    event_scores["position_aware_score"] = (
        sum(event_position_scores) / len(event_position_scores)
        if event_position_scores
        else None
    )
    event_scores["evaluated_clips"] = len(vocalsound)

    affect_references = [
        row["intended_attune_labels"]["affect"][0] for row, _, _ in crema
    ]
    affect_predictions = [
        output.affect.top_label.value
        if output.affect.top_label
        else AffectCategory.AMBIGUOUS.value
        for _, output, _ in crema
    ]
    lexical = TranscriptSentimentAdapter()
    lexical_targets = [
        lexical.classify(row["source_metadata"]["transcript"])[0].value
        for row, _, _ in crema
    ]
    affect = {
        "evaluated_clips": len(crema),
        "mapping": {"HAP": "joy", "SAD": "distress", "NEU": "neutral"},
        "accuracy": (
            sum(
                reference == prediction
                for reference, prediction in zip(
                    affect_references, affect_predictions, strict=True
                )
            )
            / len(crema)
            if crema
            else None
        ),
        "macro_f1": (
            macro_f1(
                affect_references,
                affect_predictions,
                labels=["joy", "distress", "neutral"],
            )
            if crema
            else None
        ),
        "acoustic_preference": (
            acoustic_preference_score(
                affect_predictions, affect_references, lexical_targets
            )
            if crema
            else None
        ),
    }
    errors = _qualitative_errors(crema, vocalsound, asr_capable)
    return {"asr": asr, "events": event_scores, "affect": affect}, errors


def _qualitative_errors(
    crema: list[tuple[dict[str, Any], AttuneOutput, Any]],
    vocalsound: list[tuple[dict[str, Any], AttuneOutput, Any]],
    asr_capable: bool,
) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    if asr_capable:
        ranked = []
        for row, output, _ in crema:
            reference = _normalize_asr(row["source_metadata"]["transcript"]).split()
            hypothesis = _normalize_asr(output.transcript.text).split()
            ranked.append((_edit_distance(reference, hypothesis), row, output))
        for errors, row, output in sorted(ranked, key=lambda item: item[0], reverse=True)[:2]:
            if errors:
                examples.append(
                    {
                        "source_filename": row["source_filename"],
                        "description": (
                            f"ASR made {errors} word-level edit(s); output was "
                            f"{output.transcript.text[:80]!r}."
                        ),
                    }
                )
    for row, output, _ in crema:
        reference = row["intended_attune_labels"]["affect"][0]
        predicted = output.affect.top_label.value if output.affect.top_label else "abstain"
        if predicted != reference:
            examples.append(
                {
                    "source_filename": row["source_filename"],
                    "description": (
                        f"Weak acted affect is {reference}; runner predicted {predicted}."
                    ),
                }
            )
            if len(examples) >= 4:
                break
    if vocalsound and len(examples) < 4:
        row, output, _ = vocalsound[0]
        if not output.events:
            examples.append(
                {
                    "source_filename": row["source_filename"],
                    "description": (
                        f"Weak event is {row['intended_attune_labels']['events'][0]}; "
                        "runner emitted no event."
                    ),
                }
            )
    return examples[:4]


def _normalize_asr(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def _empty_inspection_metrics() -> dict[str, Any]:
    return {
        "asr": {"evaluated_clips": 0, "wer": None, "cer": None},
        "events": {"evaluated_clips": 0, "macro_f1": None, "position_aware_score": None},
        "affect": {
            "evaluated_clips": 0,
            "accuracy": None,
            "macro_f1": None,
            "acoustic_preference": None,
        },
    }


def _empty_runtime() -> dict[str, Any]:
    return {
        "audio_seconds": 0.0,
        "elapsed_seconds": 0.0,
        "real_time_factor": None,
        "mean_latency_ms": None,
        "first_result_latency_ms": None,
    }
