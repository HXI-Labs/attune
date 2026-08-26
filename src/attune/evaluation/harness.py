"""Fixture loading and offline-safe baseline harness."""

from __future__ import annotations

import json
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
from attune.evaluation.report import EvaluationItem, evaluate_items
from attune.schema.output import AttuneOutput


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

    transcript = TranscriptSentimentAdapter()
    whisper = WhisperSmallAdapter()
    sensevoice = SenseVoiceSmallAdapter()
    emotion2vec = Emotion2VecPlusAdapter()
    candidates: list[BaselineAdapter] = [transcript, whisper, sensevoice, emotion2vec]
    candidates.extend(
        [
            ModularCascade(asr=whisper, affect=emotion2vec),
            ModularCascade(asr=sensevoice, affect=emotion2vec),
        ]
    )

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
                skipped_runners=skipped if runner is transcript else None,
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
