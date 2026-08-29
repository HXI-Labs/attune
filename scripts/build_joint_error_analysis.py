#!/usr/bin/env python3
"""Build a reproducible error-analysis report from sealed joint scores."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from attune.inference.onnx_backend import RuntimeCalibration, _sigmoid, _softmax
from attune.schema.output import AffectCategory


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _edit_distance(reference: list[str], prediction: list[str]) -> int:
    previous = list(range(len(prediction) + 1))
    for reference_index, reference_word in enumerate(reference, start=1):
        current = [reference_index]
        for prediction_index, prediction_word in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[prediction_index] + 1,
                    previous[prediction_index - 1] + (reference_word != prediction_word),
                )
            )
        previous = current
    return previous[-1]


def analyse(
    rows: list[dict[str, Any]],
    calibration: RuntimeCalibration,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    asr_errors = []
    for row in rows:
        if "reference_transcript" not in row:
            continue
        reference = _words(row["reference_transcript"])
        prediction = _words(row["predicted_transcript"])
        edits = _edit_distance(reference, prediction)
        if edits:
            asr_errors.append(
                {
                    "clip_id": row["clip_id"],
                    "dataset_id": row["dataset_id"],
                    "edits": edits,
                    "reference_words": len(reference),
                    "clip_wer": edits / max(1, len(reference)),
                    "reference": row["reference_transcript"],
                    "prediction": row["predicted_transcript"],
                }
            )
    asr_errors.sort(key=lambda item: (-item["clip_wer"], -item["edits"], item["clip_id"]))

    affect_rows = [row for row in rows if "affect_distribution" in row]
    confusion: Counter[tuple[str, str]] = Counter()
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    affect_errors = []
    conflict_counts = Counter()
    categories = tuple(AffectCategory)
    for row in affect_rows:
        probabilities = _softmax(
            np.asarray(row["affect_logits"], dtype=np.float64) / calibration.affect_temperature
        )
        target_index = int(np.argmax(row["affect_distribution"]))
        prediction_index = int(probabilities.argmax())
        target = categories[target_index].value
        prediction = categories[prediction_index].value
        confidence = float(probabilities[prediction_index])
        retained = confidence >= calibration.affect_threshold
        confusion[(target, prediction)] += 1
        class_counts[target]["support"] += 1
        class_counts[target]["correct"] += int(target == prediction)
        class_counts[target]["abstained"] += int(not retained)
        lexical = row.get("lexical_affect_label")
        if lexical is not None and lexical != target:
            conflict_counts["clips"] += 1
            conflict_counts["acoustic_target"] += int(prediction == target)
            conflict_counts["lexical_target"] += int(prediction == lexical)
            conflict_counts["other"] += int(prediction not in {target, lexical})
        if target != prediction:
            affect_errors.append(
                {
                    "clip_id": row["clip_id"],
                    "target": target,
                    "prediction": prediction,
                    "confidence": confidence,
                    "retained": retained,
                    "lexical_target": lexical,
                }
            )
    affect_errors.sort(key=lambda item: (-item["confidence"], item["clip_id"]))

    ood_errors = []
    for row in rows:
        if "ood_logit" not in row:
            continue
        probability = float(_sigmoid(np.asarray(row["ood_logit"]) / calibration.ood_temperature))
        prediction = probability >= calibration.ood_threshold
        target = bool(row.get("is_ood", False))
        if prediction != target:
            ood_errors.append(
                {
                    "clip_id": row["clip_id"],
                    "dataset_id": row["dataset_id"],
                    "target_ood": target,
                    "predicted_ood": bool(prediction),
                    "ood_probability": probability,
                }
            )
    ood_errors.sort(key=lambda item: (-abs(item["ood_probability"] - 0.5), item["clip_id"]))

    return {
        "schema_version": "1.0",
        "clips": len(rows),
        "asr": {
            "evaluated_clips": int(metrics.get("asr_clips", 0)),
            "wer": metrics.get("asr_wer"),
            "clips_with_errors": len(asr_errors),
            "worst_clips": asr_errors[:20],
        },
        "localized_events": metrics.get("event_segments", {}),
        "event_presence_f1": metrics.get("event_presence_f1", {}),
        "style_f1": metrics.get("style_f1", {}),
        "affect": {
            "evaluated_clips": len(affect_rows),
            "macro_f1": metrics.get("affect_macro_f1"),
            "coverage": metrics.get("affect_coverage"),
            "confusion": [
                {"target": target, "prediction": prediction, "count": count}
                for (target, prediction), count in sorted(confusion.items())
            ],
            "per_target": {label: dict(counts) for label, counts in sorted(class_counts.items())},
            "highest_confidence_errors": affect_errors[:20],
            "semantic_conflict": dict(conflict_counts),
        },
        "ood": {
            "f1": metrics.get("ood_f1"),
            "errors": ood_errors,
        },
        "interpretation": [
            "ASR errors are concentrated in the listed clips; inspect accent, "
            "normalization, and audio quality before changing the model.",
            "Affect labels are acted one-hot source labels, so confusion is not "
            "proof of a wrong internal-emotion inference.",
            "Only laugh, cough, and throat_clear segment errors have strong temporal ground truth.",
            "The report is descriptive and was generated after model selection; "
            "it is not a tuning set.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    event = report["localized_events"].get("per_class", {})
    ood_f1 = report["ood"]["f1"]
    ood_summary = (
        f"- OOD F1: `{ood_f1:.4f}` with {len(report['ood']['errors'])} errors."
        if ood_f1 is not None
        else "- OOD F1: unavailable because this slice contains only one OOD class."
    )
    lines = [
        "# Attune v0.1 sealed error analysis",
        "",
        "This report is descriptive. It was generated after model selection and "
        "is not a tuning set.",
        "",
        "## Summary",
        "",
        (
            f"- ASR WER: `{report['asr']['wer']:.4f}` over "
            f"{report['asr']['evaluated_clips']} clips; "
            f"{report['asr']['clips_with_errors']} clips contain at least one word edit."
        ),
        (
            "- Supported-class affect macro-F1: "
            f"`{report['affect']['macro_f1']:.4f}` at "
            f"`{report['affect']['coverage']:.4f}` coverage."
        ),
        ood_summary,
        "",
        "## Localized events",
        "",
        "| Label | F1 | TP | FP | FN | Boundary MAE ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in report["localized_events"].get("localized_labels", []):
        row = event[label]
        boundary = row["boundary_mae_ms"]
        lines.append(
            f"| {label} | {row['segment_f1']:.4f} | {row['true_positive']} | "
            f"{row['false_positive']} | {row['false_negative']} | {boundary:.1f} |"
        )
    lines.extend(["", "## Highest-WER clips", ""])
    for row in report["asr"]["worst_clips"][:10]:
        lines.append(
            f"- `{row['clip_id']}` — WER `{row['clip_wer']:.3f}`; "
            f"reference “{row['reference']}”; prediction “{row['prediction']}”"
        )
    lines.extend(["", "## Highest-confidence affect errors", ""])
    for row in report["affect"]["highest_confidence_errors"][:10]:
        lines.append(
            f"- `{row['clip_id']}` — {row['target']} → {row['prediction']} at "
            f"`{row['confidence']:.3f}`; retained `{str(row['retained']).lower()}`"
        )
    lines.extend(["", "## Interpretation boundary", ""])
    lines.extend(f"- {item}" for item in report["interpretation"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    arguments = parser.parse_args()
    rows = [json.loads(line) for line in arguments.scores.read_text().splitlines() if line]
    calibration = RuntimeCalibration.model_validate_json(arguments.calibration.read_text())
    metrics = json.loads(arguments.metrics.read_text())
    report = analyse(rows, calibration, metrics)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    arguments.markdown.parent.mkdir(parents=True, exist_ok=True)
    arguments.markdown.write_text(render_markdown(report))


if __name__ == "__main__":
    main()
