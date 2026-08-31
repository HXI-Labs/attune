#!/usr/bin/env python3
"""Select and evaluate Cadence/truncated-emotion2vec probability fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from attune.inference.affect_fusion import CADENCE_PARAMETER_COUNT, PARAMETER_LIMIT
from attune.schema.output import AffectCategory

LABELS = tuple(category.value for category in AffectCategory)
TRANSITION_TARGETS = {
    "ravdess-03-01-03-02-01-02-18": ("joy", "fear"),
    "ravdess-03-01-04-02-02-01-18": ("distress", "fear"),
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def alignment_key(row: dict[str, Any]) -> str:
    clip_id = str(row["clip_id"])
    for prefix in ("crema-joint-", "crema-perceptual-"):
        if clip_id.startswith(prefix):
            return f"crema:{clip_id.removeprefix(prefix)}"
    return f"{row['dataset_id']}:{clip_id}"


def index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {alignment_key(row): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("score rows contain duplicate alignment keys")
    return indexed


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponent = np.exp(np.clip(shifted, -80.0, 0.0))
    return exponent / exponent.sum(axis=-1, keepdims=True)


def macro_f1(prediction: np.ndarray, target: np.ndarray) -> float:
    scores = []
    for label in np.unique(target):
        predicted = prediction == label
        expected = target == label
        true_positive = int((predicted & expected).sum())
        false_positive = int((predicted & ~expected).sum())
        false_negative = int((~predicted & expected).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


def aligned_probabilities(
    cadence_rows: list[dict[str, Any]],
    student_rows: list[dict[str, Any]],
    *,
    cadence_temperature: float,
    cadence_bias: np.ndarray,
    student_temperature: float,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    cadence = index_rows(cadence_rows)
    student = index_rows(student_rows)
    keys = sorted(set(cadence) & set(student))
    if not keys:
        raise ValueError("Cadence and student scores have no shared clips")
    targets = np.asarray(
        [np.argmax(cadence[key]["affect_distribution"]) for key in keys],
        dtype=np.int64,
    )
    cadence_probabilities = softmax(
        np.asarray([cadence[key]["affect_logits"] for key in keys]) / cadence_temperature
        + cadence_bias
    )
    student_probabilities = softmax(
        np.asarray([student[key]["affect_logits"] for key in keys]) / student_temperature
    )
    return keys, targets, cadence_probabilities, student_probabilities


def evaluate_weight(
    targets: np.ndarray,
    cadence: np.ndarray,
    student: np.ndarray,
    weight: float,
) -> dict[str, float]:
    probabilities = (1.0 - weight) * cadence + weight * student
    prediction = probabilities.argmax(axis=-1)
    return {
        "macro_f1": macro_f1(prediction, targets),
        "accuracy": float((prediction == targets).mean()),
        "mean_confidence": float(probabilities.max(axis=-1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cadence-calibration", type=Path, required=True)
    parser.add_argument("--cadence-crema", type=Path, required=True)
    parser.add_argument("--cadence-berst", type=Path, required=True)
    parser.add_argument("--cadence-ravdess", type=Path, required=True)
    parser.add_argument("--student-report", type=Path, required=True)
    parser.add_argument("--student-development", type=Path, required=True)
    parser.add_argument("--student-ravdess", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    calibration = json.loads(arguments.cadence_calibration.read_text())
    student_report = json.loads(arguments.student_report.read_text())
    depth = str(student_report["selected_depth"])
    student_candidate = student_report["candidates"][depth]
    student_temperature = float(student_candidate["temperature"])
    cadence_temperature = float(calibration["affect_temperature"])
    cadence_bias = np.asarray(calibration["affect_bias"], dtype=np.float64)
    student_development = load_rows(arguments.student_development)

    cadence_crema = [
        row
        for row in load_rows(arguments.cadence_crema)
        if row["dataset_id"] == "crema_d_paired_v0.1"
    ]
    student_crema = [
        row for row in student_development if row["dataset_id"] == "crema_d_perceptual_v1"
    ]
    development = {}
    development_arrays = {}
    for name, cadence_rows, student_rows in (
        ("crema", cadence_crema, student_crema),
        ("berst", load_rows(arguments.cadence_berst), student_development),
    ):
        _, targets, cadence_probabilities, student_probabilities = aligned_probabilities(
            cadence_rows,
            student_rows,
            cadence_temperature=cadence_temperature,
            cadence_bias=cadence_bias,
            student_temperature=student_temperature,
        )
        development_arrays[name] = (targets, cadence_probabilities, student_probabilities)

    candidates = []
    for weight in np.linspace(0.0, 1.0, 101):
        per_dataset = {
            name: evaluate_weight(*arrays, float(weight))
            for name, arrays in development_arrays.items()
        }
        candidates.append(
            {
                "acoustic_weight": float(weight),
                "selection_score": float(
                    np.mean([metrics["macro_f1"] for metrics in per_dataset.values()])
                ),
                "development": per_dataset,
            }
        )
    selected = max(
        candidates,
        key=lambda candidate: (candidate["selection_score"], -candidate["acoustic_weight"]),
    )
    selected_weight = selected["acoustic_weight"]
    development = selected["development"]

    external_keys, external_targets, cadence_external, student_external = aligned_probabilities(
        load_rows(arguments.cadence_ravdess),
        load_rows(arguments.student_ravdess),
        cadence_temperature=cadence_temperature,
        cadence_bias=cadence_bias,
        student_temperature=student_temperature,
    )
    external = evaluate_weight(
        external_targets,
        cadence_external,
        student_external,
        selected_weight,
    )
    fused_external = (1.0 - selected_weight) * cadence_external + selected_weight * student_external
    transition = []
    for clip_id, (expected, contrast) in TRANSITION_TARGETS.items():
        key = f"ravdess_v1_external_affect:{clip_id}"
        index = external_keys.index(key)
        expected_probability = float(fused_external[index, LABELS.index(expected)])
        contrast_probability = float(fused_external[index, LABELS.index(contrast)])
        transition.append(
            {
                "clip_id": clip_id,
                "expected": expected,
                "contrast": contrast,
                "expected_probability": expected_probability,
                "contrast_probability": contrast_probability,
                "passed": expected_probability > contrast_probability,
            }
        )
    total_parameters = CADENCE_PARAMETER_COUNT + int(
        student_candidate["truncated_emotion2vec_parameters"]
    )
    gates = {
        "crema_macro_f1_at_least_0_64": development["crema"]["macro_f1"] >= 0.64,
        "berst_macro_f1_at_least_0_30": development["berst"]["macro_f1"] >= 0.30,
        "ravdess_macro_f1_at_least_0_40": external["macro_f1"] >= 0.40,
        "both_transition_controls_pass": all(row["passed"] for row in transition),
        "parameter_limit": total_parameters <= PARAMETER_LIMIT,
    }
    report = {
        "schema_version": "1.0",
        "method": "cadence_truncated_emotion2vec_probability_fusion",
        "selection_uses_external": False,
        "selected_acoustic_weight": selected_weight,
        "development": development,
        "ravdess_external": external,
        "transition": transition,
        "total_parameters": total_parameters,
        "parameter_limit": PARAMETER_LIMIT,
        "gates": gates,
        "candidate_passes": all(gates.values()),
        "candidates": candidates,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "candidate_passes": report["candidate_passes"],
                "selected_acoustic_weight": selected_weight,
                "development": development,
                "ravdess_external": external,
                "output": str(arguments.output),
            }
        )
    )


if __name__ == "__main__":
    main()
