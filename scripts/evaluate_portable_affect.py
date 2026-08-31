#!/usr/bin/env python3
"""Evaluate a portable affect branch with fixed Cadence probability fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from attune.evaluation.metrics import macro_f1
from attune.inference.affect_fusion import (
    DEFAULT_ACOUSTIC_WEIGHT,
    AffectProbabilityCalibration,
)
from attune.inference.emotion2vec_student import (
    OnnxTruncatedEmotion2VecPredictor,
    TorchScriptTruncatedEmotion2VecPredictor,
)
from attune.schema.output import AffectCategory
from attune.training.data import file_sha256

LABELS = tuple(category.value for category in AffectCategory)
TRANSITION_TARGETS = {
    "ravdess-03-01-03-02-01-02-18": ("joy", "fear"),
    "ravdess-03-01-04-02-02-01-18": ("distress", "fear"),
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _audio_filename(source: dict) -> str:
    if "filename" in source:
        return str(source["filename"])
    if "audio_path" in source:
        return Path(source["audio_path"]).name
    raise ValueError("source manifest row has no filename or audio_path")


def _target_affect(source: dict) -> str:
    if "target_affect" in source:
        return str(source["target_affect"])
    distribution = source.get("affect_distribution")
    if isinstance(distribution, dict) and distribution:
        return str(max(distribution, key=distribution.__getitem__))
    raise ValueError("source manifest row has no affect target")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponent = np.exp(np.clip(shifted, -80.0, 0.0))
    return exponent / exponent.sum(axis=-1, keepdims=True)


def _predictor(model: Path, batch_size: int):
    if model.suffix == ".onnx":
        return OnnxTruncatedEmotion2VecPredictor(model, batch_size=batch_size)
    if model.suffix == ".pt":
        return TorchScriptTruncatedEmotion2VecPredictor(model, batch_size=batch_size)
    raise ValueError("portable affect model must use an .onnx or .pt extension")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--cadence-scores", type=Path, required=True)
    parser.add_argument("--cadence-calibration", type=Path, required=True)
    parser.add_argument("--fp32-reference-macro-f1", type=float, required=True)
    parser.add_argument("--acoustic-weight", type=float, default=DEFAULT_ACOUSTIC_WEIGHT)
    parser.add_argument("--post-calibration", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    source_rows = _jsonl(arguments.source_manifest)
    audio_paths = list(arguments.audio_root.rglob("*.wav"))
    audio_by_name = {path.name: path for path in audio_paths}
    if len(audio_by_name) != len(audio_paths):
        raise ValueError("audio root contains duplicate WAV filenames")
    filenames = [_audio_filename(row) for row in source_rows]
    missing = [filename for filename in filenames if filename not in audio_by_name]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} evaluation WAV files")
    references = [_target_affect(row) for row in source_rows]
    cadence_by_clip = {row["clip_id"]: row for row in _jsonl(arguments.cadence_scores)}
    missing_scores = [
        row["clip_id"] for row in source_rows if row["clip_id"] not in cadence_by_clip
    ]
    if missing_scores:
        raise ValueError(f"missing {len(missing_scores)} Cadence score rows")
    source_index = {row["clip_id"]: index for index, row in enumerate(source_rows)}
    missing_transitions = set(TRANSITION_TARGETS) - source_index.keys()
    if missing_transitions:
        raise ValueError("source manifest is missing the predeclared transition controls")
    calibration = json.loads(arguments.cadence_calibration.read_text())

    predictor = _predictor(arguments.model, arguments.batch_size)
    payloads = [audio_by_name[filename].read_bytes() for filename in filenames]
    started = perf_counter()
    student_probabilities = predictor.predict_probabilities(payloads)
    elapsed = perf_counter() - started

    cadence_probabilities = _softmax(
        np.asarray([cadence_by_clip[row["clip_id"]]["affect_logits"] for row in source_rows])
        / float(calibration["affect_temperature"])
        + np.asarray(calibration["affect_bias"])
    )
    fused = (
        1.0 - arguments.acoustic_weight
    ) * cadence_probabilities + arguments.acoustic_weight * student_probabilities
    post_calibration = None
    if arguments.post_calibration is not None:
        post_calibration = AffectProbabilityCalibration.from_file(
            arguments.post_calibration,
            model_sha256=predictor.artifact_sha256,
        )
        fused = np.stack([post_calibration.apply(row) for row in fused])
    evaluation_labels = tuple(label for label in LABELS if label in set(references))
    student_labels = [LABELS[index] for index in student_probabilities.argmax(axis=-1)]
    fused_labels = [LABELS[index] for index in fused.argmax(axis=-1)]
    fused_macro_f1 = macro_f1(references, fused_labels, labels=evaluation_labels)
    audio_seconds = sum(float(row["duration_ms"]) for row in source_rows) / 1000
    transition = []
    for clip_id, (expected, contrast) in TRANSITION_TARGETS.items():
        index = source_index[clip_id]
        expected_probability = float(fused[index, LABELS.index(expected)])
        contrast_probability = float(fused[index, LABELS.index(contrast)])
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
    degradation = arguments.fp32_reference_macro_f1 - fused_macro_f1
    report = {
        "schema_version": "1.0",
        "method": "portable_artifact_fixed_cadence_probability_fusion",
        "model": {
            "path": str(arguments.model),
            "sha256": file_sha256(arguments.model),
            "quantization": predictor.quantization,
            "parameter_count": predictor.parameter_count,
        },
        "post_calibration": (
            {
                "path": str(arguments.post_calibration),
                "temperature": post_calibration.temperature,
            }
            if post_calibration is not None
            else None
        ),
        "clips": len(source_rows),
        "evaluated_labels": evaluation_labels,
        "student_macro_f1": macro_f1(
            references,
            student_labels,
            labels=evaluation_labels,
        ),
        "fused_macro_f1": fused_macro_f1,
        "fused_accuracy": float(np.mean(np.asarray(fused_labels) == np.asarray(references))),
        "fp32_reference_macro_f1": arguments.fp32_reference_macro_f1,
        "absolute_macro_f1_degradation": degradation,
        "elapsed_seconds": elapsed,
        "audio_seconds": audio_seconds,
        "real_time_factor": elapsed / audio_seconds,
        "transition": transition,
        "gates": {
            "macro_f1_degradation_at_most_0_02": degradation <= 0.02,
            "both_transition_controls_pass": all(row["passed"] for row in transition),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
