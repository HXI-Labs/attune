#!/usr/bin/env python3
"""Compare FP32 and portable INT8 fused affect calibration on development data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from evaluate_truncated_affect_fusion import alignment_key, index_rows, load_rows, softmax
from fine_tune_truncated_affect_controls import CacheOnlyEncoder
from train_truncated_emotion2vec_probe import extract_features

from attune.evaluation.metrics import macro_f1
from attune.inference.affect_fusion import DEFAULT_ACOUSTIC_WEIGHT
from attune.inference.emotion2vec_student import (
    OnnxTruncatedEmotion2VecPredictor,
    TorchScriptTruncatedEmotion2VecPredictor,
)
from attune.models.truncated_emotion2vec import TruncatedEmotion2VecHead
from attune.schema.output import AffectCategory
from attune.training.data import file_sha256
from attune.training.prepare import load_source_rows

LABELS = tuple(category.value for category in AffectCategory)


def _predictor(model: Path, batch_size: int):
    if model.suffix == ".onnx":
        return OnnxTruncatedEmotion2VecPredictor(model, batch_size=batch_size)
    if model.suffix == ".pt":
        return TorchScriptTruncatedEmotion2VecPredictor(model, batch_size=batch_size)
    raise ValueError("portable affect model must use an .onnx or .pt extension")


def _calibration_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    predicted = probabilities.argmax(axis=-1)
    expected = targets.argmax(axis=-1)
    confidence = probabilities.max(axis=-1)
    evaluated_labels = [index for index in range(len(LABELS)) if (expected == index).any()]
    f1 = macro_f1(
        [LABELS[index] for index in expected],
        [LABELS[index] for index in predicted],
        labels=[LABELS[index] for index in evaluated_labels],
    )
    calibration_error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        selected = (confidence >= lower) & (confidence < lower + 0.1)
        if selected.any():
            calibration_error += float(selected.mean()) * abs(
                float((predicted[selected] == expected[selected]).mean())
                - float(confidence[selected].mean())
            )
    retained = confidence >= 0.4
    return {
        "macro_f1": f1,
        "brier_score": float(np.square(probabilities - targets).sum(axis=-1).mean()),
        "negative_log_likelihood": float(
            -(targets * np.log(np.clip(probabilities, 1e-8, 1.0))).sum(axis=-1).mean()
        ),
        "expected_calibration_error": calibration_error,
        "coverage_at_0_4": float(retained.mean()),
        "retained_accuracy_at_0_4": (
            float((predicted[retained] == expected[retained]).mean()) if retained.any() else 0.0
        ),
    }


def _corpus_weights(dataset_ids: np.ndarray) -> torch.Tensor:
    weights = np.zeros(len(dataset_ids), dtype=np.float64)
    for dataset_id in np.unique(dataset_ids):
        selected = dataset_ids == dataset_id
        weights[selected] = 1.0 / selected.sum()
    weights /= weights.mean()
    return torch.from_numpy(weights)


def _fit_probability_calibration(
    probabilities: np.ndarray,
    targets: np.ndarray,
    dataset_ids: np.ndarray,
) -> tuple[float, np.ndarray]:
    logits = torch.from_numpy(np.log(np.clip(probabilities, 1e-8, 1.0))).double()
    soft_targets = torch.from_numpy(targets).double()
    weights = _corpus_weights(dataset_ids)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros(len(LABELS), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature, bias],
        lr=0.25,
        max_iter=150,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.25, 4.0)
        centered_bias = bias - bias.mean()
        log_probabilities = (logits / temperature + centered_bias).log_softmax(dim=-1)
        cross_entropy = (-(soft_targets * log_probabilities).sum(dim=-1) * weights).mean()
        regularization = 0.01 * (centered_bias.square().mean() + log_temperature.square())
        loss = cross_entropy + regularization
        loss.backward()
        return loss

    optimizer.step(closure)
    return (
        float(log_temperature.detach().exp().clamp(0.25, 4.0)),
        (bias.detach() - bias.detach().mean()).numpy(),
    )


def _apply_probability_calibration(
    probabilities: np.ndarray,
    temperature: float,
    bias: np.ndarray,
) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-8, 1.0)) / temperature + bias
    return softmax(logits)


def _speaker_group_folds(sources: list, fold_count: int = 5) -> np.ndarray:
    folds = np.empty(len(sources), dtype=np.int64)
    for dataset_id in sorted({source.dataset_id for source in sources}):
        groups: dict[str, list[int]] = {}
        for index, source in enumerate(sources):
            if source.dataset_id != dataset_id:
                continue
            group = source.speaker_id or source.clip_id
            groups.setdefault(group, []).append(index)
        fold_sizes = [0] * fold_count
        for _, indices in sorted(groups.items(), key=lambda row: (-len(row[1]), row[0])):
            fold = min(range(fold_count), key=lambda value: (fold_sizes[value], value))
            folds[indices] = fold
            fold_sizes[fold] += len(indices)
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, action="append", required=True)
    parser.add_argument("--cadence-scores", type=Path, action="append", required=True)
    parser.add_argument("--cadence-calibration", type=Path, required=True)
    parser.add_argument("--acoustic-weight", type=float, default=DEFAULT_ACOUSTIC_WEIGHT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    cadence_rows = index_rows([row for path in arguments.cadence_scores for row in load_rows(path)])
    sources = [
        row
        for path in arguments.source_manifest
        for row in load_source_rows(path)
        if row.split == "development"
        and row.affect_distribution is not None
        and alignment_key(row.model_dump()) in cadence_rows
    ]
    if not sources:
        raise ValueError("no development rows align with the Cadence score files")

    predictor = _predictor(arguments.model, arguments.batch_size)
    int8_student = predictor.predict_probabilities(
        [source.audio_path.read_bytes() for source in sources]
    )

    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    head = TruncatedEmotion2VecHead(
        int(checkpoint["feature_mean"].numel()),
        len(LABELS),
        kind=str(checkpoint["head_kind"]),
    ).eval()
    head.load_state_dict(checkpoint["state_dict"])
    features, _, cached_sources = extract_features(
        [(source, source.affect_distribution) for source in sources],
        arguments.feature_cache,
        CacheOnlyEncoder(),
        device=torch.device("cpu"),
        depths=(3,),
        batch_size=1,
    )
    if [source.clip_id for source in cached_sources] != [source.clip_id for source in sources]:
        raise RuntimeError("feature-cache order changed during calibration evaluation")
    normalized = (features[3] - checkpoint["feature_mean"].float()) / checkpoint[
        "feature_scale"
    ].float()
    with torch.inference_mode():
        fp32_student = (head(normalized) / float(checkpoint["temperature"])).softmax(dim=-1).numpy()

    calibration = json.loads(arguments.cadence_calibration.read_text())
    cadence = softmax(
        np.asarray(
            [
                cadence_rows[alignment_key(source.model_dump())]["affect_logits"]
                for source in sources
            ]
        )
        / float(calibration["affect_temperature"])
        + np.asarray(calibration["affect_bias"])
    )
    targets = np.asarray(
        [[source.affect_distribution[label] for label in LABELS] for source in sources]
    )
    fp32_fused = (
        1.0 - arguments.acoustic_weight
    ) * cadence + arguments.acoustic_weight * fp32_student
    int8_fused = (
        1.0 - arguments.acoustic_weight
    ) * cadence + arguments.acoustic_weight * int8_student

    per_dataset = {}
    for dataset_id in sorted({source.dataset_id for source in sources}):
        selected = np.asarray([source.dataset_id == dataset_id for source in sources])
        per_dataset[dataset_id] = {
            "clips": int(selected.sum()),
            "fp32": _calibration_metrics(fp32_fused[selected], targets[selected]),
            "int8": _calibration_metrics(int8_fused[selected], targets[selected]),
        }
    fp32_metrics = _calibration_metrics(fp32_fused, targets)
    int8_metrics = _calibration_metrics(int8_fused, targets)
    report = {
        "schema_version": "1.0",
        "method": "portable_int8_post_export_calibration_parity",
        "clips": len(sources),
        "fp32": fp32_metrics,
        "int8": int8_metrics,
        "per_dataset": per_dataset,
        "mean_l1_fused_probability_difference": float(
            np.abs(fp32_fused - int8_fused).sum(axis=-1).mean()
        ),
        "gates": {
            "macro_f1_degradation_at_most_0_02": (
                fp32_metrics["macro_f1"] - int8_metrics["macro_f1"] <= 0.02
            ),
            "brier_degradation_at_most_0_02": (
                int8_metrics["brier_score"] - fp32_metrics["brier_score"] <= 0.02
            ),
            "nll_degradation_at_most_0_05": (
                int8_metrics["negative_log_likelihood"] - fp32_metrics["negative_log_likelihood"]
                <= 0.05
            ),
        },
    }
    if arguments.calibration_output is not None:
        dataset_ids = np.asarray([source.dataset_id for source in sources])
        folds = _speaker_group_folds(sources)
        calibrated = np.zeros_like(int8_fused)
        fold_parameters = []
        for fold in range(5):
            development = folds != fold
            held_out = folds == fold
            temperature, bias = _fit_probability_calibration(
                int8_fused[development],
                targets[development],
                dataset_ids[development],
            )
            calibrated[held_out] = _apply_probability_calibration(
                int8_fused[held_out],
                temperature,
                bias,
            )
            fold_parameters.append(
                {
                    "fold": fold,
                    "training_clips": int(development.sum()),
                    "held_out_clips": int(held_out.sum()),
                    "temperature": temperature,
                    "bias": bias.tolist(),
                }
            )
        cross_validated_metrics = _calibration_metrics(calibrated, targets)
        temperature, bias = _fit_probability_calibration(int8_fused, targets, dataset_ids)
        calibration = {
            "schema_version": "1.0",
            "method": "speaker_grouped_corpus_balanced_soft_label_calibration",
            "labels": LABELS,
            "model_sha256": file_sha256(arguments.model),
            "temperature": temperature,
            "bias": bias.tolist(),
            "folds": fold_parameters,
            "cross_validated_metrics": cross_validated_metrics,
        }
        arguments.calibration_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.calibration_output.write_text(json.dumps(calibration, indent=2) + "\n")
        report["post_quantization_calibration"] = {
            "cross_validated": cross_validated_metrics,
            "temperature": temperature,
            "bias": bias.tolist(),
            "gates": {
                "macro_f1_not_worse": (
                    cross_validated_metrics["macro_f1"] >= int8_metrics["macro_f1"]
                ),
                "brier_not_worse": (
                    cross_validated_metrics["brier_score"] <= int8_metrics["brier_score"]
                ),
                "nll_not_worse": (
                    cross_validated_metrics["negative_log_likelihood"]
                    <= int8_metrics["negative_log_likelihood"]
                ),
            },
        }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
