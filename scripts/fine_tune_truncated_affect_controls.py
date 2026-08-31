#!/usr/bin/env python3
"""Reduce named-affect false alarms while preserving an incumbent affect head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from evaluate_truncated_affect_fusion import (
    alignment_key,
    index_rows,
    load_rows,
    macro_f1,
    softmax,
)
from torch import nn
from train_truncated_emotion2vec_probe import (
    LABELS,
    TRANSITION_TARGETS,
    _softmax,
    extract_features,
    infer,
    matched_rows,
)

from attune.inference.affect_fusion import DEFAULT_ACOUSTIC_WEIGHT
from attune.models.truncated_emotion2vec import TruncatedEmotion2VecHead
from attune.training.data import file_sha256, manifest_sha256
from attune.training.prepare import SourceRow, load_source_rows

CONTROL_TARGET = {label: 0.5 if label in {"neutral", "other"} else 0.0 for label in LABELS}
NON_NAMED_INDICES = (LABELS.index("neutral"), LABELS.index("other"))


class CacheOnlyEncoder(nn.Module):
    def extract_features(self, *_args, **_kwargs):
        raise RuntimeError("the truncated emotion2vec feature cache is incomplete")


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != "1.0":
        raise ValueError("invalid truncated emotion2vec checkpoint")
    if tuple(checkpoint.get("labels", ())) != LABELS or int(checkpoint.get("depth", 0)) != 3:
        raise ValueError("control fine-tuning requires a depth-three Attune affect checkpoint")
    return checkpoint


def new_head(checkpoint: dict[str, Any]) -> TruncatedEmotion2VecHead:
    feature_count = int(checkpoint["feature_mean"].numel())
    head = TruncatedEmotion2VecHead(
        feature_count,
        len(LABELS),
        kind=str(checkpoint.get("head_kind", "mlp")),
    )
    head.load_state_dict(checkpoint["state_dict"])
    return head


def control_rows(path: Path, *, split: str) -> list[tuple[SourceRow, dict[str, float]]]:
    rows = [row for row in load_source_rows(path) if row.split == split]
    if not rows:
        raise ValueError(f"{path} contains no {split} controls")
    return [(row, CONTROL_TARGET) for row in rows]


def probabilities(
    head: TruncatedEmotion2VecHead,
    features: torch.Tensor,
    temperature: float,
) -> np.ndarray:
    return _softmax(infer(head, features) / temperature)


def cadence_index(path: Path, *, dataset_id: str | None = None) -> dict[str, dict]:
    rows = load_rows(path)
    if dataset_id is not None:
        rows = [row for row in rows if row["dataset_id"] == dataset_id]
    return index_rows(rows)


def calibrated_cadence(
    rows: list[dict],
    temperature: float,
    bias: np.ndarray,
) -> np.ndarray:
    return softmax(np.asarray([row["affect_logits"] for row in rows]) / temperature + bias)


def fused_development_metrics(
    student_probabilities: np.ndarray,
    development_sources: list[SourceRow],
    control_probabilities: np.ndarray,
    control_sources: list[SourceRow],
    *,
    cadence_development: dict[str, dict[str, dict]],
    cadence_controls: dict[str, dict],
    cadence_temperature: float,
    cadence_bias: np.ndarray,
    acoustic_weight: float,
    incumbent_probabilities: np.ndarray,
) -> dict[str, float | int]:
    student = {
        alignment_key({"dataset_id": source.dataset_id, "clip_id": source.clip_id}): row
        for source, row in zip(development_sources, student_probabilities, strict=True)
    }
    dataset_scores = {}
    for name, cadence in cadence_development.items():
        keys = sorted(set(cadence) & set(student))
        targets = np.asarray(
            [np.argmax(cadence[key]["affect_distribution"]) for key in keys],
            dtype=np.int64,
        )
        cadence_probabilities = calibrated_cadence(
            [cadence[key] for key in keys],
            cadence_temperature,
            cadence_bias,
        )
        fused = (1.0 - acoustic_weight) * cadence_probabilities + acoustic_weight * np.asarray(
            [student[key] for key in keys]
        )
        dataset_scores[name] = macro_f1(fused.argmax(axis=-1), targets)

    control_cadence = calibrated_cadence(
        [cadence_controls[source.clip_id] for source in control_sources],
        cadence_temperature,
        cadence_bias,
    )
    fused_controls = (
        1.0 - acoustic_weight
    ) * control_cadence + acoustic_weight * control_probabilities
    control_prediction = fused_controls.argmax(axis=-1)
    named = ~np.isin(control_prediction, NON_NAMED_INDICES)
    confidence = fused_controls.max(axis=-1)
    return {
        "crema_macro_f1": dataset_scores["crema"],
        "berst_macro_f1": dataset_scores["berst"],
        "mean_macro_f1": float(np.mean(list(dataset_scores.values()))),
        "control_named_count": int(named.sum()),
        "control_named_at_0_4": int((named & (confidence >= 0.4)).sum()),
        "control_coverage_at_0_4": float((confidence >= 0.4).mean()),
        "mean_probability_drift": float(
            np.abs(student_probabilities - incumbent_probabilities).sum(axis=-1).mean()
        ),
    }


def fine_tune(
    checkpoint: dict[str, Any],
    training_features: torch.Tensor,
    control_features: torch.Tensor,
    replay_targets: torch.Tensor,
    *,
    temperature: float,
    control_weight: float,
    step_count: int,
    learning_rate: float,
    seed: int,
) -> TruncatedEmotion2VecHead:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    head = new_head(checkpoint).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    for _ in range(step_count):
        replay_indices = torch.randint(len(training_features), (128,), generator=generator)
        control_indices = torch.randint(len(control_features), (64,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        replay_log_probabilities = (
            head(training_features[replay_indices]) / temperature
        ).log_softmax(dim=-1)
        replay_loss = (
            -(replay_targets[replay_indices] * replay_log_probabilities).sum(dim=-1).mean()
        )
        predicted_controls = (head(control_features[control_indices]) / temperature).softmax(dim=-1)
        non_named_probability = predicted_controls[:, list(NON_NAMED_INDICES)].sum(dim=-1)
        control_loss = -non_named_probability.clamp_min(1e-8).log().mean()
        loss = replay_loss + control_weight * control_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
    return head.eval()


def evaluate_external(
    head: TruncatedEmotion2VecHead,
    features: torch.Tensor,
    targets: torch.Tensor,
    sources: list[SourceRow],
    cadence: dict[str, dict],
    *,
    temperature: float,
    cadence_temperature: float,
    cadence_bias: np.ndarray,
    acoustic_weight: float,
) -> dict[str, Any]:
    student = probabilities(head, features, temperature)
    keys = [
        alignment_key({"dataset_id": source.dataset_id, "clip_id": source.clip_id})
        for source in sources
    ]
    cadence_probabilities = calibrated_cadence(
        [cadence[key] for key in keys], cadence_temperature, cadence_bias
    )
    fused = (1.0 - acoustic_weight) * cadence_probabilities + acoustic_weight * student
    truth = targets.argmax(dim=-1).numpy()
    transition = []
    for index, source in enumerate(sources):
        if source.clip_id not in TRANSITION_TARGETS:
            continue
        expected, contrast = TRANSITION_TARGETS[source.clip_id]
        expected_probability = float(fused[index, LABELS.index(expected)])
        contrast_probability = float(fused[index, LABELS.index(contrast)])
        transition.append(
            {
                "clip_id": source.clip_id,
                "expected": expected,
                "contrast": contrast,
                "expected_probability": expected_probability,
                "contrast_probability": contrast_probability,
                "passed": expected_probability > contrast_probability,
            }
        )
    return {
        "student_macro_f1": macro_f1(student.argmax(axis=-1), truth),
        "fused_macro_f1": macro_f1(fused.argmax(axis=-1), truth),
        "fused_accuracy": float((fused.argmax(axis=-1) == truth).mean()),
        "transition": transition,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, action="append", required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--control-training-manifest", type=Path, required=True)
    parser.add_argument("--control-development-manifest", type=Path, required=True)
    parser.add_argument("--external-source-manifest", type=Path, required=True)
    parser.add_argument("--external-target-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--incumbent-checkpoint", type=Path, required=True)
    parser.add_argument("--cadence-calibration", type=Path, required=True)
    parser.add_argument("--cadence-crema", type=Path, required=True)
    parser.add_argument("--cadence-berst", type=Path, required=True)
    parser.add_argument("--cadence-controls", type=Path, required=True)
    parser.add_argument("--cadence-external", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--control-weight", type=float, action="append", default=[])
    parser.add_argument("--step-count", type=int, action="append", default=[])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--acoustic-weight", type=float, default=DEFAULT_ACOUSTIC_WEIGHT)
    parser.add_argument("--minimum-crema-macro-f1", type=float, default=0.64)
    parser.add_argument("--maximum-berst-drop", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    control_weights = sorted(set(arguments.control_weight or (0.02, 0.05, 0.1, 0.2, 0.35)))
    step_counts = sorted(set(arguments.step_count or (5, 10, 15, 20, 25)))
    if any(weight <= 0 for weight in control_weights) or any(count < 1 for count in step_counts):
        parser.error("control weights and step counts must be positive")

    checkpoint = load_checkpoint(arguments.incumbent_checkpoint)
    encoder = CacheOnlyEncoder()
    training_rows = matched_rows(
        arguments.source_manifest, arguments.target_manifest, split="train"
    )
    development_rows = matched_rows(
        arguments.source_manifest, arguments.target_manifest, split="development"
    )
    external_rows = matched_rows(
        [arguments.external_source_manifest],
        arguments.external_target_manifest,
        split="sealed_test",
    )
    training_x, _, _ = extract_features(
        training_rows,
        arguments.feature_cache,
        encoder,
        device=torch.device("cpu"),
        depths=(3,),
        batch_size=1,
    )
    development_x, _, development_sources = extract_features(
        development_rows,
        arguments.feature_cache,
        encoder,
        device=torch.device("cpu"),
        depths=(3,),
        batch_size=1,
    )
    control_training_x, _, _ = extract_features(
        control_rows(arguments.control_training_manifest, split="train"),
        arguments.feature_cache,
        encoder,
        device=torch.device("cpu"),
        depths=(3,),
        batch_size=1,
    )
    control_development_x, _, control_development_sources = extract_features(
        control_rows(arguments.control_development_manifest, split="development"),
        arguments.feature_cache,
        encoder,
        device=torch.device("cpu"),
        depths=(3,),
        batch_size=1,
    )

    feature_mean = checkpoint["feature_mean"].float()
    feature_scale = checkpoint["feature_scale"].float()

    def normalize(values: torch.Tensor) -> torch.Tensor:
        return (values - feature_mean) / feature_scale

    training_features = normalize(training_x[3])
    development_features = normalize(development_x[3])
    control_training_features = normalize(control_training_x[3])
    control_development_features = normalize(control_development_x[3])
    temperature = float(checkpoint["temperature"])
    incumbent = new_head(checkpoint).eval()
    with torch.inference_mode():
        replay_targets = (incumbent(training_features) / temperature).softmax(dim=-1)
    incumbent_development = probabilities(incumbent, development_features, temperature)

    calibration = json.loads(arguments.cadence_calibration.read_text())
    cadence_temperature = float(calibration["affect_temperature"])
    cadence_bias = np.asarray(calibration["affect_bias"], dtype=np.float64)
    cadence_development = {
        "crema": cadence_index(arguments.cadence_crema, dataset_id="crema_d_paired_v0.1"),
        "berst": cadence_index(arguments.cadence_berst, dataset_id="berst_v1"),
    }
    cadence_controls = {
        row["clip_id"]: row
        for row in load_rows(arguments.cadence_controls)
        if row["dataset_id"] == "common_voice_17_en"
    }

    def evaluate(head: TruncatedEmotion2VecHead) -> dict[str, float | int]:
        return fused_development_metrics(
            probabilities(head, development_features, temperature),
            development_sources,
            probabilities(head, control_development_features, temperature),
            control_development_sources,
            cadence_development=cadence_development,
            cadence_controls=cadence_controls,
            cadence_temperature=cadence_temperature,
            cadence_bias=cadence_bias,
            acoustic_weight=arguments.acoustic_weight,
            incumbent_probabilities=incumbent_development,
        )

    incumbent_metrics = evaluate(incumbent)
    candidates = []
    candidate_states = {}
    for control_weight in control_weights:
        for step_count in step_counts:
            head = fine_tune(
                checkpoint,
                training_features,
                control_training_features,
                replay_targets,
                temperature=temperature,
                control_weight=control_weight,
                step_count=step_count,
                learning_rate=arguments.learning_rate,
                seed=arguments.seed,
            )
            candidate = {
                "control_weight": control_weight,
                "step_count": step_count,
                **evaluate(head),
            }
            candidates.append(candidate)
            candidate_states[(control_weight, step_count)] = {
                name: value.detach().clone() for name, value in head.state_dict().items()
            }
            print(json.dumps(candidate), flush=True)

    eligible = [
        candidate
        for candidate in candidates
        if candidate["crema_macro_f1"] >= arguments.minimum_crema_macro_f1
        and candidate["berst_macro_f1"]
        >= incumbent_metrics["berst_macro_f1"] - arguments.maximum_berst_drop
    ]
    if not eligible:
        raise RuntimeError("no control fine-tuning candidate satisfies the development gates")
    selected = min(
        eligible,
        key=lambda candidate: (
            candidate["control_named_count"],
            candidate["control_named_at_0_4"],
            -candidate["mean_macro_f1"],
            candidate["mean_probability_drift"],
        ),
    )
    selected_state = candidate_states[(selected["control_weight"], selected["step_count"])]
    selected_head = new_head(checkpoint)
    selected_head.load_state_dict(selected_state)

    external_x, external_y, external_sources = extract_features(
        external_rows,
        arguments.feature_cache,
        encoder,
        device=torch.device("cpu"),
        depths=(3,),
        batch_size=1,
    )
    external = evaluate_external(
        selected_head,
        normalize(external_x[3]),
        external_y,
        external_sources,
        cadence_index(arguments.cadence_external),
        temperature=temperature,
        cadence_temperature=cadence_temperature,
        cadence_bias=cadence_bias,
        acoustic_weight=arguments.acoustic_weight,
    )

    selection_rule = (
        "minimize named-affect emissions on Common Voice development controls among "
        f"candidates with CREMA macro-F1 >= {arguments.minimum_crema_macro_f1} and "
        f"BERSt drop <= {arguments.maximum_berst_drop}; break ties by development macro-F1"
    )
    output_checkpoint = {
        **checkpoint,
        "state_dict": selected_state,
        "negative_control_finetune": {
            "control_weight": selected["control_weight"],
            "step_count": selected["step_count"],
            "learning_rate": arguments.learning_rate,
            "seed": arguments.seed,
            "selection_rule": selection_rule,
            "selection_uses_external": False,
        },
    }
    arguments.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, arguments.checkpoint_output)
    report = {
        "schema_version": "1.0",
        "method": "incumbent_probability_replay_plus_no_named_affect_control",
        "selection_uses_external": False,
        "selection_rule": selection_rule,
        "incumbent": incumbent_metrics,
        "selected": selected,
        "external": external,
        "candidates": candidates,
        "inputs": {
            "source_manifests": [
                {"path": str(path), "sha256": manifest_sha256(path)}
                for path in arguments.source_manifest
            ],
            "target_manifest": {
                "path": str(arguments.target_manifest),
                "sha256": manifest_sha256(arguments.target_manifest),
            },
            "control_training_manifest": {
                "path": str(arguments.control_training_manifest),
                "sha256": manifest_sha256(arguments.control_training_manifest),
            },
            "control_development_manifest": {
                "path": str(arguments.control_development_manifest),
                "sha256": manifest_sha256(arguments.control_development_manifest),
            },
            "incumbent_checkpoint": {
                "path": str(arguments.incumbent_checkpoint),
                "sha256": file_sha256(arguments.incumbent_checkpoint),
            },
        },
    }
    arguments.report_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.report_output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "selected": selected,
                "external": external,
                "checkpoint": str(arguments.checkpoint_output),
                "report": str(arguments.report_output),
            }
        )
    )


if __name__ == "__main__":
    main()
