#!/usr/bin/env python3
"""Distill several compact affect heads into one deployment head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from train_truncated_emotion2vec_probe import (
    LABELS,
    TRANSITION_TARGETS,
    _softmax,
    _temperature,
    extract_features,
    infer,
    matched_rows,
    metrics,
    train_head,
)

from attune.models.truncated_emotion2vec import TruncatedEmotion2VecHead
from attune.training.data import file_sha256, manifest_sha256


class CacheOnlyEncoder(nn.Module):
    def extract_features(self, *_args, **_kwargs):
        raise RuntimeError("the truncated emotion2vec feature cache is incomplete")


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != "1.0":
        raise ValueError(f"invalid teacher checkpoint: {path}")
    if tuple(checkpoint.get("labels", ())) != LABELS:
        raise ValueError(f"teacher labels do not match the affect ontology: {path}")
    if int(checkpoint.get("depth", 0)) != 3:
        raise ValueError(f"distillation requires depth-3 teachers: {path}")
    return checkpoint


def teacher_probabilities(checkpoint: dict, features: torch.Tensor) -> torch.Tensor:
    mean = checkpoint["feature_mean"].float()
    scale = checkpoint["feature_scale"].float()
    head = TruncatedEmotion2VecHead(
        mean.numel(),
        len(LABELS),
        kind=str(checkpoint.get("head_kind", "mlp")),
    )
    head.load_state_dict(checkpoint["state_dict"])
    logits = infer(head, (features - mean) / scale)
    return torch.from_numpy(_softmax(logits / float(checkpoint["temperature"]))).float()


def write_scores(
    path: Path,
    sources: list,
    targets: torch.Tensor,
    logits: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as score_file:
        for source, target, logit in zip(sources, targets, logits, strict=True):
            score_file.write(
                json.dumps(
                    {
                        "dataset_id": source.dataset_id,
                        "clip_id": source.clip_id,
                        "split": source.split,
                        "affect_distribution": target.tolist(),
                        "affect_logits": logit.tolist(),
                    }
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, action="append", required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--external-source-manifest", type=Path, required=True)
    parser.add_argument("--external-target-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--alpha", type=float, action="append", default=[])
    parser.add_argument("--minimum-berst-macro-f1", type=float, default=0.30)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--scores-output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--epoch-samples", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    alphas = sorted(set(arguments.alpha or (0.4, 0.6, 0.8)))
    if any(not 0.0 < alpha < 1.0 for alpha in alphas):
        raise ValueError("distillation alpha must be between zero and one")
    teachers = [load_checkpoint(path) for path in arguments.teacher_checkpoint]
    if len(teachers) < 2:
        raise ValueError("distillation requires at least two teacher checkpoints")

    training_rows = matched_rows(
        arguments.source_manifest,
        arguments.target_manifest,
        split="train",
    )
    development_rows = matched_rows(
        arguments.source_manifest,
        arguments.target_manifest,
        split="development",
    )
    encoder = CacheOnlyEncoder()
    train_x, train_y, train_sources = extract_features(
        training_rows,
        arguments.feature_cache,
        encoder,
        device=torch.device("cpu"),
        depths=(3,),
        batch_size=1,
    )
    development_x, development_y, development_sources = extract_features(
        development_rows,
        arguments.feature_cache,
        encoder,
        device=torch.device("cpu"),
        depths=(3,),
        batch_size=1,
    )
    feature_mean = train_x[3].mean(dim=0)
    feature_scale = train_x[3].std(dim=0).clamp_min(1e-5)
    normalized_train = (train_x[3] - feature_mean) / feature_scale
    normalized_development = (development_x[3] - feature_mean) / feature_scale
    teacher_targets = torch.stack(
        [teacher_probabilities(checkpoint, train_x[3]) for checkpoint in teachers]
    ).mean(dim=0)

    search = []
    candidates = {}
    for alpha in alphas:
        distilled_targets = (1.0 - alpha) * train_y + alpha * teacher_targets
        head, training_report = train_head(
            normalized_train,
            distilled_targets,
            train_sources,
            normalized_development,
            development_y,
            development_sources,
            seed=arguments.seed,
            epochs=arguments.epochs,
            patience=arguments.patience,
            epoch_samples=arguments.epoch_samples,
            learning_rate=arguments.learning_rate,
            head_kind="mlp",
            pair_consistency_weight=0.0,
        )
        development_logits = infer(head, normalized_development)
        temperature = _temperature(development_logits, development_y.numpy())
        development_report = metrics(
            development_logits,
            development_y.numpy(),
            development_sources,
            temperature=temperature,
        )
        by_dataset = development_report["by_dataset"]
        dataset_mean = float(np.mean([dataset["macro_f1"] for dataset in by_dataset.values()]))
        row = {
            "alpha": alpha,
            "development_dataset_macro_f1": dataset_mean,
            "berst_macro_f1": by_dataset["berst_v1"]["macro_f1"],
            "crema_macro_f1": by_dataset["crema_d_perceptual_v1"]["macro_f1"],
            "best_epoch": training_report["best_epoch"],
            "temperature": temperature,
        }
        search.append(row)
        candidates[alpha] = (
            head,
            training_report,
            temperature,
            development_logits,
            development_report,
        )
        print(json.dumps(row), flush=True)

    eligible = [row for row in search if row["berst_macro_f1"] >= arguments.minimum_berst_macro_f1]
    if not eligible:
        raise RuntimeError("no distillation candidate satisfies the BERSt development floor")
    selected = max(eligible, key=lambda row: row["development_dataset_macro_f1"])
    selected_alpha = float(selected["alpha"])
    head, training_report, temperature, development_logits, development_report = candidates[
        selected_alpha
    ]

    external_rows = matched_rows(
        [arguments.external_source_manifest],
        arguments.external_target_manifest,
        split="sealed_test",
    )
    external_x, external_y, external_sources = extract_features(
        external_rows,
        arguments.feature_cache,
        encoder,
        device=torch.device("cpu"),
        depths=(3,),
        batch_size=1,
    )
    normalized_external = (external_x[3] - feature_mean) / feature_scale
    external_logits = infer(head, normalized_external)
    external_report = metrics(
        external_logits,
        external_y.numpy(),
        external_sources,
        temperature=temperature,
    )
    transition = []
    for index, source in enumerate(external_sources):
        if source.clip_id not in TRANSITION_TARGETS:
            continue
        expected, contrast = TRANSITION_TARGETS[source.clip_id]
        logits = dict(zip(LABELS, external_logits[index].tolist(), strict=True))
        transition.append(
            {
                "clip_id": source.clip_id,
                "expected": expected,
                "contrast": contrast,
                "passed": logits[expected] > logits[contrast],
            }
        )

    base = teachers[0]
    teacher_provenance = [
        {"path": str(path), "sha256": file_sha256(path)} for path in arguments.teacher_checkpoint
    ]
    selection_rule = (
        "highest mean development macro-F1 across BERSt and CREMA-D among candidates "
        f"with BERSt macro-F1 >= {arguments.minimum_berst_macro_f1}"
    )
    output_checkpoint = {
        **base,
        "state_dict": head.state_dict(),
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "temperature": temperature,
        "distillation": {
            "teachers": teacher_provenance,
            "alpha": selected_alpha,
            "selection_rule": selection_rule,
            "selection_uses_external": False,
        },
    }
    arguments.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, arguments.checkpoint_output)

    truncated_parameters = int(base["truncated_emotion2vec_parameters"])
    candidate_report = {
        "depth": 3,
        "truncated_emotion2vec_parameters": truncated_parameters,
        "head_parameters": int(base["head_parameters"]),
        "total_with_cadence_parameters": 241_904_650 + truncated_parameters,
        "temperature": temperature,
        "training": training_report,
        "development": development_report,
        "ravdess": external_report,
        "transition": transition,
        "transition_passed": len(transition) == 2 and all(row["passed"] for row in transition),
    }
    report = {
        "schema_version": "1.0",
        "model": "truncated_emotion2vec_affect_student_distilled",
        "selected_depth": 3,
        "selection_uses_external": False,
        "selection_rule": selection_rule,
        "selected_alpha": selected_alpha,
        "distillation_search": search,
        "teachers": teacher_provenance,
        "inputs": {
            "source_manifests": [
                {"path": str(path), "sha256": manifest_sha256(path)}
                for path in arguments.source_manifest
            ],
            "target_manifest": str(arguments.target_manifest),
            "target_manifest_sha256": manifest_sha256(arguments.target_manifest),
            "external_source_manifest": str(arguments.external_source_manifest),
            "external_source_manifest_sha256": manifest_sha256(arguments.external_source_manifest),
            "external_target_manifest": str(arguments.external_target_manifest),
            "external_target_manifest_sha256": manifest_sha256(arguments.external_target_manifest),
        },
        "candidates": {"3": candidate_report},
    }
    arguments.report_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.report_output.write_text(json.dumps(report, indent=2) + "\n")
    write_scores(
        arguments.scores_output_dir / "depth-3-development.jsonl",
        development_sources,
        development_y,
        development_logits,
    )
    write_scores(
        arguments.scores_output_dir / "depth-3-external.jsonl",
        external_sources,
        external_y,
        external_logits,
    )
    print(
        json.dumps(
            {
                "selected_alpha": selected_alpha,
                "development": development_report["by_dataset"],
                "ravdess_external_macro_f1": external_report["macro_f1"],
                "transition_passed": candidate_report["transition_passed"],
                "checkpoint": str(arguments.checkpoint_output),
            }
        )
    )


if __name__ == "__main__":
    main()
