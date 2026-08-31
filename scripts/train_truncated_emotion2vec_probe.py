#!/usr/bin/env python3
"""Fit compact affect heads over truncated, frozen emotion2vec+ representations."""

from __future__ import annotations

import argparse
import array
import json
import math
import re
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from attune.models.truncated_emotion2vec import TruncatedEmotion2VecHead, pool_layer
from attune.schema.output import AffectCategory
from attune.training.data import JointManifestRow, file_sha256, manifest_sha256
from attune.training.prepare import SourceRow, load_source_rows

LABELS = tuple(category.value for category in AffectCategory)
TEACHER_CHECKPOINT_SHA256 = "60710b5aae1dbe69bdac8920028fb05882d4314fd09031922b4b61ee9e7aadbd"
TEACHER_REVISION = "b318240bfe67db81a8c572ecb37ce9c3759b81c9"
FEATURE_VERSION = "emotion2vec-plus-truncated-mean-std-v1"
CADENCE_PARAMETERS = 241_904_650
TRANSITION_TARGETS = {
    "ravdess-03-01-03-02-01-02-18": ("joy", "fear"),
    "ravdess-03-01-04-02-02-01-18": ("distress", "fear"),
}


def load_targets(path: Path) -> dict[tuple[str, str], JointManifestRow]:
    targets = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = JointManifestRow.model_validate_json(line)
        if row.affect_distribution is not None:
            targets[(row.dataset_id, row.clip_id)] = row
    return targets


def matched_rows(
    source_paths: list[Path],
    target_path: Path,
    *,
    split: str,
) -> list[tuple[SourceRow, dict[str, float]]]:
    targets = load_targets(target_path)
    matched = []
    seen: set[tuple[str, str]] = set()
    for source_path in source_paths:
        for source in load_source_rows(source_path):
            if source.split != split or source.affect_distribution is None:
                continue
            key = (source.dataset_id, source.clip_id)
            if key in seen:
                raise ValueError(f"duplicate {split} source row: {key}")
            seen.add(key)
            target = targets.get(key)
            if target is None or target.split != split or target.affect_distribution is None:
                raise ValueError(f"missing {split} target for {source.dataset_id}/{source.clip_id}")
            matched.append((source, target.affect_distribution))
    if not matched:
        raise ValueError(f"no matched {split} affect rows")
    return matched


def _read_waveform(path: Path) -> torch.Tensor:
    try:
        with wave.open(str(path), "rb") as handle:
            if (
                handle.getnchannels() != 1
                or handle.getsampwidth() != 2
                or handle.getframerate() != 16_000
            ):
                raise ValueError(f"{path}: expected mono PCM16 at 16 kHz")
            samples = array.array("h", handle.readframes(handle.getnframes()))
    except (OSError, wave.Error) as error:
        raise ValueError(f"{path}: cannot read WAV: {error}") from error
    waveform = torch.tensor(samples, dtype=torch.float32) / 32_768
    return functional.layer_norm(waveform, waveform.shape)


def _cache_name(source: SourceRow) -> str:
    identity = re.sub(r"[^a-zA-Z0-9_.-]+", "-", f"{source.dataset_id}-{source.clip_id}")
    return f"{identity}.pt"


def _pool_layers(
    layers: list[torch.Tensor],
    padding_mask: torch.Tensor,
    depths: tuple[int, ...],
) -> list[torch.Tensor]:
    return [pool_layer(layers[depth - 1], padding_mask).cpu() for depth in depths]


def _extract_batch(
    model: nn.Module,
    sources: list[SourceRow],
    *,
    device: torch.device,
    depths: tuple[int, ...],
) -> list[torch.Tensor]:
    waveforms = [_read_waveform(source.audio_path) for source in sources]
    lengths = torch.tensor([waveform.numel() for waveform in waveforms])
    samples = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    padding_mask = torch.arange(samples.shape[1]).unsqueeze(0) >= lengths.unsqueeze(1)
    with torch.inference_mode():
        extracted = model.extract_features(
            samples.to(device),
            padding_mask=padding_mask.to(device),
        )
    output_mask = extracted["padding_mask"]
    if output_mask is None:
        output_mask = torch.zeros(extracted["x"].shape[:2], dtype=torch.bool, device=device)
    return _pool_layers(extracted["layer_results"], output_mask, depths)


def extract_features(
    rows: list[tuple[SourceRow, dict[str, float]]],
    cache_dir: Path,
    model: nn.Module,
    *,
    device: torch.device,
    depths: tuple[int, ...],
    batch_size: int,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, list[SourceRow]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    by_depth: dict[int, list[torch.Tensor | None]] = {depth: [None] * len(rows) for depth in depths}
    missing: list[tuple[int, SourceRow]] = []
    for index, (source, _) in enumerate(rows):
        cache_path = cache_dir / _cache_name(source)
        if cache_path.is_file():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            valid = (
                isinstance(cached, dict)
                and cached.get("audio_sha256") == source.audio_sha256
                and cached.get("feature_version") == FEATURE_VERSION
                and cached.get("teacher_checkpoint_sha256") == TEACHER_CHECKPOINT_SHA256
            )
            embeddings = cached.get("embeddings") if valid else None
            if isinstance(embeddings, dict) and all(str(depth) in embeddings for depth in depths):
                for depth in depths:
                    by_depth[depth][index] = embeddings[str(depth)].float()
                continue
        missing.append((index, source))
    missing.sort(key=lambda row: row[1].duration_ms)
    completed = len(rows) - len(missing)
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        for _, source in batch:
            if file_sha256(source.audio_path) != source.audio_sha256:
                raise ValueError(f"audio hash mismatch for {source.clip_id}")
        pooled = _extract_batch(
            model,
            [source for _, source in batch],
            device=device,
            depths=depths,
        )
        for batch_index, (row_index, source) in enumerate(batch):
            embeddings = {}
            for depth_index, depth in enumerate(depths):
                embedding = pooled[depth_index][batch_index]
                by_depth[depth][row_index] = embedding
                embeddings[str(depth)] = embedding
            torch.save(
                {
                    "audio_sha256": source.audio_sha256,
                    "feature_version": FEATURE_VERSION,
                    "teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
                    "embeddings": embeddings,
                },
                cache_dir / _cache_name(source),
            )
        completed += len(batch)
        if completed % 100 < len(batch) or completed == len(rows):
            print(f"truncated emotion2vec features: {completed}/{len(rows)}", flush=True)
    stacked = {}
    for depth, embeddings in by_depth.items():
        if any(embedding is None for embedding in embeddings):
            raise RuntimeError(f"depth {depth} embedding cache is incomplete")
        stacked[depth] = torch.stack(embeddings)  # type: ignore[arg-type]
    targets = torch.stack(
        [
            torch.tensor([distribution[label] for label in LABELS], dtype=torch.float32)
            for _, distribution in rows
        ]
    )
    return stacked, targets, [source for source, _ in rows]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponent = np.exp(np.clip(shifted, -80.0, 0.0))
    return exponent / exponent.sum(axis=-1, keepdims=True)


def _cross_entropy(probabilities: np.ndarray, targets: np.ndarray) -> float:
    return float(-(targets * np.log(np.clip(probabilities, 1e-9, 1.0))).sum(axis=-1).mean())


def _macro_f1(probabilities: np.ndarray, targets: np.ndarray) -> float:
    prediction = probabilities.argmax(axis=-1)
    truth = targets.argmax(axis=-1)
    scores = []
    for label in np.unique(truth):
        predicted = prediction == label
        expected = truth == label
        true_positive = int((predicted & expected).sum())
        false_positive = int((predicted & ~expected).sum())
        false_negative = int((~predicted & expected).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


def metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    sources: list[SourceRow],
    *,
    temperature: float,
) -> dict[str, Any]:
    probabilities = _softmax(logits / temperature)
    prediction = probabilities.argmax(axis=-1)
    truth = targets.argmax(axis=-1)
    per_class = {}
    for index, label in enumerate(LABELS):
        support = int((truth == index).sum())
        per_class[label] = {
            "support": support,
            "recall": float(((prediction == index) & (truth == index)).sum() / support)
            if support
            else None,
            "predicted": int((prediction == index).sum()),
        }
    by_dataset = {}
    for dataset_id in sorted({source.dataset_id for source in sources}):
        indices = np.array(
            [index for index, source in enumerate(sources) if source.dataset_id == dataset_id]
        )
        by_dataset[dataset_id] = {
            "clips": len(indices),
            "accuracy": float((prediction[indices] == truth[indices]).mean()),
            "macro_f1": _macro_f1(probabilities[indices], targets[indices]),
            "cross_entropy": _cross_entropy(probabilities[indices], targets[indices]),
        }
    return {
        "clips": len(truth),
        "accuracy": float((prediction == truth).mean()),
        "macro_f1": _macro_f1(probabilities, targets),
        "cross_entropy": _cross_entropy(probabilities, targets),
        "largest_prediction_share": max(value["predicted"] for value in per_class.values())
        / len(truth),
        "per_class": per_class,
        "by_dataset": by_dataset,
    }


def _temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    return float(
        min(
            np.geomspace(0.25, 10.0, 160),
            key=lambda value: _cross_entropy(_softmax(logits / value), targets),
        )
    )


def balanced_indices(
    sources: list[SourceRow],
    targets: torch.Tensor,
    *,
    sample_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, (source, label) in enumerate(zip(sources, targets.argmax(dim=-1), strict=True)):
        groups[(source.dataset_id, int(label))].append(index)
    per_group = math.ceil(sample_count / len(groups))
    sampled = []
    for indices in groups.values():
        choices = torch.randint(len(indices), (per_group,), generator=generator)
        sampled.extend(indices[int(choice)] for choice in choices)
    order = torch.tensor(sampled)
    return order[torch.randperm(len(order), generator=generator)[:sample_count]]


def berst_intensity_pairs(
    sources: list[SourceRow],
    targets: torch.Tensor,
) -> torch.Tensor:
    groups: dict[tuple[int, int], dict[bool, list[int]]] = defaultdict(
        lambda: {False: [], True: []}
    )
    for index, (source, label) in enumerate(zip(sources, targets.argmax(dim=-1), strict=True)):
        if source.dataset_id != "berst_v1" or source.pair_id < 0:
            continue
        shouting = "shouting" in (source.styles or [])
        groups[(source.pair_id, int(label))][shouting].append(index)
    pairs = []
    for styles in groups.values():
        quiet = styles[False]
        shouting = styles[True]
        for index in range(max(len(quiet), len(shouting)) if quiet and shouting else 0):
            pairs.append((quiet[index % len(quiet)], shouting[index % len(shouting)]))
    return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)


def infer(
    head: TruncatedEmotion2VecHead,
    features: torch.Tensor,
    batch_size: int = 256,
) -> np.ndarray:
    head.eval()
    logits = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            logits.append(head(features[start : start + batch_size]))
    return torch.cat(logits).numpy()


def train_head(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    train_sources: list[SourceRow],
    development_features: torch.Tensor,
    development_targets: torch.Tensor,
    development_sources: list[SourceRow],
    *,
    seed: int,
    epochs: int,
    patience: int,
    epoch_samples: int,
    learning_rate: float,
    head_kind: str,
    pair_consistency_weight: float,
) -> tuple[TruncatedEmotion2VecHead, dict[str, Any]]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    head = TruncatedEmotion2VecHead(train_features.shape[1], len(LABELS), kind=head_kind)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-3)
    intensity_pairs = berst_intensity_pairs(train_sources, train_targets)
    best: tuple[float, float, int, dict[str, torch.Tensor]] | None = None
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        order = balanced_indices(
            train_sources,
            train_targets,
            sample_count=epoch_samples,
            generator=generator,
        )
        training_loss = 0.0
        for start in range(0, len(order), 128):
            indices = order[start : start + 128]
            optimizer.zero_grad(set_to_none=True)
            logits = head(train_features[indices])
            loss = -(train_targets[indices] * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
            if pair_consistency_weight > 0 and len(intensity_pairs):
                pair_indices = torch.randint(
                    len(intensity_pairs),
                    (max(1, len(indices) // 2),),
                    generator=generator,
                )
                pairs = intensity_pairs[pair_indices]
                quiet = head(train_features[pairs[:, 0]]).softmax(dim=-1)
                shouting = head(train_features[pairs[:, 1]]).softmax(dim=-1)
                loss = loss + pair_consistency_weight * (quiet - shouting).square().mean()
            loss.backward()
            optimizer.step()
            training_loss += float(loss.detach()) * len(indices)
        development_logits = infer(head, development_features)
        development_report = metrics(
            development_logits,
            development_targets.numpy(),
            development_sources,
            temperature=1.0,
        )
        dataset_score = float(
            np.mean([values["macro_f1"] for values in development_report["by_dataset"].values()])
        )
        epoch_report = {
            "epoch": epoch,
            "training_loss": training_loss / len(order),
            "development_macro_f1": development_report["macro_f1"],
            "development_dataset_macro_f1": dataset_score,
            "development_cross_entropy": development_report["cross_entropy"],
        }
        history.append(epoch_report)
        candidate = (dataset_score, -development_report["cross_entropy"], epoch)
        if best is None or candidate[:2] > best[:2]:
            best = (
                *candidate,
                {name: value.detach().clone() for name, value in head.state_dict().items()},
            )
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best is None:
        raise RuntimeError("training produced no affect head")
    head.load_state_dict(best[3])
    return head, {
        "best_epoch": best[2],
        "berst_intensity_pairs": len(intensity_pairs),
        "pair_consistency_weight": pair_consistency_weight,
        "history": history,
    }


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, action="append", required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--external-source-manifest", type=Path, required=True)
    parser.add_argument("--external-target-manifest", type=Path, required=True)
    parser.add_argument("--teacher-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--depth", type=int, action="append", default=[])
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--epoch-samples", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--head-kind", choices=("linear", "mlp"), default="mlp")
    parser.add_argument("--berst-pair-consistency-weight", type=float, default=0.0)
    parser.add_argument("--scores-output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()

    depths = tuple(sorted(set(arguments.depth or [2, 3])))
    if not depths or min(depths) < 1 or max(depths) > 3:
        raise ValueError("probe depths must be between 1 and 3")
    if arguments.berst_pair_consistency_weight < 0:
        raise ValueError("BERSt pair consistency weight cannot be negative")
    teacher_checkpoint = arguments.teacher_path / "model.pt"
    if file_sha256(teacher_checkpoint) != TEACHER_CHECKPOINT_SHA256:
        raise ValueError("emotion2vec+ teacher checkpoint hash mismatch")
    from funasr import AutoModel

    wrapper = AutoModel(model=str(arguments.teacher_path), disable_update=True)
    teacher = wrapper.model.eval()
    teacher.blocks = nn.ModuleList(list(teacher.blocks)[: max(depths)])
    device = choose_device(arguments.device)
    teacher.to(device)
    frontend = teacher.modality_encoders["AUDIO"]
    frontend_parameters = sum(
        parameter.numel()
        for name, parameter in frontend.named_parameters()
        if not name.startswith("decoder.")
    )
    block_parameters = [
        sum(parameter.numel() for parameter in block.parameters()) for block in teacher.blocks
    ]

    training_rows = matched_rows(
        arguments.source_manifest, arguments.target_manifest, split="train"
    )
    development_rows = matched_rows(
        arguments.source_manifest,
        arguments.target_manifest,
        split="development",
    )
    external_rows = matched_rows(
        [arguments.external_source_manifest],
        arguments.external_target_manifest,
        split="sealed_test",
    )
    train_x, train_y, train_sources = extract_features(
        training_rows,
        arguments.cache_dir,
        teacher,
        device=device,
        depths=depths,
        batch_size=arguments.feature_batch_size,
    )
    development_x, development_y, development_sources = extract_features(
        development_rows,
        arguments.cache_dir,
        teacher,
        device=device,
        depths=depths,
        batch_size=arguments.feature_batch_size,
    )
    external_x, external_y, external_sources = extract_features(
        external_rows,
        arguments.cache_dir,
        teacher,
        device=device,
        depths=depths,
        batch_size=arguments.feature_batch_size,
    )
    teacher.cpu()
    del teacher, wrapper

    candidates = {}
    checkpoints = {}
    for depth in depths:
        mean = train_x[depth].mean(dim=0)
        scale = train_x[depth].std(dim=0).clamp_min(1e-5)
        normalized_train = (train_x[depth] - mean) / scale
        normalized_development = (development_x[depth] - mean) / scale
        normalized_external = (external_x[depth] - mean) / scale
        head, training_report = train_head(
            normalized_train,
            train_y,
            train_sources,
            normalized_development,
            development_y,
            development_sources,
            seed=arguments.seed,
            epochs=arguments.epochs,
            patience=arguments.patience,
            epoch_samples=arguments.epoch_samples,
            learning_rate=arguments.learning_rate,
            head_kind=arguments.head_kind,
            pair_consistency_weight=arguments.berst_pair_consistency_weight,
        )
        development_logits = infer(head, normalized_development)
        temperature = _temperature(development_logits, development_y.numpy())
        development_report = metrics(
            development_logits,
            development_y.numpy(),
            development_sources,
            temperature=temperature,
        )
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
                    "expected_logit": logits[expected],
                    "contrast_logit": logits[contrast],
                    "passed": logits[expected] > logits[contrast],
                }
            )
        head_parameters = sum(parameter.numel() for parameter in head.parameters())
        truncated_parameters = frontend_parameters + sum(block_parameters[:depth]) + head_parameters
        candidates[str(depth)] = {
            "depth": depth,
            "truncated_emotion2vec_parameters": truncated_parameters,
            "head_parameters": head_parameters,
            "total_with_cadence_parameters": CADENCE_PARAMETERS + truncated_parameters,
            "temperature": temperature,
            "training": training_report,
            "development": development_report,
            "ravdess": external_report,
            "transition": transition,
            "transition_passed": len(transition) == 2 and all(row["passed"] for row in transition),
        }
        checkpoints[depth] = {
            "state_dict": head.state_dict(),
            "feature_mean": mean,
            "feature_scale": scale,
            "temperature": temperature,
        }
        if arguments.scores_output_dir is not None:
            arguments.scores_output_dir.mkdir(parents=True, exist_ok=True)
            for split, sources, targets, logits in (
                ("development", development_sources, development_y, development_logits),
                ("external", external_sources, external_y, external_logits),
            ):
                score_path = arguments.scores_output_dir / f"depth-{depth}-{split}.jsonl"
                with score_path.open("w") as score_file:
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
    selected_depth = max(
        depths,
        key=lambda depth: np.mean(
            [
                values["macro_f1"]
                for values in candidates[str(depth)]["development"]["by_dataset"].values()
            ]
        ),
    )
    source_manifests = [
        {"path": str(path), "sha256": manifest_sha256(path)} for path in arguments.source_manifest
    ]
    selected = candidates[str(selected_depth)]
    checkpoint = {
        "schema_version": "1.0",
        "model": "truncated_emotion2vec_affect_student",
        "feature_version": FEATURE_VERSION,
        "teacher_revision": TEACHER_REVISION,
        "teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
        "depth": selected_depth,
        "head_kind": arguments.head_kind,
        "labels": LABELS,
        "frontend_parameters": frontend_parameters,
        "block_parameters": block_parameters[:selected_depth],
        "head_parameters": selected["head_parameters"],
        "truncated_emotion2vec_parameters": selected["truncated_emotion2vec_parameters"],
        "source_manifests": source_manifests,
        "target_manifest_sha256": manifest_sha256(arguments.target_manifest),
        **checkpoints[selected_depth],
    }
    arguments.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, arguments.checkpoint_output)
    report = {
        "schema_version": "1.0",
        "model": "truncated_emotion2vec_affect_student",
        "feature_version": FEATURE_VERSION,
        "teacher_revision": TEACHER_REVISION,
        "teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
        "device": str(device),
        "head_kind": arguments.head_kind,
        "frontend_parameters": frontend_parameters,
        "block_parameters": block_parameters,
        "selected_depth": selected_depth,
        "candidates": candidates,
        "inputs": {
            "source_manifests": source_manifests,
            "target_manifest": str(arguments.target_manifest),
            "target_manifest_sha256": manifest_sha256(arguments.target_manifest),
            "external_source_manifest": str(arguments.external_source_manifest),
            "external_source_manifest_sha256": manifest_sha256(arguments.external_source_manifest),
            "external_target_manifest": str(arguments.external_target_manifest),
            "external_target_manifest_sha256": manifest_sha256(arguments.external_target_manifest),
        },
    }
    arguments.report_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.report_output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
