#!/usr/bin/env python3
"""Evaluate reviewed baselines on the licence-clean event and affect expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prepare_licence_clean_inspection import load_manifest, verify

from attune.baselines.adapters import (
    BaselineAdapter,
    BaselineInput,
    Emotion2VecPlusAdapter,
    SenseVoiceSmallAdapter,
    WhisperSmallAdapter,
)
from attune.baselines.cascade import ModularCascade
from attune.evaluation.metrics import corpus_character_error_rate, corpus_word_error_rate

MODEL_METADATA = {
    "whisper-small": {
        "revision": "973afd24965f72e36ca33b3055d56a652f456b4d",
        "attribution": "OpenAI Whisper-Small; upstream Whisper MIT licence.",
    },
    "sensevoice-small": {
        "revision": "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7",
        "attribution": (
            "SenseVoiceSmall by FunASR/FunAudioLLM; "
            "FunASR Model Open Source License Agreement v1.1."
        ),
    },
    "emotion2vec-plus": {
        "revision": "b318240bfe67db81a8c572ecb37ce9c3759b81c9",
        "attribution": (
            "emotion2vec+ base by emotion2vec and FunASR/FunAudioLLM; "
            "FunASR Model Open Source License Agreement."
        ),
    },
}
AFFECT_LABELS = ("anger", "fear", "other")


def normalize_asr(text: str) -> str:
    cleaned = "".join(
        character if character.isalnum() else " " for character in text.lower()
    )
    return " ".join(cleaned.split())


def checkpoint_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".bin", ".pt", ".safetensors"}:
            value = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    value.update(chunk)
            hashes[str(path.relative_to(root))] = value.hexdigest()
    return hashes


def categorical_metrics(references: list[str], predictions: list[str]) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for label in AFFECT_LABELS:
        true_positive = sum(
            reference == prediction == label
            for reference, prediction in zip(references, predictions, strict=True)
        )
        false_positive = sum(
            reference != label and prediction == label
            for reference, prediction in zip(references, predictions, strict=True)
        )
        false_negative = sum(
            reference == label and prediction != label
            for reference, prediction in zip(references, predictions, strict=True)
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {
            "support": references.count(label),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "accuracy": sum(
            reference == prediction
            for reference, prediction in zip(references, predictions, strict=True)
        )
        / len(references),
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
        "prediction_counts": dict(sorted(Counter(predictions).items())),
    }


def evaluate_crema(
    runner: BaselineAdapter,
    rows: list[dict[str, Any]],
    cache_root: Path,
    *,
    score_asr: bool,
) -> dict[str, Any]:
    references: list[str] = []
    hypotheses: list[str] = []
    affect_references: list[str] = []
    affect_predictions: list[str] = []
    failures: list[dict[str, str]] = []
    raw_affect_counts: Counter[str] = Counter()
    affect_source_counts: Counter[str] = Counter()
    mapping_examples: list[dict[str, Any]] = []
    examples_by_source: Counter[str] = Counter()
    elapsed = 0.0
    audio_seconds = 0.0
    for index, row in enumerate(rows, 1):
        try:
            prediction = runner.predict(
                BaselineInput(
                    audio_path=cache_root / row["cache_path"],
                    transcript_hint=row["transcript"],
                    language_hint="en",
                )
            )
        except Exception as error:
            failures.append(
                {
                    "clip_id": row["clip_id"],
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
            )
            continue
        references.append(normalize_asr(row["transcript"]))
        hypotheses.append(normalize_asr(prediction.output.transcript.text))
        affect_references.append(row["intended_attune_labels"]["affect"][0])
        affect_predictions.append(prediction.output.affect.top_label.value)
        diagnostics = prediction.diagnostics or {}
        raw_label = diagnostics.get("raw_affect_label")
        raw_affect_counts[str(raw_label) if raw_label is not None else "<not_emitted>"] += 1
        affect_source_counts[str(diagnostics.get("affect_source", "unreported"))] += 1
        source_emotion = row["intended_attune_labels"]["source_emotion"]
        if examples_by_source[source_emotion] < 2:
            mapping_examples.append(
                {
                    "clip_id": row["clip_id"],
                    "source_emotion": source_emotion,
                    "reference_schema_label": row["intended_attune_labels"]["affect"][0],
                    "raw_affect_label": raw_label,
                    "mapped_schema_label": prediction.output.affect.top_label.value,
                    "diagnostics": diagnostics,
                }
            )
            examples_by_source[source_emotion] += 1
        elapsed += prediction.runtime.elapsed_seconds
        audio_seconds += prediction.runtime.audio_seconds
        print(f"{runner.name} CREMA-D {index}/{len(rows)}", flush=True)
    return {
        "status": "completed" if not failures else ("partial_failure" if references else "failed"),
        "evaluated_clips": len(references),
        "failed_clips": len(failures),
        "asr": (
            {
                "wer": corpus_word_error_rate(references, hypotheses) if references else None,
                "cer": corpus_character_error_rate(references, hypotheses) if references else None,
                "normalization": "lowercase alphanumeric tokens; punctuation removed",
            }
            if score_asr
            else {
                "wer": None,
                "cer": None,
                "note": "Not applicable: this runner is an acoustic affect-only model.",
            }
        ),
        "categorical_affect": (
            categorical_metrics(affect_references, affect_predictions)
            if affect_references
            else None
        ),
        "runtime": {
            "audio_seconds": audio_seconds,
            "elapsed_seconds": elapsed,
            "real_time_factor": elapsed / audio_seconds if audio_seconds else None,
            "device": "cpu",
        },
        "affect_wiring_diagnostics": {
            "affect_source_counts": dict(sorted(affect_source_counts.items())),
            "raw_label_counts": dict(sorted(raw_affect_counts.items())),
            "schema_label_counts": dict(sorted(Counter(affect_predictions).items())),
            "raw_to_schema_examples": mapping_examples,
            "disgust_mapping": (
                "CREMA-D DIS remains Attune `other`. It is not remapped to distress; "
                "a runner that never emits `other` has structurally zero DIS/other recall."
            ),
        },
        "failures": failures,
    }


def _predicted_annotations(output: Any) -> tuple[set[str], set[str]]:
    events = {event.label.value for event in output.events}
    styles = {style.label.value for style in output.styles}
    return events, styles


def evaluate_sensevoice_aed(
    runner: SenseVoiceSmallAdapter,
    rows: list[dict[str, Any]],
    cache_root: Path,
) -> dict[str, Any]:
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"clips": 0, "matched": 0})
    outputs: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, row in enumerate(rows, 1):
        source_class = row["intended_attune_labels"]["source_class"]
        by_class[source_class]["clips"] += 1
        try:
            prediction = runner.predict(
                BaselineInput(audio_path=cache_root / row["cache_path"], language_hint="en")
            )
        except Exception as error:
            failures.append(
                {
                    "clip_id": row["clip_id"],
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
            )
            continue
        events, styles = _predicted_annotations(prediction.output)
        expected_events = set(row["intended_attune_labels"]["events"])
        expected_styles = set(row["intended_attune_labels"]["styles"])
        matched = bool((events & expected_events) or (styles & expected_styles))
        by_class[source_class]["matched"] += int(matched)
        outputs.append(
            {
                "clip_id": row["clip_id"],
                "source_class": source_class,
                "events": sorted(events),
                "styles": sorted(styles),
                "matched_mapped_weak_label": matched,
            }
        )
        print(f"sensevoice AED {index}/{len(rows)}", flush=True)
    return {
        "status": "completed" if not failures else ("partial_failure" if outputs else "failed"),
        "evaluated_clips": len(outputs),
        "failed_clips": len(failures),
        "by_source_class": {
            label: {
                **counts,
                "weak_label_detection_rate": counts["matched"] / counts["clips"],
            }
            for label, counts in sorted(by_class.items())
        },
        "interpretation": (
            "Crying_and_sobbing is scored only as sob; Shout only as shouting; "
            "Whispering only as whispering. Screaming is reported separately without "
            "an Attune shouting target."
        ),
        "outputs": outputs,
        "failures": failures,
    }


def evaluate_frozen_probe(
    rows: list[dict[str, Any]],
    cache_root: Path,
    checkpoint: Path | None,
    sensevoice_model: Path,
    embedding_cache: Path,
) -> dict[str, Any]:
    """Run an existing frozen probe head as OOD diagnostics; never train a head."""
    if checkpoint is None or not checkpoint.is_file():
        return {
            "status": "blocked_missing_existing_checkpoint",
            "reason": (
                "The trained frozen event-probe head is intentionally gitignored and was not "
                "available on this VM. Retraining is forbidden for this run."
            ),
            "encoder_frozen": True,
            "training_performed": False,
        }
    try:
        import torch

        from attune.models.sensevoice_probe import (
            SENSEVOICE_EMBEDDING,
            FrozenSenseVoiceEncoder,
        )
    except ImportError as error:
        return {
            "status": "blocked_missing_runtime",
            "reason": str(error),
            "encoder_frozen": True,
            "training_performed": False,
        }
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("embedding") != SENSEVOICE_EMBEDDING:
        return {
            "status": "blocked_wrong_checkpoint",
            "reason": "Checkpoint is not the existing frozen SenseVoice encoder probe.",
            "encoder_frozen": True,
            "training_performed": False,
        }
    extractor = FrozenSenseVoiceEncoder(sensevoice_model, embedding_cache, torch)
    labels = payload["labels"]
    head = torch.nn.Linear(payload["feature_mean"].numel(), len(labels))
    head.load_state_dict(payload["head_state_dict"])
    head.eval()
    outputs = []
    with torch.inference_mode():
        for row in rows:
            features = extractor(cache_root / row["cache_path"])
            normalized = (features - payload["feature_mean"]) / payload["feature_scale"]
            probabilities = torch.softmax(head(normalized.unsqueeze(0)), dim=1)[0]
            index = int(probabilities.argmax())
            outputs.append(
                {
                    "clip_id": row["clip_id"],
                    "source_class": row["intended_attune_labels"]["source_class"],
                    "predicted_vocalsound_class": labels[index],
                    "confidence": float(probabilities[index]),
                }
            )
    return {
        "status": "completed_ood_diagnostic",
        "encoder_frozen": True,
        "training_performed": False,
        "known_probe_classes": labels,
        "warning": (
            "The probe was trained for five VocalSound classes, none of which are "
            "Shout, Whispering, Screaming, or Crying_and_sobbing; outputs are OOD diagnostics."
        ),
        "outputs": outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/licence-clean-inspection.jsonl"),
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/raw/licence-clean-inspection")
    )
    parser.add_argument("--whisper-path", type=Path, required=True)
    parser.add_argument("--sensevoice-path", type=Path, required=True)
    parser.add_argument("--emotion2vec-path", type=Path, required=True)
    parser.add_argument("--probe-checkpoint", type=Path)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("artifacts/licence-clean-sensevoice-embeddings"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/licence-clean-inspection-results.json"),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    rows = load_manifest(arguments.manifest)
    verify(rows, arguments.cache_dir)
    event_rows = [row for row in rows if row["source_dataset"] == "FSD50K"]
    crema_rows = [row for row in rows if row["source_dataset"] == "CREMA-D"]
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "MODELSCOPE_OFFLINE": "1",
            "FUNASR_DISABLE_UPDATE": "1",
            "ATTUNE_SENSEVOICE_LICENSE_REVIEWED": "1",
        }
    )
    started = time.time()
    whisper = WhisperSmallAdapter(checkpoint=arguments.whisper_path)
    sensevoice = SenseVoiceSmallAdapter(checkpoint=arguments.sensevoice_path)
    emotion2vec = Emotion2VecPlusAdapter(checkpoint=arguments.emotion2vec_path)
    runners = (
        (whisper, True),
        (sensevoice, True),
        (emotion2vec, False),
        (ModularCascade(asr=whisper, affect=emotion2vec), True),
        (ModularCascade(asr=sensevoice, affect=emotion2vec), True),
    )
    crema_results = {
        runner.name: evaluate_crema(
            runner, crema_rows, arguments.cache_dir, score_asr=score_asr
        )
        for runner, score_asr in runners
    }
    payload = {
        "report_version": "1",
        "title": "Licence-clean FSD50K event and CREMA-D affect inspection",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": {
            "manifest": str(arguments.manifest),
            "clip_count": len(rows),
            "source_counts": dict(sorted(Counter(row["source_dataset"] for row in rows).items())),
            "sample_rate_hz": 16_000,
            "channels": 1,
            "fine_tuning_performed": False,
            "gate_decision": "closed",
        },
        "execution": {
            "run_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "wall_seconds": time.time() - started,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "offline_after_fetch": True,
        },
        "checkpoint_hashes": {
            "whisper-small": checkpoint_hashes(arguments.whisper_path),
            "sensevoice-small": checkpoint_hashes(arguments.sensevoice_path),
            "emotion2vec-plus": checkpoint_hashes(arguments.emotion2vec_path),
        },
        "crema_d_expansion": crema_results,
        "sensevoice_aed": evaluate_sensevoice_aed(
            sensevoice, event_rows, arguments.cache_dir
        ),
        "frozen_event_probe": evaluate_frozen_probe(
            event_rows,
            arguments.cache_dir,
            arguments.probe_checkpoint,
            arguments.sensevoice_path,
            arguments.embedding_cache,
        ),
        "model_metadata": MODEL_METADATA,
        "limitations": [
            "All dataset targets are source-labelled weak labels, not reviewed gold.",
            "FSD50K Shout clips are standalone Freesound events, not speech-embedded shouting.",
            "Crying_and_sobbing maps to sob, not crying_speech without clip-level speech review.",
            "CREMA-D intensity is metadata only and is not mapped to shouting or whispering.",
            "CREMA-D contains no surprise source category.",
            (
                "CREMA-D DIS maps to Attune other, never distress. Models without an "
                "other/disgust output therefore have structurally zero DIS recall."
            ),
            "No calibration, localization, abstention, OOD threshold, or gold review is completed.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote results to {arguments.output}")


if __name__ == "__main__":
    main()
