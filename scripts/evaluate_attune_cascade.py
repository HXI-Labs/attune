#!/usr/bin/env python3
"""Run one combined inspection of the concrete Attune cascade."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from attune.baselines.adapters import BaselineInput, TranscriptSentimentAdapter
from attune.baselines.cascade import AttuneCascade
from attune.evaluation.metrics import corpus_character_error_rate, corpus_word_error_rate
from attune.inference.packaging import package_for_trusted_channel
from attune.schema.output import AttuneOutput
from attune.schema.xml import render_xml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-manifest",
        type=Path,
        default=Path("data/manifests/inspection-set.jsonl"),
    )
    parser.add_argument(
        "--original-cache",
        type=Path,
        default=Path("data/raw/inspection-set"),
    )
    parser.add_argument(
        "--expansion-manifest",
        type=Path,
        default=Path("data/manifests/licence-clean-inspection.jsonl"),
    )
    parser.add_argument(
        "--expansion-cache",
        type=Path,
        default=Path("data/raw/licence-clean-inspection"),
    )
    parser.add_argument("--sensevoice-path", type=Path, required=True)
    parser.add_argument("--emotion2vec-path", type=Path, required=True)
    parser.add_argument("--vocalsound-probe-checkpoint", type=Path, required=True)
    parser.add_argument("--fsd50k-probe-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("artifacts/cascade-sensevoice-embeddings"),
    )
    parser.add_argument(
        "--vocalsound-training-report",
        type=Path,
        default=Path("artifacts/event-probe/metrics.json"),
    )
    parser.add_argument(
        "--fsd50k-training-report",
        type=Path,
        default=Path("artifacts/fsd50k-event-probe/metrics.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/attune-cascade-inspection-results.json"),
    )
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_slice(name: str, manifest: Path, cache: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"inspection manifest is empty: {manifest}")
    for row in rows:
        audio_path = cache / row["cache_path"]
        if not audio_path.is_file():
            raise RuntimeError(f"inspection audio is missing: {audio_path}")
        if digest(audio_path) != row["sha256"]:
            raise RuntimeError(f"inspection audio hash mismatch: {audio_path}")
        row["_inspection_slice"] = name
        row["_audio_path"] = audio_path
    return rows


def reference_transcript(row: dict[str, Any]) -> str | None:
    value = row.get("transcript")
    if value is None:
        value = row.get("source_metadata", {}).get("transcript")
    return str(value) if value is not None else None


def normalize_asr(text: str) -> str:
    return " ".join(
        "".join(character if character.isalnum() else " " for character in text.lower()).split()
    )


def categorical_metrics(
    references: list[str],
    predictions: list[str],
) -> dict[str, Any]:
    labels = sorted(set(references))
    per_class: dict[str, Any] = {}
    for label in labels:
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
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": references.count(label),
        }
    return {
        "evaluated_clips": len(references),
        "accuracy": sum(
            reference == prediction
            for reference, prediction in zip(references, predictions, strict=True)
        )
        / len(references),
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(per_class),
        "per_class": per_class,
        "prediction_counts": dict(sorted(Counter(predictions).items())),
    }


def expected_annotations(row: dict[str, Any]) -> set[str]:
    labels = row["intended_attune_labels"]
    expected = {f"event:{label}" for label in labels.get("events", [])}
    expected.update(f"style:{label}" for label in labels.get("styles", []))
    if labels.get("source_class") == "Screaming":
        expected.add("event:scream")
    return expected


def annotation_metrics(
    records: list[dict[str, Any]],
    prediction_key: str,
) -> dict[str, Any]:
    target_labels = sorted(
        {label for record in records for label in record["expected_annotations"]}
    )
    totals = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    per_class: dict[str, Any] = {}
    exact = 0
    for label in target_labels:
        true_positive = sum(
            label in record["expected_annotations"] and label in record[prediction_key]
            for record in records
        )
        false_positive = sum(
            label not in record["expected_annotations"] and label in record[prediction_key]
            for record in records
        )
        false_negative = sum(
            label in record["expected_annotations"] and label not in record[prediction_key]
            for record in records
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
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": sum(label in record["expected_annotations"] for record in records),
        }
    for record in records:
        expected = record["expected_annotations"]
        predicted = record[prediction_key]
        totals["true_positive"] += len(expected & predicted)
        totals["false_positive"] += len(predicted - expected)
        totals["false_negative"] += len(expected - predicted)
        exact += expected == predicted
    denominator = 2 * totals["true_positive"] + totals["false_positive"] + totals["false_negative"]
    return {
        "evaluated_clips": len(records),
        "target_labels": target_labels,
        "target_macro_f1": sum(row["f1"] for row in per_class.values()) / len(per_class),
        "micro_f1_all_predictions": (
            2 * totals["true_positive"] / denominator if denominator else 0.0
        ),
        "exact_set_match": exact / len(records),
        "counts": totals,
        "per_class": per_class,
        "span_policy": "whole-utterance weak labels and predictions; no localization score",
    }


def component_set(components: list[dict[str, Any]], name_fragment: str) -> set[str]:
    return {
        f"{annotation['channel']}:{annotation['label']}"
        for component in components
        if name_fragment in component["name"]
        for annotation in component["annotations"]
    }


def run_row(cascade: AttuneCascade, row: dict[str, Any]) -> dict[str, Any]:
    prediction = cascade.predict(
        BaselineInput(
            audio_path=row["_audio_path"],
            transcript_hint=reference_transcript(row),
            language_hint="en",
        )
    )
    output = AttuneOutput.model_validate(prediction.output.model_dump(mode="json"))
    diagnostics = prediction.diagnostics or {}
    components = diagnostics["event_style_components"]
    aed = component_set(components, "-aed")
    vocalsound = component_set(components, "vocalsound-frozen")
    fsd50k = component_set(components, "fsd50k-frozen")
    cascade_annotations = {
        *(f"event:{event.label.value}" for event in output.events),
        *(f"style:{style.label.value}" for style in output.styles),
    }
    lexical_label = TranscriptSentimentAdapter().classify(output.transcript.text)[0].value
    xml = render_xml(output)
    package = package_for_trusted_channel(output)
    utterance_timestamps = (
        not output.transcript.words
        and all(
            span.start_ms == 0 and span.end_ms == output.audio.duration_ms
            for span in [*output.events, *output.styles]
        )
        and output.affect.start_ms == 0
        and output.affect.end_ms == output.audio.duration_ms
    )
    structured_channels = (
        "transcript" not in package["paralinguistic_metadata"]
        and package["spoken_transcript"]["text"] == output.transcript.text
    )
    return {
        "inspection_slice": row["_inspection_slice"],
        "clip_id": row["clip_id"],
        "source_dataset": row["source_dataset"],
        "reference_transcript": reference_transcript(row),
        "transcript": output.transcript.text,
        "reference_affect": row["intended_attune_labels"].get("affect", []),
        "emotion2vec_affect": output.affect.top_label.value,
        "lexicon_affect": lexical_label,
        "expected_annotations": expected_annotations(row),
        "aed_annotations": aed,
        "vocalsound_probe_annotations": vocalsound,
        "fsd50k_probe_annotations": fsd50k,
        "probe_only_annotations": vocalsound | fsd50k,
        "cascade_annotations": cascade_annotations,
        "schema_valid": True,
        "xml_deterministic": xml == render_xml(output),
        "utterance_timestamps_only": utterance_timestamps,
        "structured_channels_separate": structured_channels,
        "elapsed_seconds": prediction.runtime.elapsed_seconds,
        "audio_seconds": prediction.runtime.audio_seconds,
    }


def summarize_slice(records: list[dict[str, Any]]) -> dict[str, Any]:
    transcript_records = [record for record in records if record["reference_transcript"]]
    affect_records = [record for record in records if record["reference_affect"]]
    event_records = [record for record in records if record["expected_annotations"]]
    references = [normalize_asr(record["reference_transcript"]) for record in transcript_records]
    hypotheses = [normalize_asr(record["transcript"]) for record in transcript_records]
    affect_references = [record["reference_affect"][0] for record in affect_records]
    return {
        "scope": {
            "clip_count": len(records),
            "source_counts": dict(
                sorted(Counter(row["source_dataset"] for row in records).items())
            ),
            "transcript_clips": len(transcript_records),
            "affect_clips": len(affect_records),
            "event_style_clips": len(event_records),
        },
        "asr": {
            "evaluated_clips": len(transcript_records),
            "wer": corpus_word_error_rate(references, hypotheses),
            "cer": corpus_character_error_rate(references, hypotheses),
            "normalization": "lowercase alphanumeric tokens; punctuation removed",
        },
        "affect": {
            "emotion2vec_plus": categorical_metrics(
                affect_references,
                [record["emotion2vec_affect"] for record in affect_records],
            ),
            "transcript_lexicon_ablation": categorical_metrics(
                affect_references,
                [record["lexicon_affect"] for record in affect_records],
            ),
        },
        "events_styles": {
            "cascade": annotation_metrics(event_records, "cascade_annotations"),
            "aed_only": annotation_metrics(event_records, "aed_annotations"),
            "probe_only": annotation_metrics(event_records, "probe_only_annotations"),
            "vocalsound_probe_only": annotation_metrics(
                event_records, "vocalsound_probe_annotations"
            ),
            "fsd50k_probe_only": annotation_metrics(event_records, "fsd50k_probe_annotations"),
        },
    }


def json_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: sorted(value) if isinstance(value, set) else value for key, value in record.items()
    }


def load_training_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"training report is missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    inspection_metrics = report.get("inspection_test_metrics") or report.get("test_metrics")
    if inspection_metrics is None:
        raise RuntimeError(f"training report has no inspection metrics: {path}")
    return {
        "path": str(path),
        "encoder_frozen": report["encoder_frozen"],
        "head": report["head"],
        "embedding": report["embedding"],
        "validation_metrics": report["validation_metrics"],
        "inspection_test_metrics": inspection_metrics,
        "partitions": report["partitions"],
    }


def main() -> None:
    arguments = parse_args()
    os.environ.update(
        {
            "ATTUNE_SENSEVOICE_LICENSE_REVIEWED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "MODELSCOPE_OFFLINE": "1",
            "FUNASR_DISABLE_UPDATE": "1",
        }
    )
    original = load_slice(
        "original_150",
        arguments.original_manifest,
        arguments.original_cache,
    )
    expansion = load_slice(
        "licence_clean_expansion_160",
        arguments.expansion_manifest,
        arguments.expansion_cache,
    )
    cascade = AttuneCascade(
        sensevoice_checkpoint=arguments.sensevoice_path,
        emotion2vec_checkpoint=arguments.emotion2vec_path,
        vocalsound_probe_checkpoint=arguments.vocalsound_probe_checkpoint,
        fsd50k_probe_checkpoint=arguments.fsd50k_probe_checkpoint,
        embedding_cache=arguments.embedding_cache,
    )
    available, reason = cascade.availability()
    if not available:
        raise RuntimeError(reason)

    started = time.time()
    records = []
    for index, row in enumerate([*original, *expansion], 1):
        records.append(run_row(cascade, row))
        print(f"Attune cascade {index}/310 {row['clip_id']}", flush=True)

    payload = {
        "report_version": "1",
        "title": "Attune cascade combined 310-clip inspection",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "gate_decision": "closed",
        "scope": {
            "clip_count": len(records),
            "original_manifest": str(arguments.original_manifest),
            "expansion_manifest": str(arguments.expansion_manifest),
            "label_status": "weak source labels / acted labels; not reviewed gold",
            "fine_tuning_performed": False,
            "encoder_frozen": True,
            "full_fsd50k_archive_downloaded": False,
        },
        "cascade": {
            "name": cascade.name,
            "transcript": "SenseVoiceSmall",
            "affect": "emotion2vec+ acoustic SER; transcript lexicon excluded",
            "events_styles": (
                "SenseVoice AED UNION frozen VocalSound linear probe "
                "UNION frozen FSD50K linear probe"
            ),
            "merge_rule": (
                "Deterministic set union by channel and label in AED, VocalSound, "
                "FSD50K order; later duplicates are suppressed. Scream remains a "
                "discrete event, not shouting. Sob remains an event, not crying_speech."
            ),
            "timestamps": "whole utterance only; no word or frame localization",
            "probe_confidence": "uncalibrated closed-set diagnostic, not gold",
        },
        "slices": {
            "original_150": summarize_slice(
                [record for record in records if record["inspection_slice"] == "original_150"]
            ),
            "licence_clean_expansion_160": summarize_slice(
                [
                    record
                    for record in records
                    if record["inspection_slice"] == "licence_clean_expansion_160"
                ]
            ),
        },
        "contract_audit": {
            "schema_valid": sum(record["schema_valid"] for record in records),
            "deterministic_xml": sum(record["xml_deterministic"] for record in records),
            "utterance_timestamps_only": sum(
                record["utterance_timestamps_only"] for record in records
            ),
            "structured_transcript_metadata_separation": sum(
                record["structured_channels_separate"] for record in records
            ),
            "total": len(records),
            "json_authoritative": True,
        },
        "training": {
            "vocalsound": load_training_report(arguments.vocalsound_training_report),
            "fsd50k": load_training_report(arguments.fsd50k_training_report),
            "checkpoint_sha256": {
                "vocalsound": digest(arguments.vocalsound_probe_checkpoint),
                "fsd50k": digest(arguments.fsd50k_probe_checkpoint),
            },
            "checkpoints_committed": False,
        },
        "execution": {
            "run_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "wall_seconds": time.time() - started,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "offline_during_inference": True,
        },
        "attribution": {
            "FunASR": "FunASR runtime project.",
            "SenseVoiceSmall": (
                "FunASR/FunAudioLLM; FunASR Model Open Source License Agreement v1.1."
            ),
            "emotion2vec+": (
                "emotion2vec and FunASR/FunAudioLLM; FunASR Model Open Source License Agreement."
            ),
            "Whisper": (
                "OpenAI Whisper; MIT. Retained as an approved standalone comparator; "
                "not executed in this SenseVoice-transcript cascade run."
            ),
            "VocalSound": ("Gong, Yu, and Glass, ICASSP 2022; CC BY-SA 4.0."),
            "FSD50K": (
                "Fonseca et al.; annotations CC BY 4.0; selected clips retain "
                "their recorded CC0 or CC BY licences."
            ),
            "CREMA-D": ("Cao et al., 2014; ODbL 1.0 database / DbCL 1.0 contents."),
        },
        "limitations": [
            "The research gate remains closed.",
            "Both probes are closed-set and have no evaluated OOD or abstention threshold.",
            "All event/style spans are whole-utterance provisional spans.",
            "Weak source labels are not reviewed Attune gold.",
            "FSD50K standalone events do not establish speech-embedded style coverage.",
            "CREMA-D DIS maps to Attune other, never distress.",
        ],
        "predictions": [json_record(record) for record in records],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
