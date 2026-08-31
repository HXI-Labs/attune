#!/usr/bin/env python3
"""Evaluate a calibrated event probe from an exported Cadence ONNX graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from attune.inference.export import PROBE_EMBEDDING_OUTPUT
from attune.inference.onnx_backend import LocalSenseVoiceFrontend
from attune.inference.probe_head import LinearProbeHead
from attune.integrity import file_digest


def load_rows(manifest: Path, root: Path, source: str | None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if source is not None:
        rows = [row for row in rows if row.get("source_dataset") == source]
    for row in rows:
        row["audio_path"] = root / row["cache_path"]
        if not row["audio_path"].is_file():
            raise ValueError(f"missing audio for {row.get('clip_id')}: {row['audio_path']}")
    return rows


def infer_decisions(
    rows: list[dict[str, Any]],
    session: ort.InferenceSession,
    frontend: LocalSenseVoiceFrontend,
    probe: LinearProbeHead,
) -> list[dict[str, Any]]:
    decisions = []
    for index, row in enumerate(rows, start=1):
        features = frontend(row["audio_path"].read_bytes())
        embedding = session.run(
            [PROBE_EMBEDDING_OUTPUT],
            {
                "speech": features[np.newaxis].astype(np.float32),
                "speech_lengths": np.asarray([features.shape[0]], dtype=np.int64),
            },
        )[0][0]
        decision = probe.predict(embedding)
        decisions.append(
            {
                "clip_id": row["clip_id"],
                "expected": row.get("expected"),
                "predicted": None if decision.abstained else decision.label,
                "abstained": decision.abstained,
                "confidence": decision.confidence,
                "score": decision.score,
            }
        )
        if index % 50 == 0 or index == len(rows):
            print(f"evaluated {index}/{len(rows)}", flush=True)
    return decisions


def metrics(
    positive: list[dict[str, Any]], ood: list[dict[str, Any]], labels: tuple[str, ...]
) -> dict[str, Any]:
    per_class = {}
    for label in labels:
        true_positive = sum(
            row["expected"] == label and row["predicted"] == label for row in positive
        )
        false_positive = sum(
            row["expected"] != label and row["predicted"] == label for row in positive
        )
        false_negative = sum(
            row["expected"] == label and row["predicted"] != label for row in positive
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
            "support": sum(row["expected"] == label for row in positive),
        }
    ood_false_positive = sum(row["predicted"] is not None for row in ood)
    macro_f1 = sum(values["f1"] for values in per_class.values()) / len(per_class)
    return {
        "macro_f1": macro_f1,
        "per_class": per_class,
        "ood": {
            "clips": len(ood),
            "false_positive": ood_false_positive,
            "false_positive_rate": ood_false_positive / len(ood),
        },
        "gate": {
            "macro_f1_minimum": 0.80,
            "ood_false_positive_rate_maximum": 0.05,
            "passed": macro_f1 >= 0.80 and ood_false_positive / len(ood) <= 0.05,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sensevoice-path", type=Path, required=True)
    parser.add_argument("--probe-head", type=Path, required=True)
    parser.add_argument("--positive-manifest", type=Path, required=True)
    parser.add_argument("--positive-root", type=Path, required=True)
    parser.add_argument("--positive-source", default="VocalSound")
    parser.add_argument("--ood-manifest", type=Path, required=True)
    parser.add_argument("--ood-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    positive_rows = load_rows(
        arguments.positive_manifest,
        arguments.positive_root,
        arguments.positive_source,
    )
    for row in positive_rows:
        expected = row.get("intended_attune_labels", {}).get("events", [])
        if len(expected) != 1:
            raise ValueError(f"positive row lacks one event label: {row['clip_id']}")
        row["expected"] = expected[0]
    ood_rows = load_rows(arguments.ood_manifest, arguments.ood_root, None)
    if not positive_rows or not ood_rows:
        raise ValueError("positive and OOD evaluation rows must both be non-empty")

    frontend = LocalSenseVoiceFrontend(arguments.sensevoice_path)
    session = ort.InferenceSession(str(arguments.model), providers=["CPUExecutionProvider"])
    available_outputs = {output.name for output in session.get_outputs()}
    if PROBE_EMBEDDING_OUTPUT not in available_outputs:
        raise ValueError("ONNX model does not expose the probe embedding")
    probe = LinearProbeHead(arguments.probe_head)
    positive_decisions = infer_decisions(positive_rows, session, frontend, probe)
    ood_decisions = infer_decisions(ood_rows, session, frontend, probe)
    event_metrics = metrics(positive_decisions, ood_decisions, probe.target_labels)
    report = {
        "schema_version": "1.0",
        "candidate_passes": event_metrics["gate"]["passed"],
        "model_sha256": file_digest(arguments.model),
        "probe_sha256": file_digest(arguments.probe_head),
        "positive_manifest_sha256": file_digest(arguments.positive_manifest),
        "ood_manifest_sha256": file_digest(arguments.ood_manifest),
        "metrics": event_metrics,
        "positive_decisions": positive_decisions,
        "ood_false_positives": [row for row in ood_decisions if row["predicted"] is not None],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["metrics"], indent=2))
    if not report["metrics"]["gate"]["passed"]:
        raise SystemExit("error: ONNX event probe gate failed")


if __name__ == "__main__":
    main()
