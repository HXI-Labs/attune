#!/usr/bin/env python3
"""Fit the predeclared frozen-output affect adapter and evaluate without deploying it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from attune.affect_adapter import (
    evaluate_affect_adapter,
    fit_affect_adapter,
    load_score_rows,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/model/affect-adapter-v0.3.json")
    )
    parser.add_argument("--adapter-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    train_path = Path(config["training_scores"])
    development_path = Path(config["selection_scores"])
    diagnostic_path = Path(config["diagnostic_scores"])
    external_path = Path(config["external_evaluation_scores"])
    train = load_score_rows(train_path)
    development = load_score_rows(development_path)
    diagnostic = load_score_rows(diagnostic_path)
    external = load_score_rows(external_path)
    adapter = fit_affect_adapter(
        train,
        development,
        candidate_c=tuple(float(value) for value in config["model"]["candidate_c"]),
    )
    reports = {
        "development": evaluate_affect_adapter(adapter, development),
        "opened_sealed_diagnostic": evaluate_affect_adapter(adapter, diagnostic),
        "external_ravdess": evaluate_affect_adapter(adapter, external),
    }
    acceptance = config["acceptance"]
    external_report = reports["external_ravdess"]
    gates = {
        "macro_f1": external_report["macro_f1"]
        >= acceptance["external_ravdess_macro_f1_minimum"],
        "anger_recall": external_report["per_class"]["anger"]["recall"]
        >= acceptance["external_ravdess_anger_recall_minimum"],
        "class_balance": external_report["maximum_predicted_class_share"]
        <= acceptance["external_maximum_predicted_class_share"],
        "coverage": external_report["coverage"] >= acceptance["external_minimum_coverage"],
        "selective_risk": bool(external_report["selective_risk_improves"]),
        "deployment_disabled": adapter.deployment_enabled is False,
    }
    report = {
        "schema_version": "1.0",
        "config_sha256": file_sha256(arguments.config),
        "score_sha256": {
            "train": file_sha256(train_path),
            "development": file_sha256(development_path),
            "diagnostic": file_sha256(diagnostic_path),
            "external": file_sha256(external_path),
        },
        "selected_c": adapter.selected_c,
        "temperature": adapter.temperature,
        "abstention_threshold": adapter.abstention_threshold,
        "supported_labels": adapter.supported_labels,
        "reports": reports,
        "acceptance_gates": gates,
        "candidate_passes": all(gates.values()),
        "deployment_enabled": False,
    }
    arguments.adapter_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.adapter_output.write_text(adapter.model_dump_json(indent=2) + "\n")
    arguments.report_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"Fitted affect adapter; external macro-F1={external_report['macro_f1']:.4f}; "
        f"candidate_passes={str(report['candidate_passes']).lower()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
