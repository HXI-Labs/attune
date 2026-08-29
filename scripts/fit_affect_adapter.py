#!/usr/bin/env python3
"""Fit the predeclared frozen-output affect adapter and evaluate without deploying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attune.affect_adapter import (
    evaluate_affect_adapter,
    fit_affect_adapter,
    load_score_rows,
)
from attune.integrity import file_digest


def _named_paths(value: str | list[str] | dict[str, str], prefix: str) -> dict[str, Path]:
    if isinstance(value, str):
        return {prefix: Path(value)}
    if isinstance(value, list):
        return {f"{prefix}_{index}": Path(path) for index, path in enumerate(value)}
    return {name: Path(path) for name, path in value.items()}


def _load_combined(paths: dict[str, Path]) -> list[dict[str, object]]:
    return [row for path in paths.values() for row in load_score_rows(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/model/affect-adapter-v0.3.json")
    )
    parser.add_argument("--adapter-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    train_paths = _named_paths(config["training_scores"], "train")
    development_paths = _named_paths(config["selection_scores"], "development")
    diagnostic_paths = _named_paths(config["diagnostic_scores"], "opened_sealed_diagnostic")
    external_paths = _named_paths(config["external_evaluation_scores"], "external_ravdess")
    train = _load_combined(train_paths)
    development = _load_combined(development_paths)
    adapter = fit_affect_adapter(
        train,
        development,
        candidate_c=tuple(float(value) for value in config["model"]["candidate_c"]),
    )
    reports = {
        "selection_combined": evaluate_affect_adapter(adapter, development),
        **{
            name: evaluate_affect_adapter(adapter, load_score_rows(path))
            for name, path in development_paths.items()
        },
        **{
            name: evaluate_affect_adapter(adapter, load_score_rows(path))
            for name, path in diagnostic_paths.items()
        },
        **{
            name: evaluate_affect_adapter(adapter, load_score_rows(path))
            for name, path in external_paths.items()
        },
    }
    acceptance = config["acceptance"]
    external_name = str(acceptance.get("primary_external_report", "external_ravdess"))
    external_report = reports[external_name]
    gates = {
        "macro_f1": external_report["macro_f1"] >= acceptance["external_ravdess_macro_f1_minimum"],
        "anger_recall": external_report["per_class"]["anger"]["recall"]
        >= acceptance["external_ravdess_anger_recall_minimum"],
        "class_balance": external_report["maximum_predicted_class_share"]
        <= acceptance["external_maximum_predicted_class_share"],
        "coverage": external_report["coverage"] >= acceptance["external_minimum_coverage"],
        "selective_risk": bool(external_report["selective_risk_improves"]),
        "deployment_disabled": adapter.deployment_enabled is False,
    }
    for report_name, minimum in acceptance.get("additional_macro_f1_minimums", {}).items():
        gates[f"{report_name}_macro_f1"] = reports[report_name]["macro_f1"] >= minimum
    report = {
        "schema_version": "1.0",
        "config_sha256": file_digest(arguments.config),
        "score_sha256": {
            **{name: file_digest(path) for name, path in train_paths.items()},
            **{name: file_digest(path) for name, path in development_paths.items()},
            **{name: file_digest(path) for name, path in diagnostic_paths.items()},
            **{name: file_digest(path) for name, path in external_paths.items()},
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
        f"Fitted affect adapter; {external_name} macro-F1="
        f"{external_report['macro_f1']:.4f}; "
        f"candidate_passes={str(report['candidate_passes']).lower()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
