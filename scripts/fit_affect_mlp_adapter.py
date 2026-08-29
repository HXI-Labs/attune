#!/usr/bin/env python3
"""Fit and audit the predeclared nonlinear affect adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from attune.affect_adapter import load_score_rows
from attune.affect_mlp_adapter import evaluate_affect_mlp, fit_affect_mlp_adapter


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def paths(value: dict[str, str]) -> dict[str, Path]:
    return {name: Path(path) for name, path in value.items()}


def combined(values: dict[str, Path]) -> list[dict[str, Any]]:
    return [row for path in values.values() for row in load_score_rows(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/model/affect-mlp-adapter-v0.5.json")
    )
    parser.add_argument("--adapter-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    training = paths(config["training_scores"])
    selection = paths(config["selection_scores"])
    diagnostics = paths(config["diagnostic_scores"])
    model = config["model"]
    adapter = fit_affect_mlp_adapter(
        combined(training),
        combined(selection),
        hidden_sizes=tuple(model["hidden_sizes"]),
        dropout=float(model["dropout"]),
        learning_rate=float(model["learning_rate"]),
        weight_decay=float(model["weight_decay"]),
        maximum_epochs=int(model["maximum_epochs"]),
        patience=int(model["patience"]),
        seed=int(config["seed"]),
    )
    reports = {
        "selection_combined": evaluate_affect_mlp(adapter, combined(selection)),
        **{
            name: evaluate_affect_mlp(adapter, load_score_rows(path))
            for name, path in selection.items()
        },
        **{
            name: evaluate_affect_mlp(adapter, load_score_rows(path))
            for name, path in diagnostics.items()
        },
    }
    acceptance = config["acceptance"]
    gates = {
        f"{name}_macro_f1": reports[name]["macro_f1"] >= minimum
        for name, minimum in acceptance["macro_f1_minimums"].items()
    }
    external = reports[acceptance["primary_external_report"]]
    gates.update(
        {
            "external_class_balance": external["maximum_predicted_class_share"]
            <= acceptance["external_maximum_predicted_class_share"],
            "external_coverage": external["coverage"]
            >= acceptance["external_minimum_coverage"],
            "external_selective_risk": bool(external["selective_risk_improves"]),
            "deployment_disabled": adapter.deployment_enabled is False,
        }
    )
    all_paths = {**training, **selection, **diagnostics}
    report = {
        "schema_version": "1.0",
        "config_sha256": file_sha256(arguments.config),
        "score_sha256": {name: file_sha256(path) for name, path in all_paths.items()},
        "best_epoch": adapter.best_epoch,
        "temperature": adapter.temperature,
        "abstention_threshold": adapter.abstention_threshold,
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
        f"Fitted affect MLP at epoch {adapter.best_epoch}; "
        f"passes={str(report['candidate_passes']).lower()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
