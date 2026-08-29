#!/usr/bin/env python3
"""Fit and audit a predeclared whisper adapter without enabling deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from attune.affect_adapter import load_score_rows
from attune.style_adapter import (
    evaluate_whisper_style_adapter,
    fit_whisper_style_adapter,
)


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def named_paths(value: str | list[str] | dict[str, str], prefix: str) -> dict[str, Path]:
    if isinstance(value, str):
        return {prefix: Path(value)}
    if isinstance(value, list):
        return {f"{prefix}_{index}": Path(path) for index, path in enumerate(value)}
    return {name: Path(path) for name, path in value.items()}


def combined(paths: dict[str, Path]) -> list[dict[str, Any]]:
    return [row for path in paths.values() for row in load_score_rows(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/model/whisper-adapter-v0.4.json")
    )
    parser.add_argument("--adapter-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    train_paths = named_paths(config["training_scores"], "train")
    selection_paths = named_paths(config["selection_scores"], "selection")
    diagnostic_paths = named_paths(config["diagnostic_scores"], "diagnostic")
    adapter = fit_whisper_style_adapter(
        combined(train_paths),
        combined(selection_paths),
        candidate_c=tuple(float(value) for value in config["model"]["candidate_c"]),
    )
    reports = {
        "selection_combined": evaluate_whisper_style_adapter(
            adapter, combined(selection_paths)
        ),
        **{
            name: evaluate_whisper_style_adapter(adapter, load_score_rows(path))
            for name, path in selection_paths.items()
        },
        **{
            name: evaluate_whisper_style_adapter(adapter, load_score_rows(path))
            for name, path in diagnostic_paths.items()
        },
    }
    gates = {
        f"{name}_{metric}_minimum": reports[name][metric] >= minimum
        for name, metric, minimum in config["acceptance"]["minimums"]
    }
    gates.update(
        {
            f"{name}_{metric}_maximum": reports[name][metric] <= maximum
            for name, metric, maximum in config["acceptance"]["maximums"]
        }
    )
    gates["deployment_disabled"] = adapter.deployment_enabled is False
    all_paths = {**train_paths, **selection_paths, **diagnostic_paths}
    report = {
        "schema_version": "1.0",
        "config_sha256": file_sha256(arguments.config),
        "score_sha256": {name: file_sha256(path) for name, path in all_paths.items()},
        "selected_c": adapter.selected_c,
        "temperature": adapter.temperature,
        "threshold": adapter.threshold,
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
        f"Fitted whisper adapter; passes={str(report['candidate_passes']).lower()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
