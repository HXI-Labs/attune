#!/usr/bin/env python3
"""Evaluate a bounded interpolation of two existing affect heads."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from attune.calibration_joint import fit_runtime_calibration, load_score_rows
from attune.evaluation.joint import evaluate_joint_scores


def _alignment_key(clip_id: str) -> str:
    for prefix in ("crema-joint-", "crema-perceptual-"):
        if clip_id.startswith(prefix):
            return f"crema:{clip_id.removeprefix(prefix)}"
    return clip_id


def _rows_by_clip(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {_alignment_key(row["clip_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("score file contains duplicate clip IDs")
    return indexed


def _blend_rows(
    primary_rows: list[dict[str, Any]],
    secondary_rows: list[dict[str, Any]],
    alpha: float,
) -> list[dict[str, Any]]:
    secondary = _rows_by_clip(secondary_rows)
    shared_ids = sorted(set(_rows_by_clip(primary_rows)) & set(secondary))
    if not shared_ids:
        raise ValueError("score files have no shared clip IDs")
    primary = _rows_by_clip(primary_rows)
    blended = []
    for clip_id in shared_ids:
        primary_row = deepcopy(primary[clip_id])
        secondary_row = secondary[clip_id]
        first = np.asarray(primary_row["affect_logits"], dtype=np.float64)
        second = np.asarray(secondary_row["affect_logits"], dtype=np.float64)
        if first.shape != second.shape:
            raise ValueError(f"affect logits differ for {clip_id}")
        primary_row["affect_logits"] = ((1.0 - alpha) * first + alpha * second).tolist()
        blended.append(primary_row)
    return blended


def _eligible(development: dict[str, Any], external: dict[str, Any]) -> bool:
    return all(
        (
            development["affect_macro_f1"] >= 0.64,
            external["affect_macro_f1"] >= 0.40,
            development["acoustic_preference_score"] > 0.0,
            external["acoustic_preference_score"] > 0.0,
            development["affect_coverage"] >= 0.50,
            external["affect_coverage"] >= 0.50,
            development["affect_selective_risk_improves"],
            external["affect_selective_risk_improves"],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-development", type=Path, required=True)
    parser.add_argument("--secondary-development", type=Path, required=True)
    parser.add_argument("--primary-external", type=Path, required=True)
    parser.add_argument("--secondary-external", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path, required=True)
    arguments = parser.parse_args()

    primary_development = load_score_rows(arguments.primary_development)
    secondary_development = load_score_rows(arguments.secondary_development)
    primary_external = load_score_rows(arguments.primary_external)
    secondary_external = load_score_rows(arguments.secondary_external)

    candidates = []
    for alpha in np.linspace(0.0, 0.5, 11):
        development_rows = _blend_rows(
            primary_development,
            secondary_development,
            float(alpha),
        )
        calibration = fit_runtime_calibration(
            development_rows,
            affect_bias_mode="aps_constrained",
        )
        development = evaluate_joint_scores(development_rows, calibration)
        external_rows = _blend_rows(primary_external, secondary_external, float(alpha))
        external = evaluate_joint_scores(external_rows, calibration)
        harmonic_mean = 2.0 / (
            1.0 / development["affect_macro_f1"] + 1.0 / external["affect_macro_f1"]
        )
        candidates.append(
            {
                "alpha": float(alpha),
                "eligible": _eligible(development, external),
                "selection_score": harmonic_mean,
                "development": development,
                "external": external,
                "calibration": json.loads(calibration.model_dump_json()),
            }
        )

    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = (
        max(
            eligible,
            key=lambda candidate: (
                candidate["selection_score"],
                -candidate["alpha"],
            ),
        )
        if eligible
        else None
    )
    report = {
        "schema_version": "1.0",
        "opened_external_diagnostic": True,
        "candidate_passes": selected is not None,
        "selected_alpha": selected["alpha"] if selected else None,
        "candidates": candidates,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    if selected is not None:
        arguments.calibration_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.calibration_output.write_text(
            json.dumps(selected["calibration"], indent=2) + "\n"
        )
    print(
        json.dumps(
            {
                "candidate_passes": report["candidate_passes"],
                "selected_alpha": report["selected_alpha"],
                "output": str(arguments.output),
            }
        )
    )


if __name__ == "__main__":
    main()
