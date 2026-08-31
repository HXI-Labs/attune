#!/usr/bin/env python3
"""Convert a reviewed PyTorch linear-probe checkpoint to a NumPy artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from attune.models.probe_inference import (
    BERST_STYLE_LABEL_MAPPING,
    FSD50K_LABEL_MAPPING,
    VOCALSOUND_LABEL_MAPPING,
)

PROFILES = {
    "vocalsound": VOCALSOUND_LABEL_MAPPING,
    "fsd50k": FSD50K_LABEL_MAPPING,
    "berst-style": BERST_STYLE_LABEL_MAPPING,
}
CALIBRATION_COMPONENTS = {
    "vocalsound": "vocalsound_probe",
    "fsd50k": "fsd50k_probe",
    "berst-style": "berst_style_probe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Validation-fitted calibration JSON; required for release artifacts",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    labels = checkpoint.get("labels")
    abstention = checkpoint.get("abstention")
    state = checkpoint.get("head_state_dict")
    if not isinstance(labels, list) or set(labels) != set(PROFILES[arguments.profile]):
        raise SystemExit("error: checkpoint labels do not match the selected profile")
    if not isinstance(abstention, dict) or abstention.get("method") != "none_logit":
        raise SystemExit("error: release probes require none-logit abstention")
    if not isinstance(state, dict) or set(state) != {"weight", "bias"}:
        raise SystemExit("error: unsupported linear head state")
    calibration = checkpoint.get("calibration")
    if arguments.calibration is not None:
        calibration_payload = json.loads(arguments.calibration.read_text())
        calibration = calibration_payload["components"][CALIBRATION_COMPONENTS[arguments.profile]]
        expected_labels = [*labels, "none"]
        if calibration.get("labels") != expected_labels:
            raise SystemExit("error: calibration labels do not match the probe checkpoint")
    temperature = (
        float(calibration.get("temperature", 1.0)) if isinstance(calibration, dict) else 1.0
    )
    mapping = PROFILES[arguments.profile]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output,
        format=np.asarray("attune_linear_probe_v1"),
        embedding=np.asarray(checkpoint["embedding"]),
        labels=np.asarray(labels),
        channels=np.asarray([mapping[label][0] for label in labels]),
        target_labels=np.asarray([mapping[label][1].value for label in labels]),
        weight=state["weight"].detach().float().numpy(),
        bias=state["bias"].detach().float().numpy(),
        feature_mean=checkpoint["feature_mean"].detach().float().numpy(),
        feature_scale=checkpoint["feature_scale"].detach().float().numpy(),
        abstention_method=np.asarray(abstention["method"]),
        abstention_threshold=np.asarray(float(abstention["threshold"]), dtype=np.float32),
        temperature=np.asarray(temperature, dtype=np.float32),
    )
    print(arguments.output)


if __name__ == "__main__":
    main()
