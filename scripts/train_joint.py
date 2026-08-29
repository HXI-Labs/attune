#!/usr/bin/env python3
"""Train a budget-bounded unified Attune candidate from a versioned manifest."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from attune.models.joint import (
    AdaptationPolicy,
    AttuneJointModel,
    load_local_sensevoice,
    warm_start_attune_heads,
)
from attune.training.trainer import TrainerConfig, train_joint_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sensevoice-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--adaptation-policy", choices=[item.value for item in AdaptationPolicy], default="frozen"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="Frozen-probe delta used to warm-start heads for an adapted candidate",
    )
    parser.add_argument("--gpu-hour-cost-gbp", type=float)
    parser.add_argument("--maximum-cost-gbp", type=float)
    arguments = parser.parse_args()
    values = {}
    if arguments.config:
        text = arguments.config.read_text()
        values = json.loads(text) if arguments.config.suffix == ".json" else yaml.safe_load(text)
        values.pop("enabled", None)
        values.pop("adaptation_candidates", None)
    if arguments.gpu_hour_cost_gbp is not None:
        values["gpu_hour_cost_gbp"] = arguments.gpu_hour_cost_gbp
    if arguments.maximum_cost_gbp is not None:
        values["maximum_cost_gbp"] = arguments.maximum_cost_gbp
    config = TrainerConfig(**values)
    backbone = load_local_sensevoice(arguments.sensevoice_path, device=config.device)
    model = AttuneJointModel(
        backbone, adaptation_policy=AdaptationPolicy(arguments.adaptation_policy)
    )
    initial_checkpoint_sha256 = None
    if arguments.initial_checkpoint:
        initial_checkpoint_sha256 = hashlib.sha256(
            arguments.initial_checkpoint.read_bytes()
        ).hexdigest()
        checkpoint = torch.load(
            arguments.initial_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        warm_start_attune_heads(model, checkpoint)
    report = train_joint_model(
        model,
        manifest=arguments.manifest,
        output_dir=arguments.output_dir,
        config=config,
        initial_checkpoint_sha256=initial_checkpoint_sha256,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
