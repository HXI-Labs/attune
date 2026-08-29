#!/usr/bin/env python3
"""Export a locally trained unified Attune checkpoint to ONNX."""

import argparse
import json
from pathlib import Path

import torch

from attune.inference.export import export_onnx, validate_onnx_parity, write_export_report
from attune.models.joint import (
    AdaptationPolicy,
    AttuneJointModel,
    load_attune_checkpoint,
    load_local_sensevoice,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensevoice-path", required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Attune delta checkpoint; omit only for the unadapted ASR reference export",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-size",
        type=int,
        help="Override inferred post-frontend feature width (normally unnecessary)",
    )
    parser.add_argument(
        "--adaptation-policy",
        choices=[item.value for item in AdaptationPolicy],
        default="frozen",
    )
    parser.add_argument(
        "--preserve-base-asr",
        action="store_true",
        help="Use frozen copies of the two base encoder tail blocks for CTC ASR",
    )
    arguments = parser.parse_args()
    backbone = load_local_sensevoice(arguments.sensevoice_path)
    model = AttuneJointModel(
        backbone,
        adaptation_policy=AdaptationPolicy(arguments.adaptation_policy),
        preserve_base_asr=arguments.preserve_base_asr,
    )
    if arguments.checkpoint:
        state = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
        load_attune_checkpoint(model, state)
    export_onnx(model, arguments.output, feature_size=arguments.feature_size)
    parity = validate_onnx_parity(model, arguments.output)
    report_path = arguments.output.with_suffix(".export.json")
    write_export_report(
        report_path,
        model_path=arguments.output,
        parity=parity,
        metadata={
            "adaptation_policy": arguments.adaptation_policy,
            "preserve_base_asr": arguments.preserve_base_asr,
            "checkpoint": str(arguments.checkpoint) if arguments.checkpoint else None,
            "purpose": "candidate" if arguments.checkpoint else "unadapted_asr_reference",
        },
    )
    print(json.dumps({"output": str(arguments.output), "report": str(report_path)}))


if __name__ == "__main__":
    main()
