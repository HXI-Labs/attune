#!/usr/bin/env python3
"""Regenerate schema-valid gold JSON from the synthetic fixture manifest."""

from __future__ import annotations

import json
from pathlib import Path

from attune.baselines.adapters import BaselineInput, build_partial_output
from attune.schema.output import AffectCategory


def main() -> None:
    root = Path(__file__).parent
    manifest = json.loads((root / "manifest.json").read_text())
    outputs = {}
    for item in manifest["items"]:
        target = AffectCategory(item["acoustic_target"])
        distribution = {category: 0.02 for category in AffectCategory}
        distribution[target] = 0.86
        output = build_partial_output(
            BaselineInput(root / item["audio"], item["transcript"]),
            model_name="synthetic-fixture-gold",
            transcript=item["transcript"],
            category=target,
            distribution=distribution,
        )
        outputs[item["id"]] = output.model_dump(mode="json")
    (root / "gold.json").write_text(json.dumps(outputs, indent=2) + "\n")


if __name__ == "__main__":
    main()
