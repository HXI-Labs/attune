from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "collect_probe_inspection_scores.py"
SPEC = importlib.util.spec_from_file_location("collect_probe_inspection_scores", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("source", "events", "styles", "component", "expected"),
    [
        ("VocalSound", ["cough"], [], "vocalsound_probe", "cough"),
        ("FSD50K", [], ["whispering"], "fsd50k_probe", "whisper"),
        ("CREMA-D", [], [], "vocalsound_probe", "none"),
    ],
)
def test_target_for_component(source, events, styles, component, expected) -> None:
    row = {
        "clip_id": "clip",
        "source_dataset": source,
        "intended_attune_labels": {"events": events, "styles": styles},
    }
    assert MODULE.target_for(row, component) == expected
