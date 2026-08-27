from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_script() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "infer.py"
    spec = importlib.util.spec_from_file_location("infer_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_probe_heads_are_skipped_independently(tmp_path: Path) -> None:
    infer = load_script()
    vocalsound = tmp_path / "vocalsound.pt"
    vocalsound.touch()

    heads, skipped = infer.available_probe_heads(
        encoder=object(),
        vocalsound_checkpoint=vocalsound,
        fsd50k_checkpoint=tmp_path / "missing-fsd50k.pt",
    )

    assert len(heads) == 1
    assert heads[0].name == "vocalsound-frozen-linear-probe"
    assert len(skipped) == 1
    assert skipped[0].startswith("FSD50K probe: missing")


def test_all_missing_probe_heads_still_build_aed_affect_mode(tmp_path: Path) -> None:
    infer = load_script()

    heads, skipped = infer.available_probe_heads(
        encoder=object(),
        vocalsound_checkpoint=tmp_path / "missing-vocalsound.pt",
        fsd50k_checkpoint=tmp_path / "missing-fsd50k.pt",
    )

    assert heads == ()
    assert len(skipped) == 2
