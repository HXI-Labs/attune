from __future__ import annotations

import importlib.util
from pathlib import Path


def test_vocalsound_partition_is_stable() -> None:
    path = Path("scripts/prepare_vocalsound_joint.py")
    spec = importlib.util.spec_from_file_location("prepare_vocalsound_joint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = {module._partition(f"vocalsound:f{index}") for index in range(50)}
    assert values == {"train", "development"}
    assert module._partition("vocalsound:f1") == module._partition("vocalsound:f1")
