from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path("scripts/prepare_crema_joint.py")
    spec = importlib.util.spec_from_file_location("prepare_crema_joint", path)
    assert spec is not None and spec.loader is not None
    sys.path.insert(0, str(path.parent.resolve()))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_crema_joint_protocol_is_speaker_disjoint_and_paired() -> None:
    module = _module()
    rows = module.protocol_rows()
    partitions = {}
    for row in rows:
        speaker = row["speaker_id"]
        assert speaker not in partitions or partitions[speaker] == row["partition"]
        partitions[speaker] = row["partition"]
    assert len(rows) == (20 + 10 + 11) * 2 * 6
    assert {row["target_affect"] for row in rows} == {
        "anger",
        "other",
        "fear",
        "joy",
        "neutral",
        "distress",
    }
