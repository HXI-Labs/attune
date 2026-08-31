from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).parents[2] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT_PATH = SCRIPT_DIR / "prepare_fsd50k_probe.py"
SPEC = importlib.util.spec_from_file_location("prepare_fsd50k_probe", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def test_selection_excludes_all_held_out_ids_and_is_balanced() -> None:
    candidates = []
    held_out = set()
    for class_index, source_class in enumerate(PREPARE.SOURCE_TO_PROBE_LABEL):
        held_id = str(class_index * 100 + 1)
        held_out.add(held_id)
        for offset in range(1, 5):
            candidates.append(
                {
                    "freesound_id": str(class_index * 100 + offset),
                    "source_label": source_class,
                    "source_split": "dev",
                }
            )

    rows = PREPARE.select_probe_rows(
        candidates,
        held_out,
        train_per_class=2,
        validation_per_class=1,
    )

    assert not ({row["freesound_id"] for row in rows} & held_out)
    assert Counter((row["partition"], row["probe_label"]) for row in rows) == {
        ("train", label): 2 for label in PREPARE.SOURCE_TO_PROBE_LABEL.values()
    } | {("validation", label): 1 for label in PREPARE.SOURCE_TO_PROBE_LABEL.values()}


def test_committed_probe_manifest_is_clip_disjoint_and_licence_clean() -> None:
    repository = Path(__file__).parents[2]
    rows = PREPARE.json.loads(
        "["
        + ",".join(
            line
            for line in (repository / "data/manifests/fsd50k-frozen-probe.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        + "]"
    )
    held_out = PREPARE.inspection_ids(repository / "data/manifests/licence-clean-inspection.jsonl")

    assert len(rows) == 320
    assert not ({row["clip_id"].removeprefix("fsd50k-") for row in rows} & held_out)
    assert {row["licence"]["clip_identifier"] for row in rows} <= {"CC0-1.0", "CC-BY-3.0"}
    assert all(len(set(row["labels"]) & set(PREPARE.SOURCE_TO_PROBE_LABEL)) == 1 for row in rows)
