from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts/prepare_ghana_english.py"
SPEC = importlib.util.spec_from_file_location("prepare_ghana_english", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PREPARE_GHANA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE_GHANA)

GhanaPreparationError = PREPARE_GHANA.GhanaPreparationError
load_manifest = PREPARE_GHANA.load_manifest


def test_committed_ghana_manifest_is_nc_research_only() -> None:
    repository = Path(__file__).parents[2]
    rows = load_manifest(repository / "data/manifests/ghana-english-wer.jsonl")

    assert 80 <= len(rows) <= 120
    assert {row["licence"] for row in rows} == {"CC-BY-NC-4.0"}
    assert all(row["research_only"] is True for row in rows)
    assert all(row["commercial_redistribution_prohibited"] is True for row in rows)
    assert all(row["speaker_id"] is None for row in rows)
    assert all(row["speaker_disjoint"] is False for row in rows)
    assert all(0.5 <= row["duration_s"] <= 30 for row in rows)
    assert all(row["sample_rate_hz"] == 16_000 and row["channels"] == 1 for row in rows)
    assert all(len(row["sha256"]) == 64 for row in rows)
    assert all(row["fetch_script"] == "scripts/prepare_ghana_english.py" for row in rows)


def test_manifest_rejects_commercially_redistributable_claim(tmp_path: Path) -> None:
    repository = Path(__file__).parents[2]
    source = repository / "data/manifests/ghana-english-wer.jsonl"
    row = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
    row["commercial_redistribution_prohibited"] = False
    manifest = tmp_path / "unsafe.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(GhanaPreparationError, match="NC research-only"):
        load_manifest(manifest)


def test_manifest_rejects_false_speaker_disjoint_claim(tmp_path: Path) -> None:
    repository = Path(__file__).parents[2]
    source = repository / "data/manifests/ghana-english-wer.jsonl"
    row = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
    row["speaker_disjoint"] = True
    manifest = tmp_path / "unsafe.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(GhanaPreparationError, match="do not claim disjointness"):
        load_manifest(manifest)
