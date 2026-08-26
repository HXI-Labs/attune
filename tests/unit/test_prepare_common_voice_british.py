from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts/prepare_common_voice_british.py"
SPEC = importlib.util.spec_from_file_location("prepare_common_voice_british", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PREPARE_CV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE_CV)

CommonVoicePreparationError = PREPARE_CV.CommonVoicePreparationError
load_manifest = PREPARE_CV.load_manifest


def test_committed_common_voice_manifest_is_cc0_and_speaker_disjoint() -> None:
    repository = Path(__file__).parents[2]
    rows = load_manifest(repository / "data/manifests/common-voice-british-wer.jsonl")

    assert 80 <= len(rows) <= 120
    assert len({row["client_id"] for row in rows}) == len(rows)
    assert {row["licence"] for row in rows} == {"CC0-1.0"}
    assert {row["accent_value"] for row in rows} == {
        "England English",
        "Scottish English",
    }
    assert all(row["speaker_disjoint"] is True for row in rows)
    assert all(0.5 <= row["duration_s"] <= 30 for row in rows)
    assert all(row["sample_rate_hz"] == 16_000 and row["channels"] == 1 for row in rows)
    assert all(len(row["sha256"]) == 64 for row in rows)
    assert all(
        row["fetch_script"] == "scripts/prepare_common_voice_british.py" for row in rows
    )


def test_manifest_rejects_duplicate_client_id(tmp_path: Path) -> None:
    repository = Path(__file__).parents[2]
    source = repository / "data/manifests/common-voice-british-wer.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    rows[1]["client_id"] = rows[0]["client_id"]
    manifest = tmp_path / "duplicate-speaker.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(CommonVoicePreparationError, match="client_id must be present and unique"):
        load_manifest(manifest)


def test_manifest_rejects_non_british_accent_claim(tmp_path: Path) -> None:
    repository = Path(__file__).parents[2]
    source = repository / "data/manifests/common-voice-british-wer.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    rows[0]["accent_value"] = "United States English"
    manifest = tmp_path / "wrong-accent.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(CommonVoicePreparationError, match="British CC0 provenance"):
        load_manifest(manifest)
