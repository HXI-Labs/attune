from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[2] / "scripts/prepare_disfluency_speech.py"
SPEC = importlib.util.spec_from_file_location("prepare_disfluency_speech", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
DISFLUENCY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DISFLUENCY)


def test_event_presence_maps_only_supported_explicit_tags() -> None:
    transcript = "<Laughter> hello <throat_clearing> there <breath> <sigh> <cough>"

    assert DISFLUENCY.event_presence(transcript) == [
        "laugh",
        "throat_clear",
        "sigh",
        "cough",
    ]


def test_event_presence_deduplicates_repeated_tags() -> None:
    assert DISFLUENCY.event_presence("<laughter> text <Laughter>") == ["laugh"]


def test_pinned_source_is_clip_disjoint_not_claimed_speaker_disjoint() -> None:
    assert DISFLUENCY.SHARDS["data/validation-00000-of-00001.parquet"]["split"] == ("development")
    assert DISFLUENCY.SHARDS["data/test-00000-of-00001.parquet"]["split"] == "sealed_test"
