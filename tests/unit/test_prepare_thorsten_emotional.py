from __future__ import annotations

import hashlib
import importlib.util
import tarfile
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts/prepare_thorsten_emotional.py"
SPEC = importlib.util.spec_from_file_location("prepare_thorsten_emotional", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
THORSTEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(THORSTEN)


def test_safe_member_rejects_traversal_and_links() -> None:
    assert str(THORSTEN._safe_member(tarfile.TarInfo("dataset/wavs/example.wav"))) == (
        "dataset/wavs/example.wav"
    )
    with pytest.raises(THORSTEN.ThorstenPreparationError, match="unsafe archive path"):
        THORSTEN._safe_member(tarfile.TarInfo("../escape.wav"))
    link = tarfile.TarInfo("dataset/link")
    link.type = tarfile.SYMTYPE
    with pytest.raises(THORSTEN.ThorstenPreparationError, match="links/devices"):
        THORSTEN._safe_member(link)


def test_sentence_partitions_are_deterministic_and_sentence_disjoint() -> None:
    sentence_ids = [hashlib.md5(str(index).encode()).hexdigest() for index in range(300)]
    partitions = THORSTEN.sentence_partitions(sentence_ids)

    assert len(partitions) == 300
    assert sum(split == "train" for split, _ in partitions.values()) == 210
    assert sum(split == "development" for split, _ in partitions.values()) == 45
    assert sum(split == "sealed_test" for split, _ in partitions.values()) == 45
    assert len({pair_id for _, pair_id in partitions.values()}) == 300


def test_delivery_mapping_does_not_confuse_angry_with_shouting() -> None:
    assert THORSTEN._affect_distribution("angry")["anger"] == 1.0
    assert THORSTEN._affect_distribution("whisper") is None
    assert "shouting" not in THORSTEN.AFFECT_MAP.values()
