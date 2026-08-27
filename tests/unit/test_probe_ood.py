import json
from pathlib import Path

import pytest

from attune.models.frozen_event_probe import ProbeDataError
from attune.models.probe_ood import crema_probe_negatives


def write_manifest(path: Path, speakers: tuple[str, str]) -> None:
    rows = [
        {
            "cache_path": f"{speaker}.wav",
            "clip_id": f"clip-{index}",
            "partition": partition,
            "speaker_id": speaker,
            "source_dataset": "CREMA-D",
        }
        for index, (speaker, partition) in enumerate(
            zip(speakers, ("train", "validation"), strict=True)
        )
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_crema_negatives_are_actor_disjoint_from_inspection(tmp_path: Path) -> None:
    manifest = tmp_path / "crema.jsonl"
    write_manifest(manifest, ("crema-d:1031", "crema-d:1051"))

    examples = crema_probe_negatives(
        manifest,
        tmp_path,
        excluded_speakers={"crema-d:1001"},
    )

    assert {example.partition for example in examples} == {"train", "validation"}


def test_crema_negatives_reject_inspection_actor(tmp_path: Path) -> None:
    manifest = tmp_path / "crema.jsonl"
    write_manifest(manifest, ("crema-d:1031", "crema-d:1051"))

    with pytest.raises(ProbeDataError, match="310-clip inspection"):
        crema_probe_negatives(
            manifest,
            tmp_path,
            excluded_speakers={"crema-d:1031"},
        )
