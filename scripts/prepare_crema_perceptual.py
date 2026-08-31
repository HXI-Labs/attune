#!/usr/bin/env python3
"""Prepare speaker-disjoint CREMA-D rows with voice-only listener distributions."""

from __future__ import annotations

import argparse
import csv
import json
import wave
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from prepare_licence_clean_inspection import (
    CREMA_D_REVISION,
    PreparationError,
    convert_to_pcm16,
    download_with_retry,
)

from attune.integrity import file_digest
from attune.schema.output import AffectCategory

ACTOR_PARTITIONS = {
    "train": (*range(1031, 1061), *range(1071, 1081)),
    "development": tuple(range(1061, 1071)),
}
KNOWN_INVALID_CLIPS = {"1064_IEO_DIS_MD", "1076_MTI_SAD_XX"}
RESPONSE_TO_AFFECT = {
    "A": "anger",
    "D": "other",
    "F": "fear",
    "H": "joy",
    "N": "neutral",
    "S": "distress",
}
ATTRIBUTION = (
    "CREMA-D by Cao et al. (IEEE Transactions on Affective Computing 2014, "
    "doi:10.1109/TAFFC.2014.2336244); database ODbL 1.0, contents DbCL 1.0."
)


def _actor_partition(actor: int) -> str | None:
    for partition, actors in ACTOR_PARTITIONS.items():
        if actor in actors:
            return partition
    return None


def protocol_rows(responses_path: Path) -> list[dict[str, object]]:
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    with responses_path.open(encoding="utf-8", newline="") as handle:
        for rating in csv.DictReader(handle):
            if rating["queryType"] != "1":
                continue
            clip_name = rating["clipName"]
            if clip_name in KNOWN_INVALID_CLIPS or _actor_partition(int(clip_name[:4])) is None:
                continue
            response = rating["respEmo"]
            if response not in RESPONSE_TO_AFFECT:
                raise ValueError(f"unsupported CREMA-D voice response: {response}")
            votes[clip_name][RESPONSE_TO_AFFECT[response]] += 1

    categories = [category.value for category in AffectCategory]
    rows = []
    for clip_name, counts in sorted(votes.items()):
        actor = int(clip_name[:4])
        rating_count = sum(counts.values())
        distribution = {label: counts[label] / rating_count for label in categories}
        filename = f"{clip_name}.wav"
        rows.append(
            {
                "attune_dataset_id": "crema_d_perceptual_v1",
                "attribution": ATTRIBUTION,
                "cache_path": f"crema_d/{filename}",
                "clip_id": f"crema-perceptual-{clip_name.lower()}",
                "licence": "ODbL-1.0 database / DbCL-1.0 contents",
                "partition": _actor_partition(actor),
                "source_dataset": "CREMA-D",
                "source_filename": filename,
                "speaker_id": f"crema-d:{actor}",
                "affect_distribution": distribution,
                "voice_rating_count": rating_count,
                "label_status": "voice-only listener distribution; not Attune gold",
                "url": (
                    "https://media.githubusercontent.com/media/"
                    f"CheyneyComputerScience/CREMA-D/{CREMA_D_REVISION}/"
                    f"AudioWAV/{filename}"
                ),
            }
        )
    return rows


def _duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        return round(handle.getnframes() / handle.getframerate() * 1000)


def _prepare_audio(
    row: dict[str, object],
    *,
    cache_dir: Path,
    download: bool,
    timeout: float,
    retries: int,
) -> None:
    target = cache_dir / str(row["cache_path"])
    if download and not target.is_file():
        source = cache_dir / "_downloads" / str(row["source_filename"])
        download_with_retry(str(row["url"]), source, timeout_s=timeout, retries=retries)
        convert_to_pcm16(source, target)
        source.unlink(missing_ok=True)
    if not target.is_file():
        raise PreparationError(f"missing CREMA-D perceptual audio: {target}")
    duration_ms = _duration_ms(target)
    if not 500 <= duration_ms <= 30_000:
        raise PreparationError(f"CREMA-D clip duration is outside Attune scope: {target}")
    row["sha256"] = file_digest(target)
    row["duration_ms"] = duration_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/crema-perceptual-v1"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/crema-perceptual-v1.jsonl")
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=4)
    arguments = parser.parse_args()
    rows = protocol_rows(arguments.responses)
    try:
        with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
            list(
                executor.map(
                    lambda row: _prepare_audio(
                        row,
                        cache_dir=arguments.cache_dir,
                        download=arguments.download,
                        timeout=arguments.timeout,
                        retries=arguments.retries,
                    ),
                    rows,
                )
            )
    except PreparationError as error:
        raise SystemExit(f"error: {error}") from error
    arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
    arguments.manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(f"Verified {len(rows)} listener-distribution CREMA-D clips")


if __name__ == "__main__":
    main()
