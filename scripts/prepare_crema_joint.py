#!/usr/bin/env python3
"""Prepare fixed speaker-disjoint, same-text CREMA-D source-label pairs."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

from prepare_licence_clean_inspection import (
    CREMA_D_REVISION,
    PreparationError,
    convert_to_pcm16,
    download_with_retry,
)

from attune.integrity import file_digest

PARTITIONS = {
    "train": range(1031, 1051),
    "development": range(1061, 1071),
    "sealed_test": range(1081, 1092),
}
SENTENCES = ("IEO", "IWL")
SOURCE_LABELS = {
    "ANG": "anger",
    "DIS": "other",
    "FEA": "fear",
    "HAP": "joy",
    "NEU": "neutral",
    "SAD": "distress",
}
ATTRIBUTION = (
    "CREMA-D by Cao et al. (IEEE Transactions on Affective Computing 2014, "
    "doi:10.1109/TAFFC.2014.2336244); database ODbL 1.0, contents DbCL 1.0."
)


def protocol_rows() -> list[dict[str, object]]:
    rows = []
    for partition, actors in PARTITIONS.items():
        for actor in actors:
            for sentence in SENTENCES:
                for source_label, target in SOURCE_LABELS.items():
                    intensity = "HI" if sentence == "IEO" and source_label != "NEU" else "XX"
                    filename = f"{actor}_{sentence}_{source_label}_{intensity}.wav"
                    rows.append(
                        {
                            "attribution": ATTRIBUTION,
                            "cache_path": f"crema_d/{filename}",
                            "clip_id": f"crema-joint-{Path(filename).stem.lower()}",
                            "licence": "ODbL-1.0 database / DbCL-1.0 contents",
                            "partition": partition,
                            "source_dataset": "CREMA-D",
                            "source_filename": filename,
                            "speaker_id": f"crema-d:{actor}",
                            "source_label": source_label,
                            "target_affect": target,
                            "label_status": "acted source label; not Attune gold",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/crema-joint"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/crema-joint-v0.1.jsonl")
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=4)
    arguments = parser.parse_args()
    rows = protocol_rows()
    try:
        for index, row in enumerate(rows, 1):
            target = arguments.cache_dir / str(row["cache_path"])
            if arguments.download and not target.is_file():
                print(f"CREMA joint {index}/{len(rows)}: {row['source_filename']}")
                source = arguments.cache_dir / "_downloads" / str(row["source_filename"])
                download_with_retry(
                    str(row["url"]),
                    source,
                    timeout_s=arguments.timeout,
                    retries=arguments.retries,
                )
                convert_to_pcm16(source, target)
                source.unlink(missing_ok=True)
            if not target.is_file():
                raise PreparationError(f"missing CREMA joint audio: {target}")
            row["sha256"] = file_digest(target)
            row["duration_ms"] = _duration_ms(target)
    except PreparationError as error:
        raise SystemExit(f"error: {error}") from error
    arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
    arguments.manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(f"Verified {len(rows)} speaker-disjoint paired CREMA-D clips")


if __name__ == "__main__":
    main()
