#!/usr/bin/env python3
"""Prepare a balanced actor-disjoint CREMA-D affect validation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_licence_clean_inspection import (
    CREMA_D_REVISION,
    PreparationError,
    convert_to_pcm16,
    digest,
    download_with_retry,
)

ACTORS = range(1051, 1061)
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


def protocol_rows() -> list[dict[str, str]]:
    rows = []
    for actor in ACTORS:
        for sentence in SENTENCES:
            for source_label, target in SOURCE_LABELS.items():
                intensity = (
                    "HI" if sentence == "IEO" and source_label != "NEU" else "XX"
                )
                filename = f"{actor}_{sentence}_{source_label}_{intensity}.wav"
                rows.append(
                    {
                        "clip_id": f"crema-affect-val-{Path(filename).stem.lower()}",
                        "speaker_id": f"crema-d:{actor}",
                        "source_dataset": "CREMA-D",
                        "source_filename": filename,
                        "cache_path": f"crema_d/{filename}",
                        "partition": "validation",
                        "source_label": source_label,
                        "target_affect": target,
                        "label_status": "acted source label; not reviewed gold",
                        "licence": "ODbL-1.0 database / DbCL-1.0 contents",
                        "attribution": ATTRIBUTION,
                        "url": (
                            "https://media.githubusercontent.com/media/"
                            f"CheyneyComputerScience/CREMA-D/{CREMA_D_REVISION}/"
                            f"AudioWAV/{filename}"
                        ),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/affect-calibration"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/affect-calibration.jsonl"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=4)
    arguments = parser.parse_args()
    rows = protocol_rows()
    try:
        for index, row in enumerate(rows, 1):
            target = arguments.cache_dir / row["cache_path"]
            if arguments.download and not target.is_file():
                print(f"Affect validation {index}/{len(rows)}: {row['source_filename']}")
                source = arguments.cache_dir / "_downloads" / row["source_filename"]
                download_with_retry(
                    row["url"],
                    source,
                    timeout_s=arguments.timeout,
                    retries=arguments.retries,
                )
                convert_to_pcm16(source, target)
                source.unlink(missing_ok=True)
            if not target.is_file():
                raise PreparationError(f"missing affect validation audio: {target}")
            row["sha256"] = digest(target)
    except PreparationError as error:
        raise SystemExit(f"error: {error}") from error
    arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
    arguments.manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"Verified {len(rows)} actor-disjoint validation clips")


if __name__ == "__main__":
    main()
