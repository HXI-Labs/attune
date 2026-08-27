#!/usr/bin/env python3
"""Prepare a bounded actor-disjoint CREMA speech-negative pool for probe abstention."""

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

DEFAULT_CACHE = Path("data/raw/crema-probe-ood")
DEFAULT_MANIFEST = Path("data/manifests/crema-probe-ood.jsonl")
SENTENCES = ("IEO", "IWL", "IOM")
PARTITION_ACTORS = {
    "train": range(1031, 1051),
    "validation": range(1051, 1061),
}
ATTRIBUTION = (
    "CREMA-D by Cao et al. (IEEE Transactions on Affective Computing 2014, "
    "doi:10.1109/TAFFC.2014.2336244); database ODbL 1.0, contents DbCL 1.0."
)


def rows_for_protocol() -> list[dict[str, str]]:
    """Return the fixed actor/sentence protocol without touching the network."""
    rows = []
    for partition, actors in PARTITION_ACTORS.items():
        for actor in actors:
            for sentence in SENTENCES:
                filename = f"{actor}_{sentence}_NEU_XX.wav"
                rows.append(
                    {
                        "attribution": ATTRIBUTION,
                        "cache_path": f"crema_d/{filename}",
                        "clip_id": f"crema-d-ood-{Path(filename).stem.lower()}",
                        "licence": "ODbL-1.0 database / DbCL-1.0 contents",
                        "partition": partition,
                        "source_dataset": "CREMA-D",
                        "source_filename": filename,
                        "speaker_id": f"crema-d:{actor}",
                        "url": (
                            "https://media.githubusercontent.com/media/"
                            f"CheyneyComputerScience/CREMA-D/{CREMA_D_REVISION}/"
                            f"AudioWAV/{filename}"
                        ),
                    }
                )
    return rows


def build(
    cache_root: Path,
    manifest: Path,
    *,
    timeout_s: float,
    retries: int,
) -> list[dict[str, str]]:
    """Fetch, convert, hash, and write the bounded protocol."""
    rows = rows_for_protocol()
    for index, row in enumerate(rows, 1):
        filename = row["source_filename"]
        source = cache_root / "_downloads" / filename
        target = cache_root / row["cache_path"]
        print(f"CREMA probe OOD {index}/{len(rows)}: {filename}", flush=True)
        if not source.is_file():
            download_with_retry(
                row["url"],
                source,
                timeout_s=timeout_s,
                retries=retries,
            )
        convert_to_pcm16(source, target)
        row["sha256"] = digest(target)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def verify(rows: list[dict[str, str]], cache_root: Path) -> None:
    """Verify every converted clip against the committed manifest hash."""
    for row in rows:
        target = cache_root / row["cache_path"]
        if not target.is_file():
            raise PreparationError(f"missing CREMA probe OOD clip: {target}")
        if digest(target) != row["sha256"]:
            raise PreparationError(f"CREMA probe OOD hash mismatch: {target}")


def load_manifest(path: Path) -> list[dict[str, str]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise PreparationError(f"cannot read CREMA probe OOD manifest: {error}") from error
    if not rows:
        raise PreparationError("CREMA probe OOD manifest is empty")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    try:
        rows = (
            build(
                arguments.cache_dir,
                arguments.manifest,
                timeout_s=arguments.timeout,
                retries=arguments.retries,
            )
            if arguments.download
            else load_manifest(arguments.manifest)
        )
        verify(rows, arguments.cache_dir)
    except PreparationError as error:
        raise SystemExit(f"error: {error}") from error
    print(f"Verified {len(rows)} CREMA probe OOD clips in {arguments.cache_dir}")


if __name__ == "__main__":
    main()
