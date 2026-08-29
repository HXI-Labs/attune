#!/usr/bin/env python3
"""Prepare a bounded Common Voice ASR replay set disjoint from British inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_common_voice_british import (  # noqa: E402
    ATTRIBUTION,
    DATASET_NAME,
    LICENCE,
    MAX_DURATION_S,
    MIN_DURATION_S,
    MIRROR_ID,
    MIRROR_REVISION,
    SOURCE_SPLIT,
    _asset_url,
    _convert,
    _source_row,
    _stream_metadata,
    file_digest,
    verify_audio,
)


def _partition(client_id: str) -> str:
    bucket = int(hashlib.sha256(client_id.encode()).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "development"
    if bucket == 1:
        return "sealed_test"
    return "train"


def _held_out_clients(path: Path) -> set[str]:
    return {
        str(row["client_id"])
        for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    }


def _retry(operation, description: str, attempts: int = 5):
    for attempt in range(attempts):
        try:
            return operation()
        except Exception:
            if attempt + 1 == attempts:
                raise
            delay = 2**attempt
            print(f"{description} failed; retrying in {delay}s", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _load_partial(path: Path, cache_root: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        row["partition"] = _partition(str(row["client_id"]))
        target = cache_root / str(row["cache_path"])
        if not target.is_file() or file_digest(target) != row["sha256"]:
            raise ValueError(f"partial replay manifest has missing/changed audio: {target}")
        verify_audio(target, float(row["duration_s"]))
    return rows


def _write_partial(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.jsonl")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


def create_replay(
    *,
    output: Path,
    cache_root: Path,
    inspection_manifest: Path,
    clip_count: int,
) -> list[dict[str, object]]:
    if not 100 <= clip_count <= 2_000:
        raise ValueError("clip_count must remain between 100 and 2,000")
    excluded = _held_out_clients(inspection_manifest)
    rows = _load_partial(output, cache_root)
    if len(rows) > clip_count:
        raise ValueError("existing replay manifest exceeds requested clip_count")
    if len(rows) == clip_count:
        _write_partial(output, rows)
        return rows
    seen = excluded | {str(row["client_id"]) for row in rows}
    completed_indices = {int(row["source_row_index"]) for row in rows}
    page_cache: dict[int, dict[int, dict]] = {}
    for row_index, metadata in enumerate(_stream_metadata()):
        if row_index in completed_indices:
            continue
        client_id = str(metadata.get("client_id", "")).strip()
        transcript = str(metadata.get("sentence", "")).strip()
        if not client_id or not transcript or client_id in seen:
            continue
        source_row = _retry(
            lambda index=row_index: _source_row(index, page_cache),
            f"Common Voice metadata row {row_index}",
        )
        if (
            str(source_row.get("client_id", "")).strip() != client_id
            or str(source_row.get("sentence", "")).strip() != transcript
        ):
            raise ValueError(f"streamed metadata mismatch at Common Voice row {row_index}")
        relative_path = f"clips/cv17-en-replay-{row_index:07d}.wav"
        target = cache_root / relative_path
        duration_s, converted_hash, source_hash = _retry(
            lambda source=source_row, path=target: _convert(_asset_url(source), path),
            f"Common Voice audio row {row_index}",
        )
        if not MIN_DURATION_S <= duration_s <= MAX_DURATION_S:
            target.unlink(missing_ok=True)
            continue
        seen.add(client_id)
        rows.append(
            {
                "clip_id": f"cv17-replay-{source_hash[:16]}",
                "source_dataset": DATASET_NAME,
                "source_mirror": MIRROR_ID,
                "dataset_revision": MIRROR_REVISION,
                "source_split": SOURCE_SPLIT,
                "source_row_index": row_index,
                "transcript": transcript,
                "client_id": client_id,
                "duration_s": round(duration_s, 6),
                "sample_rate_hz": 16_000,
                "channels": 1,
                "sha256": converted_hash,
                "source_audio_sha256": source_hash,
                "licence": LICENCE,
                "attribution": ATTRIBUTION,
                "cache_path": relative_path,
                "partition": _partition(client_id),
            }
        )
        _write_partial(output, rows)
        print(f"[{len(rows)}/{clip_count}] Common Voice replay row {row_index}", flush=True)
        if len(rows) == clip_count:
            break
    if len(rows) != clip_count:
        raise ValueError(f"stream ended after only {len(rows)} usable replay clips")
    _write_partial(output, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/common-voice-replay-v0.1.jsonl"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/common-voice-replay-v0.1"))
    parser.add_argument(
        "--inspection-manifest",
        type=Path,
        default=Path("data/manifests/common-voice-british-wer.jsonl"),
    )
    parser.add_argument("--clip-count", type=int, default=600)
    arguments = parser.parse_args()
    rows = create_replay(
        output=arguments.output,
        cache_root=arguments.cache_dir,
        inspection_manifest=arguments.inspection_manifest,
        clip_count=arguments.clip_count,
    )
    counts = {
        split: sum(row["partition"] == split for row in rows)
        for split in ("train", "development", "sealed_test")
    }
    print(json.dumps({"rows": len(rows), "partitions": counts}))


if __name__ == "__main__":
    main()
