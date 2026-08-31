#!/usr/bin/env python3
"""Build explicit auxiliary-negative source rows from the pinned British speech slice."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from prepare_common_voice_british import load_manifest, safe_target, verify_audio

from attune.integrity import file_digest
from attune.training.prepare import SourceRow


def build_rows(
    rows: list[dict[str, Any]],
    *,
    cache_root: Path,
    output_path: Path,
) -> list[SourceRow]:
    """Convert transcript-verified speech into explicit event/style negative controls."""
    prepared: list[SourceRow] = []
    for row in rows:
        audio_path = safe_target(cache_root, row["cache_path"])
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        verify_audio(audio_path, float(row["duration_s"]))
        prepared.append(
            SourceRow(
                clip_id=row["clip_id"],
                dataset_id="common_voice_17_british_external_control",
                split="sealed_test",
                speaker_id=row["client_id"],
                audio_path=Path(
                    os.path.relpath(audio_path.resolve(), output_path.parent.resolve())
                ),
                audio_sha256=file_digest(audio_path),
                duration_ms=round(float(row["duration_s"]) * 1000),
                transcript=row["transcript"],
                events=None,
                event_presence=[],
                styles=[],
                affect_distribution=None,
                vad=None,
                pair_id=-1,
                is_ood=False,
                lexical_affect_label=None,
                auxiliary_negative_tasks=["event_presence", "styles"],
            )
        )
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/common-voice-british-wer.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/common-voice-british-wer"),
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    rows = build_rows(
        load_manifest(arguments.manifest),
        cache_root=arguments.cache_dir,
        output_path=arguments.output,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(rows)} explicit British speech controls to {arguments.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
