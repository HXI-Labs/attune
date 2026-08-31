#!/usr/bin/env python3
"""Merge reviewed public-data manifests into Attune normalized source JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from attune.training.source_adapters import (
    adapt_common_voice,
    adapt_crema,
    adapt_dcase,
    adapt_fsd50k,
    load_jsonl,
    write_source_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dcase-train", type=Path)
    parser.add_argument("--dcase-test", type=Path)
    parser.add_argument("--dcase-cache", type=Path)
    parser.add_argument("--fsd", type=Path)
    parser.add_argument("--fsd-cache", type=Path)
    parser.add_argument("--crema", type=Path)
    parser.add_argument("--crema-cache", type=Path)
    parser.add_argument("--common-voice", type=Path)
    parser.add_argument("--common-voice-cache", type=Path)
    parser.add_argument(
        "--common-voice-split",
        choices=("source", "train", "development", "sealed_test"),
        default="sealed_test",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--auxiliary-negative-controls",
        action="store_true",
        help=(
            "Opt into v0.2 speech-negative supervision: Common Voice is negative for "
            "event presence/styles and CREMA-D for discrete event presence only."
        ),
    )
    parser.add_argument(
        "--normalized-source",
        type=Path,
        action="append",
        default=[],
        help="Additional already-normalized source JSONL (for example VocalSound)",
    )
    arguments = parser.parse_args()
    rows = []
    for source in arguments.normalized_source:
        rows.extend(load_jsonl(source))
    if arguments.dcase_train:
        if not arguments.dcase_cache:
            parser.error("--dcase-cache is required with DCASE manifests")
        rows.extend(
            adapt_dcase(load_jsonl(arguments.dcase_train), cache_root=arguments.dcase_cache)
        )
    if arguments.dcase_test:
        if not arguments.dcase_cache:
            parser.error("--dcase-cache is required with DCASE manifests")
        rows.extend(
            adapt_dcase(
                load_jsonl(arguments.dcase_test),
                cache_root=arguments.dcase_cache,
                sealed=True,
            )
        )
    if arguments.fsd:
        if not arguments.fsd_cache:
            parser.error("--fsd-cache is required with --fsd")
        rows.extend(adapt_fsd50k(load_jsonl(arguments.fsd), cache_root=arguments.fsd_cache))
    if arguments.crema:
        if not arguments.crema_cache:
            parser.error("--crema-cache is required with --crema")
        rows.extend(
            adapt_crema(
                load_jsonl(arguments.crema),
                cache_root=arguments.crema_cache,
                auxiliary_negative_controls=arguments.auxiliary_negative_controls,
            )
        )
    if arguments.common_voice:
        if not arguments.common_voice_cache:
            parser.error("--common-voice-cache is required with --common-voice")
        rows.extend(
            adapt_common_voice(
                load_jsonl(arguments.common_voice),
                cache_root=arguments.common_voice_cache,
                split=(
                    None
                    if arguments.common_voice_split == "source"
                    else arguments.common_voice_split
                ),
                auxiliary_negative_controls=arguments.auxiliary_negative_controls,
            )
        )
    if not rows:
        parser.error("at least one source manifest is required")
    write_source_rows(arguments.output, rows)
    print(f"Wrote {len(rows)} normalized source rows to {arguments.output}")


if __name__ == "__main__":
    main()
