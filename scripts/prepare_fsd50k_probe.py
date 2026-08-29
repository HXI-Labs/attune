#!/usr/bin/env python3
"""Prepare a bounded clip-disjoint FSD50K pool for the frozen encoder probe."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from prepare_licence_clean_inspection import (
    ALLOWED_CLIP_LICENCES,
    FSD50K_REVISION,
    _fsd_candidates,
    convert_to_pcm16,
    digest,
    download_with_retry,
    ensure_fsd_metadata,
    wav_metadata,
)

from attune.models.fsd50k_probe import SOURCE_TO_PROBE_LABEL

TRAIN_PER_CLASS = 64
VALIDATION_PER_CLASS = 16
DEFAULT_MANIFEST = Path("data/manifests/fsd50k-frozen-probe.jsonl")
DEFAULT_PROVENANCE = Path("data/manifests/fsd50k-frozen-probe.provenance.json")
DEFAULT_CACHE = Path("data/raw/fsd50k-frozen-probe")
DEFAULT_METADATA_CACHE = Path("data/raw/licence-clean-inspection")
DEFAULT_INSPECTION_MANIFEST = Path("data/manifests/licence-clean-inspection.jsonl")


class PreparationError(RuntimeError):
    """Raised when the probe pool would violate its bounded data contract."""


def inspection_ids(path: Path) -> set[str]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise PreparationError(f"cannot read inspection manifest {path}: {error}") from error
    ids = {
        str(row["clip_id"]).removeprefix("fsd50k-")
        for row in rows
        if row.get("source_dataset") == "FSD50K"
    }
    if len(ids) != 100:
        raise PreparationError(f"expected 100 held-out FSD50K inspection clips, found {len(ids)}")
    return ids


def select_probe_rows(
    candidates: list[dict[str, Any]],
    held_out_ids: set[str],
    *,
    train_per_class: int = TRAIN_PER_CLASS,
    validation_per_class: int = VALIDATION_PER_CLASS,
) -> list[dict[str, Any]]:
    """Select balanced deterministic rows after excluding all inspection clips."""
    if train_per_class < 1 or validation_per_class < 1:
        raise ValueError("train and validation quotas must be positive")
    rows: list[dict[str, Any]] = []
    quota = train_per_class + validation_per_class
    for source_class, probe_label in SOURCE_TO_PROBE_LABEL.items():
        eligible = sorted(
            (
                row
                for row in candidates
                if row["source_label"] == source_class
                and str(row["freesound_id"]) not in held_out_ids
            ),
            key=lambda row: (int(row["freesound_id"]), row["source_split"]),
        )
        if len(eligible) < quota:
            raise PreparationError(
                f"{source_class} has only {len(eligible)} eligible non-inspection clips; "
                f"{quota} required"
            )
        for index, candidate in enumerate(eligible[:quota]):
            rows.append(
                {
                    **candidate,
                    "partition": "train" if index < train_per_class else "validation",
                    "probe_label": probe_label,
                }
            )
    return rows


def fetch_row(
    selected: dict[str, Any],
    cache_root: Path,
    *,
    timeout_s: float,
    retries: int,
    hf_token: str | None,
) -> dict[str, Any]:
    clip_id = str(selected["freesound_id"])
    split = selected["source_split"]
    target = cache_root / "fsd50k" / f"{clip_id}.wav"
    if not target.is_file():
        source = cache_root / "_downloads" / f"{clip_id}.wav"
        download_with_retry(
            (
                "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/"
                f"{FSD50K_REVISION}/clips/{split}/{clip_id}.wav"
            ),
            source,
            timeout_s=timeout_s,
            retries=retries,
            token=hf_token,
        )
        convert_to_pcm16(source, target)
        source.unlink(missing_ok=True)
    duration, sample_rate, channels, sample_width = wav_metadata(target)
    licence_id, attribution_required = ALLOWED_CLIP_LICENCES[selected["licence_url"]]
    return {
        "attribution": (
            f'Freesound clip {clip_id} "{selected["title"]}" uploaded by '
            f"{selected['uploader']}; {licence_id}. FSD50K curation by Fonseca et al.; "
            "dataset annotations CC BY 4.0."
        ),
        "cache_path": f"fsd50k/{clip_id}.wav",
        "channels": channels,
        "clip_id": f"fsd50k-{clip_id}",
        "duration_s": round(duration, 6),
        "labels": selected["labels"],
        "licence": {
            "clip_identifier": licence_id,
            "clip_url": selected["licence_url"],
            "attribution_required": attribution_required,
            "dataset_curation": "CC-BY-4.0",
        },
        "partition": selected["partition"],
        "probe_label": selected["probe_label"],
        "sample_rate_hz": sample_rate,
        "sample_width_bytes": sample_width,
        "sha256": digest(target),
        "source_class": selected["source_label"],
        "source_dataset": "FSD50K",
        "source_split": split,
        "transport": {
            "repository": "Fhrozen/FSD50k",
            "revision": FSD50K_REVISION,
            "path": f"clips/{split}/{clip_id}.wav",
        },
        "uploader": selected["uploader"],
    }


def verify(rows: list[dict[str, Any]], cache_root: Path, held_out_ids: set[str]) -> None:
    expected = {("train", label): TRAIN_PER_CLASS for label in SOURCE_TO_PROBE_LABEL.values()}
    expected.update(
        {("validation", label): VALIDATION_PER_CLASS for label in SOURCE_TO_PROBE_LABEL.values()}
    )
    counts = Counter((row["partition"], row["probe_label"]) for row in rows)
    if counts != expected:
        raise PreparationError(f"unexpected partition counts: {dict(counts)}")
    ids = [str(row["clip_id"]).removeprefix("fsd50k-") for row in rows]
    if len(ids) != len(set(ids)):
        raise PreparationError("duplicate clips in FSD50K probe manifest")
    if set(ids) & held_out_ids:
        raise PreparationError("an inspection clip appears in probe training or validation")
    for row in rows:
        if row["licence"]["clip_identifier"] not in {"CC0-1.0", "CC-BY-3.0"}:
            raise PreparationError(f"forbidden clip licence: {row['clip_id']}")
        if len(set(row["labels"]) & set(SOURCE_TO_PROBE_LABEL)) != 1:
            raise PreparationError(f"cross-target clip in probe manifest: {row['clip_id']}")
        path = cache_root / row["cache_path"]
        if not path.is_file() or digest(path) != row["sha256"]:
            raise PreparationError(f"missing or changed probe audio: {path}")
        _duration, rate, channels, width = wav_metadata(path)
        if (rate, channels, width) != (16_000, 1, 2):
            raise PreparationError(f"probe audio is not mono 16 kHz PCM16: {path}")


def write_outputs(rows: list[dict[str, Any]], manifest: Path, provenance: Path) -> None:
    rows.sort(key=lambda row: (row["partition"], row["probe_label"], row["clip_id"]))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    licence_counts = Counter(row["licence"]["clip_identifier"] for row in rows)
    payload = {
        "schema_version": 1,
        "manifest": str(manifest),
        "manifest_sha256": digest(manifest),
        "row_count": len(rows),
        "partition_counts": dict(sorted(Counter(row["partition"] for row in rows).items())),
        "class_counts": dict(sorted(Counter(row["probe_label"] for row in rows).items())),
        "clip_licence_counts": dict(sorted(licence_counts.items())),
        "selection_rule": (
            "From official FSD50K metadata, retain clips with CC0 or CC BY audio and "
            "exactly one target class; exclude all 100 inspection IDs; take the first "
            "80 numeric IDs per class, assigning 64 to train and 16 to validation."
        ),
        "split_boundary": (
            "Clip-disjoint. FSD50K has no speaker identity; uploader is not treated as speaker."
        ),
        "audio_fetch": (
            "Individual pinned transport URLs only; no full audio archive was downloaded."
        ),
    }
    provenance.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--metadata-cache", type=Path, default=DEFAULT_METADATA_CACHE)
    parser.add_argument("--inspection-manifest", type=Path, default=DEFAULT_INSPECTION_MANIFEST)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    held_out = inspection_ids(arguments.inspection_manifest)
    if arguments.download:
        import os

        metadata = ensure_fsd_metadata(
            arguments.metadata_cache, arguments.timeout, arguments.retries
        )
        selected = select_probe_rows(_fsd_candidates(metadata), held_out)
        rows = []
        for index, row in enumerate(selected, 1):
            print(
                f"FSD50K probe {index}/{len(selected)}: {row['freesound_id']}",
                flush=True,
            )
            rows.append(
                fetch_row(
                    row,
                    arguments.cache_dir,
                    timeout_s=arguments.timeout,
                    retries=arguments.retries,
                    hf_token=os.environ.get("HF_TOKEN"),
                )
            )
        write_outputs(rows, arguments.manifest, arguments.provenance)
    else:
        try:
            rows = [
                json.loads(line)
                for line in arguments.manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"error: cannot read {arguments.manifest}: {error}") from error
    try:
        verify(rows, arguments.cache_dir, held_out)
    except PreparationError as error:
        raise SystemExit(f"error: {error}") from error
    print(f"Verified {len(rows)} bounded FSD50K probe clips in {arguments.cache_dir}")


if __name__ == "__main__":
    main()
