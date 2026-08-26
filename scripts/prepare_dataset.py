#!/usr/bin/env python3
"""Fetch and verify the Phase 1 inspection-set cache.

This entry point handles data preparation only. It does not train or download
model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path("data/manifests/inspection-set.jsonl")
DEFAULT_CACHE = Path("data/raw/inspection-set")
REQUIRED_FIELDS = {
    "clip_id",
    "source_dataset",
    "source_filename",
    "speaker_id",
    "split",
    "duration_s",
    "sha256",
    "licence",
    "attribution",
    "intended_attune_labels",
    "acted_status",
    "notes",
    "cache_path",
    "fetch",
}
ALLOWED_DATASETS = {"VocalSound", "CREMA-D"}


class PreparationError(RuntimeError):
    """Raised when a manifest or local cache cannot be prepared safely."""


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Return a streaming file digest."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load rows and enforce source and speaker-partition invariants."""
    rows: list[dict[str, Any]] = []
    seen_clip_ids: set[str] = set()
    speaker_splits: dict[str, str] = {}

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise PreparationError(f"cannot read manifest {path}: {error}") from error

    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise PreparationError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(row, dict):
            raise PreparationError(f"{path}:{line_number}: row must be a JSON object")

        missing = sorted(REQUIRED_FIELDS - row.keys())
        if missing:
            raise PreparationError(f"{path}:{line_number}: missing fields: {', '.join(missing)}")
        if row["source_dataset"] not in ALLOWED_DATASETS:
            raise PreparationError(
                f"{path}:{line_number}: unsupported source_dataset {row['source_dataset']!r}"
            )
        if row["clip_id"] in seen_clip_ids:
            raise PreparationError(f"{path}:{line_number}: duplicate clip_id {row['clip_id']!r}")
        seen_clip_ids.add(row["clip_id"])

        previous_split = speaker_splits.setdefault(row["speaker_id"], row["split"])
        if previous_split != row["split"]:
            raise PreparationError(
                f"speaker {row['speaker_id']!r} crosses splits "
                f"{previous_split!r} and {row['split']!r}"
            )
        rows.append(row)

    if not rows:
        raise PreparationError(f"manifest {path} contains no rows")
    return rows


def safe_cache_path(cache_root: Path, relative_path: str) -> Path:
    """Resolve a manifest cache path without permitting directory traversal."""
    root = cache_root.resolve()
    target = (root / relative_path).resolve()
    if not target.is_relative_to(root):
        raise PreparationError(f"cache_path escapes cache root: {relative_path!r}")
    return target


def download(url: str, target: Path, timeout_s: float) -> None:
    """Download a URL atomically to a local path."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part")
    request = urllib.request.Request(url, headers={"User-Agent": "attune-phase1/1.0"})
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout_s) as response,
            temporary.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        temporary.replace(target)
    except (OSError, urllib.error.URLError) as error:
        temporary.unlink(missing_ok=True)
        raise PreparationError(f"download failed for {url}: {error}") from error


def ensure_archive(fetch: dict[str, Any], archive_dir: Path, timeout_s: float) -> Path:
    """Download and validate one transport archive."""
    archive = safe_cache_path(archive_dir, fetch["archive_filename"])
    checksum = fetch.get("archive_checksum")
    if (
        archive.exists()
        and checksum
        and file_digest(archive, checksum["algorithm"]) != checksum["value"]
    ):
        raise PreparationError(
            f"transport archive checksum mismatch: {archive}; delete it and retry"
        )
    if not archive.exists():
        download(fetch["url"], archive, timeout_s)
        if checksum and file_digest(archive, checksum["algorithm"]) != checksum["value"]:
            archive.unlink(missing_ok=True)
            raise PreparationError(f"downloaded transport archive checksum mismatch: {archive}")
    return archive


def fetch_row(
    row: dict[str, Any],
    target: Path,
    archive_dir: Path,
    timeout_s: float,
) -> None:
    """Fetch one direct file or extract one exact archive member."""
    fetch = row["fetch"]
    fetch_type = fetch.get("type")
    if fetch_type == "url":
        download(fetch["url"], target, timeout_s)
        return
    if fetch_type != "tar_member":
        raise PreparationError(f"{row['clip_id']}: unsupported fetch type {fetch_type!r}")

    archive = ensure_archive(fetch, archive_dir, timeout_s)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part")
    try:
        with tarfile.open(archive, "r") as bundle:
            member = bundle.getmember(fetch["archive_member"])
            if not member.isfile():
                raise PreparationError(
                    f"{row['clip_id']}: archive member is not a file: {member.name}"
                )
            source = bundle.extractfile(member)
            if source is None:
                raise PreparationError(
                    f"{row['clip_id']}: cannot read archive member {member.name}"
                )
            with source, temporary.open("wb") as handle:
                shutil.copyfileobj(source, handle, length=1024 * 1024)
        temporary.replace(target)
    except (KeyError, OSError, tarfile.TarError) as error:
        temporary.unlink(missing_ok=True)
        raise PreparationError(
            f"{row['clip_id']}: cannot extract {fetch['archive_member']} from {archive}: {error}"
        ) from error


def selected_rows(
    rows: Iterable[dict[str, Any]],
    splits: set[str] | None,
) -> list[dict[str, Any]]:
    """Apply an optional split filter."""
    return [row for row in rows if splits is None or row["split"] in splits]


def prepare(
    manifest: Path,
    cache_root: Path,
    *,
    allow_downloads: bool,
    splits: set[str] | None = None,
    timeout_s: float = 120.0,
) -> int:
    """Fetch missing files when allowed, then verify all selected SHA-256 hashes."""
    rows = selected_rows(load_manifest(manifest), splits)
    if not rows:
        requested = ", ".join(sorted(splits or set()))
        raise PreparationError(f"manifest has no rows for requested split(s): {requested}")

    missing: list[tuple[dict[str, Any], Path]] = []
    for row in rows:
        target = safe_cache_path(cache_root, row["cache_path"])
        if target.exists():
            actual = file_digest(target)
            if actual != row["sha256"]:
                raise PreparationError(
                    f"{row['clip_id']}: SHA-256 mismatch for {target}; "
                    "remove the file and rerun with --download"
                )
        else:
            missing.append((row, target))

    if missing and not allow_downloads:
        examples = "\n".join(f"  - {target}" for _, target in missing[:10])
        remainder = len(missing) - min(10, len(missing))
        suffix = f"\n  ... and {remainder} more" if remainder else ""
        raise PreparationError(
            f"{len(missing)} manifest files are missing:\n{examples}{suffix}\n"
            "Rerun with --download to fetch the documented local inspection subset."
        )

    archive_dir = cache_root / "_archives"
    for index, (row, target) in enumerate(missing, 1):
        print(f"[{index}/{len(missing)}] fetching {row['clip_id']}")
        fetch_row(row, target, archive_dir, timeout_s)
        actual = file_digest(target)
        if actual != row["sha256"]:
            target.unlink(missing_ok=True)
            raise PreparationError(
                f"{row['clip_id']}: downloaded SHA-256 {actual} does not match manifest"
            )

    print(f"Verified {len(rows)} clips in {cache_root}")
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and verify the Phase 1 inspection-set audio cache."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download missing clips; without this flag the command only verifies",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=("inspect", "held_out_speakers"),
        help="prepare only this split (repeatable); defaults to both",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="per-request timeout seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        prepare(
            args.manifest,
            args.cache_dir,
            allow_downloads=args.download,
            splits=set(args.split) if args.split else None,
            timeout_s=args.timeout,
        )
    except PreparationError as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
