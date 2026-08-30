#!/usr/bin/env python3
"""Build a checksum-pinned manifest for an Attune Hugging Face upload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attune.integrity import file_digest


def parse_mapping(value: str) -> tuple[str, str]:
    source, separator, destination = value.partition("=")
    if not separator or not source or not destination:
        raise argparse.ArgumentTypeError("file mappings must use SOURCE=DESTINATION")
    return source, destination


def build_bundle_manifest(
    *,
    root: Path,
    release_name: str,
    gate_report: str,
    mappings: list[tuple[str, str]],
) -> dict[str, object]:
    root = root.resolve()
    if not release_name.strip():
        raise ValueError("release name must not be empty")
    if not mappings:
        raise ValueError("at least one file mapping is required")
    destinations: set[str] = set()
    records = []
    for source_name, destination in mappings:
        if destination.startswith("/") or ".." in Path(destination).parts:
            raise ValueError(f"unsafe Hugging Face destination: {destination}")
        if destination in destinations:
            raise ValueError(f"duplicate Hugging Face destination: {destination}")
        destinations.add(destination)
        source = (root / source_name).resolve()
        try:
            relative_source = source.relative_to(root)
        except ValueError as error:
            raise ValueError(f"bundle source is outside repository: {source_name}") from error
        if not source.is_file():
            raise FileNotFoundError(f"bundle source is missing: {source}")
        records.append(
            {
                "source": relative_source.as_posix(),
                "destination": destination,
                "bytes": source.stat().st_size,
                "sha256": file_digest(source),
            }
        )
    gate_path = (root / gate_report).resolve()
    if gate_path not in {(root / record["source"]).resolve() for record in records}:
        raise ValueError("gate report must be included in the file mappings")
    return {
        "schema_version": "1.0",
        "release_name": release_name,
        "gate_report": gate_path.relative_to(root).as_posix(),
        "files": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--release-name", required=True)
    parser.add_argument("--gate-report", required=True)
    parser.add_argument("--file", type=parse_mapping, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = build_bundle_manifest(
        root=arguments.root,
        release_name=arguments.release_name,
        gate_report=arguments.gate_report,
        mappings=arguments.file,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
