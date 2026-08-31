#!/usr/bin/env python3
"""Build a deterministic checksum manifest for Attune release artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attune.integrity import file_digest


def build_manifest(root: Path, artifacts: list[Path]) -> dict[str, object]:
    root = root.resolve()
    records = []
    for artifact in sorted({path.resolve() for path in artifacts}):
        if not artifact.is_file():
            raise FileNotFoundError(f"release artifact is missing: {artifact}")
        try:
            relative = artifact.relative_to(root)
        except ValueError as error:
            raise ValueError(f"release artifact is outside root {root}: {artifact}") from error
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": artifact.stat().st_size,
                "sha256": file_digest(artifact),
            }
        )
    return {
        "schema_version": "1.0",
        "hash_algorithm": "sha256",
        "artifacts": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument(
        "--artifact-list",
        type=Path,
        help="UTF-8 file containing one root-relative artifact path per line",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    artifacts = list(arguments.artifact)
    if arguments.artifact_list:
        artifacts.extend(
            arguments.root / line.strip()
            for line in arguments.artifact_list.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not artifacts:
        parser.error("at least one --artifact or --artifact-list entry is required")
    manifest = build_manifest(arguments.root, artifacts)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
