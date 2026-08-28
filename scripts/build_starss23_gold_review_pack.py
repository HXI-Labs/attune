#!/usr/bin/env python3
"""Build the STARSS23 first-60s gold-review pack from a PR 22 inspection JSONL.

This writes manifests and provenance only. It does not copy audio, train, or
claim gold. Later tiles (source_window_start_ms != 0) are excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attune.data.starss23_gold_pack import (
    PACK_RELATIVE_PATH,
    PROVENANCE_RELATIVE_PATH,
    build_pack_rows,
    load_jsonl,
    pack_provenance,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inspection-manifest",
        type=Path,
        required=True,
        help="PR 22 starss23-scene-raster-inspection.jsonl (not merged)",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    inspection = load_jsonl(arguments.inspection_manifest)
    rows = build_pack_rows(inspection)
    pack_path = arguments.repo_root / PACK_RELATIVE_PATH
    provenance_path = arguments.repo_root / PROVENANCE_RELATIVE_PATH
    write_jsonl(pack_path, rows)
    provenance_path.write_text(json.dumps(pack_provenance(rows), indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(rows)} first-60s pack rows to {pack_path} "
        f"and provenance to {provenance_path}. Gold remains closed."
    )


if __name__ == "__main__":
    main()
