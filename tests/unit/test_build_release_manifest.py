from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path(__file__).parents[2] / "scripts/build_release_manifest.py"
    specification = importlib.util.spec_from_file_location("build_release_manifest", path)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_release_manifest_is_sorted_and_hashed(tmp_path) -> None:
    script = _load_script()
    second = tmp_path / "b.txt"
    first = tmp_path / "a.txt"
    second.write_text("second")
    first.write_text("first")

    manifest = script.build_manifest(tmp_path, [second, first])

    assert [row["path"] for row in manifest["artifacts"]] == ["a.txt", "b.txt"]
    assert manifest["artifacts"][0]["sha256"] == script.sha256_file(first)
