from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from attune.integrity import file_digest

SCRIPT = Path(__file__).parents[2] / "scripts" / "build_training_lineage.py"
SPEC = importlib.util.spec_from_file_location("build_training_lineage", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_training_source_digest_binds_names_order_and_contents(monkeypatch) -> None:
    monkeypatch.setattr(MODULE.training_module, "_TRAINING_SOURCE_FILES", ("b.py", "a.py"))
    sources = {"a.py": b"first", "b.py": b"second"}

    digest = MODULE.training_source_digest(sources.__getitem__)

    assert digest == "36988ae9c8ab8de2c0928d89a8f23e25a7e603d1730ada07b16cd9de6b43c67a"


def test_lineage_requires_an_explanation_for_concurrent_source_changes(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    report = tmp_path / "report.json"
    checkpoint = tmp_path / "model.pt"
    config = tmp_path / "config.json"
    weights = tmp_path / "weights.json"
    manifest.write_text("{}\n")
    checkpoint.write_bytes(b"checkpoint")
    config.write_text("{}")
    weights.write_text("{}")
    report.write_text(
        json.dumps(
            {
                "manifest_sha256": file_digest(manifest),
                "training_source_sha256": "report-time-digest",
                "trainer": {"device": "cpu"},
            }
        )
    )
    monkeypatch.setattr(MODULE, "resolve_revision", lambda root, revision: "a" * 40)
    monkeypatch.setattr(MODULE, "git_source_digest", lambda root, revision: "start-digest")

    with pytest.raises(ValueError, match="concurrent-change-reason"):
        MODULE.build_lineage(
            root=tmp_path,
            report_path=report,
            checkpoint_path=checkpoint,
            manifest_path=manifest,
            config_path=config,
            loss_weights_path=weights,
            training_revision="HEAD",
            command="python train.py",
            concurrent_change_reason=None,
        )
