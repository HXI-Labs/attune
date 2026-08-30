from __future__ import annotations

import json
from pathlib import Path

from attune.training.controls import (
    AuxiliaryControlPolicy,
    apply_control_policy,
    build_control_manifest,
)
from attune.training.data import JointManifestRow


def _row(dataset: str, *, events=None, event_presence=None, styles=None) -> JointManifestRow:
    return JointManifestRow(
        clip_id="clip-1",
        dataset_id=dataset,
        split="train",
        speaker_id="speaker-1",
        feature_path=Path("feature.pt"),
        feature_sha256="a" * 64,
        duration_ms=1000,
        frame_hop_ms=60,
        transcript="ordinary speech",
        events=events,
        event_presence=event_presence,
        styles=styles,
    )


def _policy() -> AuxiliaryControlPolicy:
    return AuxiliaryControlPolicy.model_validate(
        {
            "schema_version": "1.0",
            "policy_id": "fixture-v1",
            "datasets": {
                "speech": {
                    "event_presence": "explicit_negative",
                    "styles": "explicit_negative",
                }
            },
            "default": {
                "event_presence": "preserve_source_annotation",
                "styles": "preserve_source_annotation",
            },
        }
    )


def test_control_policy_distinguishes_empty_targets_from_missing_labels() -> None:
    controlled = apply_control_policy(_row("speech"), _policy())

    assert controlled.event_presence == []
    assert controlled.styles == []
    assert controlled.auxiliary_negative_tasks == ["event_presence", "styles"]


def test_control_policy_refuses_to_overwrite_positive_supervision() -> None:
    try:
        apply_control_policy(_row("speech", event_presence=["laugh"]), _policy())
    except ValueError as error:
        assert "already has positive" in str(error)
    else:
        raise AssertionError("positive supervision was silently overwritten")


def test_control_policy_adds_empty_localized_event_targets() -> None:
    policy = _policy().model_copy(
        update={
            "datasets": {
                "speech": _policy()
                .datasets["speech"]
                .model_copy(update={"localized_events": "explicit_negative"})
            }
        }
    )

    controlled = apply_control_policy(_row("speech"), policy)

    assert controlled.events == []
    assert controlled.auxiliary_negative_tasks == [
        "localized_events",
        "event_presence",
        "styles",
    ]


def test_control_policy_refuses_to_overwrite_localized_events() -> None:
    policy = _policy().model_copy(
        update={
            "datasets": {
                "speech": _policy()
                .datasets["speech"]
                .model_copy(update={"localized_events": "explicit_negative"})
            }
        }
    )

    try:
        apply_control_policy(
            _row(
                "speech",
                events=[{"label": "laugh", "start_ms": 100, "end_ms": 200}],
            ),
            policy,
        )
    except ValueError as error:
        assert "positive localized_events" in str(error)
    else:
        raise AssertionError("localized event supervision was silently overwritten")


def test_control_manifest_records_policy_and_counts(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "source.jsonl"
    output = tmp_path / "output" / "controlled.jsonl"
    source.write_text(_row("speech").model_dump_json() + "\n")

    report = build_control_manifest(source, output, _policy())

    saved = json.loads(output.read_text())
    assert saved["auxiliary_negative_tasks"] == ["event_presence", "styles"]
    assert report["policy_id"] == "fixture-v1"
    assert report["negative_controls"] == {
        "speech:event_presence": 1,
        "speech:styles": 1,
    }
    assert (output.parent / saved["feature_path"]).resolve() == (
        source.parent / "feature.pt"
    ).resolve()
