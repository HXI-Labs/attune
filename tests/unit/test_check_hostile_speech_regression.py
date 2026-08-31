from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "check_hostile_speech_regression.py"
SPEC = importlib.util.spec_from_file_location("check_hostile_speech_regression", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _artifact(path: Path, content: str = "artifact") -> Path:
    path.write_text(content)
    return path


def test_human_hostile_regression_records_hashed_passing_evidence(tmp_path: Path) -> None:
    inference = {
        "transcript": {
            "text": "I hate you, I hate you so much—never call me again.",
            "confidence": 0.97,
        },
        "events": [],
        "styles": [],
    }
    inference_path = _artifact(tmp_path / "inference.json", json.dumps(inference))

    report = MODULE.build_report(
        audio_path=_artifact(tmp_path / "recording.wav"),
        inference_path=inference_path,
        model_path=_artifact(tmp_path / "model.onnx"),
        calibration_path=_artifact(tmp_path / "calibration.json"),
        expected_transcript="I hate you, I hate you so much, never call me again.",
        human_recording_confirmed=True,
        speaker_consent_confirmed=True,
    )

    assert report["passed"] is True
    assert len(report["audio"]["sha256"]) == 64
    assert all(check["passed"] for check in report["checks"])


def test_human_hostile_regression_rejects_auxiliary_hallucination(tmp_path: Path) -> None:
    inference = {
        "result": {
            "transcript": {"text": "I hate you", "confidence": 0.99},
            "events": [{"label": "cough"}],
            "styles": [],
        }
    }
    inference_path = _artifact(tmp_path / "inference.json", json.dumps(inference))

    report = MODULE.build_report(
        audio_path=_artifact(tmp_path / "recording.wav"),
        inference_path=inference_path,
        model_path=_artifact(tmp_path / "model.onnx"),
        calibration_path=_artifact(tmp_path / "calibration.json"),
        expected_transcript="I hate you",
        human_recording_confirmed=True,
        speaker_consent_confirmed=True,
    )

    assert report["passed"] is False
    event_check = next(
        check for check in report["checks"] if check["name"] == "no_vocal_event_hallucination"
    )
    assert event_check["passed"] is False
