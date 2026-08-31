#!/usr/bin/env python3
"""Create auditable evidence for the required human hostile-speech regression."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from attune.integrity import file_digest


def _normalized_transcript(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def build_report(
    *,
    audio_path: Path,
    inference_path: Path,
    model_path: Path,
    calibration_path: Path,
    expected_transcript: str,
    human_recording_confirmed: bool,
    speaker_consent_confirmed: bool,
) -> dict[str, Any]:
    inference = json.loads(inference_path.read_text())
    attune_output = inference.get("result", inference)
    transcript = attune_output.get("transcript", {})
    observed_text = transcript.get("text")
    confidence = transcript.get("confidence")
    events = attune_output.get("events")
    styles = attune_output.get("styles")
    if not isinstance(observed_text, str) or not isinstance(confidence, int | float):
        raise ValueError("inference output is missing transcript text or confidence")
    if not isinstance(events, list) or not isinstance(styles, list):
        raise ValueError("inference output is missing event or style arrays")

    checks = [
        {
            "name": "human_recording_confirmed",
            "passed": human_recording_confirmed,
            "observed": human_recording_confirmed,
            "requirement": "must be a fresh human recording",
        },
        {
            "name": "speaker_consent_confirmed",
            "passed": speaker_consent_confirmed,
            "observed": speaker_consent_confirmed,
            "requirement": "speaker consent must be confirmed",
        },
        {
            "name": "transcript_exact_after_normalization",
            "passed": _normalized_transcript(observed_text)
            == _normalized_transcript(expected_transcript),
            "observed": observed_text,
            "requirement": expected_transcript,
        },
        {
            "name": "transcript_confidence",
            "passed": float(confidence) >= 0.90,
            "observed": float(confidence),
            "requirement": ">= 0.90",
        },
        {
            "name": "no_vocal_event_hallucination",
            "passed": not events,
            "observed": events,
            "requirement": "no event for an event-free recording",
        },
        {
            "name": "no_vocal_style_hallucination",
            "passed": not styles,
            "observed": styles,
            "requirement": "no style for an ordinary-voice recording",
        },
    ]
    return {
        "schema_version": "1.0",
        "passed": all(check["passed"] for check in checks),
        "audio": {"path": str(audio_path), "sha256": file_digest(audio_path)},
        "inference": {"path": str(inference_path), "sha256": file_digest(inference_path)},
        "model": {"path": str(model_path), "sha256": file_digest(model_path)},
        "calibration": {
            "path": str(calibration_path),
            "sha256": file_digest(calibration_path),
        },
        "human_recording_confirmed": human_recording_confirmed,
        "speaker_consent_confirmed": speaker_consent_confirmed,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--expected-transcript",
        default="I hate you, I hate you so much, never call me again.",
    )
    parser.add_argument("--human-recording-confirmed", action="store_true")
    parser.add_argument("--speaker-consent-confirmed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = build_report(
        audio_path=arguments.audio,
        inference_path=arguments.inference,
        model_path=arguments.model,
        calibration_path=arguments.calibration,
        expected_transcript=arguments.expected_transcript,
        human_recording_confirmed=arguments.human_recording_confirmed,
        speaker_consent_confirmed=arguments.speaker_consent_confirmed,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
