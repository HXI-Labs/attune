from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[2] / "scripts/check_event_mixtures_candidate.py"
SPEC = importlib.util.spec_from_file_location("check_event_mixtures_candidate", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def reports(*, recall: float = 0.30, false_positive_clips: int = 2):
    wesr = {
        "localized_event_presence_macro_f1": 0.34,
        "localized_event_presence_recall": recall,
        "localized_event_presence_false_positive_rate": 0.06,
    }
    regression = {
        "event_segments": {"macro_f1": 0.60},
        "speech_controls": {
            "localized_event_false_positive_clips": false_positive_clips,
        },
    }
    return wesr, regression


def test_event_mixtures_candidate_requires_every_external_gate() -> None:
    wesr, regression = reports()

    gates = CHECK.check_candidate(wesr, regression, {"passed": True})

    assert all(gate["passed"] for gate in gates)


def test_event_mixtures_candidate_rejects_low_external_recall() -> None:
    wesr, regression = reports(recall=0.299)

    gates = CHECK.check_candidate(wesr, regression, {"passed": True})

    assert not next(gate for gate in gates if gate["name"] == "wesr_temporal_presence_recall")[
        "passed"
    ]


def test_event_mixtures_candidate_rejects_control_false_positives() -> None:
    wesr, regression = reports(false_positive_clips=3)

    gates = CHECK.check_candidate(wesr, regression, {"passed": True})

    assert not next(
        gate for gate in gates if gate["name"] == "opened_speech_control_false_positive_clips"
    )["passed"]
