from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[2] / "scripts/check_berst_affect_candidate.py"
SPEC = importlib.util.spec_from_file_location("check_berst_affect_candidate", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def report(
    macro_f1: float,
    *,
    aps: float = 0.1,
    selective: bool = True,
    first_risk: float = 0.1,
    last_risk: float = 0.4,
    largest_share: float = 0.4,
):
    return {
        "affect_macro_f1": macro_f1,
        "acoustic_preference_score": aps,
        "affect_selective_risk_improves": selective,
        "affect_risk_coverage": [
            {"coverage": 0.1, "risk": first_risk},
            {"coverage": 1.0, "risk": last_risk},
        ],
        "affect_prediction_share": {
            "neutral": largest_share,
            "joy": (1.0 - largest_share) / 2,
            "anger": (1.0 - largest_share) / 2,
        },
    }


def test_berst_affect_candidate_requires_every_gate() -> None:
    gates = CHECK.check_candidate(
        report(0.64),
        report(0.30),
        report(0.40, largest_share=0.45),
        {"passed": True},
    )

    assert all(gate["passed"] for gate in gates)


def test_berst_affect_candidate_rejects_collapsed_external_predictions() -> None:
    ravdess = report(0.42, largest_share=0.46)
    ravdess["affect_prediction_share"] = {"anger": 0.46, "other": 0.54}

    gates = CHECK.check_candidate(
        report(0.68),
        report(0.35),
        ravdess,
        {"passed": True},
    )

    share_gate = next(gate for gate in gates if gate["name"] == "ravdess_largest_prediction_share")
    assert share_gate["value"] == 0.54
    assert not share_gate["passed"]


def test_berst_affect_candidate_requires_risk_to_improve_with_abstention() -> None:
    berst = report(0.35, first_risk=0.4, last_risk=0.3)

    gates = CHECK.check_candidate(
        report(0.68),
        berst,
        report(0.42),
        {"passed": True},
    )

    assert not next(gate for gate in gates if gate["name"] == "berst_selective_risk_improves")[
        "passed"
    ]
