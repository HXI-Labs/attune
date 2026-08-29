from __future__ import annotations

import numpy as np

from attune.evaluation.joint import evaluate_joint_scores
from attune.inference.onnx_backend import RuntimeCalibration


def test_joint_report_includes_aps_and_all_task_metrics() -> None:
    calibration = RuntimeCalibration(
        event_thresholds={
            label: 0.5
            for label in ("laugh", "sob", "scream", "sigh", "cough", "throat_clear", "sneeze")
        },
        event_presence_thresholds={
            label: 0.5
            for label in ("laugh", "sob", "scream", "sigh", "cough", "throat_clear", "sneeze")
        },
        style_thresholds={"shouting": 0.5, "whispering": 0.5},
        ood_centroid=[0.0] * 4,
        ood_distance_scale=1.0,
    )
    event_target = np.zeros((4, 7))
    event_target[:, 0] = 1
    affect = np.zeros(8)
    affect[3] = 1
    rows = [
        {
            "reference_transcript": "Leave me alone!",
            "predicted_transcript": "leave me alone",
            "event_logits": (event_target * 10 - 5).tolist(),
            "event_targets": event_target.tolist(),
            "event_presence_logits": (event_target[0] * 10 - 5).tolist(),
            "event_presence_targets": event_target[0].tolist(),
            "style_logits": [5, -5],
            "style_targets": [1, 0],
            "affect_logits": (affect * 10 - 5).tolist(),
            "affect_distribution": affect.tolist(),
            "lexical_affect_label": "joy",
            "vad_prediction": [-0.5, 0.8, 0.4],
            "vad_target": [-0.5, 0.8, 0.4],
            "frame_hop_ms": 60.0,
        },
        {
            "reference_transcript": "I am fine",
            "predicted_transcript": "I fine",
            "event_logits": (event_target * 10 - 5).tolist(),
            "event_targets": event_target.tolist(),
            "event_presence_logits": (event_target[0] * 10 - 5).tolist(),
            "event_presence_targets": event_target[0].tolist(),
            "style_logits": [5, -5],
            "style_targets": [1, 0],
            "affect_logits": (affect * 10 - 5).tolist(),
            "affect_distribution": affect.tolist(),
            "lexical_affect_label": "joy",
            "vad_prediction": [-0.4, 0.7, 0.3],
            "vad_target": [-0.4, 0.7, 0.3],
            "frame_hop_ms": 60.0,
        },
    ]

    report = evaluate_joint_scores(rows, calibration)

    assert report["event_frame_f1"]["laugh"] == 1.0
    assert report["event_frame_average_precision"]["laugh"] == 1.0
    assert report["event_frame_macro_f1"] == 1 / 3
    assert report["event_frame_macro_f1_ontology"] == 1 / 7
    assert report["event_localized_labels"] == ["laugh", "cough", "throat_clear"]
    assert report["event_presence_f1"]["laugh"] == 1.0
    assert report["event_presence_average_precision"]["laugh"] == 1.0
    assert report["event_segments"]["per_class"]["laugh"]["segment_f1"] == 1.0
    assert report["style_f1"]["shouting"] == 1.0
    assert report["style_average_precision"]["shouting"] == 1.0
    assert report["acoustic_preference_score"] == 1.0
    assert report["affect_macro_f1"] == 1.0
    assert report["affect_macro_f1_ontology"] == 0.125
    assert report["affect_supported_classes"] == ["anger"]
    assert report["vad_ccc"]["arousal"] > 0.99
    assert report["asr_wer"] == 1 / 6
    assert report["affect_selective_risk_improves"] is True
    assert report["affect_mce"] < 0.01
    assert report["affect_jensen_shannon"] < 0.01
    assert report["affect_risk_coverage"][-1]["coverage"] == 1.0
    assert report["by_dataset"]["unknown"]["clips"] == 2
    assert "by_dataset" not in report["by_dataset"]["unknown"]


def test_speech_control_metrics_match_runtime_suppression_and_span_rules() -> None:
    calibration = RuntimeCalibration(
        event_thresholds={
            label: 0.5
            for label in ("laugh", "sob", "scream", "sigh", "cough", "throat_clear", "sneeze")
        },
        event_presence_thresholds={
            label: 0.5
            for label in ("laugh", "sob", "scream", "sigh", "cough", "throat_clear", "sneeze")
        },
        style_thresholds={"shouting": 0.5, "whispering": 0.5},
    )
    event_logits = np.full((4, 7), -8.0)
    event_logits[1, 0] = 5.0  # One frame is below the minimum two-frame span.
    rows = [
        {
            "reference_transcript": "ordinary speech",
            "predicted_transcript": "ordinary speech",
            "event_logits": event_logits.tolist(),
            "event_presence_logits": [8.0] * 7,
            "style_logits": [8.0, 8.0],
            "frame_hop_ms": 60.0,
        }
    ]

    report = evaluate_joint_scores(rows, calibration)

    assert report["speech_controls"] == {
        "clips": 1,
        "localized_event_false_positive_clips": 0,
        "event_presence_false_positive_clips": 0,
        "style_false_positive_clips": 0,
        "aux_false_positive_rate": 0.0,
        "localized_false_events_per_minute": 0.0,
    }
