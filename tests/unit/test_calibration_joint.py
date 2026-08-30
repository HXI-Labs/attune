from __future__ import annotations

import numpy as np

from attune.calibration_joint import fit_runtime_calibration
from attune.inference.onnx_backend import _affect_probabilities, _sigmoid


def test_joint_calibration_fits_complete_runtime_bundle() -> None:
    rng = np.random.default_rng(4)
    rows = []
    for index in range(40):
        event_target = np.zeros((5, 7), dtype=float)
        event_target[:, index % 7] = 1.0
        style_target = np.zeros(2, dtype=float)
        style_target[index % 2] = 1.0
        affect_target = np.zeros(8, dtype=float)
        affect_target[index % 8] = 1.0
        rows.append(
            {
                "event_logits": (event_target * 5 - 2.5 + rng.normal(0, 0.2, (5, 7))).tolist(),
                "event_targets": event_target.tolist(),
                "event_presence_logits": (event_target[0] * 5 - 2.5).tolist(),
                "event_presence_targets": event_target[0].tolist(),
                "style_logits": (style_target * 5 - 2.5).tolist(),
                "style_targets": style_target.tolist(),
                "affect_logits": (affect_target * 5 - 2.5).tolist(),
                "affect_distribution": affect_target.tolist(),
                "ood_embedding": rng.normal(0, 0.1, 4).tolist(),
                "ood_logit": float(-3 if index < 36 else 3),
                "is_ood": index >= 36,
            }
        )

    calibration = fit_runtime_calibration(rows)

    assert set(calibration.event_thresholds) == {
        "laugh",
        "sob",
        "scream",
        "sigh",
        "cough",
        "throat_clear",
        "sneeze",
    }
    assert set(calibration.style_thresholds) == {"shouting", "whispering"}
    assert set(calibration.event_presence_thresholds) == set(calibration.event_thresholds)
    assert set(calibration.localized_event_labels) == set(calibration.event_thresholds)
    assert calibration.localized_event_min_confidence == 0.98
    assert calibration.event_presence_enabled_labels == []
    assert calibration.style_enabled_labels == []
    assert len(calibration.affect_bias) == 8
    assert len(calibration.ood_centroid or []) == 4
    assert calibration.ood_distance_scale > 0
    assert calibration.ood_available is True


def test_localized_event_thresholds_reject_development_speech_controls() -> None:
    rng = np.random.default_rng(8)
    rows = []
    for index in range(16):
        event_target = np.zeros((5, 7), dtype=float)
        event_target[:, index % 7] = 1.0
        style_target = np.zeros(2, dtype=float)
        style_target[index % 2] = 1.0
        affect_target = np.zeros(8, dtype=float)
        affect_target[index % 8] = 1.0
        rows.append(
            {
                "event_logits": (event_target * 5 - 2.5).tolist(),
                "event_targets": event_target.tolist(),
                "event_presence_logits": (event_target[0] * 5 - 2.5).tolist(),
                "event_presence_targets": event_target[0].tolist(),
                "style_logits": (style_target * 5 - 2.5).tolist(),
                "style_targets": style_target.tolist(),
                "affect_logits": (affect_target * 5 - 2.5).tolist(),
                "affect_distribution": affect_target.tolist(),
                "ood_embedding": rng.normal(0, 0.1, 4).tolist(),
                "ood_logit": float(-3 if index < 14 else 3),
                "is_ood": index >= 14,
            }
        )
    control_logits = np.full((5, 7), -8.0)
    control_logits[:, 0] = 4.0
    rows.append(
        {
            "reference_transcript": "ordinary speech",
            "event_logits": control_logits.tolist(),
        }
    )

    calibration = fit_runtime_calibration(rows)
    control_probability = float(_sigmoid(4.0 / calibration.event_temperature))

    assert calibration.event_thresholds["laugh"] > control_probability


def test_affect_calibration_corrects_class_prior_bias() -> None:
    rng = np.random.default_rng(21)
    rows = []
    for index in range(80):
        target = np.zeros(8, dtype=float)
        target[index % 2] = 1.0
        logits = rng.normal(0, 0.1, 8)
        logits[index % 2] += 2.0
        logits[0] += 1.2
        rows.append(
            {
                "event_logits": np.full((2, 7), -2.0).tolist(),
                "event_targets": np.zeros((2, 7)).tolist(),
                "event_presence_logits": [-2.0] * 7,
                "event_presence_targets": [0.0] * 7,
                "style_logits": [-2.0] * 2,
                "style_targets": [0.0] * 2,
                "affect_logits": logits.tolist(),
                "affect_distribution": target.tolist(),
                "ood_embedding": rng.normal(0, 0.1, 4).tolist(),
                "ood_logit": float(-3 if index < 72 else 3),
                "is_ood": index >= 72,
            }
        )

    calibration = fit_runtime_calibration(rows)
    probabilities = _affect_probabilities(
        np.asarray([row["affect_logits"] for row in rows]), calibration
    )

    assert calibration.affect_bias[0] < calibration.affect_bias[1]
    assert np.mean(probabilities.argmax(axis=1) == np.arange(80) % 2) > 0.95
