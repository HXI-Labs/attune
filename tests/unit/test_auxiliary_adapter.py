from __future__ import annotations

import numpy as np

from attune.auxiliary_adapter import (
    evaluate_auxiliary_adapter,
    fit_auxiliary_adapter,
    predict_auxiliary_adapter,
)


def _row(index: int, split: str, positive: bool) -> dict:
    event_targets = [float(positive)] * 7
    style_targets = [float(positive)] * 2
    signal = 2.0 if positive else -2.0
    return {
        "clip_id": f"{split}-{index}",
        "ood_embedding": [signal] + [0.0] * 31,
        "event_presence_logits": [signal] * 7,
        "style_logits": [signal] * 2,
        "event_presence_targets": event_targets,
        "style_targets": style_targets,
        "auxiliary_negative_tasks": [] if positive else ["event_presence", "styles"],
    }


def test_auxiliary_adapter_learns_separable_controls_and_stays_disabled() -> None:
    train = [_row(index, "train", index % 2 == 0) for index in range(40)]
    development = [_row(index, "development", index % 2 == 0) for index in range(20)]
    sealed = [_row(index, "sealed", index % 2 == 0) for index in range(20)]

    adapter = fit_auxiliary_adapter(train, development)
    report = evaluate_auxiliary_adapter(adapter, sealed)

    assert np.isclose(report["event_presence"]["per_label"]["laugh"]["f1"], 1.0)
    assert report["event_presence"]["control_false_positive_clips"] == 0
    assert report["styles"]["per_label"]["shouting"]["f1"] == 1.0
    assert report["styles"]["control_false_positive_clips"] == 0
    assert adapter.event_presence.deployment_enabled_labels == []
    assert adapter.styles.deployment_enabled_labels == []

    predictions = predict_auxiliary_adapter(adapter, sealed)
    assert predictions[0]["event_presence"]["laugh"]["above_threshold"] is True
    assert predictions[1]["event_presence"]["laugh"]["above_threshold"] is False
    assert predictions[0]["event_presence"]["laugh"]["deployment_enabled"] is False
