from __future__ import annotations

import numpy as np

from attune.style_adapter import (
    evaluate_whisper_style_adapter,
    fit_whisper_style_adapter,
    predict_whisper_probability,
)


def row(value: float, whisper: bool) -> dict[str, object]:
    return {
        "ood_embedding": [value] * 32,
        "style_logits": [-value, value],
        "style_targets": [0.0, float(whisper)],
    }


def test_whisper_adapter_fits_and_stays_disabled() -> None:
    train = [row(value, value > 0) for value in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)]
    development = [row(value, value > 0) for value in (-1.5, -0.7, 0.7, 1.5)]
    adapter = fit_whisper_style_adapter(train, development)

    probabilities = predict_whisper_probability(adapter, development)
    report = evaluate_whisper_style_adapter(adapter, development)

    assert adapter.deployment_enabled is False
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert report["f1"] == 1.0
