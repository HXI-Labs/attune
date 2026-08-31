from __future__ import annotations

import numpy as np

from attune.affect_mlp_adapter import (
    evaluate_affect_mlp,
    fit_affect_mlp_adapter,
    predict_affect_mlp,
)


def row(value: float, target: int, dataset: str) -> dict[str, object]:
    distribution = [0.0] * 8
    distribution[target] = 1.0
    return {
        "dataset_id": dataset,
        "ood_embedding": [value] * 32,
        "affect_logits": [value if index == target else -value for index in range(8)],
        "vad_prediction": [value / 3] * 3,
        "affect_distribution": distribution,
    }


def test_affect_mlp_is_probabilistic_and_stays_disabled() -> None:
    train = [
        row(value, target, dataset)
        for dataset in ("a", "b")
        for target, value in ((0, -2.0), (1, 2.0))
        for _ in range(3)
    ]
    development = [
        row(value, target, dataset)
        for dataset in ("a", "b")
        for target, value in ((0, -1.5), (1, 1.5))
    ]
    adapter = fit_affect_mlp_adapter(
        train,
        development,
        hidden_sizes=(8, 4),
        dropout=0.0,
        maximum_epochs=80,
        patience=15,
    )
    probabilities = predict_affect_mlp(adapter, development)
    report = evaluate_affect_mlp(adapter, development)

    assert adapter.deployment_enabled is False
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert report["macro_f1"] == 1.0
