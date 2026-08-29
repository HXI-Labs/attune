from __future__ import annotations

from attune.affect_adapter import (
    evaluate_affect_adapter,
    fit_affect_adapter,
    predict_affect_adapter,
)


def _row(index: int, label: int) -> dict:
    signal = [0.0] * 32
    signal[label] = 3.0
    affect_logits = [0.0] * 7
    affect_logits[label] = 2.0
    target = [0.0] * 7
    target[label] = 1.0
    return {
        "clip_id": f"clip-{index}",
        "ood_embedding": signal,
        "affect_logits": affect_logits,
        "vad_prediction": [float(label), 0.0, 0.0],
        "affect_distribution": target,
        "lexical_affect_label": "neutral",
    }


def test_affect_adapter_learns_supported_classes_without_enabling_deployment() -> None:
    train = [_row(index, index % 3) for index in range(60)]
    development = [_row(index + 100, index % 3) for index in range(30)]
    adapter = fit_affect_adapter(train, development, candidate_c=(0.1, 1.0))
    report = evaluate_affect_adapter(adapter, development)
    probabilities = predict_affect_adapter(adapter, development)

    assert report["macro_f1"] == 1.0
    assert report["accuracy"] == 1.0
    assert adapter.deployment_enabled is False
    assert probabilities.shape == (30, 8)
    assert (probabilities[:, 3:] == 0).all()
