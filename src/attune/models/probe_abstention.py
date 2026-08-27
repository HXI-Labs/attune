"""Validation-selected abstention for frozen closed-set linear probes."""

from __future__ import annotations

from typing import Any, Literal

AbstentionMethod = Literal["max_softmax", "energy"]
ABSTENTION_METHODS: tuple[AbstentionMethod, ...] = ("max_softmax", "energy")


def confidence_scores(logits: Any, method: AbstentionMethod, torch: Any) -> Any:
    """Return an ID score where larger values are always more likely in-domain."""
    if method == "max_softmax":
        return torch.softmax(logits, dim=1).max(dim=1).values
    if method == "energy":
        # Energy is -logsumexp(logits). Returning its negative keeps one common
        # ``score >= threshold`` decision rule for both supported methods.
        return torch.logsumexp(logits, dim=1)
    raise ValueError(f"unsupported abstention method: {method}")


def accepts(score: float, threshold: float) -> bool:
    """Apply the checkpointed inclusive abstention boundary."""
    return score >= threshold


def calibrate_abstention(
    *,
    id_logits: Any,
    id_targets: Any,
    ood_logits: Any,
    label_count: int,
    torch: Any,
) -> dict[str, Any]:
    """Compare score methods and choose a threshold using validation data only."""
    if len(id_logits) == 0 or len(ood_logits) == 0:
        raise ValueError("abstention calibration requires non-empty ID and OOD validation")
    comparisons = {
        method: _calibrate_method(
            method=method,
            id_logits=id_logits,
            id_targets=id_targets,
            ood_logits=ood_logits,
            label_count=label_count,
            torch=torch,
        )
        for method in ABSTENTION_METHODS
    }
    selected_method = max(
        ABSTENTION_METHODS,
        key=lambda method: _selection_key(comparisons[method]["selected"]),
    )
    selected = comparisons[selected_method]["selected"]
    return {
        "method": selected_method,
        "threshold": selected["threshold"],
        "score_rule": "emit when score >= threshold; otherwise abstain",
        "score_definition": _score_definition(selected_method),
        "selection_metric": (
            "validation micro-F1 over one expected ID label per in-domain clip and "
            "an empty expected set per OOD clip; ties prefer ID macro-F1, lower OOD "
            "false-positive rate, then higher ID coverage"
        ),
        "validation": selected,
        "method_comparison": {
            method: {
                "selected_threshold": result["selected"]["threshold"],
                "validation": result["selected"],
                "threshold_sweep": result["threshold_sweep"],
            }
            for method, result in comparisons.items()
        },
    }


def evaluate_threshold(
    *,
    logits: Any,
    targets: Any,
    ood_logits: Any,
    method: AbstentionMethod,
    threshold: float,
    label_count: int,
    torch: Any,
) -> dict[str, Any]:
    """Measure one abstention operating point."""
    id_scores = confidence_scores(logits, method, torch).tolist()
    ood_scores = confidence_scores(ood_logits, method, torch).tolist()
    predictions = logits.argmax(dim=1).tolist()
    target_values = targets.tolist()
    emitted = [accepts(float(score), threshold) for score in id_scores]
    ood_emitted = [accepts(float(score), threshold) for score in ood_scores]

    per_class_f1: list[float] = []
    true_positive = false_positive = false_negative = 0
    for label_index in range(label_count):
        class_tp = sum(
            emit and target == prediction == label_index
            for target, prediction, emit in zip(
                target_values, predictions, emitted, strict=True
            )
        )
        class_fp = sum(
            emit and prediction == label_index and target != label_index
            for target, prediction, emit in zip(
                target_values, predictions, emitted, strict=True
            )
        )
        class_fn = sum(
            target == label_index and (not emit or prediction != label_index)
            for target, prediction, emit in zip(
                target_values, predictions, emitted, strict=True
            )
        )
        denominator = 2 * class_tp + class_fp + class_fn
        per_class_f1.append(2 * class_tp / denominator if denominator else 0.0)

    for target, prediction, emit in zip(target_values, predictions, emitted, strict=True):
        if not emit:
            false_negative += 1
        elif target == prediction:
            true_positive += 1
        else:
            false_positive += 1
            false_negative += 1
    false_positive += sum(ood_emitted)
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "threshold": threshold,
        "id_macro_f1": sum(per_class_f1) / label_count,
        "id_accuracy": true_positive / len(target_values),
        "id_coverage": sum(emitted) / len(emitted),
        "id_abstentions": len(emitted) - sum(emitted),
        "ood_false_positive_rate": sum(ood_emitted) / len(ood_emitted),
        "ood_false_positives": sum(ood_emitted),
        "all_prediction_micro_f1": (
            2 * true_positive / denominator if denominator else 0.0
        ),
        "counts": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
    }


def checkpoint_abstention(payload: dict[str, Any]) -> tuple[AbstentionMethod, float]:
    """Validate and return a checkpoint's mandatory abstention configuration."""
    config = payload.get("abstention")
    if not isinstance(config, dict):
        raise RuntimeError("probe checkpoint has no abstention configuration")
    method = config.get("method")
    if method not in ABSTENTION_METHODS:
        raise RuntimeError(f"probe checkpoint has unsupported abstention method: {method!r}")
    threshold = config.get("threshold")
    if not isinstance(threshold, int | float):
        raise RuntimeError("probe checkpoint has an invalid abstention threshold")
    return method, float(threshold)


def _calibrate_method(
    *,
    method: AbstentionMethod,
    id_logits: Any,
    id_targets: Any,
    ood_logits: Any,
    label_count: int,
    torch: Any,
) -> dict[str, Any]:
    all_scores = [
        *confidence_scores(id_logits, method, torch).tolist(),
        *confidence_scores(ood_logits, method, torch).tolist(),
    ]
    unique = sorted({float(score) for score in all_scores})
    epsilon = max(1e-7, (unique[-1] - unique[0]) * 1e-7)
    thresholds = [unique[0] - epsilon, *unique, unique[-1] + epsilon]
    rows = [
        evaluate_threshold(
            logits=id_logits,
            targets=id_targets,
            ood_logits=ood_logits,
            method=method,
            threshold=threshold,
            label_count=label_count,
            torch=torch,
        )
        for threshold in thresholds
    ]
    selected_index, selected = max(
        enumerate(rows),
        key=lambda indexed: (*_selection_key(indexed[1]), indexed[1]["threshold"]),
    )
    compact_indices = sorted(
        {
            0,
            len(rows) - 1,
            max(0, selected_index - 1),
            selected_index,
            min(len(rows) - 1, selected_index + 1),
        }
    )
    sweep = [
        {**rows[index], "selected": index == selected_index} for index in compact_indices
    ]
    return {"selected": selected, "threshold_sweep": sweep}


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["all_prediction_micro_f1"]),
        float(row["id_macro_f1"]),
        -float(row["ood_false_positive_rate"]),
        float(row["id_coverage"]),
    )


def _score_definition(method: AbstentionMethod) -> str:
    if method == "max_softmax":
        return "maximum closed-set softmax probability"
    return "negative energy = logsumexp(logits); energy itself is -score"
