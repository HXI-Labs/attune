import pytest

from attune.models.probe_abstention import (
    calibrate_abstention,
    checkpoint_abstention,
    confidence_scores,
    fit_none_logit_head,
)

torch = pytest.importorskip("torch")


def test_calibration_compares_max_softmax_and_energy_and_rejects_ood() -> None:
    id_logits = torch.tensor([[8.0, 0.0], [0.0, 8.0], [7.0, 0.0], [0.0, 7.0]])
    id_targets = torch.tensor([0, 1, 0, 1])
    ood_logits = torch.tensor([[0.1, 0.0], [0.0, 0.1], [0.0, 0.0]])

    result = calibrate_abstention(
        id_logits=id_logits,
        id_targets=id_targets,
        ood_logits=ood_logits,
        label_count=2,
        torch=torch,
    )

    assert set(result["method_comparison"]) == {"max_softmax", "energy"}
    assert result["validation"]["all_prediction_micro_f1"] == 1.0
    assert result["validation"]["id_macro_f1"] == 1.0
    assert result["validation"]["ood_false_positive_rate"] == 0.0
    selected_rows = result["method_comparison"][result["method"]]["threshold_sweep"]
    assert sum(row["selected"] for row in selected_rows) == 1


def test_energy_score_is_negative_energy() -> None:
    logits = torch.tensor([[1.0, 2.0]])

    score = confidence_scores(logits, "energy", torch)

    assert score.item() == pytest.approx(torch.logsumexp(logits, dim=1).item())


def test_calibration_can_select_genuine_negative_none_logit() -> None:
    closed_id = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    closed_ood = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    none_id = torch.tensor([[8.0, 0.0, -1.0], [0.0, 8.0, -1.0]])
    none_ood = torch.tensor([[0.0, 0.0, 8.0], [0.0, 0.0, 8.0]])

    result = calibrate_abstention(
        id_logits=closed_id,
        id_targets=torch.tensor([0, 1]),
        ood_logits=closed_ood,
        label_count=2,
        torch=torch,
        none_id_logits=none_id,
        none_ood_logits=none_ood,
    )

    assert result["method"] == "none_logit"
    assert isinstance(result["threshold"], float)
    assert result["validation"]["all_prediction_micro_f1"] == 1.0
    assert result["validation"]["ood_false_positive_rate"] == 0.0


def test_checkpoint_abstention_is_mandatory() -> None:
    with pytest.raises(RuntimeError, match="no abstention"):
        checkpoint_abstention({})


def test_none_logit_fit_preserves_closed_set_class_weights() -> None:
    closed = torch.nn.Linear(2, 2)
    original_weight = closed.weight.detach().clone()
    original_bias = closed.bias.detach().clone()

    combined, _state, _history = fit_none_logit_head(
        closed_head=closed,
        train_features=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        train_targets=torch.tensor([0, 1]),
        ood_train_features=torch.tensor([[-2.0, -2.0], [-1.0, -1.0]]),
        validation_features=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        validation_targets=torch.tensor([0, 1]),
        ood_validation_features=torch.tensor([[-2.0, -2.0]]),
        label_count=2,
        learning_rate=1e-2,
        batch_size=2,
        epochs=2,
        patience=2,
        seed=0,
        torch=torch,
    )

    assert torch.equal(combined.weight[:2], original_weight)
    assert torch.equal(combined.bias[:2], original_bias)
