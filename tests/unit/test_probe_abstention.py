import pytest

from attune.models.probe_abstention import (
    calibrate_abstention,
    checkpoint_abstention,
    confidence_scores,
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


def test_checkpoint_abstention_is_mandatory() -> None:
    with pytest.raises(RuntimeError, match="no abstention"):
        checkpoint_abstention({})
