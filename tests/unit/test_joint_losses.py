from __future__ import annotations

import torch

from attune.models.joint import JointOutput
from attune.training.losses import (
    JointTargets,
    compute_joint_loss,
    focal_binary_loss,
    soft_dice_loss,
)


def _output() -> JointOutput:
    batch, frames, events = 3, 5, 7
    return JointOutput(
        ctc_logits=torch.randn(batch, frames, 10, requires_grad=True),
        acoustic_lengths=torch.tensor([5, 4, 3]),
        frame_mask=torch.arange(frames).unsqueeze(0) < torch.tensor([5, 4, 3]).unsqueeze(1),
        event_logits=torch.randn(batch, frames, events, requires_grad=True),
        event_start_logits=torch.randn(batch, frames, events, requires_grad=True),
        event_end_logits=torch.randn(batch, frames, events, requires_grad=True),
        event_presence_logits=torch.randn(batch, events, requires_grad=True),
        style_logits=torch.randn(batch, 2, requires_grad=True),
        affect_logits=torch.randn(batch, 8, requires_grad=True),
        vad=torch.tanh(torch.randn(batch, 3, requires_grad=True)),
        ood_logit=torch.randn(batch, requires_grad=True),
        affect_embedding=torch.randn(batch, 16, requires_grad=True),
        ood_embedding=torch.randn(batch, 8, requires_grad=True),
    )


def test_masked_multicorpus_loss_ignores_missing_targets() -> None:
    output = _output()
    event_targets = torch.zeros_like(output.event_logits)
    event_mask = torch.zeros_like(event_targets, dtype=torch.bool)
    event_mask[0] = True
    affect = torch.zeros(3, 8)
    affect[:, 0] = 1.0
    targets = JointTargets(
        event_targets=event_targets,
        event_target_mask=event_mask,
        event_presence_targets=torch.zeros(3, 7),
        event_presence_example_mask=torch.tensor([[True] * 7, [False] * 7, [False] * 7]),
        style_targets=torch.zeros(3, 2),
        style_example_mask=torch.tensor([[True, True], [False, False], [False, False]]),
        affect_distribution=affect,
        affect_example_mask=torch.tensor([False, True, True]),
        vad_targets=torch.zeros(3, 3),
        vad_target_mask=torch.tensor([[False] * 3, [True] * 3, [True] * 3]),
        ood_targets=torch.tensor([0.0, 1.0, 0.0]),
        pair_ids=torch.tensor([1, 1, -1]),
    )
    total, metrics = compute_joint_loss(output, targets)
    total.backward()

    assert total.isfinite()
    assert {
        "event",
        "event_presence",
        "style",
        "affect",
        "vad",
        "ood",
        "paired",
        "total",
    } <= set(metrics)
    assert output.affect_logits.grad is not None


def test_empty_target_bundle_has_differentiable_zero() -> None:
    output = _output()
    total, metrics = compute_joint_loss(output, JointTargets())
    total.backward()
    assert total.item() == 0.0
    assert metrics["total"].item() == 0.0


def test_focal_binary_loss_upweights_sparse_positive_targets() -> None:
    logits = torch.zeros(1, 2)
    mask = torch.ones_like(logits, dtype=torch.bool)

    positive = focal_binary_loss(logits[:, :1], torch.ones(1, 1), mask[:, :1])
    negative = focal_binary_loss(logits[:, 1:], torch.zeros(1, 1), mask[:, 1:])

    assert torch.isclose(positive, negative * 3.0)


def test_dice_loss_is_not_diluted_by_unannotated_examples() -> None:
    logits = torch.zeros(2, 3, 1)
    targets = torch.zeros_like(logits)
    targets[0, 1, 0] = 1.0
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[0] = True

    with_unannotated = soft_dice_loss(logits, targets, mask)
    annotated_only = soft_dice_loss(logits[:1], targets[:1], mask[:1])

    assert torch.isclose(with_unannotated, annotated_only)
