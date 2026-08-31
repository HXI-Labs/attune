"""Masked multi-corpus objectives for the unified Attune model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from attune.models.joint import JointOutput


@dataclass
class JointTargets:
    ctc_targets: Tensor | None = None
    ctc_target_lengths: Tensor | None = None
    ctc_example_mask: Tensor | None = None
    event_targets: Tensor | None = None
    event_target_mask: Tensor | None = None
    event_start_targets: Tensor | None = None
    event_end_targets: Tensor | None = None
    event_presence_targets: Tensor | None = None
    event_presence_example_mask: Tensor | None = None
    style_targets: Tensor | None = None
    style_example_mask: Tensor | None = None
    affect_distribution: Tensor | None = None
    affect_example_mask: Tensor | None = None
    vad_targets: Tensor | None = None
    vad_target_mask: Tensor | None = None
    ood_targets: Tensor | None = None
    pair_ids: Tensor | None = None


@dataclass(frozen=True)
class LossWeights:
    ctc: float = 1.0
    event: float = 1.0
    boundary: float = 0.5
    event_presence: float = 1.0
    weak_event_presence: float = 0.0
    style: float = 1.0
    affect: float = 1.0
    vad: float = 1.0
    ood: float = 0.5
    paired: float = 0.2
    counterfactual: float = 0.0
    style_positive_alpha: float = 0.75


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def focal_binary_loss(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    gamma: float = 2.0,
    positive_alpha: float = 0.75,
) -> Tensor:
    """Class-balanced focal BCE for sparse multi-label supervision."""

    if not 0.0 < positive_alpha < 1.0:
        raise ValueError("positive_alpha must be between zero and one")
    base = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    correct_probability = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    class_weight = targets * positive_alpha + (1.0 - targets) * (1.0 - positive_alpha)
    return _masked_mean(base * (1.0 - correct_probability).pow(gamma) * class_weight, mask)


def soft_dice_loss(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    probabilities = torch.sigmoid(logits) * mask
    targets = targets * mask
    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    per_example_class = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    annotated = mask.any(dim=1)
    if not annotated.any():
        return logits.sum() * 0.0
    return per_example_class[annotated].mean()


def weak_temporal_presence_logits(
    frame_logits: Tensor,
    frame_mask: Tensor,
    *,
    temperature: float = 0.5,
) -> Tensor:
    """Pool frame logits for weak utterance-level event supervision.

    Log-mean-exp is a smooth multiple-instance objective: positive labels push
    the most plausible frames up, while negative labels suppress spurious
    peaks across the utterance. Padding never contributes to the pool.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if frame_logits.ndim != 3 or frame_mask.shape != frame_logits.shape[:2]:
        raise ValueError("frame logits and mask have incompatible shapes")
    valid_frames = frame_mask.sum(dim=1)
    if (valid_frames == 0).any():
        raise ValueError("each example must contain at least one valid frame")
    masked_logits = frame_logits.masked_fill(~frame_mask.unsqueeze(-1), -torch.inf)
    pooled = torch.logsumexp(masked_logits / temperature, dim=1)
    return temperature * (pooled - valid_frames.log().unsqueeze(-1))


def concordance_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    losses = []
    for dimension in range(prediction.shape[-1]):
        active = mask[:, dimension].bool()
        if active.sum() < 2:
            continue
        x = prediction[active, dimension]
        y = target[active, dimension]
        covariance = ((x - x.mean()) * (y - y.mean())).mean()
        ccc = (
            2.0
            * covariance
            / (
                x.var(unbiased=False)
                + y.var(unbiased=False)
                + (x.mean() - y.mean()).square()
                + 1e-8
            )
        )
        losses.append(1.0 - ccc)
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def paired_delivery_loss(
    embeddings: Tensor, pair_ids: Tensor, affect_distribution: Tensor | None = None
) -> Tensor:
    """Separate examples sharing text but carrying different delivery labels."""
    normalized = nn.functional.normalize(embeddings, dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    same_pair = pair_ids.unsqueeze(0).eq(pair_ids.unsqueeze(1))
    diagonal = torch.eye(pair_ids.numel(), device=pair_ids.device, dtype=torch.bool)
    selected = same_pair & ~diagonal & pair_ids.unsqueeze(0).ge(0)
    if affect_distribution is not None:
        labels = affect_distribution.argmax(dim=-1)
        selected = selected & labels.unsqueeze(0).ne(labels.unsqueeze(1))
    if not selected.any():
        return embeddings.sum() * 0.0
    return nn.functional.relu(similarity[selected] - 0.25).mean()


def counterfactual_affect_loss(
    logits: Tensor,
    pair_ids: Tensor,
    affect_distribution: Tensor,
    affect_mask: Tensor | None = None,
    *,
    margin: float = 0.25,
) -> Tensor:
    """Align same-text logit changes with listener-distribution changes."""

    valid = pair_ids.ge(0)
    if affect_mask is not None:
        valid = valid & affect_mask.bool()
    same_pair = pair_ids.unsqueeze(0).eq(pair_ids.unsqueeze(1))
    upper_triangle = torch.triu(torch.ones_like(same_pair, dtype=torch.bool), diagonal=1)
    selected = same_pair & upper_triangle & valid.unsqueeze(0) & valid.unsqueeze(1)
    if not selected.any():
        return logits.sum() * 0.0

    predicted_change = logits.unsqueeze(1) - logits.unsqueeze(0)
    target_change = affect_distribution.unsqueeze(1) - affect_distribution.unsqueeze(0)
    predicted_change = predicted_change[selected]
    target_change = target_change[selected]
    importance = target_change.abs()
    if not importance.any():
        return logits.sum() * 0.0
    ranking_loss = nn.functional.softplus(margin - target_change.sign() * predicted_change)
    return (ranking_loss * importance).sum() / importance.sum()


def compute_joint_loss(
    output: JointOutput,
    targets: JointTargets,
    weights: LossWeights | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    weights = weights or LossWeights()
    losses: dict[str, Tensor] = {}
    zero = output.affect_logits.sum() * 0.0
    if targets.ctc_targets is not None and targets.ctc_target_lengths is not None:
        ctc_logits = output.ctc_logits
        ctc_lengths = output.acoustic_lengths
        if targets.ctc_example_mask is not None:
            ctc_logits = ctc_logits[targets.ctc_example_mask]
            ctc_lengths = ctc_lengths[targets.ctc_example_mask]
        losses["ctc"] = nn.functional.ctc_loss(
            ctc_logits.log_softmax(dim=-1).transpose(0, 1),
            targets.ctc_targets,
            ctc_lengths,
            targets.ctc_target_lengths,
            blank=0,
            zero_infinity=True,
        )
    if targets.event_targets is not None:
        event_mask = output.frame_mask.unsqueeze(-1)
        if targets.event_target_mask is not None:
            event_mask = event_mask & targets.event_target_mask.bool()
        losses["event"] = focal_binary_loss(
            output.event_logits, targets.event_targets, event_mask
        ) + soft_dice_loss(output.event_logits, targets.event_targets, event_mask)
        if targets.event_start_targets is not None and targets.event_end_targets is not None:
            start = focal_binary_loss(
                output.event_start_logits, targets.event_start_targets, event_mask
            )
            end = focal_binary_loss(output.event_end_logits, targets.event_end_targets, event_mask)
            losses["boundary"] = (start + end) / 2.0
    if targets.event_presence_targets is not None:
        presence_mask = targets.event_presence_example_mask
        if presence_mask is None:
            presence_mask = torch.ones_like(targets.event_presence_targets, dtype=torch.bool)
        losses["event_presence"] = focal_binary_loss(
            output.event_presence_logits,
            targets.event_presence_targets,
            presence_mask,
        )
        losses["weak_event_presence"] = focal_binary_loss(
            weak_temporal_presence_logits(output.event_logits, output.frame_mask),
            targets.event_presence_targets,
            presence_mask,
        )
    if targets.style_targets is not None:
        mask = targets.style_example_mask
        if mask is None:
            mask = torch.ones_like(targets.style_targets, dtype=torch.bool)
        losses["style"] = focal_binary_loss(
            output.style_logits,
            targets.style_targets,
            mask,
            positive_alpha=weights.style_positive_alpha,
        )
    if targets.affect_distribution is not None:
        per_example = -(targets.affect_distribution * output.affect_logits.log_softmax(-1)).sum(-1)
        mask = targets.affect_example_mask
        if mask is None:
            mask = torch.ones_like(per_example, dtype=torch.bool)
        losses["affect"] = _masked_mean(per_example, mask)
    if targets.vad_targets is not None:
        mask = targets.vad_target_mask
        if mask is None:
            mask = torch.ones_like(targets.vad_targets, dtype=torch.bool)
        mse = _masked_mean((output.vad - targets.vad_targets).square(), mask)
        losses["vad"] = mse + concordance_loss(output.vad, targets.vad_targets, mask)
    if targets.ood_targets is not None:
        losses["ood"] = nn.functional.binary_cross_entropy_with_logits(
            output.ood_logit, targets.ood_targets.float()
        )
    if targets.pair_ids is not None:
        losses["paired"] = paired_delivery_loss(
            output.affect_embedding,
            targets.pair_ids,
            targets.affect_distribution,
        )
        if targets.affect_distribution is not None:
            losses["counterfactual"] = counterfactual_affect_loss(
                output.affect_logits,
                targets.pair_ids,
                targets.affect_distribution,
                targets.affect_example_mask,
            )
    weighted = {name: loss * getattr(weights, name) for name, loss in losses.items()}
    total = sum(weighted.values(), start=zero)
    return total, {**losses, "total": total}
