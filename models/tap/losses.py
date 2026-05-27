"""Loss functions for time-aware probing."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_binary_cross_entropy(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary cross entropy for hard or soft labels.

    Args:
        pred: probabilities in ``[0, 1]``.
        target: hard or soft targets in ``[0, 1]``.
        weight: optional per-sample weight.
        reduction: ``none``, ``mean``, or ``sum``.
    """
    pred = pred.clamp(min=1e-7, max=1.0 - 1e-7)
    loss = F.binary_cross_entropy(pred, target.float(), reduction="none")
    if weight is not None:
        loss = loss * weight.float()

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        if weight is not None:
            return loss.sum() / weight.float().sum().clamp_min(1.0)
        return loss.mean()
    raise ValueError("reduction must be one of: none, mean, sum")


def direction_margin_loss(
    mastery_before: torch.Tensor,
    mastery_after: torch.Tensor,
    responses: torch.Tensor,
    margin: float = 0.02,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Encourage mastery to move in the response-consistent direction.

    Correct responses should increase concept mastery; incorrect responses
    should decrease it. This loss is meant as a weak regularizer, not as a hard
    replacement for future-correctness supervision.
    """
    signed_delta = torch.where(
        responses.float() > 0.5,
        mastery_after - mastery_before,
        mastery_before - mastery_after,
    )
    loss = torch.relu(float(margin) - signed_delta)
    if weight is not None:
        loss = loss * weight.float()

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        if weight is not None:
            return loss.sum() / weight.float().sum().clamp_min(1.0)
        return loss.mean()
    raise ValueError("reduction must be one of: none, mean, sum")


def smoothness_loss(
    mastery_prev: torch.Tensor,
    mastery_next: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Penalize abrupt changes when no direct concept evidence is present."""
    loss = (mastery_next - mastery_prev).abs()
    if weight is not None:
        loss = loss * weight.float()

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        if weight is not None:
            return loss.sum() / weight.float().sum().clamp_min(1.0)
        return loss.mean()
    raise ValueError("reduction must be one of: none, mean, sum")


def forgetting_monotonic_loss(
    mastery_short_gap: torch.Tensor,
    mastery_long_gap: torch.Tensor,
    margin: float = 0.0,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Encourage mastery to not increase when only elapsed time increases.

    This is an Ebbinghaus-inspired weak regularizer. It should be applied to
    paired predictions that share the same hidden state and concept but differ
    in timer gap.
    """
    loss = torch.relu(mastery_long_gap - mastery_short_gap + float(margin))
    if weight is not None:
        loss = loss * weight.float()

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        if weight is not None:
            return loss.sum() / weight.float().sum().clamp_min(1.0)
        return loss.mean()
    raise ValueError("reduction must be one of: none, mean, sum")


def time_aware_probe_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    mastery_before: torch.Tensor | None = None,
    mastery_after: torch.Tensor | None = None,
    direction_responses: torch.Tensor | None = None,
    lambda_direction: float = 0.0,
    direction_margin: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Compute the standard TAP objective and return named components."""
    future = soft_binary_cross_entropy(pred, target, sample_weight, reduction="mean")
    total = future
    direction = pred.new_tensor(0.0)

    has_direction = (
        lambda_direction > 0
        and mastery_before is not None
        and mastery_after is not None
        and direction_responses is not None
    )
    if has_direction:
        direction = direction_margin_loss(
            mastery_before,
            mastery_after,
            direction_responses,
            margin=direction_margin,
            reduction="mean",
        )
        total = total + float(lambda_direction) * direction

    return {
        "loss": total,
        "future": future.detach(),
        "direction": direction.detach(),
    }
