"""Future-label construction utilities for time-aware probing."""

from __future__ import annotations

import math

import torch


def build_future_correctness_labels(
    concept_ids: torch.Tensor,
    responses: torch.Tensor,
    lengths: torch.Tensor | None = None,
    n_concepts: int | None = None,
    horizon: int = 50,
    tau: float = 20.0,
    mode: str = "decayed_average",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create temporally local future correctness labels.

    Args:
        concept_ids: ``(B, S, max_c)`` concept IDs. ``-1`` and ``0`` are
            treated as padding.
        responses: ``(B, S)`` binary responses.
        lengths: optional ``(B,)`` valid sequence lengths.
        n_concepts: maximum concept ID. If omitted, inferred from data.
        horizon: number of future steps considered after timestep ``t``.
        tau: temporal scale for ``decayed_average`` mode.
        mode: one of ``next``, ``average``, or ``decayed_average``.

    Returns:
        ``labels``: ``(B, S, K+1)`` soft labels in ``[0, 1]``.
        ``mask``: ``(B, S, K+1)`` true where a concept has at least one future
            occurrence inside the horizon.
        ``support``: ``(B, S, K+1)`` number of future occurrences used.

    Important:
        No future occurrence inside ``horizon`` means "unobserved", not
        incorrect. Those entries have ``mask=False`` and should be excluded
        from probe training/evaluation.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if mode not in {"next", "average", "decayed_average"}:
        raise ValueError("mode must be one of: next, average, decayed_average")
    if mode == "decayed_average" and tau <= 0:
        raise ValueError("tau must be positive for decayed_average mode")
    if concept_ids.dim() != 3:
        raise ValueError("concept_ids must have shape (B, S, max_c)")
    if responses.dim() != 2:
        raise ValueError("responses must have shape (B, S)")

    concept_ids_cpu = concept_ids.detach().cpu().long()
    responses_cpu = responses.detach().cpu().float()
    B, S, _ = concept_ids_cpu.shape
    if lengths is None:
        lengths_cpu = torch.full((B,), S, dtype=torch.long)
    else:
        lengths_cpu = lengths.detach().cpu().long().clamp(min=0, max=S)

    if n_concepts is None:
        valid = concept_ids_cpu[concept_ids_cpu > 0]
        n_concepts = int(valid.max().item()) if valid.numel() else 0
    K1 = n_concepts + 1

    weighted_correct = torch.zeros(B, S, K1, dtype=torch.float32)
    weight_sum = torch.zeros(B, S, K1, dtype=torch.float32)
    support = torch.zeros(B, S, K1, dtype=torch.float32)

    for b in range(B):
        seq_len = int(lengths_cpu[b].item())
        for t in range(seq_len):
            end = min(seq_len, t + horizon + 1)
            seen_next: set[int] = set()
            for j in range(t + 1, end):
                gap = j - t
                if mode == "decayed_average":
                    weight = math.exp(-gap / tau)
                else:
                    weight = 1.0

                row_concepts = _unique_valid_concepts(concept_ids_cpu[b, j], n_concepts)
                response = float(responses_cpu[b, j].item())
                for c in row_concepts:
                    if mode == "next" and c in seen_next:
                        continue
                    seen_next.add(c)
                    weighted_correct[b, t, c] += weight * response
                    weight_sum[b, t, c] += weight
                    support[b, t, c] += 1.0

    mask = weight_sum > 0
    labels = torch.zeros_like(weighted_correct)
    labels[mask] = weighted_correct[mask] / weight_sum[mask]
    return (
        labels.to(device=concept_ids.device),
        mask.to(device=concept_ids.device),
        support.to(device=concept_ids.device),
    )


def gather_labeled_probe_samples(
    hidden_states: torch.Tensor,
    timer_features: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    max_samples: int | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Flatten dense tensors into paired probe samples.

    Args:
        hidden_states: either ``(B, S, D)`` for global states or
            ``(B, S, K+1, D)`` for concept-aligned states.
        timer_features: ``(B, S, K+1, F)``.
        labels: ``(B, S, K+1)``.
        label_mask: ``(B, S, K+1)``.
        max_samples: optional random subsample size.
        generator: optional torch random generator for reproducible sampling.

    Returns:
        Dictionary with ``hidden_state``, ``concept_ids``, ``timer_features``,
        ``labels``, ``batch_index``, and ``timestep``.
    """
    if timer_features.dim() != 4:
        raise ValueError("timer_features must have shape (B, S, K+1, F)")
    if labels.shape != label_mask.shape:
        raise ValueError("labels and label_mask must have identical shape")
    if labels.shape != timer_features.shape[:3]:
        raise ValueError("labels must match timer_features on B, S, and K+1")

    index = torch.nonzero(label_mask, as_tuple=False)
    if max_samples is not None and index.shape[0] > max_samples:
        perm = torch.randperm(index.shape[0], generator=generator, device=index.device)
        index = index[perm[:max_samples]]

    b_idx = index[:, 0]
    t_idx = index[:, 1]
    c_idx = index[:, 2]

    if hidden_states.dim() == 3:
        if hidden_states.shape[:2] != labels.shape[:2]:
            raise ValueError("global hidden_states must match labels on B and S")
        hidden = hidden_states[b_idx, t_idx]
    elif hidden_states.dim() == 4:
        if hidden_states.shape[:3] != labels.shape:
            raise ValueError("concept hidden_states must match labels on B, S, and K+1")
        hidden = hidden_states[b_idx, t_idx, c_idx]
    else:
        raise ValueError("hidden_states must be 3D or 4D")

    return {
        "hidden_state": hidden,
        "concept_ids": c_idx.long(),
        "timer_features": timer_features[b_idx, t_idx, c_idx],
        "labels": labels[b_idx, t_idx, c_idx],
        "batch_index": b_idx,
        "timestep": t_idx,
    }


def _unique_valid_concepts(row: torch.Tensor, n_concepts: int) -> list[int]:
    concepts = set()
    for raw in row.tolist():
        c = int(raw)
        if 0 < c <= n_concepts:
            concepts.add(c)
    return sorted(concepts)
