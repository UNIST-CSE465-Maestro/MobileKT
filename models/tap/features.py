"""Past-only timer feature builders for time-aware probing."""

from __future__ import annotations

from collections import deque

import torch


def build_timer_features(
    concept_ids: torch.Tensor,
    responses: torch.Tensor,
    lengths: torch.Tensor | None = None,
    n_concepts: int | None = None,
    recent_window: int = 5,
    unseen_gap: float | None = None,
    prior_correct_rate: float = 0.5,
) -> torch.Tensor:
    """Build concept timer features available after each timestep.

    Args:
        concept_ids: ``(B, S, max_c)`` concept IDs. ``-1`` and ``0`` are
            treated as padding.
        responses: ``(B, S)`` binary responses.
        lengths: optional ``(B,)`` valid sequence lengths.
        n_concepts: maximum concept ID. If omitted, inferred from data.
        recent_window: number of recent attempts used for recent correctness.
        unseen_gap: gap value used for concepts never seen by the current time.
            Defaults to ``S + 1``.
        prior_correct_rate: recent correctness value for unseen concepts.

    Returns:
        ``(B, S, K+1, 3)`` where the last dimension is:

        1. ``log1p(gap_since_last_seen)``
        2. ``log1p(seen_count)``
        3. ``recent_correct_rate``

    Notes:
        Features are measured after processing timestep ``t``. They are valid
        inputs for labels drawn from ``t+1`` onward.
    """
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
    if unseen_gap is None:
        unseen_gap = float(S + 1)

    out = torch.empty(B, S, K1, 3, dtype=torch.float32)

    for b in range(B):
        last_seen = torch.full((K1,), -1, dtype=torch.long)
        seen_count = torch.zeros(K1, dtype=torch.float32)
        recent: list[deque[float]] = [deque(maxlen=recent_window) for _ in range(K1)]
        seq_len = int(lengths_cpu[b].item())

        for t in range(S):
            if t < seq_len:
                row_concepts = _unique_valid_concepts(concept_ids_cpu[b, t], n_concepts)
                response = float(responses_cpu[b, t].item())
                for c in row_concepts:
                    last_seen[c] = t
                    seen_count[c] += 1.0
                    recent[c].append(response)

            gap = torch.where(
                last_seen >= 0,
                torch.full((K1,), t, dtype=torch.float32) - last_seen.float(),
                torch.full((K1,), float(unseen_gap), dtype=torch.float32),
            )
            recent_rate = torch.full((K1,), float(prior_correct_rate), dtype=torch.float32)
            for c in range(1, K1):
                if recent[c]:
                    recent_rate[c] = float(sum(recent[c]) / len(recent[c]))

            out[b, t, :, 0] = torch.log1p(gap)
            out[b, t, :, 1] = torch.log1p(seen_count)
            out[b, t, :, 2] = recent_rate

    return out.to(device=concept_ids.device)


def _unique_valid_concepts(row: torch.Tensor, n_concepts: int) -> list[int]:
    concepts = set()
    for raw in row.tolist():
        c = int(raw)
        if 0 < c <= n_concepts:
            concepts.add(c)
    return sorted(concepts)
