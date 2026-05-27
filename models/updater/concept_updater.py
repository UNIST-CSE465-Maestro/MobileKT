"""
Concept Knowledge Updater (CU)

After the student responds to question q_t, update CS_j for all related concepts.

Formula
-------
ΔCS_j  = Tanh( CU(X_t, I_j) )                 where X_t = AE[a_t] + q'_t
CS_j  ← σ( logit(CS_j) + α_j · ΔCS_j )
I_j   = 0   (reset concept timer)

Implementation — One-hot bmm (scales to any max_c)
---------------------------------------------------
Instead of scattering per-concept updates one-by-one, we:
  1. Compute logit-space delta for each related concept:  α_j · ΔCS_j  → (B, max_c)
  2. Build a one-hot matrix:  (B, max_c, K)
  3. Project via bmm:  logit_delta_full = delta @ one_hot  → (B, K)
  4. Apply:  new_mastery = σ( logit(old_mastery) + logit_delta_full )

This is:
  - Fully differentiable (no scatter_, no detach)
  - Loop-free — works for any max_c, even max_c = K
  - Handles duplicate concept IDs (deltas accumulate additively in logit space)
  - Memory: O(B × max_c × K) — fine for K ≤ ~10K on GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..knowledge.states import encode_interval


class ConceptUpdater(nn.Module):
    def __init__(self, d: int, hidden: int, dropout: float = 0.2):
        super().__init__()
        # CU: [X_t || encode(I_j)] → ΔCS_j  (scalar per concept)
        # Input dim = 2d (acquired knowledge + time interval encoding).
        self.cu = nn.Sequential(
            nn.Linear(2 * d, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self._d = d

    def forward(
        self,
        cs: torch.Tensor,           # (B, K, 2)   concept state
        X_t: torch.Tensor,          # (B, d)      acquired knowledge = AE[a_t] + q'_t
        alpha: torch.Tensor,        # (B, max_c)  concept attention weights
        concept_ids: torch.Tensor,  # (B, max_c)  related concept indices (-1 = pad)
        response: torch.Tensor,     # (B,)        student response  {0, 1}
    ) -> torch.Tensor:
        """
        Returns:
            cs_new: (B, K, 2)  updated concept state (fully differentiable)

        Monotonicity guarantee:
            CU predicts |ΔCS| (magnitude only, always ≥ 0).
            X_t = AE[a_t] + q' is used for magnitude — AE carries information
            about the response event type (correct vs incorrect can have
            different update magnitudes even for the same question).
            Direction is determined explicitly by the response:
                correct  (response=1) → mastery increases
                incorrect(response=0) → mastery decreases
        """
        B, max_c = concept_ids.shape
        K = cs.shape[1]
        d = self._d
        eps = 1e-6
        device = cs.device

        pad_mask = (concept_ids == -1)           # (B, max_c)
        safe_ids = concept_ids.clamp(min=0)      # (B, max_c)  -1 → 0 (dummy write)

        # ── Read current mastery and intervals for related concepts ────────
        mastery   = cs[:, :, 0]                  # (B, K)  — gradient flows through
        intervals = cs[:, :, 1].detach()         # (B, K)  — timer, no grad needed

        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, max_c)
        rel_mastery   = mastery[batch_idx, safe_ids]    # (B, max_c)
        rel_intervals = intervals[batch_idx, safe_ids]  # (B, max_c)

        # ── Compute magnitude of update (always ≥ 0) ──────────────────────
        # CU predicts how much to change, response determines direction.
        enc = encode_interval(rel_intervals, d)                    # (B, max_c, d)
        X_expanded = X_t.unsqueeze(1).expand(B, max_c, d)
        cu_input = torch.cat([X_expanded, enc], dim=-1)           # (B, max_c, 2d)
        delta_mag = torch.abs(torch.tanh(
            self.cu(cu_input).squeeze(-1)                    # (B, max_c)
        ))

        # ── Apply direction from response: correct=+1, incorrect=-1 ───────
        sign  = (2 * response.float() - 1).unsqueeze(1)     # (B, 1)
        delta = sign * delta_mag                             # (B, max_c)

        # Zero out padded positions before projection
        logit_delta_rel = (alpha * delta).masked_fill(pad_mask, 0.0)  # (B, max_c)

        # ── Project to full concept space via one-hot bmm ─────────────────
        # one_hot: (B, max_c, K)  — 1 at the concept position, 0 elsewhere
        # Padded positions have safe_ids=0 but logit_delta_rel=0, so they
        # add zero to concept 0 — a safe no-op.
        one_hot = F.one_hot(safe_ids, num_classes=K).float()  # (B, max_c, K)

        # logit_delta_full[b, k] = Σ_c  logit_delta_rel[b, c] * one_hot[b, c, k]
        logit_delta_full = torch.bmm(
            logit_delta_rel.unsqueeze(1),   # (B, 1, max_c)
            one_hot,                         # (B, max_c, K)
        ).squeeze(1)                         # (B, K)

        # ── Apply update in logit space ───────────────────────────────────
        logit_m = torch.log((mastery + eps) / (1 - mastery + eps))  # (B, K)
        new_mastery = torch.sigmoid(logit_m + logit_delta_full)      # (B, K)

        # ── Update timers: reset for touched concepts, keep+1 for others ──
        # Build a boolean mask (B, K): True where concept was actually updated
        touched = (one_hot * (~pad_mask).float().unsqueeze(-1)).sum(dim=1) > 0  # (B, K)

        new_timer = torch.where(
            touched,
            torch.zeros_like(intervals),   # reset to 0 for updated concepts
            intervals,                      # keep current for untouched concepts
        )                                   # (B, K)

        cs_new = torch.stack([new_mastery, new_timer], dim=-1)  # (B, K, 2)
        return cs_new
