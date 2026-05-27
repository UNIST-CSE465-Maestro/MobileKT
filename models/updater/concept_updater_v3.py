"""
Concept Knowledge Updater v3 (CU3)

Key change from CU v1: difficulty-modulated update scaling (DIMKT-inspired).

Motivation:
  Students learn most efficiently when question difficulty ≈ current mastery
  (zone of proximal development). CU3 scales each concept's update by how
  well the question difficulty matches that concept's mastery:

      diff_norm  = σ(diff_q)                        ∈ [0, 1]
      match_j    = 1 − |mastery_j − diff_norm|      ∈ [0, 1]
      ΔCS_j      = match_j × α_j × sign × |tanh(CU(X_t, I_j))|

  When mastery_j ≈ diff_norm (difficulty matches student level):
    match ≈ 1 → full update (maximum learning signal)
  When mastery_j >> diff_norm (too easy) or mastery_j << diff_norm (too hard):
    match < 1 → reduced update (noisy or trivial signal)

The match multiplier is parameter-free. CU network input remains [X_t ‖ enc(I_j)]
(2d), identical to v1 — no additional parameters introduced.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..knowledge.states import encode_interval


class ConceptUpdaterV3(nn.Module):
    def __init__(self, d: int, hidden: int, dropout: float = 0.2):
        super().__init__()
        self.cu = nn.Sequential(
            nn.Linear(2 * d, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self._d = d

    def forward(
        self,
        cs: torch.Tensor,           # (B, K, 2)   concept state: (mastery, timer)
        X_t: torch.Tensor,          # (B, d)      acquired knowledge = AE[a_t] + q'_t
        alpha: torch.Tensor,        # (B, max_c)  concept attention weights
        concept_ids: torch.Tensor,  # (B, max_c)  related concept indices (-1 = pad)
        response: torch.Tensor,     # (B,)        student response {0, 1}
        diff: torch.Tensor,         # (B,)        question difficulty scalar (unbounded)
    ) -> torch.Tensor:
        """Returns cs_new: (B, K, 2)"""
        B, max_c = concept_ids.shape
        K = cs.shape[1]
        d = self._d
        eps = 1e-6
        device = cs.device

        pad_mask = (concept_ids == -1)
        safe_ids = concept_ids.clamp(min=0)

        mastery = cs[:, :, 0]            # (B, K)
        timers  = cs[:, :, 1].detach()   # (B, K)

        batch_idx   = torch.arange(B, device=device).unsqueeze(1).expand(B, max_c)
        rel_mastery = mastery[batch_idx, safe_ids]   # (B, max_c)
        rel_timers  = timers[batch_idx, safe_ids]    # (B, max_c)

        # ── Base update magnitude (same as v1) ────────────────────────────
        enc       = encode_interval(rel_timers, d)              # (B, max_c, d)
        X_exp     = X_t.unsqueeze(1).expand(B, max_c, d)       # (B, max_c, d)
        cu_input  = torch.cat([X_exp, enc], dim=-1)            # (B, max_c, 2d)
        delta_mag = torch.abs(torch.tanh(
            self.cu(cu_input).squeeze(-1)                       # (B, max_c)
        ))

        # ── Difficulty-match modulation (ZPD) ────────────────────────────
        # diff_norm maps unbounded difficulty to [0,1] for comparison with mastery
        diff_norm = torch.sigmoid(diff).unsqueeze(1)            # (B, 1)
        match     = 1.0 - torch.abs(rel_mastery - diff_norm)   # (B, max_c) ∈ [0, 1]

        # ── Apply sign and accumulate ─────────────────────────────────────
        sign  = (2 * response.float() - 1).unsqueeze(1)        # (B, 1)
        delta = sign * match * delta_mag                        # (B, max_c)
        logit_delta_rel = (alpha * delta).masked_fill(pad_mask, 0.0)

        one_hot = F.one_hot(safe_ids, num_classes=K).float()   # (B, max_c, K)
        logit_delta_full = torch.bmm(
            logit_delta_rel.unsqueeze(1), one_hot
        ).squeeze(1)                                            # (B, K)

        logit_m     = torch.log((mastery + eps) / (1 - mastery + eps))
        new_mastery = torch.sigmoid(logit_m + logit_delta_full)

        touched   = (one_hot * (~pad_mask).float().unsqueeze(-1)).sum(dim=1) > 0
        new_timer = torch.where(touched, torch.zeros_like(timers), timers)

        return torch.stack([new_mastery, new_timer], dim=-1)   # (B, K, 2)
