"""
Concept Knowledge Updater v2 (CU2)

Differences from CU v1
----------------------
v1:  CS ∈ R^{K×2}  (mastery, timer)
     Δ = sign × |tanh(MLP([X_t ∥ enc(I_j)]))|
     CS_j ← σ(logit(CS_j) + α_j · Δ)

v2:  CS ∈ R^{K×3}  (mastery, momentum_logit, timer)
     ① Adaptive gate  z_j = σ(W_gate([X_t ∥ enc(I_j) ∥ σ(mom_j)]))
     ② Gated update   CS_j ← σ(logit(CS_j) + α_j · z_j · sign · |Δ|)
     ③ Momentum EMA   mom_j ← β·mom_j + (1−β)·(α_j·logit_delta_j)
        — parameter-free, tracks recent signed logit-space updates

Momentum interpretation
-----------------------
mom_j is stored as a raw logit-scale value (unbounded, initialised to 0).
For display:  σ(mom_j) ∈ (0,1),  0.5 = neutral trend
              > 0.5 = consistently improving
              < 0.5 = consistently declining

Adaptive gate rationale
-----------------------
The gate z_j is conditioned on (X_t, encode(I_j), σ(mom_j)):
  — Concepts visited recently with consistent trend → larger z (accept bigger update)
  — Concepts in flux / not seen for a while → smaller z (conservative update)
This implements per-concept adaptive learning rate without per-concept parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..knowledge.states import encode_interval


class ConceptUpdaterV2(nn.Module):
    def __init__(self, d: int, hidden: int, dropout: float = 0.2, mom_decay: float = 0.85):
        super().__init__()
        self._d = d
        self._mom_decay = mom_decay

        # Base update magnitude (same as v1)
        self.cu = nn.Sequential(
            nn.Linear(2 * d, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # Adaptive gate: conditioned on X_t, recency, and current momentum
        # Input dim = 2d + 1  (X_t:d, enc:d, σ(mom):1)
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d + 1, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 1),
        )

    def forward(
        self,
        cs: torch.Tensor,           # (B, K, 3)   concept state: (mastery, momentum_logit, timer)
        X_t: torch.Tensor,          # (B, d)
        alpha: torch.Tensor,        # (B, max_c)
        concept_ids: torch.Tensor,  # (B, max_c)  (-1 = pad)
        response: torch.Tensor,     # (B,)
    ) -> torch.Tensor:
        """Returns cs_new: (B, K, 3)"""
        B, max_c = concept_ids.shape
        K = cs.shape[1]
        d = self._d
        eps = 1e-6
        device = cs.device

        pad_mask = (concept_ids == -1)
        safe_ids = concept_ids.clamp(min=0)

        mastery  = cs[:, :, 0]           # (B, K)
        momentum = cs[:, :, 1]           # (B, K)  raw logit-scale
        timers   = cs[:, :, 2].detach()  # (B, K)

        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, max_c)
        rel_mastery  = mastery[batch_idx, safe_ids]   # (B, max_c)
        rel_momentum = momentum[batch_idx, safe_ids]  # (B, max_c)
        rel_timers   = timers[batch_idx, safe_ids]    # (B, max_c)

        enc = encode_interval(rel_timers, d)                      # (B, max_c, d)
        X_expanded = X_t.unsqueeze(1).expand(B, max_c, d)        # (B, max_c, d)

        # ── Base update magnitude (same as v1) ────────────────────────────
        cu_input = torch.cat([X_expanded, enc], dim=-1)           # (B, max_c, 2d)
        delta_mag = torch.abs(torch.tanh(
            self.cu(cu_input).squeeze(-1)                          # (B, max_c)
        ))

        # ── Adaptive gate (NEW) ───────────────────────────────────────────
        mom_normalized = torch.sigmoid(rel_momentum).unsqueeze(-1)  # (B, max_c, 1)
        gate_input = torch.cat([X_expanded, enc, mom_normalized], dim=-1)  # (B, max_c, 2d+1)
        z = torch.sigmoid(self.gate_net(gate_input).squeeze(-1))   # (B, max_c)

        # ── Gated signed delta ────────────────────────────────────────────
        sign  = (2 * response.float() - 1).unsqueeze(1)           # (B, 1)
        delta = sign * z * delta_mag                               # (B, max_c)
        logit_delta_rel = (alpha * delta).masked_fill(pad_mask, 0.0)

        # ── Project to full concept space ─────────────────────────────────
        one_hot = F.one_hot(safe_ids, num_classes=K).float()      # (B, max_c, K)
        logit_delta_full = torch.bmm(
            logit_delta_rel.unsqueeze(1), one_hot
        ).squeeze(1)                                               # (B, K)

        # ── Update mastery ────────────────────────────────────────────────
        logit_m   = torch.log((mastery + eps) / (1 - mastery + eps))
        new_mastery = torch.sigmoid(logit_m + logit_delta_full)

        # ── Update momentum (EMA, parameter-free) ─────────────────────────
        # For each touched concept: mom ← β·mom + (1−β)·logit_delta
        # For untouched: keep current momentum
        touched = (one_hot * (~pad_mask).float().unsqueeze(-1)).sum(dim=1) > 0  # (B, K)
        beta = self._mom_decay
        new_momentum = torch.where(
            touched,
            beta * momentum + (1.0 - beta) * logit_delta_full,
            momentum,
        )

        # ── Update timers ─────────────────────────────────────────────────
        new_timer = torch.where(
            touched,
            torch.zeros_like(timers),
            timers,
        )

        return torch.stack([new_mastery, new_momentum, new_timer], dim=-1)
