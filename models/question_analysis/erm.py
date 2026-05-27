"""
Expanded Rasch Model (ERM)  —  adapted from MIKT (WWW'24)

Given question embedding q_t and its related concept set C_{q_t}, produces
a Multi-Concept projected Question Embedding Vector q'_t.

Formulas
--------
α_j  = softmax over j ∈ C_{q_t}  of  (q_t^T CE_j / √d)
MC_{q_t}  = Σ_j  α_j · CE_j                           ← weighted concept embedding
OF_{q_t}  = Diff_{q_t} × (W1 · mean(CE_j) + b1)      ← difficulty-modulated direction
q'_t      = MC_{q_t} + OF_{q_t}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ExpandedRaschModel(nn.Module):
    def __init__(self, d: int, hidden: int, dropout: float = 0.2):
        """
        Args:
            d:      embedding dimension (shared for questions and concepts)
            hidden: hidden size for the direction transform W1
            dropout: dropout rate
        """
        super().__init__()
        self.d = d
        self.scale = math.sqrt(d)

        # W1, b1: maps average concept embedding → direction vector ∈ ℝ^d
        self.direction_transform = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d),
        )

    def forward(
        self,
        q: torch.Tensor,               # (B, d)  question embedding
        diff: torch.Tensor,            # (B,)    difficulty scalar from QDE
        concept_embeds: torch.Tensor,  # (K, d)  full concept embedding matrix CE
        concept_ids: torch.Tensor,     # (B, max_c)  concept indices per question (-1 = padding)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            q_prime: (B, d)   Multi-Concept projected question embedding
            alpha:   (B, max_c)  attention weights over concepts (padded positions = 0)
        """
        B, max_c = concept_ids.shape
        device = q.device

        # ── Gather concept embeddings for this batch ───────────────────────
        # concept_ids: (B, max_c),  pad index = -1  →  clamp to 0 for embedding
        pad_mask = (concept_ids == -1)                           # (B, max_c)
        safe_ids = concept_ids.clamp(min=0)                      # (B, max_c)
        ce = F.embedding(safe_ids, concept_embeds)               # (B, max_c, d)

        # ── Attention weights α_j ─────────────────────────────────────────
        # q: (B, d) → (B, 1, d) for batched dot-product
        scores = torch.bmm(q.unsqueeze(1), ce.transpose(1, 2))  # (B, 1, max_c)
        scores = scores.squeeze(1) / self.scale                  # (B, max_c)
        scores = scores.masked_fill(pad_mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)                    # (B, max_c)
        # Replace NaN rows (all-padding edge case) with zeros
        alpha = alpha.nan_to_num(0.0)

        # ── MC_{q_t}: weighted average of concept embeddings ──────────────
        MC = torch.bmm(alpha.unsqueeze(1), ce).squeeze(1)       # (B, d)

        # ── OF_{q_t}: difficulty × direction ─────────────────────────────
        valid = (~pad_mask).float()                              # (B, max_c)
        n_valid = valid.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, 1)
        avg_ce = (ce * valid.unsqueeze(-1)).sum(dim=1) / n_valid # (B, d)
        direction = self.direction_transform(avg_ce)             # (B, d)
        OF = diff.unsqueeze(-1) * direction                      # (B, d)

        # Residual: preserve original question identity alongside concept projection.
        # Without this, q is fully replaced by MC+OF and question-specific signal is lost.
        q_prime = q + MC + OF                                    # (B, d)
        return q_prime, alpha
