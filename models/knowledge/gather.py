"""
Knowledge Gather Function

Combines the student's Concept State and Domain State into a final
knowledge vector FK_{q_t} ∈ ℝ^{2d} relative to the current question.

Steps
-----
1. Forgetness modeling:  CS_j^Decay = CS_j × σ(W_f · encode(I_j) + b_f)
2. Domain knowledge:     DK = g ⊙ D̄                              ∈ ℝ^d
3. Concept knowledge:    CK = (1-g) ⊙ Σ_j (α_j · CS_j^D) · CE_j  ∈ ℝ^d
4. Final knowledge:      FK = concat(DK, CK)                       ∈ ℝ^{2d}

g ∈ (0,1)^d is the per-dimension blend gate from DRE (vector, not scalar).
Element-wise gating lets each latent dimension independently decide how much
to rely on domain-level vs concept-level knowledge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .states import encode_interval


class KnowledgeGatherFunction(nn.Module):
    def __init__(self, d: int):
        """
        Args:
            d: embedding dimension
        """
        super().__init__()
        self._d = d
        # Learnable forgetness: 2-layer MLP  d → d//2 → 1
        # Single linear was too shallow to capture non-linear decay patterns.
        self.forget_net = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1),
        )

    def apply_concept_decay(
        self,
        cs: torch.Tensor,     # (B, K, 2)  full concept state
        concept_ids: torch.Tensor,  # (B, max_c)  related concept indices
    ) -> torch.Tensor:
        """
        Apply forgetness decay to the concept mastery values of related concepts.

        Returns:
            cs_decay: (B, max_c)  decayed mastery scalars for each related concept
        """
        B, max_c = concept_ids.shape
        d = self._d

        pad_mask = (concept_ids == -1)
        safe_ids = concept_ids.clamp(min=0)           # (B, max_c)

        # Gather mastery and intervals for related concepts
        mastery   = cs[:, :, 0]                       # (B, K)
        intervals = cs[:, :, 1]                       # (B, K)

        batch_idx = torch.arange(B, device=cs.device).unsqueeze(1).expand(B, max_c)
        rel_mastery   = mastery[batch_idx, safe_ids]  # (B, max_c)
        rel_intervals = intervals[batch_idx, safe_ids]# (B, max_c)

        # encode(I_j): (B, max_c, d)
        enc = encode_interval(rel_intervals, d)       # (B, max_c, d)
        forget_gate = torch.sigmoid(
            self.forget_net(enc).squeeze(-1)          # (B, max_c)
        )

        cs_decay = rel_mastery * forget_gate          # (B, max_c)
        cs_decay = cs_decay.masked_fill(pad_mask, 0.0)
        return cs_decay

    def forward(
        self,
        cs_decay: torch.Tensor,        # (B, max_c)  decayed mastery scalars
        alpha: torch.Tensor,           # (B, max_c)  concept attention weights
        g: torch.Tensor,               # (B, d)      per-dimension blend gate ∈ (0,1)^d
        d_bar: torch.Tensor,           # (B, d)      aggregated domain vector
        concept_embeds: torch.Tensor,  # (K, d)      concept embedding matrix CE
        concept_ids: torch.Tensor,     # (B, max_c)  related concept indices
    ) -> torch.Tensor:
        """
        Returns:
            fk: (B, 2d)  final knowledge vector FK_{q_t}
        """
        safe_ids = concept_ids.clamp(min=0)
        ce = F.embedding(safe_ids, concept_embeds)    # (B, max_c, d)

        # DK_{q_t} = g ⊙ D̄  — element-wise per-dimension domain blend
        DK = g * d_bar                                # (B, d)

        # CK_{q_t} = (1-g) ⊙ Σ_j (α_j · CS_j^D) · CE_j
        weights  = (alpha * cs_decay).unsqueeze(-1)   # (B, max_c, 1)
        ck_raw   = (weights * ce).sum(dim=1)          # (B, d)
        CK = (1 - g) * ck_raw                        # (B, d)

        fk = torch.cat([DK, CK], dim=-1)              # (B, 2d)
        return fk
