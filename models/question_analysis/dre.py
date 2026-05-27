"""
Domain Relevance Extractor (DRE)

From q'_t, the student's Domain State (DS), and decayed Concept State (CS_decay),
produces a per-dimension gate vector g ∈ (0,1)^d.

g controls how much each feature dimension relies on domain-level vs concept-level
knowledge — element-wise, independently per dimension:
    DK = g       ⊙ D̄                           (domain contribution)
    CK = (1 - g) ⊙ Σ α_j CS_j^D · CE_j         (concept contribution)

This extends the original scalar D_α gate to a vector gate, allowing the model
to learn that some latent dimensions are better explained by domain trends while
others are better captured by concept-level mastery — a strictly richer blend.

Formula
-------
D̄ = (1/T) Σ_{i=1}^T  DS_i · DE_i                    (aggregate domain knowledge vector)
g  = σ( (W2 · q'_t) ⊙ D̄  +  W3 · Σ_j α_j · CS_j^Decay )
       ^^^^^^^^^^^^^^^^                ^^^^^^^^^^^^^^^^^^^^
       per-dim domain-question match   concept scalar broadcast to d dims
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainRelevanceExtractor(nn.Module):
    def __init__(self, d: int, dropout: float = 0.2):
        """
        Args:
            d:  embedding dimension
        """
        super().__init__()
        self.d = d

        # W2: projects q'_t into domain embedding space  (d → d)
        self.W2 = nn.Linear(d, d, bias=False)

        # W3: broadcasts concept knowledge scalar → d-dimensional gate contribution
        # Input: scalar ck ∈ ℝ  →  Output: per-dim weight ∈ ℝ^d
        self.W3 = nn.Linear(1, d, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q_prime: torch.Tensor,         # (B, d)  multi-concept question embedding
        alpha: torch.Tensor,           # (B, max_c)  concept attention weights
        cs_decay: torch.Tensor,        # (B, max_c)  decayed concept mastery scalars
        ds: torch.Tensor,              # (B, T)  domain mastery window
        domain_embeds: torch.Tensor,   # (T, d)  domain embedding matrix DE
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            g:      (B, d)  per-dimension blend gate ∈ (0,1)^d
            d_bar:  (B, d)  aggregated domain knowledge vector
        """
        T = domain_embeds.shape[0]

        # ── D̄: aggregate domain knowledge vector ─────────────────────────
        # ds: (B, T),  domain_embeds: (T, d)
        # Weighted sum: (B, T) × (T, d) → (B, d)
        d_bar = torch.matmul(ds, domain_embeds) / T              # (B, d)

        # ── Concept knowledge scalar Σ α_j · CS_j^Decay ──────────────────
        ck_scalar = (alpha * cs_decay).sum(dim=-1, keepdim=True) # (B, 1)

        # ── Vector gate g ∈ (0,1)^d ───────────────────────────────────────
        # domain_term[b, j] = (W2·q'_t)[b,j] * D̄[b,j]  — per-dim affinity
        # concept_term[b, j] = W3[j] * ck_scalar[b]      — per-dim scaling
        q_proj       = self.W2(self.dropout(q_prime))    # (B, d)
        domain_term  = q_proj * d_bar                    # (B, d)  element-wise
        concept_term = self.W3(ck_scalar)                # (B, d)  Linear(1→d)
        g = torch.sigmoid(domain_term + concept_term)    # (B, d)

        return g, d_bar
