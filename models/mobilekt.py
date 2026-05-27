"""
MobileKT — Full Model

Architecture overview (per time step t):
─────────────────────────────────────────────────────────────────────────────
  Cloud side:
    q_t = QTV(raw_question)   ← large model, cached per question

  On-device forward pass:
    1. QDE:  Diff_{q_t}   = QDE(q_t)
    2. ERM:  q'_t, α      = ERM(q_t, Diff, CE, C_{q_t})
    3. CS_decay            = KGF.apply_concept_decay(CS, C_{q_t})
    4. DRE:  g, D̄         = DRE(q'_t, α, CS_decay, DS[:T], DE)
    5. KGF:  FK_{q_t}     = KGF(CS_decay, α, g, D̄, CE, C_{q_t})
    6. SAE:  FA_{q_t}     = SAE(concat(FK_{q_t}, q'_t))  ← q' skip-connection
    7. IRT:  y_t           = σ(C × (FA_{q_t} - Diff_{q_t}))

  On-device update (after response a_t):
    X_t = AE[a_t] + q'_t
    8. CU:  CS  ← CU(CS, X_t, α, C_{q_t})
    9. DU:  DS  ← DU(DS, X_t)
   10. Tick all concept timers by 1

Key design decisions:
  - QDE + ERM precomputed for full sequence (no double call per step)
  - q'_{t+1} concatenated to SAE input (skip connection from question to prediction)
  - Learnable initial concept mastery per concept (CS prior)
  - Learnable initial domain mastery per domain step (DS prior)
  - DRE produces vector gate g ∈ (0,1)^d instead of scalar D_α
  - IRT scale C is learnable
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .question_analysis import (
    QuestionDifficultyEstimator,
    ExpandedRaschModel,
    DomainRelevanceExtractor,
)
from .knowledge import KnowledgeGatherFunction
from .irt import StudentAbilityEstimator, IRTPrediction
from .updater import ConceptUpdater, DomainUpdater
from config import MobileKTConfig


class MobileKT(nn.Module):
    def __init__(self, cfg: MobileKTConfig):
        super().__init__()
        self.cfg = cfg
        d  = cfg.d
        K  = cfg.n_concepts
        T  = cfg.n_domains

        # ── Embeddings ────────────────────────────────────────────────────
        # Question Embedding — replaces QTV for standard datasets (no raw text).
        self.question_embed = nn.Embedding(cfg.n_questions + 1, d, padding_idx=0)
        self.concept_embed  = nn.Embedding(K + 1, d, padding_idx=0)    # CE
        self.domain_embed   = nn.Parameter(torch.randn(T, d) * 0.01)   # DE
        self.answer_embed   = nn.Embedding(2, d)                         # AE

        # Learnable initial mastery per concept (instead of fixed 0.5)
        # This lets the model learn "prior difficulty" of each concept.
        self.init_mastery = nn.Parameter(torch.full((K + 1,), 0.0))    # logit scale → sigmoid ≈ 0.5 init

        # Learnable initial domain mastery (one value per domain window slot)
        # Symmetric to CS prior: lets the model learn the average starting
        # domain proficiency rather than always initialising to 0.5.
        self.init_domain_mastery = nn.Parameter(torch.zeros(T))        # logit scale → sigmoid ≈ 0.5 init

        # ── Modules ───────────────────────────────────────────────────────
        # QDE: 2-layer MLP  d → qde_hidden → 1
        self.qde = QuestionDifficultyEstimator(d, cfg.qde_hidden, cfg.dropout)
        self.erm = ExpandedRaschModel(d, cfg.erm_hidden, cfg.dropout)
        self.kgf = KnowledgeGatherFunction(d)
        self.dre = DomainRelevanceExtractor(d, cfg.dropout)

        # SAE input: FK(2d) only — no q' skip-connection
        # Architecture: 2-layer MLP  2d → sae_hidden → 1
        self.sae = StudentAbilityEstimator(d, cfg.sae_hidden, cfg.dropout,
                                           input_dim=2 * d)
        self.irt = IRTPrediction(cfg.irt_scale)
        self.cu  = ConceptUpdater(d, cfg.cu_hidden, cfg.dropout)
        self.du  = DomainUpdater(d, cfg.du_hidden, T, cfg.dropout)

        self.dropout = nn.Dropout(cfg.dropout)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _init_concept_state(self, B: int, device: torch.device) -> torch.Tensor:
        """
        (B, K+1, 2) — mastery initialized from learnable per-concept prior,
        timers initialized to 0.
        """
        mastery = torch.sigmoid(self.init_mastery).unsqueeze(0).expand(B, -1)  # (B, K+1)
        timer   = torch.zeros(B, self.cfg.n_concepts + 1, device=device)
        return torch.stack([mastery, timer], dim=-1)

    def _init_domain_state(self, B: int, device: torch.device) -> torch.Tensor:
        """
        (B, T+1) — domain mastery window initialized from learnable per-slot prior,
        timer initialized to 0.
        """
        mastery = torch.sigmoid(self.init_domain_mastery).unsqueeze(0).expand(B, -1)  # (B, T)
        timer   = torch.zeros(B, 1, device=device)
        return torch.cat([mastery, timer], dim=-1)  # (B, T+1)

    def _precompute_sequence(
        self,
        question_ids: torch.Tensor,   # (B, S)
        concept_ids:  torch.Tensor,   # (B, S, max_c)
        CE: torch.Tensor,             # (K+1, d)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute QDE + ERM for the full sequence at once.
        Returns:
            all_q_prime: (B, S, d)
            all_alpha:   (B, S, max_c)
            all_diff:    (B, S)
        """
        B, S = question_ids.shape
        max_c = concept_ids.shape[2]

        # Embed all questions: (B, S, d)
        all_q = self.dropout(self.question_embed(question_ids))   # (B, S, d)

        # QDE on flattened sequence, then reshape
        all_diff = self.qde(all_q)                                # (B, S)

        # ERM on flattened (B*S, d) + (B*S, max_c), then reshape
        q_flat    = all_q.view(B * S, -1)                         # (B*S, d)
        diff_flat = all_diff.view(B * S)                          # (B*S,)
        c_flat    = concept_ids.view(B * S, max_c)                # (B*S, max_c)

        q_prime_flat, alpha_flat = self.erm(q_flat, diff_flat, CE, c_flat)

        all_q_prime = q_prime_flat.view(B, S, -1)                 # (B, S, d)
        all_alpha   = alpha_flat.view(B, S, max_c)                # (B, S, max_c)

        return all_q_prime, all_alpha, all_diff

    # ─────────────────────────────────────────────────────────────────────
    def forward(
        self,
        question_ids: torch.Tensor,  # (B, S)
        concept_ids:  torch.Tensor,  # (B, S, max_c)
        responses:    torch.Tensor,  # (B, S)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full sequence forward pass (teacher-forced).
        Returns:
            y_pred: (B, S-1)
            mask:   (B, S-1)  valid positions (question_id != 0)
        """
        B, S = question_ids.shape
        device = question_ids.device
        T = self.cfg.n_domains
        CE = self.concept_embed.weight                             # (K+1, d)

        # ── Precompute QDE+ERM for all positions ──────────────────────────
        all_q_prime, all_alpha, all_diff = self._precompute_sequence(
            question_ids, concept_ids, CE
        )

        # ── Initialise on-device states ───────────────────────────────────
        cs = self._init_concept_state(B, device)                   # (B, K+1, 2)
        ds = self._init_domain_state(B, device)                    # (B, T+1)

        preds = []

        for t in range(S - 1):
            # ── Step t: state, Step t+1: prediction target ────────────────
            c_ids_next  = concept_ids[:, t + 1]                    # (B, max_c)
            q_prime_next = all_q_prime[:, t + 1]                   # (B, d)
            alpha_next   = all_alpha[:, t + 1]                     # (B, max_c)
            diff_next    = all_diff[:, t + 1]                      # (B,)

            # ── 3. Concept decay ──────────────────────────────────────────
            cs_decay = self.kgf.apply_concept_decay(cs, c_ids_next)  # (B, max_c)

            # ── 4. DRE ────────────────────────────────────────────────────
            ds_mastery = ds[:, :T]
            g, d_bar = self.dre(
                q_prime_next, alpha_next, cs_decay, ds_mastery, self.domain_embed
            )

            # ── 5. Knowledge Gather ───────────────────────────────────────
            fk = self.kgf(cs_decay, alpha_next, g, d_bar, CE, c_ids_next)  # (B, 2d)

            # ── 6. SAE ───────────────────────────────────────────────────
            fa = self.sae(fk)                                        # (B,)

            # ── 7. IRT prediction ─────────────────────────────────────────
            y_t = self.irt(fa, diff_next)                           # (B,)
            preds.append(y_t)

            # ── 8-9. Update state with step-t response ────────────────────
            valid_t     = (question_ids[:, t] != 0)                 # (B,) — skip padding
            a_t         = responses[:, t]                           # (B,)
            c_ids_t     = concept_ids[:, t]                         # (B, max_c)
            q_prime_t   = all_q_prime[:, t]                         # (B, d)
            alpha_t     = all_alpha[:, t]                           # (B, max_c)

            A_t = self.answer_embed(a_t.long())                     # (B, d)
            X_t = A_t + q_prime_t                                   # (B, d)

            # CU: X_t magnitude + response direction → concept state update
            # DU: X_t = AE + q' (domain update uses full acquired-knowledge signal)
            cs_new = self.cu(cs, X_t, alpha_t, c_ids_t, a_t)
            ds_new = self.du(ds, X_t)

            # Apply updates only for valid (non-padded) positions
            valid_t3 = valid_t.unsqueeze(-1).unsqueeze(-1)          # (B, 1, 1)
            cs = torch.where(valid_t3.expand_as(cs), cs_new, cs)
            valid_t1 = valid_t.unsqueeze(-1)                        # (B, 1)
            ds = torch.where(valid_t1.expand_as(ds), ds_new, ds)

            # ── 10. Tick concept timers (valid positions only) ────────────
            new_timer = torch.where(
                valid_t.unsqueeze(-1),                              # (B, 1)
                cs[:, :, 1].detach() + 1.0,
                cs[:, :, 1].detach(),
            )
            cs = torch.stack([cs[:, :, 0], new_timer], dim=-1)

        y_pred = torch.stack(preds, dim=1)                         # (B, S-1)
        mask   = (question_ids[:, 1:] != 0)                        # (B, S-1)
        return y_pred, mask

    @torch.no_grad()
    def predict_next(
        self,
        q_next_id:   torch.Tensor,   # (B,)
        c_next_ids:  torch.Tensor,   # (B, max_c)
        cs:          torch.Tensor,   # (B, K+1, 2)
        ds:          torch.Tensor,   # (B, T+1)
    ) -> torch.Tensor:
        """
        Single-step on-device inference (prediction only, no state update).
        Call update_state() after the student responds.
        """
        T  = self.cfg.n_domains
        CE = self.concept_embed.weight

        q     = self.dropout(self.question_embed(q_next_id))
        diff  = self.qde(q)
        q_prime, alpha = self.erm(q, diff, CE, c_next_ids)

        cs_decay = self.kgf.apply_concept_decay(cs, c_next_ids)
        g, d_bar = self.dre(q_prime, alpha, cs_decay, ds[:, :T], self.domain_embed)
        fk = self.kgf(cs_decay, alpha, g, d_bar, CE, c_next_ids)
        fa = self.sae(fk)
        return self.irt(fa, diff)

    @torch.no_grad()
    def update_state(
        self,
        q_id:       torch.Tensor,   # (B,)
        c_ids:      torch.Tensor,   # (B, max_c)
        response:   torch.Tensor,   # (B,)  0 or 1
        cs:         torch.Tensor,   # (B, K+1, 2)
        ds:         torch.Tensor,   # (B, T+1)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update CS and DS after the student responds.
        Returns updated (cs, ds).

        On-device usage:
            prob = model.predict_next(q_id, c_ids, cs, ds)
            # student answers ...
            cs, ds = model.update_state(q_id, c_ids, answer, cs, ds)
        """
        CE = self.concept_embed.weight

        q       = self.dropout(self.question_embed(q_id))
        diff    = self.qde(q)
        q_prime, alpha = self.erm(q, diff, CE, c_ids)

        A = self.answer_embed(response.long())    # (B, d)
        X = A + q_prime                           # (B, d)

        cs = self.cu(cs, X, alpha, c_ids, response)
        ds = self.du(ds, X)

        # Tick concept timers
        cs = torch.stack([cs[:, :, 0], cs[:, :, 1].detach() + 1.0], dim=-1)
        return cs, ds

    @torch.no_grad()
    def apply_time_decay(
        self,
        ds:      torch.Tensor,   # (B, T+1)
        delta_t: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply time-only DS decay when the student is idle (no question answered).
        DS forgets domain mastery based on elapsed time.

        On-device usage (e.g., student hasn't studied for N days):
            ds = model.apply_time_decay(ds, delta_t=N)
        """
        return self.du(ds, X_t=None, delta_t=delta_t)
