"""
MobileKT v2 — Momentum-Augmented Adaptive Knowledge Tracing

Architecture changes from v1
─────────────────────────────────────────────────────────────────────────────
 ① Concept State: (K, 2) → (K, 3)
      Added channel: momentum_logit ∈ ℝ
      Readable as σ(momentum_logit) ∈ (0,1): learning velocity per concept
      State size: ~1.0 KB → ~1.5 KB  (still well within mobile budget)

 ② ConceptUpdater v2 (CU2)
      Adaptive gate z_j = σ(W_gate([X_t ‖ enc(I_j) ‖ σ(mom_j)]))
      Gated update: logit(CS_j) += α_j · z_j · sign · |Δ|
      Momentum EMA: mom_j ← 0.85·mom_j + 0.15·(α_j·Δ_logit)  [no extra params]

 ③ Momentum-boosted Knowledge Gather
      effective_decay = cs_decay × (1 + mom_scale × (2·σ(mom_j) − 1))
      mom_scale: single learnable scalar (init 0.1)
      Concepts on upward trend contribute more strongly to FK

 ④ SAE v2: richer ability estimate
      input = LayerNorm(FK(2d) ‖ q'(d) ‖ mom_summary(1)) = 3d+1
      mom_summary = Σ_j α_j · σ(mom_j)  — attention-weighted momentum average
      Restores q' skip-connection (ablation showed +0.001 benefit)

 ⑤ Everything else (QDE, ERM, DRE, KGF, DU, IRT) unchanged from v1

On-device forward pass (per step t):
─────────────────────────────────────────────────────────────────────────────
  1. QDE:  Diff_{q_t}          = QDE(q_t)
  2. ERM:  q'_t, α             = ERM(q_t, Diff, CE, C_{q_t})
  3. CS_decay                   = KGF.apply_concept_decay(CS, C_{q_t})
  4. DRE:  g, D̄                = DRE(q'_t, α, CS_decay, DS[:T], DE)
  5. KGF:  FK_{q_t}            = KGF(effective_decay, α, g, D̄, CE, C_{q_t})
  6. SAE:  FA_{q_t}            = SAE(LayerNorm([FK ‖ q' ‖ mom_summary]))
  7. IRT:  y_t                  = σ(C × (FA_{q_t} − Diff_{q_t}))

  After response a_t:
  8.  CU2: CS ← CU2(CS, X_t, α, C_{q_t}, a_t)   [updates mastery + momentum]
  9.  DU:  DS ← DU(DS, X_t)
  10. Tick timers
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
from .knowledge.states import encode_interval
from .irt import StudentAbilityEstimator, IRTPrediction
from .updater import DomainUpdater
from .updater.concept_updater_v2 import ConceptUpdaterV2
from config import MobileKTConfig


class MobileKTV2(nn.Module):
    def __init__(self, cfg: MobileKTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        K = cfg.n_concepts
        T = cfg.n_domains

        # ── Embeddings ────────────────────────────────────────────────────
        self.question_embed     = nn.Embedding(cfg.n_questions + 1, d, padding_idx=0)
        self.concept_embed      = nn.Embedding(K + 1, d, padding_idx=0)
        self.domain_embed       = nn.Parameter(torch.randn(T, d) * 0.01)
        self.answer_embed       = nn.Embedding(2, d)

        # Learnable concept / domain priors (same as v1)
        self.init_mastery        = nn.Parameter(torch.zeros(K + 1))
        self.init_domain_mastery = nn.Parameter(torch.zeros(T))

        # ── Shared modules (unchanged from v1) ────────────────────────────
        self.qde = QuestionDifficultyEstimator(d, cfg.qde_hidden, cfg.dropout)
        self.erm = ExpandedRaschModel(d, cfg.erm_hidden, cfg.dropout)
        self.kgf = KnowledgeGatherFunction(d)
        self.dre = DomainRelevanceExtractor(d, cfg.dropout)
        self.irt = IRTPrediction(cfg.irt_scale)
        self.du  = DomainUpdater(d, cfg.du_hidden, T, cfg.dropout)

        # ── New in v2 ─────────────────────────────────────────────────────
        # CU2: adaptive gate + momentum EMA
        self.cu = ConceptUpdaterV2(d, cfg.cu_hidden, cfg.dropout,
                                   mom_decay=getattr(cfg, 'mom_decay', 0.85))

        # Momentum boost scale for KGF (single learnable scalar, init small)
        self.mom_scale = nn.Parameter(torch.tensor(0.1))

        # SAE v2: input 3d+1  (FK:2d + q':d + mom_summary:1)
        self.sae_norm = nn.LayerNorm(3 * d + 1)
        self.sae = StudentAbilityEstimator(d, cfg.sae_hidden, cfg.dropout,
                                           input_dim=3 * d + 1)

        self.dropout = nn.Dropout(cfg.dropout)

    # ── State init ────────────────────────────────────────────────────────

    def _init_concept_state(self, B: int, device: torch.device) -> torch.Tensor:
        """(B, K+1, 3): (mastery, momentum_logit=0, timer=0)"""
        mastery  = torch.sigmoid(self.init_mastery).unsqueeze(0).expand(B, -1)  # (B, K+1)
        momentum = torch.zeros(B, self.cfg.n_concepts + 1, device=device)
        timer    = torch.zeros(B, self.cfg.n_concepts + 1, device=device)
        return torch.stack([mastery, momentum, timer], dim=-1)  # (B, K+1, 3)

    def _init_domain_state(self, B: int, device: torch.device) -> torch.Tensor:
        """(B, T+1): (mastery window, timer)"""
        mastery = torch.sigmoid(self.init_domain_mastery).unsqueeze(0).expand(B, -1)
        timer   = torch.zeros(B, 1, device=device)
        return torch.cat([mastery, timer], dim=-1)

    # ── Helper: gather momentum for related concepts ──────────────────────

    def _gather_momentum(
        self,
        cs: torch.Tensor,           # (B, K+1, 3)
        concept_ids: torch.Tensor,  # (B, max_c)
        alpha: torch.Tensor,        # (B, max_c)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mom_summary: (B, 1)  attention-weighted average momentum ∈ (0,1)
            mom_signal:  (B, max_c)  per-concept σ(momentum) for boost computation
        """
        B, max_c = concept_ids.shape
        device = cs.device

        pad_mask = (concept_ids == -1)
        safe_ids = concept_ids.clamp(min=0)

        momentum = cs[:, :, 1]  # (B, K+1)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, max_c)
        rel_mom   = momentum[batch_idx, safe_ids]              # (B, max_c)
        rel_mom   = rel_mom.masked_fill(pad_mask, 0.0)        # pad → neutral 0
        mom_signal = torch.sigmoid(rel_mom)                    # (B, max_c) ∈ (0,1)

        # Attention-weighted average
        weights     = alpha.masked_fill(pad_mask, 0.0)
        mom_summary = (weights * mom_signal).sum(dim=1, keepdim=True)  # (B, 1)
        return mom_summary, mom_signal

    # ── Precompute sequence features (same structure as v1) ───────────────

    def _precompute_sequence(
        self,
        question_ids: torch.Tensor,   # (B, S)
        concept_ids: torch.Tensor,    # (B, S, max_c)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (all_diff, all_q_prime, all_alpha): all (B, S, ...)"""
        B, S = question_ids.shape
        CE = self.concept_embed.weight

        q_flat    = self.dropout(self.question_embed(question_ids.reshape(B * S)))
        diff_flat = self.qde(q_flat)
        c_flat    = concept_ids.reshape(B * S, -1)
        q_prime_flat, alpha_flat = self.erm(q_flat, diff_flat, CE, c_flat)

        all_diff    = diff_flat.reshape(B, S)
        all_q_prime = q_prime_flat.reshape(B, S, -1)
        all_alpha   = alpha_flat.reshape(B, S, -1)
        return all_diff, all_q_prime, all_alpha

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        question_ids: torch.Tensor,  # (B, S)
        concept_ids: torch.Tensor,   # (B, S, max_c)
        responses: torch.Tensor,     # (B, S)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            preds: (B, S-1)  predicted correctness probabilities
            mask:  (B, S-1)  bool, True = valid (non-padding) position
        """
        B, S = question_ids.shape
        device = question_ids.device
        T = self.cfg.n_domains
        CE = self.concept_embed.weight

        cs = self._init_concept_state(B, device)
        ds = self._init_domain_state(B, device)

        all_diff, all_q_prime, all_alpha = self._precompute_sequence(
            question_ids, concept_ids
        )

        preds, masks = [], []

        for t in range(S - 1):
            # ── Indices: predict step t+1 using state built from 0..t ──────
            q_prime_next = all_q_prime[:, t + 1]   # (B, d)
            diff_next    = all_diff[:, t + 1]       # (B,)
            alpha_next   = all_alpha[:, t + 1]      # (B, max_c)
            c_ids_next   = concept_ids[:, t + 1]    # (B, max_c)
            masks.append(question_ids[:, t + 1] != 0)

            # ── 3. Apply concept decay ─────────────────────────────────────
            # KGF expects (B, K, 2): (mastery, timer) — extract from 3-channel CS
            cs_mastery_timer = torch.stack([cs[:, :, 0], cs[:, :, 2]], dim=-1)  # (B, K, 2)
            cs_decay = self.kgf.apply_concept_decay(cs_mastery_timer, c_ids_next)

            # ── NEW: Momentum gather (single call, reused below) ──────────
            mom_summary, mom_signal = self._gather_momentum(cs, c_ids_next, alpha_next)
            # mom_signal ∈ (0,1), boost ∈ (-1, 1)
            mom_boost = (2 * mom_signal - 1)                  # (B, max_c)
            effective_decay = (cs_decay * (1 + self.mom_scale * mom_boost)).clamp(min=0.0)

            # ── 4. DRE ─────────────────────────────────────────────────────
            g, d_bar = self.dre(
                q_prime_next, alpha_next, effective_decay,
                ds[:, :T], self.domain_embed
            )

            # ── 5. KGF ────────────────────────────────────────────────────
            fk = self.kgf(effective_decay, alpha_next, g, d_bar, CE, c_ids_next)  # (B, 2d)

            # ── 6. SAE v2 ─────────────────────────────────────────────────
            sae_in = self.sae_norm(
                torch.cat([fk, q_prime_next, mom_summary], dim=-1)   # (B, 3d+1)
            )
            fa = self.sae(sae_in)                                      # (B,)

            # ── 7. IRT prediction ──────────────────────────────────────────
            preds.append(self.irt(fa, diff_next))

            # ── 8-9. Update state with step-t response ────────────────────
            valid_t   = (question_ids[:, t] != 0)
            a_t       = responses[:, t]
            c_ids_t   = concept_ids[:, t]
            q_prime_t = all_q_prime[:, t]
            alpha_t   = all_alpha[:, t]

            X_t = self.answer_embed(a_t.long()) + q_prime_t  # (B, d)

            cs_new = self.cu(cs, X_t, alpha_t, c_ids_t, a_t)
            ds_new = self.du(ds, X_t)

            valid3 = valid_t.unsqueeze(-1).unsqueeze(-1)
            valid1 = valid_t.unsqueeze(-1)
            cs = torch.where(valid3.expand_as(cs), cs_new, cs)
            ds = torch.where(valid1.expand_as(ds), ds_new, ds)

            # ── 10. Tick timers ────────────────────────────────────────────
            new_timer = torch.where(
                valid_t.unsqueeze(-1),
                cs[:, :, 2].detach() + 1.0,
                cs[:, :, 2].detach(),
            )
            cs = torch.stack([cs[:, :, 0], cs[:, :, 1], new_timer], dim=-1)

        preds = torch.stack(preds, dim=1)   # (B, S-1)
        mask  = torch.stack(masks, dim=1)   # (B, S-1)
        return preds, mask

    # ── On-device API (same interface as v1) ──────────────────────────────

    @torch.no_grad()
    def predict_next(
        self,
        q_next_id:  torch.Tensor,   # (B,)
        c_next_ids: torch.Tensor,   # (B, max_c)
        cs: torch.Tensor,           # (B, K+1, 3)
        ds: torch.Tensor,           # (B, T+1)
    ) -> torch.Tensor:
        T  = self.cfg.n_domains
        CE = self.concept_embed.weight

        q       = self.dropout(self.question_embed(q_next_id))
        diff    = self.qde(q)
        q_prime, alpha = self.erm(q, diff, CE, c_next_ids)

        cs_mastery_timer = torch.stack([cs[:, :, 0], cs[:, :, 2]], dim=-1)
        cs_decay = self.kgf.apply_concept_decay(cs_mastery_timer, c_next_ids)
        _, mom_signal = self._gather_momentum(cs, c_next_ids, alpha)
        mom_boost = (2 * mom_signal - 1)
        effective_decay = (cs_decay * (1 + self.mom_scale * mom_boost)).clamp(min=0.0)

        g, d_bar = self.dre(q_prime, alpha, effective_decay, ds[:, :T], self.domain_embed)
        fk = self.kgf(effective_decay, alpha, g, d_bar, CE, c_next_ids)

        mom_summary, _ = self._gather_momentum(cs, c_next_ids, alpha)
        sae_in = self.sae_norm(torch.cat([fk, q_prime, mom_summary], dim=-1))
        fa = self.sae(sae_in)
        return self.irt(fa, diff)

    @torch.no_grad()
    def update_state(
        self,
        q_id:     torch.Tensor,   # (B,)
        c_ids:    torch.Tensor,   # (B, max_c)
        response: torch.Tensor,   # (B,)
        cs: torch.Tensor,         # (B, K+1, 3)
        ds: torch.Tensor,         # (B, T+1)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        CE = self.concept_embed.weight

        q       = self.dropout(self.question_embed(q_id))
        diff    = self.qde(q)
        q_prime, alpha = self.erm(q, diff, CE, c_ids)

        A = self.answer_embed(response.long())
        X = A + q_prime

        cs = self.cu(cs, X, alpha, c_ids, response)
        ds = self.du(ds, X)

        # Tick timers
        cs = torch.stack([cs[:, :, 0], cs[:, :, 1], cs[:, :, 2].detach() + 1.0], dim=-1)
        return cs, ds

    @torch.no_grad()
    def apply_time_decay(
        self,
        ds: torch.Tensor,
        delta_t: float = 1.0,
    ) -> torch.Tensor:
        return self.du(ds, X_t=None, delta_t=delta_t)

    # ── Interpretability helper ───────────────────────────────────────────

    @torch.no_grad()
    def get_concept_mastery(self, cs: torch.Tensor) -> torch.Tensor:
        """Returns mastery channel: (B, K+1)"""
        return cs[:, :, 0]

    @torch.no_grad()
    def get_concept_momentum(self, cs: torch.Tensor) -> torch.Tensor:
        """Returns σ(momentum_logit): (B, K+1) ∈ (0,1), 0.5=neutral"""
        return torch.sigmoid(cs[:, :, 1])
