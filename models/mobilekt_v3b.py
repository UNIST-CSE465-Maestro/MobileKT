"""
MobileKT v3b — IRT Backbone + DIMKT Concept Update

Single targeted change over v1:
  - ConceptUpdater → ConceptUpdaterV3  (DIMKT difficulty-modulated update)
  - All other components identical to v1 (SAE + IRT prediction)

Motivation:
  V3's direct prediction head sacrifices the IRT structural prior, which is
  a strong inductive bias on small datasets (< 3000 training sequences).
  V3b retains the IRT formulation while adding the only v3 change that is
  both parameter-free and theoretically well-grounded (ZPD / DIMKT).

  V1 with lr=2e-4 achieved test_auc=0.7559. V3b targets improvement over
  that by letting concept updates scale with difficulty-mastery proximity.
"""

import torch
import torch.nn as nn

from .question_analysis import (
    QuestionDifficultyEstimator,
    ExpandedRaschModel,
    DomainRelevanceExtractor,
)
from .knowledge import KnowledgeGatherFunction
from .irt import StudentAbilityEstimator, IRTPrediction
from .updater import ConceptUpdaterV3, DomainUpdater
from config import MobileKTConfig


class MobileKTV3b(nn.Module):
    def __init__(self, cfg: MobileKTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        K = cfg.n_concepts
        T = cfg.n_domains

        self.question_embed      = nn.Embedding(cfg.n_questions + 1, d, padding_idx=0)
        self.concept_embed       = nn.Embedding(K + 1, d, padding_idx=0)
        self.domain_embed        = nn.Parameter(torch.randn(T, d) * 0.01)
        self.answer_embed        = nn.Embedding(2, d)
        self.init_mastery        = nn.Parameter(torch.full((K + 1,), 0.0))
        self.init_domain_mastery = nn.Parameter(torch.zeros(T))

        self.qde = QuestionDifficultyEstimator(d, cfg.qde_hidden, cfg.dropout)
        self.erm = ExpandedRaschModel(d, cfg.erm_hidden, cfg.dropout)
        self.kgf = KnowledgeGatherFunction(d)
        self.dre = DomainRelevanceExtractor(d, cfg.dropout)
        self.sae = StudentAbilityEstimator(d, cfg.sae_hidden, cfg.dropout, input_dim=2 * d)
        self.irt = IRTPrediction(cfg.irt_scale)
        self.cu  = ConceptUpdaterV3(d, cfg.cu_hidden, cfg.dropout)   # DIMKT
        self.du  = DomainUpdater(d, cfg.du_hidden, T, cfg.dropout)

        self.dropout = nn.Dropout(cfg.dropout)

    def _init_concept_state(self, B: int, device: torch.device) -> torch.Tensor:
        mastery = torch.sigmoid(self.init_mastery).unsqueeze(0).expand(B, -1)
        timer   = torch.zeros(B, self.cfg.n_concepts + 1, device=device)
        return torch.stack([mastery, timer], dim=-1)

    def _init_domain_state(self, B: int, device: torch.device) -> torch.Tensor:
        mastery = torch.sigmoid(self.init_domain_mastery).unsqueeze(0).expand(B, -1)
        timer   = torch.zeros(B, 1, device=device)
        return torch.cat([mastery, timer], dim=-1)

    def _precompute_sequence(self, question_ids, concept_ids, CE):
        B, S  = question_ids.shape
        max_c = concept_ids.shape[2]

        all_q    = self.dropout(self.question_embed(question_ids))
        all_diff = self.qde(all_q)

        q_flat    = all_q.view(B * S, -1)
        diff_flat = all_diff.view(B * S)
        c_flat    = concept_ids.view(B * S, max_c)
        q_prime_flat, alpha_flat = self.erm(q_flat, diff_flat, CE, c_flat)

        return q_prime_flat.view(B, S, -1), alpha_flat.view(B, S, max_c), all_diff

    def forward(self, question_ids, concept_ids, responses):
        B, S   = question_ids.shape
        device = question_ids.device
        T  = self.cfg.n_domains
        CE = self.concept_embed.weight

        all_q_prime, all_alpha, all_diff = self._precompute_sequence(
            question_ids, concept_ids, CE
        )

        cs = self._init_concept_state(B, device)
        ds = self._init_domain_state(B, device)
        preds = []

        for t in range(S - 1):
            c_ids_next   = concept_ids[:, t + 1]
            q_prime_next = all_q_prime[:, t + 1]
            alpha_next   = all_alpha[:, t + 1]
            diff_next    = all_diff[:, t + 1]

            cs_decay = self.kgf.apply_concept_decay(cs, c_ids_next)
            g, d_bar = self.dre(q_prime_next, alpha_next, cs_decay,
                                ds[:, :T], self.domain_embed)
            fk  = self.kgf(cs_decay, alpha_next, g, d_bar, CE, c_ids_next)
            fa  = self.sae(fk)
            y_t = self.irt(fa, diff_next)
            preds.append(y_t)

            valid_t   = (question_ids[:, t] != 0)
            a_t       = responses[:, t]
            c_ids_t   = concept_ids[:, t]
            q_prime_t = all_q_prime[:, t]
            alpha_t   = all_alpha[:, t]
            diff_t    = all_diff[:, t]

            A_t = self.answer_embed(a_t.long())
            X_t = A_t + q_prime_t

            cs_new = self.cu(cs, X_t, alpha_t, c_ids_t, a_t, diff_t)   # DIMKT
            ds_new = self.du(ds, X_t)

            valid_t3 = valid_t.unsqueeze(-1).unsqueeze(-1)
            cs = torch.where(valid_t3.expand_as(cs), cs_new, cs)
            valid_t1 = valid_t.unsqueeze(-1)
            ds = torch.where(valid_t1.expand_as(ds), ds_new, ds)

            new_timer = torch.where(
                valid_t.unsqueeze(-1),
                cs[:, :, 1].detach() + 1.0,
                cs[:, :, 1].detach(),
            )
            cs = torch.stack([cs[:, :, 0], new_timer], dim=-1)

        y_pred = torch.stack(preds, dim=1)
        mask   = (question_ids[:, 1:] != 0)
        return y_pred, mask

    @torch.no_grad()
    def predict_next(self, q_next_id, c_next_ids, cs, ds):
        T  = self.cfg.n_domains
        CE = self.concept_embed.weight

        q       = self.dropout(self.question_embed(q_next_id))
        diff    = self.qde(q)
        q_prime, alpha = self.erm(q, diff, CE, c_next_ids)

        cs_decay = self.kgf.apply_concept_decay(cs, c_next_ids)
        g, d_bar = self.dre(q_prime, alpha, cs_decay, ds[:, :T], self.domain_embed)
        fk = self.kgf(cs_decay, alpha, g, d_bar, CE, c_next_ids)
        fa = self.sae(fk)
        return self.irt(fa, diff)

    @torch.no_grad()
    def update_state(self, q_id, c_ids, response, cs, ds):
        CE = self.concept_embed.weight

        q       = self.dropout(self.question_embed(q_id))
        diff    = self.qde(q)
        q_prime, alpha = self.erm(q, diff, CE, c_ids)

        A = self.answer_embed(response.long())
        X = A + q_prime

        cs = self.cu(cs, X, alpha, c_ids, response, diff)
        ds = self.du(ds, X)
        cs = torch.stack([cs[:, :, 0], cs[:, :, 1].detach() + 1.0], dim=-1)
        return cs, ds

    @torch.no_grad()
    def apply_time_decay(self, ds, delta_t=1.0):
        return self.du(ds, X_t=None, delta_t=delta_t)
