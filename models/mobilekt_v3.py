"""
MobileKT v3 — Full Model

Three targeted changes over v1, each with strong empirical justification:

  A. Per-question difficulty bias  (simpleKT, NeurIPS'22)
     diff_q = QDE(q) + diff_bias[q_id]
     Adds a learnable scalar offset per question on top of the embedding-derived
     difficulty. QDE captures systematic difficulty from content; diff_bias
     captures residual question-specific variation (e.g. tricky wording, unusual
     concept combinations). Init to 0 → training starts identical to v1.
     Parameter overhead: n_questions scalars (~17K for assist09).

  B. Direct prediction head  (replaces SAE + IRTPrediction)
     y_t = σ(Pred([FK(2d) ‖ q'(d)]))
     The rigid IRT formula σ(C·(FA−Diff)) forces FA to live in the same
     1-D scale as Diff. Replacing with an MLP over [FK ‖ q'] lets the model
     jointly learn ability-difficulty matching in the full 3d feature space.
     q' already encodes difficulty direction (via ERM's OF = Diff × direction),
     so the prediction head retains indirect IRT structure while being more
     expressive.

  C. Difficulty-modulated concept update  (DIMKT, NeurIPS'22)
     match_j = 1 − |mastery_j − σ(diff_q)|   ∈ [0, 1]
     Scales each concept's update by its proximity to the question's difficulty
     (zone of proximal development). No new parameters.

On-device state: CS ∈ ℝ^{K×2} (mastery, timer), DS ∈ ℝ^{T+1} — identical to v1.
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
from .updater import ConceptUpdaterV3, DomainUpdater
from config import MobileKTConfig


class MobileKTV3(nn.Module):
    def __init__(self, cfg: MobileKTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        K = cfg.n_concepts
        T = cfg.n_domains

        # ── Embeddings ────────────────────────────────────────────────────
        self.question_embed = nn.Embedding(cfg.n_questions + 1, d, padding_idx=0)
        self.concept_embed  = nn.Embedding(K + 1, d, padding_idx=0)
        self.domain_embed   = nn.Parameter(torch.randn(T, d) * 0.01)
        self.answer_embed   = nn.Embedding(2, d)

        # Per-question difficulty bias: one trainable scalar per question.
        # padding_idx=0 → bias for padding questions is always 0.
        # Controlled by cfg.use_diff_bias (default True).
        if getattr(cfg, 'use_diff_bias', True):
            self.diff_bias = nn.Embedding(cfg.n_questions + 1, 1, padding_idx=0)
            nn.init.zeros_(self.diff_bias.weight)
        else:
            self.diff_bias = None

        # Learnable initial mastery priors (logit scale → sigmoid ≈ 0.5)
        self.init_mastery        = nn.Parameter(torch.full((K + 1,), 0.0))
        self.init_domain_mastery = nn.Parameter(torch.zeros(T))

        # ── Modules ───────────────────────────────────────────────────────
        self.qde = QuestionDifficultyEstimator(d, cfg.qde_hidden, cfg.dropout)
        self.erm = ExpandedRaschModel(d, cfg.erm_hidden, cfg.dropout)
        self.kgf = KnowledgeGatherFunction(d)
        self.dre = DomainRelevanceExtractor(d, cfg.dropout)

        # Direct prediction head: FK(2d) ‖ q'(d) → y ∈ (0,1)
        # Reuses sae_hidden as pred_hidden for config compatibility.
        pred_hidden = cfg.sae_hidden
        self.pred = nn.Sequential(
            nn.Linear(3 * d, pred_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(pred_hidden, 1),
        )

        self.cu = ConceptUpdaterV3(d, cfg.cu_hidden, cfg.dropout)
        self.du = DomainUpdater(d, cfg.du_hidden, T, cfg.dropout)

        self.dropout = nn.Dropout(cfg.dropout)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _init_concept_state(self, B: int, device: torch.device) -> torch.Tensor:
        mastery = torch.sigmoid(self.init_mastery).unsqueeze(0).expand(B, -1)
        timer   = torch.zeros(B, self.cfg.n_concepts + 1, device=device)
        return torch.stack([mastery, timer], dim=-1)

    def _init_domain_state(self, B: int, device: torch.device) -> torch.Tensor:
        mastery = torch.sigmoid(self.init_domain_mastery).unsqueeze(0).expand(B, -1)
        timer   = torch.zeros(B, 1, device=device)
        return torch.cat([mastery, timer], dim=-1)

    def _precompute_sequence(
        self,
        question_ids: torch.Tensor,   # (B, S)
        concept_ids:  torch.Tensor,   # (B, S, max_c)
        CE: torch.Tensor,             # (K+1, d)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Precompute QDE (+ per-Q bias) + ERM for the full sequence.
        Returns: all_q_prime (B,S,d), all_alpha (B,S,max_c), all_diff (B,S)
        """
        B, S  = question_ids.shape
        max_c = concept_ids.shape[2]

        all_q    = self.dropout(self.question_embed(question_ids))    # (B, S, d)
        all_diff = self.qde(all_q)                                    # (B, S)

        # Add per-question difficulty bias (skipped if use_diff_bias=False)
        if self.diff_bias is not None:
            bias     = self.diff_bias(question_ids).squeeze(-1)       # (B, S)
            all_diff = all_diff + bias

        q_flat    = all_q.view(B * S, -1)
        diff_flat = all_diff.view(B * S)
        c_flat    = concept_ids.view(B * S, max_c)

        q_prime_flat, alpha_flat = self.erm(q_flat, diff_flat, CE, c_flat)

        all_q_prime = q_prime_flat.view(B, S, -1)
        all_alpha   = alpha_flat.view(B, S, max_c)

        return all_q_prime, all_alpha, all_diff

    def _predict(self, fk: torch.Tensor, q_prime: torch.Tensor) -> torch.Tensor:
        """FK(2d) ‖ q'(d) → direct MLP → y ∈ (0, 1)."""
        return torch.sigmoid(
            self.pred(torch.cat([fk, q_prime], dim=-1)).squeeze(-1)
        )

    # ─────────────────────────────────────────────────────────────────────
    def forward(
        self,
        question_ids: torch.Tensor,  # (B, S)
        concept_ids:  torch.Tensor,  # (B, S, max_c)
        responses:    torch.Tensor,  # (B, S)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, S   = question_ids.shape
        device = question_ids.device
        T  = self.cfg.n_domains
        CE = self.concept_embed.weight

        all_q_prime, all_alpha, all_diff = self._precompute_sequence(
            question_ids, concept_ids, CE
        )

        cs = self._init_concept_state(B, device)    # (B, K+1, 2)
        ds = self._init_domain_state(B, device)     # (B, T+1)

        preds = []

        for t in range(S - 1):
            c_ids_next   = concept_ids[:, t + 1]    # (B, max_c)
            q_prime_next = all_q_prime[:, t + 1]    # (B, d)
            alpha_next   = all_alpha[:, t + 1]      # (B, max_c)

            # ── Gather knowledge ──────────────────────────────────────────
            cs_decay = self.kgf.apply_concept_decay(cs, c_ids_next)
            g, d_bar = self.dre(
                q_prime_next, alpha_next, cs_decay, ds[:, :T], self.domain_embed
            )
            fk = self.kgf(cs_decay, alpha_next, g, d_bar, CE, c_ids_next)  # (B, 2d)

            # ── Predict ───────────────────────────────────────────────────
            y_t = self._predict(fk, q_prime_next)    # (B,)
            preds.append(y_t)

            # ── Update state with step-t response ─────────────────────────
            valid_t   = (question_ids[:, t] != 0)
            a_t       = responses[:, t]
            c_ids_t   = concept_ids[:, t]
            q_prime_t = all_q_prime[:, t]
            alpha_t   = all_alpha[:, t]
            diff_t    = all_diff[:, t]

            A_t = self.answer_embed(a_t.long())
            X_t = A_t + q_prime_t

            cs_new = self.cu(cs, X_t, alpha_t, c_ids_t, a_t, diff_t)
            ds_new = self.du(ds, X_t)

            valid_t3 = valid_t.unsqueeze(-1).unsqueeze(-1)
            cs = torch.where(valid_t3.expand_as(cs), cs_new, cs)
            valid_t1 = valid_t.unsqueeze(-1)
            ds = torch.where(valid_t1.expand_as(ds), ds_new, ds)

            # Tick concept timers
            new_timer = torch.where(
                valid_t.unsqueeze(-1),
                cs[:, :, 1].detach() + 1.0,
                cs[:, :, 1].detach(),
            )
            cs = torch.stack([cs[:, :, 0], new_timer], dim=-1)

        y_pred = torch.stack(preds, dim=1)           # (B, S-1)
        mask   = (question_ids[:, 1:] != 0)          # (B, S-1)
        return y_pred, mask

    @torch.no_grad()
    def predict_next(
        self,
        q_next_id:  torch.Tensor,   # (B,)
        c_next_ids: torch.Tensor,   # (B, max_c)
        cs:         torch.Tensor,   # (B, K+1, 2)
        ds:         torch.Tensor,   # (B, T+1)
    ) -> torch.Tensor:
        T  = self.cfg.n_domains
        CE = self.concept_embed.weight

        q       = self.dropout(self.question_embed(q_next_id))
        diff    = self.qde(q)
        if self.diff_bias is not None:
            diff = diff + self.diff_bias(q_next_id).squeeze(-1)
        q_prime, alpha = self.erm(q, diff, CE, c_next_ids)

        cs_decay = self.kgf.apply_concept_decay(cs, c_next_ids)
        g, d_bar = self.dre(q_prime, alpha, cs_decay, ds[:, :T], self.domain_embed)
        fk = self.kgf(cs_decay, alpha, g, d_bar, CE, c_next_ids)
        return self._predict(fk, q_prime)

    @torch.no_grad()
    def update_state(
        self,
        q_id:     torch.Tensor,   # (B,)
        c_ids:    torch.Tensor,   # (B, max_c)
        response: torch.Tensor,   # (B,)
        cs:       torch.Tensor,   # (B, K+1, 2)
        ds:       torch.Tensor,   # (B, T+1)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        CE = self.concept_embed.weight

        q    = self.dropout(self.question_embed(q_id))
        diff = self.qde(q)
        if self.diff_bias is not None:
            diff = diff + self.diff_bias(q_id).squeeze(-1)
        q_prime, alpha = self.erm(q, diff, CE, c_ids)

        A = self.answer_embed(response.long())
        X = A + q_prime

        cs = self.cu(cs, X, alpha, c_ids, response, diff)
        ds = self.du(ds, X)
        cs = torch.stack([cs[:, :, 0], cs[:, :, 1].detach() + 1.0], dim=-1)
        return cs, ds

    @torch.no_grad()
    def apply_time_decay(self, ds: torch.Tensor, delta_t: float = 1.0) -> torch.Tensor:
        return self.du(ds, X_t=None, delta_t=delta_t)
