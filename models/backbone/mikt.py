"""MIKT-style backbone adapted to the MobileKT sequence interface.

The original pyKT MIKT implementation builds a global ``pro2skill`` matrix from
the pyKT dataset object. MobileKT batches already carry multi-concept IDs per
interaction, so this module uses that concept set directly while preserving the
important MIKT ingredients: per-concept vector state, global state, elapsed-step
forgetting, Rasch-style item difficulty, and multi-concept attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MIKTBackbone(nn.Module):
    """A compact MIKT-style KT engine for ``(q, concepts, response)`` streams."""

    def __init__(
        self,
        n_concepts: int,
        d: int,
        state_d: int | None = None,
        dropout: float = 0.2,
        max_seq_len: int = 200,
        output_scale: float = 5.0,
    ):
        super().__init__()
        self.n_concepts = n_concepts
        self.d = d
        self.state_d = state_d or d
        self.max_seq_len = max_seq_len
        self.output_scale = output_scale

        D = self.state_d
        self.skill_embed = nn.Embedding(n_concepts + 1, d, padding_idx=0)
        self.answer_embed = nn.Embedding(2, d)
        self.time_state = nn.Parameter(torch.empty(max_seq_len + 1, D))
        self.skill_state = nn.Parameter(torch.empty(n_concepts + 1, D))
        self.all_state = nn.Parameter(torch.empty(1, D))

        self.pro_linear = nn.Linear(d, d)
        self.skill_linear = nn.Linear(d, d)
        self.pro_change = nn.Linear(d, d)
        self.question_to_state = nn.Linear(d, D)

        self.all_forget = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.ReLU(),
            nn.Linear(D, D),
            nn.Sigmoid(),
        )
        self.skill_forget = nn.Sequential(
            nn.Linear(3 * D, D),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(D, D),
            nn.Sigmoid(),
        )

        self.now_obtain = nn.Sequential(
            nn.Linear(d, D),
            nn.Tanh(),
            nn.Linear(D, D),
            nn.Tanh(),
        )
        self.all_obtain = nn.Linear(d, D)
        self.predict_attn = nn.Linear(2 * D + d, D)
        self.pro_ability = nn.Sequential(
            nn.Linear(2 * D + d, D),
            nn.ReLU(),
            nn.Linear(D, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.time_state)
        nn.init.xavier_uniform_(self.skill_state)
        nn.init.xavier_uniform_(self.all_state)
        with torch.no_grad():
            self.skill_embed.weight[0].zero_()
            self.skill_state[0].zero_()

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        skill_state = self.skill_state.unsqueeze(0).expand(batch_size, -1, -1).clone()
        all_state = self.all_state.expand(batch_size, -1).clone()
        last_skill_time = torch.zeros(batch_size, self.n_concepts + 1, device=device)
        return skill_state, all_state, last_skill_time

    def build_item_embedding(
        self,
        question_embedding: torch.Tensor,
        difficulty: torch.Tensor,
        concept_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Build a MIKT-compatible item embedding from QE output and KC set."""
        valid = self._valid_concept_mask(concept_ids)
        safe_ids = concept_ids.clamp(min=0, max=self.n_concepts)
        skill_emb = self.skill_embed(safe_ids)

        q_query = self.pro_linear(question_embedding).unsqueeze(1)
        c_key = self.skill_linear(skill_emb)
        scores = torch.bmm(q_query, c_key.transpose(1, 2)).squeeze(1) / math.sqrt(self.d)
        scores = scores.masked_fill(~valid, float("-inf"))
        alpha = torch.softmax(scores, dim=-1).nan_to_num(0.0)

        skill_attn = torch.bmm(alpha.unsqueeze(1), skill_emb).squeeze(1)
        denom = valid.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        skill_mean = (skill_emb * valid.unsqueeze(-1)).sum(dim=1) / denom
        diff_direction = difficulty.unsqueeze(-1) * self.pro_change(skill_mean)
        return self.dropout(question_embedding + skill_attn + diff_direction)

    def forward(
        self,
        question_embedding: torch.Tensor,
        difficulty: torch.Tensor,
        concept_ids: torch.Tensor,
        responses: torch.Tensor,
        question_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, _ = question_embedding.shape
        device = question_embedding.device

        item_embedding = self.build_item_embedding(
            question_embedding.reshape(B * S, -1),
            difficulty.reshape(B * S),
            concept_ids.reshape(B * S, concept_ids.shape[-1]),
        ).view(B, S, -1)

        skill_state, all_state, last_skill_time = self.initial_state(B, device)
        preds = []

        for t in range(S - 1):
            valid_t = self._valid_question(question_ids, concept_ids, t)
            skill_state, all_state, last_skill_time = self._update_observed(
                skill_state,
                all_state,
                last_skill_time,
                item_embedding[:, t],
                concept_ids[:, t],
                responses[:, t],
                step=t,
                valid=valid_t,
            )
            pred = self._predict_from_state(
                skill_state,
                all_state,
                last_skill_time,
                item_embedding[:, t + 1],
                difficulty[:, t + 1],
                concept_ids[:, t + 1],
                step=t + 1,
            )
            preds.append(pred)

        y_pred = torch.stack(preds, dim=1)
        if question_ids is not None:
            mask = question_ids[:, 1:] != 0
        else:
            mask = self._valid_concept_mask(concept_ids[:, 1:]).any(dim=-1)
        return y_pred, mask

    @torch.no_grad()
    def predict_next(
        self,
        question_embedding: torch.Tensor,
        difficulty: torch.Tensor,
        concept_ids: torch.Tensor,
        skill_state: torch.Tensor,
        all_state: torch.Tensor,
        last_skill_time: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        item_embedding = self.build_item_embedding(question_embedding, difficulty, concept_ids)
        return self._predict_from_state(
            skill_state,
            all_state,
            last_skill_time,
            item_embedding,
            difficulty,
            concept_ids,
            step,
        )

    @torch.no_grad()
    def update_state(
        self,
        question_embedding: torch.Tensor,
        concept_ids: torch.Tensor,
        response: torch.Tensor,
        skill_state: torch.Tensor,
        all_state: torch.Tensor,
        last_skill_time: torch.Tensor,
        step: int,
        difficulty: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if difficulty is None:
            difficulty = question_embedding.new_zeros(question_embedding.shape[0])
        item_embedding = self.build_item_embedding(question_embedding, difficulty, concept_ids)
        valid = self._valid_concept_mask(concept_ids).any(dim=-1)
        return self._update_observed(
            skill_state,
            all_state,
            last_skill_time,
            item_embedding,
            concept_ids,
            response,
            step,
            valid,
        )

    def _update_observed(
        self,
        skill_state: torch.Tensor,
        all_state: torch.Tensor,
        last_skill_time: torch.Tensor,
        item_embedding: torch.Tensor,
        concept_ids: torch.Tensor,
        response: torch.Tensor,
        step: int,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decayed_skill, decayed_all = self._decay_for_concepts(
            skill_state, all_state, last_skill_time, concept_ids, step
        )
        x = item_embedding + self.answer_embed(response.long().clamp(0, 1))
        next_all = decayed_all + torch.tanh(self.all_obtain(self.dropout(x)))

        to_get = self.now_obtain(self.dropout(x)).unsqueeze(1)
        related = self._concept_multi_hot(concept_ids, skill_state.device)
        scores = torch.bmm(to_get, decayed_skill.transpose(1, 2)).squeeze(1) / math.sqrt(self.state_d)
        scores = scores.masked_fill(~related, float("-inf"))
        alpha = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        delta = alpha.unsqueeze(-1) * to_get
        next_skill = decayed_skill + delta

        valid2 = valid.unsqueeze(-1)
        valid3 = valid2.unsqueeze(-1)
        skill_state = torch.where(valid3, next_skill, skill_state)
        all_state = torch.where(valid2, next_all, all_state)

        current_time = torch.full_like(last_skill_time, float(step))
        next_last = torch.where(related & valid2, current_time, last_skill_time)
        return skill_state, all_state, next_last

    def _predict_from_state(
        self,
        skill_state: torch.Tensor,
        all_state: torch.Tensor,
        last_skill_time: torch.Tensor,
        item_embedding: torch.Tensor,
        difficulty: torch.Tensor,
        concept_ids: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        decayed_skill, decayed_all = self._decay_for_concepts(
            skill_state, all_state, last_skill_time, concept_ids, step
        )
        related = self._concept_multi_hot(concept_ids, skill_state.device)
        q_state = self.question_to_state(item_embedding).unsqueeze(1)
        scores = torch.bmm(q_state, decayed_skill.transpose(1, 2)).squeeze(1) / math.sqrt(self.state_d)
        scores = scores.masked_fill(~related, float("-inf"))
        alpha = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        need_state = torch.bmm(alpha.unsqueeze(1), decayed_skill).squeeze(1)

        gate = torch.sigmoid(
            self.predict_attn(self.dropout(torch.cat([need_state, decayed_all, item_embedding], dim=-1)))
        )
        fused_state = torch.cat([(1.0 - gate) * need_state, gate * decayed_all], dim=-1)
        ability = torch.sigmoid(
            self.pro_ability(torch.cat([fused_state, item_embedding], dim=-1)).squeeze(-1)
        )
        return torch.sigmoid(self.output_scale * (ability - torch.sigmoid(difficulty)))

    def _decay_for_concepts(
        self,
        skill_state: torch.Tensor,
        all_state: torch.Tensor,
        last_skill_time: torch.Tensor,
        concept_ids: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = skill_state.shape[0]
        device = skill_state.device
        one_gap = torch.ones(B, dtype=torch.long, device=device).clamp(max=self.max_seq_len)
        all_gap_embed = F.embedding(one_gap, self.time_state)
        decayed_all = all_state * self.all_forget(self.dropout(torch.cat([all_state, all_gap_embed], dim=-1)))

        gap = (float(step) - last_skill_time).clamp(min=0, max=self.max_seq_len).long()
        gap_embed = F.embedding(gap, self.time_state)
        all_effect = decayed_all.unsqueeze(1).expand_as(skill_state)
        forget = self.skill_forget(self.dropout(torch.cat([skill_state, gap_embed, all_effect], dim=-1)))
        related = self._concept_multi_hot(concept_ids, device).unsqueeze(-1)
        forget = torch.where(related, forget, torch.ones_like(forget))
        return skill_state * forget, decayed_all

    def _concept_multi_hot(self, concept_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        valid = self._valid_concept_mask(concept_ids)
        safe_ids = concept_ids.clamp(min=0, max=self.n_concepts)
        out = torch.zeros(concept_ids.shape[0], self.n_concepts + 1, dtype=torch.bool, device=device)
        out.scatter_(1, safe_ids.to(device), valid.to(device))
        out[:, 0] = False
        return out

    def _valid_concept_mask(self, concept_ids: torch.Tensor) -> torch.Tensor:
        return (concept_ids > 0) & (concept_ids <= self.n_concepts)

    def _valid_question(
        self,
        question_ids: torch.Tensor | None,
        concept_ids: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        if question_ids is not None:
            return question_ids[:, step] != 0
        return self._valid_concept_mask(concept_ids[:, step]).any(dim=-1)
