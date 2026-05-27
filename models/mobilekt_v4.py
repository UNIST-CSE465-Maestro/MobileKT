"""MobileKT v4: Question Encoder + MIKT backbone.

This version follows the v2 architecture note: keep a high-capacity
multi-dimensional KT state in the MIKT engine, and make the question side an
adapter that maps cached raw-question features to MIKT-compatible
``(q_hat, diff_hat)``. TAP is intentionally not included here; it can be trained
as a separate readout over frozen states.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import MobileKTConfig

from .backbone import MIKTBackbone
from .qe import MIKTQuestionEncoder, QuestionEncoderOutput


class MobileKTV4(nn.Module):
    """QE + MIKT model with the same training interface as earlier MobileKT."""

    def __init__(self, cfg: MobileKTConfig):
        super().__init__()
        self.cfg = cfg
        state_d = getattr(cfg, "mikt_state_dim", cfg.d)
        qe_input_mode = getattr(cfg, "qe_input_mode", "features")
        self.question_encoder = MIKTQuestionEncoder(
            n_questions=cfg.n_questions,
            d=cfg.d,
            hidden=cfg.qde_hidden,
            dropout=cfg.dropout,
            feature_dim=getattr(cfg, "question_feature_dim", None),
            use_diff_bias=getattr(cfg, "use_diff_bias", True),
            input_mode=qe_input_mode,
        )
        self.backbone = MIKTBackbone(
            n_concepts=cfg.n_concepts,
            d=cfg.d,
            state_d=state_d,
            dropout=cfg.dropout,
            max_seq_len=cfg.max_seq_len,
            output_scale=getattr(cfg, "mikt_output_scale", 5.0),
        )

    def encode_questions(
        self,
        *,
        question_features: torch.Tensor | None = None,
        question_ids: torch.Tensor | None = None,
    ) -> QuestionEncoderOutput:
        """Encode raw-question features, or question IDs for the MIKT-ID baseline."""
        return self.question_encoder(
            question_ids=question_ids,
            question_features=question_features,
        )

    def forward(
        self,
        question_ids: torch.Tensor,
        concept_ids: torch.Tensor,
        responses: torch.Tensor,
        question_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode_questions(
            question_features=question_features,
            question_ids=question_ids if self.question_encoder.input_mode == "id" else None,
        )
        return self.backbone(
            encoded.embedding,
            encoded.difficulty,
            concept_ids,
            responses,
            question_ids=question_ids,
        )

    @torch.no_grad()
    def initial_state(self, batch_size: int, device: torch.device):
        return self.backbone.initial_state(batch_size, device)

    @torch.no_grad()
    def predict_next(
        self,
        q_next_id: torch.Tensor | None,
        c_next_ids: torch.Tensor,
        skill_state: torch.Tensor,
        all_state: torch.Tensor,
        last_skill_time: torch.Tensor,
        step: int,
        question_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self.encode_questions(
            question_features=question_features,
            question_ids=q_next_id if self.question_encoder.input_mode == "id" else None,
        )
        return self.backbone.predict_next(
            encoded.embedding,
            encoded.difficulty,
            c_next_ids,
            skill_state,
            all_state,
            last_skill_time,
            step,
        )

    @torch.no_grad()
    def update_state(
        self,
        q_id: torch.Tensor | None,
        c_ids: torch.Tensor,
        response: torch.Tensor,
        skill_state: torch.Tensor,
        all_state: torch.Tensor,
        last_skill_time: torch.Tensor,
        step: int,
        question_features: torch.Tensor | None = None,
    ):
        encoded = self.encode_questions(
            question_features=question_features,
            question_ids=q_id if self.question_encoder.input_mode == "id" else None,
        )
        return self.backbone.update_state(
            encoded.embedding,
            c_ids,
            response,
            skill_state,
            all_state,
            last_skill_time,
            step,
            difficulty=encoded.difficulty,
        )
