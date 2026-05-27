"""Question Encoder adapters for MobileKT v4.

The research architecture treats the Question Encoder as an adapter from raw
question content to the latent item representation expected by the KT engine.
In the paper-facing architecture the encoder input is raw question metadata
(``Question``, ``Options``, optional ``Visual Description``), precomputed at
authoring/server time into a cached feature vector. A question-ID lookup is kept
only as an explicit ``MIKT-ID`` baseline/fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class QuestionEncoderOutput:
    """MIKT-compatible item representation."""

    embedding: torch.Tensor
    difficulty: torch.Tensor


class MIKTQuestionEncoder(nn.Module):
    """Produce ``(q_hat, diff_hat)`` for MIKT-style KT backbones.

    Args:
        input_mode: ``"features"`` uses cached raw-question features and is the
            document-aligned path. ``"id"`` uses a question-ID embedding table
            for the MIKT-ID baseline.
    """

    def __init__(
        self,
        n_questions: int,
        d: int,
        hidden: int,
        dropout: float = 0.2,
        feature_dim: int | None = None,
        use_diff_bias: bool = True,
        input_mode: str = "features",
    ):
        super().__init__()
        if input_mode not in {"features", "id"}:
            raise ValueError("input_mode must be one of: features, id")
        if input_mode == "features" and feature_dim is None:
            raise ValueError("feature_dim is required when input_mode='features'")

        self.d = d
        self.input_mode = input_mode
        self.question_embed: nn.Embedding | None = None
        if input_mode == "id":
            self.question_embed = nn.Embedding(n_questions + 1, d, padding_idx=0)

        self.feature_proj: nn.Module | None = None
        if feature_dim is not None:
            self.feature_proj = nn.Sequential(
                nn.Linear(feature_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, d),
            )

        self.diff_head = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.diff_bias = None
        if use_diff_bias and input_mode == "id":
            self.diff_bias = nn.Embedding(n_questions + 1, 1, padding_idx=0)
            nn.init.zeros_(self.diff_bias.weight)

    def forward(
        self,
        question_ids: torch.Tensor | None = None,
        question_features: torch.Tensor | None = None,
    ) -> QuestionEncoderOutput:
        if self.input_mode == "features":
            if question_features is None:
                raise ValueError("question_features are required when input_mode='features'")
            if self.feature_proj is None:
                raise ValueError("feature_dim was not configured for question_features")
            q = self.feature_proj(question_features.float())
        else:
            if question_ids is None:
                raise ValueError("question_ids are required when input_mode='id'")
            if self.question_embed is None:
                raise ValueError("question_embed is only initialized for input_mode='id'")
            q = self.question_embed(question_ids.long())

        diff = self.diff_head(q).squeeze(-1)
        if self.diff_bias is not None and question_ids is not None:
            diff = diff + self.diff_bias(question_ids.long()).squeeze(-1)
        return QuestionEncoderOutput(embedding=q, difficulty=diff)
