"""
Question Difficulty Estimator (QDE)

From question embedding q_t ∈ ℝ^d  →  difficulty scalar Diff_{q_t} ∈ ℝ

The output is unbounded (no sigmoid) so it can directly participate in the
IRT formula:  y_t = σ(C * (FA_{q_t} - Diff_{q_t}))
"""

import torch
import torch.nn as nn


class QuestionDifficultyEstimator(nn.Module):
    def __init__(self, d: int, hidden: int, dropout: float = 0.2):
        """
        Args:
            d:       input dimension (question embedding dim)
            hidden:  hidden layer size
            dropout: dropout rate

        Architecture: 2-layer MLP  d → hidden → 1
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),  # scalar output
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: question embeddings  (batch, seq, d)  or  (batch, d)
        Returns:
            diff: difficulty scalars  (batch, seq)  or  (batch,)
        """
        return self.net(q).squeeze(-1)
