"""
Student Ability Estimator (SAE)

Converts the Final Knowledge vector FK_{q_t} ∈ ℝ^{2d} into a scalar
ability estimate FA_{q_t} ∈ ℝ.

The output is unbounded — it will be used directly in the IRT formula:
    y_t = σ(C * (FA_{q_t} - Diff_{q_t}))
"""

import torch
import torch.nn as nn


class StudentAbilityEstimator(nn.Module):
    def __init__(self, d: int, hidden: int, dropout: float = 0.2,
                 input_dim: int | None = None):
        """
        Args:
            d:         embedding dimension
            hidden:    hidden layer size
            dropout:   dropout rate
            input_dim: total input size (default: 2d from FK = concat(DK, CK))
                       Set to 3d when q' skip-connection is used.

        Architecture: 2-layer MLP  input_dim → hidden → 1
        """
        super().__init__()
        in_dim = input_dim if input_dim is not None else 2 * d
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),  # scalar output
        )

    def forward(self, fk: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fk: final knowledge vector  (B, 2d)
        Returns:
            fa: student ability scalar  (B,)
        """
        return self.net(fk).squeeze(-1)
