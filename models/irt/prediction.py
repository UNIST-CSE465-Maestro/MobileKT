"""
IRT Final Prediction

1PL IRT (Rasch model) adapted for MobileKT:

    y_t = σ( C × (FA_{q_t} - Diff_{q_t}) )

where:
    FA_{q_t}   : student ability  (from SAE)
    Diff_{q_t} : question difficulty  (from QDE)
    C          : learned or fixed scale factor (default 3.0)

Training objective: Binary Cross-Entropy
    Loss = -[ a_t log y_t + (1-a_t) log(1-y_t) ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IRTPrediction(nn.Module):
    def __init__(self, irt_scale: float = 3.0):
        """
        Args:
            irt_scale: Initial value for the C scalar (learnable).
        """
        super().__init__()
        self.C = nn.Parameter(torch.tensor(irt_scale))

    def forward(
        self,
        fa: torch.Tensor,    # (B,)  student ability
        diff: torch.Tensor,  # (B,)  question difficulty
    ) -> torch.Tensor:
        """
        Returns:
            y: (B,)  predicted probability of correct response ∈ (0,1)
        """
        return torch.sigmoid(self.C * (fa - diff))

    @staticmethod
    def loss(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Binary Cross-Entropy loss with optional padding mask.

        Args:
            y_pred: (B, seq)  or  (B,)  predicted probabilities
            y_true: (B, seq)  or  (B,)  ground-truth labels  {0, 1}
            mask:   (B, seq)  bool mask (True = valid position)

        Returns:
            scalar loss
        """
        loss = F.binary_cross_entropy(y_pred, y_true.float(), reduction="none")
        if mask is not None:
            loss = loss[mask]
        return loss.mean()
