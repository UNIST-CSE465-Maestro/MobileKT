"""
On-device Student Knowledge States

These are the interpretable states stored locally on the student's device.
All mastery values are in [0, 1].

ConceptState  — ℝ^{K × 2}  (mastery_j, interval_j)
DomainState   — ℝ^{T + 1}  (mastery_window[0..T-1], interval)

Design principle: states are plain tensors, not nn.Modules, because they
represent runtime data that changes per student — not learnable parameters.
The Updater modules (CU, DU) hold the learnable weights that modify these states.

State initialization is handled by MobileKT._init_concept_state() and
MobileKT._init_domain_state(), which use learnable priors.
"""

import torch


def encode_interval(interval: torch.Tensor, d: int) -> torch.Tensor:
    """
    Encode a time interval scalar to a d-dim vector for forget modeling.

        encode(I) = tile(log(I + 1), d)

    Args:
        interval: (B,)  or  (B, K)   non-negative step count
        d:        target dimension

    Returns:
        encoded: (..., d)
    """
    log_I = torch.log(interval.float() + 1.0)   # (...,)
    return log_I.unsqueeze(-1).expand(*log_I.shape, d)  # (..., d)
