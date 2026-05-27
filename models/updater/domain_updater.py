"""
Domain Knowledge Updater (DU)

Updates the Domain State (DS) sliding window after each response.

Two modes
---------
1. Response mode  (X_t is available — normal update after student answers):
        ΔDS  = Tanh( DU(X_t, I) )
        D̄_s = mean(DS[:T])
        DS  ←push  σ( logit(D̄_s) + ΔDS )    (push new mastery, pop oldest)
        I    = 0

2. Time-only mode  (no response — DS decays over time):
        D̄_s = mean(DS[:T])
        DS  ←push  D̄_s × σ( W_f · encode(I) + b_f )
        I   += Δt
"""

import torch
import torch.nn as nn

from ..knowledge.states import encode_interval


class DomainUpdater(nn.Module):
    def __init__(self, d: int, hidden: int, n_domains: int, dropout: float = 0.2):
        """
        Args:
            d:         embedding dimension
            hidden:    DU hidden size
            n_domains: T — the domain window size
        """
        super().__init__()
        self.T = n_domains
        self._d = d

        # DU maps [X_t || encode(I)] → ΔDS  (scalar)
        self.du = nn.Sequential(
            nn.Linear(2 * d, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # Forgetness gate for time-only decay
        self.forget_net = nn.Linear(d, 1)

    def forward(
        self,
        ds: torch.Tensor,   # (B, T+1)  domain state (last entry is interval)
        X_t: torch.Tensor,  # (B, d)  acquired knowledge,  or None for time-only
        delta_t: float = 1.0,
    ) -> torch.Tensor:
        """
        Returns:
            ds: (B, T+1)  updated domain state
        """
        B, T1 = ds.shape
        T = self.T
        d = self._d
        device = ds.device

        mastery_window = ds[:, :T]     # (B, T)
        interval = ds[:, T]            # (B,)

        D_bar = mastery_window.mean(dim=-1)  # (B,)  D̄_s

        if X_t is not None:
            # ── Response mode ─────────────────────────────────────────────
            enc = encode_interval(interval, d)           # (B, d)
            du_input = torch.cat([X_t, enc], dim=-1)    # (B, 2d)
            delta_ds = torch.tanh(self.du(du_input).squeeze(-1))  # (B,)

            eps = 1e-6
            logit_dbar = torch.log(
                (D_bar + eps) / (1 - D_bar + eps)
            )
            new_mastery = torch.sigmoid(logit_dbar + delta_ds)    # (B,)
            new_interval = torch.zeros_like(interval)
        else:
            # ── Time-only mode ────────────────────────────────────────────
            enc = encode_interval(interval, d)           # (B, d)
            forget_gate = torch.sigmoid(
                self.forget_net(enc).squeeze(-1)         # (B,)
            )
            new_mastery = D_bar * forget_gate            # (B,)
            new_interval = interval + delta_t

        # Push new mastery: drop oldest (index 0), append at the end
        new_window = torch.cat(
            [mastery_window[:, 1:], new_mastery.unsqueeze(-1)], dim=-1
        )                                               # (B, T)

        ds_new = torch.cat(
            [new_window, new_interval.unsqueeze(-1)], dim=-1
        )                                               # (B, T+1)
        return ds_new
