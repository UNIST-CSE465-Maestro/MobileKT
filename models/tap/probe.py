"""Neural modules for time-aware mastery probing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TimeAwareProbeConfig


class TimeAwareProbe(nn.Module):
    """Decode concept-level mastery from frozen KT states and timer features.

    The module supports two common usage patterns:

    1. Paired samples:
       ``forward(hidden, concept_ids, timer_features)`` where each row is one
       (student, timestep, concept) sample.

    2. Dense inference:
       ``forward_all_concepts(...)`` to produce a full mastery vector for all
       concepts from either a global hidden state or concept-aligned states.
    """

    def __init__(self, cfg: TimeAwareProbeConfig):
        super().__init__()
        if cfg.n_layers < 1:
            raise ValueError("TimeAwareProbeConfig.n_layers must be >= 1")

        self.cfg = cfg
        self.concept_embed = None
        concept_dim = 0
        if cfg.use_concept_embedding:
            self.concept_embed = nn.Embedding(
                cfg.n_concepts + 1,
                cfg.concept_dim,
                padding_idx=0,
            )
            concept_dim = cfg.concept_dim

        input_dim = cfg.state_dim + cfg.timer_dim + concept_dim
        layers: list[nn.Module] = []
        if cfg.n_layers == 1:
            layers.append(nn.Linear(input_dim, 1))
        else:
            layers.extend(
                [
                    nn.Linear(input_dim, cfg.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                ]
            )
            for _ in range(cfg.n_layers - 2):
                layers.extend(
                    [
                        nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(cfg.dropout),
                    ]
                )
            layers.append(nn.Linear(cfg.hidden_dim, 1))

        self.readout = nn.Sequential(*layers)

    def forward(
        self,
        hidden_state: torch.Tensor,
        concept_ids: torch.Tensor | None = None,
        timer_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return mastery for paired probe samples.

        Args:
            hidden_state: ``(N, state_dim)`` tensor.
            concept_ids: ``(N,)`` concept IDs. Required when
                ``use_concept_embedding=True``.
            timer_features: ``(N, timer_dim)`` past-only timer features.

        Returns:
            ``(N,)`` mastery values in ``[0, 1]``.
        """
        if hidden_state.dim() != 2:
            raise ValueError("hidden_state must have shape (N, state_dim)")

        pieces = [hidden_state]
        if self.concept_embed is not None:
            if concept_ids is None:
                raise ValueError("concept_ids are required when concept embedding is enabled")
            pieces.append(self.concept_embed(concept_ids.long()))

        if timer_features is None:
            timer_features = hidden_state.new_zeros(
                hidden_state.shape[0],
                self.cfg.timer_dim,
            )
        if timer_features.dim() != 2:
            raise ValueError("timer_features must have shape (N, timer_dim)")
        pieces.append(timer_features.to(hidden_state.dtype))

        z = torch.cat(pieces, dim=-1)
        return torch.sigmoid(self.readout(z).squeeze(-1))

    def forward_all_concepts(
        self,
        hidden_state: torch.Tensor,
        timer_features: torch.Tensor,
        concept_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode a full mastery vector from a global state.

        Args:
            hidden_state: ``(B, state_dim)`` global KT state.
            timer_features: ``(B, K+1, timer_dim)`` concept timer features.
            concept_ids: optional ``(K+1,)`` or ``(B, K+1)`` concept IDs.

        Returns:
            ``(B, K+1)`` mastery matrix. Column 0 is the padding concept and
            should usually be ignored by downstream code.
        """
        if hidden_state.dim() != 2:
            raise ValueError("hidden_state must have shape (B, state_dim)")
        if timer_features.dim() != 3:
            raise ValueError("timer_features must have shape (B, K+1, timer_dim)")

        B, K1, _ = timer_features.shape
        expanded_state = hidden_state.unsqueeze(1).expand(B, K1, -1)
        flat_state = expanded_state.reshape(B * K1, -1)
        flat_timer = timer_features.reshape(B * K1, -1)
        flat_concepts = self._expand_concept_ids(B, K1, hidden_state.device, concept_ids)
        return self.forward(flat_state, flat_concepts, flat_timer).view(B, K1)

    def forward_concept_states(
        self,
        concept_state: torch.Tensor,
        timer_features: torch.Tensor,
        concept_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode a full mastery vector from concept-aligned states.

        Args:
            concept_state: ``(B, K+1, state_dim)`` concept-specific state.
            timer_features: ``(B, K+1, timer_dim)`` concept timer features.
            concept_ids: optional ``(K+1,)`` or ``(B, K+1)`` concept IDs.

        Returns:
            ``(B, K+1)`` mastery matrix.
        """
        if concept_state.dim() != 3:
            raise ValueError("concept_state must have shape (B, K+1, state_dim)")
        if timer_features.dim() != 3:
            raise ValueError("timer_features must have shape (B, K+1, timer_dim)")
        if concept_state.shape[:2] != timer_features.shape[:2]:
            raise ValueError("concept_state and timer_features must share B and K+1")

        B, K1, _ = concept_state.shape
        flat_state = concept_state.reshape(B * K1, -1)
        flat_timer = timer_features.reshape(B * K1, -1)
        flat_concepts = self._expand_concept_ids(B, K1, concept_state.device, concept_ids)
        return self.forward(flat_state, flat_concepts, flat_timer).view(B, K1)

    @staticmethod
    def _expand_concept_ids(
        batch_size: int,
        n_concepts_with_pad: int,
        device: torch.device,
        concept_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if concept_ids is None:
            ids = torch.arange(n_concepts_with_pad, device=device)
            return ids.unsqueeze(0).expand(batch_size, -1).reshape(-1)

        concept_ids = concept_ids.to(device)
        if concept_ids.dim() == 1:
            if concept_ids.numel() != n_concepts_with_pad:
                raise ValueError("1D concept_ids must have length K+1")
            return concept_ids.unsqueeze(0).expand(batch_size, -1).reshape(-1)
        if concept_ids.dim() == 2:
            if concept_ids.shape != (batch_size, n_concepts_with_pad):
                raise ValueError("2D concept_ids must have shape (B, K+1)")
            return concept_ids.reshape(-1)
        raise ValueError("concept_ids must be None, 1D, or 2D")


class EbbinghausTimeAwareProbe(nn.Module):
    """Decode mastery with a structured elapsed-time decay term.

    The readout separates stable mastery evidence from elapsed-time forgetting:

    ``mastery = sigmoid(base_logit - decay_rate * log1p(gap))``

    ``decay_rate`` is learned per sample from the frozen KT state, concept, and
    non-gap timer features. The softplus parameterization keeps it non-negative,
    so increasing only the timer gap cannot increase mastery.
    """

    def __init__(self, cfg: TimeAwareProbeConfig):
        super().__init__()
        if cfg.n_layers < 1:
            raise ValueError("TimeAwareProbeConfig.n_layers must be >= 1")

        self.cfg = cfg
        self.concept_embed = None
        concept_dim = 0
        if cfg.use_concept_embedding:
            self.concept_embed = nn.Embedding(
                cfg.n_concepts + 1,
                cfg.concept_dim,
                padding_idx=0,
            )
            concept_dim = cfg.concept_dim

        context_dim = cfg.state_dim + concept_dim + max(cfg.timer_dim - 1, 0)
        self.base_readout = self._build_mlp(context_dim, 1, cfg)
        self.decay_readout = self._build_mlp(context_dim, 1, cfg)
        final_decay = self.decay_readout[-1]
        if isinstance(final_decay, nn.Linear) and final_decay.bias is not None:
            nn.init.constant_(final_decay.bias, -4.0)

    @staticmethod
    def _build_mlp(input_dim: int, output_dim: int, cfg: TimeAwareProbeConfig) -> nn.Sequential:
        layers: list[nn.Module] = []
        if cfg.n_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.extend(
                [
                    nn.Linear(input_dim, cfg.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                ]
            )
            for _ in range(cfg.n_layers - 2):
                layers.extend(
                    [
                        nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(cfg.dropout),
                    ]
                )
            layers.append(nn.Linear(cfg.hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(
        self,
        hidden_state: torch.Tensor,
        concept_ids: torch.Tensor | None = None,
        timer_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_state.dim() != 2:
            raise ValueError("hidden_state must have shape (N, state_dim)")

        if timer_features is None:
            timer_features = hidden_state.new_zeros(hidden_state.shape[0], self.cfg.timer_dim)
        if timer_features.dim() != 2:
            raise ValueError("timer_features must have shape (N, timer_dim)")

        pieces = [hidden_state]
        if self.concept_embed is not None:
            if concept_ids is None:
                raise ValueError("concept_ids are required when concept embedding is enabled")
            pieces.append(self.concept_embed(concept_ids.long()))

        timer_features = timer_features.to(hidden_state.dtype)
        if self.cfg.timer_dim > 1:
            pieces.append(timer_features[:, 1:])
        context = torch.cat(pieces, dim=-1)

        log_gap = timer_features[:, 0].clamp_min(0.0)
        base_logit = self.base_readout(context).squeeze(-1)
        raw_decay = self.decay_readout(context).squeeze(-1)
        decay_rate = F.softplus(raw_decay)
        if self.cfg.max_decay_rate > 0:
            decay_rate = decay_rate.clamp(max=float(self.cfg.max_decay_rate))
        return torch.sigmoid(base_logit - decay_rate * log_gap)

    def forward_all_concepts(
        self,
        hidden_state: torch.Tensor,
        timer_features: torch.Tensor,
        concept_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_state.dim() != 2:
            raise ValueError("hidden_state must have shape (B, state_dim)")
        if timer_features.dim() != 3:
            raise ValueError("timer_features must have shape (B, K+1, timer_dim)")

        B, K1, _ = timer_features.shape
        expanded_state = hidden_state.unsqueeze(1).expand(B, K1, -1)
        flat_state = expanded_state.reshape(B * K1, -1)
        flat_timer = timer_features.reshape(B * K1, -1)
        flat_concepts = TimeAwareProbe._expand_concept_ids(B, K1, hidden_state.device, concept_ids)
        return self.forward(flat_state, flat_concepts, flat_timer).view(B, K1)

    def forward_concept_states(
        self,
        concept_state: torch.Tensor,
        timer_features: torch.Tensor,
        concept_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if concept_state.dim() != 3:
            raise ValueError("concept_state must have shape (B, K+1, state_dim)")
        if timer_features.dim() != 3:
            raise ValueError("timer_features must have shape (B, K+1, timer_dim)")
        if concept_state.shape[:2] != timer_features.shape[:2]:
            raise ValueError("concept_state and timer_features must share B and K+1")

        B, K1, _ = concept_state.shape
        flat_state = concept_state.reshape(B * K1, -1)
        flat_timer = timer_features.reshape(B * K1, -1)
        flat_concepts = TimeAwareProbe._expand_concept_ids(B, K1, concept_state.device, concept_ids)
        return self.forward(flat_state, flat_concepts, flat_timer).view(B, K1)
