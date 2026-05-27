"""Configuration objects for time-aware probing."""

from dataclasses import dataclass


@dataclass
class TimeAwareProbeConfig:
    """Small readout head for decoding concept mastery from frozen KT states.

    The probe is intentionally modest in capacity. It should read information
    from a trained KT model, not become a large replacement predictor.
    """

    state_dim: int
    n_concepts: int
    timer_dim: int = 3
    concept_dim: int = 32
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.1
    use_concept_embedding: bool = True

    # Label-generation defaults. These are used by the helper utilities, not
    # by the neural module itself.
    horizon: int = 50
    tau: float = 20.0
    label_mode: str = "decayed_average"  # next | average | decayed_average

    # Regularization defaults for probe training.
    direction_margin: float = 0.02
    lambda_direction: float = 0.1
    lambda_smooth: float = 0.0

    # Ebbinghaus-style forgetting defaults. These are used by the structured
    # decay probe and optional monotonic forgetting regularizer.
    lambda_forgetting: float = 0.0
    forgetting_margin: float = 0.0
    forgetting_short_gap: float = 3.0
    forgetting_long_gap: float = 50.0
    max_decay_rate: float = 2.0
