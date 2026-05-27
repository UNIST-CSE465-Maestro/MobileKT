"""Time-aware probing tools for interpretable KT mastery readout."""

from .config import TimeAwareProbeConfig
from .features import build_timer_features
from .labels import build_future_correctness_labels, gather_labeled_probe_samples
from .losses import (
    direction_margin_loss,
    forgetting_monotonic_loss,
    smoothness_loss,
    soft_binary_cross_entropy,
    time_aware_probe_loss,
)
from .probe import EbbinghausTimeAwareProbe, TimeAwareProbe

__all__ = [
    "TimeAwareProbe",
    "EbbinghausTimeAwareProbe",
    "TimeAwareProbeConfig",
    "build_timer_features",
    "build_future_correctness_labels",
    "gather_labeled_probe_samples",
    "soft_binary_cross_entropy",
    "direction_margin_loss",
    "forgetting_monotonic_loss",
    "smoothness_loss",
    "time_aware_probe_loss",
]
