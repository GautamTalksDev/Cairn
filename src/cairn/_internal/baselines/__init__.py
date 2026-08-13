"""Baseline package: integrity monitors without independent corroboration."""

from cairn._internal.baselines.naive import (
    NaiveCovarianceMonitor,
    protection_levels_from_covariance,
    with_claimed_diagonal_covariance,
)

__all__ = [
    "NaiveCovarianceMonitor",
    "protection_levels_from_covariance",
    "with_claimed_diagonal_covariance",
]
