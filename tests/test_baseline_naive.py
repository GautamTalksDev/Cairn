"""Tests for the naive covariance integrity baseline."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from cairn._internal.baselines.naive import (
    NaiveCovarianceMonitor,
    chi2_scale,
    protection_levels_from_covariance,
)
from cairn.types import IntegrityState, Pose, PoseEstimate


def test_protection_levels_match_chi2_formula() -> None:
    sig = 0.05
    cov = np.diag([sig**2, sig**2, sig**2, 0.01, 0.01, 0.01])
    hpl, vpl = protection_levels_from_covariance(cov, confidence=0.999)
    assert hpl == pytest.approx(sig * np.sqrt(chi2.ppf(0.999, 2)))
    assert vpl == pytest.approx(sig * np.sqrt(chi2.ppf(0.999, 1)))


def test_naive_monitor_always_nominal() -> None:
    cov = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    est = PoseEstimate(
        pose=Pose(
            timestamp=1.0,
            position=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        ),
        covariance=cov,
        source="test",
    )
    report = NaiveCovarianceMonitor(0.999).report_one(est)
    assert report.state is IntegrityState.NOMINAL
    assert report.protection_level_horizontal > 0.0
    assert chi2_scale(0.999, 2) == pytest.approx(np.sqrt(chi2.ppf(0.999, 2)))
