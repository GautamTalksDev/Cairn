"""Tests for solution-separation (RAIM-style) fault detection."""

from __future__ import annotations

import numpy as np
import pytest

from cairn._internal.detectors.consistency import SolutionSeparationDetector, fuse_positions_wls
from cairn.types import Pose, PoseEstimate


def _estimate(
    name: str,
    position: list[float],
    *,
    std: float = 0.05,
    timestamp: float = 0.0,
) -> PoseEstimate:
    cov = np.diag([std**2, std**2, std**2, 0.01, 0.01, 0.01])
    return PoseEstimate(
        pose=Pose(
            timestamp=timestamp,
            position=np.asarray(position, dtype=np.float64),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            frame="map",
        ),
        covariance=cov,
        source=name,
    )


class TestSolutionSeparationFaultInjection:
    def test_identifies_biased_source(self) -> None:
        # Three consistent sources near the origin; lidar carries a 1 m bias in x.
        truth = [0.0, 0.0, 0.0]
        sources = {
            "vio": _estimate("vio", [0.01, -0.02, 0.005], std=0.04),
            "lidar": _estimate("lidar", [1.00, 0.01, -0.01], std=0.04),
            "imu_prop": _estimate("imu_prop", [-0.02, 0.015, 0.0], std=0.06),
        }
        # Sanity: truth unused except to document intent.
        assert truth[0] == 0.0

        result = SolutionSeparationDetector(p_fa=0.01).detect(sources)
        assert result.faulted_source == "lidar"
        lidar_test = next(t for t in result.tests if t.excluded_source == "lidar")
        assert lidar_test.faulted is True
        assert result.protection_level_horizontal > 0.0

    def test_no_fault_on_single_clean_epoch(self) -> None:
        sources = {
            "a": _estimate("a", [0.01, 0.0, 0.0], std=0.05),
            "b": _estimate("b", [-0.01, 0.01, 0.0], std=0.05),
            "c": _estimate("c", [0.0, -0.01, 0.005], std=0.05),
        }
        result = SolutionSeparationDetector(p_fa=0.01).detect(sources)
        assert result.faulted_source is None
        assert all(not t.faulted for t in result.tests)

    def test_pl_grows_with_source_disagreement(self) -> None:
        """HPL must track observed separation, not stay frozen at the χ² floor."""
        detector = SolutionSeparationDetector(p_fa=0.01, pl_confidence=0.999)
        clean = {
            "a": _estimate("a", [0.0, 0.0, 0.0], std=0.05),
            "b": _estimate("b", [0.01, 0.0, 0.0], std=0.05),
            "c": _estimate("c", [-0.01, 0.01, 0.0], std=0.05),
        }
        mild = {
            "a": _estimate("a", [0.0, 0.0, 0.0], std=0.05),
            "b": _estimate("b", [0.25, 0.0, 0.0], std=0.05),
            "c": _estimate("c", [-0.01, 0.01, 0.0], std=0.05),
        }
        strong = {
            "a": _estimate("a", [0.0, 0.0, 0.0], std=0.05),
            "b": _estimate("b", [1.50, 0.0, 0.0], std=0.05),
            "c": _estimate("c", [-0.01, 0.01, 0.0], std=0.05),
        }
        pl_clean = detector.detect(clean).protection_level_horizontal
        pl_mild = detector.detect(mild).protection_level_horizontal
        pl_strong = detector.detect(strong).protection_level_horizontal
        assert pl_mild > pl_clean
        assert pl_strong > pl_mild
        # Separation for equal-weight sensors is ~⅓ of the bias; require clear growth.
        assert pl_strong > 2.0 * pl_clean


class TestFalseAlarmRateCalibration:
    def test_clean_data_false_alarm_rate_matches_p_fa(self) -> None:
        """Monte Carlo under H0: empirical any-fault rate ≈ configured p_fa."""
        p_fa = 0.05
        n_trials = 3000
        rng = np.random.default_rng(42)
        detector = SolutionSeparationDetector(p_fa=p_fa)
        truth = np.array([1.0, -2.0, 0.5])
        stds = {"a": 0.05, "b": 0.06, "c": 0.04, "d": 0.07}

        false_alarms = 0
        for _ in range(n_trials):
            sources = {}
            for name, std in stds.items():
                noise = rng.normal(scale=std, size=3)
                sources[name] = _estimate(
                    name, (truth + noise).tolist(), std=std
                )
            if detector.detect(sources).faulted_source is not None:
                false_alarms += 1

        empirical = false_alarms / n_trials
        # Binomial standard error; allow ~3σ slack for statistical noise.
        se = np.sqrt(p_fa * (1.0 - p_fa) / n_trials)
        assert abs(empirical - p_fa) <= 3.0 * se, (
            f"empirical FAR {empirical:.4f} vs configured p_fa {p_fa} "
            f"(3σ band ±{3 * se:.4f})"
        )


def test_wls_fusion_reduces_variance() -> None:
    sources = {
        "a": _estimate("a", [0.0, 0.0, 0.0], std=0.10),
        "b": _estimate("b", [0.0, 0.0, 0.0], std=0.10),
    }
    _mean, cov = fuse_positions_wls(sources)
    # Two equal sensors → fused variance ≈ half.
    assert cov[0, 0] == pytest.approx(0.005, rel=1e-6)
