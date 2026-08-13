"""CP-13 — Adversarial / filthy input handling.

Documents the contract in ``cairn.validate`` and asserts every case either
raises a clear exception naming timestamp/source or degrades to
``DO_NOT_TRUST`` with a finite protection level (never a silent wrong answer).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from cairn._internal.baselines.naive import NaiveCovarianceMonitor
from cairn._internal.detectors.consistency import SINGLE_SOURCE_TAG, SolutionSeparationDetector
from cairn._internal.detectors.degeneracy import DegeneracyAwareMonitor
from cairn._internal.validate import (
    CairnInputError,
    associate_sources_at_time,
    describe_rate_mismatch,
    validate_trajectory,
)
from cairn.types import IntegrityState, Pose, PoseEstimate, Trajectory


def _pose(t: float, xyz: list[float] | None = None) -> Pose:
    return Pose(
        timestamp=t,
        position=np.asarray(xyz if xyz is not None else [0.0, 0.0, 0.0], dtype=np.float64),
        orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )


def _estimate(
    t: float,
    *,
    source: str = "test",
    xyz: list[float] | None = None,
    std: float = 0.05,
) -> PoseEstimate:
    cov = np.diag([std**2, std**2, std**2, 0.01, 0.01, 0.01])
    return PoseEstimate(pose=_pose(t, xyz), covariance=cov, source=source)


# ---------------------------------------------------------------------------
# Construction / type boundary
# ---------------------------------------------------------------------------


class TestNaNInfRejected:
    def test_nan_position_raises_with_timestamp(self) -> None:
        with pytest.raises(ValueError, match="non-finite|timestamp"):
            _pose(1.0, [0.0, float("nan"), 0.0])

    def test_inf_position_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            _pose(2.5, [float("inf"), 0.0, 0.0])

    def test_nan_covariance_raises_naming_source(self) -> None:
        cov = np.eye(6)
        cov[0, 0] = float("nan")
        with pytest.raises(ValueError, match="source='bad_cov'|non-finite"):
            PoseEstimate(pose=_pose(3.0), covariance=cov, source="bad_cov")

    def test_inf_covariance_raises(self) -> None:
        cov = np.eye(6)
        cov[2, 2] = float("inf")
        with pytest.raises(ValueError, match="non-finite"):
            PoseEstimate(pose=_pose(4.0), covariance=cov, source="inf_cov")


class TestNonPositiveDefiniteCovariance:
    def test_indefinite_covariance_raises_naming_source(self) -> None:
        cov = np.eye(6)
        cov[0, 0] = -1.0
        with pytest.raises(ValueError, match="positive-definite|source='indef'"):
            PoseEstimate(pose=_pose(5.0), covariance=cov, source="indef")

    def test_mutated_non_pd_degrades_at_monitor(self) -> None:
        """In-place mutation after construction: monitor must not emit NaN PL."""
        est = _estimate(6.0, source="mutated")
        est.covariance[0, 0] = -1.0  # break PD after the fact
        report = NaiveCovarianceMonitor().report_one(est)
        assert report.state == IntegrityState.DO_NOT_TRUST
        assert np.isfinite(report.protection_level_horizontal)
        assert "source=mutated" in report.reason or "mutated" in report.reason


class TestTimestampOrdering:
    def test_duplicate_timestamps_raise(self) -> None:
        with pytest.raises(ValueError, match="duplicate timestamp"):
            Trajectory([_estimate(1.0, source="a"), _estimate(1.0, source="b")])

    def test_out_of_order_timestamps_raise(self) -> None:
        with pytest.raises(ValueError, match="out of order"):
            Trajectory([_estimate(2.0, source="a"), _estimate(1.0, source="b")])

    def test_empty_trajectory_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            Trajectory([])

    def test_length_one_trajectory_allowed_for_naive(self) -> None:
        traj = Trajectory([_estimate(10.0, source="solo")])
        reports = NaiveCovarianceMonitor().report(traj)
        assert len(reports) == 1
        assert reports[0].state == IntegrityState.NOMINAL
        assert np.isfinite(reports[0].protection_level_horizontal)


class TestLargeTimeGaps:
    def test_large_gap_warns_and_continues(self, caplog: pytest.LogCaptureFixture) -> None:
        traj = Trajectory(
            [
                _estimate(0.0, source="gapper"),
                _estimate(0.1, source="gapper"),
                _estimate(5.0, source="gapper"),  # 4.9 s gap
            ]
        )
        with caplog.at_level(logging.WARNING, logger="cairn.validate"):
            warnings = validate_trajectory(traj, gap_warn_s=1.0)
        assert len(warnings) == 1
        assert "large time gap" in warnings[0]
        assert "0.100000000" in warnings[0] or "0.1" in warnings[0]
        assert "5.000000000" in warnings[0] or "5.0" in warnings[0]
        assert "gapper" in warnings[0]
        # Still processable.
        reports = NaiveCovarianceMonitor().report(traj)
        assert len(reports) == 3
        assert all(np.isfinite(r.protection_level_horizontal) for r in reports)


# ---------------------------------------------------------------------------
# Consistency / multi-source
# ---------------------------------------------------------------------------


class TestSingleSourceConsistency:
    def test_single_source_degrades_not_crash(self) -> None:
        det = SolutionSeparationDetector(p_fa=0.05)
        result = det.detect({"only": _estimate(1.0, source="only")})
        assert result.faulted_source == SINGLE_SOURCE_TAG
        assert np.isfinite(result.protection_level_horizontal)
        assert result.protection_level_horizontal > 0.0

    def test_single_source_report_is_do_not_trust(self) -> None:
        det = SolutionSeparationDetector(p_fa=0.05)
        report = det.report_one({"only": _estimate(2.0, source="lidar")})
        assert report.state == IntegrityState.DO_NOT_TRUST
        assert np.isfinite(report.protection_level_horizontal)
        assert "single_source" in report.reason

    def test_empty_sources_degrades(self) -> None:
        det = SolutionSeparationDetector(p_fa=0.05)
        result = det.detect({})
        assert result.faulted_source == SINGLE_SOURCE_TAG
        assert np.isfinite(result.protection_level_horizontal)


class TestMismatchedSampleRates:
    def test_associate_and_degrade_sparse_epochs(self) -> None:
        # Fast stream @ 10 Hz for 1 s; slow stream @ 1 Hz.
        fast = Trajectory(
            [_estimate(i * 0.1, source="fast", xyz=[0.0, 0.0, 0.0]) for i in range(11)]
        )
        slow = Trajectory(
            [
                _estimate(0.0, source="slow", xyz=[0.01, 0.0, 0.0]),
                _estimate(1.0, source="slow", xyz=[0.01, 0.0, 0.0]),
            ]
        )
        summary = describe_rate_mismatch({"fast": fast, "slow": slow})
        assert "fast" in summary and "slow" in summary

        epochs = associate_sources_at_time(
            {"fast": fast, "slow": slow}, max_dt=0.05
        )
        assert len(epochs) >= 2
        det = SolutionSeparationDetector(p_fa=0.05)
        n_dnt = 0
        n_ok = 0
        for epoch in epochs:
            report = det.report_one(epoch)
            assert np.isfinite(report.protection_level_horizontal)
            if len(epoch) < 2:
                assert report.state == IntegrityState.DO_NOT_TRUST
                n_dnt += 1
            else:
                n_ok += 1
        assert n_dnt > 0  # mid-rate epochs with only "fast"
        assert n_ok >= 2  # endpoints where both align


class TestMutatedNanDoesNotPropagateToPL:
    def test_naive_nan_position_after_construction(self) -> None:
        est = _estimate(7.0, source="poison")
        est.pose.position[0] = float("nan")
        report = NaiveCovarianceMonitor().report_one(est)
        assert report.state == IntegrityState.DO_NOT_TRUST
        assert np.isfinite(report.protection_level_horizontal)
        assert np.isfinite(report.protection_level_vertical)

    def test_consistency_nan_source_degrades(self, caplog: pytest.LogCaptureFixture) -> None:
        good = _estimate(8.0, source="good", xyz=[0.0, 0.0, 0.0])
        bad = _estimate(8.0, source="bad", xyz=[0.1, 0.0, 0.0])
        bad.pose.position[1] = float("nan")
        with caplog.at_level(logging.WARNING, logger="cairn.detectors.consistency"):
            result = SolutionSeparationDetector().detect({"good": good, "bad": bad})
        assert result.faulted_source == SINGLE_SOURCE_TAG
        assert np.isfinite(result.protection_level_horizontal)
        assert any("rejected" in r.message.lower() or "non-finite" in r.message.lower()
                   for r in caplog.records)


class TestDegeneracyFilthyInput:
    def test_nan_scores_do_not_trust(self) -> None:
        est = _estimate(9.0, source="deg")
        scores = np.ones(6)
        scores[0] = float("nan")
        report = DegeneracyAwareMonitor().report_one(est, scores)
        assert report.state == IntegrityState.DO_NOT_TRUST
        assert np.isfinite(report.protection_level_horizontal)


class TestBehaviourContractDocumented:
    """Sanity: CairnInputError messages always name the locus."""

    def test_validate_trajectory_names_source_on_gap(self, caplog: pytest.LogCaptureFixture) -> None:
        traj = Trajectory([_estimate(0.0, source="A"), _estimate(10.0, source="A")])
        with caplog.at_level(logging.WARNING, logger="cairn.validate"):
            msgs = validate_trajectory(traj, gap_warn_s=1.0)
        assert msgs and "source='A'" in msgs[0]

    def test_associate_empty_streams_raises(self) -> None:
        with pytest.raises(CairnInputError, match="at least one stream"):
            associate_sources_at_time({})
