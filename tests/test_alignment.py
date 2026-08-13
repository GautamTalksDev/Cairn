"""Tests for trajectory alignment and ATE cross-check against evo."""

from __future__ import annotations

import numpy as np
import pytest

from cairn.metrics.alignment import associate, umeyama_alignment
from cairn.metrics.error import ate_rmse, position_errors
from cairn.types import Pose, PoseEstimate, Trajectory


def _rotz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _quat_from_R(R: np.ndarray) -> np.ndarray:
    from cairn.metrics.alignment import _rotmat_to_quat_wxyz

    return _rotmat_to_quat_wxyz(R)


def _traj_from_positions(
    positions: np.ndarray,
    timestamps: np.ndarray | None = None,
    R_body: np.ndarray | None = None,
) -> Trajectory:
    """Build a trajectory from ``(N, 3)`` positions (identity orientation unless given)."""
    n = positions.shape[0]
    if timestamps is None:
        timestamps = np.arange(n, dtype=np.float64) * 0.1
    if R_body is None:
        R_body = np.eye(3, dtype=np.float64)
    q = _quat_from_R(R_body)
    estimates = [
        PoseEstimate(
            pose=Pose(
                timestamp=float(timestamps[i]),
                position=positions[i],
                orientation=q,
            ),
            covariance=np.eye(6),
            source="test",
        )
        for i in range(n)
    ]
    return Trajectory(estimates)


class TestUmeyama:
    def test_recovers_known_rotation_and_translation_exactly(self) -> None:
        rng = np.random.default_rng(0)
        src = rng.normal(size=(3, 20))
        R_true = _rotz(0.37)
        t_true = np.array([1.5, -2.0, 0.75], dtype=np.float64)
        dst = R_true @ src + t_true[:, None]

        R, t, scale = umeyama_alignment(src, dst, with_scale=False)

        assert scale == pytest.approx(1.0)
        np.testing.assert_allclose(R, R_true, atol=1e-12)
        np.testing.assert_allclose(t, t_true, atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)

    def test_recovers_under_small_gaussian_noise(self) -> None:
        rng = np.random.default_rng(1)
        src = rng.normal(size=(3, 50))
        R_true = _rotz(-0.55)
        t_true = np.array([-0.3, 0.8, 1.1], dtype=np.float64)
        dst = R_true @ src + t_true[:, None] + rng.normal(scale=1e-3, size=src.shape)

        R, t, scale = umeyama_alignment(src, dst, with_scale=False)

        assert scale == pytest.approx(1.0)
        np.testing.assert_allclose(R, R_true, atol=5e-3)
        np.testing.assert_allclose(t, t_true, atol=5e-3)

    def test_reflection_case_yields_proper_rotation(self) -> None:
        """When SVD prefers a reflection, the Kabsch/Umeyama S fix keeps det(R)=+1."""
        src = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        # Reflect across XY then apply a proper rotation + translation.
        reflect = np.diag([1.0, 1.0, -1.0])
        R_proper = _rotz(0.4)
        t_true = np.array([0.2, -0.1, 0.3], dtype=np.float64)
        dst = R_proper @ (reflect @ src) + t_true[:, None]

        # Confirm the uncorrected SVD product is improper on this data.
        mean_src = src.mean(axis=1)
        mean_dst = dst.mean(axis=1)
        cov = ((dst - mean_dst[:, None]) @ (src - mean_src[:, None]).T) / src.shape[1]
        U, _, Vt = np.linalg.svd(cov)
        assert np.linalg.det(U @ Vt) == pytest.approx(-1.0, abs=1e-12)

        R, _t, scale = umeyama_alignment(src, dst, with_scale=False)

        assert scale == pytest.approx(1.0)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


class TestTrajectoryAlignment:
    def test_self_alignment_yields_zero_position_error(self) -> None:
        rng = np.random.default_rng(2)
        positions = np.cumsum(rng.normal(scale=0.05, size=(30, 3)), axis=0)
        traj = _traj_from_positions(positions)

        for mode in ("none", "origin", "umeyama"):
            errs = position_errors(traj, traj, align_mode=mode)  # type: ignore[arg-type]
            np.testing.assert_allclose(errs.euclidean, 0.0, atol=1e-12)
            np.testing.assert_allclose(errs.horizontal, 0.0, atol=1e-12)
            np.testing.assert_allclose(errs.vertical, 0.0, atol=1e-12)
            assert ate_rmse(traj, traj, align_mode=mode) == pytest.approx(0.0, abs=1e-12)  # type: ignore[arg-type]

    def test_associate_respects_max_dt(self) -> None:
        est = _traj_from_positions(
            np.zeros((3, 3)), timestamps=np.array([0.0, 0.1, 0.2])
        )
        gt = _traj_from_positions(
            np.zeros((3, 3)), timestamps=np.array([0.0, 0.15, 1.0])
        )
        pairs = associate(est, gt, max_dt=0.06)
        assert pairs == [(0, 0), (1, 1)]


class TestEvoCrossCheck:
    def test_ate_matches_evo_to_1e_6(self) -> None:
        pytest.importorskip("evo")
        from evo.core import metrics as evo_metrics
        from evo.core import sync
        from evo.core.trajectory import PoseTrajectory3D

        rng = np.random.default_rng(3)
        n = 40
        timestamps = np.arange(n, dtype=np.float64) * 0.05
        gt_pos = np.cumsum(rng.normal(scale=0.02, size=(n, 3)), axis=0)
        R = _rotz(0.25)
        t = np.array([0.5, -0.25, 0.1], dtype=np.float64)
        est_pos = (R @ gt_pos.T).T + t + rng.normal(scale=1e-4, size=gt_pos.shape)

        gt = _traj_from_positions(gt_pos, timestamps=timestamps)
        est = _traj_from_positions(est_pos, timestamps=timestamps)

        cairn_ate = ate_rmse(est, gt, align_mode="umeyama", with_scale=False)

        q_wxyz = np.repeat([[1.0, 0.0, 0.0, 0.0]], n, axis=0)
        traj_gt = PoseTrajectory3D(
            positions_xyz=gt_pos,
            orientations_quat_wxyz=q_wxyz.copy(),
            timestamps=timestamps,
        )
        traj_est = PoseTrajectory3D(
            positions_xyz=est_pos,
            orientations_quat_wxyz=q_wxyz.copy(),
            timestamps=timestamps,
        )
        traj_gt, traj_est = sync.associate_trajectories(traj_gt, traj_est, max_diff=0.02)
        traj_est.align(traj_gt, correct_scale=False)

        ape = evo_metrics.APE(evo_metrics.PoseRelation.translation_part)
        ape.process_data((traj_gt, traj_est))
        evo_ate = float(ape.get_statistic(evo_metrics.StatisticsType.rmse))

        assert cairn_ate == pytest.approx(evo_ate, abs=1e-6), (
            f"cairn ATE {cairn_ate} vs evo ATE {evo_ate}"
        )
