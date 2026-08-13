"""Tests for cairn core domain types."""

from __future__ import annotations

import numpy as np
import pytest

from cairn.types import Pose, PoseEstimate, Trajectory


def _quat_wxyz(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array([w, x, y, z], dtype=np.float64)


def _identity_quat() -> np.ndarray:
    return _quat_wxyz(1.0, 0.0, 0.0, 0.0)


def _rotz_quat(angle_rad: float) -> np.ndarray:
    """w-first quaternion for a rotation of ``angle_rad`` about +z."""
    half = 0.5 * angle_rad
    return _quat_wxyz(np.cos(half), 0.0, 0.0, np.sin(half))


def _angle_about_z(q: np.ndarray) -> float:
    """Recover rotation angle about +z from a w-first quaternion (radians)."""
    q = q / np.linalg.norm(q)
    # Prefer positive w for a unique angle in (-pi, pi].
    if q[0] < 0.0:
        q = -q
    return float(2.0 * np.arctan2(q[3], q[0]))


def _pose(t: float, position: list[float], orientation: np.ndarray) -> Pose:
    return Pose(timestamp=t, position=np.asarray(position), orientation=orientation)


def _estimate(pose: Pose) -> PoseEstimate:
    return PoseEstimate(
        pose=pose,
        covariance=np.eye(6),
        source="test",
    )


class TestShapeValidation:
    def test_pose_rejects_bad_position_shape(self) -> None:
        with pytest.raises(ValueError, match=r"position must have shape \(3,\)"):
            Pose(
                timestamp=0.0,
                position=np.zeros(2),
                orientation=_identity_quat(),
            )

    def test_pose_rejects_bad_orientation_shape(self) -> None:
        with pytest.raises(ValueError, match=r"orientation must have shape \(4,\)"):
            Pose(
                timestamp=0.0,
                position=np.zeros(3),
                orientation=np.zeros(3),
            )

    def test_pose_estimate_rejects_bad_covariance_shape(self) -> None:
        pose = _pose(0.0, [0.0, 0.0, 0.0], _identity_quat())
        with pytest.raises(ValueError, match=r"covariance must have shape \(6, 6\)"):
            PoseEstimate(pose=pose, covariance=np.eye(3), source="imu")


class TestTrajectoryInterpolate:
    def test_interpolate_at_exact_sample_returns_same_pose(self) -> None:
        p0 = _pose(0.0, [0.0, 0.0, 0.0], _identity_quat())
        p1 = _pose(1.0, [1.0, 0.0, 0.0], _rotz_quat(np.pi / 2))
        traj = Trajectory([_estimate(p0), _estimate(p1)])

        result = traj.interpolate_at(1.0)

        assert result.timestamp == pytest.approx(1.0)
        np.testing.assert_allclose(result.position, p1.position)
        np.testing.assert_allclose(result.orientation, p1.orientation)

    def test_slerp_midpoint_of_90_deg_is_45_deg(self) -> None:
        p0 = _pose(0.0, [0.0, 0.0, 0.0], _identity_quat())
        p1 = _pose(2.0, [2.0, 0.0, 0.0], _rotz_quat(np.pi / 2))
        traj = Trajectory([_estimate(p0), _estimate(p1)])

        mid = traj.interpolate_at(1.0)

        assert mid.timestamp == pytest.approx(1.0)
        np.testing.assert_allclose(mid.position, [1.0, 0.0, 0.0])
        assert _angle_about_z(mid.orientation) == pytest.approx(np.pi / 4, abs=1e-9)

    def test_interpolate_out_of_bounds_raises(self) -> None:
        p0 = _pose(0.0, [0.0, 0.0, 0.0], _identity_quat())
        p1 = _pose(1.0, [1.0, 0.0, 0.0], _identity_quat())
        traj = Trajectory([_estimate(p0), _estimate(p1)])

        with pytest.raises(ValueError, match="outside trajectory bounds"):
            traj.interpolate_at(-0.1)
        with pytest.raises(ValueError, match="outside trajectory bounds"):
            traj.interpolate_at(1.1)
