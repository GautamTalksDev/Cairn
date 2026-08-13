"""Tests for cairn trajectory I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cairn._internal.io import load_ros2_bag, load_tum, save_tum
from cairn.types import Pose, PoseEstimate, Trajectory

FIXTURES = Path(__file__).parent / "fixtures"


def _quat_wxyz(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array([w, x, y, z], dtype=np.float64)


def assert_quaternions_close(
    q_a: np.ndarray,
    q_b: np.ndarray,
    *,
    atol: float = 1e-9,
) -> None:
    """Assert unit quaternions represent the same rotation (q and -q equivalent)."""
    if np.allclose(q_a, q_b, atol=atol, rtol=0.0):
        return
    if np.allclose(q_a, -q_b, atol=atol, rtol=0.0):
        return
    raise AssertionError(
        f"quaternions differ beyond sign ambiguity:\n  {q_a}\n  {q_b}"
    )


def _synthetic_trajectory(n: int = 10) -> Trajectory:
    estimates: list[PoseEstimate] = []
    for i in range(n):
        t = float(i) * 0.1
        # Varying yaw; include a negative-w quaternion so sign handling matters.
        angle = 0.2 * i
        if i == 7:
            # Same rotation as +q, written with opposite hemisphere.
            half = 0.5 * angle
            q = _quat_wxyz(-np.cos(half), 0.0, 0.0, -np.sin(half))
        else:
            half = 0.5 * angle
            q = _quat_wxyz(np.cos(half), 0.0, 0.0, np.sin(half))
        pose = Pose(
            timestamp=t,
            position=np.array([0.5 * i, 0.1 * i, -0.01 * i], dtype=np.float64),
            orientation=q,
            frame="map",
        )
        estimates.append(
            PoseEstimate(pose=pose, covariance=np.eye(6), source="synthetic")
        )
    return Trajectory(estimates)


class TestQuaternionSignAmbiguity:
    def test_helper_accepts_q_and_neg_q(self) -> None:
        q = _quat_wxyz(0.9238795325, 0.0, 0.0, 0.3826834324)
        assert_quaternions_close(q, q)
        assert_quaternions_close(q, -q)

    def test_helper_rejects_different_rotation(self) -> None:
        """Ensures sign-aware compare is not a no-op that always passes."""
        q_id = _quat_wxyz(1.0, 0.0, 0.0, 0.0)
        q_90 = _quat_wxyz(np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))
        with pytest.raises(AssertionError, match="sign ambiguity"):
            assert_quaternions_close(q_id, q_90)


class TestTumIO:
    def test_round_trip_positions_and_quaternions(self, tmp_path: Path) -> None:
        original = _synthetic_trajectory(10)
        path = tmp_path / "traj.txt"

        save_tum(original, path)
        loaded = load_tum(path)

        assert len(loaded) == len(original)
        for got, exp in zip(loaded, original, strict=True):
            assert got.pose.timestamp == pytest.approx(exp.pose.timestamp)
            np.testing.assert_allclose(
                got.pose.position, exp.pose.position, atol=1e-9, rtol=0.0
            )
            assert_quaternions_close(
                got.pose.orientation, exp.pose.orientation, atol=1e-9
            )

    def test_round_trip_with_negated_quaternion_in_file(self, tmp_path: Path) -> None:
        """File-level q -> -q must still round-trip as the same rotation."""
        original = _synthetic_trajectory(3)
        path = tmp_path / "negated.txt"
        save_tum(original, path)

        lines = path.read_text(encoding="utf-8").splitlines()
        rewritten: list[str] = []
        flipped = False
        for line in lines:
            if line.startswith("#") or not line.strip():
                rewritten.append(line)
                continue
            parts = line.split()
            if not flipped and len(parts) == 8:
                # Negate TUM w-last quaternion in place (same rotation).
                qx, qy, qz, qw = (float(parts[i]) for i in range(4, 8))
                parts[4:8] = [f"{-qx:.9f}", f"{-qy:.9f}", f"{-qz:.9f}", f"{-qw:.9f}"]
                rewritten.append(" ".join(parts))
                flipped = True
            else:
                rewritten.append(line)
        assert flipped
        path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

        loaded = load_tum(path)
        # Direct equality would fail on the flipped sample; sign-aware must pass.
        assert not np.allclose(
            loaded[0].pose.orientation,
            original[0].pose.orientation,
            atol=1e-9,
            rtol=0.0,
        )
        assert_quaternions_close(
            loaded[0].pose.orientation,
            original[0].pose.orientation,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            loaded[0].pose.position, original[0].pose.position, atol=1e-9, rtol=0.0
        )

    def test_load_fixture(self) -> None:
        traj = load_tum(FIXTURES / "sample_tum.txt")
        assert len(traj) == 3
        np.testing.assert_allclose(traj[0].pose.position, [0.0, 0.0, 0.0])
        # Fixture qw=1, qx=qy=qz=0 → w-first (1, 0, 0, 0)
        np.testing.assert_allclose(traj[0].pose.orientation, [1.0, 0.0, 0.0, 0.0])
        assert traj.timestamps()[0] == pytest.approx(0.0)

    def test_default_covariance_is_identity(self) -> None:
        traj = load_tum(FIXTURES / "sample_tum.txt")
        np.testing.assert_array_equal(traj[0].covariance, np.eye(6))


class TestBagStub:
    def test_load_ros2_bag_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="rosbags"):
            load_ros2_bag("/tmp/example.bag", "/odom")
