"""Synthetic point-cloud tests for ICP-Hessian degeneracy detection."""

from __future__ import annotations

import numpy as np
import pytest

from cairn._internal.detectors.degeneracy import (
    DegeneracyDetector,
    localizability,
    point_to_plane_hessian,
)


def _corridor_cloud(
    rng: np.random.Generator,
    *,
    n: int = 2500,
    length: float = 12.0,
    width: float = 2.0,
    height: float = 2.5,
) -> np.ndarray:
    """Axis-aligned corridor along +x (walls at y=±w/2, floor/ceiling)."""
    pts: list[np.ndarray] = []
    n_wall = n // 4
    # +y / -y walls
    for y in (width / 2.0, -width / 2.0):
        x = rng.uniform(0.0, length, size=n_wall)
        z = rng.uniform(0.0, height, size=n_wall)
        pts.append(np.column_stack([x, np.full(n_wall, y), z]))
    # floor / ceiling
    for z in (0.0, height):
        x = rng.uniform(0.0, length, size=n_wall)
        y = rng.uniform(-width / 2.0, width / 2.0, size=n_wall)
        pts.append(np.column_stack([x, y, np.full(n_wall, z)]))
    cloud = np.vstack(pts)
    return cloud + rng.normal(scale=0.005, size=cloud.shape)


def _plane_cloud(rng: np.random.Generator, *, n: int = 2500) -> np.ndarray:
    """Horizontal plane z=0 (unconstrained in x and y translation)."""
    x = rng.uniform(-5.0, 5.0, size=n)
    y = rng.uniform(-5.0, 5.0, size=n)
    z = np.zeros(n)
    cloud = np.column_stack([x, y, z])
    return cloud + rng.normal(scale=0.005, size=cloud.shape)


def _corner_cloud(rng: np.random.Generator, *, n: int = 2700) -> np.ndarray:
    """Three mutually orthogonal planes (fully constrained translations)."""
    n_face = n // 3
    faces: list[np.ndarray] = []
    # z=0
    faces.append(
        np.column_stack(
            [
                rng.uniform(0.0, 4.0, n_face),
                rng.uniform(0.0, 4.0, n_face),
                np.zeros(n_face),
            ]
        )
    )
    # y=0
    faces.append(
        np.column_stack(
            [
                rng.uniform(0.0, 4.0, n_face),
                np.zeros(n_face),
                rng.uniform(0.0, 4.0, n_face),
            ]
        )
    )
    # x=0
    faces.append(
        np.column_stack(
            [
                np.zeros(n_face),
                rng.uniform(0.0, 4.0, n_face),
                rng.uniform(0.0, 4.0, n_face),
            ]
        )
    )
    cloud = np.vstack(faces)
    return cloud + rng.normal(scale=0.005, size=cloud.shape)


def _pair(cloud: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Identity registration pair with independent sensor noise."""
    scan = cloud + rng.normal(scale=0.003, size=cloud.shape)
    prev = cloud + rng.normal(scale=0.003, size=cloud.shape)
    return scan, prev


class TestLocalizabilitySynthetic:
    def test_corridor_unconstrained_along_axis(self) -> None:
        # Expected: translation along +x is weak; y/z walls+floor constrain lateral/vertical.
        rng = np.random.default_rng(0)
        scan, prev = _pair(_corridor_cloud(rng), rng)
        det = DegeneracyDetector(score_threshold=0.3)
        result = det.update(scan, prev)
        assert result.scores.shape == (6,)
        assert float(result.scores[0]) < 0.3  # x free
        assert float(result.scores[1]) > 0.5  # y constrained
        assert float(result.scores[2]) > 0.5  # z constrained
        assert "x" in result.degenerate_axes
        assert result.degenerate is True

    def test_plane_two_unconstrained_translations(self) -> None:
        # Expected: only z translation strongly constrained among xyz.
        rng = np.random.default_rng(1)
        scan, prev = _pair(_plane_cloud(rng), rng)
        det = DegeneracyDetector(score_threshold=0.3)
        result = det.update(scan, prev)
        assert float(result.scores[2]) > 0.7  # z
        assert float(result.scores[0]) < 0.3  # x free
        assert float(result.scores[1]) < 0.3  # y free
        assert "x" in result.degenerate_axes
        assert "y" in result.degenerate_axes

    def test_corner_fully_constrained(self) -> None:
        # Expected: all three translational axes well constrained.
        rng = np.random.default_rng(2)
        scan, prev = _pair(_corner_cloud(rng), rng)
        det = DegeneracyDetector(score_threshold=0.25)
        result = det.update(scan, prev)
        assert float(result.scores[0]) > 0.4
        assert float(result.scores[1]) > 0.4
        assert float(result.scores[2]) > 0.4
        # No translational axis should be flagged degenerate.
        assert "x" not in result.degenerate_axes
        assert "y" not in result.degenerate_axes
        assert "z" not in result.degenerate_axes


def test_localizability_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="\\(6, 6\\)"):
        localizability(np.eye(3))


def test_hessian_positive_semidefinite_on_corner() -> None:
    rng = np.random.default_rng(3)
    scan, prev = _pair(_corner_cloud(rng), rng)
    H = point_to_plane_hessian(scan, prev)
    eig = np.linalg.eigvalsh(H)
    assert float(eig.min()) >= -1e-8
