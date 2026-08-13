"""Trajectory error metrics (ATE / RPE and per-axis position error).

Units: metres (position errors), seconds (timestamps / windows).
Frames: estimate and ground truth must share a common frame after alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cairn.config import CairnConfig
from cairn.metrics.alignment import (
    AlignMode,
    _pose_to_se3,
    _se3_inverse,
    align_trajectories,
    associate,
)
from cairn.types import Trajectory

__all__ = [
    "PositionErrorResult",
    "ate_rmse",
    "position_errors",
    "rte",
]


def _default_align_dt() -> float:
    return CairnConfig().association.alignment_max_dt_s


@dataclass(frozen=True, slots=True)
class PositionErrorResult:
    """Per-matched-timestamp position errors in metres."""

    euclidean: NDArray[np.float64]
    horizontal: NDArray[np.float64]
    vertical: NDArray[np.float64]
    matches: tuple[tuple[int, int], ...]


def position_errors(
    est: Trajectory,
    gt: Trajectory,
    align_mode: AlignMode = "none",
    *,
    max_dt: float | None = None,
    with_scale: bool = False,
) -> PositionErrorResult:
    """Per-timestamp Euclidean / horizontal / vertical position error (metres).

    Horizontal is ``hypot(dx, dy)``; vertical is ``|dz|``, assuming z-up.
    """
    if max_dt is None:
        max_dt = _default_align_dt()
    aligned = align_trajectories(
        est, gt, align_mode, max_dt=max_dt, with_scale=with_scale
    )
    matches = associate(aligned, gt, max_dt=max_dt)
    if not matches:
        raise ValueError("No timestamp matches within max_dt")

    euclidean = np.empty(len(matches), dtype=np.float64)
    horizontal = np.empty(len(matches), dtype=np.float64)
    vertical = np.empty(len(matches), dtype=np.float64)
    for k, (i, j) in enumerate(matches):
        d = aligned[i].pose.position - gt[j].pose.position
        euclidean[k] = float(np.linalg.norm(d))
        horizontal[k] = float(np.hypot(d[0], d[1]))
        vertical[k] = float(abs(d[2]))

    return PositionErrorResult(
        euclidean=euclidean,
        horizontal=horizontal,
        vertical=vertical,
        matches=tuple(matches),
    )


def ate_rmse(
    est: Trajectory,
    gt: Trajectory,
    align_mode: AlignMode = "umeyama",
    *,
    max_dt: float | None = None,
    with_scale: bool = False,
) -> float:
    """Absolute trajectory error RMSE after optional alignment.

    Parameters
    ----------
    est, gt:
        Estimated and ground-truth trajectories (positions in metres).
    align_mode:
        ``none`` | ``origin`` | ``umeyama`` (SE3 / Sim3 gauge removal).
    max_dt:
        Max |Δt| for timestamp association in seconds (default from config).
    with_scale:
        If True, Umeyama estimates a scale (Sim3).

    Returns
    -------
    float
        RMSE in metres.

    Raises
    ------
    ValueError
        No timestamp matches within ``max_dt``, or Umeyama needs ≥3 matches.
    """
    errs = position_errors(
        est, gt, align_mode, max_dt=max_dt, with_scale=with_scale
    )
    return float(np.sqrt(np.mean(errs.euclidean**2)))


def rte(
    est: Trajectory,
    gt: Trajectory,
    align_mode: AlignMode = "none",
    *,
    window: float = 1.0,
    max_dt: float | None = None,
    with_scale: bool = False,
    window_tol: float | None = None,
) -> NDArray[np.float64]:
    """Relative trajectory error (translation) over a time window (metres).

    For each matched pair of poses separated by approximately ``window``
    seconds, compares relative SE(3) motion and returns the translational
    magnitude of the pose residual
    ``inv(T_gt_i^{-1} T_gt_j) (T_est_i^{-1} T_est_j)``.
    """
    if window <= 0.0:
        raise ValueError(f"window must be positive, got {window}")
    if max_dt is None:
        max_dt = _default_align_dt()
    if window_tol is None:
        window_tol = max_dt

    aligned = align_trajectories(
        est, gt, align_mode, max_dt=max_dt, with_scale=with_scale
    )
    matches = associate(aligned, gt, max_dt=max_dt)
    if len(matches) < 2:
        raise ValueError("Need at least 2 matched poses for RTE")

    # Build synchronized pose lists at matched timestamps.
    T_est = [_pose_to_se3(aligned[i].pose) for i, _ in matches]
    T_gt = [_pose_to_se3(gt[j].pose) for _, j in matches]
    times = np.array(
        [aligned[i].pose.timestamp for i, _ in matches], dtype=np.float64
    )

    errors: list[float] = []
    for i, t_i in enumerate(times):
        target = float(t_i) + float(window)
        # Nearest later sample near t_i + window.
        j = int(np.argmin(np.abs(times - target)))
        if j <= i:
            continue
        if abs(float(times[j]) - target) > window_tol:
            continue
        rel_est = _se3_inverse(T_est[i]) @ T_est[j]
        rel_gt = _se3_inverse(T_gt[i]) @ T_gt[j]
        err = _se3_inverse(rel_gt) @ rel_est
        errors.append(float(np.linalg.norm(err[:3, 3])))

    if not errors:
        raise ValueError(
            f"No pose pairs found near window={window}s (tol={window_tol})"
        )
    return np.asarray(errors, dtype=np.float64)
