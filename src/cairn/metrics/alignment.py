"""Trajectory association and SE(3)/Sim(3) alignment (public metrics helpers).

Units: metres (positions), seconds (timestamps). Rotations are SO(3) / unit
quaternions (w-first).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from cairn.config import CairnConfig
from cairn.types import Pose, PoseEstimate, Trajectory

__all__ = [
    "AlignMode",
    "align_trajectories",
    "associate",
    "umeyama_alignment",
]

AlignMode = Literal["none", "origin", "umeyama"]


def _default_align_dt() -> float:
    return CairnConfig().association.alignment_max_dt_s


def _quat_wxyz_to_rotmat(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert w-first unit quaternion ``(w, x, y, z)`` to a 3×3 rotation matrix."""
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotmat_to_quat_wxyz(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a 3×3 rotation matrix to a w-first unit quaternion."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = float(m00 + m11 + m22)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _pose_to_se3(pose: Pose) -> NDArray[np.float64]:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_wxyz_to_rotmat(pose.orientation)
    T[:3, 3] = pose.position
    return T


def _se3_inverse(T: NDArray[np.float64]) -> NDArray[np.float64]:
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def _apply_sim3(
    traj: Trajectory,
    R: NDArray[np.float64],
    t: NDArray[np.float64],
    scale: float,
) -> Trajectory:
    """Map each position ``p -> scale * R @ p + t`` and rotate orientations by ``R``."""
    estimates: list[PoseEstimate] = []
    for est in traj:
        p = est.pose.position
        R_pose = _quat_wxyz_to_rotmat(est.pose.orientation)
        new_pose = Pose(
            timestamp=est.pose.timestamp,
            position=scale * (R @ p) + t,
            orientation=_rotmat_to_quat_wxyz(R @ R_pose),
            frame=est.pose.frame,
        )
        estimates.append(
            PoseEstimate(
                pose=new_pose,
                covariance=est.covariance.copy(),
                source=est.source,
            )
        )
    return Trajectory(estimates)


def associate(
    est: Trajectory,
    gt: Trajectory,
    max_dt: float | None = None,
) -> list[tuple[int, int]]:
    """Greedy nearest-neighbour timestamp association.

    For each estimate timestamp, the nearest ground-truth sample within
    ``max_dt`` is proposed; candidate pairs are then accepted greedily in
    ascending ``|Δt|`` so each index is used at most once.

    Returns
    -------
    list of ``(est_index, gt_index)`` pairs in ascending estimate-index order.
    """
    if max_dt is None:
        max_dt = _default_align_dt()
    if max_dt < 0.0:
        raise ValueError(f"max_dt must be non-negative, got {max_dt}")

    t_est = est.timestamps()
    t_gt = gt.timestamps()
    # O(N log M) candidate generation via binary search (not N×M).
    candidates: list[tuple[float, int, int]] = []
    for i, te in enumerate(t_est):
        j = int(np.searchsorted(t_gt, te, side="left"))
        best_j: int | None = None
        best_dt = float("inf")
        for k in (j - 1, j):
            if 0 <= k < t_gt.size:
                dt = abs(float(te) - float(t_gt[k]))
                if dt < best_dt:
                    best_dt = dt
                    best_j = k
        if best_j is not None and best_dt <= max_dt:
            candidates.append((best_dt, i, best_j))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used_est: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, i, j in candidates:
        if i in used_est or j in used_gt:
            continue
        used_est.add(i)
        used_gt.add(j)
        matches.append((i, j))

    matches.sort(key=lambda pair: pair[0])
    return matches


def umeyama_alignment(
    src: NDArray[np.float64],
    dst: NDArray[np.float64],
    with_scale: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Closed-form Umeyama (1991) alignment via SVD.

    Parameters
    ----------
    src, dst:
        Point sets of shape ``(3, N)``. Finds ``R``, ``t``, ``s`` minimizing
        ``|| dst - (s * R @ src + t) ||``.
    with_scale:
        If True, estimate Sim(3) scale; otherwise scale is fixed to 1 (SE(3)).

    Returns
    -------
    R, t, s
        Rotation (3×3), translation (3,), and scale.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape:
        raise ValueError(f"src and dst must share shape, got {src.shape} vs {dst.shape}")
    if src.ndim != 2 or src.shape[0] != 3:
        raise ValueError(f"src/dst must have shape (3, N), got {src.shape}")
    _m, n = src.shape
    if n < 3:
        raise ValueError(f"Umeyama alignment needs at least 3 points, got {n}")

    mean_src = src.mean(axis=1)
    mean_dst = dst.mean(axis=1)
    src_c = src - mean_src[:, None]
    dst_c = dst - mean_dst[:, None]

    # Variance of source (Umeyama eq. 36).
    sigma_src = float(np.sum(src_c * src_c) / n)

    # Covariance (Umeyama eq. 38): (1/n) * dst_c @ src_c.T
    cov = (dst_c @ src_c.T) / n

    U, D, Vt = np.linalg.svd(cov)
    if np.count_nonzero(D > np.finfo(D.dtype).eps) < 2:
        raise ValueError("Degenerate covariance rank; Umeyama alignment is not possible")

    # Reflection fix (Umeyama / Kabsch): ensure det(R) = +1.
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0.0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    if with_scale:
        if sigma_src == 0.0:
            raise ValueError("Source variance is zero; cannot estimate scale")
        scale = float(np.trace(np.diag(D) @ S) / sigma_src)
    else:
        scale = 1.0

    t = mean_dst - scale * (R @ mean_src)
    return R, t, scale


def align_trajectories(
    est: Trajectory,
    gt: Trajectory,
    mode: AlignMode,
    *,
    max_dt: float | None = None,
    with_scale: bool = False,
) -> Trajectory:
    """Align ``est`` to ``gt`` according to ``mode``.

    Modes
    -----
    none:
        Return ``est`` unchanged (new Trajectory wrapping the same estimates).
    origin:
        SE(3) first-pose alignment: ``T = T_gt0 @ inv(T_est0)``.
    umeyama:
        Umeyama SE(3)/Sim(3) on associated positions, then apply to all poses.
    """
    if mode == "none":
        return Trajectory(est.estimates)

    if mode == "origin":
        T_est0 = _pose_to_se3(est[0].pose)
        T_gt0 = _pose_to_se3(gt[0].pose)
        T_align = T_gt0 @ _se3_inverse(T_est0)
        return _apply_sim3(
            est,
            R=T_align[:3, :3],
            t=T_align[:3, 3],
            scale=1.0,
        )

    if max_dt is None:
        max_dt = _default_align_dt()

    if mode == "umeyama":
        matches = associate(est, gt, max_dt=max_dt)
        if len(matches) < 3:
            raise ValueError(
                f"Umeyama alignment needs at least 3 matched poses, got {len(matches)}"
            )
        src = np.column_stack([est[i].pose.position for i, _ in matches])
        dst = np.column_stack([gt[j].pose.position for _, j in matches])
        R, t, scale = umeyama_alignment(src, dst, with_scale=with_scale)
        return _apply_sim3(est, R=R, t=t, scale=scale)

    raise ValueError(f"Unknown align mode {mode!r}; expected 'none', 'origin', or 'umeyama'")
