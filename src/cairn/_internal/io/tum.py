"""TUM trajectory file format.

TUM lines are whitespace-separated::

    timestamp tx ty tz qx qy qz qw

Quaternion order in the file is **w-last** ``(x, y, z, w)``.
Cairn :class:`~cairn.types.Pose` uses **w-first** ``(w, x, y, z)``.
Load/save convert explicitly between these conventions.
"""

from __future__ import annotations

__all__ = [
    "load_tum",
    "save_tum",
]


from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from cairn.types import Pose, PoseEstimate, Trajectory


def _tum_xyzw_to_wxyz(qx: float, qy: float, qz: float, qw: float) -> NDArray[np.float64]:
    """Convert TUM w-last ``(x, y, z, w)`` to Pose w-first ``(w, x, y, z)``."""
    return np.array([qw, qx, qy, qz], dtype=np.float64)


def _wxyz_to_tum_xyzw(q_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert Pose w-first ``(w, x, y, z)`` to TUM w-last ``(x, y, z, w)``."""
    w, x, y, z = (float(v) for v in q_wxyz)
    return np.array([x, y, z, w], dtype=np.float64)


def load_tum(
    path: str | Path,
    *,
    default_covariance: NDArray[np.float64] | None = None,
    source: str = "tum",
    frame: str | None = None,
) -> Trajectory:
    """Load a TUM trajectory file into a :class:`~cairn.types.Trajectory`.

    TUM carries no covariance; each estimate uses ``default_covariance``
    (identity 6×6 when omitted).
    """
    path = Path(path)
    cov = (
        np.eye(6, dtype=np.float64)
        if default_covariance is None
        else np.asarray(default_covariance, dtype=np.float64)
    )
    estimates: list[PoseEstimate] = []

    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(
                    f"{path}:{line_no}: expected 8 fields "
                    f"(timestamp tx ty tz qx qy qz qw), got {len(parts)}"
                )
            ts, tx, ty, tz, qx, qy, qz, qw = (float(p) for p in parts)
            # TUM is w-last (qx qy qz qw); Pose stores w-first (w, x, y, z).
            orientation = _tum_xyzw_to_wxyz(qx, qy, qz, qw)
            pose = Pose(
                timestamp=ts,
                position=np.array([tx, ty, tz], dtype=np.float64),
                orientation=orientation,
                frame=frame,
            )
            estimates.append(
                PoseEstimate(pose=pose, covariance=cov.copy(), source=source)
            )

    if not estimates:
        raise ValueError(f"{path}: no pose lines found")
    return Trajectory(estimates)


def save_tum(traj: Trajectory, path: str | Path) -> None:
    """Write a :class:`~cairn.types.Trajectory` as a TUM trajectory file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# timestamp tx ty tz qx qy qz qw",
        "# quaternion is TUM w-last (x, y, z, w); converted from Pose w-first",
    ]
    for estimate in traj:
        pose = estimate.pose
        tx, ty, tz = (float(v) for v in pose.position)
        # Pose is w-first (w, x, y, z); TUM expects w-last (qx qy qz qw).
        qx, qy, qz, qw = (float(v) for v in _wxyz_to_tum_xyzw(pose.orientation))
        lines.append(
            f"{pose.timestamp:.9f} {tx:.9f} {ty:.9f} {tz:.9f} "
            f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
