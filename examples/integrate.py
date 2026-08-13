#!/usr/bin/env python3
"""Two pose sources → Cairn integrity reports. Units: m, s; quat w-first (w,x,y,z);
covariance 6×6 [x,y,z,roll,pitch,yaw] (m²/rad²). Both sources share one nav frame."""

from __future__ import annotations

import numpy as np

from cairn import CairnConfig, CairnMonitor, Pose, PoseEstimate

_Q = np.array([1.0, 0.0, 0.0, 0.0])
_COV = np.diag([0.05**2, 0.05**2, 0.05**2, 0.01, 0.01, 0.01])


def make_estimate(t: float, xyz: list[float], source: str) -> PoseEstimate:
    """One estimate; raises ValueError on bad shapes / non-finite values."""
    return PoseEstimate(
        pose=Pose(timestamp=t, position=np.asarray(xyz, float), orientation=_Q.copy()),
        covariance=_COV.copy(),
        source=source,
    )


def main() -> None:
    # consistency needs ≥2 independent sources. Swap make_estimate for your estimators.
    mon = CairnMonitor(CairnConfig())  # or CairnConfig.from_yaml("configs/default.yaml")
    for k in range(5):
        t = 0.1 * k
        report = mon.update(
            t,
            {
                "lidar": make_estimate(t, [0.01 * k, 0.0, 0.0], "lidar"),
                "vio": make_estimate(t, [0.01 * k + 0.02, 0.0, 0.0], "vio"),
            },
        )
        # state: NOMINAL | DEGRADED | DO_NOT_TRUST; protection levels in metres
        print(f"t={t:.1f}  {report.state.name:12s}  HPL={report.protection_level_horizontal:.3f}m")
    mon.close()


if __name__ == "__main__":
    main()
