#!/usr/bin/env python3
"""Feed a pose source Cairn has never heard of.

Source names are opaque provenance tags, not a plugin registry. If your
estimator is called ``terrain_relative`` or ``my_ekf``, wrap each sample as a
:class:`~cairn.types.PoseEstimate` and pass it under that string key.

This demo:

1. Loads the **synthetic** quickstart estimate (``scripts/make_quickstart_data.py``)
   as one known stream (``compslam``).
2. Builds a second stream ``terrain_relative`` that Cairn has never seen: a
   simple lateral offset of the same poses (stand-in for "your" filter).
3. Runs the consistency monitor on the pair.

Units and frames:

* Timestamps: seconds.
* Positions: metres in frame ``map``.
* Orientation: w-first ``(w, x, y, z)`` after TUM load.
* Covariance: m² / rad² on ``[x, y, z, roll, pitch, yaw]``.
* Protection levels: metres.

Run from the repo root::

    python examples/custom_source.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cairn import CairnConfig, CairnMonitor, Pose, PoseEstimate
from cairn._internal.io.tum import load_tum

_HERE = Path(__file__).resolve().parent
_EST_TUM = _HERE / "quickstart" / "est.tum"

# Claimed uncertainty for both streams (metres / radians).
_STD_XY_M = 0.08
_STD_Z_M = 0.08
_STD_ROT_RAD = 0.05

# Synthetic lateral bias for the "never seen" source (metres).
_CUSTOM_Y_OFFSET_M = 0.15


def _covariance_m2_rad2() -> np.ndarray:
    return np.diag(
        [
            _STD_XY_M**2,
            _STD_XY_M**2,
            _STD_Z_M**2,
            _STD_ROT_RAD**2,
            _STD_ROT_RAD**2,
            _STD_ROT_RAD**2,
        ]
    )


def wrap_custom_estimator_sample(
    *,
    timestamp_s: float,
    position_m: np.ndarray,
    orientation_wxyz: np.ndarray,
    covariance_m2_rad2: np.ndarray,
    source_name: str,
    frame: str,
) -> PoseEstimate:
    """Convert raw estimator output into the only type Cairn understands.

    Parameters use explicit units: seconds, metres, w-first quaternion,
    covariance in m²/rad². ``source_name`` can be any non-empty string.
    """
    pose = Pose(
        timestamp=timestamp_s,
        position=np.asarray(position_m, dtype=np.float64),
        orientation=np.asarray(orientation_wxyz, dtype=np.float64),
        frame=frame,
    )
    return PoseEstimate(pose=pose, covariance=covariance_m2_rad2.copy(), source=source_name)


def main() -> None:
    if not _EST_TUM.is_file():
        raise SystemExit(
            "missing examples/quickstart/est.tum; run:\n"
            "  python scripts/make_quickstart_data.py --seed 0"
        )

    cov = _covariance_m2_rad2()
    # Synthetic quickstart estimate (metres, frame map).
    compslam = list(
        load_tum(
            _EST_TUM,
            default_covariance=cov,
            source="compslam",
            frame="map",
        )
    )
    demo = compslam[:40]

    print(
        f"custom_source.py: pairing compslam with terrain_relative "
        f"({len(demo)} epochs from synthetic {_EST_TUM.name})"
    )
    print(
        "Note: source name 'terrain_relative' is not special-cased anywhere "
        "in Cairn; any non-empty string works."
    )

    cfg = CairnConfig()
    with CairnMonitor(cfg) as mon:
        for est in demo:
            t_s = float(est.pose.timestamp)

            # --- Your filter would produce these numbers somehow. ---
            # Here we invent them: same pose as compslam with +0.15 m in y.
            custom_xyz_m = est.pose.position.copy()
            custom_xyz_m[1] += _CUSTOM_Y_OFFSET_M

            custom = wrap_custom_estimator_sample(
                timestamp_s=t_s,
                position_m=custom_xyz_m,
                orientation_wxyz=est.pose.orientation,
                covariance_m2_rad2=cov,
                source_name="terrain_relative",  # never registered; that is fine
                frame="map",  # must match the other source's navigation frame
            )

            report = mon.update(
                t_s,
                {
                    "compslam": est,
                    "terrain_relative": custom,
                },
            )
            print(
                f"t={t_s:.3f}s  {report.state.name:12s}  "
                f"HPL={report.protection_level_horizontal:.3f}m  "
                f"sources={list(mon.state.sources_present_last)}"
            )


if __name__ == "__main__":
    main()
