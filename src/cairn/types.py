"""Core domain types for Cairn (public).

Conventions
-----------
* **Time** — Unix-like seconds as ``float`` (not datetime).
* **Position** — metres ``(x, y, z)`` in the named ``frame`` (or implied world).
* **Orientation** — unit quaternion **w-first** ``(w, x, y, z)``: rotation from
  body into ``frame``.
* **Covariance** — 6×6, ordered ``[x, y, z, roll, pitch, yaw]``
  (m² / rad² and cross terms), local tangent / small-angle Euler about the estimate.
* **Protection levels** — metres (horizontal / vertical).

Raises
------
ValueError
    Non-finite values, wrong shapes, empty / unsorted trajectories, or
    non-positive-definite covariances at construction time.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import overload

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Pose",
    "PoseEstimate",
    "Trajectory",
    "IntegrityState",
    "IntegrityReport",
]


def _as_float_array(name: str, value: object, shape: tuple[int, ...]) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return np.array(arr, dtype=np.float64, copy=True)


def _normalize_quat(q: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("orientation quaternion must be non-zero")
    return q / norm


def _slerp(
    q0: NDArray[np.float64],
    q1: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    """Spherical linear interpolation of w-first unit quaternions."""
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    # Take the short path on the quaternion sphere.
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return _normalize_quat(result)
    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * alpha
    s0 = float(np.sin(theta_0 - theta) / sin_theta_0)
    s1 = float(np.sin(theta) / sin_theta_0)
    return s0 * q0 + s1 * q1


@dataclass(frozen=True, slots=True)
class Pose:
    """A timed pose in space.

    Parameters
    ----------
    timestamp:
        Time in seconds.
    position:
        Translation ``(x, y, z)`` in metres.
    orientation:
        Unit quaternion ``(w, x, y, z)`` (w-first).
    frame:
        Optional name of the reference frame for position/orientation.
    """

    timestamp: float
    position: NDArray[np.float64]
    orientation: NDArray[np.float64]
    frame: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", float(self.timestamp))
        if not np.isfinite(self.timestamp):
            raise ValueError(f"timestamp must be finite, got {self.timestamp!r}")
        object.__setattr__(
            self, "position", _as_float_array("position", self.position, (3,))
        )
        if not np.all(np.isfinite(self.position)):
            raise ValueError(
                f"position contains non-finite values at timestamp={self.timestamp:.9f}"
            )
        orientation = _as_float_array("orientation", self.orientation, (4,))
        if not np.all(np.isfinite(orientation)):
            raise ValueError(
                f"orientation contains non-finite values at timestamp={self.timestamp:.9f}"
            )
        object.__setattr__(self, "orientation", _normalize_quat(orientation))


@dataclass(frozen=True, slots=True)
class PoseEstimate:
    """A :class:`Pose` with uncertainty and provenance.

    Parameters
    ----------
    pose:
        Estimated pose (timestamp, position, orientation, frame).
    covariance:
        6×6 covariance, ordered ``[x, y, z, roll, pitch, yaw]``
        (metres² / rad² and cross terms).
    source:
        Sensor or algorithm that produced this estimate.
    """

    pose: Pose
    covariance: NDArray[np.float64]
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "covariance",
            _as_float_array("covariance", self.covariance, (6, 6)),
        )
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty string")
        if not np.all(np.isfinite(self.covariance)):
            raise ValueError(
                f"covariance contains non-finite values; "
                f"source={self.source!r}, timestamp={self.pose.timestamp:.9f}"
            )
        # Symmetric positive-definite (numerical floor).
        sym = 0.5 * (self.covariance + self.covariance.T)
        eig = np.linalg.eigvalsh(sym)
        if float(eig.min()) < -1e-10:
            raise ValueError(
                f"covariance is not positive-definite "
                f"(min eigenvalue={float(eig.min()):.3e}); "
                f"source={self.source!r}, timestamp={self.pose.timestamp:.9f}"
            )
        object.__setattr__(self, "covariance", np.array(sym, dtype=np.float64, copy=True))


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Ordered sequence of :class:`PoseEstimate` samples.

    Samples must be strictly increasing in ``pose.timestamp``.
    """

    estimates: tuple[PoseEstimate, ...]

    def __init__(self, estimates: Sequence[PoseEstimate]) -> None:
        items = tuple(estimates)
        if not items:
            raise ValueError("Trajectory is empty (requires at least one PoseEstimate)")
        timestamps = [e.pose.timestamp for e in items]
        for i in range(len(timestamps) - 1):
            t0, t1 = timestamps[i], timestamps[i + 1]
            if t0 > t1:
                raise ValueError(
                    f"Trajectory timestamps out of order: "
                    f"{t0:.9f} then {t1:.9f} "
                    f"(sources {items[i].source!r} → {items[i + 1].source!r})"
                )
            if t0 == t1:
                raise ValueError(
                    f"Trajectory has duplicate timestamp {t0:.9f} "
                    f"(sources {items[i].source!r} and {items[i + 1].source!r})"
                )
        object.__setattr__(self, "estimates", items)

    def __len__(self) -> int:
        return len(self.estimates)

    def __iter__(self) -> Iterator[PoseEstimate]:
        return iter(self.estimates)

    @overload
    def __getitem__(self, index: int) -> PoseEstimate: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[PoseEstimate, ...]: ...

    def __getitem__(self, index: int | slice) -> PoseEstimate | tuple[PoseEstimate, ...]:
        return self.estimates[index]

    def timestamps(self) -> NDArray[np.float64]:
        """Return sample timestamps in seconds as a 1-D array."""
        return np.array([e.pose.timestamp for e in self.estimates], dtype=np.float64)

    def interpolate_at(self, t: float) -> Pose:
        """Interpolate pose at time ``t`` (seconds).

        Position uses linear interpolation; orientation uses quaternion SLERP.
        Raises ``ValueError`` if ``t`` is outside the trajectory time span.
        """
        t = float(t)
        ts = self.timestamps()
        t0 = float(ts[0])
        t1 = float(ts[-1])
        if t < t0 or t > t1:
            raise ValueError(
                f"t={t} is outside trajectory bounds [{t0}, {t1}]"
            )

        # Exact sample hit: return that pose unchanged.
        exact = np.where(np.isclose(ts, t))[0]
        if exact.size:
            return self.estimates[int(exact[0])].pose

        i1 = int(np.searchsorted(ts, t, side="right"))
        i0 = i1 - 1
        left = self.estimates[i0].pose
        right = self.estimates[i1].pose
        span = right.timestamp - left.timestamp
        alpha = (t - left.timestamp) / span

        position = (1.0 - alpha) * left.position + alpha * right.position
        orientation = _slerp(left.orientation, right.orientation, alpha)
        frame = left.frame if left.frame == right.frame else None
        return Pose(
            timestamp=t,
            position=position,
            orientation=orientation,
            frame=frame,
        )


class IntegrityState(Enum):
    """Coarse integrity classification for a navigation solution."""

    NOMINAL = "nominal"
    DEGRADED = "degraded"
    DO_NOT_TRUST = "do_not_trust"


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """Integrity alert for a navigation solution at one epoch.

    Parameters
    ----------
    timestamp:
        Time in seconds.
    protection_level_horizontal:
        Horizontal protection level in metres.
    protection_level_vertical:
        Vertical protection level in metres.
    state:
        Discrete integrity classification.
    reason:
        Human-readable explanation of the reported state.
    time_since_corroboration:
        Seconds since the last successful cross-check / corroboration.
    """

    timestamp: float
    protection_level_horizontal: float
    protection_level_vertical: float
    state: IntegrityState
    reason: str
    time_since_corroboration: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(
            self,
            "protection_level_horizontal",
            float(self.protection_level_horizontal),
        )
        object.__setattr__(
            self,
            "protection_level_vertical",
            float(self.protection_level_vertical),
        )
        object.__setattr__(
            self,
            "time_since_corroboration",
            float(self.time_since_corroboration),
        )
        if not isinstance(self.state, IntegrityState):
            raise ValueError("state must be an IntegrityState")
        if not np.isfinite(self.protection_level_horizontal):
            raise ValueError(
                f"protection_level_horizontal must be finite, "
                f"got {self.protection_level_horizontal!r} at timestamp={self.timestamp:.9f}"
            )
        if not np.isfinite(self.protection_level_vertical):
            raise ValueError(
                f"protection_level_vertical must be finite, "
                f"got {self.protection_level_vertical!r} at timestamp={self.timestamp:.9f}"
            )
        if self.protection_level_horizontal < 0.0:
            raise ValueError("protection_level_horizontal must be non-negative")
        if self.protection_level_vertical < 0.0:
            raise ValueError("protection_level_vertical must be non-negative")
        if self.time_since_corroboration < 0.0:
            raise ValueError("time_since_corroboration must be non-negative")
