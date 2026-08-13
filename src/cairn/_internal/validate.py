"""Adversarial / filthy-input validation at Cairn public API boundaries.

Behaviour contract (CP-13)
--------------------------
| Condition                         | Behaviour                                      |
| --------------------------------- | ---------------------------------------------- |
| NaN / Inf in position             | ``ValueError`` naming timestamp + source       |
| NaN / Inf in covariance           | ``ValueError`` naming timestamp + source       |
| Non-positive-definite covariance  | ``ValueError`` (construction / validate) or    |
|                                   | ``DO_NOT_TRUST`` at monitor if forced through  |
| Duplicate timestamps              | ``ValueError`` on ``Trajectory``               |
| Out-of-order timestamps           | ``ValueError`` on ``Trajectory``               |
| Empty trajectory                  | ``ValueError`` on ``Trajectory``               |
| Length-1 trajectory               | Allowed for single-stream monitors             |
| Large mid-sequence time gap       | ``logging.warning`` naming both timestamps;    |
|                                   | processing continues                           |
| Single-source consistency         | ``DO_NOT_TRUST`` (no crash)                    |
| Mismatched source rates           | Associate within ``max_dt``; epochs with <2    |
|                                   | sources → ``DO_NOT_TRUST``                     |

Never let a non-finite value become a protection level: monitors replace
failed PL computation with ``DO_NOT_TRUST`` and a finite fallback PL.
"""

from __future__ import annotations

__all__ = [
    "CairnInputError",
    "DEFAULT_GAP_WARN_S",
    "UNTRUSTED_PL_M",
    "associate_sources_at_time",
    "describe_rate_mismatch",
    "require_finite",
    "require_finite_scalar",
    "require_spd_covariance",
    "untrusted_report",
    "validate_pose",
    "validate_pose_estimate",
    "validate_trajectory",
]


import logging
import math
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from cairn.errors import CairnInputError
from cairn.types import IntegrityReport, IntegrityState, Pose, PoseEstimate, Trajectory

logger = logging.getLogger("cairn.validate")

# Numerical PSD tolerance (not a partner knob).
_EIG_FLOOR = -1e-10


def _cfg():
    """Lazy default config (avoids import cycles at module load)."""
    from cairn.config import CairnConfig

    return CairnConfig()


def _default_untrusted_pl_m() -> float:
    return _cfg().integrity.untrusted_pl_m


def _default_gap_warn_s() -> float:
    return _cfg().association.gap_warn_s


def _default_assoc_max_dt_s() -> float:
    return _cfg().association.max_dt_s


# Backward-compatible aliases — must match CairnConfig defaults (see test_config).
UNTRUSTED_PL_M = 1.0e6
DEFAULT_GAP_WARN_S = 1.0


def _ts_src(timestamp: float | None, source: str | None) -> str:
    parts = []
    if source is not None:
        parts.append(f"source={source!r}")
    if timestamp is not None and math.isfinite(float(timestamp)):
        parts.append(f"timestamp={float(timestamp):.9f}")
    elif timestamp is not None:
        parts.append(f"timestamp={timestamp!r}")
    return ", ".join(parts) if parts else "unknown location"


def require_finite(
    name: str,
    array: NDArray[np.float64],
    *,
    timestamp: float | None = None,
    source: str | None = None,
) -> None:
    """Raise if ``array`` contains NaN or Inf."""
    if not np.all(np.isfinite(array)):
        bad = ~np.isfinite(array)
        n_bad = int(np.count_nonzero(bad))
        raise CairnInputError(
            f"non-finite values in {name} ({n_bad} entries); {_ts_src(timestamp, source)}"
        )


def require_finite_scalar(
    name: str,
    value: float,
    *,
    timestamp: float | None = None,
    source: str | None = None,
) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise CairnInputError(
            f"non-finite {name}={value!r}; {_ts_src(timestamp, source)}"
        )
    return v


def require_spd_covariance(
    covariance: NDArray[np.float64],
    *,
    timestamp: float | None = None,
    source: str | None = None,
    name: str = "covariance",
) -> NDArray[np.float64]:
    """Require a finite, symmetric, positive-definite 6×6 covariance."""
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape != (6, 6):
        raise CairnInputError(
            f"{name} must have shape (6, 6), got {cov.shape}; "
            f"{_ts_src(timestamp, source)}"
        )
    require_finite(name, cov, timestamp=timestamp, source=source)
    sym = 0.5 * (cov + cov.T)
    if not np.allclose(cov, sym, rtol=1e-8, atol=1e-10):
        raise CairnInputError(
            f"{name} is not symmetric; {_ts_src(timestamp, source)}"
        )
    eig = np.linalg.eigvalsh(sym)
    if float(eig.min()) < _EIG_FLOOR:
        raise CairnInputError(
            f"{name} is not positive-definite "
            f"(min eigenvalue={float(eig.min()):.3e}); {_ts_src(timestamp, source)}"
        )
    return sym


def validate_pose(pose: Pose, *, source: str | None = None) -> None:
    """Validate a pose; raises :class:`CairnInputError` on filthy fields."""
    ts = require_finite_scalar(
        "timestamp", pose.timestamp, timestamp=pose.timestamp, source=source
    )
    require_finite("position", pose.position, timestamp=ts, source=source)
    require_finite("orientation", pose.orientation, timestamp=ts, source=source)
    if not math.isclose(float(np.linalg.norm(pose.orientation)), 1.0, rel_tol=1e-3, abs_tol=1e-3):
        # Pose already normalises; this catches corruption after construction.
        raise CairnInputError(
            f"orientation is not unit length; {_ts_src(ts, source)}"
        )


def validate_pose_estimate(estimate: PoseEstimate) -> None:
    """Validate pose + covariance at a public API boundary."""
    validate_pose(estimate.pose, source=estimate.source)
    require_spd_covariance(
        estimate.covariance,
        timestamp=estimate.pose.timestamp,
        source=estimate.source,
    )


def validate_trajectory(
    trajectory: Trajectory,
    *,
    gap_warn_s: float | None = None,
) -> list[str]:
    """Validate every estimate; warn on large gaps.

    Returns
    -------
    warnings
        Human-readable gap warnings (also emitted via ``logging``).
    """
    if gap_warn_s is None:
        gap_warn_s = _default_gap_warn_s()
    if len(trajectory) == 0:
        raise CairnInputError("trajectory is empty")
    warnings: list[str] = []
    prev_ts: float | None = None
    for est in trajectory:
        validate_pose_estimate(est)
        ts = float(est.pose.timestamp)
        if prev_ts is not None:
            if ts <= prev_ts:
                raise CairnInputError(
                    f"timestamps not strictly increasing "
                    f"(prev={prev_ts:.9f}, curr={ts:.9f}); "
                    f"source={est.source!r}"
                )
            gap = ts - prev_ts
            if gap_warn_s > 0.0 and gap > gap_warn_s:
                msg = (
                    f"large time gap {gap:.3f}s between timestamps "
                    f"{prev_ts:.9f} and {ts:.9f}; source={est.source!r}"
                )
                warnings.append(msg)
                logger.warning(msg)
        prev_ts = ts
    return warnings


def untrusted_report(
    timestamp: float,
    *,
    reason: str,
    source: str | None = None,
    pl_m: float | None = None,
) -> IntegrityReport:
    """Finite ``DO_NOT_TRUST`` report — never carries NaN protection levels."""
    if pl_m is None:
        pl_m = _default_untrusted_pl_m()
    ts = require_finite_scalar("timestamp", timestamp, timestamp=timestamp, source=source)
    pl = require_finite_scalar("pl_m", pl_m, timestamp=ts, source=source)
    if pl < 0.0:
        raise CairnInputError(f"pl_m must be non-negative; {_ts_src(ts, source)}")
    full_reason = reason if source is None else f"{reason}; source={source}"
    return IntegrityReport(
        timestamp=ts,
        protection_level_horizontal=pl,
        protection_level_vertical=pl,
        state=IntegrityState.DO_NOT_TRUST,
        reason=full_reason,
        time_since_corroboration=0.0,
    )


def associate_sources_at_time(
    streams: Mapping[str, Trajectory],
    *,
    max_dt: float | None = None,
) -> list[dict[str, PoseEstimate]]:
    """Associate multi-rate streams onto a common timeline (union of stamps).

    For each unique timestamp across streams, collect every source with a sample
    within ``max_dt``. Epochs are returned even when only one source matches so
    the caller can degrade to ``DO_NOT_TRUST``.

    Raises
    ------
    CairnInputError
        If ``streams`` is empty or any stream fails validation.
    """
    if max_dt is None:
        max_dt = _default_assoc_max_dt_s()
    if not streams:
        raise CairnInputError("associate_sources_at_time requires at least one stream")
    if max_dt < 0.0:
        raise CairnInputError(f"max_dt must be non-negative, got {max_dt}")

    for name, traj in streams.items():
        if len(traj) == 0:
            raise CairnInputError(f"stream {name!r} is empty")
        validate_trajectory(traj)

    # Union of timestamps from all streams.
    all_times: list[float] = []
    for traj in streams.values():
        all_times.extend(float(t) for t in traj.timestamps())
    unique_times = np.unique(np.asarray(all_times, dtype=np.float64))

    epochs: list[dict[str, PoseEstimate]] = []
    for t in unique_times:
        epoch: dict[str, PoseEstimate] = {}
        for name, traj in streams.items():
            ts = traj.timestamps()
            idx = int(np.argmin(np.abs(ts - t)))
            dt = abs(float(ts[idx]) - float(t))
            if dt <= max_dt + 1e-12:
                epoch[name] = traj[idx]
        if epoch:
            epochs.append(epoch)
        else:
            logger.warning(
                "no sources within max_dt=%.3fs of timestamp=%.9f", max_dt, float(t)
            )
    return epochs


def describe_rate_mismatch(
    streams: Mapping[str, Trajectory],
) -> str:
    """Return a short summary of per-source mean sample rates (for logs/tests)."""
    parts = []
    for name, traj in streams.items():
        if len(traj) < 2:
            parts.append(f"{name}: n={len(traj)} (rate undefined)")
            continue
        ts = traj.timestamps()
        dt = np.diff(ts)
        rate = float(1.0 / np.mean(dt)) if float(np.mean(dt)) > 0 else float("nan")
        parts.append(f"{name}: n={len(traj)} mean_rate≈{rate:.2f} Hz")
    return "; ".join(parts)
