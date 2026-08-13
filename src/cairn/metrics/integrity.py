"""Integrity metrics and Stanford-ESA diagram summaries (public).

``true_errors`` are paired with each report's horizontal protection level
(``protection_level_horizontal``). Units: metres for errors / PL / alert limit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cairn.types import IntegrityReport, IntegrityState

__all__ = [
    "IntegrityReportSummary",
    "StanfordDiagramData",
    "availability",
    "false_alarm_rate",
    "integrity_report_summary",
    "missed_detection_rate",
    "stanford_diagram",
]


def _as_error_array(
    reports: Sequence[IntegrityReport],
    true_errors: Sequence[float] | NDArray[np.float64],
) -> NDArray[np.float64]:
    errors = np.asarray(true_errors, dtype=np.float64)
    if errors.ndim != 1:
        raise ValueError(f"true_errors must be 1-D, got shape {errors.shape}")
    if len(reports) != errors.size:
        raise ValueError(
            f"reports and true_errors length mismatch: {len(reports)} vs {errors.size}"
        )
    if np.any(errors < 0.0):
        raise ValueError("true_errors must be non-negative")
    return errors


def _protection_levels(reports: Sequence[IntegrityReport]) -> NDArray[np.float64]:
    return np.array(
        [r.protection_level_horizontal for r in reports], dtype=np.float64
    )


def missed_detection_rate(
    reports: Sequence[IntegrityReport],
    true_errors: Sequence[float] | NDArray[np.float64],
) -> float:
    """Fraction of epochs that are confidently wrong (dimensionless in ``[0, 1]``).

    A missed detection is ``true_error > protection_level`` (metres) while
    ``state == NOMINAL``.

    Raises
    ------
    ValueError
        Length mismatch, negative errors, or non-1-D ``true_errors``.
    """
    if not reports:
        return 0.0
    errors = _as_error_array(reports, true_errors)
    pls = _protection_levels(reports)
    missed = [
        (err > pl) and (rep.state is IntegrityState.NOMINAL)
        for rep, err, pl in zip(reports, errors, pls, strict=True)
    ]
    return float(np.mean(missed))


def false_alarm_rate(
    reports: Sequence[IntegrityReport],
    true_errors: Sequence[float] | NDArray[np.float64],
) -> float:
    """Fraction of epochs with an unnecessary integrity flag (``[0, 1]``).

    A false alarm is ``state in {DEGRADED, DO_NOT_TRUST}`` while
    ``true_error <= protection_level`` (metres).

    Raises
    ------
    ValueError
        Length mismatch, negative errors, or non-1-D ``true_errors``.
    """
    if not reports:
        return 0.0
    errors = _as_error_array(reports, true_errors)
    pls = _protection_levels(reports)
    alarmed = (
        IntegrityState.DEGRADED,
        IntegrityState.DO_NOT_TRUST,
    )
    false = [
        (rep.state in alarmed) and (err <= pl)
        for rep, err, pl in zip(reports, errors, pls, strict=True)
    ]
    return float(np.mean(false))


def availability(reports: Sequence[IntegrityReport]) -> float:
    """Fraction of epochs declared ``NOMINAL`` (dimensionless in ``[0, 1]``)."""
    if not reports:
        return 0.0
    return float(np.mean([r.state is IntegrityState.NOMINAL for r in reports]))


@dataclass(frozen=True, slots=True)
class StanfordDiagramData:
    """Stanford-ESA integrity diagram inputs and quadrant counts.

    Axes: true error on ``x``, protection level on ``y``. Regions are partitioned
    mutually exclusively using alert limit ``AL``:

    * **unavailable** — ``PL >= AL``
    * **hazardously_misleading** — ``error > AL`` and ``PL < AL``
    * **misleading** — ``error > PL`` and ``error <= AL`` and ``PL < AL``
    * **nominal** — ``error <= PL`` and ``PL < AL``
    """

    true_errors: NDArray[np.float64]
    protection_levels: NDArray[np.float64]
    alert_limit: float
    nominal_count: int
    misleading_count: int
    hazardously_misleading_count: int
    unavailable_count: int


def stanford_diagram(
    reports: Sequence[IntegrityReport],
    true_errors: Sequence[float] | NDArray[np.float64],
    alert_limit: float,
) -> StanfordDiagramData:
    """Build Stanford-ESA diagram arrays and per-quadrant counts.

    Parameters
    ----------
    reports:
        Integrity reports (HPL in metres).
    true_errors:
        Horizontal true errors in metres, one per report.
    alert_limit:
        Operational alert limit in metres (must be > 0).

    Raises
    ------
    ValueError
        Non-positive ``alert_limit``, length mismatch, or negative errors.
    """
    if alert_limit <= 0.0:
        raise ValueError(f"alert_limit must be positive, got {alert_limit}")

    if not reports:
        empty = np.asarray([], dtype=np.float64)
        return StanfordDiagramData(
            true_errors=empty,
            protection_levels=empty,
            alert_limit=float(alert_limit),
            nominal_count=0,
            misleading_count=0,
            hazardously_misleading_count=0,
            unavailable_count=0,
        )

    errors = _as_error_array(reports, true_errors)
    pls = _protection_levels(reports)
    al = float(alert_limit)

    nominal = 0
    misleading = 0
    hmi = 0
    unavailable = 0
    for err, pl in zip(errors, pls, strict=True):
        if pl >= al:
            unavailable += 1
        elif err > al:
            hmi += 1
        elif err > pl:
            misleading += 1
        else:
            nominal += 1

    return StanfordDiagramData(
        true_errors=errors.copy(),
        protection_levels=pls.copy(),
        alert_limit=al,
        nominal_count=nominal,
        misleading_count=misleading,
        hazardously_misleading_count=hmi,
        unavailable_count=unavailable,
    )


@dataclass(frozen=True, slots=True)
class IntegrityReportSummary:
    """Aggregate integrity metrics for a report sequence."""

    epoch_count: int
    missed_detection_rate: float
    false_alarm_rate: float
    availability: float
    stanford: StanfordDiagramData


def integrity_report_summary(
    reports: Sequence[IntegrityReport],
    true_errors: Sequence[float] | NDArray[np.float64],
    alert_limit: float,
) -> IntegrityReportSummary:
    """Compute MDR, FAR, availability, and Stanford diagram in one pass.

    Parameters
    ----------
    reports:
        Integrity reports (protection levels in metres).
    true_errors:
        Horizontal true errors in metres.
    alert_limit:
        Alert limit in metres.

    Raises
    ------
    ValueError
        See :func:`stanford_diagram` / :func:`missed_detection_rate`.
    """
    return IntegrityReportSummary(
        epoch_count=len(reports),
        missed_detection_rate=missed_detection_rate(reports, true_errors),
        false_alarm_rate=false_alarm_rate(reports, true_errors),
        availability=availability(reports),
        stanford=stanford_diagram(reports, true_errors, alert_limit),
    )
