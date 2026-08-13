"""Naive integrity baseline: trust the estimator's self-reported covariance."""

from __future__ import annotations

__all__ = [
    "NaiveCovarianceMonitor",
    "chi2_scale",
    "protection_levels_from_covariance",
    "with_claimed_diagonal_covariance",
]


import logging

import numpy as np
from scipy.stats import chi2

from cairn._internal.validate import (
    untrusted_report,
    validate_pose_estimate,
    validate_trajectory,
)
from cairn.config import CairnConfig
from cairn.errors import CairnInputError
from cairn.types import IntegrityReport, IntegrityState, PoseEstimate, Trajectory

logger = logging.getLogger("cairn.baselines.naive")


def chi2_scale(confidence: float, dof: int) -> float:
    """Return ``√χ²_{confidence, dof}`` used to inflate 1σ into a protection level."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if dof < 1:
        raise ValueError(f"dof must be >= 1, got {dof}")
    return float(np.sqrt(chi2.ppf(confidence, dof)))


def protection_levels_from_covariance(
    covariance: np.ndarray,
    *,
    confidence: float | None = None,
) -> tuple[float, float]:
    """Convert a 6×6 pose covariance into horizontal / vertical protection levels.

    Covariance order is ``[x, y, z, roll, pitch, yaw]`` (metres / radians).

    * **HPL** — ``√(χ²_{conf,2} · λ_max(C_{xy}))`` (2-DoF horizontal ellipse bound).
    * **VPL** — ``√(χ²_{conf,1} · C_{zz})`` (1-DoF vertical bound).

    Raises
    ------
    ValueError
        If covariance is the wrong shape or yields a non-finite PL.
    """
    if confidence is None:
        confidence = CairnConfig().integrity.chi2_confidence
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape != (6, 6):
        raise ValueError(f"covariance must have shape (6, 6), got {cov.shape}")
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance contains non-finite values")

    c_xy = 0.5 * (cov[0:2, 0:2] + cov[0:2, 0:2].T)
    eigvals = np.linalg.eigvalsh(c_xy)
    lam_max = float(max(eigvals.max(), 0.0))
    var_z = float(max(cov[2, 2], 0.0))

    hpl = chi2_scale(confidence, 2) * float(np.sqrt(lam_max))
    vpl = chi2_scale(confidence, 1) * float(np.sqrt(var_z))
    if not (np.isfinite(hpl) and np.isfinite(vpl)):
        raise ValueError(
            f"non-finite protection levels from covariance (HPL={hpl}, VPL={vpl})"
        )
    return hpl, vpl


class NaiveCovarianceMonitor:
    """Integrity monitor that only re-packages the filter's own covariance.

    Protection levels are chi-squared scaled self-reported uncertainties.
    ``state`` is always :attr:`IntegrityState.NOMINAL` when the estimate is
    well-formed. Filthy inputs degrade to :attr:`IntegrityState.DO_NOT_TRUST`
    with a finite fallback PL (never NaN).
    """

    def __init__(self, confidence: float | None = None) -> None:
        if confidence is None:
            confidence = CairnConfig().integrity.chi2_confidence
        self.confidence = float(confidence)
        # Validate early.
        _ = chi2_scale(self.confidence, 2)

    def report_one(self, estimate: PoseEstimate) -> IntegrityReport:
        try:
            validate_pose_estimate(estimate)
            hpl, vpl = protection_levels_from_covariance(
                estimate.covariance, confidence=self.confidence
            )
        except (CairnInputError, ValueError) as exc:
            logger.warning("naive monitor rejecting estimate: %s", exc)
            return untrusted_report(
                estimate.pose.timestamp,
                reason=f"naive_input_rejected:{exc}",
                source=estimate.source,
            )
        return IntegrityReport(
            timestamp=estimate.pose.timestamp,
            protection_level_horizontal=hpl,
            protection_level_vertical=vpl,
            state=IntegrityState.NOMINAL,
            reason=(
                f"naive_chi2_cov confidence={self.confidence:g} "
                f"source={estimate.source}"
            ),
            time_since_corroboration=0.0,
        )

    def report(self, trajectory: Trajectory) -> list[IntegrityReport]:
        """Emit one report per pose estimate (NOMINAL or DO_NOT_TRUST)."""
        validate_trajectory(trajectory)
        return [self.report_one(est) for est in trajectory]


def with_claimed_diagonal_covariance(
    trajectory: Trajectory,
    *,
    std_xy: float | None = None,
    std_z: float | None = None,
    std_rot: float | None = None,
    source: str | None = None,
    config: CairnConfig | None = None,
) -> Trajectory:
    """Attach a diagonal self-reported covariance (metres / radians).

    CERBERUS TUM exports carry no covariance; this builds the overconfident
    filter self-report that a naive monitor would trust. Defaults come from
    ``config.default_covariance`` (or :class:`~cairn.config.CairnConfig`).
    """
    cov_cfg = (config or CairnConfig()).covariance_for(source)
    if std_xy is None:
        std_xy = cov_cfg.std_xy_m
    if std_z is None:
        std_z = cov_cfg.std_z_m
    if std_rot is None:
        std_rot = cov_cfg.std_rot_rad
    if std_xy <= 0.0 or std_z <= 0.0 or std_rot <= 0.0:
        raise ValueError("claimed standard deviations must be positive")
    diag = np.array(
        [
            std_xy**2,
            std_xy**2,
            std_z**2,
            std_rot**2,
            std_rot**2,
            std_rot**2,
        ],
        dtype=np.float64,
    )
    cov = np.diag(diag)
    estimates: list[PoseEstimate] = []
    for est in trajectory:
        estimates.append(
            PoseEstimate(
                pose=est.pose,
                covariance=cov.copy(),
                source=source or est.source,
            )
        )
    return Trajectory(estimates)
