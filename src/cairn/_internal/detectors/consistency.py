"""Solution-separation fault detection across independent pose sources.

This is the navigation analogue of aviation receiver autonomous integrity
monitoring (RAIM) / solution-separation: fuse all sources, then fuse each
leave-one-out subset, and test whether any source is inconsistent with the
rest.

Assumptions
-----------
1. **Independence.** Pose sources are conditionally independent given the true
   pose (uncorrelated estimation errors). Weighted least-squares fusion and
   the χ² separation statistic are derived under that model. If two "sources"
   share a filter state, map, or time-sync fault, separations become dependent
   and both the false-alarm calibration and the isolation logic can fail —
   typically by **missing a common-mode fault** or by **blaming the wrong
   residual source**.
2. **Gaussian errors.** Each source reports an unbiased mean and a covariance
   that correctly describes its error. Optimistic covariances shrink separations
   relative to the modelled Σ_ss and inflate missed detections.
3. **Single fault.** Isolation returns at most one faulted source (the largest
   normalized separation). Multiple simultaneous faults are not modelled.
4. **Position-only fusion.** Orientation is not fused here; the 3×3 position
   block of each :class:`~cairn.types.PoseEstimate` covariance is used. Horizontal
   protection level is derived from the horizontal components of the SS bound.

What breaks when independence fails
-----------------------------------
Correlated sources (e.g. LiDAR odometry and LiDAR-inertial SLAM on the same
cloud, or two algorithms locked to the same degeneracy) produce small
separations even when both are wrong. Solution-separation then reports
**nominal consistency with a misleading protection level** — the failure mode
this product exists to catch must still be covered by geometry-aware detectors
(CP-8) and by ensuring source diversity.
"""

from __future__ import annotations

__all__ = [
    "SINGLE_SOURCE_TAG",
    "SeparationTest",
    "SolutionSeparationDetector",
    "SolutionSeparationResult",
    "diagnostic_parts_from_ss",
    "fuse_positions_wls",
    "solution_separation",
]


import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from cairn._internal.diagnostics import TestStatRecord, TriggerRecord
from cairn._internal.validate import untrusted_report, validate_pose_estimate
from cairn.config import CairnConfig
from cairn.errors import CairnInputError
from cairn.types import IntegrityReport, IntegrityState, PoseEstimate

logger = logging.getLogger("cairn.detectors.consistency")

DOF = 3  # position fusion in R^3
SINGLE_SOURCE_TAG = "__single_source__"


def _position_cov(estimate: PoseEstimate) -> NDArray[np.float64]:
    cov = np.asarray(estimate.covariance, dtype=np.float64)
    return 0.5 * (cov[:3, :3] + cov[:3, :3].T)


def _as_sources(
    sources: dict[str, PoseEstimate] | list[PoseEstimate],
) -> dict[str, PoseEstimate]:
    if isinstance(sources, dict):
        return dict(sources)
    return {f"source_{i}": est for i, est in enumerate(sources)}


def _validate_sources(src: dict[str, PoseEstimate]) -> None:
    if not src:
        raise CairnInputError("solution separation received an empty source set")
    for name, est in src.items():
        # Ensure dict key and estimate.source are both present in errors.
        try:
            validate_pose_estimate(est)
        except CairnInputError as exc:
            raise CairnInputError(f"{exc} (dict_key={name!r})") from exc


def fuse_positions_wls(
    sources: dict[str, PoseEstimate],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Information-form WLS fusion of position means.

    Returns
    -------
    mean, cov
        Fused position (3,) and covariance (3, 3).
    """
    info = np.zeros((DOF, DOF), dtype=np.float64)
    rhs = np.zeros(DOF, dtype=np.float64)
    for est in sources.values():
        c = _position_cov(est)
        try:
            w = np.linalg.inv(c)
        except np.linalg.LinAlgError as exc:
            raise ValueError("source covariance is singular") from exc
        info += w
        rhs += w @ np.asarray(est.pose.position, dtype=np.float64)
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError as exc:
        raise ValueError("fused information matrix is singular") from exc
    mean = cov @ rhs
    return mean, 0.5 * (cov + cov.T)


@dataclass(frozen=True, slots=True)
class SeparationTest:
    """One leave-one-out solution-separation test."""

    excluded_source: str
    separation: NDArray[np.float64]
    separation_covariance: NDArray[np.float64]
    test_statistic: float
    threshold: float
    faulted: bool


@dataclass(frozen=True, slots=True)
class SolutionSeparationResult:
    """Outcome of solution-separation over a shared epoch."""

    full_solution: NDArray[np.float64]
    full_covariance: NDArray[np.float64]
    tests: tuple[SeparationTest, ...]
    faulted_source: str | None
    protection_level_horizontal: float
    protection_level_vertical: float
    p_fa: float
    reason: str = ""


def diagnostic_parts_from_ss(
    result: SolutionSeparationResult,
) -> tuple[float | None, tuple[TestStatRecord, ...], TriggerRecord | None]:
    """Extract JSONL-friendly test stats / trigger from an SS result."""
    if not result.tests:
        trigger = None
        if result.faulted_source is not None:
            trigger = TriggerRecord(
                name=result.faulted_source,
                statistic=None,
                threshold=None,
                kind="untrusted",
                comparison="no_cross_consistency_or_input_failure",
            )
        return None, (), trigger

    threshold = float(result.tests[0].threshold)
    tests = tuple(
        TestStatRecord(
            name=t.excluded_source,
            statistic=float(t.test_statistic),
            threshold=float(t.threshold),
            crossed=bool(t.faulted),
            kind="separation",
        )
        for t in result.tests
    )
    trigger = None
    if result.faulted_source is not None and result.faulted_source != SINGLE_SOURCE_TAG:
        hit = next(
            (t for t in result.tests if t.excluded_source == result.faulted_source),
            None,
        )
        if hit is not None:
            trigger = TriggerRecord(
                name=hit.excluded_source,
                statistic=float(hit.test_statistic),
                threshold=float(hit.threshold),
                kind="separation",
                comparison="stat > threshold",
            )
    return threshold, tests, trigger


class SolutionSeparationDetector:
    """RAIM-style solution separation across N pose sources at one epoch."""

    def __init__(
        self,
        *,
        p_fa: float | None = None,
        pl_confidence: float | None = None,
        untrusted_pl_m: float | None = None,
    ) -> None:
        cfg = CairnConfig().integrity
        p_fa = cfg.p_fa if p_fa is None else float(p_fa)
        pl_confidence = cfg.pl_confidence if pl_confidence is None else float(pl_confidence)
        untrusted_pl_m = (
            cfg.untrusted_pl_m if untrusted_pl_m is None else float(untrusted_pl_m)
        )
        if not 0.0 < p_fa < 1.0:
            raise ValueError(f"p_fa must be in (0, 1), got {p_fa}")
        if not 0.0 < pl_confidence < 1.0:
            raise ValueError(f"pl_confidence must be in (0, 1), got {pl_confidence}")
        if untrusted_pl_m <= 0.0:
            raise ValueError(f"untrusted_pl_m must be > 0, got {untrusted_pl_m}")
        self.p_fa = p_fa
        self.pl_confidence = pl_confidence
        self.untrusted_pl_m = untrusted_pl_m

    def detect(
        self,
        sources: dict[str, PoseEstimate] | list[PoseEstimate],
    ) -> SolutionSeparationResult:
        """Run full-set / leave-one-out separation tests.

        The per-test chi-squared false-alarm allocation is ``p_fa / N``
        (Bonferroni) so the family-wise rate of *any* false isolation under
        independent H0 trials is approximately ``p_fa``.

        Single-source input does **not** crash: it returns a result with
        ``faulted_source="__single_source__"`` and a large finite PL, which
        callers should treat as ``DO_NOT_TRUST`` (cross-consistency impossible).
        """
        src = _as_sources(sources)
        try:
            _validate_sources(src)
        except CairnInputError as exc:
            logger.warning("solution separation rejected sources: %s", exc)
            any_est = next(iter(src.values()), None)
            ts = float(any_est.pose.timestamp) if any_est is not None else 0.0
            return self._untrusted_result(
                reason=f"input_rejected:{exc}",
                timestamp=ts,
                position=(
                    np.asarray(any_est.pose.position, dtype=np.float64)
                    if any_est is not None
                    else np.zeros(3)
                ),
            )

        names = list(src.keys())
        if len(names) < 2:
            only = names[0] if names else None
            logger.warning(
                "solution separation: only %d source(s) at timestamp=%s — "
                "cross-consistency impossible; degrading to DO_NOT_TRUST",
                len(names),
                f"{src[only].pose.timestamp:.9f}" if only else "n/a",
            )
            est = src[only] if only else None
            return self._untrusted_result(
                reason="single_source_no_cross_consistency",
                timestamp=float(est.pose.timestamp) if est else 0.0,
                position=(
                    np.asarray(est.pose.position, dtype=np.float64)
                    if est is not None
                    else np.zeros(3)
                ),
                faulted_source=SINGLE_SOURCE_TAG,
            )

        n_src = len(names)
        alpha = self.p_fa / n_src
        threshold = float(chi2.ppf(1.0 - alpha, DOF))

        try:
            full_mean, full_cov = fuse_positions_wls(src)
        except ValueError as exc:
            logger.warning("solution separation fusion failed: %s", exc)
            est0 = src[names[0]]
            return self._untrusted_result(
                reason=f"fusion_failed:{exc}",
                timestamp=float(est0.pose.timestamp),
                position=np.asarray(est0.pose.position, dtype=np.float64),
            )

        tests: list[SeparationTest] = []

        for name in names:
            subset = {k: v for k, v in src.items() if k != name}
            sub_mean, sub_cov = fuse_positions_wls(subset)
            separation = full_mean - sub_mean
            ss_cov = sub_cov - full_cov
            ss_cov = 0.5 * (ss_cov + ss_cov.T)
            eig = np.linalg.eigvalsh(ss_cov)
            if float(eig.min()) < 0.0:
                ss_cov = ss_cov - 1.01 * float(eig.min()) * np.eye(DOF)

            try:
                stat = float(separation @ np.linalg.solve(ss_cov, separation))
            except np.linalg.LinAlgError:
                scale = max(float(np.trace(ss_cov)) / DOF, 1e-12)
                stat = float(separation @ separation) / scale

            if not np.isfinite(stat):
                logger.warning(
                    "non-finite separation statistic excluding %r at timestamp=%.9f",
                    name,
                    float(src[name].pose.timestamp),
                )
                est0 = src[names[0]]
                return self._untrusted_result(
                    reason=f"non_finite_stat:excluded={name}",
                    timestamp=float(est0.pose.timestamp),
                    position=full_mean,
                )

            tests.append(
                SeparationTest(
                    excluded_source=name,
                    separation=separation,
                    separation_covariance=ss_cov,
                    test_statistic=stat,
                    threshold=threshold,
                    faulted=stat > threshold,
                )
            )

        faulted = [t for t in tests if t.faulted]
        if faulted:
            faulted_source = max(faulted, key=lambda t: t.test_statistic).excluded_source
        else:
            faulted_source = None

        hpl, vpl = self._protection_levels(full_cov, tests, threshold)
        if not (np.isfinite(hpl) and np.isfinite(vpl)):
            logger.warning(
                "non-finite protection levels (HPL=%r VPL=%r); forcing untrusted",
                hpl,
                vpl,
            )
            return self._untrusted_result(
                reason="non_finite_protection_level",
                timestamp=float(src[names[0]].pose.timestamp),
                position=full_mean,
                faulted_source=faulted_source or SINGLE_SOURCE_TAG,
            )

        if faulted_source is not None:
            hit = next(t for t in tests if t.excluded_source == faulted_source)
            reason = (
                f"solution_separation:faulted={faulted_source}:"
                f"stat={hit.test_statistic:.6g}>threshold={hit.threshold:.6g}"
            )
        else:
            reason = "solution_separation:consistent"

        logger.debug(
            "SS epoch sources=%s threshold=%.6g stats=%s faulted=%s",
            names,
            threshold,
            {t.excluded_source: t.test_statistic for t in tests},
            faulted_source,
        )

        return SolutionSeparationResult(
            full_solution=full_mean,
            full_covariance=full_cov,
            tests=tuple(tests),
            faulted_source=faulted_source,
            protection_level_horizontal=hpl,
            protection_level_vertical=vpl,
            p_fa=self.p_fa,
            reason=reason,
        )

    def report_from_result(
        self,
        result: SolutionSeparationResult,
        sources: dict[str, PoseEstimate] | list[PoseEstimate],
    ) -> IntegrityReport:
        """Map an SS result to an :class:`~cairn.types.IntegrityReport`."""
        src = _as_sources(sources)
        ts = (
            float(next(iter(src.values())).pose.timestamp)
            if src
            else 0.0
        )
        if result.faulted_source == SINGLE_SOURCE_TAG or len(src) < 2:
            reason = result.reason or "single_source_no_cross_consistency"
            return untrusted_report(
                ts,
                reason=reason,
                source=next(iter(src.keys()), None) if src else None,
                pl_m=float(result.protection_level_horizontal),
            )
        if result.faulted_source is not None:
            return IntegrityReport(
                timestamp=ts,
                protection_level_horizontal=float(result.protection_level_horizontal),
                protection_level_vertical=float(result.protection_level_vertical),
                state=IntegrityState.DO_NOT_TRUST,
                reason=result.reason
                or f"solution_separation:faulted={result.faulted_source}",
                time_since_corroboration=0.0,
            )
        return IntegrityReport(
            timestamp=ts,
            protection_level_horizontal=float(result.protection_level_horizontal),
            protection_level_vertical=float(result.protection_level_vertical),
            state=IntegrityState.NOMINAL,
            reason=result.reason or "solution_separation:consistent",
            time_since_corroboration=0.0,
        )

    def report_one(
        self,
        sources: dict[str, PoseEstimate] | list[PoseEstimate],
    ) -> IntegrityReport:
        """Integrity report for one epoch; single-source → ``DO_NOT_TRUST``."""
        result = self.detect(sources)
        return self.report_from_result(result, sources)

    def _untrusted_result(
        self,
        *,
        reason: str,
        timestamp: float,
        position: NDArray[np.float64],
        faulted_source: str | None = SINGLE_SOURCE_TAG,
    ) -> SolutionSeparationResult:
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        pl = self.untrusted_pl_m
        cov = np.eye(DOF, dtype=np.float64) * (pl**2)
        return SolutionSeparationResult(
            full_solution=pos,
            full_covariance=cov,
            tests=tuple(),
            faulted_source=faulted_source,
            protection_level_horizontal=pl,
            protection_level_vertical=pl,
            p_fa=self.p_fa,
            reason=reason,
        )

    def _protection_levels(
        self,
        full_cov: NDArray[np.float64],
        tests: list[SeparationTest],
        threshold: float,
    ) -> tuple[float, float]:
        """Noise-only bound plus per-hypothesis SS contribution.

        For each leave-one-out test the SS term maps ``max(stat, threshold)``
        through Σ_ss into position space::

            r = sqrt(max(stat, threshold) · λ)

        so HPL grows when sources disagree (large Mahalanobis separation), with
        a floor at the χ² decision-region radius (integrity against the largest
        undetected fault still under the alarm line). Using only the threshold
        floor — the previous behaviour — froze HPL whenever source covariances
        were fixed.
        """
        k_full = float(np.sqrt(chi2.ppf(self.pl_confidence, DOF)))
        noise_h = k_full * float(
            np.sqrt(max(np.linalg.eigvalsh(full_cov[:2, :2]).max(), 0.0))
        )
        noise_v = k_full * float(np.sqrt(max(full_cov[2, 2], 0.0)))

        ss_h = 0.0
        ss_v = 0.0
        for t in tests:
            lam_h = float(
                max(np.linalg.eigvalsh(t.separation_covariance[:2, :2]).max(), 0.0)
            )
            lam_v = float(max(t.separation_covariance[2, 2], 0.0))
            # Inflate with observed disagreement; never below the detection floor.
            scale = max(float(t.test_statistic), float(threshold))
            ss_h = max(ss_h, float(np.sqrt(scale * lam_h)))
            ss_v = max(ss_v, float(np.sqrt(scale * lam_v)))
            # Also cover the Euclidean separation itself (fault pull on the fuse).
            ss_h = max(ss_h, float(np.linalg.norm(t.separation[:2])))
            ss_v = max(ss_v, float(abs(t.separation[2])))

        return ss_h + noise_h, ss_v + noise_v


def solution_separation(
    sources: dict[str, PoseEstimate] | list[PoseEstimate],
    *,
    p_fa: float | None = None,
    pl_confidence: float | None = None,
    untrusted_pl_m: float | None = None,
) -> SolutionSeparationResult:
    """Functional wrapper around :class:`SolutionSeparationDetector`."""
    return SolutionSeparationDetector(
        p_fa=p_fa,
        pl_confidence=pl_confidence,
        untrusted_pl_m=untrusted_pl_m,
    ).detect(sources)
