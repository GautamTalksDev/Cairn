"""LiDAR registration degeneracy / localizability from the ICP Hessian.

Point-to-plane ICP contributes an information (Hessian) matrix whose small
eigenvalues mark pose directions that the scan cannot constrain — the classic
failure mode in self-similar underground corridors.
"""

from __future__ import annotations

__all__ = [
    "DOF_NAMES",
    "DegeneracyAwareMonitor",
    "DegeneracyDetector",
    "DegeneracyUpdate",
    "estimate_normals",
    "inflate_covariance_anisotropic",
    "localizability",
    "nearest_neighbor_indices",
    "point_to_plane_hessian",
]


import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cairn._internal.baselines.naive import protection_levels_from_covariance
from cairn._internal.diagnostics import TestStatRecord, TriggerRecord
from cairn.config import CairnConfig
from cairn.types import IntegrityReport, IntegrityState, PoseEstimate

# DoF order matches PoseEstimate covariance: [x, y, z, roll, pitch, yaw].
DOF_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
logger = logging.getLogger("cairn.detectors.degeneracy")


def estimate_normals(
    points: NDArray[np.float64],
    *,
    k: int = 20,
) -> NDArray[np.float64]:
    """Estimate unit normals via local PCA (eigenvector of smallest eigenvalue)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    n = pts.shape[0]
    if n < 4:
        raise ValueError("need at least 4 points to estimate normals")
    k = int(min(max(k, 3), n))
    normals = np.zeros((n, 3), dtype=np.float64)
    # Brute-force kNN — fine for synthetic / modest scans.
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    nn_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    for i in range(n):
        nbrs = pts[nn_idx[i]]
        cov = np.cov(nbrs.T)
        w, v = np.linalg.eigh(cov)
        normals[i] = v[:, 0]
    # Orient consistently toward the cloud centroid (inward/outward arbitrary but stable).
    centroid = pts.mean(axis=0)
    flip = np.sum(normals * (centroid - pts), axis=1) < 0.0
    normals[flip] *= -1.0
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)
    return normals


def nearest_neighbor_indices(
    query: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Return index in ``reference`` of the nearest neighbour of each ``query`` point."""
    q = np.asarray(query, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    d2 = np.sum((q[:, None, :] - r[None, :, :]) ** 2, axis=2)
    return np.argmin(d2, axis=1).astype(np.int64)


def point_to_plane_hessian(
    scan: NDArray[np.float64],
    reference: NDArray[np.float64],
    *,
    reference_normals: NDArray[np.float64] | None = None,
    max_distance: float = 1.0,
) -> NDArray[np.float64]:
    """Accumulate the 6×6 point-to-plane ICP Hessian at the identity pose.

    For a correspondence ``(p, n)`` the residual Jacobian is
    ``J = [nᵀ, (p × n)ᵀ]`` and the Hessian contribution is ``Jᵀ J``.
    """
    scan = np.asarray(scan, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if scan.shape[1] != 3 or reference.shape[1] != 3:
        raise ValueError("scan and reference must be (N, 3) point clouds")
    if reference_normals is None:
        reference_normals = estimate_normals(reference)
    else:
        reference_normals = np.asarray(reference_normals, dtype=np.float64)

    idx = nearest_neighbor_indices(scan, reference)
    H = np.zeros((6, 6), dtype=np.float64)
    for i, j in enumerate(idx):
        p = scan[i]
        q = reference[j]
        n = reference_normals[j]
        if float(np.linalg.norm(p - q)) > max_distance:
            continue
        cross = np.cross(p, n)
        J = np.concatenate([n, cross])  # (6,)
        H += np.outer(J, J)
    return 0.5 * (H + H.T)


def localizability(hessian: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-axis localizability scores in ``[0, 1]`` for the six pose DoF.

    Scores are the absolute Hessian diagonal entries, normalized separately
    within the translational block ``[x,y,z]`` and the rotational block
    ``[roll,pitch,yaw]`` so that a direction with near-zero information
    maps to ~0 and the strongest direction in each block maps to ~1.
    """
    H = np.asarray(hessian, dtype=np.float64)
    if H.shape != (6, 6):
        raise ValueError(f"hessian must have shape (6, 6), got {H.shape}")
    diag = np.abs(np.diag(H))
    scores = np.zeros(6, dtype=np.float64)
    for sl in (slice(0, 3), slice(3, 6)):
        block = diag[sl]
        peak = float(block.max())
        if peak <= 0.0:
            scores[sl] = 0.0
        else:
            scores[sl] = block / peak
    return np.clip(scores, 0.0, 1.0)


def inflate_covariance_anisotropic(
    covariance: NDArray[np.float64],
    scores: NDArray[np.float64],
    *,
    loc_floor: float | None = None,
    max_inflation: float | None = None,
) -> NDArray[np.float64]:
    """Inflate diagonal covariance entries along under-constrained DoF.

    Variance along axis ``i`` is multiplied by
    ``(1 / max(score_i, loc_floor))²``, capped by ``max_inflation²``.
    Off-diagonals are left unchanged (conservative for PL extraction).
    """
    deg = CairnConfig().degeneracy
    if loc_floor is None:
        loc_floor = deg.loc_floor
    if max_inflation is None:
        max_inflation = deg.max_inflation
    cov = np.array(covariance, dtype=np.float64, copy=True)
    s = np.asarray(scores, dtype=np.float64)
    if cov.shape != (6, 6) or s.shape != (6,):
        raise ValueError("covariance must be (6,6) and scores (6,)")
    for i in range(6):
        infl = 1.0 / max(float(s[i]), loc_floor)
        infl = min(infl, max_inflation)
        cov[i, i] *= infl * infl
    return cov


@dataclass(frozen=True, slots=True)
class DegeneracyUpdate:
    """Result of one :meth:`DegeneracyDetector.update` call."""

    scores: NDArray[np.float64]
    hessian: NDArray[np.float64]
    degenerate: bool
    degenerate_axes: tuple[str, ...]


class DegeneracyDetector:
    """Online point-cloud localizability detector (ICP Hessian eigenvalues/axes)."""

    def __init__(
        self,
        *,
        score_threshold: float | None = None,
        normal_k: int = 20,
        max_correspondence_distance: float = 1.0,
    ) -> None:
        if score_threshold is None:
            score_threshold = CairnConfig().degeneracy.score_threshold
        self.score_threshold = float(score_threshold)
        self.normal_k = int(normal_k)
        self.max_correspondence_distance = float(max_correspondence_distance)

    def update(
        self,
        scan: NDArray[np.float64],
        prev_scan: NDArray[np.float64],
    ) -> DegeneracyUpdate:
        """Compare ``scan`` to ``prev_scan`` and score per-axis constraints."""
        normals = estimate_normals(prev_scan, k=self.normal_k)
        H = point_to_plane_hessian(
            scan,
            prev_scan,
            reference_normals=normals,
            max_distance=self.max_correspondence_distance,
        )
        scores = localizability(H)
        weak = [
            DOF_NAMES[i]
            for i in range(6)
            if float(scores[i]) < self.score_threshold
        ]
        return DegeneracyUpdate(
            scores=scores,
            hessian=H,
            degenerate=bool(weak),
            degenerate_axes=tuple(weak),
        )


class DegeneracyAwareMonitor:
    """Naive covariance PL, inflated anisotropically by localizability scores.

    Under-constrained epochs are flagged ``DEGRADED`` (or ``DO_NOT_TRUST`` when
    the weakest translational score is extremely low) so the detector is not a
    silent always-NOMINAL rubber stamp.
    """

    def __init__(
        self,
        *,
        confidence: float | None = None,
        score_threshold: float | None = None,
        dnt_threshold: float | None = None,
        loc_floor: float | None = None,
        max_inflation: float | None = None,
    ) -> None:
        cfg = CairnConfig()
        self.confidence = float(
            cfg.integrity.chi2_confidence if confidence is None else confidence
        )
        self.score_threshold = float(
            cfg.degeneracy.score_threshold if score_threshold is None else score_threshold
        )
        self.dnt_threshold = float(
            cfg.degeneracy.dnt_threshold if dnt_threshold is None else dnt_threshold
        )
        self.loc_floor = float(cfg.degeneracy.loc_floor if loc_floor is None else loc_floor)
        self.max_inflation = float(
            cfg.degeneracy.max_inflation if max_inflation is None else max_inflation
        )

    def report_one(
        self,
        estimate: PoseEstimate,
        scores: NDArray[np.float64],
    ) -> IntegrityReport:
        from cairn._internal.validate import untrusted_report, validate_pose_estimate
        from cairn.errors import CairnInputError

        try:
            validate_pose_estimate(estimate)
            scores = np.asarray(scores, dtype=np.float64)
            if scores.shape != (6,) or not np.all(np.isfinite(scores)):
                raise CairnInputError(
                    f"localizability scores must be finite shape (6,), "
                    f"got shape={getattr(scores, 'shape', None)}; "
                    f"source={estimate.source!r}, "
                    f"timestamp={estimate.pose.timestamp:.9f}"
                )
            inflated = inflate_covariance_anisotropic(
                estimate.covariance,
                scores,
                loc_floor=self.loc_floor,
                max_inflation=self.max_inflation,
            )
            hpl, vpl = protection_levels_from_covariance(
                inflated, confidence=self.confidence
            )
        except (CairnInputError, ValueError) as exc:
            logger.warning("degeneracy monitor rejecting estimate: %s", exc)
            return untrusted_report(
                estimate.pose.timestamp,
                reason=f"degeneracy_input_rejected:{exc}",
                source=estimate.source,
            )
        min_trans = float(np.min(scores[:3]))
        if min_trans < self.dnt_threshold:
            state = IntegrityState.DO_NOT_TRUST
            reason = (
                f"degeneracy_do_not_trust:"
                f"min_trans={min_trans:.6g}<dnt_threshold={self.dnt_threshold:.6g}"
            )
        elif min_trans < self.score_threshold:
            state = IntegrityState.DEGRADED
            reason = (
                f"degeneracy_degraded:"
                f"min_trans={min_trans:.6g}<score_threshold={self.score_threshold:.6g}"
            )
        else:
            state = IntegrityState.NOMINAL
            reason = (
                f"degeneracy_nominal:"
                f"min_trans={min_trans:.6g}>=score_threshold={self.score_threshold:.6g}"
            )
        weak = [DOF_NAMES[i] for i in range(6) if float(scores[i]) < self.score_threshold]
        if weak:
            reason = f"{reason}:axes={','.join(weak)}"
        return IntegrityReport(
            timestamp=estimate.pose.timestamp,
            protection_level_horizontal=hpl,
            protection_level_vertical=vpl,
            state=state,
            reason=reason,
            time_since_corroboration=0.0,
        )

    def diagnostic_parts(
        self,
        scores: NDArray[np.float64],
        report: IntegrityReport,
    ) -> tuple[float | None, tuple[TestStatRecord, ...], TriggerRecord | None]:
        """Localizability scores vs score/DNT thresholds for JSONL dumps."""
        s = np.asarray(scores, dtype=np.float64)
        tests = tuple(
            TestStatRecord(
                name=DOF_NAMES[i],
                statistic=float(s[i]),
                threshold=float(self.score_threshold),
                crossed=float(s[i]) < self.score_threshold,
                kind="localizability",
            )
            for i in range(6)
        )
        min_trans = float(np.min(s[:3]))
        min_axis = DOF_NAMES[int(np.argmin(s[:3]))]
        trigger = None
        if report.state is IntegrityState.DO_NOT_TRUST:
            trigger = TriggerRecord(
                name=min_axis,
                statistic=min_trans,
                threshold=float(self.dnt_threshold),
                kind="localizability_dnt",
                comparison="score < dnt_threshold",
            )
        elif report.state is IntegrityState.DEGRADED:
            trigger = TriggerRecord(
                name=min_axis,
                statistic=min_trans,
                threshold=float(self.score_threshold),
                kind="localizability",
                comparison="score < score_threshold",
            )
        return float(self.score_threshold), tests, trigger
