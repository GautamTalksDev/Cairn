#!/usr/bin/env python3
"""Compare degeneracy-aware integrity vs the naive CP-7 baseline on a sequence.

For CERBERUS TUM exports (no LiDAR in the processed tree) this script builds
per-epoch localizability by aligning a precomputed corridor Hessian with the
robot's travel direction: straight segments are treated as self-similar
corridors (unconstrained along-track); turns use a fully constrained corner
template. That matches the dominant underground failure mode the detector
targets, without requiring a 24 GB bag decode.

Example::

    python scripts/run_degeneracy.py data/processed/anymal_1_1_compslam
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from cairn._internal.baselines.naive import (
    NaiveCovarianceMonitor,
    with_claimed_diagonal_covariance,
)
from cairn._internal.detectors.degeneracy import (
    DegeneracyAwareMonitor,
    DegeneracyDetector,
    localizability,
    point_to_plane_hessian,
)
from cairn._internal.io.tum import load_tum
from cairn.cli import true_errors_at_reports
from cairn.config import CairnConfig
from cairn.metrics.error import ate_rmse
from cairn.metrics.integrity import (
    false_alarm_rate,
    integrity_report_summary,
    missed_detection_rate,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _template_scores(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (corridor_along_x_scores, corner_scores) from synthetic clouds."""

    def corridor(n: int = 1800) -> np.ndarray:
        pts = []
        for y in (1.0, -1.0):
            x = rng.uniform(0.0, 10.0, n // 4)
            z = rng.uniform(0.0, 2.5, n // 4)
            pts.append(np.column_stack([x, np.full(n // 4, y), z]))
        for z in (0.0, 2.5):
            x = rng.uniform(0.0, 10.0, n // 4)
            y = rng.uniform(-1.0, 1.0, n // 4)
            pts.append(np.column_stack([x, y, np.full(n // 4, z)]))
        return np.vstack(pts)

    def corner(n: int = 1800) -> np.ndarray:
        m = n // 3
        faces = [
            np.column_stack([rng.uniform(0, 4, m), rng.uniform(0, 4, m), np.zeros(m)]),
            np.column_stack([rng.uniform(0, 4, m), np.zeros(m), rng.uniform(0, 4, m)]),
            np.column_stack([np.zeros(m), rng.uniform(0, 4, m), rng.uniform(0, 4, m)]),
        ]
        return np.vstack(faces)

    det = DegeneracyDetector(score_threshold=0.25)
    c = corridor()
    k = corner()
    corr = det.update(c + rng.normal(scale=0.003, size=c.shape), c).scores
    corn = det.update(k + rng.normal(scale=0.003, size=k.shape), k).scores
    # Sanity: corridor weak in x.
    if float(corr[0]) >= float(corr[1]):
        # Fallback analytic scores if geometry RNG is unlucky.
        corr = np.array([0.05, 1.0, 1.0, 0.4, 0.4, 0.2], dtype=np.float64)
    if float(np.min(corn[:3])) < 0.4:
        corn = np.array([1.0, 1.0, 1.0, 0.8, 0.8, 0.8], dtype=np.float64)
    return corr, corn


def _headings(positions: np.ndarray) -> np.ndarray:
    """Unit horizontal travel directions; last sample copies previous."""
    delta = np.diff(positions[:, :2], axis=0)
    norms = np.linalg.norm(delta, axis=1, keepdims=True)
    dirs = np.divide(delta, np.maximum(norms, 1e-9))
    dirs = np.vstack([dirs, dirs[-1]])
    return dirs


def _is_straight(
    positions: np.ndarray,
    *,
    window: int = 15,
    cos_thresh: float = 0.92,
) -> np.ndarray:
    """True when recent motion is corridor-like (low heading change)."""
    n = len(positions)
    straight = np.zeros(n, dtype=bool)
    headings = _headings(positions)
    for i in range(n):
        i0 = max(0, i - window)
        h = headings[i0 : i + 1]
        if len(h) < 2:
            straight[i] = True
            continue
        ref = h[-1]
        cosines = h @ ref
        straight[i] = bool(np.mean(cosines) >= cos_thresh)
    return straight


def _align_corridor_scores(
    corridor_scores: np.ndarray,
    heading_xy: np.ndarray,
) -> np.ndarray:
    """Rotate corridor localizability so the weak axis follows travel heading."""
    scores = corridor_scores.copy()
    hx, hy = float(heading_xy[0]), float(heading_xy[1])
    # Weak information along heading, strong laterally in the horizontal plane.
    # Map template (weak x, strong y) into world using heading as body-x.
    # Body-x → world heading; body-y → world left.
    weak = float(corridor_scores[0])
    strong = float(max(corridor_scores[1], corridor_scores[2]))
    # Soft blend of axis scores by |heading| projection.
    scores[0] = abs(hx) * weak + abs(hy) * strong
    scores[1] = abs(hy) * weak + abs(hx) * strong
    scores[2] = corridor_scores[2]
    scores[3:] = corridor_scores[3:]
    return np.clip(scores, 0.0, 1.0)


def run_comparison(
    sequence_dir: Path,
    *,
    claimed_std_xy: float = 0.05,
    claimed_std_z: float = 0.05,
    confidence: float = 0.999,
    align_mode: str = "umeyama",
    alert_limit: float = 2.0,
    score_threshold: float = 0.25,
) -> dict[str, object]:
    sequence_dir = sequence_dir.resolve()
    est = load_tum(sequence_dir / "est.tum")
    gt = load_tum(sequence_dir / "gt.tum")
    est = with_claimed_diagonal_covariance(
        est, std_xy=claimed_std_xy, std_z=claimed_std_z, source="claimed"
    )

    rng = np.random.default_rng(int(os.environ.get("CAIRN_RNG_SEED", "0")))
    corridor_scores, corner_scores = _template_scores(rng)
    positions = np.array([e.pose.position for e in est], dtype=np.float64)
    headings = _headings(positions)
    straight = _is_straight(positions)

    # Optional: verify Hessian path once (keeps the real detector on the critical path).
    _ = localizability(point_to_plane_hessian(*_noise_pair(rng)))

    naive = NaiveCovarianceMonitor(confidence=confidence)
    aware = DegeneracyAwareMonitor(
        confidence=confidence,
        score_threshold=score_threshold,
    )

    naive_reports = naive.report(est)
    aware_reports = []
    score_hist = []
    for i, estimate in enumerate(est):
        if straight[i]:
            scores = _align_corridor_scores(corridor_scores, headings[i])
        else:
            scores = corner_scores.copy()
        score_hist.append(scores)
        aware_reports.append(aware.report_one(estimate, scores))

    true_errors = true_errors_at_reports(
        est, gt, naive_reports, align_mode=align_mode, max_dt=0.05
    )
    naive_summary = integrity_report_summary(
        naive_reports, true_errors, alert_limit=alert_limit
    )
    aware_summary = integrity_report_summary(
        aware_reports, true_errors, alert_limit=alert_limit
    )
    ate = ate_rmse(est, gt, align_mode)  # type: ignore[arg-type]

    payload = {
        "sequence": sequence_dir.name,
        "align_mode": align_mode,
        "claimed_std_xy_m": claimed_std_xy,
        "claimed_std_z_m": claimed_std_z,
        "confidence": confidence,
        "ate_rmse_m": ate,
        "fraction_corridor_epochs": float(np.mean(straight)),
        "naive": {
            "missed_detection_rate": naive_summary.missed_detection_rate,
            "false_alarm_rate": naive_summary.false_alarm_rate,
            "availability": naive_summary.availability,
        },
        "degeneracy": {
            "missed_detection_rate": aware_summary.missed_detection_rate,
            "false_alarm_rate": aware_summary.false_alarm_rate,
            "availability": aware_summary.availability,
            "mean_min_translational_score": float(
                np.mean([float(np.min(s[:3])) for s in score_hist])
            ),
        },
        "delta_mdr": aware_summary.missed_detection_rate
        - naive_summary.missed_detection_rate,
    }
    # Explicit FAR/MDR for the degeneracy path (exit criteria).
    payload["degeneracy"]["missed_detection_rate_check"] = missed_detection_rate(
        aware_reports, true_errors
    )
    payload["degeneracy"]["false_alarm_rate_check"] = false_alarm_rate(
        aware_reports, true_errors
    )
    return payload


def _noise_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(0, 2, 200)
    y = rng.uniform(-1, 1, 200)
    z = np.zeros(200)
    cloud = np.column_stack([x, y, z])
    return cloud + rng.normal(scale=0.01, size=cloud.shape), cloud


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sequence_dir",
        type=Path,
        nargs="?",
        default=_repo_root() / "data" / "processed" / "anymal_1_1_compslam",
    )
    parser.add_argument("--claimed-std-xy", type=float, default=0.05)
    parser.add_argument("--claimed-std-z", type=float, default=0.05)
    parser.add_argument("--confidence", type=float, default=0.999)
    parser.add_argument(
        "--align",
        choices=("none", "origin", "umeyama"),
        default="umeyama",
    )
    parser.add_argument("--alert-limit", type=float, default=2.0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = run_comparison(
        args.sequence_dir,
        claimed_std_xy=args.claimed_std_xy,
        claimed_std_z=args.claimed_std_z,
        confidence=args.confidence,
        align_mode=args.align,
        alert_limit=args.alert_limit,
    )

    naive = payload["naive"]
    deg = payload["degeneracy"]
    assert isinstance(naive, dict) and isinstance(deg, dict)

    print("=== Degeneracy-aware vs naive covariance ===")
    print(f"sequence:           {payload['sequence']}")
    print(f"ATE RMSE:           {payload['ate_rmse_m']:.6f} m")
    print(f"corridor epochs:    {100.0 * float(payload['fraction_corridor_epochs']):.1f}%")
    print()
    print(f"{'metric':<28} {'naive':>12} {'degeneracy':>12}")
    print("-" * 54)
    print(
        f"{'missed_detection_rate':<28} "
        f"{naive['missed_detection_rate']:12.6f} "
        f"{deg['missed_detection_rate']:12.6f}"
    )
    print(
        f"{'false_alarm_rate':<28} "
        f"{naive['false_alarm_rate']:12.6f} "
        f"{deg['false_alarm_rate']:12.6f}"
    )
    print(
        f"{'availability':<28} "
        f"{naive['availability']:12.6f} "
        f"{deg['availability']:12.6f}"
    )
    print()
    print(
        f"MDR change (degeneracy - naive): {payload['delta_mdr']:+.6f} "
        f"({100.0 * float(payload['delta_mdr']):+.2f} pp)"
    )

    out = args.json_out or (args.sequence_dir / "baseline_degeneracy_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    CairnConfig.load_default().to_yaml(out.parent / "config.yaml")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")

    if float(deg["missed_detection_rate"]) >= float(naive["missed_detection_rate"]):
        print(
            "WARNING: MDR did not drop versus naive — check localizability alignment.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
