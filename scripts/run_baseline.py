#!/usr/bin/env python3
"""Run the naive covariance integrity baseline on a prepared sequence.

Example::

    python scripts/run_baseline.py data/processed/anymal_1_1_compslam

CERBERUS TUM files have no covariance column. By default this script attaches a
diagonal self-reported covariance with ``--claimed-std-xy`` / ``--claimed-std-z``
(default 5 cm), representing a typical overconfident filter self-report, then
runs :class:`~cairn.baselines.naive.NaiveCovarianceMonitor`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cairn._internal.baselines.naive import (
    NaiveCovarianceMonitor,
    with_claimed_diagonal_covariance,
)
from cairn._internal.io.reports import save_integrity_reports
from cairn._internal.io.tum import load_tum
from cairn.cli import true_errors_at_reports
from cairn.config import CairnConfig
from cairn.metrics.error import ate_rmse
from cairn.metrics.integrity import integrity_report_summary


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_baseline(
    sequence_dir: Path,
    *,
    confidence: float = 0.999,
    claimed_std_xy: float = 0.05,
    claimed_std_z: float = 0.05,
    align_mode: str = "umeyama",
    alert_limit: float = 2.0,
    write_reports: bool = True,
) -> dict[str, object]:
    sequence_dir = sequence_dir.resolve()
    est_path = sequence_dir / "est.tum"
    gt_path = sequence_dir / "gt.tum"
    if not est_path.is_file() or not gt_path.is_file():
        raise FileNotFoundError(
            f"Expected est.tum and gt.tum under {sequence_dir} "
            "(run scripts/prepare_cerberus.py first)"
        )

    est = load_tum(est_path)
    gt = load_tum(gt_path)

    # Attach filter self-report (CERBERUS CSV has none).
    est = with_claimed_diagonal_covariance(
        est,
        std_xy=claimed_std_xy,
        std_z=claimed_std_z,
        source="naive_claimed_cov",
    )

    monitor = NaiveCovarianceMonitor(confidence=confidence)
    reports = monitor.report(est)
    if write_reports:
        save_integrity_reports(reports, sequence_dir / "reports_naive.txt")

    true_errors = true_errors_at_reports(
        est, gt, reports, align_mode=align_mode, max_dt=0.05
    )
    summary = integrity_report_summary(reports, true_errors, alert_limit=alert_limit)
    ate = ate_rmse(est, gt, align_mode)  # type: ignore[arg-type]

    payload: dict[str, object] = {
        "sequence": sequence_dir.name,
        "monitor": "NaiveCovarianceMonitor",
        "confidence": confidence,
        "claimed_std_xy_m": claimed_std_xy,
        "claimed_std_z_m": claimed_std_z,
        "align_mode": align_mode,
        "alert_limit_m": alert_limit,
        "epoch_count": summary.epoch_count,
        "missed_detection_rate": summary.missed_detection_rate,
        "false_alarm_rate": summary.false_alarm_rate,
        "availability": summary.availability,
        "ate_rmse_m": ate,
        "stanford": {
            "nominal_count": summary.stanford.nominal_count,
            "misleading_count": summary.stanford.misleading_count,
            "hazardously_misleading_count": summary.stanford.hazardously_misleading_count,
            "unavailable_count": summary.stanford.unavailable_count,
        },
        "n_missed_detections": int(
            sum(
                (err > r.protection_level_horizontal)
                and (r.state.name == "NOMINAL")
                for r, err in zip(reports, true_errors, strict=True)
            )
        ),
    }
    return payload


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
        help="Prepared sequence directory (default: anymal_1 CompSLAM)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.999,
        help="Chi-squared confidence for PL scaling (default: 0.999 = 99.9%%)",
    )
    parser.add_argument(
        "--claimed-std-xy",
        type=float,
        default=0.05,
        help="Self-reported horizontal position σ in metres (default: 0.05)",
    )
    parser.add_argument(
        "--claimed-std-z",
        type=float,
        default=0.05,
        help="Self-reported vertical position σ in metres (default: 0.05)",
    )
    parser.add_argument(
        "--align",
        choices=("none", "origin", "umeyama"),
        default="umeyama",
        help="Trajectory alignment before true-error comparison",
    )
    parser.add_argument(
        "--alert-limit",
        type=float,
        default=2.0,
        help="Stanford diagram alert limit in metres",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the summary JSON",
    )
    args = parser.parse_args(argv)

    payload = run_baseline(
        args.sequence_dir,
        confidence=args.confidence,
        claimed_std_xy=args.claimed_std_xy,
        claimed_std_z=args.claimed_std_z,
        align_mode=args.align,
        alert_limit=args.alert_limit,
    )

    mdr = float(payload["missed_detection_rate"])  # type: ignore[arg-type]
    print("=== Naive covariance baseline ===")
    print(f"sequence:              {payload['sequence']}")
    print(f"claimed σ_xy / σ_z:    {payload['claimed_std_xy_m']} / {payload['claimed_std_z_m']} m")
    print(f"chi² confidence:       {payload['confidence']}")
    print(f"align:                 {payload['align_mode']}")
    print(f"epochs:                {payload['epoch_count']}")
    print(f"ATE RMSE:              {payload['ate_rmse_m']:.6f} m")
    print(f"availability:          {payload['availability']:.6f}")
    print(f"false_alarm_rate:      {payload['false_alarm_rate']:.6f}")
    print(f"missed_detection_rate: {mdr:.6f}")
    print(f"missed_detection_pct:  {100.0 * mdr:.2f}%")
    print(f"n_missed_detections:   {payload['n_missed_detections']}")
    stanford = payload["stanford"]  # type: ignore[assignment]
    assert isinstance(stanford, dict)
    print(
        "stanford counts:       "
        f"N={stanford['nominal_count']} MI={stanford['misleading_count']} "
        f"HMI={stanford['hazardously_misleading_count']} "
        f"UA={stanford['unavailable_count']}"
    )

    out = args.json_out or (args.sequence_dir / "baseline_naive_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    CairnConfig.load_default().to_yaml(out.parent / "config.yaml")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
