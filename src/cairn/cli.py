"""Cairn command-line harness (``cairn`` console script).

Not required for library integration — prefer
``from cairn import CairnMonitor, CairnConfig``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from cairn._internal.io.reports import load_integrity_reports
from cairn._internal.io.tum import load_tum
from cairn._internal.plots import plot_error_vs_pl, plot_stanford, plot_trajectories
from cairn.config import CairnConfig
from cairn.metrics.alignment import AlignMode
from cairn.metrics.error import ate_rmse, position_errors
from cairn.metrics.integrity import IntegrityReportSummary, integrity_report_summary
from cairn.types import IntegrityReport, Trajectory

__all__ = [
    "build_parser",
    "main",
    "run_score",
    "true_errors_at_reports",
]


def true_errors_at_reports(
    est: Trajectory,
    gt: Trajectory,
    reports: Sequence[IntegrityReport],
    *,
    align_mode: AlignMode = "umeyama",
    max_dt: float | None = None,
) -> NDArray[np.float64]:
    """Horizontal true error (m) nearest each integrity-report timestamp."""
    if max_dt is None:
        max_dt = CairnConfig().association.max_dt_s
    result = position_errors(est, gt, align_mode, max_dt=max_dt)
    match_times = np.array(
        [gt[j].pose.timestamp for _, j in result.matches], dtype=np.float64
    )
    if match_times.size == 0:
        raise ValueError("No matched poses between estimate and ground truth")

    errors = np.empty(len(reports), dtype=np.float64)
    for i, report in enumerate(reports):
        idx = int(np.argmin(np.abs(match_times - report.timestamp)))
        dt = abs(float(match_times[idx]) - float(report.timestamp))
        if dt > max_dt:
            raise ValueError(
                f"No trajectory match within {max_dt}s for report "
                f"timestamp {report.timestamp} (nearest dt={dt})"
            )
        errors[i] = result.horizontal[idx]
    return errors


def _summary_to_dict(
    summary: IntegrityReportSummary,
    *,
    ate: float,
) -> dict[str, object]:
    return {
        "epoch_count": summary.epoch_count,
        "missed_detection_rate": summary.missed_detection_rate,
        "false_alarm_rate": summary.false_alarm_rate,
        "availability": summary.availability,
        "ate_rmse_m": ate,
        "alert_limit_m": summary.stanford.alert_limit,
        "stanford": {
            "nominal_count": summary.stanford.nominal_count,
            "misleading_count": summary.stanford.misleading_count,
            "hazardously_misleading_count": summary.stanford.hazardously_misleading_count,
            "unavailable_count": summary.stanford.unavailable_count,
        },
    }


def _write_report_md(path: Path, payload: dict[str, object]) -> None:
    stanford = payload["stanford"]
    assert isinstance(stanford, dict)
    lines = [
        "# Cairn integrity score report",
        "",
        f"- Epochs: **{payload['epoch_count']}**",
        f"- Missed detection rate: **{payload['missed_detection_rate']:.6f}**",
        f"- False alarm rate: **{payload['false_alarm_rate']:.6f}**",
        f"- Availability: **{payload['availability']:.6f}**",
        f"- ATE RMSE: **{payload['ate_rmse_m']:.6f} m**",
        f"- Alert limit: **{payload['alert_limit_m']} m**",
        "",
        "## Stanford diagram counts",
        "",
        f"- Nominal: {stanford['nominal_count']}",
        f"- Misleading information: {stanford['misleading_count']}",
        f"- Hazardously misleading information: {stanford['hazardously_misleading_count']}",
        f"- Unavailable: {stanford['unavailable_count']}",
        "",
        "## Artifacts",
        "",
        "- `summary.json`",
        "- `config.yaml` (effective configuration for reproducibility)",
        "- `stanford.png`",
        "- `error_vs_pl.png`",
        "- `trajectories.png`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_score(
    *,
    est_path: Path,
    gt_path: Path,
    reports_path: Path,
    out_dir: Path,
    config: CairnConfig | None = None,
    alert_limit: float | None = None,
    align_mode: AlignMode | None = None,
) -> IntegrityReportSummary:
    """Run the full scoring pipeline and write artifacts to ``out_dir``.

    Always dumps the effective :class:`~cairn.config.CairnConfig` as
    ``config.yaml`` so any results directory is reproducible.
    """
    cfg = config if config is not None else CairnConfig.load_default()
    if alert_limit is not None:
        cfg = replace(
            cfg,
            integrity=replace(cfg.integrity, alert_limit_m=float(alert_limit)),
        )
    if align_mode is not None:
        cfg = replace(
            cfg,
            evaluation=replace(cfg.evaluation, align_mode=align_mode),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out_dir / "config.yaml")

    alert = cfg.integrity.alert_limit_m
    align = cfg.evaluation.align_mode
    max_dt = cfg.association.max_dt_s
    align_dt = cfg.association.alignment_max_dt_s

    est = load_tum(est_path)
    gt = load_tum(gt_path)
    reports = load_integrity_reports(reports_path)
    true_errors = true_errors_at_reports(
        est, gt, reports, align_mode=align, max_dt=max_dt
    )
    summary = integrity_report_summary(reports, true_errors, alert_limit=alert)
    ate = ate_rmse(est, gt, align, max_dt=align_dt)
    payload = _summary_to_dict(summary, ate=ate)

    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_report_md(out_dir / "report.md", payload)

    fig_s, ax_s = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
    plot_stanford(summary, ax=ax_s)
    fig_s.savefig(out_dir / "stanford.png", dpi=180, facecolor="white")
    plt.close(fig_s)

    fig_e, ax_e = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)
    plot_error_vs_pl(reports, true_errors, ax=ax_e)
    fig_e.savefig(out_dir / "error_vs_pl.png", dpi=160, facecolor="white")
    plt.close(fig_e)

    fig_t, axs_t = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    plot_trajectories(est, gt, axs=(axs_t[0], axs_t[1]))
    fig_t.savefig(out_dir / "trajectories.png", dpi=160, facecolor="white")
    plt.close(fig_t)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cairn", description="Cairn integrity tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="Score estimate vs ground truth with integrity reports")
    score.add_argument("--est", type=Path, required=True, help="Estimated trajectory (TUM)")
    score.add_argument("--gt", type=Path, required=True, help="Ground-truth trajectory (TUM)")
    score.add_argument(
        "--reports",
        type=Path,
        required=True,
        help="Integrity reports text file",
    )
    score.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config path (default: configs/default.yaml when present)",
    )
    score.add_argument(
        "--alert-limit",
        type=float,
        default=None,
        help="Horizontal alert limit in metres (overrides config.integrity.alert_limit_m)",
    )
    score.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for summary.json, config.yaml, report.md, and PNGs",
    )
    score.add_argument(
        "--align",
        choices=("none", "origin", "umeyama"),
        default=None,
        help="Trajectory alignment mode (overrides config.evaluation.align_mode)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "score":
        if args.config is not None:
            cfg = CairnConfig.from_yaml(args.config)
        else:
            cfg = CairnConfig.load_default()
        if args.alert_limit is not None and args.alert_limit <= 0.0:
            parser.error("--alert-limit must be positive")
        summary = run_score(
            est_path=args.est,
            gt_path=args.gt,
            reports_path=args.reports,
            out_dir=args.out,
            config=cfg,
            alert_limit=args.alert_limit,
            align_mode=args.align,
        )
        print(
            f"Wrote artifacts to {args.out} "
            f"(MDR={summary.missed_detection_rate:.6f}, "
            f"epochs={summary.epoch_count})"
        )
        return 0

    parser.error(f"Unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
