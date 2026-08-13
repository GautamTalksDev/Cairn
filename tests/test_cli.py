"""CLI harness tests for ``cairn score``."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cairn._internal.io.reports import save_integrity_reports
from cairn._internal.io.tum import save_tum
from cairn.cli import main
from cairn.types import IntegrityReport, IntegrityState, Pose, PoseEstimate, Trajectory


def _quat_yaw(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def _synthetic_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build matched est/gt TUM files plus integrity reports."""
    n = 20
    timestamps = np.arange(n, dtype=np.float64) * 0.1
    gt_estimates: list[PoseEstimate] = []
    est_estimates: list[PoseEstimate] = []
    reports: list[IntegrityReport] = []

    for i, t in enumerate(timestamps):
        yaw = 0.05 * i
        gt_pos = np.array([0.5 * i, 0.1 * math.sin(0.3 * i), 0.02 * i], dtype=np.float64)
        # Small horizontal bias so errors are non-zero but usually under PL.
        bias = np.array([0.05, -0.03, 0.0], dtype=np.float64)
        if i == 12:
            bias = np.array([1.5, 0.0, 0.0], dtype=np.float64)  # large spike
        est_pos = gt_pos + bias

        q = _quat_yaw(yaw)
        gt_estimates.append(
            PoseEstimate(
                pose=Pose(timestamp=float(t), position=gt_pos, orientation=q),
                covariance=np.eye(6),
                source="gt",
            )
        )
        est_estimates.append(
            PoseEstimate(
                pose=Pose(timestamp=float(t), position=est_pos, orientation=q),
                covariance=np.eye(6),
                source="est",
            )
        )

        # PL=0.5 m; most epochs safe; spike epoch is a missed detection if NOMINAL.
        state = IntegrityState.NOMINAL
        pl = 0.5
        if i in (5, 6):
            state = IntegrityState.DEGRADED
            pl = 1.0
        reports.append(
            IntegrityReport(
                timestamp=float(t),
                protection_level_horizontal=pl,
                protection_level_vertical=pl,
                state=state,
                reason="synthetic",
                time_since_corroboration=0.0,
            )
        )

    est_path = tmp_path / "est.txt"
    gt_path = tmp_path / "gt.txt"
    reports_path = tmp_path / "reports.txt"
    save_tum(Trajectory(est_estimates), est_path)
    save_tum(Trajectory(gt_estimates), gt_path)
    save_integrity_reports(reports, reports_path)
    return est_path, gt_path, reports_path


def test_cairn_score_writes_expected_artifacts(tmp_path: Path) -> None:
    est_path, gt_path, reports_path = _synthetic_bundle(tmp_path)
    out_dir = tmp_path / "artifacts"

    rc = main(
        [
            "score",
            "--est",
            str(est_path),
            "--gt",
            str(gt_path),
            "--reports",
            str(reports_path),
            "--alert-limit",
            "2.0",
            "--out",
            str(out_dir),
        ]
    )

    assert rc == 0
    expected = [
        out_dir / "summary.json",
        out_dir / "config.yaml",
        out_dir / "report.md",
        out_dir / "stanford.png",
        out_dir / "error_vs_pl.png",
        out_dir / "trajectories.png",
    ]
    for path in expected:
        assert path.is_file(), f"missing artifact: {path}"
        assert path.stat().st_size > 0

    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["epoch_count"] == 20
    assert "missed_detection_rate" in payload
    assert (out_dir / "report.md").read_text(encoding="utf-8").startswith("# Cairn")
    assert "alert_limit_m" in (out_dir / "config.yaml").read_text(encoding="utf-8")


def test_cairn_score_uses_config_alert_limit(tmp_path: Path) -> None:
    est_path, gt_path, reports_path = _synthetic_bundle(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "integrity:\n  alert_limit_m: 3.5\n  chi2_confidence: 0.999\n"
        "  p_fa: 0.05\n  pl_confidence: 0.999\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "from_cfg"
    rc = main(
        [
            "score",
            "--est",
            str(est_path),
            "--gt",
            str(gt_path),
            "--reports",
            str(reports_path),
            "--config",
            str(cfg_path),
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["alert_limit_m"] == 3.5


def test_cairn_score_rejects_nonpositive_alert_limit(tmp_path: Path) -> None:
    est_path, gt_path, reports_path = _synthetic_bundle(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "score",
                "--est",
                str(est_path),
                "--gt",
                str(gt_path),
                "--reports",
                str(reports_path),
                "--alert-limit",
                "0",
                "--out",
                str(tmp_path / "out"),
            ]
        )
    assert excinfo.value.code == 2
