"""CP-12 — End-to-end regression against stored golden headline numbers.

Locks the publishable table (clean naive@σ*, synthetic drift for all monitors,
common-mode limitation, drift-timing duty metric) so refactors cannot silently
change them.

Requires prepared sequences under ``data/processed/``. Skip otherwise.

Run::

    pytest -m slow -q
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cairn._internal.baselines.naive import with_claimed_diagonal_covariance
from cairn._internal.io.tum import load_tum
from cairn.cli import true_errors_at_reports
from cairn.metrics.error import ate_rmse
from cairn.metrics.integrity import integrity_report_summary
from cairn.types import IntegrityState, Pose, PoseEstimate, Trajectory

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"
GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "expected_results.json"

CLEAN = "anymal_1_1_compslam"
DRIFT = "anymal_1_1_compslam_drift"

REL_TOL = 1e-3
ABS_FLOOR = 1e-12


def _load_script(module_name: str, relative: str) -> Any:
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _require_sequences() -> None:
    missing = [
        name
        for name in (CLEAN, DRIFT)
        if not (PROCESSED / name / "est.tum").is_file()
        or not (PROCESSED / name / "gt.tum").is_file()
    ]
    if missing:
        pytest.skip(
            "prepared CERBERUS sequences required for slow e2e regression: "
            + ", ".join(missing)
            + " (run scripts/prepare_cerberus.py and scripts/inject_faults.py)"
        )


def _load_golden() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _expected(entry: dict[str, Any] | float | bool) -> Any:
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


def assert_close(
    actual: float,
    expected: float,
    *,
    name: str,
    rel: float = REL_TOL,
) -> None:
    """Relative comparison with an absolute floor; prints both sides on failure."""
    exp = float(expected)
    act = float(actual)
    scale = max(abs(exp), abs(act), ABS_FLOOR)
    if abs(act - exp) > rel * scale:
        pytest.fail(
            f"{name}: expected {exp!r}, got {act!r} "
            f"(rel_tol={rel}, abs_diff={abs(act - exp)!r})"
        )


def assert_bool(actual: bool, expected: bool, *, name: str) -> None:
    if bool(actual) is not bool(expected):
        pytest.fail(f"{name}: expected {expected!r}, got {actual!r}")


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    return _load_golden()


@pytest.fixture(scope="module")
def pipeline(golden: dict[str, Any]) -> dict[str, Any]:
    """Run the full scored pipeline once per module (slow)."""
    _require_sequences()
    meta = golden["_meta"]
    sigma = float(_expected(golden["sigma_star_m"]))
    align = str(meta["align"])
    confidence = float(meta["confidence"])
    alert_limit = float(meta["alert_limit_m"])
    score_threshold = float(meta["score_threshold"])
    rate = float(meta["drift_rate_m_per_s"])
    start_frac = float(meta["drift_start_frac"])

    ms = _load_script("cairn_e2e_multiseq", "scripts/run_all_sequences.py")
    cm = _load_script("cairn_e2e_common_mode", "scripts/common_mode_fault.py")
    dt = _load_script("cairn_e2e_drift_timing", "scripts/drift_timing.py")

    def score_sequence(seq_name: str) -> dict[str, Any]:
        est = load_tum(PROCESSED / seq_name / "est.tum")
        gt = load_tum(PROCESSED / seq_name / "gt.tum")
        ate = float(ate_rmse(est, gt, align))  # type: ignore[arg-type]
        out: dict[str, Any] = {"ate_rmse_m": ate, "n_epochs": len(est)}

        naive_r, naive_s = ms.run_naive(
            est,
            gt,
            sigma=sigma,
            confidence=confidence,
            align=align,
            alert_limit=alert_limit,
        )
        out["naive"] = {
            "missed_detection_rate": float(naive_s.missed_detection_rate),
            "false_alarm_rate": float(naive_s.false_alarm_rate),
            "availability": float(naive_s.availability),
            "mean_pl_m": float(
                np.mean([r.protection_level_horizontal for r in naive_r])
            ),
            "p95_pl_m": float(
                np.percentile([r.protection_level_horizontal for r in naive_r], 95)
            ),
        }

        deg_r, deg_s = ms.run_degeneracy(
            est,
            gt,
            sigma=sigma,
            confidence=confidence,
            align=align,
            alert_limit=alert_limit,
            score_threshold=score_threshold,
        )
        out["degeneracy"] = {
            "missed_detection_rate": float(deg_s.missed_detection_rate),
            "false_alarm_rate": float(deg_s.false_alarm_rate),
            "availability": float(deg_s.availability),
            "mean_pl_m": float(np.mean([r.protection_level_horizontal for r in deg_r])),
            "p95_pl_m": float(
                np.percentile([r.protection_level_horizontal for r in deg_r], 95)
            ),
        }

        con_r, con_s = ms.run_consistency(
            est, gt, sigma=sigma, align=align, alert_limit=alert_limit
        )
        out["consistency"] = {
            "missed_detection_rate": float(con_s.missed_detection_rate),
            "false_alarm_rate": float(con_s.false_alarm_rate),
            "availability": float(con_s.availability),
            "mean_pl_m": float(np.mean([r.protection_level_horizontal for r in con_r])),
            "p95_pl_m": float(
                np.percentile([r.protection_level_horizontal for r in con_r], 95)
            ),
            "reports": con_r,
        }
        out["est"] = est
        out["gt"] = gt
        return out

    clean = score_sequence(CLEAN)
    drift = score_sequence(DRIFT)

    # Common-mode limitation (same offset on all SS sources).
    clean_est = load_tum(PROCESSED / CLEAN / "est.tum")
    clean_gt = load_tum(PROCESSED / CLEAN / "gt.tum")
    offsets, _t_onset, onset_idx = cm._common_mode_offset(
        clean_est, rate_m_per_s=rate, start_frac=start_frac
    )
    cm_reports, _stats, _hpls = cm.run_consistency_common_mode(
        clean_est, offsets, sigma=sigma
    )
    drifted_est = Trajectory(
        [
            PoseEstimate(
                pose=Pose(
                    timestamp=pe.pose.timestamp,
                    position=pe.pose.position + offsets[i],
                    orientation=pe.pose.orientation,
                    frame=pe.pose.frame,
                ),
                covariance=pe.covariance.copy(),
                source="common_mode",
            )
            for i, pe in enumerate(clean_est)
        ]
    )
    est_c = with_claimed_diagonal_covariance(
        drifted_est, std_xy=sigma, std_z=sigma, source="cm"
    )
    cm_errors = true_errors_at_reports(
        est_c, clean_gt, cm_reports, align_mode=align, max_dt=0.05
    )
    cm_summary = integrity_report_summary(
        cm_reports, cm_errors, alert_limit=alert_limit
    )
    cm_alarmed = np.array(
        [r.state != IntegrityState.NOMINAL for r in cm_reports], dtype=bool
    )
    common_mode = {
        "missed_detection_rate": float(cm_summary.missed_detection_rate),
        "false_alarm_rate": float(cm_summary.false_alarm_rate),
        "availability": float(cm_summary.availability),
        "alarm_fraction_after_onset": float(np.mean(cm_alarmed[onset_idx:])),
    }

    # Drift timing — headline duty metric (reuse drift consistency reports).
    _, _, t_onset_abs, _ = dt._injected_drift_series(
        clean_est, rate_m_per_s=rate, start_frac=start_frac
    )
    t0_abs = float(clean_est[0].pose.timestamp)
    est_d = with_claimed_diagonal_covariance(
        drift["est"], std_xy=sigma, std_z=sigma, source="timing"
    )
    timing_errors = true_errors_at_reports(
        est_d, drift["gt"], drift["consistency"]["reports"], align_mode=align, max_dt=0.05
    )
    duty = dt._alarm_duty_crossing(
        drift["consistency"]["reports"],
        timing_errors,
        t_onset_abs=t_onset_abs,
        t0_abs=t0_abs,
        window_s=5.0,
        duty=0.5,
    )

    return {
        "sigma": sigma,
        "clean": clean,
        "drift": drift,
        "common_mode": common_mode,
        "drift_timing": duty,
    }


@pytest.mark.slow
class TestEndToEndGolden:
    def test_sigma_star_matches_golden(
        self, golden: dict[str, Any], pipeline: dict[str, Any]
    ) -> None:
        assert_close(
            pipeline["sigma"],
            _expected(golden["sigma_star_m"]),
            name="sigma_star_m",
        )

    def test_clean_naive_at_sigma_star(
        self, golden: dict[str, Any], pipeline: dict[str, Any]
    ) -> None:
        exp = golden["clean_anymal_1"]
        act = pipeline["clean"]
        assert_close(act["ate_rmse_m"], _expected(exp["ate_rmse_m"]), name="clean.ate")
        for key in ("missed_detection_rate", "false_alarm_rate", "availability", "mean_pl_m"):
            assert_close(
                act["naive"][key],
                _expected(exp["naive"][key]),
                name=f"clean.naive.{key}",
            )

    def test_clean_degeneracy_and_consistency(
        self, golden: dict[str, Any], pipeline: dict[str, Any]
    ) -> None:
        exp = golden["clean_anymal_1"]
        act = pipeline["clean"]
        for key in ("missed_detection_rate", "false_alarm_rate", "mean_pl_m", "p95_pl_m"):
            assert_close(
                act["degeneracy"][key],
                _expected(exp["degeneracy"][key]),
                name=f"clean.degeneracy.{key}",
            )
        for key in ("missed_detection_rate", "false_alarm_rate", "availability"):
            assert_close(
                act["consistency"][key],
                _expected(exp["consistency"][key]),
                name=f"clean.consistency.{key}",
            )

    def test_drift_all_monitors(
        self, golden: dict[str, Any], pipeline: dict[str, Any]
    ) -> None:
        exp = golden["drift_injected"]
        act = pipeline["drift"]
        assert_close(act["ate_rmse_m"], _expected(exp["ate_rmse_m"]), name="drift.ate")

        for key in ("missed_detection_rate", "false_alarm_rate", "availability"):
            assert_close(
                act["naive"][key],
                _expected(exp["naive"][key]),
                name=f"drift.naive.{key}",
            )
        for key in ("missed_detection_rate", "false_alarm_rate"):
            assert_close(
                act["degeneracy"][key],
                _expected(exp["degeneracy"][key]),
                name=f"drift.degeneracy.{key}",
            )
        for key in (
            "missed_detection_rate",
            "false_alarm_rate",
            "availability",
            "mean_pl_m",
            "p95_pl_m",
        ):
            assert_close(
                act["consistency"][key],
                _expected(exp["consistency"][key]),
                name=f"drift.consistency.{key}",
            )

        # Order-of-magnitude product claim: consistency MDR ≪ naive MDR.
        assert act["consistency"]["missed_detection_rate"] < 0.25
        assert act["naive"]["missed_detection_rate"] > 0.85

    def test_common_mode_limitation(
        self, golden: dict[str, Any], pipeline: dict[str, Any]
    ) -> None:
        exp = golden["common_mode"]
        act = pipeline["common_mode"]
        for key in (
            "missed_detection_rate",
            "false_alarm_rate",
            "availability",
            "alarm_fraction_after_onset",
        ):
            assert_close(act[key], _expected(exp[key]), name=f"common_mode.{key}")
        # Must remain a miss: high MDR, low post-onset alarm fraction.
        assert act["missed_detection_rate"] > 0.5
        assert act["alarm_fraction_after_onset"] < 0.25

    def test_drift_timing_duty_metric(
        self, golden: dict[str, Any], pipeline: dict[str, Any]
    ) -> None:
        exp = golden["drift_timing"]
        act = pipeline["drift_timing"]
        assert_bool(
            bool(act.get("detected")),
            bool(_expected(exp["consistency_detected"])),
            name="drift_timing.detected",
        )
        assert_close(
            float(act["time_to_alarm_s"]),
            _expected(exp["consistency_time_to_detection_s"]),
            name="drift_timing.time_to_detection_s",
        )
        assert_close(
            float(act["true_error_at_alarm_m"]),
            _expected(exp["consistency_true_error_at_detection_m"]),
            name="drift_timing.true_error_at_detection_m",
        )
