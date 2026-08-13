#!/usr/bin/env python3
"""Common-mode fault — the honest limitation of solution separation.

Solution separation assumes conditionally independent sources. When the *same*
drift is applied to every source simultaneously, leave-one-out separations stay
near zero and consistency reports NOMINAL while true error grows.

This is stated in ``cairn.detectors.consistency`` (CP-9). Reporting it on a
limitations slide *before* a reviewer finds it is what makes the single-source
drift result credible.

**Synthetic fault.** Not observed mine data.

Outputs: ``outputs/common_mode/``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cairn._internal.baselines.naive import with_claimed_diagonal_covariance
from cairn._internal.detectors.consistency import SolutionSeparationDetector
from cairn._internal.io.tum import load_tum
from cairn.cli import true_errors_at_reports
from cairn.config import CairnConfig
from cairn.metrics.error import ate_rmse
from cairn.metrics.integrity import integrity_report_summary
from cairn.types import IntegrityReport, IntegrityState, Pose, PoseEstimate, Trajectory


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_sigma_star(default: float = 0.0694) -> float:
    path = _repo_root() / "outputs" / "sigma_sweep" / "sigma_sweep.json"
    if not path.is_file():
        return default
    payload = json.loads(path.read_text(encoding="utf-8"))
    crossing = payload.get("crossing") or {}
    sigma = crossing.get("sigma_xy_m")
    return float(sigma) if sigma is not None else default


def _headings_xy(positions: np.ndarray) -> np.ndarray:
    delta = np.diff(positions[:, :2], axis=0)
    norms = np.linalg.norm(delta, axis=1, keepdims=True)
    dirs = np.divide(delta, np.maximum(norms, 1e-9))
    return np.vstack([dirs, dirs[-1]])


def _common_mode_offset(
    clean: Trajectory,
    *,
    rate_m_per_s: float,
    start_frac: float,
) -> tuple[np.ndarray, float, int]:
    """Per-epoch world-XY offset applied to ALL sources (common-mode drift)."""
    n = len(clean)
    i0 = int(n * start_frac)
    t0 = float(clean[i0].pose.timestamp)
    positions = np.array([e.pose.position for e in clean], dtype=np.float64)
    headings = _headings_xy(positions)
    offsets = np.zeros((n, 3), dtype=np.float64)
    for i in range(i0, n):
        dt = float(clean[i].pose.timestamp) - t0
        mag = rate_m_per_s * max(dt, 0.0)
        offsets[i, :2] = mag * headings[i]
    return offsets, t0, i0


def run_consistency_common_mode(
    clean: Trajectory,
    offsets: np.ndarray,
    *,
    sigma: float,
    lag_steps: int = 10,
) -> tuple[list[IntegrityReport], np.ndarray, np.ndarray]:
    """Apply the *same* epoch offset to primary, prop, and lag.

    Prop/lag geometry comes from the clean trajectory; then ``offsets[i]`` is
    added to every source at epoch ``i``. Separations stay near the clean
    baseline — the independence-breaking case CP-9 warns about.
    """
    clean_pos = np.array([e.pose.position for e in clean], dtype=np.float64)
    ref = SolutionSeparationDetector(p_fa=0.05, pl_confidence=0.999)
    diag = np.diag([sigma**2, sigma**2, sigma**2, 0.05**2, 0.05**2, 0.05**2])
    diag_prop = np.diag(
        [(2 * sigma) ** 2, (2 * sigma) ** 2, (2 * sigma) ** 2, 0.1**2, 0.1**2, 0.1**2]
    )
    reports: list[IntegrityReport] = []
    max_stats = np.empty(len(clean), dtype=np.float64)
    hpls = np.empty(len(clean), dtype=np.float64)
    n = len(clean)
    for i, cur in enumerate(clean):
        off = offsets[i]
        primary = clean_pos[i] + off
        if i == 0:
            prop = primary.copy()
        else:
            vel = clean_pos[i - 1] - clean_pos[i - 2] if i >= 2 else np.zeros(3)
            # Clean constant-velocity prop, then the *same* common-mode offset.
            prop = clean_pos[i - 1] + vel + off
        lag = clean_pos[max(0, i - lag_steps)] + off

        def _pe(name: str, pos: np.ndarray, cov: np.ndarray) -> PoseEstimate:
            return PoseEstimate(
                pose=Pose(
                    timestamp=cur.pose.timestamp,
                    position=pos,
                    orientation=cur.pose.orientation,
                    frame=cur.pose.frame,
                ),
                covariance=cov.copy(),
                source=name,
            )

        sources = {
            "primary": _pe("primary", primary, diag),
            "prop": _pe("prop", prop, diag_prop),
            "lag": _pe("lag", lag, diag_prop),
        }
        result = ref.detect(sources)
        max_stats[i] = max(t.test_statistic for t in result.tests)
        hpls[i] = result.protection_level_horizontal
        alarmed = result.faulted_source is not None
        reports.append(
            IntegrityReport(
                timestamp=float(cur.pose.timestamp),
                protection_level_horizontal=float(result.protection_level_horizontal),
                protection_level_vertical=float(result.protection_level_vertical),
                state=(
                    IntegrityState.DO_NOT_TRUST if alarmed else IntegrityState.NOMINAL
                ),
                reason=(
                    f"ss:faulted={result.faulted_source}"
                    if alarmed
                    else "ss:consistent"
                ),
                time_since_corroboration=0.0,
            )
        )
        if (i + 1) % max(1, n // 5) == 0:
            _log(f"  common-mode SS {i + 1}/{n}")
    return reports, max_stats, hpls


def run_consistency_single_source_drift(
    clean: Trajectory,
    offsets: np.ndarray,
    *,
    sigma: float,
    lag_steps: int = 10,
) -> list[IntegrityReport]:
    """Control: only the primary pose carries the drift (same as multiseq drift)."""
    import importlib.util

    drifted = Trajectory(
        [
            PoseEstimate(
                pose=Pose(
                    timestamp=pe.pose.timestamp,
                    position=pe.pose.position + offsets[i],
                    orientation=pe.pose.orientation,
                    frame=pe.pose.frame,
                ),
                covariance=pe.covariance.copy(),
                source=f"{pe.source}+drift",
            )
            for i, pe in enumerate(clean)
        ]
    )
    path = _repo_root() / "scripts" / "drift_timing.py"
    spec = importlib.util.spec_from_file_location("cairn_drift_timing_cm", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._consistency_reports(drifted, sigma=sigma, lag_steps=lag_steps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=_repo_root() / "data" / "processed" / "anymal_1_1_compslam",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_repo_root() / "outputs" / "common_mode",
    )
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--rate", type=float, default=0.02)
    parser.add_argument("--start-frac", type=float, default=0.30)
    parser.add_argument(
        "--align", choices=("none", "origin", "umeyama"), default="umeyama"
    )
    args = parser.parse_args(argv)

    source = args.source.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    CairnConfig.load_default().to_yaml(out_dir / "config.yaml")
    sigma = float(args.sigma) if args.sigma is not None else _load_sigma_star()

    clean = load_tum(source / "est.tum")
    gt = load_tum(source / "gt.tum")
    offsets, t_onset, onset_idx = _common_mode_offset(
        clean, rate_m_per_s=args.rate, start_frac=args.start_frac
    )
    drift_mag = np.linalg.norm(offsets[:, :2], axis=1)
    t_rel = np.array([e.pose.timestamp for e in clean], dtype=np.float64)
    t_rel = t_rel - t_rel[0]
    t_onset_rel = float(t_onset - float(clean[0].pose.timestamp))

    _log("consistency under COMMON-MODE drift (all sources)…")
    cm_reports, max_stats, hpls = run_consistency_common_mode(
        clean, offsets, sigma=sigma
    )

    # Drifted trajectory for ATE / true error (common-mode on the published est).
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
            for i, pe in enumerate(clean)
        ]
    )
    est_c = with_claimed_diagonal_covariance(
        drifted_est, std_xy=sigma, std_z=sigma, source="cm"
    )
    errors = true_errors_at_reports(
        est_c, gt, cm_reports, align_mode=args.align, max_dt=0.05
    )
    summary = integrity_report_summary(cm_reports, errors, alert_limit=2.0)
    ate = float(ate_rmse(drifted_est, gt, args.align))  # type: ignore[arg-type]

    alarmed = np.array(
        [r.state != IntegrityState.NOMINAL for r in cm_reports], dtype=bool
    )
    post = np.arange(len(clean)) >= onset_idx
    alarm_frac_post = float(np.mean(alarmed[post])) if np.any(post) else float("nan")

    # First alarm after onset (expect: never / rare).
    first_alarm_dt = None
    err_at_alarm = None
    for i in range(onset_idx, len(cm_reports)):
        if cm_reports[i].state != IntegrityState.NOMINAL:
            first_alarm_dt = float(cm_reports[i].timestamp) - float(t_onset)
            err_at_alarm = float(errors[i])
            break

    _log("control: single-source drift (primary only)…")
    ss_reports = run_consistency_single_source_drift(clean, offsets, sigma=sigma)
    ss_alarmed = np.array(
        [r.state != IntegrityState.NOMINAL for r in ss_reports], dtype=bool
    )
    ss_summary = integrity_report_summary(ss_reports, errors, alert_limit=2.0)
    ss_first = None
    ss_err = None
    for i in range(onset_idx, len(ss_reports)):
        if ss_reports[i].state != IntegrityState.NOMINAL:
            ss_first = float(ss_reports[i].timestamp) - float(t_onset)
            ss_err = float(errors[i])
            break

    payload = {
        "synthetic_fault": True,
        "disclaimer": (
            "Synthetic common-mode drift — not field-observed. "
            "Documents the independence assumption failure mode."
        ),
        "sequence": source.name,
        "rate_m_per_s": args.rate,
        "start_frac": args.start_frac,
        "ate_rmse_m": ate,
        "common_mode": {
            "missed_detection_rate": summary.missed_detection_rate,
            "false_alarm_rate": summary.false_alarm_rate,
            "availability": summary.availability,
            "alarm_fraction_after_onset": alarm_frac_post,
            "time_to_first_alarm_s": first_alarm_dt,
            "true_error_at_first_alarm_m": err_at_alarm,
            "mean_pl_m": float(np.mean(hpls)),
            "p95_pl_m": float(np.percentile(hpls, 95)),
            "stat_p50": float(np.median(max_stats)),
            "stat_p95": float(np.percentile(max_stats, 95)),
            "max_true_error_m": float(np.max(errors)),
        },
        "single_source_control": {
            "missed_detection_rate": ss_summary.missed_detection_rate,
            "false_alarm_rate": ss_summary.false_alarm_rate,
            "availability": ss_summary.availability,
            "alarm_fraction_after_onset": float(np.mean(ss_alarmed[post])),
            "time_to_first_alarm_s": ss_first,
            "true_error_at_first_alarm_m": ss_err,
        },
        "limitation": (
            "Solution separation does not detect common-mode faults: when every "
            "source carries the same bias, separations stay small and the monitor "
            "reports NOMINAL while true error grows. Source diversity is a "
            "requirement, not a nice-to-have."
        ),
    }
    (out_dir / "common_mode.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    axes[0].plot(t_rel, drift_mag, color="#1f4e79", lw=2, label="Injected common-mode |drift|")
    axes[0].plot(t_rel, errors, color="#c45c26", lw=1.3, label="True horizontal error")
    axes[0].axvline(t_onset_rel, color="#333", ls="--", lw=1)
    axes[0].set_ylabel("metres")
    axes[0].set_title(
        "Common-mode fault (synthetic) — consistency should miss this",
        loc="left",
        fontsize=11,
    )
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t_rel, max_stats, color="#2a6f97", lw=1.2, label="max SS statistic")
    axes[1].axvline(t_onset_rel, color="#333", ls="--", lw=1)
    axes[1].set_ylabel("χ² statistic")
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].grid(True, alpha=0.25)

    cm_state = np.array(
        [0.0 if r.state == IntegrityState.NOMINAL else 1.0 for r in cm_reports]
    )
    ss_state = np.array(
        [0.0 if r.state == IntegrityState.NOMINAL else 1.0 for r in ss_reports]
    )
    axes[2].step(t_rel, cm_state, where="post", color="#a4161a", lw=1.4, label="common-mode")
    axes[2].step(
        t_rel, ss_state, where="post", color="#2a6f97", lw=1.2, alpha=0.85, label="single-source (control)"
    )
    axes[2].axvline(t_onset_rel, color="#333", ls="--", lw=1)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["NOMINAL", "ALARM"])
    axes[2].set_xlabel("time since start (s)")
    axes[2].legend(fontsize=8, frameon=False)
    axes[2].grid(True, alpha=0.25)
    fig.text(
        0.01,
        0.01,
        "SYNTHETIC FAULT — documents independence failure; not field evidence.",
        fontsize=8,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_dir / "common_mode.png", dpi=160)
    plt.close(fig)

    cm = payload["common_mode"]
    ss = payload["single_source_control"]
    assert isinstance(cm, dict) and isinstance(ss, dict)

    lines = [
        "# Common-mode fault (synthetic limitation)",
        "",
        "**Synthetic correlated drift on all SS sources — not field data.**",
        "",
        str(payload["limitation"]),
        "",
        "## Results",
        "",
        "| Case | MDR | availability | alarm frac after onset | max true error |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| common-mode (all sources) | {100*float(cm['missed_detection_rate']):.1f}% | "
            f"{100*float(cm['availability']):.1f}% | "
            f"{100*float(cm['alarm_fraction_after_onset']):.1f}% | "
            f"{float(cm['max_true_error_m']):.2f} m |"
        ),
        (
            f"| single-source control | {100*float(ss['missed_detection_rate']):.1f}% | "
            f"{100*float(ss['availability']):.1f}% | "
            f"{100*float(ss['alarm_fraction_after_onset']):.1f}% | "
            f"(same injected magnitude) |"
        ),
        "",
        "Raw time-to-first-alarm is not used here: both cases can trip a chance FAR "
        "at onset. The discriminating signal is post-onset alarm fraction "
        f"({100*float(cm['alarm_fraction_after_onset']):.0f}% common-mode vs "
        f"{100*float(ss['alarm_fraction_after_onset']):.0f}% single-source) and MDR "
        f"({100*float(cm['missed_detection_rate']):.0f}% vs "
        f"{100*float(ss['missed_detection_rate']):.0f}%).",
        "",
        "## Slide wording",
        "",
        "> Solution separation catches *disagreement between sources*. "
        "It cannot catch a fault that corrupts every source the same way. "
        "That is a named limitation (CP-9), not an accidental omission — "
        "and why source diversity and geometry-aware monitors still matter.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    _log("")
    _log(payload["limitation"])
    _log(
        f"common-mode: MDR={100*float(cm['missed_detection_rate']):.1f}%  "
        f"availability={100*float(cm['availability']):.1f}%  "
        f"first alarm={cm['time_to_first_alarm_s']}"
    )
    _log(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
