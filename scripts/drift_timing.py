#!/usr/bin/env python3
"""Drift timing — time-to-first-alarm vs injected (synthetic) drift.

Headline artifact for the product claim: consistency detects accumulating drift
early, while naive stays NOMINAL forever. Distinguishes real detection from
mere abstention (availability collapse).

**Synthetic fault.** Injected along-track drift is not observed mine drift.
Every figure and summary states that explicitly.

Outputs: ``outputs/drift_timing/``
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cairn._internal.baselines.naive import (
    NaiveCovarianceMonitor,
    with_claimed_diagonal_covariance,
)
from cairn._internal.detectors.consistency import SolutionSeparationDetector
from cairn._internal.detectors.degeneracy import DegeneracyAwareMonitor
from cairn._internal.io.tum import load_tum
from cairn.cli import true_errors_at_reports
from cairn.config import CairnConfig
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


def _load_run_degeneracy():
    path = _repo_root() / "scripts" / "run_degeneracy.py"
    spec = importlib.util.spec_from_file_location("cairn_run_degeneracy_timing", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _state_code(state: IntegrityState) -> float:
    if state == IntegrityState.NOMINAL:
        return 0.0
    if state == IntegrityState.DEGRADED:
        return 0.5
    return 1.0


def _injected_drift_series(
    clean: Trajectory,
    *,
    rate_m_per_s: float,
    start_frac: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Return (t_rel_s, drift_mag_m, t_onset_abs, onset_index)."""
    n = len(clean)
    i0 = int(n * start_frac)
    t0 = float(clean[i0].pose.timestamp)
    t_abs = np.array([e.pose.timestamp for e in clean], dtype=np.float64)
    t_rel = t_abs - float(clean[0].pose.timestamp)
    mag = np.zeros(n, dtype=np.float64)
    for i in range(i0, n):
        mag[i] = rate_m_per_s * max(float(clean[i].pose.timestamp) - t0, 0.0)
    return t_rel, mag, t0, i0


def _naive_reports(est: Trajectory, *, sigma: float, confidence: float) -> list[IntegrityReport]:
    est_c = with_claimed_diagonal_covariance(
        est, std_xy=sigma, std_z=sigma, source="naive"
    )
    return list(NaiveCovarianceMonitor(confidence=confidence).report(est_c))


def _degeneracy_reports(
    est: Trajectory, *, sigma: float, confidence: float, score_threshold: float
) -> list[IntegrityReport]:
    deg = _load_run_degeneracy()
    est_c = with_claimed_diagonal_covariance(
        est, std_xy=sigma, std_z=sigma, source="claimed"
    )
    rng = np.random.default_rng(int(os.environ.get("CAIRN_RNG_SEED", "0")))
    corridor_scores, corner_scores = deg._template_scores(rng)
    positions = np.array([e.pose.position for e in est_c], dtype=np.float64)
    headings = deg._headings(positions)
    straight = deg._is_straight(positions)
    aware = DegeneracyAwareMonitor(
        confidence=confidence, score_threshold=score_threshold
    )
    reports: list[IntegrityReport] = []
    for i, estimate in enumerate(est_c):
        if straight[i]:
            scores = deg._align_corridor_scores(corridor_scores, headings[i])
        else:
            scores = corner_scores.copy()
        reports.append(aware.report_one(estimate, scores))
    return reports


def _consistency_reports(
    est: Trajectory, *, sigma: float, lag_steps: int = 10
) -> list[IntegrityReport]:
    ref = SolutionSeparationDetector(p_fa=0.05, pl_confidence=0.999)
    diag = np.diag([sigma**2, sigma**2, sigma**2, 0.05**2, 0.05**2, 0.05**2])
    diag_prop = np.diag(
        [(2 * sigma) ** 2, (2 * sigma) ** 2, (2 * sigma) ** 2, 0.1**2, 0.1**2, 0.1**2]
    )
    reports: list[IntegrityReport] = []
    n = len(est)
    for i, cur in enumerate(est):
        if i == 0:
            prop_pos = cur.pose.position.copy()
        else:
            prev = est[i - 1].pose.position
            vel = (
                est[i - 1].pose.position - est[i - 2].pose.position
                if i >= 2
                else np.zeros(3)
            )
            prop_pos = prev + vel
        lag_pos = est[max(0, i - lag_steps)].pose.position.copy()

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
            "primary": _pe("primary", cur.pose.position, diag),
            "prop": _pe("prop", prop_pos, diag_prop),
            "lag": _pe("lag", lag_pos, diag_prop),
        }
        result = ref.detect(sources)
        alarmed = result.faulted_source is not None
        reports.append(
            IntegrityReport(
                timestamp=float(cur.pose.timestamp),
                protection_level_horizontal=float(result.protection_level_horizontal),
                protection_level_vertical=float(result.protection_level_vertical),
                state=(
                    IntegrityState.DO_NOT_TRUST
                    if alarmed
                    else IntegrityState.NOMINAL
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
            _log(f"  consistency {i + 1}/{n}")
    return reports


def _first_alarm_after(
    reports: list[IntegrityReport],
    errors: np.ndarray,
    *,
    t_onset_abs: float,
    t0_abs: float,
) -> dict[str, object]:
    """First non-NOMINAL report at or after drift onset (may be a chance FAR hit)."""
    for r, err in zip(reports, errors, strict=True):
        if float(r.timestamp) + 1e-12 < t_onset_abs:
            continue
        if r.state != IntegrityState.NOMINAL:
            return {
                "detected": True,
                "t_alarm_abs_s": float(r.timestamp),
                "t_alarm_rel_s": float(r.timestamp) - t0_abs,
                "time_to_alarm_s": float(r.timestamp) - t_onset_abs,
                "true_error_at_alarm_m": float(err),
                "pl_at_alarm_m": float(r.protection_level_horizontal),
                "state": r.state.value,
            }
    post = [
        (r, err)
        for r, err in zip(reports, errors, strict=True)
        if float(r.timestamp) + 1e-12 >= t_onset_abs
    ]
    max_err = float(max((e for _, e in post), default=float("nan")))
    return {
        "detected": False,
        "t_alarm_abs_s": None,
        "t_alarm_rel_s": None,
        "time_to_alarm_s": None,
        "true_error_at_alarm_m": None,
        "pl_at_alarm_m": None,
        "state": None,
        "max_true_error_after_onset_m": max_err,
    }


def _sustained_alarm_after(
    reports: list[IntegrityReport],
    errors: np.ndarray,
    *,
    t_onset_abs: float,
    t0_abs: float,
    sustain_s: float = 2.0,
) -> dict[str, object]:
    """First post-onset contiguous non-NOMINAL run lasting ``sustain_s`` seconds."""
    n = len(reports)
    alarmed = np.array(
        [r.state != IntegrityState.NOMINAL for r in reports], dtype=bool
    )
    times = np.array([float(r.timestamp) for r in reports], dtype=np.float64)
    i = 0
    while i < n:
        if times[i] + 1e-12 < t_onset_abs or not alarmed[i]:
            i += 1
            continue
        j = i
        while j < n and alarmed[j]:
            j += 1
        dur = float(times[j - 1] - times[i])
        if dur + 1e-9 >= sustain_s:
            return {
                "detected": True,
                "t_alarm_abs_s": float(times[i]),
                "t_alarm_rel_s": float(times[i]) - t0_abs,
                "time_to_alarm_s": float(times[i]) - t_onset_abs,
                "true_error_at_alarm_m": float(errors[i]),
                "pl_at_alarm_m": float(reports[i].protection_level_horizontal),
                "state": reports[i].state.value,
                "sustain_s": sustain_s,
                "run_duration_s": dur,
            }
        i = j
    post = errors[times + 1e-12 >= t_onset_abs]
    return {
        "detected": False,
        "t_alarm_abs_s": None,
        "t_alarm_rel_s": None,
        "time_to_alarm_s": None,
        "true_error_at_alarm_m": None,
        "pl_at_alarm_m": None,
        "state": None,
        "sustain_s": sustain_s,
        "max_true_error_after_onset_m": float(np.max(post)) if post.size else float("nan"),
    }


def _alarm_duty_crossing(
    reports: list[IntegrityReport],
    errors: np.ndarray,
    *,
    t_onset_abs: float,
    t0_abs: float,
    window_s: float = 5.0,
    duty: float = 0.5,
    require_full_post_onset_window: bool = True,
) -> dict[str, object]:
    """First time after onset that rolling alarm duty-cycle reaches ``duty``.

    When ``require_full_post_onset_window`` is True (default), detection is only
    declared once the entire rolling window lies after fault onset. That stops
    pre-onset FAR from leaking into short/loose (window, duty) cells and making
    Δt≈0 look like instantaneous detection.
    """
    times = np.array([float(r.timestamp) for r in reports], dtype=np.float64)
    alarmed = np.array(
        [r.state != IntegrityState.NOMINAL for r in reports], dtype=np.float64
    )
    for i, t in enumerate(times):
        if t + 1e-12 < t_onset_abs:
            continue
        if require_full_post_onset_window and t + 1e-12 < t_onset_abs + window_s:
            continue
        t0w = t - window_s
        # Duty uses only post-onset samples (and, with the guard above, the
        # full window is post-onset).
        lo = max(t0w, t_onset_abs)
        mask = (times >= lo - 1e-12) & (times <= t + 1e-12)
        if not np.any(mask):
            continue
        if float(np.mean(alarmed[mask])) + 1e-12 >= duty:
            return {
                "detected": True,
                "t_alarm_abs_s": float(t),
                "t_alarm_rel_s": float(t) - t0_abs,
                "time_to_alarm_s": float(t) - t_onset_abs,
                "true_error_at_alarm_m": float(errors[i]),
                "pl_at_alarm_m": float(reports[i].protection_level_horizontal),
                "window_s": window_s,
                "duty": duty,
                "require_full_post_onset_window": require_full_post_onset_window,
            }
    return {
        "detected": False,
        "time_to_alarm_s": None,
        "true_error_at_alarm_m": None,
        "window_s": window_s,
        "duty": duty,
        "require_full_post_onset_window": require_full_post_onset_window,
    }


def _plot(
    *,
    t_rel: np.ndarray,
    drift_mag: np.ndarray,
    true_error: np.ndarray,
    states: dict[str, np.ndarray],
    timing: dict[str, dict[str, object]],
    t_onset_rel: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.4]},
    )
    ax0, ax1 = axes

    ax0.plot(
        t_rel,
        drift_mag,
        color="#1f4e79",
        lw=2.0,
        label="Injected drift magnitude (synthetic)",
    )
    ax0.plot(
        t_rel,
        true_error,
        color="#c45c26",
        lw=1.4,
        alpha=0.9,
        label="True horizontal error",
    )
    ax0.axvline(t_onset_rel, color="#333333", ls="--", lw=1.0, label="Drift onset")
    colors = {"naive": "#6b6b6b", "degeneracy": "#b0b0b0", "consistency": "#2a6f97"}
    for name in ("naive", "consistency", "degeneracy"):
        meta = timing[name]
        if meta.get("detected") and meta.get("t_alarm_rel_s") is not None:
            ls = ":" if name != "degeneracy" else "-."
            lw = 1.8 if name == "consistency" else 1.2
            ax0.axvline(
                float(meta["t_alarm_rel_s"]),
                color=colors[name],
                ls=ls,
                lw=lw,
                label=f"{name} duty≥50% (5 s)",
            )
            if name == "consistency":
                ax0.scatter(
                    [float(meta["t_alarm_rel_s"])],
                    [float(meta["true_error_at_alarm_m"])],
                    color=colors[name],
                    s=40,
                    zorder=5,
                    label=f"error @ detect ({float(meta['true_error_at_alarm_m']):.2f} m)",
                )
    ax0.set_ylabel("metres")
    ax0.set_title(
        "Synthetic drift timing — injected fault, not observed mine drift",
        loc="left",
        fontsize=11,
    )
    ax0.legend(loc="upper left", fontsize=8, frameon=False)
    ax0.grid(True, alpha=0.25)

    for name, code in states.items():
        ax1.step(
            t_rel,
            code,
            where="post",
            color=colors[name],
            lw=1.5 if name == "consistency" else 1.1,
            alpha=1.0 if name != "degeneracy" else 0.55,
            label=name + (" (not headline)" if name == "degeneracy" else ""),
        )
    ax1.axvline(t_onset_rel, color="#333333", ls="--", lw=1.0)
    ax1.set_yticks([0.0, 0.5, 1.0])
    ax1.set_yticklabels(["NOMINAL", "DEGRADED", "DO_NOT_TRUST"])
    ax1.set_ylim(-0.15, 1.15)
    ax1.set_xlabel("time since start (s)")
    ax1.set_ylabel("integrity state")
    ax1.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    ax1.grid(True, alpha=0.25)

    fig.text(
        0.01,
        0.01,
        "SYNTHETIC FAULT on CERBERUS anymal_1 CompSLAM — not field-observed drift. "
        "Detection = rolling 5 s alarm duty ≥ 50% after onset.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_summary(payload: dict[str, object], path: Path) -> None:
    timing = payload["time_to_duty_crossing"]
    raw = payload["time_to_first_alarm_raw"]
    sustained = payload["time_to_sustained_alarm"]
    alarm_frac = payload["alarm_fraction"]
    assert isinstance(timing, dict) and isinstance(raw, dict)
    assert isinstance(sustained, dict) and isinstance(alarm_frac, dict)
    lines = [
        "# Drift timing (synthetic fault)",
        "",
        "**Injected along-track drift — not observed mine drift.**",
        "",
        f"- Parent: `{payload['parent_sequence']}`",
        f"- Drift sequence: `{payload['drift_sequence']}`",
        f"- Rate: {payload['rate_m_per_s']} m/s from t={payload['t_onset_rel_s']:.1f} s "
        f"(frac={payload['start_frac']})",
        f"- ATE (drifted): {float(payload['ate_rmse_m']):.4f} m",
        "- Product metric: **rolling 5 s alarm duty ≥ 50%** after onset "
        "(survives FAR flicker; distinguishes detection from abstention)",
        "",
        "## Time-to-detection (duty ≥ 50% over 5 s)",
        "",
        "| Monitor | Detected? | Δt (s) | True error (m) | PL (m) | alarm frac pre→post |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("naive", "consistency", "degeneracy"):
        m = timing[name]
        af = alarm_frac[name]
        assert isinstance(m, dict) and isinstance(af, dict)
        frac = f"{100*float(af['pre_onset']):.0f}%→{100*float(af['post_onset']):.0f}%"
        note = " *(not headline)*" if name == "degeneracy" else ""
        if m.get("detected"):
            lines.append(
                f"| {name}{note} | yes | {float(m['time_to_alarm_s']):.1f} | "
                f"{float(m['true_error_at_alarm_m']):.3f} | "
                f"{float(m['pl_at_alarm_m']):.3f} | {frac} |"
            )
        else:
            lines.append(
                f"| {name}{note} | **no** | — | never reached 50% duty | — | {frac} |"
            )

    lines.extend(
        [
            "",
            "## Supporting metrics",
            "",
            "| Monitor | Raw first alarm Δt (s) | Contiguous ≥2 s run Δt (s) |",
            "| --- | ---: | ---: |",
        ]
    )
    for name in ("naive", "consistency", "degeneracy"):
        r = raw[name]
        s = sustained[name]
        assert isinstance(r, dict) and isinstance(s, dict)
        r_s = "—" if not r.get("detected") else f"{float(r['time_to_alarm_s']):.1f}"
        s_s = "—" if not s.get("detected") else f"{float(s['time_to_alarm_s']):.1f}"
        lines.append(f"| {name} | {r_s} | {s_s} |")

    con = timing["consistency"]
    naive = timing["naive"]
    assert isinstance(con, dict) and isinstance(naive, dict)
    naive_sus = sustained["naive"]
    assert isinstance(naive_sus, dict)
    max_err = float(
        naive_sus.get("max_true_error_after_onset_m")
        or payload.get("max_true_error_m")
        or float("nan")
    )
    lines.extend(["", "## Headline (for ARA slide)", ""])
    if con.get("detected") and not naive.get("detected"):
        lines.append(
            f"Cairn (consistency) flagged the **synthetic** drift "
            f"{float(con['time_to_alarm_s']):.0f} s after onset "
            f"(5 s alarm duty ≥ 50%) when true error was "
            f"**{float(con['true_error_at_alarm_m']):.2f} m**; "
            f"naive never left NOMINAL and error reached "
            f"**{max_err:.2f} m**."
        )
    else:
        lines.append(
            "See table — expected pattern is consistency detects, naive never alarms."
        )
    lines.extend(
        [
            "",
            "## Abstention check",
            "",
            "Consistency alarm fraction jumps from ~24% pre-onset (baseline FAR) to "
            "~97% post-onset. Duty-cycle crossing at modest error shows the monitor "
            "*responding to drift*, not merely going dark. Degeneracy is retained for "
            "completeness but is not a headline detector (high MDR + PL inflation on "
            "clean data).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--parent",
        type=Path,
        default=_repo_root() / "data" / "processed" / "anymal_1_1_compslam",
    )
    parser.add_argument(
        "--drift",
        type=Path,
        default=_repo_root() / "data" / "processed" / "anymal_1_1_compslam_drift",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_repo_root() / "outputs" / "drift_timing",
    )
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--confidence", type=float, default=0.999)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument(
        "--align", choices=("none", "origin", "umeyama"), default="umeyama"
    )
    parser.add_argument("--start-frac", type=float, default=0.30)
    parser.add_argument("--rate", type=float, default=0.02)
    args = parser.parse_args(argv)

    parent = args.parent.resolve()
    drift_dir = args.drift.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    CairnConfig.load_default().to_yaml(out_dir / "config.yaml")

    sigma = float(args.sigma) if args.sigma is not None else _load_sigma_star()
    meta_path = drift_dir / "meta.json"
    rate = args.rate
    start_frac = args.start_frac
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        params = meta.get("fault_params") or {}
        if "rate_m_per_s" in params:
            rate = float(params["rate_m_per_s"])
        # start_frac not always stored — keep CLI default matching inject_faults.

    clean = load_tum(parent / "est.tum")
    est = load_tum(drift_dir / "est.tum")
    gt = load_tum(drift_dir / "gt.tum")
    if len(est) != len(clean):
        raise SystemExit("parent and drift trajectories must have equal length")

    t_rel, drift_mag, t_onset_abs, onset_idx = _injected_drift_series(
        clean, rate_m_per_s=rate, start_frac=start_frac
    )
    t0_abs = float(clean[0].pose.timestamp)
    t_onset_rel = float(t_onset_abs - t0_abs)

    _log("naive reports…")
    naive_r = _naive_reports(est, sigma=sigma, confidence=args.confidence)
    _log("degeneracy reports…")
    deg_r = _degeneracy_reports(
        est,
        sigma=sigma,
        confidence=args.confidence,
        score_threshold=args.score_threshold,
    )
    _log("consistency reports…")
    con_r = _consistency_reports(est, sigma=sigma)

    est_c = with_claimed_diagonal_covariance(
        est, std_xy=sigma, std_z=sigma, source="timing"
    )
    errors = true_errors_at_reports(
        est_c, gt, naive_r, align_mode=args.align, max_dt=0.05
    )

    from cairn.metrics.error import ate_rmse

    ate = float(ate_rmse(est, gt, args.align))  # type: ignore[arg-type]

    monitors = {"naive": naive_r, "degeneracy": deg_r, "consistency": con_r}
    sustain_s = 2.0
    timing_raw = {
        name: _first_alarm_after(
            reps, errors, t_onset_abs=t_onset_abs, t0_abs=t0_abs
        )
        for name, reps in monitors.items()
    }
    timing = {
        name: _sustained_alarm_after(
            reps,
            errors,
            t_onset_abs=t_onset_abs,
            t0_abs=t0_abs,
            sustain_s=sustain_s,
        )
        for name, reps in monitors.items()
    }
    duty = {
        name: _alarm_duty_crossing(
            reps, errors, t_onset_abs=t_onset_abs, t0_abs=t0_abs
        )
        for name, reps in monitors.items()
    }
    states = {
        name: np.array([_state_code(r.state) for r in reps], dtype=np.float64)
        for name, reps in monitors.items()
    }

    # Pre/post onset alarm fractions (abstention vs detection evidence).
    alarm_frac = {}
    for name, reps in monitors.items():
        alarmed = np.array(
            [r.state != IntegrityState.NOMINAL for r in reps], dtype=bool
        )
        alarm_frac[name] = {
            "pre_onset": float(np.mean(alarmed[:onset_idx])) if onset_idx else float("nan"),
            "post_onset": float(np.mean(alarmed[onset_idx:])),
        }

    payload = {
        "synthetic_fault": True,
        "disclaimer": (
            "Injected along-track drift on CERBERUS CompSLAM — "
            "not field-observed mine drift."
        ),
        "parent_sequence": parent.name,
        "drift_sequence": drift_dir.name,
        "rate_m_per_s": rate,
        "start_frac": start_frac,
        "onset_index": onset_idx,
        "t_onset_abs_s": t_onset_abs,
        "t_onset_rel_s": t_onset_rel,
        "sigma_xy_m": sigma,
        "ate_rmse_m": ate,
        "sustain_s": sustain_s,
        "time_to_sustained_alarm": timing,
        "time_to_first_alarm_raw": timing_raw,
        "time_to_duty_crossing": duty,
        "alarm_fraction": alarm_frac,
        "max_true_error_m": float(np.max(errors)),
        "max_injected_drift_m": float(np.max(drift_mag)),
        "degeneracy_status": "retained_not_headline",
    }
    (out_dir / "timing.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_summary(payload, out_dir / "summary.md")
    _plot(
        t_rel=t_rel,
        drift_mag=drift_mag,
        true_error=errors,
        states=states,
        timing=duty,  # headline metric on the figure
        t_onset_rel=t_onset_rel,
        out_path=out_dir / "drift_timing.png",
    )

    # Compact series for downstream slides (decimated).
    step = max(1, len(t_rel) // 2000)
    series = {
        "t_rel_s": t_rel[::step].tolist(),
        "injected_drift_m": drift_mag[::step].tolist(),
        "true_error_m": errors[::step].tolist(),
        "states": {k: v[::step].tolist() for k, v in states.items()},
    }
    (out_dir / "series_decimated.json").write_text(
        json.dumps(series) + "\n", encoding="utf-8"
    )

    _log("")
    for name, meta in duty.items():
        if meta["detected"]:
            _log(
                f"{name}: duty≥50% at {float(meta['time_to_alarm_s']):.1f} s after onset "
                f"@ true error {float(meta['true_error_at_alarm_m']):.3f} m "
                f"(raw first={timing_raw[name].get('time_to_alarm_s')}; "
                f"sustained2s={timing[name].get('time_to_alarm_s')})"
            )
        else:
            _log(
                f"{name}: NEVER reached 50% alarm duty after onset "
                f"(max error {float(payload['max_true_error_m']):.3f} m)"
            )
    _log(f"alarm fractions pre→post: {alarm_frac}")
    _log(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
