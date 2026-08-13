#!/usr/bin/env python3
"""ROC curves at matched FAR, with hard-epoch (high true-error) split.

Static covariance tuned on full-run GT (σ*) can look safe *on average* while
failing in the high-error tail. This script:

1. Sweeps each monitor's decision threshold → MDR vs FAR (ROC).
2. Reports MDR at fixed FAR ∈ {1%, 5%, 10%}.
3. Splits epochs into **hard** (true error > 90th percentile) vs **easy**,
   and reports MDR on each subset at those matched FAR points.

Naive never alarms (always NOMINAL), so it only occupies FAR = 0. That is not
a bug in the ROC — it is the claim under test.

Outputs: ``outputs/roc/``
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
    protection_levels_from_covariance,
    with_claimed_diagonal_covariance,
)
from cairn._internal.detectors.consistency import SolutionSeparationDetector
from cairn._internal.detectors.degeneracy import inflate_covariance_anisotropic
from cairn._internal.io.tum import load_tum
from cairn.cli import true_errors_at_reports
from cairn.config import CairnConfig
from cairn.metrics.error import ate_rmse
from cairn.types import Pose, PoseEstimate, Trajectory


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_run_degeneracy():
    path = _repo_root() / "scripts" / "run_degeneracy.py"
    spec = importlib.util.spec_from_file_location("cairn_run_degeneracy_roc", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _default_sequence() -> Path:
    root = _repo_root()
    full = root / "data" / "processed" / "anymal_1_1_compslam"
    if (full / "est.tum").is_file():
        return full
    return root / "examples" / "quickstart"


def _load_sigma_star(default: float = 0.0694) -> float:
    path = _repo_root() / "outputs" / "sigma_sweep" / "sigma_sweep.json"
    if not path.is_file():
        return default
    payload = json.loads(path.read_text(encoding="utf-8"))
    crossing = payload.get("crossing") or {}
    sigma = crossing.get("sigma_xy_m")
    return float(sigma) if sigma is not None else default


def _log(msg: str) -> None:
    print(msg, flush=True)


def rates_from_arrays(
    errors: np.ndarray,
    hpl: np.ndarray,
    alarmed: np.ndarray,
    hard: np.ndarray,
    easy: np.ndarray,
) -> dict[str, float]:
    """MDR / FAR / PL tightness from parallel arrays."""
    exceed = errors > hpl
    nominal = ~alarmed
    miss = exceed & nominal
    false_alarm = alarmed & ~exceed
    ratio = hpl / np.maximum(errors, 1e-12)
    return {
        "far": float(np.mean(false_alarm)),
        "mdr": float(np.mean(miss)),
        "mdr_hard": float(np.mean(miss[hard])) if np.any(hard) else float("nan"),
        "mdr_easy": float(np.mean(miss[easy])) if np.any(easy) else float("nan"),
        "availability": float(np.mean(nominal)),
        "mean_pl_m": float(np.mean(hpl)),
        "p95_pl_m": float(np.percentile(hpl, 95)),
        "pl_over_err_p50": float(np.percentile(ratio, 50)),
        "pl_over_err_p95": float(np.percentile(ratio, 95)),
        "mean_pl_hard_m": float(np.mean(hpl[hard])) if np.any(hard) else float("nan"),
        "mean_pl_easy_m": float(np.mean(hpl[easy])) if np.any(easy) else float("nan"),
    }


_PL_KEYS = (
    "mean_pl_m",
    "p95_pl_m",
    "pl_over_err_p50",
    "pl_over_err_p95",
    "mean_pl_hard_m",
    "mean_pl_easy_m",
)


def _interpolate_at_far(
    curve: list[dict[str, float]],
    target_far: float,
    *,
    preferred_threshold: float | None = None,
) -> dict[str, float] | None:
    """Linear interpolate metrics onto a target overall FAR.

    When several thresholds share the same FAR (e.g. naive always FAR=0),
    prefer ``preferred_threshold`` if given; else the lowest overall MDR,
    then the tightest mean PL (cowardice tie-break).
    """
    if not curve:
        return None
    fars = np.array([p["far"] for p in curve], dtype=np.float64)
    near = np.where(np.abs(fars - target_far) <= 0.005)[0]
    if len(near) > 0:
        if preferred_threshold is not None:
            best = min(
                near,
                key=lambda i: abs(curve[i]["threshold"] - preferred_threshold),
            )
        else:
            best = min(
                near,
                key=lambda i: (
                    curve[i]["mdr"],
                    curve[i].get("mean_pl_m", 0.0),
                    abs(fars[i] - target_far),
                ),
            )
        out = dict(curve[best])
        out["far"] = float(fars[best]) if abs(fars[best] - target_far) >= 1e-12 else float(
            target_far
        )
        return out

    order = np.argsort(fars)
    fars_s = fars[order]
    if target_far < fars_s[0] - 1e-12 or target_far > fars_s[-1] + 1e-12:
        return None

    for a, b in zip(order[:-1], order[1:], strict=False):
        f0, f1 = float(fars[a]), float(fars[b])
        if f0 == f1:
            continue
        if (f0 <= target_far <= f1) or (f1 <= target_far <= f0):
            t = (target_far - f0) / (f1 - f0)
            out: dict[str, float] = {"far": target_far}
            keys = (
                "mdr",
                "mdr_hard",
                "mdr_easy",
                "availability",
                "threshold",
                *_PL_KEYS,
            )
            for key in keys:
                if key not in curve[a] or key not in curve[b]:
                    continue
                out[key] = float(curve[a][key] + t * (curve[b][key] - curve[a][key]))
            return out
    return None


def _matched_table(
    name: str,
    curve: list[dict[str, float]],
    targets: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10),
    *,
    preferred_threshold: float | None = None,
) -> list[dict[str, object]]:
    rows = []
    for far in targets:
        hit = _interpolate_at_far(
            curve, far, preferred_threshold=preferred_threshold
        )
        if hit is None:
            rows.append(
                {
                    "monitor": name,
                    "far_target": far,
                    "reachable": False,
                    "mdr": None,
                    "mdr_hard": None,
                    "mdr_easy": None,
                    "mean_pl_m": None,
                    "p95_pl_m": None,
                    "pl_over_err_p50": None,
                    "pl_over_err_p95": None,
                    "note": "FAR not reachable on this ROC",
                }
            )
        else:
            row = {
                "monitor": name,
                "far_target": far,
                "reachable": True,
                "far_actual": hit["far"],
                "threshold": hit.get("threshold"),
                "mdr": hit["mdr"],
                "mdr_hard": hit["mdr_hard"],
                "mdr_easy": hit["mdr_easy"],
                "availability": hit.get("availability"),
            }
            for key in _PL_KEYS:
                row[key] = hit.get(key)
            rows.append(row)
    return rows


def _localizability_series(est: Trajectory) -> np.ndarray:
    deg = _load_run_degeneracy()
    rng = np.random.default_rng(int(os.environ.get("CAIRN_RNG_SEED", "0")))
    corridor_scores, corner_scores = deg._template_scores(rng)
    positions = np.array([e.pose.position for e in est], dtype=np.float64)
    headings = deg._headings(positions)
    straight = deg._is_straight(positions)
    scores = []
    for i in range(len(est)):
        if straight[i]:
            scores.append(deg._align_corridor_scores(corridor_scores, headings[i]))
        else:
            scores.append(corner_scores.copy())
    return np.asarray(scores, dtype=np.float64)


def _naive_curve(
    est: Trajectory,
    errors: np.ndarray,
    hard: np.ndarray,
    easy: np.ndarray,
    *,
    confidence: float,
) -> list[dict[str, float]]:
    """Sweep claimed σ. Always NOMINAL ⇒ FAR ≡ 0."""
    monitor = NaiveCovarianceMonitor(confidence=confidence)
    sigmas = np.unique(
        np.concatenate(
            [np.linspace(0.01, 0.12, 25), np.linspace(0.15, 0.50, 8), np.array([1.0])]
        )
    )
    curve = []
    for sigma in sigmas:
        est_s = with_claimed_diagonal_covariance(
            est, std_xy=float(sigma), std_z=float(sigma), source="naive"
        )
        reports = monitor.report(est_s)
        hpl = np.array([r.protection_level_horizontal for r in reports], dtype=np.float64)
        alarmed = np.zeros(len(reports), dtype=bool)
        point = rates_from_arrays(errors, hpl, alarmed, hard, easy)
        point["threshold"] = float(sigma)
        curve.append(point)
    return curve


def _degeneracy_curve(
    est: Trajectory,
    errors: np.ndarray,
    hard: np.ndarray,
    easy: np.ndarray,
    scores: np.ndarray,
    *,
    sigma_xy: float,
    confidence: float,
) -> list[dict[str, float]]:
    """Sweep localizability alarm threshold (lower → more alarms).

    HPL depends only on inflated cov (score-driven), not on the alarm threshold,
    so it is precomputed once.
    """
    est_s = with_claimed_diagonal_covariance(
        est, std_xy=sigma_xy, std_z=sigma_xy, source="claimed"
    )
    hpl = np.empty(len(est_s), dtype=np.float64)
    min_trans = np.min(scores[:, :3], axis=1)
    for i, pe in enumerate(est_s):
        inflated = inflate_covariance_anisotropic(pe.covariance, scores[i])
        h, _ = protection_levels_from_covariance(inflated, confidence=confidence)
        hpl[i] = h

    thresholds = np.linspace(0.0, 1.0, 41)
    curve = []
    for thr in thresholds:
        # DEGRADED or DO_NOT_TRUST both count as alarms for FAR/MDR.
        alarmed = min_trans < float(thr)
        point = rates_from_arrays(errors, hpl, alarmed, hard, easy)
        point["threshold"] = float(thr)
        curve.append(point)
    return curve


def _consistency_sources_and_stats(
    est: Trajectory,
    *,
    sigma_xy: float,
    lag_steps: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-epoch SS stats using primary / CV-prop / lagged pose sources."""
    ref = SolutionSeparationDetector(p_fa=0.05, pl_confidence=0.999)
    diag = np.diag(
        [sigma_xy**2, sigma_xy**2, sigma_xy**2, 0.05**2, 0.05**2, 0.05**2]
    )
    diag_prop = np.diag(
        [(2 * sigma_xy) ** 2, (2 * sigma_xy) ** 2, (2 * sigma_xy) ** 2, 0.1**2, 0.1**2, 0.1**2]
    )

    max_stats = np.empty(len(est), dtype=np.float64)
    hpls = np.empty(len(est), dtype=np.float64)
    vpls = np.empty(len(est), dtype=np.float64)
    n = len(est)
    report_every = max(1, n // 10)

    for i, cur in enumerate(est):
        if i == 0:
            prop_pos = cur.pose.position.copy()
        else:
            prev = est[i - 1].pose.position
            if i >= 2:
                vel = est[i - 1].pose.position - est[i - 2].pose.position
            else:
                vel = np.zeros(3)
            prop_pos = prev + vel

        lag_i = max(0, i - lag_steps)
        lag_pos = est[lag_i].pose.position.copy()

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
        max_stats[i] = max(t.test_statistic for t in result.tests)
        hpls[i] = result.protection_level_horizontal
        vpls[i] = result.protection_level_vertical
        if (i + 1) % report_every == 0 or i + 1 == n:
            _log(f"  consistency SS {i + 1}/{n}")

    return hpls, vpls, max_stats


def _consistency_curve(
    est: Trajectory,
    errors: np.ndarray,
    hard: np.ndarray,
    easy: np.ndarray,
    *,
    sigma_xy: float,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    hpls, _vpls, stats = _consistency_sources_and_stats(est, sigma_xy=sigma_xy)
    lo = float(np.quantile(stats, 0.05))
    hi = float(np.quantile(stats, 0.995))
    # Fixed grid only — do not append every unique statistic (that is O(n²)).
    thresholds = np.linspace(lo, hi, 41)

    curve = []
    alarm_fractions = []
    for thr in thresholds:
        alarmed = stats > float(thr)
        alarm_fractions.append(float(np.mean(alarmed)))
        point = rates_from_arrays(errors, hpls, alarmed, hard, easy)
        point["threshold"] = float(thr)
        curve.append(point)

    diag = {
        "stat_min": float(np.min(stats)),
        "stat_p50": float(np.median(stats)),
        "stat_p95": float(np.percentile(stats, 95)),
        "stat_max": float(np.max(stats)),
        "threshold_lo": lo,
        "threshold_hi": hi,
        "alarm_frac_min": float(np.min(alarm_fractions)),
        "alarm_frac_max": float(np.max(alarm_fractions)),
        "mean_pl_m": float(np.mean(hpls)),
        "p95_pl_m": float(np.percentile(hpls, 95)),
        "pl_unique": int(np.unique(np.round(hpls, 6)).size),
        "mdr_unique_on_curve": int(np.unique(np.round([p["mdr"] for p in curve], 8)).size),
        "far_unique_on_curve": int(np.unique(np.round([p["far"] for p in curve], 8)).size),
        "threshold_moves_alarms": float(np.max(alarm_fractions) - np.min(alarm_fractions))
        > 1e-6,
    }
    return curve, diag


def _plot_roc(
    curves: dict[str, list[dict[str, float]]],
    out: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)

    ax0 = axes[0]
    colors = {"naive": "#455a64", "degeneracy": "#c62828", "consistency": "#1565c0"}
    for name, curve in curves.items():
        far = np.array([p["far"] for p in curve])
        mdr = np.array([p["mdr"] for p in curve])
        order = np.argsort(far)
        ax0.plot(
            100.0 * far[order],
            100.0 * mdr[order],
            "-o",
            ms=3,
            lw=1.8,
            color=colors.get(name, "k"),
            label=name,
        )
    ax0.set_xlabel("False-alarm rate (%)")
    ax0.set_ylabel("Missed-detection rate (%)")
    ax0.set_title("ROC (all epochs)")
    ax0.grid(True, color="#cfd8dc", lw=0.7)
    ax0.legend(fontsize=9)

    ax1 = axes[1]
    for name, curve in curves.items():
        far = np.array([p["far"] for p in curve])
        mdr_h = np.array([p["mdr_hard"] for p in curve])
        order = np.argsort(far)
        ax1.plot(
            100.0 * far[order],
            100.0 * mdr_h[order],
            "-o",
            ms=3,
            lw=1.8,
            color=colors.get(name, "k"),
            label=name,
        )
    ax1.set_xlabel("False-alarm rate (%) — matched on all epochs")
    ax1.set_ylabel("MDR on hard epochs (%)")
    ax1.set_title("Hard-epoch MDR (error > P90)")
    ax1.grid(True, color="#cfd8dc", lw=0.7)
    ax1.legend(fontsize=9)

    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


def _interpret(
    matched: list[dict[str, object]],
    naive_star: dict[str, float],
    consistency_diag: dict[str, float] | None = None,
) -> str:
    def row(monitor: str, far: float) -> dict[str, object] | None:
        for r in matched:
            if r["monitor"] == monitor and abs(float(r["far_target"]) - far) < 1e-12:
                return r
        return None

    lines = [
        f"Naive at σ* (FAR=0): overall MDR={100*naive_star['mdr']:.2f}%, "
        f"hard MDR={100*naive_star['mdr_hard']:.2f}%, "
        f"easy MDR={100*naive_star['mdr_easy']:.2f}%, "
        f"mean PL={naive_star.get('mean_pl_m', float('nan')):.3f} m, "
        f"PL/err P50={naive_star.get('pl_over_err_p50', float('nan')):.1f}×.",
    ]
    deg5 = row("degeneracy", 0.05)
    con5 = row("consistency", 0.05)
    if deg5 and deg5.get("reachable"):
        mean_pl = deg5.get("mean_pl_m")
        pl_note = (
            f", mean PL={float(mean_pl):.3f} m, "
            f"PL/err P50={float(deg5.get('pl_over_err_p50') or float('nan')):.1f}×"
            if mean_pl is not None
            else ""
        )
        lines.append(
            f"Degeneracy at FAR≈5%: overall MDR={100*float(deg5['mdr']):.2f}%, "
            f"hard MDR={100*float(deg5['mdr_hard']):.2f}% "
            f"(vs naive hard {100*naive_star['mdr_hard']:.2f}%){pl_note}."
        )
        p95_pl = deg5.get("p95_pl_m")
        naive_pl = float(naive_star.get("mean_pl_m") or 0.26)
        if mean_pl is not None and float(mean_pl) > 2.0 * naive_pl:
            lines.append(
                "WARNING: degeneracy mean PL ≫ naive@σ* PL — apparent MDR wins may be "
                "inflation (cowardice), not detection."
            )
        if p95_pl is not None and float(p95_pl) > 4.0 * naive_pl:
            lines.append(
                f"WARNING: degeneracy P95 PL={float(p95_pl):.3f} m vs naive PL="
                f"{naive_pl:.3f} m — tail inflation blankets hard epochs."
            )
    else:
        lines.append("Degeneracy did not reach FAR=5% on this sweep.")
    if con5 and con5.get("reachable"):
        lines.append(
            f"Consistency at FAR≈5%: overall MDR={100*float(con5['mdr']):.2f}%, "
            f"hard MDR={100*float(con5['mdr_hard']):.2f}%, "
            f"mean PL={float(con5.get('mean_pl_m') or float('nan')):.3f} m."
        )
    if consistency_diag is not None:
        if not consistency_diag.get("threshold_moves_alarms"):
            lines.append(
                "RED FLAG: consistency threshold sweep does not change alarm rate — "
                "statistic may be stuck relative to the threshold grid."
            )
        elif int(consistency_diag.get("mdr_unique_on_curve", 0)) <= 1:
            lines.append(
                "RED FLAG: consistency MDR is flat across the ROC despite moving FAR — "
                f"mean PL={float(consistency_diag['mean_pl_m']):.3f} m likely blankets "
                "all true errors (cowardice or inert separations)."
            )
        lines.append(
            "Consistency diagnostics: "
            f"stat P50/P95/max="
            f"{consistency_diag['stat_p50']:.3g}/"
            f"{consistency_diag['stat_p95']:.3g}/"
            f"{consistency_diag['stat_max']:.3g}; "
            f"alarm frac span "
            f"[{100*consistency_diag['alarm_frac_min']:.1f}%, "
            f"{100*consistency_diag['alarm_frac_max']:.1f}%]."
        )

    hard_gap = float(naive_star["mdr_hard"]) - float(naive_star["mdr_easy"])
    if hard_gap > 1e-6 and float(naive_star["mdr_hard"]) < 0.01:
        lines.append(
            "Directional hard/easy split exists for naive@σ* (hard MDR > easy), "
            "but absolute hard MDR is <1% on this run — anymal_1 looks too well-behaved "
            "to stress the thesis; multi-sequence is required."
        )
    elif deg5 and deg5.get("reachable") and float(deg5["mdr_hard"]) + 1e-6 < float(
        naive_star["mdr_hard"]
    ):
        lines.append(
            "Hard-epoch split: degeneracy lowers hard MDR vs naive@σ* at matched FAR — "
            "confirm via PL tightness that this is not inflation."
        )
    elif deg5 and deg5.get("reachable"):
        lines.append(
            "At matched FAR, degeneracy does not beat naive on hard epochs — "
            "detector may not separate, or the sequence may be too easy."
        )
    return " ".join(lines)


def _write_summary(payload: dict[str, object], path: Path) -> None:
    naive_star = payload["naive_at_sigma_star"]
    assert isinstance(naive_star, dict)
    lines = [
        "# ROC + hard-epoch split",
        "",
        f"- Sequence: `{payload['sequence']}`",
        f"- ATE RMSE: **{float(payload['ate_rmse_m']):.4f} m**",
        f"- σ* (from sigma sweep): **{100*float(payload['sigma_star_m']):.2f} cm**",
        (
            f"- Hard epochs: true error > P{payload['hard_percentile']:g} "
            f"= **{float(payload['hard_error_threshold_m']):.4f} m** "
            f"({payload['n_hard']} / {payload['n_epochs']})"
        ),
        "",
        "## Naive at σ* (FAR = 0 — never alarms)",
        "",
        "| subset | MDR | mean PL (m) |",
        "| --- | ---: | ---: |",
        (
            f"| all | {100*float(naive_star['mdr']):.2f}% | "
            f"{float(naive_star.get('mean_pl_m') or float('nan')):.3f} |"
        ),
        (
            f"| hard | {100*float(naive_star['mdr_hard']):.2f}% | "
            f"{float(naive_star.get('mean_pl_hard_m') or float('nan')):.3f} |"
        ),
        (
            f"| easy | {100*float(naive_star['mdr_easy']):.2f}% | "
            f"{float(naive_star.get('mean_pl_easy_m') or float('nan')):.3f} |"
        ),
        (
            f"| PL/err P50 / P95 | — | "
            f"{float(naive_star.get('pl_over_err_p50') or float('nan')):.1f}× / "
            f"{float(naive_star.get('pl_over_err_p95') or float('nan')):.1f}× |"
        ),
        "",
        "## MDR at matched FAR (with PL tightness)",
        "",
        "| Monitor | FAR | Reachable | MDR | MDR hard | mean PL (m) | P95 PL (m) | "
        "PL/err P50 | PL/err P95 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in payload["matched_far"]:  # type: ignore[assignment]
        assert isinstance(r, dict)
        if not r["reachable"]:
            lines.append(
                f"| {r['monitor']} | {100*float(r['far_target']):.0f}% | no | "
                "— | — | — | — | — | — |"
            )
        else:
            lines.append(
                f"| {r['monitor']} | {100*float(r['far_target']):.0f}% | yes | "
                f"{100*float(r['mdr']):.2f}% | "
                f"{100*float(r['mdr_hard']):.2f}% | "
                f"{float(r.get('mean_pl_m') or float('nan')):.3f} | "
                f"{float(r.get('p95_pl_m') or float('nan')):.3f} | "
                f"{float(r.get('pl_over_err_p50') or float('nan')):.1f}× | "
                f"{float(r.get('pl_over_err_p95') or float('nan')):.1f}× |"
            )
    if payload.get("consistency_diagnostics"):
        d = payload["consistency_diagnostics"]
        assert isinstance(d, dict)
        lines.extend(
            [
                "",
                "## Consistency threshold diagnostics",
                "",
                f"- Statistic P50/P95/max: "
                f"{d['stat_p50']:.4g} / {d['stat_p95']:.4g} / {d['stat_max']:.4g}",
                f"- Threshold grid: [{d['threshold_lo']:.4g}, {d['threshold_hi']:.4g}]",
                f"- Alarm fraction span: "
                f"[{100*d['alarm_frac_min']:.2f}%, {100*d['alarm_frac_max']:.2f}%]",
                f"- Unique FAR / MDR points on curve: "
                f"{d['far_unique_on_curve']} / {d['mdr_unique_on_curve']}",
                f"- Mean / P95 PL (fixed across thresholds): "
                f"{d['mean_pl_m']:.3f} m / {d['p95_pl_m']:.3f} m",
                f"- Threshold moves alarms: {d['threshold_moves_alarms']}",
            ]
        )
    lines.extend(["", "## Interpretation", "", str(payload["interpretation"]), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sequence-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--align", choices=("none", "origin", "umeyama"), default="umeyama")
    parser.add_argument("--confidence", type=float, default=0.999)
    parser.add_argument("--hard-percentile", type=float, default=90.0)
    args = parser.parse_args(argv)

    sequence_dir = (args.sequence_dir or _default_sequence()).resolve()
    out_dir = args.out_dir or (_repo_root() / "outputs" / "roc")
    out_dir.mkdir(parents=True, exist_ok=True)
    CairnConfig.load_default().to_yaml(out_dir / "config.yaml")

    sigma_star = _load_sigma_star()
    est = load_tum(sequence_dir / "est.tum")
    gt = load_tum(sequence_dir / "gt.tum")
    ate = ate_rmse(est, gt, args.align)  # type: ignore[arg-type]

    est0 = with_claimed_diagonal_covariance(
        est, std_xy=sigma_star, std_z=sigma_star, source="roc"
    )
    reports0 = NaiveCovarianceMonitor(args.confidence).report(est0)
    errors = true_errors_at_reports(
        est0, gt, reports0, align_mode=args.align, max_dt=0.05
    )
    assert len(errors) == len(est)

    p90 = float(np.percentile(errors, args.hard_percentile))
    hard = errors > p90
    easy = ~hard

    _log(
        f"sequence={sequence_dir.name}  n={len(est)}  ATE={ate:.4f} m  "
        f"σ*={100*sigma_star:.2f} cm  hard_P{args.hard_percentile:g}={p90:.4f} m  "
        f"n_hard={int(hard.sum())} ({100*hard.mean():.1f}%)"
    )

    _log("localizability scores…")
    scores = _localizability_series(est)
    _log("naive σ sweep…")
    naive = _naive_curve(est, errors, hard, easy, confidence=args.confidence)
    _log("degeneracy threshold sweep…")
    degeneracy = _degeneracy_curve(
        est,
        errors,
        hard,
        easy,
        scores,
        sigma_xy=sigma_star,
        confidence=args.confidence,
    )
    _log("consistency SS + threshold sweep…")
    consistency, consistency_diag = _consistency_curve(
        est, errors, hard, easy, sigma_xy=sigma_star
    )
    _log(
        "  consistency diag: "
        f"stat P50={consistency_diag['stat_p50']:.3g} "
        f"P95={consistency_diag['stat_p95']:.3g} max={consistency_diag['stat_max']:.3g}; "
        f"alarm span "
        f"[{100*consistency_diag['alarm_frac_min']:.1f}%, "
        f"{100*consistency_diag['alarm_frac_max']:.1f}%]; "
        f"mean PL={consistency_diag['mean_pl_m']:.3f} m"
    )

    curves = {"naive": naive, "degeneracy": degeneracy, "consistency": consistency}

    naive_at_star = min(naive, key=lambda p: abs(p["threshold"] - sigma_star))
    matched = []
    for name, curve in curves.items():
        preferred = sigma_star if name == "naive" else None
        matched.extend(_matched_table(name, curve, preferred_threshold=preferred))

    payload = {
        "sequence": sequence_dir.name,
        "align_mode": args.align,
        "ate_rmse_m": ate,
        "sigma_star_m": sigma_star,
        "hard_percentile": args.hard_percentile,
        "hard_error_threshold_m": p90,
        "n_epochs": len(est),
        "n_hard": int(hard.sum()),
        "n_easy": int(easy.sum()),
        "naive_at_sigma_star": naive_at_star,
        "curves": curves,
        "matched_far": matched,
        "consistency_diagnostics": consistency_diag,
        "interpretation": _interpret(matched, naive_at_star, consistency_diag),
    }

    (out_dir / "roc.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _plot_roc(curves, out_dir / "roc.png")
    _write_summary(payload, out_dir / "summary.md")

    _log("")
    _log(payload["interpretation"])
    _log(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
