#!/usr/bin/env python3
"""Run every prepared CERBERUS (and fault-injected) sequence through all monitors.

For each directory under ``data/processed/`` that contains ``est.tum`` + ``gt.tum``,
evaluate naive / degeneracy / consistency monitors and emit a ranked table::

    sequence, ATE, duration, MDR, FAR, mean PL, P95 PL, availability

Sequences are ranked by ATE (hardest first). Outputs: ``outputs/multiseq/``.

Claimed σ defaults to σ* from ``outputs/sigma_sweep/`` when available so PL
tightness is comparable to the ROC / sigma-sweep work — not an oracle per
sequence at runtime, but an honest shared operating point for the table.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

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


def _load_run_degeneracy():
    path = _repo_root() / "scripts" / "run_degeneracy.py"
    spec = importlib.util.spec_from_file_location("cairn_run_degeneracy_ms", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover_sequences(processed_root: Path) -> list[Path]:
    root = processed_root.resolve()
    if not root.is_dir():
        return []
    found = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "est.tum").is_file() and (child / "gt.tum").is_file():
            found.append(child)
    return found


def _duration_s(traj: Trajectory) -> float:
    if len(traj) < 2:
        return 0.0
    return float(traj[-1].pose.timestamp - traj[0].pose.timestamp)


def _pl_stats(reports: list[IntegrityReport]) -> tuple[float, float]:
    hpl = np.array([r.protection_level_horizontal for r in reports], dtype=np.float64)
    return float(np.mean(hpl)), float(np.percentile(hpl, 95))


def _row(
    *,
    sequence: str,
    monitor: str,
    ate: float,
    duration: float,
    summary: object,
    reports: list[IntegrityReport],
) -> dict[str, object]:
    mean_pl, p95_pl = _pl_stats(reports)
    return {
        "sequence": sequence,
        "monitor": monitor,
        "ate_rmse_m": ate,
        "duration_s": duration,
        "missed_detection_rate": float(summary.missed_detection_rate),  # type: ignore[attr-defined]
        "false_alarm_rate": float(summary.false_alarm_rate),  # type: ignore[attr-defined]
        "availability": float(summary.availability),  # type: ignore[attr-defined]
        "mean_pl_m": mean_pl,
        "p95_pl_m": p95_pl,
        "n_epochs": int(summary.epoch_count),  # type: ignore[attr-defined]
    }


def run_naive(
    est: Trajectory,
    gt: Trajectory,
    *,
    sigma: float,
    confidence: float,
    align: str,
    alert_limit: float,
) -> tuple[list[IntegrityReport], object]:
    est_c = with_claimed_diagonal_covariance(
        est, std_xy=sigma, std_z=sigma, source="naive"
    )
    reports = NaiveCovarianceMonitor(confidence=confidence).report(est_c)
    errors = true_errors_at_reports(
        est_c, gt, reports, align_mode=align, max_dt=0.05
    )
    return reports, integrity_report_summary(reports, errors, alert_limit=alert_limit)


def run_degeneracy(
    est: Trajectory,
    gt: Trajectory,
    *,
    sigma: float,
    confidence: float,
    align: str,
    alert_limit: float,
    score_threshold: float,
) -> tuple[list[IntegrityReport], object]:
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
    errors = true_errors_at_reports(
        est_c, gt, reports, align_mode=align, max_dt=0.05
    )
    return reports, integrity_report_summary(reports, errors, alert_limit=alert_limit)


def run_consistency(
    est: Trajectory,
    gt: Trajectory,
    *,
    sigma: float,
    align: str,
    alert_limit: float,
    lag_steps: int = 10,
) -> tuple[list[IntegrityReport], object]:
    """Synthetic primary / CV-prop / lag sources → SS reports (same as ROC)."""
    ref = SolutionSeparationDetector(p_fa=0.05, pl_confidence=0.999)
    diag = np.diag(
        [sigma**2, sigma**2, sigma**2, 0.05**2, 0.05**2, 0.05**2]
    )
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
        alarmed = result.faulted_source is not None
        state = IntegrityState.DO_NOT_TRUST if alarmed else IntegrityState.NOMINAL
        reason = (
            f"solution_separation:faulted={result.faulted_source}"
            if alarmed
            else "solution_separation:consistent"
        )
        reports.append(
            IntegrityReport(
                timestamp=float(cur.pose.timestamp),
                protection_level_horizontal=float(result.protection_level_horizontal),
                protection_level_vertical=float(result.protection_level_vertical),
                state=state,
                reason=reason,
                time_since_corroboration=0.0,
            )
        )
        if (i + 1) % max(1, n // 5) == 0:
            _log(f"    consistency {i + 1}/{n}")

    # Errors need a same-length estimate stream; use claimed-cov est for association.
    est_c = with_claimed_diagonal_covariance(
        est, std_xy=sigma, std_z=sigma, source="consistency_assoc"
    )
    errors = true_errors_at_reports(
        est_c, gt, reports, align_mode=align, max_dt=0.05
    )
    return reports, integrity_report_summary(reports, errors, alert_limit=alert_limit)


def evaluate_sequence(
    sequence_dir: Path,
    *,
    sigma: float,
    confidence: float,
    align: str,
    alert_limit: float,
    score_threshold: float,
) -> list[dict[str, object]]:
    est = load_tum(sequence_dir / "est.tum")
    gt = load_tum(sequence_dir / "gt.tum")
    ate = float(ate_rmse(est, gt, align))  # type: ignore[arg-type]
    duration = _duration_s(est)
    name = sequence_dir.name
    rows: list[dict[str, object]] = []

    _log("  naive…")
    naive_r, naive_s = run_naive(
        est, gt, sigma=sigma, confidence=confidence, align=align, alert_limit=alert_limit
    )
    rows.append(
        _row(
            sequence=name,
            monitor="naive",
            ate=ate,
            duration=duration,
            summary=naive_s,
            reports=naive_r,
        )
    )

    _log("  degeneracy…")
    deg_r, deg_s = run_degeneracy(
        est,
        gt,
        sigma=sigma,
        confidence=confidence,
        align=align,
        alert_limit=alert_limit,
        score_threshold=score_threshold,
    )
    rows.append(
        _row(
            sequence=name,
            monitor="degeneracy",
            ate=ate,
            duration=duration,
            summary=deg_s,
            reports=deg_r,
        )
    )

    _log("  consistency…")
    con_r, con_s = run_consistency(
        est, gt, sigma=sigma, align=align, alert_limit=alert_limit
    )
    rows.append(
        _row(
            sequence=name,
            monitor="consistency",
            ate=ate,
            duration=duration,
            summary=con_s,
            reports=con_r,
        )
    )
    return rows


def _write_table(rows: list[dict[str, object]], path: Path) -> None:
    # Rank sequences by ATE (desc); within sequence keep monitor order.
    seq_ate = {}
    for r in rows:
        seq_ate[str(r["sequence"])] = float(r["ate_rmse_m"])
    order = sorted(seq_ate, key=lambda s: seq_ate[s], reverse=True)
    monitor_order = {"naive": 0, "degeneracy": 1, "consistency": 2}
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            -seq_ate[str(r["sequence"])],
            monitor_order.get(str(r["monitor"]), 9),
        ),
    )

    lines = [
        "# Multi-sequence integrity table",
        "",
        "Sequences ranked by ATE (hardest first). Claimed σ shared across monitors.",
        "",
        "| sequence | monitor | ATE (m) | duration (s) | MDR | FAR | "
        "mean PL (m) | P95 PL (m) | availability |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted_rows:
        lines.append(
            f"| {r['sequence']} | {r['monitor']} | "
            f"{float(r['ate_rmse_m']):.4f} | {float(r['duration_s']):.1f} | "
            f"{100*float(r['missed_detection_rate']):.2f}% | "
            f"{100*float(r['false_alarm_rate']):.2f}% | "
            f"{float(r['mean_pl_m']):.3f} | {float(r['p95_pl_m']):.3f} | "
            f"{100*float(r['availability']):.2f}% |"
        )

    # Call out inflation: mean/P95 PL ≫ ATE with MDR≈0 (cowardice on easy data).
    lines.extend(["", "## PL tightness flags", ""])
    flagged = False
    for r in sorted_rows:
        ate = float(r["ate_rmse_m"])
        mean_pl = float(r["mean_pl_m"])
        p95_pl = float(r["p95_pl_m"])
        mdr = float(r["missed_detection_rate"])
        if mdr >= 1e-4 or ate < 1e-3:
            continue
        if mean_pl > 1.4 * ate or p95_pl > 3.0 * ate:
            flagged = True
            lines.append(
                f"- **{r['sequence']} / {r['monitor']}**: MDR≈0 with mean/P95 PL="
                f"{mean_pl:.3f}/{p95_pl:.3f} m vs ATE={ate:.3f} m "
                f"(P95/ATE={p95_pl/ate:.1f}×) — PL inflation, not detection."
            )
    if not flagged:
        lines.append(
            "- No MDR≈0 / PL≫ATE cowardice flags at the shared σ operating point."
        )

    faulted = [s for s in order if any(tag in s for tag in ("_bias", "_drift", "_dropout"))]
    if faulted:
        lines.extend(
            [
                "",
                "## Fault injection",
                "",
                "Hard sequences with `_bias` / `_dropout` / `_drift` suffixes are "
                "controlled faults on CompSLAM (`scripts/inject_faults.py`). "
                "Compare MDR under faults to the clean parent: that gap is the "
                "product claim when clean underground data alone looks too easy.",
            ]
        )
    elif len(order) == 1:
        lines.extend(
            [
                "",
                "## Dataset ceiling",
                "",
                "Only one prepared CERBERUS sequence. Run "
                "`python scripts/inject_faults.py` then re-run this script "
                "so the table includes controlled failure modes.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=_repo_root() / "data" / "processed",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_repo_root() / "outputs" / "multiseq",
    )
    parser.add_argument("--sigma", type=float, default=None, help="Claimed σ (default: σ*)")
    parser.add_argument("--confidence", type=float, default=0.999)
    parser.add_argument(
        "--align", choices=("none", "origin", "umeyama"), default="umeyama"
    )
    parser.add_argument("--alert-limit", type=float, default=2.0)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument(
        "--skip-consistency",
        action="store_true",
        help="Skip SS monitor (slow on long sequences)",
    )
    args = parser.parse_args(argv)

    sigma = float(args.sigma) if args.sigma is not None else _load_sigma_star()
    sequences = discover_sequences(args.processed_root)
    if not sequences:
        print(f"no prepared sequences under {args.processed_root}", file=sys.stderr)
        return 1

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    CairnConfig.load_default().to_yaml(out_dir / "config.yaml")

    _log(f"sequences={len(sequences)}  σ={100*sigma:.2f} cm  → {out_dir}")
    all_rows: list[dict[str, object]] = []
    for seq in sequences:
        _log(f"\n=== {seq.name} ===")
        if args.skip_consistency:
            est = load_tum(seq / "est.tum")
            gt = load_tum(seq / "gt.tum")
            ate = float(ate_rmse(est, gt, args.align))  # type: ignore[arg-type]
            duration = _duration_s(est)
            naive_r, naive_s = run_naive(
                est,
                gt,
                sigma=sigma,
                confidence=args.confidence,
                align=args.align,
                alert_limit=args.alert_limit,
            )
            deg_r, deg_s = run_degeneracy(
                est,
                gt,
                sigma=sigma,
                confidence=args.confidence,
                align=args.align,
                alert_limit=args.alert_limit,
                score_threshold=args.score_threshold,
            )
            all_rows.extend(
                [
                    _row(
                        sequence=seq.name,
                        monitor="naive",
                        ate=ate,
                        duration=duration,
                        summary=naive_s,
                        reports=naive_r,
                    ),
                    _row(
                        sequence=seq.name,
                        monitor="degeneracy",
                        ate=ate,
                        duration=duration,
                        summary=deg_s,
                        reports=deg_r,
                    ),
                ]
            )
        else:
            all_rows.extend(
                evaluate_sequence(
                    seq,
                    sigma=sigma,
                    confidence=args.confidence,
                    align=args.align,
                    alert_limit=args.alert_limit,
                    score_threshold=args.score_threshold,
                )
            )

    payload = {
        "sigma_xy_m": sigma,
        "align": args.align,
        "confidence": args.confidence,
        "n_sequences": len(sequences),
        "sequences": [s.name for s in sequences],
        "rows": all_rows,
    }
    (out_dir / "multiseq.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_table(all_rows, out_dir / "summary.md")
    _log(f"\nwrote {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
