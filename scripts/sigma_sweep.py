#!/usr/bin/env python3
"""Sweep claimed σ_xy for the naive baseline and expose the safety–usefulness tradeoff.

CompSLAM TUM exports carry no covariance. The CP-7 headline MDR at σ=5 cm is therefore
an assumption, not a measurement. This sweep answers the reviewer's question:

* At what claimed σ does MDR fall below 0.1%?
* What horizontal protection level (HPL) does that σ imply?

If that HPL is large relative to ATE, static self-reported covariance cannot be both
safe and useful — the product case. If some σ is both safe and tight, naive covariance
is enough and the product case is weaker.

Outputs under ``outputs/sigma_sweep/``:
  - ``sigma_sweep.csv`` / ``sigma_sweep.json``
  - ``mdr_far_vs_sigma.png``
  - ``summary.md``
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cairn._internal.baselines.naive import (
    NaiveCovarianceMonitor,
    chi2_scale,
    with_claimed_diagonal_covariance,
)
from cairn._internal.io.tum import load_tum
from cairn.cli import true_errors_at_reports
from cairn.config import CairnConfig
from cairn.metrics.error import ate_rmse
from cairn.metrics.integrity import false_alarm_rate, missed_detection_rate


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_sequence() -> Path:
    root = _repo_root()
    full = root / "data" / "processed" / "anymal_1_1_compslam"
    if (full / "est.tum").is_file():
        return full
    return root / "examples" / "quickstart"


def hpl_from_sigma(sigma_xy: float, *, confidence: float = 0.999) -> float:
    """HPL for isotropic horizontal variance σ² (chi², 2 DoF)."""
    return float(chi2_scale(confidence, 2) * sigma_xy)


def run_sigma_sweep(
    sequence_dir: Path,
    *,
    sigmas_m: np.ndarray,
    confidence: float = 0.999,
    align_mode: str = "umeyama",
    alert_limit: float = 2.0,
    mdr_target: float = 0.001,
) -> dict[str, object]:
    sequence_dir = sequence_dir.resolve()
    est = load_tum(sequence_dir / "est.tum")
    gt = load_tum(sequence_dir / "gt.tum")
    ate = ate_rmse(est, gt, align_mode)  # type: ignore[arg-type]

    # True errors depend on alignment, not on claimed σ — compute once.
    seed = with_claimed_diagonal_covariance(
        est, std_xy=float(sigmas_m[0]), std_z=float(sigmas_m[0]), source="sigma_sweep"
    )
    seed_reports = NaiveCovarianceMonitor(confidence=confidence).report(seed)
    true_errors = true_errors_at_reports(
        seed, gt, seed_reports, align_mode=align_mode, max_dt=0.05
    )

    rows: list[dict[str, float]] = []
    monitor = NaiveCovarianceMonitor(confidence=confidence)

    for sigma in sigmas_m:
        sigma = float(sigma)
        est_s = with_claimed_diagonal_covariance(
            est, std_xy=sigma, std_z=sigma, source="sigma_sweep"
        )
        reports = monitor.report(est_s)
        mdr = missed_detection_rate(reports, true_errors)
        far = false_alarm_rate(reports, true_errors)
        hpl = hpl_from_sigma(sigma, confidence=confidence)
        mean_pl = float(np.mean([r.protection_level_horizontal for r in reports]))
        rows.append(
            {
                "sigma_xy_m": sigma,
                "sigma_xy_cm": 100.0 * sigma,
                "hpl_m": hpl,
                "mean_reported_pl_m": mean_pl,
                "mdr": mdr,
                "far": far,
                "availability": 1.0,  # naive is always NOMINAL
            }
        )

    # First σ (ascending) where MDR <= target; interpolate if we bracket the crossing.
    sigmas = np.array([r["sigma_xy_m"] for r in rows], dtype=np.float64)
    mdrs = np.array([r["mdr"] for r in rows], dtype=np.float64)
    crossing: dict[str, object] | None = None
    below = np.where(mdrs <= mdr_target)[0]
    if below.size:
        i = int(below[0])
        if i == 0:
            sigma_star = float(sigmas[0])
        else:
            # Linear interpolate in σ between i-1 (MDR > target) and i (MDR <= target).
            s0, s1 = float(sigmas[i - 1]), float(sigmas[i])
            m0, m1 = float(mdrs[i - 1]), float(mdrs[i])
            if m0 == m1:
                sigma_star = s1
            else:
                # mdr decreases with σ; solve mdr_target = m0 + t (m1-m0)
                t = (mdr_target - m0) / (m1 - m0)
                t = float(np.clip(t, 0.0, 1.0))
                sigma_star = s0 + t * (s1 - s0)
        crossing = {
            "mdr_target": mdr_target,
            "sigma_xy_m": sigma_star,
            "sigma_xy_cm": 100.0 * sigma_star,
            "hpl_m": hpl_from_sigma(sigma_star, confidence=confidence),
            "hpl_over_ate": hpl_from_sigma(sigma_star, confidence=confidence) / ate
            if ate > 0
            else None,
            "bracket_sigma_m": [float(sigmas[max(i - 1, 0)]), float(sigmas[i])],
            "bracket_mdr": [float(mdrs[max(i - 1, 0)]), float(mdrs[i])],
        }
    else:
        crossing = {
            "mdr_target": mdr_target,
            "sigma_xy_m": None,
            "sigma_xy_cm": None,
            "hpl_m": None,
            "hpl_over_ate": None,
            "note": f"MDR never fell to {mdr_target} in the swept range "
            f"[{float(sigmas.min()):.3f}, {float(sigmas.max()):.3f}] m",
        }

    return {
        "sequence": sequence_dir.name,
        "align_mode": align_mode,
        "confidence": confidence,
        "alert_limit_m": alert_limit,
        "ate_rmse_m": ate,
        "mdr_target": mdr_target,
        "rows": rows,
        "crossing": crossing,
        "finding": _interpret(ate, crossing),
    }


def _interpret(ate: float, crossing: dict[str, object]) -> str:
    hpl = crossing.get("hpl_m")
    if hpl is None:
        return (
            "No σ in the sweep drives MDR to the target. "
            "Either widen the sweep or the error distribution has a heavy tail."
        )
    hpl_f = float(hpl)
    ratio = hpl_f / ate if ate > 0 else float("inf")
    if ratio >= 5.0:
        return (
            f"To reach MDR ≤ {crossing['mdr_target']}, claimed σ implies "
            f"HPL ≈ {hpl_f:.3f} m, which is {ratio:.1f}× the ATE RMSE "
            f"({ate:.3f} m). Static self-reported covariance cannot be both "
            "safe (low MDR) and useful (tight PL) on this sequence."
        )
    if ratio >= 2.0:
        return (
            f"Acceptable MDR requires HPL ≈ {hpl_f:.3f} m "
            f"({ratio:.1f}× ATE={ate:.3f} m). Usability is strained; "
            "an integrity layer that adapts PL is still motivated."
        )
    return (
        f"Some σ yields MDR ≤ {crossing['mdr_target']} with HPL ≈ {hpl_f:.3f} m "
        f"only {ratio:.1f}× ATE ({ate:.3f} m). On this sequence, a well-tuned "
        "static covariance may already be competitive — product case is weaker."
    )


def _plot(rows: list[dict[str, float]], crossing: dict[str, object], out: Path) -> None:
    sig_cm = np.array([r["sigma_xy_cm"] for r in rows])
    mdr = np.array([r["mdr"] for r in rows])
    far = np.array([r["far"] for r in rows])
    hpl = np.array([r["hpl_m"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)

    ax0 = axes[0]
    ax0.plot(sig_cm, 100.0 * mdr, color="#b71c1c", lw=2.0, label="MDR")
    ax0.plot(sig_cm, 100.0 * far, color="#1565c0", lw=2.0, ls="--", label="FAR")
    ax0.axhline(0.1, color="#455a64", lw=1.2, ls=":", label="MDR = 0.1%")
    if crossing.get("sigma_xy_cm") is not None:
        ax0.axvline(
            float(crossing["sigma_xy_cm"]),
            color="#ef6c00",
            lw=1.4,
            ls="-.",
            label=f"σ* ≈ {float(crossing['sigma_xy_cm']):.1f} cm",
        )
    ax0.set_xlabel("Claimed σ_xy (cm)")
    ax0.set_ylabel("Rate (%)")
    ax0.set_title("Naive baseline: MDR / FAR vs claimed σ")
    ax0.set_xscale("log")
    ax0.grid(True, which="both", color="#cfd8dc", lw=0.7)
    ax0.legend(fontsize=9)

    ax1 = axes[1]
    ax1.plot(sig_cm, hpl, color="#6a1b9a", lw=2.0, label="HPL (χ² 99.9%, 2 DoF)")
    if crossing.get("hpl_m") is not None:
        ax1.axhline(
            float(crossing["hpl_m"]),
            color="#ef6c00",
            lw=1.4,
            ls="-.",
            label=f"HPL(σ*) ≈ {float(crossing['hpl_m']):.2f} m",
        )
    ax1.set_xlabel("Claimed σ_xy (cm)")
    ax1.set_ylabel("Protection level (m)")
    ax1.set_title("Implied horizontal protection level")
    ax1.set_xscale("log")
    ax1.grid(True, which="both", color="#cfd8dc", lw=0.7)
    ax1.legend(fontsize=9)

    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


def _write_summary(payload: dict[str, object], path: Path) -> None:
    c = payload["crossing"]  # type: ignore[index]
    assert isinstance(c, dict)
    lines = [
        "# Sigma sweep — naive covariance baseline",
        "",
        f"- Sequence: `{payload['sequence']}`",
        f"- Alignment: `{payload['align_mode']}`",
        f"- ATE RMSE: **{float(payload['ate_rmse_m']):.4f} m**",
        f"- Chi² confidence: {payload['confidence']}",
        f"- MDR target: {payload['mdr_target']}",
        "",
        "## Crossing (MDR ≤ target)",
        "",
    ]
    if c.get("sigma_xy_m") is None:
        lines.append(f"- {c.get('note', 'No crossing found.')}")
    else:
        lines.extend(
            [
                f"- σ* ≈ **{float(c['sigma_xy_cm']):.2f} cm** "
                f"({float(c['sigma_xy_m']):.4f} m)",
                f"- Implied HPL ≈ **{float(c['hpl_m']):.3f} m**",
                f"- HPL / ATE ≈ **{float(c['hpl_over_ate']):.2f}×**",
            ]
        )
    lines.extend(
        [
            "",
            "## Finding",
            "",
            str(payload["finding"]),
            "",
            "## Note on FAR",
            "",
            "The naive monitor never leaves `NOMINAL`, so FAR is identically zero "
            "for every σ. The relevant tradeoff is **MDR vs protection-level size**, "
            "not MDR vs FAR.",
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
        "--sequence-dir",
        type=Path,
        default=None,
        help="Prepared sequence (default: full anymal_1 if present, else quickstart)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/sigma_sweep)",
    )
    parser.add_argument(
        "--mdr-target",
        type=float,
        default=0.001,
        help="MDR crossing threshold (default: 0.001 = 0.1%%)",
    )
    parser.add_argument("--align", choices=("none", "origin", "umeyama"), default="umeyama")
    parser.add_argument("--confidence", type=float, default=0.999)
    args = parser.parse_args(argv)

    sequence_dir = args.sequence_dir or _default_sequence()
    out_dir = args.out_dir or (_repo_root() / "outputs" / "sigma_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    CairnConfig.load_default().to_yaml(out_dir / "config.yaml")

    # 1 cm → 100 cm, denser at the low end where the MDR knee lives.
    sigmas = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.10, 19),
                np.linspace(0.12, 0.30, 10),
                np.linspace(0.35, 1.00, 14),
            ]
        )
    )

    payload = run_sigma_sweep(
        sequence_dir,
        sigmas_m=sigmas,
        confidence=args.confidence,
        align_mode=args.align,
        mdr_target=args.mdr_target,
    )

    rows = payload["rows"]  # type: ignore[assignment]
    assert isinstance(rows, list)

    csv_path = out_dir / "sigma_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "sigma_sweep.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _plot(rows, payload["crossing"], out_dir / "mdr_far_vs_sigma.png")  # type: ignore[arg-type]
    _write_summary(payload, out_dir / "summary.md")

    crossing = payload["crossing"]
    assert isinstance(crossing, dict)
    print("=== Naive σ sweep ===")
    print(f"sequence: {payload['sequence']}")
    print(f"ATE RMSE: {float(payload['ate_rmse_m']):.4f} m")
    if crossing.get("sigma_xy_m") is None:
        print(crossing.get("note"))
    else:
        print(
            f"MDR ≤ {payload['mdr_target']}: σ* ≈ {float(crossing['sigma_xy_cm']):.2f} cm "
            f"→ HPL ≈ {float(crossing['hpl_m']):.3f} m "
            f"({float(crossing['hpl_over_ate']):.1f}× ATE)"
        )
    print()
    print(payload["finding"])
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
