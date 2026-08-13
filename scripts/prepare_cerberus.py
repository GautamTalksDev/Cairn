#!/usr/bin/env python3
"""Prepare a downloaded CERBERUS Final Event sequence for Cairn.

Coordinate frames
-----------------
CERBERUS publishes robot poses **with respect to the DARPA Finals course
frame** (``darpa``): a right-handed Cartesian frame fixed to the Louisville
Mega Cavern course, defined by DARPA surveyed fiducials / starting-gate
markers (see Team CERBERUS technical overview and the SubT finals ground-truth
documentation).

Each CSV pose line is already TUM-ordered::

    timestamp  tx ty tz  qx qy qz qw

where ``(tx, ty, tz)`` is the robot base position in metres in ``darpa``, and
``(qx, qy, qz, qw)`` is the body→``darpa`` orientation as a **w-last** unit
quaternion (TUM / evo convention).

Transform applied by this script
--------------------------------
* **SE(3):** identity. Poses are left in the DARPA frame; we do **not** map
  into UTM, base_link, or LiDAR frames.
* **Timestamp:** values ``>= 1e12`` are treated as nanoseconds and divided by
  ``1e9`` to yield Unix seconds (float). Values already in seconds are kept.
* **Quaternion order:** unchanged on disk (TUM w-last). Cairn's ``load_tum``
  converts to the library's internal w-first convention at read time.

Expected input layout (official libdrive share)::

    <sequence_dir>/
      trajectories/
        1_compslam/{gt.csv,compslam_tum.csv}
        2_compslam_post/...
        3_m3rm/...

``sequence_dir`` may also be the method folder itself (containing ``gt.csv``
and an estimate CSV).

Outputs under ``data/processed/<sequence_name>/``::

    gt.tum
    est.tum
    reports.txt   # placeholder integrity reports for the score harness
    meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cairn._internal.io.reports import save_integrity_reports
from cairn._internal.io.tum import load_tum, save_tum
from cairn.types import IntegrityReport, IntegrityState, Pose, PoseEstimate, Trajectory

# Official method folder → estimate filename.
_METHOD_ESTIMATE = {
    "1_compslam": "compslam_tum.csv",
    "compslam": "compslam_tum.csv",
    "2_compslam_post": "compslam_post_tum.csv",
    "compslam_post": "compslam_post_tum.csv",
    "3_m3rm": "m3rm_tum.csv",
    "m3rm": "m3rm_tum.csv",
}

_GAP_THRESHOLD_S = 0.5


@dataclass(frozen=True)
class TrajectoryStats:
    name: str
    n_poses: int
    duration_s: float
    path_length_m: float
    mean_rate_hz: float
    n_gaps_over_0_5s: float
    max_gap_s: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize_timestamp(ts: float) -> float:
    """Convert CERBERUS timestamps to Unix seconds."""
    ts = float(ts)
    if ts >= 1e12:  # nanoseconds (integer or float)
        return ts / 1e9
    return ts


def load_cerberus_tum_csv(path: Path) -> Trajectory:
    """Load a CERBERUS TUM CSV (DARPA frame, w-last quaternion) into a Trajectory.

    No SE(3) remapping is applied; see module docstring.
    """
    estimates: list[PoseEstimate] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(
                    f"{path}:{line_no}: expected 8 TUM fields, got {len(parts)}"
                )
            ts, tx, ty, tz, qx, qy, qz, qw = (float(p) for p in parts)
            # File is TUM w-last; Pose stores w-first.
            pose = Pose(
                timestamp=_normalize_timestamp(ts),
                position=np.array([tx, ty, tz], dtype=np.float64),
                orientation=np.array([qw, qx, qy, qz], dtype=np.float64),
                frame="darpa",
            )
            estimates.append(
                PoseEstimate(
                    pose=pose,
                    covariance=np.eye(6),
                    source=path.stem,
                )
            )
    if not estimates:
        raise ValueError(f"{path}: no poses found")
    return Trajectory(estimates)


def validate_timestamps(traj: Trajectory, label: str) -> tuple[int, float]:
    """Check strict monotonicity; report gaps larger than 0.5 s.

    Returns ``(n_gaps, max_gap_s)``.
    """
    ts = traj.timestamps()
    if ts.size < 2:
        print(f"[{label}] fewer than 2 poses; skip gap analysis")
        return 0, 0.0
    dts = np.diff(ts)
    if np.any(dts <= 0.0):
        bad = int(np.sum(dts <= 0.0))
        raise ValueError(f"{label}: timestamps not strictly increasing ({bad} non-positive Δt)")
    gap_idx = np.where(dts > _GAP_THRESHOLD_S)[0]
    max_gap = float(dts.max())
    if gap_idx.size:
        print(
            f"[{label}] WARNING: {gap_idx.size} gap(s) > {_GAP_THRESHOLD_S} s "
            f"(max Δt = {max_gap:.3f} s)"
        )
        for i in gap_idx[:20]:
            print(
                f"  gap after pose {i}: t={ts[i]:.6f} → {ts[i + 1]:.6f} "
                f"(Δt={dts[i]:.3f} s)"
            )
        if gap_idx.size > 20:
            print(f"  ... and {gap_idx.size - 20} more")
    else:
        print(f"[{label}] timestamps strictly increasing; no gaps > {_GAP_THRESHOLD_S} s")
    return int(gap_idx.size), max_gap


def trajectory_stats(name: str, traj: Trajectory, n_gaps: int, max_gap: float) -> TrajectoryStats:
    ts = traj.timestamps()
    positions = np.array([e.pose.position for e in traj], dtype=np.float64)
    duration = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    path = (
        float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
        if len(positions) > 1
        else 0.0
    )
    rate = (len(traj) - 1) / duration if duration > 0.0 else 0.0
    return TrajectoryStats(
        name=name,
        n_poses=len(traj),
        duration_s=duration,
        path_length_m=path,
        mean_rate_hz=rate,
        n_gaps_over_0_5s=float(n_gaps),
        max_gap_s=max_gap,
    )


def print_summary_table(rows: list[TrajectoryStats]) -> None:
    headers = (
        "sequence",
        "duration_s",
        "path_m",
        "n_poses",
        "rate_Hz",
        "gaps>0.5s",
        "max_gap_s",
    )
    table = [
        [
            r.name,
            f"{r.duration_s:.1f}",
            f"{r.path_length_m:.1f}",
            str(r.n_poses),
            f"{r.mean_rate_hz:.2f}",
            f"{int(r.n_gaps_over_0_5s)}",
            f"{r.max_gap_s:.3f}",
        ]
        for r in rows
    ]
    widths = [len(h) for h in headers]
    for row in table:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cols: list[str]) -> str:
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    print()
    print(fmt(list(headers)))
    print("-+-".join("-" * w for w in widths))
    for row in table:
        print(fmt(row))


def resolve_inputs(sequence_dir: Path, method: str) -> tuple[Path, Path, str]:
    """Return ``(gt_csv, est_csv, sequence_name)``."""
    sequence_dir = sequence_dir.resolve()
    method_key = method.strip().lower()

    # Case A: pointing at a method folder that already contains the CSVs.
    if (sequence_dir / "gt.csv").is_file():
        est_name = _METHOD_ESTIMATE.get(method_key) or _METHOD_ESTIMATE.get(sequence_dir.name)
        if est_name is None:
            # Prefer any known estimate CSV present.
            for candidate in (
                "compslam_tum.csv",
                "compslam_post_tum.csv",
                "m3rm_tum.csv",
            ):
                if (sequence_dir / candidate).is_file():
                    est_name = candidate
                    break
        if est_name is None or not (sequence_dir / est_name).is_file():
            raise FileNotFoundError(
                f"No estimate CSV found in {sequence_dir}; "
                f"expected one of {sorted(set(_METHOD_ESTIMATE.values()))}"
            )
        seq_name = sequence_dir.parent.parent.name  # .../anymal_1/trajectories/1_compslam
        if seq_name == "trajectories":
            seq_name = sequence_dir.parent.name
        # Prefer robot name when layout is anymal_X/trajectories/<method>
        parts = sequence_dir.parts
        for part in reversed(parts):
            if part.startswith("anymal_"):
                seq_name = f"{part}_{sequence_dir.name}"
                break
        else:
            seq_name = sequence_dir.name
        return sequence_dir / "gt.csv", sequence_dir / est_name, seq_name

    # Case B: pointing at robot root with trajectories/<method>/
    method_dir_name = {
        "compslam": "1_compslam",
        "1_compslam": "1_compslam",
        "compslam_post": "2_compslam_post",
        "2_compslam_post": "2_compslam_post",
        "m3rm": "3_m3rm",
        "3_m3rm": "3_m3rm",
    }.get(method_key)
    if method_dir_name is None:
        raise ValueError(
            f"Unknown method {method!r}; expected compslam, compslam_post, or m3rm"
        )
    method_dir = sequence_dir / "trajectories" / method_dir_name
    if not method_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {method_dir} under downloaded CERBERUS sequence directory"
        )
    est_name = _METHOD_ESTIMATE[method_dir_name]
    gt = method_dir / "gt.csv"
    est = method_dir / est_name
    if not gt.is_file() or not est.is_file():
        raise FileNotFoundError(f"Missing {gt.name} or {est_name} in {method_dir}")
    seq_name = f"{sequence_dir.name}_{method_dir_name}"
    return gt, est, seq_name


def placeholder_reports(gt: Trajectory, pl_m: float = 1.0) -> list[IntegrityReport]:
    """Build NOMINAL integrity reports at GT timestamps (pre-integrity-estimator)."""
    return [
        IntegrityReport(
            timestamp=e.pose.timestamp,
            protection_level_horizontal=pl_m,
            protection_level_vertical=pl_m,
            state=IntegrityState.NOMINAL,
            reason="placeholder_until_integrity_estimator",
            time_since_corroboration=0.0,
        )
        for e in gt
    ]


def prepare(
    sequence_dir: Path,
    *,
    method: str = "compslam",
    out_root: Path | None = None,
    sequence_name: str | None = None,
    alert_pl_m: float = 1.0,
) -> Path:
    gt_path, est_path, default_name = resolve_inputs(sequence_dir, method)
    name = sequence_name or default_name
    out_root = out_root or (_repo_root() / "data" / "processed")
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading GT:  {gt_path}")
    print(f"Loading EST: {est_path}")
    print("Frame: darpa (CERBERUS / DARPA Finals course); SE(3) transform: identity")
    gt = load_cerberus_tum_csv(gt_path)
    est = load_cerberus_tum_csv(est_path)

    gt_gaps, gt_max_gap = validate_timestamps(gt, "gt")
    est_gaps, est_max_gap = validate_timestamps(est, "est")

    save_tum(gt, out_dir / "gt.tum")
    save_tum(est, out_dir / "est.tum")
    reports = placeholder_reports(gt, pl_m=alert_pl_m)
    save_integrity_reports(reports, out_dir / "reports.txt")

    # Round-trip sanity via Cairn TUM loader.
    _ = load_tum(out_dir / "gt.tum")
    _ = load_tum(out_dir / "est.tum")

    stats = [
        trajectory_stats(f"{name}/gt", gt, gt_gaps, gt_max_gap),
        trajectory_stats(f"{name}/est", est, est_gaps, est_max_gap),
    ]
    print_summary_table(stats)

    meta = {
        "sequence": name,
        "source_gt": str(gt_path),
        "source_est": str(est_path),
        "frame": "darpa",
        "se3_transform": "identity",
        "quaternion_file_order": "xyzw (TUM w-last)",
        "dataset_doi": "https://doi.org/10.3929/ethz-b-000552248",
        "license": "In Copyright - Non-Commercial Use Permitted "
        "(http://rightsstatements.org/page/InC-NC/1.0/)",
        "stats": [
            {
                "name": s.name,
                "n_poses": s.n_poses,
                "duration_s": s.duration_s,
                "path_length_m": s.path_length_m,
                "mean_rate_hz": s.mean_rate_hz,
                "n_gaps_over_0_5s": int(s.n_gaps_over_0_5s),
                "max_gap_s": s.max_gap_s,
            }
            for s in stats
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote processed artifacts to {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sequence_dir",
        type=Path,
        help="Downloaded CERBERUS robot dir (anymal_N) or method dir (1_compslam)",
    )
    parser.add_argument(
        "--method",
        default="compslam",
        choices=("compslam", "compslam_post", "m3rm", "1_compslam", "2_compslam_post", "3_m3rm"),
        help="Which estimate to pair with GT (default: compslam)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Processed output root (default: <repo>/data/processed)",
    )
    parser.add_argument(
        "--sequence-name",
        default=None,
        help="Override output folder name under data/processed/",
    )
    parser.add_argument(
        "--pl",
        type=float,
        default=1.0,
        help="Placeholder horizontal/vertical protection level in metres",
    )
    args = parser.parse_args(argv)
    prepare(
        args.sequence_dir,
        method=args.method,
        out_root=args.out_root,
        sequence_name=args.sequence_name,
        alert_pl_m=args.pl,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
