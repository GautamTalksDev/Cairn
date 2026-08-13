#!/usr/bin/env python3
"""Inject controlled faults into a prepared CERBERUS trajectory.

Clean underground runs (e.g. anymal_1 CompSLAM) can sit under a dataset ceiling:
well-tuned static covariance looks both safe and useful. Fault injection creates
failures on demand so monitors can be scored when the thesis actually bites.

Faults (applied to ``est.tum`` only; GT unchanged)::

    bias     — constant XY offset over a mid-run window
    dropout  — freeze pose for a contiguous block (dead reckoning stall)
    drift    — linearly growing along-track bias

Outputs under ``data/processed/<source>_<fault>/`` with ``meta.json`` noting
the injection so multi-seq ranking stays honest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cairn._internal.io.tum import load_tum, save_tum
from cairn.types import Pose, PoseEstimate, Trajectory


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _window_indices(n: int, start_frac: float, duration_frac: float) -> slice:
    i0 = int(n * start_frac)
    i1 = min(n, i0 + max(1, int(n * duration_frac)))
    return slice(i0, i1)


def _headings_xy(positions: np.ndarray) -> np.ndarray:
    delta = np.diff(positions[:, :2], axis=0)
    norms = np.linalg.norm(delta, axis=1, keepdims=True)
    dirs = np.divide(delta, np.maximum(norms, 1e-9))
    return np.vstack([dirs, dirs[-1]])


def inject_bias(
    est: Trajectory,
    *,
    offset_m: float = 1.0,
    start_frac: float = 0.40,
    duration_frac: float = 0.15,
) -> Trajectory:
    """Constant world-XY bias over a contiguous window."""
    n = len(est)
    win = _window_indices(n, start_frac, duration_frac)
    offset = np.array([offset_m, 0.0, 0.0], dtype=np.float64)
    out: list[PoseEstimate] = []
    for i, pe in enumerate(est):
        pos = pe.pose.position.copy()
        if win.start <= i < win.stop:
            pos = pos + offset
        out.append(
            PoseEstimate(
                pose=Pose(
                    timestamp=pe.pose.timestamp,
                    position=pos,
                    orientation=pe.pose.orientation,
                    frame=pe.pose.frame,
                ),
                covariance=pe.covariance.copy(),
                source=f"{pe.source}+bias",
            )
        )
    return Trajectory(out)


def inject_dropout(
    est: Trajectory,
    *,
    start_frac: float = 0.55,
    duration_frac: float = 0.08,
) -> Trajectory:
    """Freeze position/orientation for a contiguous block (filter stall)."""
    n = len(est)
    win = _window_indices(n, start_frac, duration_frac)
    freeze = est[win.start]
    out: list[PoseEstimate] = []
    for i, pe in enumerate(est):
        if win.start <= i < win.stop:
            out.append(
                PoseEstimate(
                    pose=Pose(
                        timestamp=pe.pose.timestamp,
                        position=freeze.pose.position.copy(),
                        orientation=freeze.pose.orientation.copy(),
                        frame=pe.pose.frame,
                    ),
                    covariance=pe.covariance.copy(),
                    source=f"{pe.source}+dropout",
                )
            )
        else:
            out.append(pe)
    return Trajectory(out)


def inject_drift(
    est: Trajectory,
    *,
    rate_m_per_s: float = 0.02,
    start_frac: float = 0.30,
) -> Trajectory:
    """Along-track bias that grows linearly after ``start_frac``."""
    n = len(est)
    i0 = int(n * start_frac)
    positions = np.array([e.pose.position for e in est], dtype=np.float64)
    headings = _headings_xy(positions)
    t0 = float(est[i0].pose.timestamp)
    out: list[PoseEstimate] = []
    for i, pe in enumerate(est):
        pos = pe.pose.position.copy()
        if i >= i0:
            dt = float(pe.pose.timestamp) - t0
            mag = rate_m_per_s * max(dt, 0.0)
            h = headings[i]
            pos[:2] = pos[:2] + mag * h
        out.append(
            PoseEstimate(
                pose=Pose(
                    timestamp=pe.pose.timestamp,
                    position=pos,
                    orientation=pe.pose.orientation,
                    frame=pe.pose.frame,
                ),
                covariance=pe.covariance.copy(),
                source=f"{pe.source}+drift",
            )
        )
    return Trajectory(out)


_INJECTORS = {
    "bias": inject_bias,
    "dropout": inject_dropout,
    "drift": inject_drift,
}


def write_faulted_sequence(
    source_dir: Path,
    fault: str,
    out_dir: Path,
    **kwargs: object,
) -> Path:
    source_dir = source_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    est = load_tum(source_dir / "est.tum")
    gt = load_tum(source_dir / "gt.tum")
    injector = _INJECTORS[fault]
    faulted = injector(est, **kwargs)  # type: ignore[arg-type]

    save_tum(gt, out_dir / "gt.tum")
    save_tum(faulted, out_dir / "est.tum")

    meta = {
        "sequence": out_dir.name,
        "parent_sequence": source_dir.name,
        "fault": fault,
        "fault_params": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in kwargs.items()},
        "n_poses": len(faulted),
        "duration_s": float(faulted[-1].pose.timestamp - faulted[0].pose.timestamp),
        "note": "Controlled fault injection; GT unchanged. For monitor stress tests.",
    }
    if (source_dir / "meta.json").is_file():
        parent = json.loads((source_dir / "meta.json").read_text(encoding="utf-8"))
        meta["parent_meta"] = {
            k: parent[k]
            for k in ("dataset_doi", "license", "frame")
            if k in parent
        }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out_dir


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
        "--faults",
        nargs="+",
        choices=tuple(_INJECTORS.keys()) + ("all",),
        default=["all"],
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=_repo_root() / "data" / "processed",
    )
    parser.add_argument("--bias-m", type=float, default=1.0)
    parser.add_argument("--drift-rate", type=float, default=0.02, help="m/s along-track")
    args = parser.parse_args(argv)

    faults = list(_INJECTORS.keys()) if "all" in args.faults else list(args.faults)
    source = args.source.resolve()
    if not (source / "est.tum").is_file():
        print(f"missing est.tum under {source}", file=sys.stderr)
        return 1

    for fault in faults:
        name = f"{source.name}_{fault}"
        out = args.out_root / name
        kwargs: dict[str, object] = {}
        if fault == "bias":
            kwargs = {"offset_m": args.bias_m}
        elif fault == "drift":
            kwargs = {"rate_m_per_s": args.drift_rate}
        path = write_faulted_sequence(source, fault, out, **kwargs)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
