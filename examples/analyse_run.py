#!/usr/bin/env python3
"""Post-hoc analysis of a Cairn diagnostics JSONL file.

Flow:

1. Run the monitor on the **synthetic** quickstart pair
   (``scripts/make_quickstart_data.py``) with diagnostics enabled.
2. Load the JSONL with :func:`cairn._internal.diagnostics.load_diagnostics`.
3. Print state histogram, protection-level stats, and state transitions.

Units in the JSONL match the live API:

* ``timestamp``: seconds
* ``protection_level_horizontal`` / ``protection_level_vertical``: metres
* ``state``: ``nominal`` | ``degraded`` | ``do_not_trust``

Run from the repo root::

    python examples/analyse_run.py
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from cairn import CairnConfig, CairnMonitor, PoseEstimate
from cairn._internal.diagnostics import load_diagnostics
from cairn._internal.io.tum import load_tum
from cairn.config import DiagnosticsConfig

_HERE = Path(__file__).resolve().parent
_QUICKSTART = _HERE / "quickstart"
_EST_TUM = _QUICKSTART / "est.tum"
_GT_TUM = _QUICKSTART / "gt.tum"

_STD_XY_M = 0.05
_STD_Z_M = 0.05
_STD_ROT_RAD = 0.05


def _covariance() -> np.ndarray:
    """Diagonal covariance in m² / rad²."""
    return np.diag(
        [
            _STD_XY_M**2,
            _STD_XY_M**2,
            _STD_Z_M**2,
            _STD_ROT_RAD**2,
            _STD_ROT_RAD**2,
            _STD_ROT_RAD**2,
        ]
    )


def _pair_epochs(
    a: list[PoseEstimate],
    b: list[PoseEstimate],
    *,
    max_dt_s: float = 0.05,
) -> list[tuple[float, dict[str, PoseEstimate]]]:
    """Nearest-neighbour association; timestamps in seconds."""
    b_ts = np.array([e.pose.timestamp for e in b], dtype=np.float64)
    out: list[tuple[float, dict[str, PoseEstimate]]] = []
    used: set[int] = set()
    for ea in a:
        t = float(ea.pose.timestamp)
        j = int(np.searchsorted(b_ts, t))
        best_j = None
        best_dt = float("inf")
        for k in (j - 1, j):
            if 0 <= k < b_ts.size and k not in used:
                dt = abs(float(b_ts[k]) - t)
                if dt < best_dt:
                    best_dt = dt
                    best_j = k
        if best_j is not None and best_dt <= max_dt_s:
            used.add(best_j)
            eb = b[best_j]
            out.append((t, {ea.source: ea, eb.source: eb}))
    return out


def run_monitor_to_jsonl(path: Path, *, max_epochs: int = 40) -> int:
    """Write diagnostics while scoring the bundled trajectories.

    Returns the number of epochs written.
    """
    cov = _covariance()
    lidar = list(load_tum(_EST_TUM, default_covariance=cov, source="lidar", frame="map"))
    vio = list(load_tum(_GT_TUM, default_covariance=cov, source="vio", frame="map"))
    epochs = _pair_epochs(lidar, vio)[:max_epochs]
    if not epochs:
        raise RuntimeError(
            "no paired epochs from synthetic quickstart; run "
            "python scripts/make_quickstart_data.py --seed 0"
        )

    cfg = CairnConfig(
        diagnostics=DiagnosticsConfig(enabled=True, path=str(path)),
    )
    with CairnMonitor(cfg) as mon:
        for t_s, estimates in epochs:
            mon.update(t_s, estimates)
    return len(epochs)


def analyse_jsonl(path: Path) -> None:
    """Summarise a diagnostics JSONL produced by CairnMonitor."""
    rows = load_diagnostics(path)
    if not rows:
        print(f"no rows in {path}")
        return

    states = Counter(str(r.get("state", "?")) for r in rows)
    hpls = np.array(
        [float(r["protection_level_horizontal"]) for r in rows],
        dtype=np.float64,
    )
    times_s = np.array([float(r["timestamp"]) for r in rows], dtype=np.float64)

    print(f"file: {path}")
    print(f"epochs: {len(rows)}")
    print(
        f"time span: {float(times_s[-1] - times_s[0]):.3f} s "
        f"(from {times_s[0]:.3f} to {times_s[-1]:.3f})"
    )
    print("state counts:")
    for name, count in sorted(states.items()):
        print(f"  {name:16s}  {count:4d}  ({100.0 * count / len(rows):.1f}%)")

    print(
        f"HPL (metres): min={hpls.min():.3f}  median={np.median(hpls):.3f}  "
        f"max={hpls.max():.3f}"
    )

    # State transitions (compare consecutive epoch states).
    transitions = 0
    print("state transitions:")
    prev = None
    for r in rows:
        cur = str(r.get("state"))
        if prev is not None and cur != prev:
            transitions += 1
            trigger = r.get("trigger") or {}
            print(
                f"  t={float(r['timestamp']):.3f}s  {prev} → {cur}  "
                f"reason={r.get('reason')!r}  "
                f"trigger={trigger.get('name', 'n/a')}"
            )
        prev = cur
    if transitions == 0:
        print("  (none)")


def main() -> None:
    if not _EST_TUM.is_file() or not _GT_TUM.is_file():
        raise SystemExit(
            "missing examples/quickstart/*.tum; run:\n"
            "  python scripts/make_quickstart_data.py --seed 0"
        )
    with tempfile.TemporaryDirectory(prefix="cairn_diag_") as tmp:
        jsonl = Path(tmp) / "run.jsonl"
        n = run_monitor_to_jsonl(jsonl)
        print(f"analyse_run.py: wrote {n} diagnostic epochs → {jsonl.name}")
        print("---")
        analyse_jsonl(jsonl)


if __name__ == "__main__":
    main()
