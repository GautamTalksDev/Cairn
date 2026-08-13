#!/usr/bin/env python3
"""Plot Cairn JSONL diagnostics: per-source test statistics vs thresholds.

Example::

    python scripts/replay_diagnostics.py outputs/run/diagnostics.jsonl \\
        --out outputs/run/diagnostics.png

Does not re-run detectors — only reads the opt-in JSONL dump from a prior
:class:`~cairn.monitor.CairnMonitor` session.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cairn._internal.diagnostics import load_diagnostics


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def plot_diagnostics(
    rows: list[dict],
    out_path: Path,
    *,
    title: str | None = None,
) -> None:
    """Plot test statistics against thresholds over time."""
    if not rows:
        raise ValueError("diagnostics file is empty")

    times = np.asarray([float(r["timestamp"]) for r in rows], dtype=np.float64)
    t0 = float(times[0])
    t_rel = times - t0

    # Collect unique test names across epochs (sources or axes).
    names: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for t in r.get("tests") or []:
            n = str(t["name"])
            if n not in seen:
                seen.add(n)
                names.append(n)

    thresholds = []
    for r in rows:
        thr = r.get("threshold")
        thresholds.append(float(thr) if thr is not None else np.nan)
    thr_arr = np.asarray(thresholds, dtype=np.float64)

    states = [str(r.get("state", "")) for r in rows]
    state_changed = [bool(r.get("state_changed")) for r in rows]

    n_panels = max(len(names), 1) + 1  # stats + state strip
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(10.0, 2.2 * n_panels),
        sharex=True,
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]
    axes = list(axes)

    if names:
        for i, name in enumerate(names):
            ax = axes[i]
            stats = []
            thr_line = []
            for r in rows:
                hit = next(
                    (t for t in (r.get("tests") or []) if str(t["name"]) == name),
                    None,
                )
                if hit is None:
                    stats.append(np.nan)
                    thr_line.append(np.nan)
                else:
                    s = hit.get("statistic")
                    th = hit.get("threshold")
                    stats.append(float(s) if s is not None else np.nan)
                    thr_line.append(float(th) if th is not None else np.nan)
            ax.plot(t_rel, stats, color="#1565c0", lw=1.4, label=f"{name} statistic")
            thr_plot = np.asarray(thr_line, dtype=np.float64)
            if np.any(np.isfinite(thr_plot)):
                ax.plot(
                    t_rel,
                    thr_plot,
                    color="#c62828",
                    lw=1.2,
                    ls="--",
                    label="threshold",
                )
            elif np.any(np.isfinite(thr_arr)):
                ax.plot(
                    t_rel,
                    thr_arr,
                    color="#c62828",
                    lw=1.2,
                    ls="--",
                    label="threshold",
                )
            ax.set_ylabel(name)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "no per-source tests in file", ha="center", va="center")
        axes[0].set_axis_off()

    ax_s = axes[-1]
    state_map = {"NOMINAL": 0, "DEGRADED": 1, "DO_NOT_TRUST": 2}
    y = np.asarray([state_map.get(s, -1) for s in states], dtype=np.float64)
    ax_s.step(t_rel, y, where="post", color="#37474f", lw=1.5)
    for i, changed in enumerate(state_changed):
        if changed:
            ax_s.axvline(t_rel[i], color="#ef6c00", alpha=0.45, lw=1.0)
    ax_s.set_yticks([0, 1, 2])
    ax_s.set_yticklabels(["NOMINAL", "DEGRADED", "DO_NOT_TRUST"])
    ax_s.set_xlabel(f"Time since t0={t0:.3f} (s)")
    ax_s.set_ylabel("state")
    ax_s.grid(True, alpha=0.3)
    ax_s.set_ylim(-0.4, 2.4)

    fig.suptitle(title or "Cairn diagnostics replay", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "diagnostics",
        type=Path,
        help="JSONL diagnostics file from CairnMonitor",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: <diagnostics_stem>_replay.png beside the JSONL)",
    )
    args = parser.parse_args(argv)

    path = args.diagnostics.resolve()
    if not path.is_file():
        print(f"diagnostics file not found: {path}", file=sys.stderr)
        return 1
    out = args.out or path.with_name(path.stem + "_replay.png")
    rows = load_diagnostics(path)
    plot_diagnostics(rows, out, title=f"Diagnostics: {path.name}")
    print(f"wrote {out} ({len(rows)} epochs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
